//! Fetching of `http(s)://` image URLs for vision requests (OpenAI `image_url`, Anthropic `source.type=url`).
//!
//! Off by default (`--allow-remote-images`). The fetch is hardened against server-side request forgery:
//! only http and https, no credentials in the URL, the host name is resolved here and every address is
//! checked against the private/loopback/link-local/metadata ranges before connecting, the connection is
//! pinned to the checked address (so a second DNS answer cannot redirect it), redirects are followed by hand
//! (at most [`MAX_REDIRECTS`], each hop checked again), the body is size- and time-limited, and the content is
//! identified by its magic bytes instead of the server's `Content-Type`.
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

pub const MAX_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_REDIRECTS: usize = 3;
pub const TOTAL_TIMEOUT: Duration = Duration::from_secs(15);

fn blocked_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local() // 169.254.0.0/16, includes the cloud metadata address
        || ip.is_broadcast()
        || ip.is_multicast()
        || a == 0
        || (a == 100 && (64..128).contains(&b)) // carrier-grade NAT 100.64.0.0/10
        || (a == 192 && b == 0 && c == 0) // IETF protocol assignments
        || (a == 192 && b == 0 && c == 2) // TEST-NET-1
        || (a == 198 && (b == 18 || b == 19)) // benchmarking
        || (a == 198 && b == 51 && c == 100) // TEST-NET-2
        || (a == 203 && b == 0 && c == 113) // TEST-NET-3
        || a >= 240 // reserved
}

fn blocked_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return blocked_v4(v4);
    }
    let s = ip.segments();
    // NAT64 64:ff9b::/96 and the local-use 64:ff9b:1::/48 embed an IPv4 address or are not public
    if s[0] == 0x64 && s[1] == 0xff9b {
        return s[2..6] != [0, 0, 0, 0] || blocked_v4(Ipv4Addr::new((s[6] >> 8) as u8, s[6] as u8, (s[7] >> 8) as u8, s[7] as u8));
    }
    // 6to4 2002::/16 embeds the IPv4 address in bits 16..48
    if s[0] == 0x2002 {
        return blocked_v4(Ipv4Addr::new((s[1] >> 8) as u8, s[1] as u8, (s[2] >> 8) as u8, s[2] as u8));
    }
    ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || (s[0] & 0xfe00) == 0xfc00 // unique local fc00::/7 (also fd00:ec2::254, the AWS IPv6 metadata address)
        || (s[0] & 0xffc0) == 0xfe80 // link local
        || (s[0] & 0xffc0) == 0xfec0 // deprecated site local
        || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation
        || (s[0] == 0x0100 && s[1..4] == [0, 0, 0]) // discard-only 100::/64
}

/// `true` for every address a remote image fetch must never connect to.
pub fn is_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => blocked_v4(v4),
        IpAddr::V6(v6) => blocked_v6(v6),
    }
}

/// Which kind of image the bytes are, by magic number (the formats the vision encoder reads).
pub fn sniff(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else {
        None
    }
}

#[derive(Clone, Copy, Default)]
pub struct Policy {
    /// Test hook: also allow loopback and private addresses.
    #[cfg(test)]
    pub allow_private: bool,
}

impl Policy {
    fn permits(&self, ip: IpAddr) -> bool {
        #[cfg(test)]
        if self.allow_private {
            return true;
        }
        !is_blocked(ip)
    }
}

/// Resolves `host:port` and returns an address that passes the policy.
async fn checked_address(host: &str, port: u16, policy: Policy) -> Result<SocketAddr, String> {
    let addresses: Vec<SocketAddr> = if let Ok(ip) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        vec![SocketAddr::new(ip, port)]
    } else {
        tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| format!("cannot resolve image host '{host}': {e}"))?
            .collect()
    };
    if addresses.is_empty() {
        return Err(format!("image host '{host}' has no address"));
    }
    // all answers must be public: a name that also resolves to an internal address is refused outright
    if let Some(bad) = addresses.iter().find(|a| !policy.permits(a.ip())) {
        return Err(format!(
            "image URL refused: host '{host}' resolves to {} (private, loopback, link-local or reserved addresses are blocked)",
            bad.ip()
        ));
    }
    Ok(addresses[0])
}

pub const DISABLED: &str = "remote image URLs are disabled; send the image as a data: URL, or start the server with --allow-remote-images";

/// Downloads an image and returns it as a `data:` URL.
pub async fn fetch(url: &str, policy: Policy) -> Result<String, String> {
    match tokio::time::timeout(TOTAL_TIMEOUT, fetch_inner(url, policy)).await {
        Ok(result) => result,
        Err(_) => Err(format!("image download exceeded {} s", TOTAL_TIMEOUT.as_secs())),
    }
}

async fn fetch_inner(url: &str, policy: Policy) -> Result<String, String> {
    let mut current = reqwest::Url::parse(url).map_err(|e| format!("invalid image URL: {e}"))?;
    for hop in 0..=MAX_REDIRECTS {
        if !matches!(current.scheme(), "http" | "https") {
            return Err("image URLs must use http or https".into());
        }
        if !current.username().is_empty() || current.password().is_some() {
            return Err("image URLs must not contain credentials".into());
        }
        let host = current.host_str().ok_or("image URL has no host")?.to_owned();
        let port = current.port_or_known_default().ok_or("image URL has no port")?;
        let address = checked_address(&host, port, policy).await?;
        // pin the connection to the address that was checked
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .resolve(&host, address)
            .timeout(TOTAL_TIMEOUT)
            .connect_timeout(Duration::from_secs(5))
            .user_agent("fllm-image-fetch")
            .no_proxy()
            .build()
            .map_err(|e| format!("cannot start image download: {e}"))?;
        let mut response = client
            .get(current.clone())
            .header("accept", "image/png, image/jpeg")
            .send()
            .await
            .map_err(|e| format!("image download failed: {}", e.without_url()))?;
        let status = response.status();
        if status.is_redirection() {
            if hop == MAX_REDIRECTS {
                return Err(format!("image URL redirects more than {MAX_REDIRECTS} times"));
            }
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or("image redirect without a Location header")?;
            current = current
                .join(location)
                .map_err(|e| format!("invalid image redirect target: {e}"))?;
            continue;
        }
        if !status.is_success() {
            return Err(format!("image download failed with HTTP status {}", status.as_u16()));
        }
        if response.content_length().is_some_and(|n| n > MAX_BYTES as u64) {
            return Err(format!("image is larger than {} MiB", MAX_BYTES >> 20));
        }
        let mut body: Vec<u8> = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| format!("image download failed: {}", e.without_url()))?
        {
            if body.len() + chunk.len() > MAX_BYTES {
                return Err(format!("image is larger than {} MiB", MAX_BYTES >> 20));
            }
            body.extend_from_slice(&chunk);
        }
        let kind = sniff(&body).ok_or("the downloaded file is not a PNG or JPEG image")?;
        use base64::Engine;
        return Ok(format!("data:{kind};base64,{}", base64::engine::general_purpose::STANDARD.encode(&body)));
    }
    Err(format!("image URL redirects more than {MAX_REDIRECTS} times"))
}

/// Turns the image URLs of a request into data URLs: `data:` URLs pass unchanged, http(s) URLs are fetched when
/// allowed, anything else is an error that tells the caller what to do.
pub async fn resolve_all(urls: Vec<String>, allow_remote: bool, policy: Policy) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(urls.len());
    for url in urls {
        if url.starts_with("data:") {
            out.push(url);
        } else if url.starts_with("http://") || url.starts_with("https://") {
            if !allow_remote {
                return Err(DISABLED.into());
            }
            out.push(fetch(&url, policy).await?);
        } else {
            return Err("image URLs must be data: URLs, or http(s) URLs with --allow-remote-images".into());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn blocked(text: &str) -> bool {
        is_blocked(text.parse().unwrap())
    }

    #[test]
    fn internal_addresses_are_blocked() {
        for ip in [
            "127.0.0.1", "127.8.9.10", "10.0.0.1", "172.16.5.4", "172.31.255.255", "192.168.1.1", "169.254.169.254", "0.0.0.0",
            "100.64.0.1", "100.127.255.254", "224.0.0.1", "255.255.255.255", "240.0.0.1", "198.18.0.1", "192.0.2.5",
            "::1", "::", "fe80::1", "fc00::1", "fd00:ec2::254", "ff02::1", "::ffff:127.0.0.1", "::ffff:10.1.2.3",
            "::ffff:169.254.169.254", "64:ff9b::7f00:1", "2002:7f00:1::1", "2001:db8::1", "fec0::1",
        ] {
            assert!(blocked(ip), "{ip} must be blocked");
        }
    }

    #[test]
    fn public_addresses_pass() {
        for ip in ["1.1.1.1", "8.8.8.8", "93.184.216.34", "172.32.0.1", "100.63.255.255", "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808"] {
            assert!(!blocked(ip), "{ip} must be allowed");
        }
    }

    #[test]
    fn magic_numbers_identify_png_and_jpeg_only() {
        assert_eq!(sniff(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0]), Some("image/png"));
        assert_eq!(sniff(&[0xff, 0xd8, 0xff, 0xe0]), Some("image/jpeg"));
        assert_eq!(sniff(b"GIF89a"), None);
        assert_eq!(sniff(b"<html>"), None);
    }

    #[tokio::test]
    async fn bad_urls_are_refused_before_any_connection() {
        let policy = Policy::default();
        for (url, expect) in [
            ("ftp://example.com/a.png", "http or https"),
            ("file:///etc/passwd", "http or https"),
            ("http://user:pw@example.com/a.png", "credentials"),
            ("http://127.0.0.1:1/a.png", "blocked"),
            ("http://[::1]/a.png", "blocked"),
            ("http://169.254.169.254/latest/meta-data/", "blocked"),
            ("http://localhost/a.png", "blocked"),
            ("not a url", "invalid image URL"),
        ] {
            let error = fetch(url, policy).await.unwrap_err();
            assert!(error.contains(expect), "{url}: {error}");
        }
    }

    #[tokio::test]
    async fn remote_images_are_off_unless_enabled() {
        let urls = vec!["https://example.com/a.png".to_owned()];
        let error = resolve_all(urls.clone(), false, Policy::default()).await.unwrap_err();
        assert!(error.contains("--allow-remote-images"), "{error}");
        let data = vec!["data:image/png;base64,AAAA".to_owned()];
        assert_eq!(resolve_all(data.clone(), false, Policy::default()).await.unwrap(), data);
        assert!(resolve_all(vec!["gopher://x".into()], true, Policy::default()).await.is_err());
    }

    /// One-shot HTTP server on loopback answering each connection with the next canned response.
    fn serve(responses: Vec<Vec<u8>>) -> (u16, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = [0u8; 2048];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(&response);
            }
        });
        (port, handle)
    }

    fn http(status: &str, headers: &str, body: &[u8]) -> Vec<u8> {
        let mut out = format!("HTTP/1.1 {status}\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n", body.len()).into_bytes();
        out.extend_from_slice(body);
        out
    }

    #[tokio::test]
    async fn allowed_hosts_are_fetched_followed_and_sniffed() {
        let png = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3];
        let policy = Policy { allow_private: true };
        let (port, server) = serve(vec![http("200 OK", "content-type: text/plain\r\n", &png)]);
        let url = fetch(&format!("http://127.0.0.1:{port}/x"), policy).await.unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "content is sniffed, not taken from the header");
        server.join().unwrap();

        let (port, server) = serve(vec![http("200 OK", "", b"<html>nope</html>")]);
        let error = fetch(&format!("http://127.0.0.1:{port}/x"), policy).await.unwrap_err();
        assert!(error.contains("not a PNG or JPEG"), "{error}");
        server.join().unwrap();

        let (port, server) = serve(vec![http("404 Not Found", "", b"")]);
        assert!(fetch(&format!("http://127.0.0.1:{port}/x"), policy).await.unwrap_err().contains("404"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn redirects_are_limited_and_rechecked() {
        let policy = Policy { allow_private: true };
        let (port, server) = serve(vec![
            http("302 Found", "location: /b\r\n", b""),
            http("302 Found", "location: /c\r\n", b""),
            http("302 Found", "location: /d\r\n", b""),
            http("302 Found", "location: /e\r\n", b""),
        ]);
        let error = fetch(&format!("http://127.0.0.1:{port}/a"), policy).await.unwrap_err();
        assert!(error.contains("redirects more than"), "{error}");
        server.join().unwrap();
    }

    #[tokio::test]
    async fn oversized_bodies_are_cut_off() {
        let policy = Policy { allow_private: true };
        let mut body = vec![0xff, 0xd8, 0xff];
        body.resize(MAX_BYTES + 1, 0);
        let (port, server) = serve(vec![http("200 OK", "", &body)]);
        let error = fetch(&format!("http://127.0.0.1:{port}/big"), policy).await.unwrap_err();
        assert!(error.contains("larger than"), "{error}");
        // joining on the async thread would keep the client connection from closing, and the writer would never stop
        let _ = tokio::task::spawn_blocking(move || server.join()).await;
    }
}
