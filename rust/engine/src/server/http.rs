//! The HTTP layer around the API routes: request ids, CORS, authentication and rate limits, the
//! non-streaming request timeout, uniform error bodies and the access log (all in [`layer`]), and
//! the connection loop with TLS, header/idle timeouts, a connection limit and graceful shutdown
//! ([`serve`]).
use super::{
    errors,
    options::HttpOptions,
    security::{Caller, RequestStats, Security, Verdict, CALLER},
};
use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Router,
};
use http_body::{Frame, SizeHint};
use hyper_util::rt::TokioIo;
use std::{
    convert::Infallible,
    future::Future,
    net::SocketAddr,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, AsyncWriteExt, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::{watch, OwnedSemaphorePermit, Semaphore},
    task::JoinSet,
};
use tower::ServiceExt;

/// Peer address of the connection, for the access log.
#[derive(Clone, Copy, Debug)]
pub struct ClientAddr(pub SocketAddr);

#[derive(Default)]
pub struct HttpCounters {
    pub connections_open: AtomicU64,
    pub connections_rejected: AtomicU64,
    pub request_timeouts: AtomicU64,
}

pub struct HttpContext {
    pub options: HttpOptions,
    pub security: Security,
    pub counters: HttpCounters,
}

impl HttpContext {
    pub fn new(options: HttpOptions) -> Arc<Self> {
        Arc::new(Self {
            security: Security::new(&options),
            options,
            counters: HttpCounters::default(),
        })
    }
}

const REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

fn fresh_request_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    format!("req-{:012x}{:04x}", nanos & 0xffff_ffff_ffff, COUNTER.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

/// A client-supplied id is kept when it is short and made of harmless characters (it ends up in logs and headers).
fn request_id_of(headers: &HeaderMap) -> String {
    headers
        .get(&REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .filter(|v| {
            !v.is_empty()
                && v.len() <= 128
                && v.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b':'))
        })
        .map_or_else(fresh_request_id, str::to_owned)
}

const CORS_METHODS: &str = "GET, POST, PUT, DELETE, OPTIONS, HEAD";
const CORS_HEADERS: &str = "Authorization, x-api-key, api-key, Content-Type, anthropic-version, anthropic-beta, x-request-id, OpenAI-Organization, OpenAI-Project, x-stainless-os, x-stainless-lang, x-stainless-package-version, x-stainless-runtime, x-stainless-runtime-version, x-stainless-arch, x-stainless-retry-count, x-stainless-timeout";
const CORS_EXPOSE: &str = "x-request-id, retry-after, content-type";

fn allowed_origin(options: &HttpOptions, origin: &str) -> Option<String> {
    if options.cors_origins.iter().any(|o| o == "*") {
        return Some("*".into());
    }
    options
        .cors_origins
        .iter()
        .any(|o| o.eq_ignore_ascii_case(origin.trim_end_matches('/')))
        .then(|| origin.to_owned())
}

fn set(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn preflight(options: &HttpOptions, request_headers: &HeaderMap, origin: &str, id: &str, path: &str) -> Response {
    let Some(allow) = allowed_origin(options, origin) else {
        return errors::for_path(
            path,
            StatusCode::FORBIDDEN,
            "cross-origin requests from this origin are not allowed; start the server with --cors-origins",
            Some("cors_origin_not_allowed"),
            id,
        );
    };
    let mut response = StatusCode::NO_CONTENT.into_response();
    let headers = response.headers_mut();
    set(headers, "access-control-allow-origin", &allow);
    set(headers, "access-control-allow-methods", CORS_METHODS);
    let requested = request_headers
        .get("access-control-request-headers")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 1024);
    set(headers, "access-control-allow-headers", requested.unwrap_or(CORS_HEADERS));
    set(headers, "access-control-max-age", "600");
    if request_headers.get("access-control-request-private-network").is_some_and(|v| v == "true") {
        set(headers, "access-control-allow-private-network", "true");
    }
    set(headers, "vary", "Origin, Access-Control-Request-Headers");
    response
}

/// Gives plain-text rejections of the framework (body too large, bad JSON, unknown route, wrong method) the
/// error shape of the route, and puts the request id into JSON error bodies.
async fn normalize_error(response: Response, path: &str, id: &str) -> Response {
    let status = response.status();
    let is_json = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"));
    let (mut parts, body) = response.into_parts();
    let bytes = match axum::body::to_bytes(body, 1 << 20).await {
        Ok(bytes) => bytes,
        Err(_) => return errors::for_path(path, status, "error response too large", None, id),
    };
    if is_json {
        if let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let mut changed = false;
            if value["error"].is_object() && value["error"].get("request_id").is_none() {
                value["error"]["request_id"] = serde_json::json!(id);
                changed = true;
            } else if value["type"] == "error" && value.get("request_id").is_none() {
                value["request_id"] = serde_json::json!(id);
                changed = true;
            }
            if changed {
                parts.headers.remove(header::CONTENT_LENGTH);
                return Response::from_parts(parts, Body::from(value.to_string()));
            }
        }
        return Response::from_parts(parts, Body::from(bytes));
    }
    let text = String::from_utf8_lossy(&bytes);
    let message = match status {
        StatusCode::PAYLOAD_TOO_LARGE => "request body too large; raise --max-body-size".to_owned(),
        StatusCode::NOT_FOUND if text.trim().is_empty() => format!("unknown route {path}"),
        StatusCode::METHOD_NOT_ALLOWED if text.trim().is_empty() => "method not allowed for this route".to_owned(),
        _ if text.trim().is_empty() => status.canonical_reason().unwrap_or("error").to_owned(),
        _ => text.trim().to_owned(),
    };
    let mut replacement = errors::for_path(path, status, &message, None, id);
    for name in [header::ALLOW, header::RETRY_AFTER, header::WWW_AUTHENTICATE] {
        if let Some(value) = parts.headers.get(&name) {
            replacement.headers_mut().insert(name, value.clone());
        }
    }
    replacement
}

/// Runs `on_end` when the body is finished or dropped (client gone), without changing the body.
pub struct Watched<F: FnOnce() + Unpin> {
    inner: Body,
    on_end: Option<F>,
}

impl<F: FnOnce() + Unpin> Watched<F> {
    pub fn new(inner: Body, on_end: F) -> Self {
        Self { inner, on_end: Some(on_end) }
    }
}

impl<F: FnOnce() + Unpin> http_body::Body for Watched<F> {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let polled = Pin::new(&mut self.inner).poll_frame(cx);
        if let Poll::Ready(None) = polled {
            if let Some(done) = self.on_end.take() {
                done();
            }
        }
        polled
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl<F: FnOnce() + Unpin> Drop for Watched<F> {
    fn drop(&mut self) {
        if let Some(done) = self.on_end.take() {
            done();
        }
    }
}

/// The Studio routes that change something (settings, model downloads and deletions, API keys) are
/// administration. Without a configured API key they are only open to a client on the same machine:
/// otherwise anybody who can reach the port could delete models or register a key and lock the owner out.
/// The agent tools (files and shell) additionally refuse every other machine even when a key is configured.
/// A web page in the owner's own browser reaches the port from this machine as well, so the request must
/// also address the server by `localhost` or an IP address (a DNS-rebinding name is refused) and, when the
/// browser names an origin, come from the page of this very server.
pub fn studio_admin_blocked(
    auth_required: bool,
    method: &Method,
    path: &str,
    peer: Option<SocketAddr>,
    headers: &HeaderMap,
) -> bool {
    // the agent tools read and write files and run programs: never from another machine, with or without a key
    // (the web fetch and search routes reach out from this machine and read files of the agent: same rule)
    let agent = path.starts_with("/api/studio/agent/") || path.starts_with("/api/studio/web/");
    // the list of changes names files of the workspace, so even this read stays on this machine
    let reads_workspace = path == "/api/studio/agent/changes";
    if (auth_required && !agent)
        || !path.starts_with("/api/studio/")
        || (matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS) && !reads_workspace)
    {
        return false;
    }
    if !peer.is_some_and(|addr| addr.ip().to_canonical().is_loopback()) {
        return true;
    }
    let host = headers.get(header::HOST).and_then(|v| v.to_str().ok()).unwrap_or_default();
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or_default(),
        None => host.rsplit_once(':').map_or(host, |(name, _)| name),
    };
    if !(name.eq_ignore_ascii_case("localhost") || name.parse::<std::net::IpAddr>().is_ok()) {
        return true;
    }
    match headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        None => false,
        Some(origin) => origin
            .split_once("://")
            .is_none_or(|(_, authority)| !authority.eq_ignore_ascii_case(host)),
    }
}

/// The middleware of every route. Order: request id, CORS preflight, authentication and rate limit, handler
/// under the request timeout, error normalisation, CORS and id headers, access log.
pub async fn layer(State(ctx): State<Arc<HttpContext>>, request: Request, next: Next) -> Response {
    let started = Instant::now();
    let id = request_id_of(request.headers());
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let peer = request.extensions().get::<ClientAddr>().map(|a| a.0);
    let stats = Arc::new(RequestStats::default());

    let mut response = if method == Method::OPTIONS
        && origin.is_some()
        && request.headers().contains_key("access-control-request-method")
    {
        preflight(&ctx.options, request.headers(), origin.as_deref().unwrap_or_default(), &id, &path)
    } else if studio_admin_blocked(ctx.security.auth_required(), &method, &path, peer, request.headers()) {
        errors::for_path(
            &path,
            StatusCode::FORBIDDEN,
            "the Studio administration needs an API key when the request does not come from this machine; start the server with --api-key",
            Some("studio_admin_forbidden"),
            &id,
        )
    } else {
        match ctx.security.check(&method, &path, request.headers()) {
            Verdict::Allowed(limiter) => {
                let caller = Caller { limiter, stats: stats.clone(), request_id: id.clone() };
                let run = CALLER.scope(caller, next.run(request));
                let limit = ctx.options.request_timeout;
                if limit.is_zero() {
                    run.await
                } else {
                    match tokio::time::timeout(limit, run).await {
                        Ok(response) => response,
                        Err(_) => {
                            ctx.counters.request_timeouts.fetch_add(1, Ordering::Relaxed);
                            tracing::warn!(request_id = %id, %path, "request exceeded --request-timeout of {}s and was cancelled", limit.as_secs());
                            errors::for_path(
                                &path,
                                StatusCode::GATEWAY_TIMEOUT,
                                &format!("the request did not finish within --request-timeout ({} s)", limit.as_secs()),
                                Some("request_timeout"),
                                &id,
                            )
                        }
                    }
                }
            }
            Verdict::Missing => {
                let mut r = errors::for_path(
                    &path,
                    StatusCode::UNAUTHORIZED,
                    "missing API key; send it as 'Authorization: Bearer KEY' or 'x-api-key: KEY'",
                    Some("missing_api_key"),
                    &id,
                );
                r.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
                r
            }
            Verdict::Invalid => {
                let mut r = errors::for_path(&path, StatusCode::UNAUTHORIZED, "invalid API key", Some("invalid_api_key"), &id);
                r.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
                r
            }
            Verdict::RateLimited { retry_after, label } => {
                tracing::info!(request_id = %id, key = %label, "rate limit reached");
                let mut r = errors::for_path(
                    &path,
                    StatusCode::TOO_MANY_REQUESTS,
                    "rate limit reached for this API key; retry after the time in the Retry-After header",
                    Some("rate_limit_exceeded"),
                    &id,
                );
                set(r.headers_mut(), "retry-after", &retry_after.as_secs().max(1).to_string());
                r
            }
        }
    };

    if response.status().as_u16() >= 400 {
        response = normalize_error(response, &path, &id).await;
    }
    if let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert(REQUEST_ID, value);
    }
    if let Some(origin) = origin.as_deref() {
        if let Some(allow) = allowed_origin(&ctx.options, origin) {
            let headers = response.headers_mut();
            if !headers.contains_key("access-control-allow-origin") {
                set(headers, "access-control-allow-origin", &allow);
                set(headers, "access-control-expose-headers", CORS_EXPOSE);
                headers.append(header::VARY, HeaderValue::from_static("Origin"));
            }
        }
    }
    if ctx.options.access_log {
        let status = response.status().as_u16();
        let (parts, body) = response.into_parts();
        let line_id = id.clone();
        let body = Body::new(Watched::new(body, move || {
            tracing::info!(
                target: "fllm::access",
                request_id = %line_id,
                client = %peer.map(|p| p.ip().to_string()).unwrap_or_default(),
                method = %method,
                path = %path,
                status,
                latency_ms = started.elapsed().as_millis() as u64,
                prompt_tokens = stats.prompt_tokens.load(Ordering::Relaxed),
                completion_tokens = stats.completion_tokens.load(Ordering::Relaxed),
                "request"
            );
        }));
        return Response::from_parts(parts, body);
    }
    response
}

/// Per-connection bookkeeping for the idle and header timeouts.
struct ConnState {
    started: Instant,
    in_flight: AtomicUsize,
    requests: AtomicU64,
    /// Milliseconds since `started` when the connection became idle.
    idle_since: AtomicU64,
    /// Milliseconds since `started` plus one when the first byte of a request head arrived; 0 when none is pending.
    head_started: AtomicU64,
}

impl ConnState {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            in_flight: AtomicUsize::new(0),
            requests: AtomicU64::new(0),
            idle_since: AtomicU64::new(0),
            head_started: AtomicU64::new(0),
        }
    }
    fn now(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }
    fn bytes_read(&self) {
        if self.in_flight.load(Ordering::Relaxed) == 0 && self.head_started.load(Ordering::Relaxed) == 0 {
            self.head_started.store(self.now() + 1, Ordering::Relaxed);
        }
    }
    fn request_started(&self) {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.head_started.store(0, Ordering::Relaxed);
    }
    fn request_ended(&self) {
        if self.in_flight.fetch_sub(1, Ordering::Relaxed) == 1 {
            self.idle_since.store(self.now(), Ordering::Relaxed);
        }
    }
    /// Why the connection must be closed, if it must.
    fn expired(&self, options: &HttpOptions) -> Option<&'static str> {
        if self.in_flight.load(Ordering::Relaxed) > 0 {
            return None;
        }
        let now = self.now();
        let head = self.head_started.load(Ordering::Relaxed);
        let header_limit = options.header_read_timeout.as_millis() as u64;
        if head > 0 {
            return (!options.header_read_timeout.is_zero() && now.saturating_sub(head - 1) > header_limit)
                .then_some("request headers not received in time");
        }
        let idle = now.saturating_sub(self.idle_since.load(Ordering::Relaxed));
        if self.requests.load(Ordering::Relaxed) == 0 {
            return (!options.header_read_timeout.is_zero() && idle > header_limit).then_some("no request received in time");
        }
        (!options.keep_alive_timeout.is_zero() && idle > options.keep_alive_timeout.as_millis() as u64)
            .then_some("keep-alive timeout")
    }
}

/// Stream wrapper that tells the connection state when bytes arrive.
struct Observed<S> {
    inner: S,
    state: Arc<ConnState>,
}

impl<S: AsyncRead + Unpin> AsyncRead for Observed<S> {
    fn poll_read(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        let polled = Pin::new(&mut self.inner).poll_read(cx, buf);
        if matches!(polled, Poll::Ready(Ok(()))) && buf.filled().len() > before {
            self.state.bytes_read();
        }
        polled
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for Observed<S> {
    fn poll_write(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
    fn poll_write_vectored(mut self: Pin<&mut Self>, cx: &mut Context<'_>, bufs: &[std::io::IoSlice<'_>]) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }
    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

async fn drive<S>(io: S, peer: SocketAddr, state: Arc<ConnState>, app: Router, options: Arc<HttpOptions>, mut stop: watch::Receiver<bool>)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service_state = state.clone();
    let service = hyper::service::service_fn(move |request: hyper::Request<hyper::body::Incoming>| {
        let app = app.clone();
        let state = service_state.clone();
        async move {
            state.request_started();
            let mut request = request.map(Body::new);
            request.extensions_mut().insert(ClientAddr(peer));
            let response = app.oneshot(request).await.unwrap_or_else(|never: Infallible| match never {});
            let (parts, body) = response.into_parts();
            let body = Body::new(Watched::new(body, move || state.request_ended()));
            Ok::<_, Infallible>(Response::from_parts(parts, body))
        }
    });
    let mut builder = hyper::server::conn::http1::Builder::new();
    // header and idle timeouts are enforced below, where the two can be told apart
    builder.keep_alive(true).header_read_timeout(None);
    let connection = builder.serve_connection(TokioIo::new(Observed { inner: io, state: state.clone() }), service);
    let mut connection = std::pin::pin!(connection.with_upgrades());
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    let mut draining = false;
    loop {
        tokio::select! {
            result = connection.as_mut() => {
                if let Err(error) = result {
                    tracing::debug!(%peer, "connection ended: {error}");
                }
                return;
            }
            _ = tick.tick() => {
                if let Some(reason) = state.expired(&options) {
                    tracing::debug!(%peer, "closing connection: {reason}");
                    return;
                }
            }
            _ = stop.changed(), if !draining => {
                draining = true;
                connection.as_mut().graceful_shutdown();
            }
        }
    }
}

async fn refuse(mut stream: TcpStream) {
    let reply = b"HTTP/1.1 503 Service Unavailable\r\nretry-after: 1\r\nconnection: close\r\ncontent-length: 0\r\n\r\n";
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        let _ = stream.write_all(reply).await;
        let _ = stream.shutdown().await;
    })
    .await;
}

#[allow(clippy::too_many_arguments)]
async fn connection(
    stream: TcpStream,
    peer: SocketAddr,
    tls: Option<tokio_rustls::TlsAcceptor>,
    app: Router,
    ctx: Arc<HttpContext>,
    stop: watch::Receiver<bool>,
    _permit: OwnedSemaphorePermit,
) {
    struct Open<'a>(&'a AtomicU64);
    impl Drop for Open<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::Relaxed);
        }
    }
    ctx.counters.connections_open.fetch_add(1, Ordering::Relaxed);
    let _open = Open(&ctx.counters.connections_open);
    let _ = stream.set_nodelay(true);
    let state = Arc::new(ConnState::new());
    let options = Arc::new(ctx.options.clone());
    match tls {
        None => drive(stream, peer, state, app, options, stop).await,
        Some(acceptor) => {
            let limit = options.header_read_timeout.max(Duration::from_secs(5));
            match tokio::time::timeout(limit, acceptor.accept(stream)).await {
                Ok(Ok(tls_stream)) => drive(tls_stream, peer, state, app, options, stop).await,
                Ok(Err(error)) => tracing::debug!(%peer, "TLS handshake failed: {error}"),
                Err(_) => tracing::debug!(%peer, "TLS handshake timed out"),
            }
        }
    }
}

/// Accepts connections until `shutdown` completes, then stops listening, flips `ready` to false and waits up to
/// `--shutdown-grace` for the requests in flight.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    ctx: Arc<HttpContext>,
    ready: Arc<AtomicBool>,
    shutdown: impl Future<Output = ()>,
) -> Result<(), String> {
    let tls = match (&ctx.options.tls_cert, &ctx.options.tls_key) {
        (Some(cert), Some(key)) => Some(super::tls::acceptor(cert, key)?),
        _ => None,
    };
    let (stop_tx, stop_rx) = watch::channel(false);
    let slots = Arc::new(Semaphore::new(ctx.options.max_connections));
    let mut tasks = JoinSet::new();
    let mut shutdown = std::pin::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => match accepted {
                Ok((stream, peer)) => {
                    while tasks.try_join_next().is_some() {}
                    match slots.clone().try_acquire_owned() {
                        Ok(permit) => {
                            tasks.spawn(connection(stream, peer, tls.clone(), app.clone(), ctx.clone(), stop_rx.clone(), permit));
                        }
                        Err(_) => {
                            ctx.counters.connections_rejected.fetch_add(1, Ordering::Relaxed);
                            tracing::warn!(%peer, "connection refused: --max-connections reached");
                            tasks.spawn(refuse(stream));
                        }
                    }
                }
                Err(error) => {
                    // typically EMFILE: back off instead of spinning
                    tracing::warn!("accept failed: {error}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }
    ready.store(false, Ordering::Relaxed);
    drop(listener);
    let _ = stop_tx.send(true);
    if tasks.is_empty() {
        return Ok(());
    }
    tracing::info!("shutting down: waiting up to {} s for {} connection(s)", ctx.options.shutdown_grace.as_secs(), tasks.len());
    let drain = async { while tasks.join_next().await.is_some() {} };
    if tokio::time::timeout(ctx.options.shutdown_grace, drain).await.is_err() {
        tracing::warn!("shutdown grace period over; closing {} connection(s)", tasks.len());
        tasks.abort_all();
    }
    Ok(())
}

#[cfg(test)]
mod studio_gate_tests {
    use super::*;

    fn addr(text: &str) -> Option<SocketAddr> {
        Some(text.parse().unwrap())
    }

    fn headers(host: &str, origin: Option<&str>) -> HeaderMap {
        let mut map = HeaderMap::new();
        map.insert(header::HOST, HeaderValue::from_str(host).unwrap());
        if let Some(origin) = origin {
            map.insert(header::ORIGIN, HeaderValue::from_str(origin).unwrap());
        }
        map
    }

    const PULL: &str = "/api/studio/models/pull";

    #[test]
    fn studio_administration_is_local_without_an_api_key() {
        let local = headers("127.0.0.1:8000", None);
        let post = Method::POST;
        assert!(!studio_admin_blocked(false, &post, PULL, addr("127.0.0.1:5000"), &local));
        assert!(!studio_admin_blocked(false, &post, PULL, addr("[::1]:5000"), &local));
        assert!(!studio_admin_blocked(false, &post, PULL, addr("[::ffff:127.0.0.1]:5000"), &local));
        assert!(studio_admin_blocked(false, &post, PULL, addr("192.168.1.20:5000"), &local));
        assert!(studio_admin_blocked(false, &Method::DELETE, "/api/studio/keys/key-1", addr("10.7.0.4:1"), &local));
        assert!(studio_admin_blocked(false, &post, PULL, None, &local), "an unknown peer counts as remote");
    }

    #[test]
    fn studio_administration_refuses_pages_of_other_sites_and_rebinding_names() {
        let post = Method::POST;
        let peer = addr("127.0.0.1:5000");
        // the page of this server itself, by name or address
        assert!(!studio_admin_blocked(false, &post, PULL, peer, &headers("localhost:8000", Some("http://localhost:8000"))));
        assert!(!studio_admin_blocked(false, &post, PULL, peer, &headers("[::1]:8000", Some("http://[::1]:8000"))));
        // another site's page talking to the local server
        assert!(studio_admin_blocked(false, &post, PULL, peer, &headers("127.0.0.1:8000", Some("https://evil.example"))));
        assert!(studio_admin_blocked(false, &post, PULL, peer, &headers("localhost:8000", Some("http://localhost:3000"))));
        // a name that resolves to 127.0.0.1 (DNS rebinding), even with a matching origin
        assert!(studio_admin_blocked(false, &post, PULL, peer, &headers("rebind.example:8000", Some("http://rebind.example:8000"))));
        assert!(studio_admin_blocked(false, &post, PULL, peer, &headers("rebind.example:8000", None)));
    }

    #[test]
    fn studio_reading_and_other_routes_are_not_gated() {
        let remote = addr("192.168.1.20:5000");
        let h = headers("192.168.1.5:8000", None);
        assert!(!studio_admin_blocked(false, &Method::GET, "/api/studio/status", remote, &h));
        assert!(!studio_admin_blocked(false, &Method::POST, "/v1/chat/completions", remote, &h));
        // with an API key configured the normal key check decides
        assert!(!studio_admin_blocked(true, &Method::POST, "/api/studio/keys", remote, &h));
        // ... except for the agent tools, which never leave this machine
        assert!(studio_admin_blocked(true, &Method::POST, "/api/studio/agent/tool", remote, &h));
        let local = headers("127.0.0.1:8000", None);
        assert!(!studio_admin_blocked(true, &Method::POST, "/api/studio/agent/tool", addr("127.0.0.1:1"), &local));
    }

    #[test]
    fn web_and_change_routes_of_the_agent_stay_on_this_machine() {
        let remote = addr("192.168.1.20:5000");
        let h = headers("192.168.1.5:8000", None);
        let local = headers("127.0.0.1:8000", None);
        let here = addr("127.0.0.1:1");
        for path in ["/api/studio/web/fetch", "/api/studio/web/search", "/api/studio/agent/undo", "/api/studio/agent/end"] {
            for auth in [false, true] {
                assert!(studio_admin_blocked(auth, &Method::POST, path, remote, &h), "{path} from another machine (key: {auth})");
                assert!(!studio_admin_blocked(auth, &Method::POST, path, here, &local), "{path} from this machine (key: {auth})");
            }
        }
        // the list of changes names workspace files: also a read stays local
        for auth in [false, true] {
            assert!(studio_admin_blocked(auth, &Method::GET, "/api/studio/agent/changes", remote, &h));
            assert!(!studio_admin_blocked(auth, &Method::GET, "/api/studio/agent/changes", here, &local));
        }
        // the tool catalog is plain information
        assert!(!studio_admin_blocked(false, &Method::GET, "/api/studio/agent/tools", remote, &h));
        // a rebinding name is refused for the new routes as for the old ones
        assert!(studio_admin_blocked(false, &Method::POST, "/api/studio/web/fetch", here, &headers("rebind.example:8000", None)));
    }
}