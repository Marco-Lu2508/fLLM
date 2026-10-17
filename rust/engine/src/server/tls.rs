//! Optional TLS for the listener (`--tls-cert`, `--tls-key`), using rustls with the `ring` provider that the HTTP
//! client of the engine already links. HTTP/1.1 only.
use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer};
use std::{path::Path, sync::Arc};
use tokio_rustls::TlsAcceptor;

pub fn acceptor(cert: &Path, key: &Path) -> Result<TlsAcceptor, String> {
    let certs = CertificateDer::pem_file_iter(cert)
        .map_err(|e| format!("cannot read --tls-cert {}: {e}", cert.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("invalid certificate in {}: {e}", cert.display()))?;
    if certs.is_empty() {
        return Err(format!("--tls-cert {} contains no certificate", cert.display()));
    }
    let key = PrivateKeyDer::from_pem_file(key)
        .map_err(|e| format!("cannot read a private key from --tls-key {}: {e}", key.display()))?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("TLS protocol setup failed: {e}"))?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| format!("certificate and key do not fit together: {e}"))?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(TlsAcceptor::from(Arc::new(config)))
}
