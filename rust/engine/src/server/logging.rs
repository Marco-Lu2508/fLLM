//! Process logging: `tracing` with a text or JSON formatter on stderr.
//! `--log-level` wins over `RUST_LOG`; the default is `info`. Access lines (`--access-log`) use the
//! target `fllm::access`, so `RUST_LOG=fllm::access=off,info` silences just them.
use super::options::{HttpOptions, LogFormat};
use tracing_subscriber::EnvFilter;

pub fn init(options: &HttpOptions) {
    let filter = match &options.log_level {
        Some(level) => EnvFilter::try_new(level).unwrap_or_else(|e| {
            eprintln!("invalid --log-level '{level}': {e}; using info");
            EnvFilter::new("info")
        }),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()));
    // a second call (tests start several servers in one process) keeps the first subscriber
    let _ = match options.log_format {
        LogFormat::Text => builder.try_init(),
        LogFormat::Json => builder.json().try_init(),
    };
}
