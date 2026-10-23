//! Process logging: `tracing` with a text or JSON formatter on stderr.
//! `--log-level` wins over `RUST_LOG`; the default is `info`. Access lines (`--access-log`) use the
//! target `fllm::access`, so `RUST_LOG=fllm::access=off,info` silences just them.
use super::options::{HttpOptions, LogFormat};
use tracing_subscriber::EnvFilter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_codes_are_removed() {
        assert_eq!(strip_ansi("\u{1b}[2m2026\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m ready"), "2026  INFO ready");
    }

    #[test]
    fn the_ring_keeps_whole_lines_and_the_last_rewrite_of_a_progress_line() {
        ring_push(b"ring-test first\nring-test sec");
        ring_push(b"ond\nring-test bar: 10%\rring-test bar: 90%\n");
        let lines = recent_lines(RING_LINES);
        let mine: Vec<&String> = lines.iter().filter(|l| l.starts_with("ring-test")).collect();
        assert_eq!(mine, ["ring-test first", "ring-test second", "ring-test bar: 90%"]);
        assert!(recent_lines(1).len() <= 1);
    }
}

/// The most recent log lines, kept for the Studio's log view.
const RING_LINES: usize = 2000;
const MAX_LINE_CHARS: usize = 2000;

#[derive(Default)]
struct Ring {
    lines: std::collections::VecDeque<String>,
    partial: String,
}

static RING: std::sync::Mutex<Option<Ring>> = std::sync::Mutex::new(None);

/// Removes terminal colour codes (`ESC [ ... letter`).
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn ring_push(bytes: &[u8]) {
    let text = strip_ansi(&String::from_utf8_lossy(bytes));
    let mut guard = RING.lock().unwrap_or_else(|e| e.into_inner());
    let ring = guard.get_or_insert_with(Ring::default);
    // progress lines end in a carriage return and are rewritten in place: they are not worth keeping
    for piece in text.split_inclusive('\n') {
        ring.partial.push_str(piece);
        if ring.partial.ends_with('\n') {
            let line = ring.partial.trim_end_matches(['\n', '\r']).rsplit('\r').next().unwrap_or_default().to_owned();
            ring.partial.clear();
            if !line.trim().is_empty() {
                if ring.lines.len() == RING_LINES {
                    ring.lines.pop_front();
                }
                ring.lines.push_back(line.chars().take(MAX_LINE_CHARS).collect());
            }
        } else if ring.partial.len() > 4 * MAX_LINE_CHARS {
            ring.partial.clear();
        }
    }
}

/// The last `count` log lines, oldest first.
pub fn recent_lines(count: usize) -> Vec<String> {
    let guard = RING.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map_or_else(Vec::new, |ring| {
        let skip = ring.lines.len().saturating_sub(count);
        ring.lines.iter().skip(skip).cloned().collect()
    })
}

/// Writes to the standard error output and keeps a copy of the lines.
struct Tee;

impl std::io::Write for Tee {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        ring_push(buf);
        std::io::stderr().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

struct MakeTee;

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for MakeTee {
    type Writer = Tee;
    fn make_writer(&'a self) -> Tee {
        Tee
    }
}

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
        .with_writer(MakeTee)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()));
    // a second call (tests start several servers in one process) keeps the first subscriber
    let _ = match options.log_format {
        LogFormat::Text => builder.try_init(),
        LogFormat::Json => builder.json().try_init(),
    };
}
