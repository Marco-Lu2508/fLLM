//! Reading aloud: Markdown to text that sounds right, cutting it into pieces, and handing it to
//! the computer's own voice (the Studio's voice.js — `stripForSpeech` and `chunkForSpeech` are
//! checked against that code in the tests).

use regex::Regex;

/// Turns Markdown into plain text that sounds right when read aloud: no code, no marks, no web
/// addresses (the Studio's `stripForSpeech`).
pub fn strip_for_speech(markdown: &str) -> String {
    let mut s = markdown.to_string();
    // think blocks
    s = re(r"(?i)<think>[\s\S]*?(</think>|$)").replace_all(&s, " ").into_owned();
    // fenced code blocks (an unfinished one runs to the end)
    let opener = re(r"^[ \t]*(```|~~~)");
    let mut kept: Vec<&str> = Vec::new();
    let mut fence = String::new();
    for line in s.split('\n') {
        let found = opener.captures(line);
        if !fence.is_empty() {
            if let Some(c) = found {
                if c[1] == fence {
                    fence.clear();
                }
            }
            continue;
        }
        if let Some(c) = found {
            fence = c[1].to_string();
            continue;
        }
        kept.push(line);
    }
    s = kept.join("\n");
    s = re(r"<[^>\n]+>").replace_all(&s, " ").into_owned();
    s = re(r"!\[([^\]]*)\]\([^)]*\)").replace_all(&s, "$1").into_owned();
    s = re(r"\[([^\]]+)\]\([^)]*\)").replace_all(&s, "$1").into_owned();
    s = re(r"\bhttps?://[^\s)]+").replace_all(&s, " ").into_owned();
    s = re(r"`+([^`\n]*)`+").replace_all(&s, "$1").into_owned();
    s = re(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+").replace_all(&s, "").into_owned();
    s = re(r"(?m)^[ \t]{0,3}>[ \t]?").replace_all(&s, "").into_owned();
    s = re(r"(?m)^[ \t]*([-*+]|\d+[.)])[ \t]+").replace_all(&s, "").into_owned();
    s = re(r"(?m)^[ \t]*([-*_][ \t]*){3,}$").replace_all(&s, "").into_owned();
    // table separator rows
    s = re(r"(?m)^\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")
        .replace_all(&s, "")
        .into_owned();
    s = re(r"\|").replace_all(&s, " ").into_owned();
    // **bold** and __bold__ (the JavaScript uses one backreference, Rust two patterns)
    s = re(r"\*\*(.+?)\*\*").replace_all(&s, "$1").into_owned();
    s = re(r"__(.+?)__").replace_all(&s, "$1").into_owned();
    // *italic* and _italic_: the boundary behind the closing mark is kept (the lookahead of the
    // JavaScript is written as a group here, because Rust's regex has no lookahead)
    s = re(r"(?m)(^|[\s(])[*_]([^*_\n]+?)[*_]([\s).,!?;:]|$)").replace_all(&s, "$1$2$3").into_owned();
    s = s.replace("~~", "");
    s = re(r"\*+").replace_all(&s, "").into_owned();
    s.split('\n')
        .map(|l| re(r"[ \t]+").replace_all(l, " ").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect::<Vec<String>>()
        .join("\n")
}

/// Cuts plain text into pieces of at most `max` characters at sentence ends (long sentences at
/// commas or spaces) (the Studio's `chunkForSpeech`; `max` 0 means the Studio's 220).
pub fn chunk_for_speech(text: &str, max: usize) -> Vec<String> {
    let limit = if max == 0 { 220 } else { max };
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let sentence = re(r#"[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$"#);
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        cur.clear();
    };
    let mut sentences: Vec<String> = Vec::new();
    for line in text.split('\n') {
        for cap in sentence.captures_iter(line) {
            sentences.push(cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default());
        }
        sentences.push("\n".to_string());
    }
    for s in &sentences {
        if s == "\n" {
            flush(&mut cur, &mut out);
            continue;
        }
        let mut rest: Vec<char> = s.chars().collect();
        while rest.len() > limit {
            let window = limit.min(rest.len() - 1);
            let mut cut: Option<usize> = rest[..=window].iter().rposition(|&c| c == ',');
            if cut.map(|c| c < limit / 3).unwrap_or(true) {
                cut = rest[..=window].iter().rposition(|&c| c == ' ');
            }
            let cut = cut.filter(|&c| c >= 1).unwrap_or(limit);
            let cut = cut.min(rest.len() - 1);
            if !cur.is_empty() {
                flush(&mut cur, &mut out);
            }
            out.push(rest[..=cut].iter().collect::<String>().trim().to_string());
            rest = rest[cut + 1..].to_vec();
        }
        let piece: String = rest.into_iter().collect();
        if cur.chars().count() + piece.chars().count() > limit {
            flush(&mut cur, &mut out);
        }
        cur.push_str(&piece);
    }
    flush(&mut cur, &mut out);
    out.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

/// Hands the text to the computer's own voice: the first of the known programs that starts takes
/// it (`spd-say`, `espeak-ng`, `espeak`, `say`, the speech synthesizer of Windows). Returns false
/// when none of them could be started — the caller then says so (the Studio's `fllmSpeak` says
/// "This browser cannot read aloud." when the browser has no voice).
pub fn speak_out(text: &str) -> bool {
    if text.trim().is_empty() {
        return true; // nothing to say is not a failure (the Studio returns early as well)
    }
    #[cfg(target_os = "macos")]
    let candidates: &[(&str, &[&str])] = &[
        ("say", &[]),
        ("spd-say", &["-w"]),
        ("espeak-ng", &[]),
        ("espeak", &[]),
    ];
    #[cfg(target_os = "windows")]
    let candidates: &[(&str, &[&str])] = &[("powershell", &["-NoProfile", "-Command"])];
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let candidates: &[(&str, &[&str])] = &[("spd-say", &["-w"]), ("espeak-ng", &[]), ("espeak", &[]), ("say", &[])];
    for (cmd, base) in candidates {
        let mut command = std::process::Command::new(cmd);
        command.args(*base);
        if *cmd == "powershell" {
            // the text goes in as a single-quoted literal; a quote inside it is doubled
            let escaped = text.replace('\'', "''");
            command.arg(format!("(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{escaped}')"));
        } else {
            command.arg(text);
        }
        if command.spawn().is_ok() {
            return true;
        }
    }
    false
}

/// One compiled expression, kept for the many uses above (md::plain uses it too).
pub(crate) fn re(pattern: &str) -> &'static Regex {
    static CACHE: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, &'static Regex>>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut cache = CACHE.lock().unwrap();
    if let Some(found) = cache.get(pattern) {
        return found;
    }
    let made = Box::leak(Box::new(Regex::new(pattern).unwrap_or_else(|e| panic!("the pattern {pattern:?} is wrong: {e}"))));
    cache.insert(pattern.to_string(), made);
    made
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The strips and cuts of the Studio against its own output (the fixtures come from
    /// voice.js, see fixtures/speech_fixtures.js).
    #[test]
    fn the_speech_helpers_are_the_studios() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/speech_fixtures.json")).expect("the fixtures are json");
        let strips = data["strips"].as_array().expect("the strips");
        assert_eq!(strips.len(), 11, "every input of the fixture list runs");
        for case in strips {
            let input = case["input"].as_str().expect("an input");
            assert_eq!(strip_for_speech(input), case["out"], "strip for {input:?}");
        }
        let chunks = data["chunks"].as_array().expect("the chunks");
        assert_eq!(chunks.len(), 7, "every cutting of the fixture list runs");
        for case in chunks {
            let text = case["text"].as_str().expect("a text");
            let max = case["max"].as_u64().map(|m| m as usize).unwrap_or(0);
            let got = chunk_for_speech(text, max);
            let want: Vec<String> = case["out"].as_array().expect("the pieces").iter().map(|s| s.as_str().unwrap().to_string()).collect();
            assert_eq!(got, want, "chunk {text:?} with {max}");
        }
    }

    #[test]
    fn a_text_without_letters_is_nothing_to_say() {
        assert!(strip_for_speech("").is_empty());
        assert!(chunk_for_speech("", 0).is_empty());
        assert!(chunk_for_speech("\n\n  \n", 100).is_empty());
    }
}
