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

/// The short tone of the alerts (`chime` of `voice.js`: two sine tones, 880 Hz and 1175 Hz, each
/// with a quick rise and a fall). The wave is written next and handed to the first player that
/// starts (`aplay`, `paplay`, `afplay`, `afplay`-like tools, the sound of Windows); without one
/// there is no tone, which is not a failure the caller needs to hear about.
pub fn chime() {
    let sample_rate: u32 = 44_100;
    let seconds: u32 = 1;
    let mut samples: Vec<i16> = Vec::with_capacity((sample_rate * seconds) as usize);
    // the two tones of the Studio: 880 Hz from the start, 1175 Hz after 0.14 seconds
    for i in 0..(sample_rate * seconds) {
        let t = i as f32 / sample_rate as f32;
        let mut v = 0.0f32;
        for (hz, at) in [(880.0f32, 0.0f32), (1175.0f32, 0.14f32)] {
            let dt = t - at;
            if !(0.0..0.22).contains(&dt) {
                continue;
            }
            // the envelope of the Studio: rise in 0.02 seconds, fall from 0.2
            let env = if dt < 0.02 {
                (dt / 0.02).powf(4.0) * 0.12
            } else {
                (1.0 - ((dt - 0.02) / 0.18)).clamp(0.0, 1.0).powf(2.0) * 0.12
            };
            v += (2.0 * std::f32::consts::PI * hz * dt).sin() * env;
        }
        samples.push((v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
    }
    let mut wav: Vec<u8> = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    fn tag(wav: &mut Vec<u8>, id: &[u8; 4], len: u32) {
        wav.extend_from_slice(id);
        wav.extend_from_slice(&len.to_le_bytes());
    }
    tag(&mut wav, b"RIFF", 36 + data_len);
    tag(&mut wav, b"WAVE", 0);
    tag(&mut wav, b"fmt ", 16);
    wav.extend_from_slice(&1u16.to_le_bytes()); // pcm
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    tag(&mut wav, b"data", data_len);
    for s in samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    let dir = std::env::temp_dir();
    let path = dir.join(format!("fllm-chime-{}.wav", std::process::id()));
    if std::fs::write(&path, &wav).is_err() {
        return;
    }
    let players: &[(&str, &[&str])] = &[
        ("aplay", &["-q"]),
        ("paplay", &[]),
        ("afplay", &[]),
        ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("powershell", &[]),
    ];
    for (cmd, base) in players {
        let mut c = std::process::Command::new(cmd);
        c.args(*base);
        if *cmd == "powershell" {
            c.arg(format!(
                "(New-Object Media.SoundPlayer '{}').PlaySync()",
                path.display()
            ));
        } else {
            c.arg(&path);
        }
        if c.spawn().is_ok() {
            return;
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// The system notification of the alerts (`announce` of `voice.js`: the title *fLLM*, the body, one
/// tag so a second one replaces the first). The first of the known programs that starts takes it;
/// without one there is no notification.
pub fn notify(body: &str) {
    let body = body.replace(['\'', '"'], " ");
    let body: String = body.chars().take(160).collect();
    #[cfg(target_os = "macos")]
    let candidates: &[(&str, &[&str])] = &[(
        "osascript",
        &["-e", &format!("display notification \"{body}\" with title \"fLLM\"")],
    )];
    #[cfg(target_os = "windows")]
    let candidates: &[(&str, &[&str])] = &[(
        "powershell",
        &["-NoProfile", "-Command", &format!(
            "(New-Object -ComObject Wscript.Shell).Popup(\"{body}\", 10, \"fLLM\", 0x40)"
        )],
    )];
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let candidates: &[(&str, &[&str])] = &[(
        "notify-send",
        &["--app-name=fLLM", "--replace-id=1", "fLLM", body.as_str()],
    )];
    for (cmd, args) in candidates {
        if std::process::Command::new(cmd).args(*args).spawn().is_ok() {
            return;
        }
    }
}

/// The voice names this computer can say (`getVoices` of the Studio, read from the speech
/// programs instead of a browser): one per line of `spd-say --list-voices` or `espeak --voices`,
/// sorted by name. Without a program there is no list, and that is not an error — the settings
/// page then says that this computer cannot read aloud.
pub fn list_voices() -> Vec<String> {
    let out = |cmd: &str, args: &[&str], pick: fn(&str) -> Option<String>| {
        let text = std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())?;
        let mut names: Vec<String> = text
            .lines()
            .filter_map(pick)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if names.is_empty() {
            None
        } else {
            names.sort();
            names.dedup();
            Some(names)
        }
    };
    // speech-dispatcher: a table whose third word is the name
    if let Some(names) = out("spd-say", &["--list-voices"], |line| {
        line.split_whitespace().nth(2).map(str::to_string)
    }) {
        return names;
    }
    // espeak: the columns after the index, the name ends before the language
    if let Some(names) = out("espeak-ng", &["--voices"], |line| {
        parse_espeak_voices(line)
    }) {
        return names;
    }
    out("espeak", &["--voices"], parse_espeak_voices).unwrap_or_default()
}

/// One line of `espeak --voices` (`Pty Language Age/Gender VoiceName File Other`): the voice name
/// is the fifth word.
fn parse_espeak_voices(line: &str) -> Option<String> {
    let mut parts = line.split_whitespace();
    let first = parts.next()?;
    if !first.chars().all(|c| c.is_ascii_digit()) {
        return None; // the header line
    }
    parts.next()?; // language
    parts.next()?; // age/gender
    parts.next().map(str::to_string)
}

/// Whether this computer can show a system notification (the Studio asks the browser for the
/// permission; here the question is whether a program for it is installed).
pub fn can_notify() -> bool {
    [
        ("notify-send", &[][..]),
        ("osascript", &["-e", "display notification \"x\""][..]),
        ("powershell", &["-NoProfile", "-Command", "1"][..]),
    ]
    .iter()
    .any(|(cmd, args)| std::process::Command::new(cmd).args(*args).spawn().is_ok())
}

/// The languages of the dictation select in Settings, Voice and alerts (`LANGUAGES` of
/// `voice.js`: empty = the language of the system).
pub const LANGUAGES: [(&str, &str); 17] = [
    ("", "Browser language"),
    ("en-US", "English (US)"),
    ("en-GB", "English (UK)"),
    ("de-DE", "German"),
    ("fr-FR", "French"),
    ("es-ES", "Spanish"),
    ("it-IT", "Italian"),
    ("pt-BR", "Portuguese (Brazil)"),
    ("nl-NL", "Dutch"),
    ("pl-PL", "Polish"),
    ("ru-RU", "Russian"),
    ("tr-TR", "Turkish"),
    ("ja-JP", "Japanese"),
    ("ko-KR", "Korean"),
    ("zh-CN", "Chinese (Mandarin)"),
    ("hi-IN", "Hindi"),
    ("ar-SA", "Arabic"),
];

/// The arguments one speech program gets for the chosen voice (`rate` and `pitch` of the Studio
/// are 0.5 to 2, 1 is normal; every program reads them on its own scale). Pure, so the mapping
/// itself is what the tests check.
pub fn voice_args(program: &str, v: &crate::store::VoicePrefs) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let rate = v.rate.clamp(0.5, 2.0);
    let pitch = v.pitch.clamp(0.5, 2.0);
    match program {
        // speech-dispatcher: both scales run from -100 to 100, 0 is normal
        "spd-say" => {
            out.push("-r".into());
            out.push(format!("{}", ((rate - 1.0) * 100.0).round() as i32));
            out.push("-p".into());
            out.push(format!("{}", ((pitch - 1.0) * 100.0).round() as i32));
            if !v.lang.is_empty() {
                out.push("-l".into());
                out.push(v.lang.clone());
            }
        }
        // espeak: words per minute (175 is normal) and pitch 0 to 99 (50 is normal)
        "espeak-ng" | "espeak" => {
            out.push("-s".into());
            out.push(format!("{}", (rate * 175.0).round() as i32));
            out.push("-p".into());
            out.push(format!("{}", (pitch * 50.0).round().clamp(0.0, 99.0) as i32));
            if !v.voice.is_empty() {
                out.push("-v".into());
                out.push(v.voice.clone());
            } else if !v.lang.is_empty() {
                out.push("-v".into());
                out.push(v.lang.clone());
            }
        }
        // the synthesizer of macOS: words per minute, pitch 0 to 20 (10 is normal)
        "say" => {
            out.push("-r".into());
            out.push(format!("{}", (rate * 175.0).round() as i32));
            out.push("-p".into());
            out.push(format!("{}", (pitch * 10.0).round().clamp(0.0, 20.0) as i32));
            if !v.voice.is_empty() {
                out.push("-v".into());
                out.push(v.voice.clone());
            }
        }
        // Windows: the synthesizer's own scales (-10 to 10, 0 is normal)
        "powershell" => {
            let rs = ((rate - 1.0) * 10.0).round() as i32;
            let ps = ((pitch - 1.0) * 10.0).round() as i32;
            let voice = if v.voice.is_empty() {
                String::new()
            } else {
                format!("$s.SelectVoice('{}');", v.voice.replace('\'', "''"))
            };
            let lang = if v.lang.is_empty() {
                String::new()
            } else {
                format!("$s.Locale='{}';", v.lang.replace('\'', "''"))
            };
            out.push(format!(
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate={rs}; $s.Pitch={ps}; {voice}{lang}"
            ));
        }
        _ => {}
    }
    out
}

/// Hands the text to the computer's own voice: the first of the known programs that starts takes
/// it (`spd-say`, `espeak-ng`, `espeak`, `say`, the speech synthesizer of Windows). Returns false
/// when none of them could be started — the caller then says so (the Studio's `fllmSpeak` says
/// "This browser cannot read aloud." when the browser has no voice).
pub fn speak_out(text: &str) -> bool {
    speak_with(text, &crate::store::VoicePrefs::default())
}

/// The same, with the chosen voice: every program gets the language, the voice, how fast and how
/// high on its own scale (`voice_args`), and the text as its last argument.
pub fn speak_with(text: &str, v: &crate::store::VoicePrefs) -> bool {
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
        let args = voice_args(cmd, v);
        if *cmd == "powershell" {
            // the settings and the text go in as one script, the text as a single-quoted literal
            let escaped = text.replace('\'', "''");
            let head = args.into_iter().next().unwrap_or_default();
            command.arg(format!("{head}$s.Speak('{escaped}');"));
        } else {
            command.args(args);
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
