//! Jinja environment for Hugging Face chat templates.
//!
//! Templates are written for the Python `ImmutableSandboxedEnvironment` that `transformers` builds in
//! `apply_chat_template` (`trim_blocks=True`, `lstrip_blocks=True`, the `loopcontrols` extension, `tojson` with
//! `json.dumps` keyword arguments, `raise_exception`, `strftime_now`). This module configures minijinja to behave the
//! same way and fills the gaps between the two engines that matter for printing: Python's `True`/`None` spelling and
//! `repr()` of lists and dicts, `str.zfill()` and friends.
//!
//! Known differences that cannot be closed (minijinja values are immutable and its `~` operator prints with its own
//! `Display`) are listed in `docs/changes/formats.md` and pinned by `KNOWN_DIFFERENCES` in the tests below.
use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind, State};
use std::fmt::Write;

#[cfg(test)]
pub(crate) static TEST_NOW: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// Hugging Face chat templates wrap assistant spans in `{% generation %}` ... `{% endgeneration %}`
/// (training-mask markers). They do not change the rendered text and minijinja does not know them,
/// so they become `{% if true %}` ... `{% endif %}` with the whitespace-control markers preserved.
pub fn normalize(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("%}") else {
            out.push_str(&rest[start..]);
            return out;
        };
        let inner = &after[..end];
        let core = inner.trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        let replacement = match core {
            "generation" => Some(" if true "),
            "endgeneration" => Some(" endif "),
            _ => None,
        };
        if let Some(body) = replacement {
            let lead = inner.chars().next().filter(|c| *c == '-' || *c == '+');
            let trail = inner.chars().last().filter(|c| *c == '-');
            out.push_str("{%");
            if let Some(c) = lead {
                out.push(c);
            }
            out.push_str(body);
            if let Some(c) = trail {
                out.push(c);
            }
            out.push_str("%}");
        } else {
            out.push_str(&rest[start..start + 2 + end + 2]);
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    out
}

/// The environment `transformers` uses for `apply_chat_template`, as far as minijinja can reproduce it.
pub fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(|state, value, method, args| {
        match python_methods(state, value, method, args) {
            Err(e) if e.kind() == ErrorKind::UnknownMethod => {
                minijinja_contrib::pycompat::unknown_method_callback(state, value, method, args)
            }
            other => other,
        }
    });
    env.set_formatter(|out, state, value| match value.kind() {
        ValueKind::Bool => out.write_str(if value.is_true() { "True" } else { "False" }).map_err(fmt_err),
        ValueKind::None => out.write_str("None").map_err(fmt_err),
        ValueKind::Seq | ValueKind::Map | ValueKind::Iterable => out.write_str(&py_repr(value)).map_err(fmt_err),
        _ => minijinja::escape_formatter(out, state, value),
    });
    // Python: `None` and numbers are not iterable (minijinja iterates `none` as an empty sequence)
    env.add_test("iterable", |v: Value| {
        v.is_undefined() || matches!(v.kind(), ValueKind::String | ValueKind::Seq | ValueKind::Map | ValueKind::Iterable | ValueKind::Bytes)
    });
    // Python: `len(undefined)` is 0 in jinja2
    env.add_filter("length", |v: Value| -> Result<usize, Error> {
        if v.is_undefined() {
            return Ok(0);
        }
        v.len().ok_or_else(|| Error::new(ErrorKind::InvalidOperation, format!("cannot calculate length of value of type {}", v.kind())))
    });
    // `x|trim` on a list or dict trims `str(x)` in Python; minijinja would print JSON
    env.add_filter("trim", |v: Value, chars: Option<String>| -> String {
        let text = py_str(&v);
        match chars {
            Some(chars) => text.trim_matches(|c| chars.contains(c)).to_owned(),
            None => text.trim().to_owned(),
        }
    });
    env.add_filter("tojson", crate::template_json::tojson);
    env.add_filter("string", |v: Value| -> Value {
        if v.is_safe() || v.as_str().is_some() {
            v
        } else {
            Value::from(py_str(&v))
        }
    });
    env.add_function("raise_exception", |message: String| -> Result<String, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, message))
    });
    env.add_function("strftime_now", |format: String| -> String {
        #[cfg(test)]
        {
            let ts = TEST_NOW.load(std::sync::atomic::Ordering::Relaxed);
            if ts != 0 {
                if let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) {
                    return dt.format(&format).to_string();
                }
            }
        }
        if let Ok(epoch_str) = std::env::var("FLLM_NOW").or_else(|_| std::env::var("SOURCE_DATE_EPOCH")) {
            if let Ok(ts) = epoch_str.parse::<i64>() {
                if let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) {
                    return dt.format(&format).to_string();
                }
            }
        }
        chrono::Local::now().format(&format).to_string()
    });
    env
}

fn fmt_err(_: std::fmt::Error) -> Error {
    Error::new(ErrorKind::WriteFailure, "formatting failed")
}

/// Render `template` (already passed through [`normalize`]) with the given context.
pub fn render(template: &str, context: &serde_json::Value) -> Result<String, String> {
    environment().render_str(template, context).map_err(|e| format!("{e:#}"))
}

/// `str(value)` as Python prints it: `True`, `None`, `['a', 1]`, `{'k': 'v'}`; plain strings stay as they are.
fn py_str(value: &Value) -> String {
    match value.kind() {
        ValueKind::String => value.as_str().unwrap_or_default().to_owned(),
        _ => py_repr(value),
    }
}

/// `repr(value)`.
fn py_repr(value: &Value) -> String {
    let mut out = String::new();
    write_repr(value, &mut out);
    out
}

fn write_repr(value: &Value, out: &mut String) {
    match value.kind() {
        ValueKind::Undefined => {}
        ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if value.is_true() { "True" } else { "False" }),
        ValueKind::Number => {
            let _ = write!(out, "{value}");
        }
        ValueKind::String => out.push_str(&py_quote(value.as_str().unwrap_or_default())),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            if let Ok(items) = value.try_iter() {
                for (i, item) in items.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_repr(&item, out);
                }
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            if let Ok(keys) = value.try_iter() {
                for (i, key) in keys.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_repr(&key, out);
                    out.push_str(": ");
                    if let Ok(item) = value.get_item(&key) {
                        write_repr(&item, out);
                    }
                }
            }
            out.push('}');
        }
        _ => {
            let _ = write!(out, "{value}");
        }
    }
}

/// Python `repr()` of a string: single quotes unless the text contains `'` and no `"`.
fn py_quote(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn arg_str(args: &[Value], i: usize) -> Result<Option<String>, Error> {
    match args.get(i) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_owned()))
            .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "argument must be a string")),
    }
}

fn arg_int(args: &[Value], i: usize) -> Result<Option<i64>, Error> {
    match args.get(i) {
        None => Ok(None),
        Some(v) => v
            .as_i64()
            .map(Some)
            .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "argument must be an integer")),
    }
}

fn pad_fill(args: &[Value]) -> Result<char, Error> {
    let fill = arg_str(args, 1)?.unwrap_or_else(|| " ".to_owned());
    let mut chars = fill.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(Error::new(ErrorKind::InvalidOperation, "the fill character must be exactly one character long")),
    }
}

/// Python `str` methods that minijinja-contrib's pycompat layer does not implement.
fn python_methods(_state: &State, value: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    let Some(s) = value.as_str() else {
        return Err(Error::from(ErrorKind::UnknownMethod));
    };
    let width = |args: &[Value]| -> Result<usize, Error> {
        Ok(arg_int(args, 0)?.unwrap_or(0).max(0) as usize)
    };
    let len = s.chars().count();
    match method {
        "swapcase" => Ok(Value::from(
            s.chars()
                .flat_map(|c| {
                    if c.is_uppercase() {
                        c.to_lowercase().collect::<Vec<_>>()
                    } else {
                        c.to_uppercase().collect::<Vec<_>>()
                    }
                })
                .collect::<String>(),
        )),
        "casefold" => Ok(Value::from(s.to_lowercase().replace('\u{df}', "ss"))),
        "isspace" => Ok(Value::from(!s.is_empty() && s.chars().all(char::is_whitespace))),
        "zfill" => {
            let w = width(args)?;
            if len >= w {
                return Ok(Value::from(s));
            }
            let (sign, digits) = match s.chars().next() {
                Some(c @ ('+' | '-')) => (c.to_string(), &s[1..]),
                _ => (String::new(), s),
            };
            Ok(Value::from(format!("{sign}{}{digits}", "0".repeat(w - len))))
        }
        "ljust" | "rjust" | "center" => {
            let w = width(args)?;
            let fill = pad_fill(args)?;
            if len >= w {
                return Ok(Value::from(s));
            }
            let total = w - len;
            let (left, right) = match method {
                "ljust" => (0, total),
                "rjust" => (total, 0),
                // CPython: left = pad/2 + (pad & width & 1)
                _ => {
                    let left = total / 2 + (total & w & 1);
                    (left, total - left)
                }
            };
            let pad = |n: usize| fill.to_string().repeat(n);
            Ok(Value::from(format!("{}{s}{}", pad(left), pad(right))))
        }
        "partition" | "rpartition" => {
            let sep = arg_str(args, 0)?.ok_or_else(|| Error::new(ErrorKind::MissingArgument, "separator"))?;
            if sep.is_empty() {
                return Err(Error::new(ErrorKind::InvalidOperation, "empty separator"));
            }
            let found = if method == "partition" { s.find(&sep) } else { s.rfind(&sep) };
            let parts = match found {
                Some(i) => vec![s[..i].to_owned(), sep.clone(), s[i + sep.len()..].to_owned()],
                None if method == "partition" => vec![s.to_owned(), String::new(), String::new()],
                None => vec![String::new(), String::new(), s.to_owned()],
            };
            Ok(Value::from(parts))
        }
        "rsplit" => {
            let sep = args.first().filter(|v| !v.is_none()).and_then(|v| v.as_str().map(str::to_owned));
            let max = arg_int(args, 1)?.unwrap_or(-1);
            let parts: Vec<String> = match sep {
                Some(sep) if max >= 0 => s.rsplitn(max as usize + 1, sep.as_str()).map(str::to_owned).collect::<Vec<_>>().into_iter().rev().collect(),
                Some(sep) => s.split(sep.as_str()).map(str::to_owned).collect(),
                None if max >= 0 => {
                    let mut parts = Vec::new();
                    let mut rest = s.trim_end();
                    while parts.len() < max as usize {
                        match rest.rfind(char::is_whitespace) {
                            Some(i) => {
                                let word_start = i + rest[i..].chars().next().map_or(1, char::len_utf8);
                                parts.push(rest[word_start..].to_owned());
                                rest = rest[..i].trim_end();
                            }
                            None => break,
                        }
                    }
                    parts.push(rest.trim_start().to_owned());
                    parts.reverse();
                    parts
                }
                None => s.split_whitespace().map(str::to_owned).collect(),
            };
            Ok(Value::from(parts))
        }
        "removeprefix" => {
            let p = arg_str(args, 0)?.unwrap_or_default();
            Ok(Value::from(s.strip_prefix(p.as_str()).unwrap_or(s)))
        }
        "removesuffix" => {
            let p = arg_str(args, 0)?.unwrap_or_default();
            Ok(Value::from(s.strip_suffix(p.as_str()).unwrap_or(s)))
        }
        "index" | "rindex" => {
            let needle = arg_str(args, 0)?.unwrap_or_default();
            let found = if method == "index" { s.find(&needle) } else { s.rfind(&needle) };
            match found {
                Some(i) => Ok(Value::from(s[..i].chars().count())),
                None => Err(Error::new(ErrorKind::InvalidOperation, "substring not found")),
            }
        }
        "istitle" => {
            let mut previous_cased = false;
            let mut any_cased = false;
            for c in s.chars() {
                if c.is_uppercase() {
                    if previous_cased {
                        return Ok(Value::from(false));
                    }
                    previous_cased = true;
                    any_cased = true;
                } else if c.is_lowercase() {
                    if !previous_cased {
                        return Ok(Value::from(false));
                    }
                    any_cased = true;
                } else {
                    previous_cased = false;
                }
            }
            Ok(Value::from(any_cased))
        }
        "expandtabs" => {
            let tab = arg_int(args, 0)?.unwrap_or(8).max(0) as usize;
            let mut out = String::new();
            let mut column = 0usize;
            for c in s.chars() {
                match c {
                    '\t' => {
                        if tab > 0 {
                            let n = tab - column % tab;
                            out.push_str(&" ".repeat(n));
                            column += n;
                        }
                    }
                    '\n' | '\r' => {
                        out.push(c);
                        column = 0;
                    }
                    c => {
                        out.push(c);
                        column += 1;
                    }
                }
            }
            Ok(Value::from(out))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value as Json};
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("chat_templates")
    }

    /// Probes whose output differs from CPython's jinja2 and cannot be fixed with minijinja (name -> reason).
    /// The test fails if an entry stops differing, so the list cannot go stale.
    const KNOWN_DIFFERENCES: &[(&str, &str)] = &[
        ("str_percent", "printf-style `%` on strings is not an operator in minijinja"),
        ("e_float_format", "printf-style `%` on strings is not an operator in minijinja"),
        ("str_partition", "minijinja has no tuple type: Python prints (a, b), minijinja [a, b]"),
        ("e_tuple", "minijinja has no tuple type: Python prints (a, b), minijinja [a, b]"),
        ("f_items_dictsort", "dict items are tuples in Python, lists in minijinja (printing only)"),
        ("f_dictsort_by_value", "dict items are tuples in Python, lists in minijinja (printing only)"),
        ("str_encode", "no bytes type"),
        ("f_join_attr", "join(attribute=) is not supported"),
        ("f_abs_round_int_float", "round(precision, method) and int(default) take fewer arguments in minijinja"),
        ("f_truncate", "filter truncate is not registered"),
        ("f_wordcount_center", "filters wordcount/center are not registered"),
        ("f_urlencode_escape", "filter urlencode is not registered"),
        ("f_filesizeformat", "filter filesizeformat is not registered"),
        ("f_pprint", "pprint prints JSON-like text"),
        ("f_attr", "|attr on a dict returns the key in minijinja, undefined in jinja2"),
        ("e_negative_index_msgs", "|attr on a dict returns the key in minijinja, undefined in jinja2"),
        ("f_xmlattr", "filter xmlattr is not registered"),
        ("f_safe_striptags", "filter striptags is not registered"),
        ("f_forceescape", "filter forceescape is not registered"),
        ("t_callable_string", "test callable is not registered"),
        ("e_global_functions", "globals lipsum/cycler/joiner are not registered"),
        ("e_dict_func", "dict() accepts keyword arguments only"),
        ("e_joiner", "global joiner is not registered"),
        ("e_cycler", "global cycler is not registered"),
    ];

    fn read_json(path: &std::path::Path) -> Json {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
    }

    #[test]
    fn probes_match_python_jinja2() {
        let dir = data_dir();
        let doc = read_json(&dir.join("probes.json"));
        let probes = &doc["probes"];
        let golden = read_json(&dir.join("probes.golden.json"));
        let mut failures = Vec::new();
        let mut known_seen = Vec::new();
        for probe in probes.as_array().unwrap() {
            let name = probe["name"].as_str().unwrap();
            let template = normalize(probe["template"].as_str().unwrap());
            let expected = &golden[name];
            let actual = render(&template, &doc["vars"]);
            let matches = match (expected.get("ok"), &actual) {
                (Some(ok), Ok(text)) => ok.as_str() == Some(text.as_str()),
                (None, Err(_)) => true,
                _ => false,
            };
            let known = KNOWN_DIFFERENCES.iter().any(|(n, _)| *n == name);
            if known {
                known_seen.push(name.to_owned());
            }
            if !matches && !known {
                failures.push(format!("{name}\n  template: {}\n  python:   {expected}\n  rust:     {actual:?}", probe["template"]));
            } else if matches && known {
                failures.push(format!("{name}: listed in KNOWN_DIFFERENCES but now matches"));
            }
        }
        assert!(failures.is_empty(), "{} probe(s) differ from jinja2:\n{}", failures.len(), failures.join("\n"));
        assert_eq!(known_seen.len(), KNOWN_DIFFERENCES.len(), "KNOWN_DIFFERENCES names an unknown probe");
    }

    /// Renders every template under `dir` for every case in `cases.json` and compares with the `<name>.golden.json`
    /// files that `scripts/chat_template_golden.py golden` produced with transformers' jinja2 environment.
    pub(crate) fn set_test_now(ts: i64) {
        TEST_NOW.store(ts, std::sync::atomic::Ordering::Relaxed);
    }

    struct ResetTestNow;
    impl Drop for ResetTestNow {
        fn drop(&mut self) {
            set_test_now(0);
        }
    }

    fn compare_corpus(dir: &std::path::Path, cases: &Json) -> (usize, Vec<String>) {
        set_test_now(1792065600); // 2026-10-15 12:00:00 UTC (golden fixture generation date)
        let _reset = ResetTestNow;
        let mut compared = 0usize;
        let mut failures = Vec::new();
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        files.sort();
        for path in files {
            let file = path.file_name().unwrap().to_string_lossy().to_string();
            let Some(stem) = file.strip_suffix(".jinja") else { continue };
            let (name, tname) = stem.split_once('@').unwrap_or((stem, "default"));
            let template = normalize(&std::fs::read_to_string(&path).unwrap());
            let meta = read_json(&dir.join(format!("{name}.meta.json")));
            let golden = read_json(&dir.join(format!("{name}.golden.json")));
            for case in cases.as_array().unwrap() {
                let case_name = case["name"].as_str().unwrap();
                let key = format!("{tname}/{case_name}");
                let Some(expected) = golden.get(&key) else { continue };
                // the server only selects a `tool_use` template when the request carries tools
                if tname == "tool_use" && case.get("tools").is_none() {
                    continue;
                }
                let mut ctx = json!({
                    "messages": case["messages"],
                    "add_generation_prompt": case.get("add_generation_prompt").cloned().unwrap_or(json!(true)),
                    "bos_token": meta["bos"].as_str().unwrap_or(""),
                    "eos_token": meta["eos"].as_str().unwrap_or(""),
                    "pad_token": meta["pad"].as_str().unwrap_or(""),
                    "unk_token": meta["unk"].as_str().unwrap_or(""),
                });
                // transformers passes `tools=None` / `documents=None` when absent; minijinja sees JSON null the same way
                ctx["tools"] = case.get("tools").cloned().unwrap_or(Json::Null);
                ctx["documents"] = case.get("documents").cloned().unwrap_or(Json::Null);
                if let Some(kwargs) = case.get("kwargs").and_then(Json::as_object) {
                    for (k, v) in kwargs {
                        ctx[k] = v.clone();
                    }
                }
                let actual = render(&template, &ctx);
                compared += 1;
                let ok = match (expected.get("ok"), &actual) {
                    (Some(ok), Ok(text)) => ok.as_str() == Some(text.as_str()),
                    (None, Err(_)) => true,
                    _ => false,
                };
                if !ok {
                    failures.push(format!("{key} ({name})\n  python: {expected}\n  rust:   {actual:?}"));
                }
            }
        }
        (compared, failures)
    }

    #[test]
    fn corpus_matches_transformers() {
        let dir = data_dir().join("templates");
        let cases = read_json(&data_dir().join("cases.json"));
        let (compared, failures) = compare_corpus(&dir, &cases);
        assert!(compared > 100, "only {compared} template renderings were compared");
        assert!(failures.is_empty(), "{} of {compared} renderings differ:\n{}", failures.len(), failures.join("\n"));
    }

    /// Full corpus including templates whose licence does not allow redistribution in this repository:
    /// `FLLM_CHAT_TEMPLATE_CORPUS=/path cargo test -p fllm-engine --release chat_template::tests::external_corpus`
    #[test]
    fn external_corpus_matches_transformers() {
        let Ok(dir) = std::env::var("FLLM_CHAT_TEMPLATE_CORPUS") else { return };
        let cases = read_json(&data_dir().join("cases.json"));
        let (compared, failures) = compare_corpus(std::path::Path::new(&dir), &cases);
        assert!(failures.is_empty(), "{} of {compared} renderings differ:\n{}", failures.len(), failures.join("\n"));
        eprintln!("external corpus: {compared} renderings identical to transformers");
    }

    #[test]
    fn generation_tags_become_noop_blocks_and_keep_whitespace_control() {
        let template = "a{%- generation -%}b{% endgeneration %}c{%- endgeneration %}";
        assert_eq!(normalize(template), "a{%- if true -%}b{% endif %}c{%- endif %}");
        assert_eq!(normalize("{% if x %}y{% endif %}"), "{% if x %}y{% endif %}");
    }

    #[test]
    fn python_values_print_like_python() {
        let out = render("{{ x }} {{ y }} {{ z }} {{ w }}", &json!({"x": true, "y": null, "z": ["a", 1, null], "w": {"k": "it's"}})).unwrap();
        assert_eq!(out, "True None ['a', 1, None] {'k': \"it's\"}");
    }
}
