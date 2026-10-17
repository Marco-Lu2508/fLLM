//! Tool-call parsers: one implementation per model family behind the [`ToolParser`] trait.
//!
//! Every family wraps tool calls in its own markers. The server feeds the *accumulated* raw
//! text of a choice to [`extract`] after every token, so a parser is a pure function of that
//! text: half-written markers stay buffered, complete spans become calls, and anything that
//! cannot be decoded is handed back as plain content instead of failing the request.
//! [`extract`] guarantees that `content` only ever grows and that the list of calls only
//! ever gets longer, which is what makes streamed deltas safe.
//!
//! Many model families put their markers into *control* tokens that the detokenizer drops.
//! [`ToolParser::markers`] lists every literal that has to survive detokenization; the server
//! keeps the matching token ids visible (see `TokenizerBundle::decode_keeping`).
//!
//! Formats and their provenance are listed in `docs/server-api.md` ("Tools und Reasoning").
//! Formats taken from a chat template that ships with the model (or with this repository
//! under `models/templates`) are *derived*; everything else carries an `// UNVERIFIED:`
//! comment where it is implemented.

use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------

/// One argument of an XML-style call, before it is typed.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    /// Already JSON (JSON-bodied formats, DSML `string="false"` after parsing, ...).
    Json(Value),
    /// Plain text; typed afterwards from the tool's parameter schema.
    Text(String),
    /// Text plus the type the model declared for it (`<ifm|arg_type>`, DSML `string=`).
    Typed(String, String),
}

/// The arguments of a decoded call.
#[derive(Clone, Debug, PartialEq)]
pub enum CallArgs {
    /// A complete JSON value (normally an object; a JSON string is parsed later).
    Json(Value),
    /// Named parameters of an XML-style format, in the order the model wrote them.
    Params(Vec<(String, Arg)>),
}

/// One tool call recovered from raw model output.
#[derive(Clone, Debug, PartialEq)]
pub struct RawCall {
    /// Function name as written by the model.
    pub name: String,
    pub args: CallArgs,
}

impl RawCall {
    pub fn json(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            args: CallArgs::Json(arguments),
        }
    }
    pub fn params(name: impl Into<String>, params: Vec<(String, Arg)>) -> Self {
        Self {
            name: name.into(),
            args: CallArgs::Params(params),
        }
    }
    /// Arguments as compact JSON text without a schema: texts are guessed (`3` is a number).
    pub fn arguments_text(&self) -> Result<String, String> {
        arguments_json(&self.args, None).map(|v| v.to_string())
    }
}

/// Result of scanning accumulated model text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Extracted {
    /// Everything that is not part of a tool call, never containing half a marker.
    pub content: String,
    /// Complete, decoded calls, in generation order.
    pub calls: Vec<RawCall>,
}

/// An opening literal and the literal that closes its span (`None`: the span is one JSON value).
pub type MarkerPair = (&'static str, Option<&'static str>);

/// A parser for one model family.
///
/// Implementations describe *where* a call span starts and ends and how its body is laid
/// out; buffering, fallback-to-content and marker safety live in [`extract`].
pub trait ToolParser: Send + Sync {
    /// Stable id, used by `--tool-parser` and in the documentation.
    fn id(&self) -> &'static str;

    /// Opening and closing literals of a call span. A parser may list several pairs (the
    /// earliest opening wins); a span with no closing literal ends with its JSON value.
    fn pairs(&self) -> &'static [MarkerPair];

    /// Literals inside a span that must stay visible while detokenizing but are not span
    /// markers (separators and nested tags that are special tokens in most tokenizers).
    fn extra_markers(&self) -> &'static [&'static str] {
        &[]
    }

    /// Every literal of the format (span markers and extras): none of them may be split
    /// across two streamed chunks.
    fn hold_literals(&self) -> Vec<&'static str> {
        let mut all = Vec::new();
        for (open, close) in self.pairs() {
            if !open.is_empty() {
                all.push(*open);
            }
            if let Some(close) = close {
                all.push(*close);
            }
        }
        all.extend_from_slice(self.extra_markers());
        all.sort_unstable();
        all.dedup();
        all
    }

    /// Every literal the server must keep when detokenizing, because the vocabulary stores it
    /// as a control token that plain decoding drops. Literals that are not tokens are ignored.
    fn markers(&self) -> Vec<&'static str> {
        self.hold_literals()
    }

    /// `true` when the marker must be followed by a JSON value; occurrences in prose
    /// (`functools` is an ordinary word) are then skipped instead of buffered.
    fn requires_json_payload(&self) -> bool {
        false
    }

    /// `true` when a span that is still open when generation ends is decoded anyway.
    /// Models stop on the closing marker (`<|call|>`, `</s>`) more often than they write it.
    /// A body that does not decode still comes back as content, unchanged.
    fn lenient_end(&self) -> bool {
        true
    }

    /// Byte index of the next span-opening marker at or after `from`.
    ///
    /// A marker at the very end of the text is returned even when the payload has not
    /// started yet, so the extractor buffers it instead of streaming it out.
    fn find_open(&self, text: &str, from: usize) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (open, _) in self.pairs() {
            if open.is_empty() {
                continue;
            }
            let mut at = from;
            while let Some(offset) = text[at..].find(open) {
                let start = at + offset;
                let after = start + open.len();
                if self.requires_json_payload() {
                    match text[after..].chars().find(|c| !c.is_whitespace()) {
                        Some('[' | '{') | None => {}
                        Some(_) => {
                            at = after;
                            continue;
                        }
                    }
                }
                if best.is_none_or(|b| start < b) {
                    best = Some(start);
                }
                break;
            }
        }
        best
    }

    /// The marker pair whose opening literal is at `start`.
    fn pair_at(&self, text: &str, start: usize) -> MarkerPair {
        let mut found: Option<MarkerPair> = None;
        for pair in self.pairs() {
            if !pair.0.is_empty()
                && text[start..].starts_with(pair.0)
                && found.is_none_or(|f| pair.0.len() > f.0.len())
            {
                found = Some(*pair);
            }
        }
        found.unwrap_or(("", None))
    }

    /// Byte index just after the span that starts at `start`, or `None` while the span is
    /// still incomplete.
    fn span_end(&self, text: &str, start: usize) -> Option<usize> {
        let (open, close) = self.pair_at(text, start);
        let after_open = start + open.len();
        match close {
            Some(close) => text[after_open..]
                .find(close)
                .map(|offset| after_open + offset + close.len()),
            None => json_span_end(text, after_open),
        }
    }

    /// The span body between the markers (default: everything inside them).
    fn payload<'t>(&self, text: &'t str, start: usize, end: usize) -> &'t str {
        let (open, close) = self.pair_at(text, start);
        let from = start + open.len();
        let to = match close {
            Some(close) if text[..end].ends_with(close) => end - close.len(),
            _ => end,
        };
        &text[from.min(to)..to]
    }

    /// Decode one span body. `Err` means the body is malformed; the extractor then returns
    /// the span verbatim as content, so a broken call never fails a request.
    fn decode(&self, payload: &str) -> Result<Vec<RawCall>, String>;
}

// ---------------------------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------------------------

/// Scan accumulated model text into visible content plus complete tool calls.
///
/// * a span whose end has not arrived stays buffered while generation runs; when generation
///   is `complete` it is decoded if the parser is lenient, and returned as content otherwise;
/// * a span whose body does not decode is returned as content, unchanged;
/// * a trailing partial marker is never part of `content`.
pub fn extract(parser: &dyn ToolParser, text: &str, complete: bool) -> Extracted {
    extract_filtered(parser, text, complete, &|_| true)
}

/// [`extract`] that only accepts spans whose calls all satisfy `accept` (the server passes
/// "the function is declared in the request"); any other span stays in `content`, unchanged.
pub fn extract_filtered(
    parser: &dyn ToolParser,
    text: &str,
    complete: bool,
    accept: &dyn Fn(&RawCall) -> bool,
) -> Extracted {
    let mut out = Extracted::default();
    let mut cursor = 0usize;
    loop {
        let Some(start) = parser.find_open(text, cursor) else {
            out.content.push_str(&text[cursor..]);
            break;
        };
        out.content.push_str(&text[cursor..start]);
        let Some(end) = parser.span_end(text, start) else {
            if complete {
                match parser
                    .lenient_end()
                    .then(|| parser.decode(parser.payload(text, start, text.len())))
                {
                    Some(Ok(calls)) if !calls.is_empty() && calls.iter().all(accept) => {
                        out.calls.extend(calls)
                    }
                    // Truncated or broken span: hand the remainder back untouched.
                    _ => out.content.push_str(&text[start..]),
                }
            }
            break;
        };
        match parser.decode(parser.payload(text, start, end)) {
            Ok(calls) if !calls.is_empty() && calls.iter().all(accept) => out.calls.extend(calls),
            _ => out.content.push_str(&text[start..end]),
        }
        if end <= cursor {
            // Cannot happen for a span that starts at or after `cursor`; never loop forever.
            out.content.push_str(&text[end.max(cursor)..]);
            break;
        }
        cursor = end;
    }
    hold_marker_prefix(&mut out.content, &parser.hold_literals(), complete);
    out
}

/// Hold back a trailing partial marker so a chunk boundary never splits one.
///
/// Only *proper* prefixes are held: a marker that was already skipped as prose stays
/// visible, and a complete marker always belongs to a span that the extractor buffers.
fn hold_marker_prefix(content: &mut String, markers: &[&str], complete: bool) {
    if complete {
        return;
    }
    let mut hold = 0usize;
    for marker in markers {
        for len in (1..marker.len()).rev() {
            if len <= hold {
                break;
            }
            if !marker.is_char_boundary(len) {
                continue;
            }
            if content.ends_with(&marker[..len]) {
                hold = len;
                break;
            }
        }
    }
    if hold > 0 && content.is_char_boundary(content.len() - hold) {
        content.truncate(content.len() - hold);
    }
}

// ---------------------------------------------------------------------------------------------
// Shared building blocks
// ---------------------------------------------------------------------------------------------

/// Scan a JSON object/array starting at `from`; `None` while it is still incomplete.
///
/// Strings, escapes and nesting are tracked, so a `}` inside an argument does not end the
/// span. Structurally broken JSON also yields `None`: the extractor buffers it and, once
/// generation ends, hands the text back unchanged.
pub(crate) fn json_span_end(text: &str, from: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = from;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    if !matches!(bytes.get(index), Some(b'[' | b'{')) {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        index += 1;
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// One call object: `{"name":…,"arguments":…}`, Llama's `parameters`, OpenAI's nested
/// `{"type":"function","function":{…}}` and the `args` spelling.
fn json_call(item: &Value) -> Result<RawCall, String> {
    let item = match item.get("function") {
        Some(inner) if inner.is_object() => inner,
        _ => item,
    };
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| "tool call has no function name".to_owned())?;
    let raw = ["arguments", "parameters", "args"]
        .iter()
        .find_map(|key| item.get(*key))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = match raw {
        Value::String(inner) => serde_json::from_str::<Value>(&inner)
            .map_err(|e| format!("tool arguments are not JSON: {e}"))?,
        Value::Null => json!({}),
        value => value,
    };
    Ok(RawCall::json(name, arguments))
}

/// Decode a span body made of JSON values: calls objects, arrays of them, or several values
/// one after another (MiniMax-M1 writes one object per line; Llama separates with `;`).
fn json_calls(payload: &str) -> Result<Vec<RawCall>, String> {
    let mut calls = Vec::new();
    let mut rest = payload;
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',' || c == ';');
        if rest.is_empty() {
            break;
        }
        let mut values = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        let value = match values.next() {
            Some(Ok(value)) => value,
            Some(Err(e)) => return Err(format!("invalid tool call JSON: {e}")),
            None => break,
        };
        let used = values.byte_offset();
        rest = &rest[used..];
        match value {
            Value::Array(items) => {
                for item in &items {
                    calls.push(json_call(item)?);
                }
            }
            item => calls.push(json_call(&item)?),
        }
    }
    if calls.is_empty() {
        return Err("no tool call in the span".into());
    }
    Ok(calls)
}

/// Text between `open` and `close` starting the search at `from`: `(inner, index after close)`.
fn between<'t>(text: &'t str, from: usize, open: &str, close: &str) -> Option<(&'t str, usize)> {
    let start = text.get(from..)?.find(open)? + from + open.len();
    let end = text.get(start..)?.find(close)? + start;
    Some((&text[start..end], end + close.len()))
}

/// Value of `name="…"` (or `name='…'`) in a tag's attribute text.
fn attribute<'t>(attrs: &'t str, name: &str) -> Option<&'t str> {
    let mut from = 0;
    while let Some(offset) = attrs[from..].find(name) {
        let at = from + offset;
        let before_ok = at == 0 || attrs[..at].ends_with(|c: char| c.is_whitespace());
        let rest = attrs[at + name.len()..].trim_start();
        if before_ok && rest.starts_with('=') {
            let rest = rest[1..].trim_start();
            let quote = rest.chars().next().filter(|c| matches!(c, '"' | '\''))?;
            let body = &rest[1..];
            let end = body.find(quote)?;
            return Some(&body[..end]);
        }
        from = at + name.len();
    }
    None
}

/// `true` for names a function can have; keeps prose out of the name-only formats.
fn plausible_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':' | '/'))
}

// ---------------------------------------------------------------------------------------------
// Argument typing
// ---------------------------------------------------------------------------------------------

/// JSON types a parameter schema allows, from `type`, `anyOf`/`oneOf` and `enum`.
fn schema_types(schema: Option<&Value>) -> Vec<String> {
    fn collect(schema: &Value, out: &mut Vec<String>, depth: usize) {
        if depth > 3 {
            return;
        }
        match schema.get("type") {
            Some(Value::String(t)) => out.push(t.clone()),
            Some(Value::Array(ts)) => {
                out.extend(ts.iter().filter_map(Value::as_str).map(str::to_owned))
            }
            _ => {}
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(options) = schema.get(key).and_then(Value::as_array) {
                for option in options {
                    collect(option, out, depth + 1);
                }
            }
        }
    }
    let mut out = Vec::new();
    if let Some(schema) = schema {
        collect(schema, &mut out, 0);
    }
    out
}

/// Interpret text without a schema: JSON literals become values, everything else stays text.
fn guess_value(text: &str) -> Value {
    let trimmed = text.trim();
    match trimmed {
        "true" | "True" => return Value::Bool(true),
        "false" | "False" => return Value::Bool(false),
        "null" | "None" => return Value::Null,
        _ => {}
    }
    let numeric = trimmed
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit() || c == '-');
    if numeric || trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            if !value.is_string() {
                return value;
            }
        }
    }
    Value::String(text.to_owned())
}

/// Interpret `text` as one of the schema's types, the strictest first; text is the fallback.
fn coerce_text(text: &str, schema: Option<&Value>) -> Value {
    let types = schema_types(schema);
    if types.is_empty() {
        return guess_value(text);
    }
    let trimmed = text.trim();
    for ty in types.iter().filter(|t| t.as_str() != "string") {
        let parsed = match ty.as_str() {
            "integer" => trimmed
                .parse::<i64>()
                .map(Value::from)
                .ok()
                .or_else(|| trimmed.parse::<u64>().map(Value::from).ok()),
            "number" => trimmed.parse::<i64>().map(Value::from).ok().or_else(|| {
                trimmed
                    .parse::<f64>()
                    .ok()
                    .and_then(|f| serde_json::Number::from_f64(f).map(Value::Number))
            }),
            "boolean" => match trimmed {
                "true" | "True" => Some(Value::Bool(true)),
                "false" | "False" => Some(Value::Bool(false)),
                _ => None,
            },
            "null" => matches!(trimmed, "null" | "None").then_some(Value::Null),
            "array" => serde_json::from_str::<Value>(trimmed)
                .ok()
                .filter(Value::is_array),
            "object" => serde_json::from_str::<Value>(trimmed)
                .ok()
                .filter(Value::is_object),
            _ => None,
        };
        if let Some(value) = parsed {
            return value;
        }
    }
    Value::String(text.to_owned())
}

/// Turn decoded arguments into the JSON object of the API call, typing text parameters with
/// the tool's parameter schema (`parameters` of the declared function).
pub fn arguments_json(args: &CallArgs, parameters: Option<&Value>) -> Result<Value, String> {
    match args {
        CallArgs::Json(Value::String(text)) => {
            serde_json::from_str(text).map_err(|e| format!("tool arguments are not JSON: {e}"))
        }
        CallArgs::Json(value) => Ok(value.clone()),
        CallArgs::Params(params) => {
            let mut object = Map::new();
            for (key, arg) in params {
                let property = parameters
                    .and_then(|p| p.get("properties"))
                    .and_then(|p| p.get(key));
                let value = match arg {
                    Arg::Json(value) => value.clone(),
                    Arg::Text(text) => coerce_text(text, property),
                    Arg::Typed(text, ty) => match ty.as_str() {
                        "any" | "" => coerce_text(text, property),
                        declared => coerce_text(text, Some(&json!({ "type": declared }))),
                    },
                };
                object.insert(key.clone(), value);
            }
            Ok(Value::Object(object))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Body decoders (one per wire format)
// ---------------------------------------------------------------------------------------------

type Body = fn(&str) -> Result<Vec<RawCall>, String>;

/// Hermes / Qwen / Granite-4 / GLM / Qwen3-Coder: everything between `<tool_call>` markers.
/// The three families share the marker, so the body decides: a JSON object or array
/// (Hermes, Qwen2.5/3, Granite), `<function=NAME><parameter=KEY>…` (Qwen3-Coder, Qwen3.5,
/// Nemotron) or `NAME<arg_key>…</arg_key><arg_value>…</arg_value>` (GLM, Laguna).
fn body_tool_call(payload: &str) -> Result<Vec<RawCall>, String> {
    let body = payload.trim();
    if body.starts_with('{') || body.starts_with('[') {
        return json_calls(body);
    }
    if body.starts_with("<function=") {
        return qwen_function_calls(body);
    }
    glm_call(body).map(|call| vec![call])
}

/// `<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n…</function>`, repeated.
fn qwen_function_calls(body: &str) -> Result<Vec<RawCall>, String> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = body[cursor..].find("<function=") {
        let name_start = cursor + offset + "<function=".len();
        let name_end = body[name_start..]
            .find('>')
            .ok_or("unterminated function tag")?
            + name_start;
        let name = body[name_start..name_end].trim();
        if !plausible_name(name) {
            return Err(format!("implausible function name {name:?}"));
        }
        let inner_start = name_end + 1;
        let inner_end = body[inner_start..]
            .find("</function>")
            .ok_or("function without closing tag")?
            + inner_start;
        let inner = &body[inner_start..inner_end];
        let mut params = Vec::new();
        let mut at = 0;
        while let Some(offset) = inner[at..].find("<parameter=") {
            let key_start = at + offset + "<parameter=".len();
            let key_end = inner[key_start..]
                .find('>')
                .ok_or("unterminated parameter tag")?
                + key_start;
            let key = inner[key_start..key_end].trim().to_owned();
            let value_start = key_end + 1;
            // A missing `</parameter>` ends at the next parameter or the end of the function.
            let (value_end, next) = match inner[value_start..].find("</parameter>") {
                Some(close) => (
                    value_start + close,
                    value_start + close + "</parameter>".len(),
                ),
                None => {
                    let end = inner[value_start..]
                        .find("<parameter=")
                        .map_or(inner.len(), |o| value_start + o);
                    (end, end)
                }
            };
            let value = &inner[value_start..value_end];
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            params.push((key, Arg::Text(value.to_owned())));
            at = next;
        }
        calls.push(RawCall::params(name, params));
        cursor = inner_end + "</function>".len();
    }
    if calls.is_empty() {
        return Err("no function in the tool call".into());
    }
    Ok(calls)
}

/// `NAME<arg_key>K</arg_key><arg_value>V</arg_value>…` with optional whitespace between.
fn glm_call(body: &str) -> Result<RawCall, String> {
    let (name, mut rest) = match body.find("<arg_key>") {
        Some(at) => (body[..at].trim(), &body[at..]),
        None => (body.trim(), ""),
    };
    if !plausible_name(name) {
        return Err(format!("implausible function name {name:?}"));
    }
    let mut params = Vec::new();
    while let Some((key, after_key)) = between(rest, 0, "<arg_key>", "</arg_key>") {
        let (value, after_value) = between(rest, after_key, "<arg_value>", "</arg_value>")
            .ok_or("argument key without a complete value")?;
        params.push((key.trim().to_owned(), Arg::Text(value.to_owned())));
        rest = &rest[after_value..];
    }
    if !rest.trim().is_empty() {
        return Err("text after the last argument".into());
    }
    Ok(RawCall::params(name, params))
}

/// A tag-style call format: `<{p}invoke name="F"><{p}parameter name="K" [string="…"]>V</{p}parameter>…`.
/// MiniMax-M2 has an empty prefix; DeepSeek-V3.2/V4 DSML uses `｜DSML｜`.
fn invoke_calls(payload: &str, prefix: &str) -> Result<Vec<RawCall>, String> {
    let invoke_open = format!("<{prefix}invoke");
    let invoke_close = format!("</{prefix}invoke>");
    let param_open = format!("<{prefix}parameter");
    let param_close = format!("</{prefix}parameter>");
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = payload[cursor..].find(&invoke_open) {
        let attrs_start = cursor + offset + invoke_open.len();
        let attrs_end = payload[attrs_start..]
            .find('>')
            .ok_or("unterminated invoke tag")?
            + attrs_start;
        let name = attribute(&payload[attrs_start..attrs_end], "name")
            .filter(|n| plausible_name(n))
            .ok_or("invoke without a usable name")?;
        let inner_start = attrs_end + 1;
        let inner_end = payload[inner_start..]
            .find(&invoke_close)
            .ok_or("invoke without closing tag")?
            + inner_start;
        let inner = &payload[inner_start..inner_end];
        let mut params = Vec::new();
        let mut at = 0;
        while let Some(offset) = inner[at..].find(&param_open) {
            let attrs_start = at + offset + param_open.len();
            let attrs_end = inner[attrs_start..]
                .find('>')
                .ok_or("unterminated parameter tag")?
                + attrs_start;
            let attrs = &inner[attrs_start..attrs_end];
            let key = attribute(attrs, "name").ok_or("parameter without a name")?;
            let value_start = attrs_end + 1;
            let value_end = inner[value_start..]
                .find(&param_close)
                .ok_or("parameter without closing tag")?
                + value_start;
            let text = inner[value_start..value_end].to_owned();
            let arg = match attribute(attrs, "string") {
                Some("true") => Arg::Typed(text, "string".into()),
                Some("false") => {
                    serde_json::from_str::<Value>(text.trim()).map_or(Arg::Text(text), Arg::Json)
                }
                _ => Arg::Text(text),
            };
            params.push((key.to_owned(), arg));
            at = value_end + param_close.len();
        }
        calls.push(RawCall::params(name, params));
        cursor = inner_end + invoke_close.len();
    }
    if calls.is_empty() {
        return Err("no invoke in the tool call block".into());
    }
    Ok(calls)
}

fn body_minimax_m2(payload: &str) -> Result<Vec<RawCall>, String> {
    invoke_calls(payload, "")
}

fn body_dsml(payload: &str) -> Result<Vec<RawCall>, String> {
    invoke_calls(payload, "｜DSML｜")
}

/// MiniCPM5: ` name="F"><param name="K">V</param>…` after the `<function` token. A value
/// that contains `<`, `&` or a newline is wrapped in `<![CDATA[…]]>`.
fn body_minicpm(payload: &str) -> Result<Vec<RawCall>, String> {
    let attrs_end = payload.find('>').ok_or("unterminated function tag")?;
    let name = attribute(&payload[..attrs_end], "name")
        .filter(|n| plausible_name(n))
        .ok_or("function without a usable name")?;
    let mut rest = &payload[attrs_end + 1..];
    let mut params = Vec::new();
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let tail = rest
            .strip_prefix("<param")
            .ok_or("text between parameters")?;
        let attrs_end = tail.find('>').ok_or("unterminated param tag")?;
        let key = attribute(&tail[..attrs_end], "name").ok_or("param without a name")?;
        let value = &tail[attrs_end + 1..];
        let (text, after) = if let Some(cdata) = value.strip_prefix("<![CDATA[") {
            let end = cdata.find("]]>").ok_or("unterminated CDATA")?;
            let after = cdata[end + 3..]
                .strip_prefix("</param>")
                .ok_or("CDATA not followed by </param>")?;
            (cdata[..end].to_owned(), after)
        } else {
            let end = value.find("</param>").ok_or("param without closing tag")?;
            (value[..end].to_owned(), &value[end + "</param>".len()..])
        };
        params.push((key.to_owned(), Arg::Text(text)));
        rest = after;
    }
    Ok(vec![RawCall::params(name, params)])
}

/// DeepSeek V3 / V3.1 / R1: calls between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`,
/// written either `NAME<｜tool▁sep｜>{json}` (V3.1) or
/// `function<｜tool▁sep｜>NAME\n```json\n{json}\n```` (V3, R1 and its distillations).
fn body_deepseek(payload: &str) -> Result<Vec<RawCall>, String> {
    const BEGIN: &str = "<｜tool▁call▁begin｜>";
    const END: &str = "<｜tool▁call▁end｜>";
    const SEP: &str = "<｜tool▁sep｜>";
    let mut calls = Vec::new();
    let segments: Vec<&str> = if payload.contains(BEGIN) {
        payload
            .split(BEGIN)
            .skip(1)
            .map(|s| s.split(END).next().unwrap_or(s))
            .collect()
    } else {
        vec![payload.split(END).next().unwrap_or(payload)]
    };
    for segment in segments {
        let (head, tail) = segment.split_once(SEP).ok_or("call without a separator")?;
        let head = head.trim();
        let tail = tail.trim();
        let (name, args) = if head == "function" || tail.contains("```") {
            let (name, args) = tail.split_once('\n').unwrap_or((tail, ""));
            (name.trim(), args)
        } else {
            (head, tail)
        };
        if !plausible_name(name) {
            return Err(format!("implausible function name {name:?}"));
        }
        let args = args.trim();
        let args = args
            .strip_prefix("```json")
            .or_else(|| args.strip_prefix("```"))
            .map_or(args, |a| a.trim_end().trim_end_matches("```"))
            .trim();
        let value = if args.is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(args)
                .map_err(|e| format!("arguments are not JSON: {e}"))?
        };
        calls.push(RawCall::json(name, value));
    }
    Ok(calls)
}

/// Kimi K2: `<|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>{json}<|tool_call_end|>`
/// inside a `<|tool_calls_section_begin|>` … `<|tool_calls_section_end|>` section.
fn body_kimi(payload: &str) -> Result<Vec<RawCall>, String> {
    const BEGIN: &str = "<|tool_call_begin|>";
    const END: &str = "<|tool_call_end|>";
    const ARGS: &str = "<|tool_call_argument_begin|>";
    let segments: Vec<&str> = if payload.contains(BEGIN) {
        payload
            .split(BEGIN)
            .skip(1)
            .map(|s| s.split(END).next().unwrap_or(s))
            .collect()
    } else {
        vec![payload.split(END).next().unwrap_or(payload)]
    };
    let mut calls = Vec::new();
    for segment in segments {
        let (id, args) = segment
            .split_once(ARGS)
            .ok_or("call without arguments marker")?;
        let id = id.trim();
        let id = id.strip_prefix("functions.").unwrap_or(id);
        // `NAME:INDEX` — the index is a counter, names may contain dots.
        let name = match id.rsplit_once(':') {
            Some((name, index))
                if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) =>
            {
                name
            }
            _ => id,
        };
        if !plausible_name(name) {
            return Err(format!("implausible function name {name:?}"));
        }
        let args = args.trim();
        let value = if args.is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(args)
                .map_err(|e| format!("arguments are not JSON: {e}"))?
        };
        calls.push(RawCall::json(name, value));
    }
    Ok(calls)
}

/// Map a type name written by a model to a JSON schema type.
fn normalize_type(declared: &str) -> String {
    let lower = declared.trim().to_ascii_lowercase();
    match lower.as_str() {
        "int" | "integer" | "long" => "integer".into(),
        "float" | "double" | "number" | "decimal" => "number".into(),
        "bool" | "boolean" => "boolean".into(),
        "str" | "string" | "text" => "string".into(),
        "null" | "none" => "null".into(),
        other if other.starts_with("array") || other.starts_with("list") => "array".into(),
        other if other.starts_with("object") || other.starts_with("dict") => "object".into(),
        _ => "any".into(),
    }
}

/// K2-Horizon: `<ifm|tool_call>` holds `{"name":…,"arguments":…}` (format `json`) or
/// `NAME\n<ifm|arg_key>K</ifm|arg_key>\n[<ifm|arg_type>T</ifm|arg_type>\n]<ifm|arg_value>V</ifm|arg_value>`
/// (`xml`, `xml_typed`), all inside one `<ifm|tool_calls>` block.
fn body_k2horizon(payload: &str) -> Result<Vec<RawCall>, String> {
    const OPEN: &str = "<ifm|tool_call>";
    const CLOSE: &str = "</ifm|tool_call>";
    let segments: Vec<&str> = if payload.contains(OPEN) {
        payload
            .split(OPEN)
            .skip(1)
            .map(|s| s.split(CLOSE).next().unwrap_or(s))
            .collect()
    } else {
        vec![payload.split(CLOSE).next().unwrap_or(payload)]
    };
    let mut calls = Vec::new();
    for segment in segments {
        let body = segment.trim();
        if body.starts_with('{') || body.starts_with('[') {
            calls.extend(json_calls(body)?);
            continue;
        }
        let (name, mut rest) = match body.find("<ifm|arg_key>") {
            Some(at) => (body[..at].trim(), &body[at..]),
            None => (body, ""),
        };
        if !plausible_name(name) {
            return Err(format!("implausible function name {name:?}"));
        }
        let mut params = Vec::new();
        while let Some((key, after_key)) = between(rest, 0, "<ifm|arg_key>", "</ifm|arg_key>") {
            let tail = &rest[after_key..];
            // The type, when present, sits between the key and the value.
            let value_at = tail
                .find("<ifm|arg_value>")
                .ok_or("argument key without a value")?;
            let declared = between(&tail[..value_at], 0, "<ifm|arg_type>", "</ifm|arg_type>")
                .map(|(t, _)| normalize_type(t));
            let (value, after_value) = between(tail, 0, "<ifm|arg_value>", "</ifm|arg_value>")
                .ok_or("argument value without closing tag")?;
            let arg = match declared {
                Some(ty) => Arg::Typed(value.to_owned(), ty),
                None => Arg::Text(value.to_owned()),
            };
            params.push((key.trim().to_owned(), arg));
            rest = &tail[after_value..];
        }
        if !rest.trim().is_empty() {
            return Err("text after the last argument".into());
        }
        calls.push(RawCall::params(name, params));
    }
    Ok(calls)
}

// --- Gemma-style literals -------------------------------------------------------------------

/// Reader for the `{key:value,…}` literals of Gemma 4 and FunctionGemma. Strings are wrapped
/// in a quote token (`<|"|>` or `<escape>`), keys are bare words.
struct LiteralReader<'a> {
    text: &'a str,
    at: usize,
    quote: &'static str,
}

impl<'a> LiteralReader<'a> {
    fn skip_space(&mut self) {
        while self.text[self.at..].starts_with(|c: char| c.is_whitespace()) {
            self.at += self.text[self.at..]
                .chars()
                .next()
                .map_or(1, char::len_utf8);
        }
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_space();
        if self.text[self.at..].starts_with(token) {
            self.at += token.len();
            true
        } else {
            false
        }
    }

    fn object(&mut self) -> Result<Map<String, Value>, String> {
        let mut map = Map::new();
        if self.eat("}") {
            return Ok(map);
        }
        loop {
            self.skip_space();
            let key = if self.text[self.at..].starts_with(self.quote) {
                self.string()?
            } else {
                let rest = &self.text[self.at..];
                let end = rest.find(':').ok_or("object key without a colon")?;
                self.at += end;
                rest[..end].trim().to_owned()
            };
            if !self.eat(":") {
                return Err("object key without a colon".into());
            }
            let value = self.value()?;
            map.insert(key, value);
            if self.eat(",") {
                continue;
            }
            if self.eat("}") {
                return Ok(map);
            }
            return Err("object not closed".into());
        }
    }

    fn string(&mut self) -> Result<String, String> {
        let body = &self.text[self.at + self.quote.len()..];
        let end = body.find(self.quote).ok_or("unterminated string")?;
        self.at += self.quote.len() + end + self.quote.len();
        Ok(body[..end].to_owned())
    }

    fn value(&mut self) -> Result<Value, String> {
        self.skip_space();
        let rest = &self.text[self.at..];
        if rest.starts_with(self.quote) {
            return self.string().map(Value::String);
        }
        if rest.starts_with('{') {
            self.at += 1;
            return self.object().map(Value::Object);
        }
        if rest.starts_with('[') {
            self.at += 1;
            let mut items = Vec::new();
            if self.eat("]") {
                return Ok(Value::Array(items));
            }
            loop {
                items.push(self.value()?);
                if self.eat(",") {
                    continue;
                }
                if self.eat("]") {
                    return Ok(Value::Array(items));
                }
                return Err("array not closed".into());
            }
        }
        let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
        let word = rest[..end].trim();
        if word.is_empty() {
            return Err("missing value".into());
        }
        self.at += end;
        Ok(match word {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" | "none" => Value::Null,
            other => serde_json::from_str::<Value>(other)
                .ok()
                .filter(Value::is_number)
                .unwrap_or_else(|| Value::String(other.to_owned())),
        })
    }
}

/// `call:NAME{key:value,…}` with the given string quote token.
fn gemma_call(payload: &str, quote: &'static str) -> Result<Vec<RawCall>, String> {
    let body = payload.trim();
    let body = body.strip_prefix("call:").ok_or("missing call: prefix")?;
    let brace = body.find('{').ok_or("call without an argument object")?;
    let name = body[..brace].trim();
    if !plausible_name(name) {
        return Err(format!("implausible function name {name:?}"));
    }
    let mut reader = LiteralReader {
        text: body,
        at: brace + 1,
        quote,
    };
    let args = reader.object()?;
    if !body[reader.at..].trim().is_empty() {
        return Err("text after the argument object".into());
    }
    Ok(vec![RawCall::json(name, Value::Object(args))])
}

fn body_gemma4(payload: &str) -> Result<Vec<RawCall>, String> {
    gemma_call(payload, "<|\"|>")
}

fn body_functiongemma(payload: &str) -> Result<Vec<RawCall>, String> {
    gemma_call(payload, "<escape>")
}

// --- LFM2 -----------------------------------------------------------------------------------

/// Reader for Python call lists: `[name(key="value", n=1, flag=True), other()]`.
struct PythonReader<'a> {
    text: &'a str,
    at: usize,
}

impl PythonReader<'_> {
    fn skip_space(&mut self) {
        while self.text[self.at..].starts_with(|c: char| c.is_whitespace()) {
            self.at += self.text[self.at..]
                .chars()
                .next()
                .map_or(1, char::len_utf8);
        }
    }

    fn eat(&mut self, ch: char) -> bool {
        self.skip_space();
        if self.text[self.at..].starts_with(ch) {
            self.at += ch.len_utf8();
            true
        } else {
            false
        }
    }

    fn identifier(&mut self) -> Result<String, String> {
        self.skip_space();
        let rest = &self.text[self.at..];
        let end = rest
            .find(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '.')))
            .unwrap_or(rest.len());
        if end == 0 {
            return Err("expected a name".into());
        }
        self.at += end;
        Ok(rest[..end].to_owned())
    }

    fn string(&mut self, quote: char) -> Result<String, String> {
        let mut out = String::new();
        self.at += 1;
        let mut chars = self.text[self.at..].char_indices();
        while let Some((index, ch)) = chars.next() {
            if ch == quote {
                self.at += index + 1;
                return Ok(out);
            }
            if ch != '\\' {
                out.push(ch);
                continue;
            }
            match chars.next().map(|(_, c)| c) {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some(other) => out.push(other),
                None => break,
            }
        }
        Err("unterminated string".into())
    }

    fn value(&mut self) -> Result<Value, String> {
        self.skip_space();
        let rest = &self.text[self.at..];
        match rest.chars().next() {
            Some(q @ ('"' | '\'')) => self.string(q).map(Value::String),
            Some('[') | Some('(') => {
                let close = if rest.starts_with('[') { ']' } else { ')' };
                self.at += 1;
                let mut items = Vec::new();
                if self.eat(close) {
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    if self.eat(',') {
                        if self.eat(close) {
                            return Ok(Value::Array(items));
                        }
                        continue;
                    }
                    if self.eat(close) {
                        return Ok(Value::Array(items));
                    }
                    return Err("list not closed".into());
                }
            }
            Some('{') => {
                self.at += 1;
                let mut map = Map::new();
                if self.eat('}') {
                    return Ok(Value::Object(map));
                }
                loop {
                    let key = match self.value()? {
                        Value::String(key) => key,
                        other => other.to_string(),
                    };
                    if !self.eat(':') {
                        return Err("dict key without a colon".into());
                    }
                    map.insert(key, self.value()?);
                    if self.eat(',') {
                        if self.eat('}') {
                            return Ok(Value::Object(map));
                        }
                        continue;
                    }
                    if self.eat('}') {
                        return Ok(Value::Object(map));
                    }
                    return Err("dict not closed".into());
                }
            }
            _ => {
                let end = rest
                    .find(|c: char| matches!(c, ',' | ')' | ']' | '}' | ':') || c.is_whitespace())
                    .unwrap_or(rest.len());
                let word = &rest[..end];
                self.at += end;
                Ok(match word {
                    "True" | "true" => Value::Bool(true),
                    "False" | "false" => Value::Bool(false),
                    "None" | "null" => Value::Null,
                    other => serde_json::from_str::<Value>(other)
                        .ok()
                        .filter(Value::is_number)
                        .ok_or_else(|| format!("unsupported Python value {other:?}"))?,
                })
            }
        }
    }
}

/// LFM2 writes `[get_weather(city="Köln"), other(x=1)]` between `<|tool_call_start|>` and
/// `<|tool_call_end|>`; with a JSON system prompt it writes `[{"name":…,"arguments":…}]`.
fn body_lfm2(payload: &str) -> Result<Vec<RawCall>, String> {
    let body = payload.trim();
    let inner = body.strip_prefix('[').unwrap_or(body);
    if inner.trim_start().starts_with('{') {
        return json_calls(body);
    }
    let mut reader = PythonReader { text: body, at: 0 };
    let bracketed = reader.eat('[');
    let mut calls = Vec::new();
    loop {
        if bracketed && reader.eat(']') {
            break;
        }
        let name = reader.identifier()?;
        if !plausible_name(&name) {
            return Err(format!("implausible function name {name:?}"));
        }
        if !reader.eat('(') {
            return Err("call without parentheses".into());
        }
        let mut args = Map::new();
        if !reader.eat(')') {
            loop {
                let key = reader.identifier()?;
                if !reader.eat('=') {
                    return Err("positional arguments are not supported".into());
                }
                args.insert(key, reader.value()?);
                if reader.eat(',') {
                    if reader.eat(')') {
                        break;
                    }
                    continue;
                }
                if reader.eat(')') {
                    break;
                }
                return Err("call not closed".into());
            }
        }
        calls.push(RawCall::json(name, Value::Object(args)));
        if reader.eat(',') {
            continue;
        }
        if bracketed && !reader.eat(']') {
            return Err("call list not closed".into());
        }
        break;
    }
    reader.skip_space();
    if reader.at < body.len() {
        return Err("text after the call list".into());
    }
    Ok(calls)
}

// ---------------------------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------------------------

/// A family whose calls are `open … close` spans (or `open JSON`) with a plain body decoder.
#[derive(Clone, Copy)]
pub struct SimpleParser {
    id: &'static str,
    pairs: &'static [MarkerPair],
    extra: &'static [&'static str],
    body: Body,
    requires_json: bool,
}

impl ToolParser for SimpleParser {
    fn id(&self) -> &'static str {
        self.id
    }
    fn pairs(&self) -> &'static [MarkerPair] {
        self.pairs
    }
    fn extra_markers(&self) -> &'static [&'static str] {
        self.extra
    }
    fn requires_json_payload(&self) -> bool {
        self.requires_json
    }
    fn decode(&self, payload: &str) -> Result<Vec<RawCall>, String> {
        (self.body)(payload)
    }
}

const fn simple(
    id: &'static str,
    pairs: &'static [MarkerPair],
    extra: &'static [&'static str],
    body: Body,
) -> SimpleParser {
    SimpleParser {
        id,
        pairs,
        extra,
        body,
        requires_json: false,
    }
}

/// Qwen2.5/3, Hermes 2/3, Granite 4: `<tool_call>{"name":…,"arguments":…}</tool_call>`.
/// The same marker carries Qwen3-Coder's XML and GLM's key/value tags, see [`body_tool_call`].
const TOOL_CALL_PAIRS: &[MarkerPair] = &[("<tool_call>", Some("</tool_call>"))];
const GLM_EXTRA: &[&str] = &["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"];
/// MiniMax-M1 and Apriel: one JSON object per line (or an array) in `<tool_calls>`.
const TOOL_CALLS_PAIRS: &[MarkerPair] = &[("<tool_calls>", Some("</tool_calls>"))];
const MINIMAX_PAIRS: &[MarkerPair] = &[("<minimax:tool_call>", Some("</minimax:tool_call>"))];
/// Granite 3.x: `<|tool_call|>[{"name":…,"arguments":…}]`.
const GRANITE3_PAIRS: &[MarkerPair] = &[("<|tool_call|>", None)];
/// Phi-4-mini and Firefunction: `functools[{"name":…,"arguments":…}]`.
// UNVERIFIED: the `functools` marker comes from vLLM's phi4_mini_json parser and the
// firefunction-v2 template; no Phi-4-mini template was available to check the exact spelling.
const PHI4_PAIRS: &[MarkerPair] = &[("functools", None)];
const GEMMA4_PAIRS: &[MarkerPair] = &[("<|tool_call>", Some("<tool_call|>"))];
// UNVERIFIED: FunctionGemma markers and `<escape>` quoting are from Google's model card, not
// from a template in this repository.
const FUNCTIONGEMMA_PAIRS: &[MarkerPair] =
    &[("<start_function_call>", Some("<end_function_call>"))];
const LFM2_PAIRS: &[MarkerPair] = &[("<|tool_call_start|>", Some("<|tool_call_end|>"))];
const KIMI_PAIRS: &[MarkerPair] = &[
    (
        "<|tool_calls_section_begin|>",
        Some("<|tool_calls_section_end|>"),
    ),
    ("<|tool_call_begin|>", Some("<|tool_call_end|>")),
];
const KIMI_EXTRA: &[&str] = &["<|tool_call_argument_begin|>"];
const DEEPSEEK_PAIRS: &[MarkerPair] = &[
    ("<｜tool▁calls▁begin｜>", Some("<｜tool▁calls▁end｜>")),
    ("<｜tool▁call▁begin｜>", Some("<｜tool▁call▁end｜>")),
];
const DEEPSEEK_EXTRA: &[&str] = &["<｜tool▁sep｜>"];
// UNVERIFIED: whether `<｜DSML｜function_calls>` is one special token or ordinary text in the
// DeepSeek V3.2/V4 vocabulary; the parser works either way as long as the text is visible.
const DSML_V32_PAIRS: &[MarkerPair] = &[(
    "<｜DSML｜function_calls>",
    Some("</｜DSML｜function_calls>"),
)];
const DSML_V4_PAIRS: &[MarkerPair] = &[("<｜DSML｜tool_calls>", Some("</｜DSML｜tool_calls>"))];
const K2HORIZON_PAIRS: &[MarkerPair] = &[
    ("<ifm|tool_calls>", Some("</ifm|tool_calls>")),
    ("<ifm|tool_call>", Some("</ifm|tool_call>")),
];
const K2HORIZON_EXTRA: &[&str] = &[
    "<ifm|arg_key>",
    "</ifm|arg_key>",
    "<ifm|arg_type>",
    "</ifm|arg_type>",
    "<ifm|arg_value>",
    "</ifm|arg_value>",
];

const HERMES: SimpleParser = simple("hermes", TOOL_CALL_PAIRS, &[], body_tool_call);
const QWEN3_CODER: SimpleParser = simple("qwen3_coder", TOOL_CALL_PAIRS, &[], body_tool_call);
const GLM: SimpleParser = simple("glm", TOOL_CALL_PAIRS, GLM_EXTRA, body_tool_call);
const MINIMAX_M1: SimpleParser = simple("minimax_m1", TOOL_CALLS_PAIRS, &[], json_calls);
const MINIMAX: SimpleParser = simple("minimax", MINIMAX_PAIRS, &[], body_minimax_m2);
const GRANITE3: SimpleParser = simple("granite3", GRANITE3_PAIRS, &[], json_calls);
const PHI4: SimpleParser = SimpleParser {
    id: "phi4",
    pairs: PHI4_PAIRS,
    extra: &[],
    body: json_calls,
    requires_json: true,
};
const GEMMA4: SimpleParser = simple("gemma4", GEMMA4_PAIRS, &[], body_gemma4);
const FUNCTIONGEMMA: SimpleParser = simple(
    "functiongemma",
    FUNCTIONGEMMA_PAIRS,
    &[],
    body_functiongemma,
);
const LFM2: SimpleParser = simple("lfm2", LFM2_PAIRS, &[], body_lfm2);
const KIMI: SimpleParser = simple("kimi", KIMI_PAIRS, KIMI_EXTRA, body_kimi);
const DEEPSEEK: SimpleParser = simple("deepseek", DEEPSEEK_PAIRS, DEEPSEEK_EXTRA, body_deepseek);
const DEEPSEEK_V32: SimpleParser = simple("deepseek_v32", DSML_V32_PAIRS, &[], body_dsml);
const DEEPSEEK_V4: SimpleParser = simple("deepseek_v4", DSML_V4_PAIRS, &[], body_dsml);
const K2HORIZON: SimpleParser = simple(
    "k2horizon",
    K2HORIZON_PAIRS,
    K2HORIZON_EXTRA,
    body_k2horizon,
);

// --- more families ----------------------------------------------------------------------------

/// InternLM2/2.5: `<|action_start|><|plugin|>\n{"name":…,"parameters":…}<|action_end|>` (vLLM `internlm2`).
const INTERNLM_PAIRS: &[MarkerPair] = &[("<|action_start|>", Some("<|action_end|>"))];
const INTERNLM_EXTRA: &[&str] = &["<|plugin|>"];

fn body_internlm(payload: &str) -> Result<Vec<RawCall>, String> {
    let body = payload.trim().strip_prefix("<|plugin|>").unwrap_or(payload).trim();
    // `<|interpreter|>` blocks carry code, not a function call
    json_calls(body)
}

/// Granite-20B-FC: `<function_call> {"name":…,"arguments":…}`, the marker repeated for every call.
const GRANITE20B_PAIRS: &[MarkerPair] = &[("<function_call>", None)];
/// Seed-OSS: the Qwen3-Coder XML inside `<seed:tool_call>` tags.
const SEED_PAIRS: &[MarkerPair] = &[("<seed:tool_call>", Some("</seed:tool_call>"))];
/// Step-3: calls inside `<｜tool_calls_begin｜>` (note the plain underscores, unlike DeepSeek's `▁`).
const STEP3_PAIRS: &[MarkerPair] = &[
    ("<｜tool_calls_begin｜>", Some("<｜tool_calls_end｜>")),
    ("<｜tool_call_begin｜>", Some("<｜tool_call_end｜>")),
];
const STEP3_EXTRA: &[&str] = &["<｜tool_sep｜>"];

/// `function<｜tool_sep｜><steptml:invoke name="F"><steptml:parameter name="K">V</steptml:parameter></steptml:invoke>`.
fn body_step3(payload: &str) -> Result<Vec<RawCall>, String> {
    invoke_calls(payload, "steptml:")
}

/// Llama 4 and other Pythonic models may wrap the list in `<|python_start|>` … `<|python_end|>`.
const LLAMA4_PAIRS: &[MarkerPair] = &[("<|python_start|>", Some("<|python_end|>"))];
/// OLMo-3: one Python call per line between `<function_calls>` tags.
const OLMO3_PAIRS: &[MarkerPair] = &[("<function_calls>", Some("</function_calls>"))];
/// Gemma 3 function calling (and ToolACE-style prompts): a ```` ```tool_code ```` block with one call per line.
const TOOL_CODE_PAIRS: &[MarkerPair] = &[("```tool_code", Some("```"))];

/// Calls written one per line (`get_weather(city="Köln")`), without the surrounding list.
fn body_call_lines(payload: &str) -> Result<Vec<RawCall>, String> {
    let lines: Vec<&str> = payload.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return Err("no call in the block".into());
    }
    body_lfm2(&format!("[{}]", lines.join(",")))
}

const INTERNLM: SimpleParser = simple("internlm", INTERNLM_PAIRS, INTERNLM_EXTRA, body_internlm);
const GRANITE20B: SimpleParser = SimpleParser {
    id: "granite20b",
    pairs: GRANITE20B_PAIRS,
    extra: &[],
    body: json_calls,
    requires_json: true,
};
const SEED_OSS: SimpleParser = simple("seed_oss", SEED_PAIRS, &[], body_tool_call);
const STEP3: SimpleParser = simple("step3_steptml", STEP3_PAIRS, STEP3_EXTRA, body_step3);
const LLAMA4_PYTHONIC: SimpleParser = simple("llama4_pythonic", LLAMA4_PAIRS, &[], body_lfm2);
const OLMO3: SimpleParser = simple("olmo3", OLMO3_PAIRS, &[], body_call_lines);
const TOOL_CODE: SimpleParser = simple("tool_code", TOOL_CODE_PAIRS, &[], body_call_lines);

/// A bare Python call list as the whole reply or a whole line: `[get_weather(city="SF"), other(x=1)]`
/// (Llama 4, ToolACE, Gemma-3 prompts; vLLM `pythonic`). Prose with brackets (`[1]`, `[note]`) stays content:
/// the bracket must open a line and be followed by `name(`.
pub struct PythonicParser;

impl PythonicParser {
    /// `Some(true)` once `[ name (` is complete, `None` while the text ends before the decision is possible.
    fn looks_like_calls(text: &str, open: usize) -> Option<bool> {
        let rest = text[open + 1..].trim_start();
        let first = rest.chars().next()?;
        if !(first.is_alphabetic() || first == '_') {
            return Some(false);
        }
        for (index, ch) in rest.char_indices() {
            if ch.is_alphanumeric() || matches!(ch, '_' | '.') {
                continue;
            }
            let after = rest[index..].trim_start();
            return if after.is_empty() { None } else { Some(after.starts_with('(')) };
        }
        None
    }
}

impl ToolParser for PythonicParser {
    fn id(&self) -> &'static str {
        "pythonic"
    }
    fn pairs(&self) -> &'static [MarkerPair] {
        &[("[", None)]
    }
    fn lenient_end(&self) -> bool {
        false
    }
    fn find_open(&self, text: &str, from: usize) -> Option<usize> {
        let mut at = from;
        while let Some(offset) = text[at..].find('[') {
            let start = at + offset;
            let opens_line = text[..start].rsplit('\n').next().is_none_or(|line| line.trim().is_empty());
            if opens_line && Self::looks_like_calls(text, start) != Some(false) {
                return Some(start);
            }
            at = start + 1;
        }
        None
    }
    fn span_end(&self, text: &str, start: usize) -> Option<usize> {
        let mut depth = 0usize;
        let mut quote: Option<u8> = None;
        let mut escaped = false;
        for (index, byte) in text.bytes().enumerate().skip(start) {
            if let Some(q) = quote {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == q {
                    quote = None;
                }
                continue;
            }
            match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'[' | b'(' | b'{' => depth += 1,
                b']' | b')' | b'}' => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        return Some(index + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }
    fn payload<'t>(&self, text: &'t str, start: usize, end: usize) -> &'t str {
        &text[start..end]
    }
    fn decode(&self, payload: &str) -> Result<Vec<RawCall>, String> {
        body_lfm2(payload)
    }
}

/// xLAM (Salesforce): a JSON array of calls, bare (`[{"name":…,"arguments":…}]`), in a ```` ```json ```` fence
/// or after `[TOOL_CALLS]`; the array may follow reasoning (`</think>`).
pub struct XlamParser;

impl XlamParser {
    const FENCE: &'static str = "```json";
    const TAG: &'static str = "[TOOL_CALLS]";

    /// `[` followed by `{`, or by nothing yet.
    fn bare_start(text: &str, open: usize) -> bool {
        let opens_line = text[..open].rsplit('\n').next().is_none_or(|line| line.trim().is_empty());
        opens_line && matches!(text[open + 1..].trim_start().chars().next(), Some('{') | None)
    }
}

impl ToolParser for XlamParser {
    fn id(&self) -> &'static str {
        "xlam"
    }
    fn pairs(&self) -> &'static [MarkerPair] {
        &[(Self::FENCE, Some("```")), (Self::TAG, None)]
    }
    fn find_open(&self, text: &str, from: usize) -> Option<usize> {
        let mut best: Option<usize> = None;
        for marker in [Self::FENCE, Self::TAG] {
            if let Some(at) = text[from..].find(marker) {
                best = Some(best.map_or(from + at, |b: usize| b.min(from + at)));
            }
        }
        let mut at = from;
        while let Some(offset) = text[at..].find('[') {
            let start = at + offset;
            if text[start..].starts_with(Self::TAG) {
                at = start + 1;
                continue;
            }
            if Self::bare_start(text, start) {
                best = Some(best.map_or(start, |b| b.min(start)));
                break;
            }
            at = start + 1;
        }
        best
    }
    fn span_end(&self, text: &str, start: usize) -> Option<usize> {
        if text[start..].starts_with(Self::FENCE) {
            let after = start + Self::FENCE.len();
            return text[after..].find("```").map(|o| after + o + 3);
        }
        let json_from = if text[start..].starts_with(Self::TAG) { start + Self::TAG.len() } else { start };
        json_span_end(text, json_from)
    }
    fn payload<'t>(&self, text: &'t str, start: usize, end: usize) -> &'t str {
        if text[start..].starts_with(Self::FENCE) {
            let inner = &text[start + Self::FENCE.len()..end];
            return inner.strip_suffix("```").unwrap_or(inner);
        }
        if text[start..].starts_with(Self::TAG) {
            return &text[start + Self::TAG.len()..end];
        }
        &text[start..end]
    }
    fn decode(&self, payload: &str) -> Result<Vec<RawCall>, String> {
        json_calls(payload)
    }
}

/// Mistral: `[TOOL_CALLS]` followed by either a JSON array of calls (Mistral 7B v0.3, Nemo)
/// or `NAME[ARGS]{json}` with an optional `[CALL_ID]xxxxxxxxx` (Mistral Small 3.x, Devstral,
/// Ministral); several calls repeat the marker.
pub struct MistralParser;

impl MistralParser {
    const OPEN: &'static str = "[TOOL_CALLS]";
}

impl ToolParser for MistralParser {
    fn id(&self) -> &'static str {
        "mistral"
    }
    fn pairs(&self) -> &'static [MarkerPair] {
        &[(Self::OPEN, None)]
    }
    fn extra_markers(&self) -> &'static [&'static str] {
        &["[ARGS]", "[CALL_ID]"]
    }
    fn span_end(&self, text: &str, start: usize) -> Option<usize> {
        let after = start + Self::OPEN.len();
        let rest = text[after..].trim_start();
        if rest.starts_with('[') || rest.starts_with('{') {
            return json_span_end(text, after);
        }
        let args = text[after..].find("[ARGS]")? + after + "[ARGS]".len();
        json_span_end(text, args)
    }
    fn decode(&self, payload: &str) -> Result<Vec<RawCall>, String> {
        let body = payload.trim_start();
        if body.starts_with('[') || body.starts_with('{') {
            return json_calls(body);
        }
        let (head, args) = body.split_once("[ARGS]").ok_or("call without [ARGS]")?;
        let name = head.split("[CALL_ID]").next().unwrap_or(head).trim();
        if !plausible_name(name) {
            return Err(format!("implausible function name {name:?}"));
        }
        let value = serde_json::from_str::<Value>(args.trim())
            .map_err(|e| format!("arguments are not JSON: {e}"))?;
        Ok(vec![RawCall::json(name, value)])
    }
}

/// Llama 3.1/3.2/3.3: `{"name":…,"parameters":…}` as the whole reply (JSON tool use), or
/// after `<|python_tag|>`. Calls may be separated by `;`.
pub struct LlamaParser;

impl LlamaParser {
    const TAG: &'static str = "<|python_tag|>";

    /// `{"name"` or `{"type"` (Llama 3.2 writes `"type":"function"` first), whitespace allowed,
    /// or a prefix of that: the only ways a bare call object starts. `Some(true)` once the key
    /// is complete, `Some(false)` while it still could become one.
    fn call_start(text: &str) -> Option<bool> {
        let rest = text.strip_prefix('{')?.trim_start();
        const KEYS: [&str; 2] = ["\"name\"", "\"type\""];
        if KEYS.iter().any(|key| rest.starts_with(key)) {
            Some(true)
        } else if KEYS
            .iter()
            .any(|key| rest.len() < key.len() && key.starts_with(rest))
        {
            Some(false)
        } else {
            None
        }
    }
}

impl ToolParser for LlamaParser {
    fn id(&self) -> &'static str {
        "llama3"
    }
    fn pairs(&self) -> &'static [MarkerPair] {
        &[(Self::TAG, None)]
    }
    fn find_open(&self, text: &str, from: usize) -> Option<usize> {
        let tagged = text[from..].find(Self::TAG).map(|o| o + from);
        // A bare call object: at the start of the reply, or right after another call object.
        let skipped = text[from..]
            .find(|c: char| !(c.is_whitespace() || c == ';'))
            .map(|o| o + from);
        let bare = skipped.filter(|&at| {
            (from == 0 || text[..from].ends_with('}')) && Self::call_start(&text[at..]).is_some()
        });
        match (tagged, bare) {
            (Some(t), Some(b)) => Some(t.min(b)),
            (t, b) => t.or(b),
        }
    }
    fn span_end(&self, text: &str, start: usize) -> Option<usize> {
        let open = if text[start..].starts_with(Self::TAG) {
            Self::TAG.len()
        } else {
            0
        };
        json_span_end(text, start + open)
    }
    fn payload<'t>(&self, text: &'t str, start: usize, end: usize) -> &'t str {
        let open = if text[start..].starts_with(Self::TAG) {
            Self::TAG.len()
        } else {
            0
        };
        &text[(start + open).min(end)..end]
    }
    fn decode(&self, payload: &str) -> Result<Vec<RawCall>, String> {
        json_calls(payload)
    }
}

/// MiniCPM5: `<function name="F"><param name="K">V</param>…</function>`. `<function`,
/// `<param`, `</param>` and `</function>` are control tokens of the vocabulary.
pub struct MiniCpmParser;

impl MiniCpmParser {
    /// Index of the first `</function>` that is not inside a CDATA section.
    fn close_at(text: &str, from: usize) -> Option<usize> {
        let mut at = from;
        loop {
            let close = text[at..].find("</function>")? + at;
            match text[at..close].rfind("<![CDATA[") {
                Some(open) => {
                    let open = at + open;
                    if text[open..close].contains("]]>") {
                        return Some(close);
                    }
                    // Inside CDATA: continue behind its end.
                    let end = text[close..].find("]]>")? + close + 3;
                    at = end;
                }
                None => return Some(close),
            }
        }
    }
}

impl ToolParser for MiniCpmParser {
    fn id(&self) -> &'static str {
        "minicpm"
    }
    fn pairs(&self) -> &'static [MarkerPair] {
        &[("<function", Some("</function>"))]
    }
    fn extra_markers(&self) -> &'static [&'static str] {
        &["<param", "</param>"]
    }
    fn find_open(&self, text: &str, from: usize) -> Option<usize> {
        // `<function name=` only: `<function>` or `<function=` belong to other formats and
        // plain text such as `<functional>` must not open a span.
        let mut at = from;
        while let Some(offset) = text[at..].find("<function") {
            let start = at + offset;
            let rest = &text[start + "<function".len()..];
            if rest.is_empty() || rest.starts_with(|c: char| c.is_whitespace()) {
                return Some(start);
            }
            at = start + 1;
        }
        None
    }
    fn span_end(&self, text: &str, start: usize) -> Option<usize> {
        Self::close_at(text, start + "<function".len()).map(|c| c + "</function>".len())
    }
    fn decode(&self, payload: &str) -> Result<Vec<RawCall>, String> {
        body_minicpm(payload)
    }
}

/// gpt-oss (Harmony): `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{json}<|call|>`.
///
/// The reasoning splitter (`reasoning::HarmonyReasoning`) hands over such spans in exactly
/// this spelling, also when the model addressed the function in the role header. `<|call|>`
/// is a stop token, so it is usually missing from the text: the span then ends with the text.
pub struct HarmonyParser;

impl HarmonyParser {
    const OPEN: &'static str = "<|channel|>commentary to=functions.";
}

impl ToolParser for HarmonyParser {
    fn id(&self) -> &'static str {
        "gpt_oss"
    }
    fn pairs(&self) -> &'static [MarkerPair] {
        &[(Self::OPEN, Some("<|call|>"))]
    }
    fn markers(&self) -> Vec<&'static str> {
        vec![
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|end|>",
            "<|call|>",
            "<|return|>",
        ]
    }
    fn decode(&self, payload: &str) -> Result<Vec<RawCall>, String> {
        let (head, body) = payload
            .split_once("<|message|>")
            .ok_or("call without a message body")?;
        let name: String = head
            .trim_start()
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '<')
            .collect();
        if !plausible_name(&name) {
            return Err(format!("implausible function name {name:?}"));
        }
        let body = body.trim();
        let value = if body.is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(body)
                .map_err(|e| format!("arguments are not JSON: {e}"))?
        };
        Ok(vec![RawCall::json(name, value)])
    }
}

/// The model output is passed through unchanged: no markers, no calls.
pub struct PlainParser;

impl ToolParser for PlainParser {
    fn id(&self) -> &'static str {
        "none"
    }
    fn pairs(&self) -> &'static [MarkerPair] {
        &[]
    }
    fn find_open(&self, _text: &str, _from: usize) -> Option<usize> {
        None
    }
    fn decode(&self, _payload: &str) -> Result<Vec<RawCall>, String> {
        Ok(Vec::new())
    }
}

// ---------------------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------------------

/// Canonical id and accepted aliases of every parser, for `--tool-parser`.
const NAMES: &[(&str, &[&str])] = &[
    (
        "hermes",
        &[
            "qwen", "qwen2", "qwen2_5", "qwen3", "nous", "granite", "granite4", "hermes2",
            "hermes3", "hermes_2_pro", "hermes-2-pro", "ernie", "ernie45", "ernie4_5", "qwen3_omni",
        ],
    ),
    (
        "qwen3_coder",
        &["qwen3-coder", "qwen3_xml", "qwen3.5", "nemotron", "step3"],
    ),
    (
        "glm",
        &["glm4", "glm45", "glm-4.5", "glm-4.6", "glm-4.7", "laguna"],
    ),
    ("minicpm", &["minicpm5"]),
    (
        "llama3",
        &[
            "llama",
            "llama3_json",
            "llama3.1",
            "llama3.2",
            "llama3.3",
            "python_tag",
        ],
    ),
    ("mistral", &["ministral", "devstral", "mixtral"]),
    (
        "deepseek",
        &["deepseek_v3", "deepseek_v31", "deepseek-r1", "r1"],
    ),
    ("deepseek_v32", &["deepseek_v3.2", "dsml"]),
    ("deepseek_v4", &["deepseek-v4"]),
    ("kimi", &["kimi_k2", "kimi-k2", "k2"]),
    ("gpt_oss", &["gpt-oss", "gptoss", "harmony", "openai"]),
    ("gemma4", &["gemma", "gemma-4"]),
    ("functiongemma", &["function_gemma", "function-gemma"]),
    ("minimax", &["minimax_m2", "minimax-m2"]),
    ("minimax_m1", &["minimax-m1", "apriel", "hunyuan", "hunyuan_a13b", "jamba"]),
    ("pythonic", &["toolace", "python"]),
    ("llama4_pythonic", &["llama4", "llama-4"]),
    ("olmo3", &["olmo-3"]),
    ("tool_code", &["gemma3", "gemma-3"]),
    ("xlam", &["salesforce"]),
    ("internlm", &["internlm2", "internlm2.5", "internlm3"]),
    ("granite20b", &["granite-20b-fc", "granite_20b_fc"]),
    ("seed_oss", &["seed-oss", "seed"]),
    ("step3_steptml", &["step-3", "step3_v1"]),
    ("granite3", &["granite-3", "granite3.3"]),
    (
        "phi4",
        &["phi4_mini", "phi-4-mini", "functools", "firefunction"],
    ),
    ("lfm2", &["lfm", "lfm2.5", "liquid"]),
    ("k2horizon", &["k2-horizon", "ifm"]),
    ("none", &["off", "disabled"]),
];

/// Canonical ids, for help texts and error messages.
pub fn parser_ids() -> Vec<&'static str> {
    NAMES.iter().map(|(id, _)| *id).collect()
}

fn canonical(name: &str) -> Option<&'static str> {
    let lower = name.trim().to_ascii_lowercase();
    NAMES
        .iter()
        .find(|(id, aliases)| *id == lower || aliases.contains(&lower.as_str()))
        .map(|(id, _)| *id)
}

/// `true` when `--tool-parser` accepts the value (`auto` included).
pub fn is_known(name: &str) -> bool {
    name.trim().eq_ignore_ascii_case("auto") || canonical(name).is_some()
}

/// Build the parser with the given id or alias.
pub fn by_name(name: &str) -> Result<std::sync::Arc<dyn ToolParser>, String> {
    use std::sync::Arc;
    let id = canonical(name).ok_or_else(|| {
        format!(
            "unknown tool parser '{name}' (available: auto, {})",
            parser_ids().join(", ")
        )
    })?;
    let parser: Arc<dyn ToolParser> = match id {
        "hermes" => Arc::new(HERMES),
        "qwen3_coder" => Arc::new(QWEN3_CODER),
        "glm" => Arc::new(GLM),
        "minicpm" => Arc::new(MiniCpmParser),
        "llama3" => Arc::new(LlamaParser),
        "mistral" => Arc::new(MistralParser),
        "deepseek" => Arc::new(DEEPSEEK),
        "deepseek_v32" => Arc::new(DEEPSEEK_V32),
        "deepseek_v4" => Arc::new(DEEPSEEK_V4),
        "kimi" => Arc::new(KIMI),
        "gpt_oss" => Arc::new(HarmonyParser),
        "gemma4" => Arc::new(GEMMA4),
        "functiongemma" => Arc::new(FUNCTIONGEMMA),
        "minimax" => Arc::new(MINIMAX),
        "minimax_m1" => Arc::new(MINIMAX_M1),
        "granite3" => Arc::new(GRANITE3),
        "phi4" => Arc::new(PHI4),
        "lfm2" => Arc::new(LFM2),
        "k2horizon" => Arc::new(K2HORIZON),
        "pythonic" => Arc::new(PythonicParser),
        "llama4_pythonic" => Arc::new(LLAMA4_PYTHONIC),
        "olmo3" => Arc::new(OLMO3),
        "tool_code" => Arc::new(TOOL_CODE),
        "xlam" => Arc::new(XlamParser),
        "internlm" => Arc::new(INTERNLM),
        "granite20b" => Arc::new(GRANITE20B),
        "seed_oss" => Arc::new(SEED_OSS),
        "step3_steptml" => Arc::new(STEP3),
        _ => Arc::new(PlainParser),
    };
    Ok(parser)
}

/// The parser used when a request carries tools and nothing was configured.
pub fn default_parser() -> std::sync::Arc<dyn ToolParser> {
    std::sync::Arc::new(HERMES)
}

/// Pick the parser for a loaded model. An explicit `flag` (anything but `auto`) wins; then the
/// markers of the chat template (the most specific first); then `general.architecture`; the
/// Hermes format is the default, because the Qwen/Hermes template is the most common one.
pub fn select(
    arch: &str,
    chat_template: &str,
    flag: Option<&str>,
) -> Result<std::sync::Arc<dyn ToolParser>, String> {
    if let Some(name) = flag.filter(|n| !n.trim().eq_ignore_ascii_case("auto")) {
        return by_name(name);
    }
    // (marker in the template, parser id), most specific first.
    const BY_TEMPLATE: &[(&str, &str)] = &[
        ("<ifm|tool_calls>", "k2horizon"),
        ("<param name=", "minicpm"),
        ("<|tool_calls_section_begin|>", "kimi"),
        ("DSML｜tool_calls", "deepseek_v4"),
        ("DSML｜function_calls", "deepseek_v32"),
        ("<｜tool▁calls▁begin｜>", "deepseek"),
        ("<|start|>assistant<|channel|>", "gpt_oss"),
        ("to=functions", "gpt_oss"),
        ("<|tool_call>call:", "gemma4"),
        ("<start_function_call>", "functiongemma"),
        ("<minimax:tool_call>", "minimax"),
        ("<|tool_call_start|>", "lfm2"),
        ("<seed:tool_call>", "seed_oss"),
        ("<|action_start|>", "internlm"),
        ("<｜tool_calls_begin｜>", "step3_steptml"),
        ("<function_call>", "granite20b"),
        ("<|python_start|>", "llama4_pythonic"),
        ("```tool_code", "tool_code"),
        ("<function_calls>", "olmo3"),
        ("[TOOL_CALLS]", "mistral"),
        ("<|python_tag|>", "llama3"),
        ("<arg_key>", "glm"),
        ("<function=", "qwen3_coder"),
        ("<tool_calls>", "minimax_m1"),
        ("<|tool_call|>", "granite3"),
        ("functools", "phi4"),
        ("<tool_call>", "hermes"),
    ];
    for (marker, id) in BY_TEMPLATE {
        if chat_template.contains(marker) {
            return by_name(id);
        }
    }
    // Llama 3.1+ templates for JSON tool use mention neither marker.
    if chat_template.contains("Environment: ipython")
        || (chat_template.contains("\"parameters\": dictionary of argument name"))
    {
        return by_name("llama3");
    }
    let arch = arch.to_ascii_lowercase();
    const BY_ARCH: &[(&str, &str)] = &[
        ("gpt-oss", "gpt_oss"),
        ("gptoss", "gpt_oss"),
        ("minicpm", "minicpm"),
        ("internlm", "internlm"),
        ("seed_oss", "seed_oss"),
        ("kimi", "kimi"),
        ("glm", "glm"),
        ("minimax", "minimax"),
        ("lfm", "lfm2"),
        ("gemma", "gemma4"),
        ("phi", "phi4"),
        ("granite", "hermes"),
    ];
    for (fragment, id) in BY_ARCH {
        if arch.contains(fragment) {
            return by_name(id);
        }
    }
    Ok(default_parser())
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `text` in every possible chunking (every prefix, which is what the server sees)
    /// and assert that content only grows, completed calls never change and no half marker
    /// is shown. Returns the result of the final call.
    fn run(parser: &dyn ToolParser, text: &str) -> Extracted {
        let mut previous = Extracted::default();
        let boundaries = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()));
        for end in boundaries {
            let step = extract(parser, &text[..end], false);
            assert!(
                step.content.starts_with(&previous.content),
                "{}: content shrank at byte {end}: {:?} -> {:?}",
                parser.id(),
                previous.content,
                step.content
            );
            assert!(
                step.calls.len() >= previous.calls.len()
                    && step.calls[..previous.calls.len()] == previous.calls[..],
                "{}: a completed call changed or vanished at byte {end}",
                parser.id()
            );
            previous = step;
        }
        let last = extract(parser, text, true);
        assert!(
            last.content.starts_with(&previous.content),
            "{}: the final call dropped streamed content",
            parser.id()
        );
        assert!(last.calls.len() >= previous.calls.len());
        last
    }

    /// Random, uneven chunking of the same text must give the same final result.
    fn run_random(parser: &dyn ToolParser, text: &str, seed: u64) -> Extracted {
        let mut state = seed | 1;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut end = 0;
        let mut previous = Extracted::default();
        while end < text.len() {
            end = (end + 1 + next() % 7).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            let step = extract(parser, &text[..end], end == text.len());
            assert!(step.content.starts_with(&previous.content));
            previous = step;
        }
        previous
    }

    fn parser(id: &str) -> std::sync::Arc<dyn ToolParser> {
        by_name(id).unwrap()
    }

    fn names(out: &Extracted) -> Vec<&str> {
        out.calls.iter().map(|c| c.name.as_str()).collect()
    }

    fn args(call: &RawCall) -> Value {
        arguments_json(&call.args, None).unwrap()
    }

    // --- Hermes / Qwen ---------------------------------------------------------------------

    #[test]
    fn hermes_single_call_and_content() {
        let p = parser("hermes");
        let text = "Let me check.<tool_call>\n{\"name\":\"weather\",\"arguments\":{\"city\":\"Berlin\"}}\n</tool_call>Done";
        let out = run(p.as_ref(), text);
        assert_eq!(out.content, "Let me check.Done");
        assert_eq!(names(&out), ["weather"]);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin"}));
        assert_eq!(run_random(p.as_ref(), text, 7), out);
    }

    #[test]
    fn hermes_several_calls_array_and_string_arguments() {
        let p = parser("hermes");
        let text = "a<tool_call>[{\"name\":\"one\",\"arguments\":{\"x\":1}},{\"name\":\"two\",\"parameters\":{\"y\":\"z\"}}]</tool_call>b<tool_call>{\"name\":\"three\",\"arguments\":\"{\\\"k\\\":\\\"}\\\"}\"}</tool_call>c";
        let out = run(p.as_ref(), text);
        assert_eq!(out.content, "abc");
        assert_eq!(names(&out), ["one", "two", "three"]);
        assert_eq!(args(&out.calls[1]), json!({"y":"z"}));
        assert_eq!(args(&out.calls[2]), json!({"k":"}"}));
    }

    #[test]
    fn hermes_malformed_or_truncated_bodies_stay_content() {
        let p = parser("hermes");
        let bad = "x<tool_call>{\"name\":\"weather\",oops}</tool_call>y";
        let out = run(p.as_ref(), bad);
        assert_eq!(out.content, bad);
        assert!(out.calls.is_empty());
        let truncated = "hi<tool_call>{\"name\":\"weather\",\"arguments\":{\"ci";
        assert_eq!(extract(p.as_ref(), truncated, false).content, "hi");
        let out = extract(p.as_ref(), truncated, true);
        assert_eq!(out.content, truncated);
        assert!(out.calls.is_empty());
    }

    #[test]
    fn hermes_unterminated_but_complete_json_is_a_call_at_the_end() {
        let p = parser("hermes");
        let text = "<tool_call>{\"name\":\"weather\",\"arguments\":{}}";
        assert!(extract(p.as_ref(), text, false).calls.is_empty());
        assert_eq!(extract(p.as_ref(), text, true).calls.len(), 1);
    }

    #[test]
    fn half_markers_are_never_streamed() {
        let p = parser("hermes");
        for cut in ["<", "<tool", "<tool_cal", "</tool_ca"] {
            let text = format!("text{cut}");
            assert_eq!(extract(p.as_ref(), &text, false).content, "text", "{cut}");
            assert_eq!(extract(p.as_ref(), &text, true).content, text);
        }
    }

    // --- Qwen3-Coder, GLM -------------------------------------------------------------------

    #[test]
    fn qwen3_coder_xml_with_multiline_values_and_two_functions() {
        let p = parser("qwen3_coder");
        let text = "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n<parameter=text>\nline 1\nline 2\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=weather>\n<parameter=city>\nKöln\n</parameter>\n</function>\n</tool_call>";
        let out = run(p.as_ref(), text);
        assert_eq!(names(&out), ["write_file", "weather"]);
        assert_eq!(
            args(&out.calls[0]),
            json!({"path":"/tmp/a.txt","text":"line 1\nline 2"})
        );
        assert_eq!(args(&out.calls[1]), json!({"city":"Köln"}));
        assert_eq!(out.content.trim(), "");
        // The hermes parser reads the same bytes: the marker is shared.
        assert_eq!(
            names(&extract(parser("hermes").as_ref(), text, true)),
            ["write_file", "weather"]
        );
    }

    #[test]
    fn qwen3_coder_broken_xml_is_content() {
        let p = parser("qwen3_coder");
        let text = "<tool_call>\n<function=weather>\n<parameter=city>\nKöln\n</tool_call>";
        let out = run(p.as_ref(), text);
        assert!(out.calls.is_empty());
        assert_eq!(out.content, text);
    }

    #[test]
    fn glm_key_value_tags() {
        let p = parser("glm");
        let text = "ok\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Berlin</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>";
        let out = run(p.as_ref(), text);
        assert_eq!(out.content.trim(), "ok");
        assert_eq!(out.calls[0].name, "get_weather");
        // Untyped text: JSON literals become values.
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin","days":3}));
        let no_args = run(p.as_ref(), "<tool_call>get_time</tool_call>");
        assert_eq!(names(&no_args), ["get_time"]);
    }

    // --- MiniCPM5 -------------------------------------------------------------------------

    #[test]
    fn minicpm_function_param_tags() {
        let p = parser("minicpm");
        let text = "Ich schaue nach.<function name=\"get_weather\"><param name=\"city\">Köln</param><param name=\"days\">3</param></function>";
        let out = run(p.as_ref(), text);
        assert_eq!(out.content, "Ich schaue nach.");
        assert_eq!(out.calls[0].name, "get_weather");
        assert_eq!(args(&out.calls[0]), json!({"city":"Köln","days":3}));
        assert_eq!(run_random(p.as_ref(), text, 3), out);
    }

    #[test]
    fn minicpm_cdata_values_may_contain_markup() {
        let p = parser("minicpm");
        let text = "<function name=\"write\"><param name=\"code\"><![CDATA[if a < b {\n  </function> }]]></param></function><function name=\"two\"></function>";
        let out = run(p.as_ref(), text);
        assert_eq!(names(&out), ["write", "two"]);
        assert_eq!(
            args(&out.calls[0]),
            json!({"code":"if a < b {\n  </function> }"})
        );
        assert_eq!(args(&out.calls[1]), json!({}));
    }

    #[test]
    fn minicpm_text_that_only_looks_like_a_tag_is_content() {
        let p = parser("minicpm");
        for text in [
            "<functional> programming",
            "<function>x</function>",
            "a < b",
        ] {
            let out = run(p.as_ref(), text);
            assert!(out.calls.is_empty(), "{text}");
            assert_eq!(out.content, text);
        }
        // What the old server produced when the control tokens were dropped.
        let stripped = " name=\"get_weather\"> name=\"city\">Köln";
        assert_eq!(run(p.as_ref(), stripped).content, stripped);
    }

    // --- Llama 3.x ------------------------------------------------------------------------

    #[test]
    fn llama_bare_json_tagged_json_and_parallel_calls() {
        let p = parser("llama3");
        let bare = run(
            p.as_ref(),
            "{\"name\": \"weather\", \"parameters\": {\"city\": \"Berlin\"}}",
        );
        assert_eq!(bare.content, "");
        assert_eq!(args(&bare.calls[0]), json!({"city":"Berlin"}));
        let typed = run(
            p.as_ref(),
            "{\"type\":\"function\",\"name\":\"weather\",\"parameters\":{}}",
        );
        assert_eq!(names(&typed), ["weather"]);
        let tagged = run(p.as_ref(), "<|python_tag|>{\"name\":\"a\",\"parameters\":{\"p\":1}}; {\"name\":\"b\",\"parameters\":{}}");
        assert_eq!(names(&tagged), ["a", "b"]);
        let many = run(
            p.as_ref(),
            "{\"name\":\"a\",\"parameters\":{}};{\"name\":\"b\",\"parameters\":{}}",
        );
        assert_eq!(names(&many), ["a", "b"]);
    }

    #[test]
    fn llama_json_that_is_not_a_call_is_content() {
        let p = parser("llama3");
        for text in [
            "{\"answer\": 42}",
            "Here: {\"name\": \"x\", \"parameters\": {}}",
            "{ not json",
        ] {
            let out = run(p.as_ref(), text);
            assert!(out.calls.is_empty(), "{text}");
            assert_eq!(out.content, text);
        }
    }

    // --- Mistral --------------------------------------------------------------------------

    #[test]
    fn mistral_json_array_and_named_formats() {
        let p = parser("mistral");
        let old = "I will call it.[TOOL_CALLS][{\"name\":\"weather\",\"arguments\":{\"city\":\"Berlin\"}}] and then stop";
        let out = run(p.as_ref(), old);
        assert_eq!(out.content, "I will call it. and then stop");
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin"}));
        let new = "[TOOL_CALLS]weather[CALL_ID]abc123xyz[ARGS]{\"city\":\"Berlin\"}[TOOL_CALLS]time[ARGS]{}";
        let out = run(p.as_ref(), new);
        assert_eq!(names(&out), ["weather", "time"]);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin"}));
        let broken = run(p.as_ref(), "text[TOOL_CALLS] oops");
        assert_eq!(broken.content, "text[TOOL_CALLS] oops");
        assert!(broken.calls.is_empty());
    }

    // --- DeepSeek -------------------------------------------------------------------------

    #[test]
    fn deepseek_v31_and_r1_formats() {
        let p = parser("deepseek");
        let v31 = "Sure.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>weather<｜tool▁sep｜>{\"city\": \"Berlin\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>time<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let out = run(p.as_ref(), v31);
        assert_eq!(out.content, "Sure.");
        assert_eq!(names(&out), ["weather", "time"]);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin"}));
        let r1 = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>weather\n```json\n{\"city\": \"Berlin\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let out = run(p.as_ref(), r1);
        assert_eq!(names(&out), ["weather"]);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin"}));
        // The section end is optional once generation stops.
        let open =
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>weather<｜tool▁sep｜>{}<｜tool▁call▁end｜>";
        assert_eq!(extract(p.as_ref(), open, true).calls.len(), 1);
        assert!(extract(p.as_ref(), open, false).calls.is_empty());
    }

    #[test]
    fn deepseek_dsml_formats() {
        let text = "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Berlin</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n<｜DSML｜parameter name=\"tags\" string=\"false\">[\"a\",\"b\"]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>";
        let p = parser("deepseek_v32");
        let out = run(p.as_ref(), text);
        assert_eq!(names(&out), ["weather"]);
        assert_eq!(
            arguments_json(&out.calls[0].args, None).unwrap(),
            json!({"city":"Berlin","days":3,"tags":["a","b"]})
        );
        let v4 = text.replace("function_calls", "tool_calls");
        let out = run(parser("deepseek_v4").as_ref(), &v4);
        assert_eq!(names(&out), ["weather"]);
    }

    // --- Kimi K2 --------------------------------------------------------------------------

    #[test]
    fn kimi_k2_sections() {
        let p = parser("kimi");
        let text = "Thinking done.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Berlin\"}<|tool_call_end|><|tool_call_begin|>functions.mcp.search:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>";
        let out = run(p.as_ref(), text);
        assert_eq!(out.content, "Thinking done.");
        assert_eq!(names(&out), ["get_weather", "mcp.search"]);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin"}));
    }

    // --- gpt-oss ----------------------------------------------------------------------------

    #[test]
    fn gpt_oss_canonical_and_raw_spans() {
        let p = parser("gpt_oss");
        let canonical = "<|channel|>commentary to=functions.get_weather<|message|>{\"city\":\"Berlin\"}<|call|>";
        let out = run(p.as_ref(), canonical);
        assert_eq!(names(&out), ["get_weather"]);
        let raw = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Berlin\"}";
        let out = run(p.as_ref(), raw);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin"}));
        let bad = "<|channel|>commentary to=functions.get_weather<|message|>{oops}<|call|>";
        assert!(run(p.as_ref(), bad).calls.is_empty());
    }

    // --- Gemma ------------------------------------------------------------------------------

    #[test]
    fn gemma4_literals() {
        let p = parser("gemma4");
        let text = "<|tool_call>call:get_weather{city:<|\"|>Köln, DE<|\"|>,days:3,metric:true,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],opts:{units:<|\"|>c<|\"|>}}<tool_call|>";
        let out = run(p.as_ref(), text);
        assert_eq!(
            args(&out.calls[0]),
            json!({"city":"Köln, DE","days":3,"metric":true,"tags":["a","b"],"opts":{"units":"c"}})
        );
        let empty = run(p.as_ref(), "<|tool_call>call:ping{}<tool_call|>");
        assert_eq!(args(&empty.calls[0]), json!({}));
        let bad = run(p.as_ref(), "<|tool_call>call:ping{a:<|\"|>x}<tool_call|>");
        assert!(bad.calls.is_empty());
    }

    #[test]
    fn functiongemma_literals() {
        let p = parser("functiongemma");
        let text = "<start_function_call>call:get_weather{city:<escape>Berlin<escape>,days:2}<end_function_call>";
        let out = run(p.as_ref(), text);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin","days":2}));
    }

    // --- MiniMax, Granite, Phi-4, LFM2, K2-Horizon ---------------------------------------

    #[test]
    fn minimax_m2_invoke_tags_and_m1_json_lines() {
        let m2 = "<minimax:tool_call>\n<invoke name=\"weather\">\n<parameter name=\"city\">Berlin</parameter>\n<parameter name=\"days\">3</parameter>\n</invoke>\n<invoke name=\"time\">\n</invoke>\n</minimax:tool_call>";
        let out = run(parser("minimax").as_ref(), m2);
        assert_eq!(names(&out), ["weather", "time"]);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin","days":3}));
        let m1 = "<tool_calls>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n{\"name\": \"b\", \"arguments\": {}}\n</tool_calls>";
        let out = run(parser("minimax_m1").as_ref(), m1);
        assert_eq!(names(&out), ["a", "b"]);
    }

    #[test]
    fn granite3_and_phi4_json_lists() {
        let g = "<|tool_call|>[{\"name\":\"a\",\"arguments\":{\"x\":1}},{\"name\":\"b\",\"arguments\":{}}]";
        let out = run(parser("granite3").as_ref(), g);
        assert_eq!(names(&out), ["a", "b"]);
        let p = parser("phi4");
        let text = "import functools\nfunctools[{\"name\":\"weather\",\"arguments\":{\"city\":\"Berlin\"}}]";
        let out = run(p.as_ref(), text);
        assert_eq!(out.content, "import functools\n");
        assert_eq!(names(&out), ["weather"]);
        let prose = run(p.as_ref(), "use functools to reduce");
        assert_eq!(prose.content, "use functools to reduce");
    }

    #[test]
    fn lfm2_pythonic_and_json_calls() {
        let p = parser("lfm2");
        let text = "<|tool_call_start|>[get_weather(city=\"Köln\", days=3, metric=True, tags=['a', \"b\"]), time()]<|tool_call_end|>";
        let out = run(p.as_ref(), text);
        assert_eq!(names(&out), ["get_weather", "time"]);
        assert_eq!(
            args(&out.calls[0]),
            json!({"city":"Köln","days":3,"metric":true,"tags":["a","b"]})
        );
        let json_mode = run(
            p.as_ref(),
            "<|tool_call_start|>[{\"name\":\"a\",\"arguments\":{\"x\":1}}]<|tool_call_end|>",
        );
        assert_eq!(names(&json_mode), ["a"]);
        let bad = run(
            p.as_ref(),
            "<|tool_call_start|>[get_weather(\"positional\")]<|tool_call_end|>",
        );
        assert!(bad.calls.is_empty());
    }

    #[test]
    fn k2horizon_json_xml_and_typed_formats() {
        let p = parser("k2horizon");
        let json_fmt = "<ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Berlin\"}}</ifm|tool_call>\n</ifm|tool_calls>";
        let out = run(p.as_ref(), json_fmt);
        assert_eq!(args(&out.calls[0]), json!({"city":"Berlin"}));
        let xml = "<ifm|tool_calls>\n<ifm|tool_call>weather\n<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Berlin</ifm|arg_value>\n<ifm|arg_key>days</ifm|arg_key>\n<ifm|arg_value>3</ifm|arg_value>\n</ifm|tool_call>\n<ifm|tool_call>time\n</ifm|tool_call>\n</ifm|tool_calls>";
        let out = run(p.as_ref(), xml);
        assert_eq!(names(&out), ["weather", "time"]);
        let typed = "<ifm|tool_calls>\n<ifm|tool_call>weather\n<ifm|arg_key>zip</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n<ifm|arg_value>12345</ifm|arg_value>\n<ifm|arg_key>days</ifm|arg_key>\n<ifm|arg_type>integer</ifm|arg_type>\n<ifm|arg_value>3</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>";
        let out = run(p.as_ref(), typed);
        assert_eq!(args(&out.calls[0]), json!({"zip":"12345","days":3}));
        // A block without the wrapper still works.
        let bare = run(p.as_ref(), "<ifm|tool_call>weather\n<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>X</ifm|arg_value>\n</ifm|tool_call>");
        assert_eq!(names(&bare), ["weather"]);
    }

    // --- Typing, selection, robustness -------------------------------------------------------

    #[test]
    fn text_arguments_follow_the_schema() {
        let schema = json!({"type":"object","properties":{
            "zip":{"type":"string"},"n":{"type":"integer"},"x":{"type":"number"},
            "flag":{"type":"boolean"},"list":{"type":"array"},"obj":{"type":"object"},
            "maybe":{"anyOf":[{"type":"integer"},{"type":"null"}]}}});
        let args = CallArgs::Params(vec![
            ("zip".into(), Arg::Text("12345".into())),
            ("n".into(), Arg::Text(" 7 ".into())),
            ("x".into(), Arg::Text("2.5".into())),
            ("flag".into(), Arg::Text("True".into())),
            ("list".into(), Arg::Text("[1, 2]".into())),
            ("obj".into(), Arg::Text("{\"a\": 1}".into())),
            ("maybe".into(), Arg::Text("null".into())),
            ("free".into(), Arg::Text("hello".into())),
            ("count".into(), Arg::Text("12".into())),
            ("bad".into(), Arg::Text("abc".into())),
        ]);
        let value = arguments_json(&args, Some(&schema)).unwrap();
        assert_eq!(
            value,
            json!({"zip":"12345","n":7,"x":2.5,"flag":true,"list":[1,2],"obj":{"a":1},
                   "maybe":null,"free":"hello","count":12,"bad":"abc"})
        );
        let wrong = CallArgs::Params(vec![("n".into(), Arg::Text("seven".into()))]);
        assert_eq!(
            arguments_json(&wrong, Some(&schema)).unwrap(),
            json!({"n":"seven"})
        );
    }

    #[test]
    fn selection_uses_flag_then_template_then_architecture() {
        let id = |arch: &str, template: &str, flag: Option<&str>| {
            select(arch, template, flag).unwrap().id()
        };
        assert_eq!(id("llama", "{% if tools %}<tool_call>", None), "hermes");
        assert_eq!(
            id("llama", "<function name=\"x\"><param name=\"y\">", None),
            "minicpm"
        );
        assert_eq!(
            id("llama", "<function=ex>\n<parameter=a>", None),
            "qwen3_coder"
        );
        assert_eq!(id("x", "<tool_call>f<arg_key>k</arg_key>", None), "glm");
        assert_eq!(id("x", "[TOOL_CALLS]", None), "mistral");
        assert_eq!(id("x", "<|python_tag|>", None), "llama3");
        assert_eq!(id("x", "<ifm|tool_calls>", None), "k2horizon");
        assert_eq!(id("x", "<｜tool▁calls▁begin｜>", None), "deepseek");
        assert_eq!(id("x", "<｜DSML｜tool_calls>", None), "deepseek_v4");
        assert_eq!(id("x", "<|tool_calls_section_begin|>", None), "kimi");
        assert_eq!(id("x", "<|tool_call>call:", None), "gemma4");
        assert_eq!(id("x", "<minimax:tool_call>", None), "minimax");
        assert_eq!(id("x", "<|tool_call_start|>", None), "lfm2");
        assert_eq!(id("gpt-oss", "", None), "gpt_oss");
        assert_eq!(id("x", "to=functions.x", None), "gpt_oss");
        assert_eq!(id("minicpm", "", None), "minicpm");
        assert_eq!(id("llama", "", None), "hermes");
        assert_eq!(id("llama", "[TOOL_CALLS]", Some("llama3")), "llama3");
        assert_eq!(id("llama", "[TOOL_CALLS]", Some("auto")), "mistral");
        assert_eq!(id("llama", "", Some("Qwen3")), "hermes");
        assert!(select("llama", "", Some("nope")).is_err());
        assert!(is_known("auto") && is_known("glm-4.6") && !is_known("nope"));
        for id in parser_ids() {
            assert_eq!(by_name(id).unwrap().id(), id);
        }
    }

    #[test]
    fn every_parser_survives_hostile_input() {
        let mut inputs: Vec<String> = [
            "",
            "<",
            "{",
            "[",
            "}}}}",
            "\u{0}",
            "\u{FFFD}",
            "€€€",
            "日本語",
            "```json",
            "<<<>>>",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        // Every marker alone, doubled, followed by junk, and cut at every byte.
        for id in parser_ids() {
            let p = parser(id);
            for marker in p.hold_literals() {
                for suffix in ["", "{", "{\"name\":", "[", "x", "\n", "}{", "<|"] {
                    inputs.push(format!("{marker}{suffix}"));
                    inputs.push(format!("a{marker}{marker}{suffix}{marker}"));
                }
            }
        }
        for id in parser_ids() {
            let p = parser(id);
            for text in &inputs {
                for complete in [false, true] {
                    let out = extract(p.as_ref(), text, complete);
                    assert!(out.content.len() <= text.len(), "{id}: {text:?}");
                }
                // Cut inside multi-byte characters is not possible for &str; cut everywhere else.
                for (cut, _) in text.char_indices() {
                    let _ = extract(p.as_ref(), &text[..cut], false);
                }
            }
        }
    }

    // --- pythonic, xLAM, InternLM, Granite-20B-FC, Seed-OSS, Step-3, OLMo-3, tool_code, Hunyuan/Jamba ------------
    // Model outputs follow the examples in vLLM's parser tests (tests/entrypoints/openai/tool_parsers).

    #[test]
    fn pythonic_single_multiple_and_nested_arguments() {
        let p = parser("pythonic");
        let out = run(p.as_ref(), "[get_weather(city='San Francisco', metric='celsius')]");
        assert_eq!(names(&out), ["get_weather"]);
        assert_eq!(args(&out.calls[0]), json!({"city": "San Francisco", "metric": "celsius"}));
        assert_eq!(out.content, "");

        let text = "[get_weather(city='San Francisco', metric='celsius'), register_user(name='John Doe', age=37, address={'city': 'San Francisco', 'state': 'CA'}, role=None, passed_test=True, aliases=['John', 'Johnny'])]";
        let out = run(p.as_ref(), text);
        assert_eq!(names(&out), ["get_weather", "register_user"]);
        assert_eq!(
            args(&out.calls[1]),
            json!({"name": "John Doe", "age": 37, "address": {"city": "San Francisco", "state": "CA"}, "role": null, "passed_test": true, "aliases": ["John", "Johnny"]})
        );
        let out = run(p.as_ref(), "[get_current_time()]");
        assert_eq!(args(&out.calls[0]), json!({}));
    }

    #[test]
    fn pythonic_text_before_quotes_with_brackets_and_prose_brackets() {
        let p = parser("pythonic");
        let out = run(p.as_ref(), "Let me check.\n[search(query=\"a ] b ) c\", tags=['x', \"y'z\"])]");
        assert_eq!(out.content, "Let me check.\n");
        assert_eq!(args(&out.calls[0]), json!({"query": "a ] b ) c", "tags": ["x", "y'z"]}));
        // brackets in prose are not calls
        let prose = "See [1] and [note] or [a, b] and arr[0](x).";
        let out = run(p.as_ref(), prose);
        assert_eq!(out.content, prose);
        assert!(out.calls.is_empty());
    }

    #[test]
    fn pythonic_malformed_and_truncated_stay_content() {
        let p = parser("pythonic");
        for bad in ["[get_weather(city=", "[get_weather('x')]", "[get_weather(city='x'", "[f(a=1) g(b=2)]"] {
            let streamed = run(p.as_ref(), bad);
            assert!(streamed.calls.is_empty(), "{bad}: {:?}", streamed.calls);
            assert_eq!(streamed.content, bad, "{bad}");
        }
    }

    #[test]
    fn pythonic_text_after_the_list_stays_content() {
        let out = run(parser("pythonic").as_ref(), "[get_weather(city='x')] trailing");
        assert_eq!(names(&out), ["get_weather"]);
        assert_eq!(out.content, " trailing");
    }

    #[test]
    fn llama4_python_tags_olmo3_lines_and_tool_code_blocks() {
        let out = run(parser("llama4_pythonic").as_ref(), "<|python_start|>[get_weather(city=\"SF\")]<|python_end|>");
        assert_eq!(names(&out), ["get_weather"]);
        assert_eq!(out.content, "");
        let out = run(parser("olmo3").as_ref(), "Checking.\n<function_calls>\nget_weather(city='SF')\nget_time(zone=\"UTC\", fmt='24h')\n</function_calls>");
        assert_eq!(names(&out), ["get_weather", "get_time"]);
        assert_eq!(args(&out.calls[1]), json!({"zone": "UTC", "fmt": "24h"}));
        assert_eq!(out.content, "Checking.\n");
        let out = run(parser("tool_code").as_ref(), "```tool_code\nget_weather(city='SF')\n```\nDone.");
        assert_eq!(names(&out), ["get_weather"]);
        assert_eq!(out.content, "\nDone.");
        // truncated and empty blocks come back unchanged
        let cut = "<function_calls>\nget_weather(city=";
        assert_eq!(run(parser("olmo3").as_ref(), cut).content, cut);
        let empty = "<function_calls>\n</function_calls>";
        assert_eq!(run(parser("olmo3").as_ref(), empty).content, empty);
    }

    #[test]
    fn xlam_bare_fenced_and_tagged_arrays() {
        let p = parser("xlam");
        let out = run(p.as_ref(), "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}, {\"name\": \"calc\", \"arguments\": {\"expr\": \"2+2\"}}]");
        assert_eq!(names(&out), ["get_weather", "calc"]);
        assert_eq!(out.content, "");
        let out = run(p.as_ref(), "I will look it up.\n```json\n[{\"name\": \"f\", \"arguments\": {\"a\": 1}}]\n```");
        assert_eq!(names(&out), ["f"]);
        assert_eq!(out.content, "I will look it up.\n");
        let out = run(p.as_ref(), "</think>\n[TOOL_CALLS][{\"name\": \"g\", \"arguments\": {}}]");
        assert_eq!(names(&out), ["g"]);
        assert_eq!(out.content, "</think>\n");
        let prose = "The list [1, 2] and [x] stay.";
        assert_eq!(run(p.as_ref(), prose).content, prose);
        let bad = "[{\"name\": \"f\", \"arguments\": ";
        assert_eq!(run(p.as_ref(), bad).content, bad);
    }

    #[test]
    fn internlm_plugin_calls() {
        let p = parser("internlm");
        let text = "I need the weather.<|action_start|><|plugin|>\n{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Berlin\"}}<|action_end|>";
        let out = run(p.as_ref(), text);
        assert_eq!(out.content, "I need the weather.");
        assert_eq!(names(&out), ["get_weather"]);
        assert_eq!(args(&out.calls[0]), json!({"city": "Berlin"}));
        // the code interpreter block is not a function call
        let code = "<|action_start|><|interpreter|>\nprint(1)<|action_end|>";
        assert_eq!(run(p.as_ref(), code).content, code);
    }

    #[test]
    fn granite20b_repeated_function_call_markers() {
        let p = parser("granite20b");
        let text = "<function_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}\n<function_call> {\"name\": \"calc\", \"arguments\": {\"expr\": \"2+2\"}}";
        let out = run(p.as_ref(), text);
        assert_eq!(names(&out), ["get_weather", "calc"]);
        assert_eq!(out.content.trim(), "");
        // prose that merely mentions the marker is not buffered forever
        assert_eq!(run(p.as_ref(), "use <function_call> like this").content, "use <function_call> like this");
    }

    #[test]
    fn seed_oss_and_step3_xml_calls() {
        let text = "<seed:think>need weather</seed:think>\n<seed:tool_call>\n<function=get_weather>\n<parameter=location>Barcelona, Spain</parameter>\n<parameter=days>3</parameter>\n</function>\n</seed:tool_call>";
        let out = run(parser("seed_oss").as_ref(), text);
        assert_eq!(names(&out), ["get_weather"]);
        assert_eq!(args(&out.calls[0]), json!({"location": "Barcelona, Spain", "days": 3}));
        assert!(out.content.starts_with("<seed:think>"));

        let text = "Checking.<｜tool_calls_begin｜>\n<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name=\"get_weather\">\n<steptml:parameter name=\"location\">Paris</steptml:parameter>\n<steptml:parameter name=\"unit\">celsius</steptml:parameter>\n</steptml:invoke><｜tool_call_end｜>\n<｜tool_calls_end｜>";
        let out = run(parser("step3_steptml").as_ref(), text);
        assert_eq!(names(&out), ["get_weather"]);
        assert_eq!(args(&out.calls[0]), json!({"location": "Paris", "unit": "celsius"}));
        assert_eq!(out.content, "Checking.");
    }

    #[test]
    fn hunyuan_jamba_and_ernie_reuse_existing_formats() {
        for id in ["hunyuan", "jamba"] {
            let out = run(parser(id).as_ref(), "<tool_calls>[{\"name\": \"f\", \"arguments\": {\"a\": 1}}]</tool_calls>");
            assert_eq!(names(&out), ["f"], "{id}");
        }
        for id in ["ernie", "qwen3_omni", "hermes-2-pro"] {
            let out = run(parser(id).as_ref(), "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>");
            assert_eq!(names(&out), ["f"], "{id}");
        }
    }

    #[test]
    fn new_parsers_are_picked_from_their_templates() {
        for (template, id) in [
            ("{{ '<seed:tool_call>' }}<function=", "seed_oss"),
            ("<|action_start|><|plugin|>", "internlm"),
            ("<｜tool_calls_begin｜><steptml:invoke", "step3_steptml"),
            ("<function_call> {", "granite20b"),
            ("<|python_start|>[", "llama4_pythonic"),
            ("```tool_code", "tool_code"),
            ("<function_calls>", "olmo3"),
        ] {
            assert_eq!(select("llama", template, None).unwrap().id(), id, "{template}");
        }
    }

    #[test]
    fn plain_parser_passes_text_through() {
        let out = run(parser("none").as_ref(), "anything <tool_call> [TOOL_CALLS]");
        assert_eq!(out.content, "anything <tool_call> [TOOL_CALLS]");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn json_span_end_handles_strings_escapes_and_nesting() {
        assert_eq!(json_span_end("{\"a\":\"}\"}", 0), Some(9));
        assert_eq!(json_span_end("{\"a\":\"\\\\\"}", 0), Some(10));
        assert_eq!(json_span_end("{\"a\":{\"b\":[1,2]}}", 0), Some(17));
        assert_eq!(json_span_end("{\"a\":", 0), None);
        assert_eq!(json_span_end("not json", 0), None);
        assert_eq!(json_span_end("  [{\"a\":1}] tail", 0), Some(11));
    }

    #[test]
    fn filter_keeps_undeclared_calls_as_content() {
        let p = parser("hermes");
        let text = "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>";
        let out = extract_filtered(p.as_ref(), text, true, &|c| c.name == "weather");
        assert!(out.calls.is_empty());
        assert_eq!(out.content, text);
    }
}
