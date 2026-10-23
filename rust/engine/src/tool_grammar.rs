//! Grammars that make a model answer with plain text or with tool calls its tool parser can read.
//!
//! Small models often write a command as prose or break the JSON of a call. With
//! `tool_call_grammar` the server constrains decoding to
//! `text | call (separator call)*`, where `text` cannot contain the opening marker of a call
//! (so prose never turns into a call by accident), the function name is one of the declared
//! tools and the arguments follow that tool's JSON schema. The result is a Lark grammar for
//! llguidance; `%json` embeds the schema of the arguments.
//!
//! Supported call formats (by parser id): `hermes`, `qwen3_coder`, `llama3`, `mistral`,
//! `granite3`. Any other parser returns `None` and generation stays unconstrained.
use serde_json::{json, Value};

/// Lark string literal for `text`.
fn lit(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_else(|_| "\"\"".into())
}

/// Escape `text` for a Lark `/regex/` literal.
fn re_escape(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        if "\\.+*?()|[]{}^$#&-~/".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Definition of the terminal `name` for text that contains none of `markers` and, when `first`
/// is given, starts with a character of the class `first` (a regex class body such as `^{\s`).
/// llguidance writes `&` and `~` between the `/regex/` literals, not inside them.
fn text_terminal(name: &str, markers: &[&str], first: Option<&str>) -> String {
    let head = match first {
        Some(class) => format!("[{class}](?s:.*)"),
        None => "(?s:.+)".to_string(),
    };
    let mut out = format!("{name}: /{head}/");
    for marker in markers {
        out.push_str(&format!(" & ~/(?s:.*){}(?s:.*)/", re_escape(marker)));
    }
    out.push('\n');
    out
}

fn name_of(function: &Value) -> &str {
    function["name"].as_str().unwrap_or_default()
}

fn params_of(function: &Value) -> Value {
    function.get("parameters").cloned().unwrap_or(json!({"type":"object"}))
}

/// `%json` item for the whole call object `{"name":…,<key>:…}`.
fn call_object_schema(functions: &[Value], key: &str) -> Value {
    let any: Vec<Value> = functions
        .iter()
        .map(|f| {
            json!({"type":"object","properties":{"name":{"const":name_of(f)},key:params_of(f)},
                "required":["name",key],"additionalProperties":false})
        })
        .collect();
    json!({"anyOf": any})
}

fn json_item(schema: &Value) -> String {
    format!("%json {}", serde_json::to_string(schema).unwrap_or_else(|_| "{}".into()))
}

/// Text of one enum value in an XML-style parameter.
fn enum_text(value: &Value) -> String {
    value.as_str().map_or_else(|| value.to_string(), str::to_owned)
}

/// Grammar expression for the value of one XML-style parameter, typed by its schema.
fn xml_value(schema: &Value) -> String {
    if let Some(values) = schema.get("enum").and_then(Value::as_array).filter(|v| !v.is_empty()) {
        return format!("({})", values.iter().map(|v| lit(&enum_text(v))).collect::<Vec<_>>().join(" | "));
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("integer") => r"/-?[0-9]+/".into(),
        Some("number") => r"/-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?/".into(),
        Some("boolean") => r#"("true" | "false")"#.into(),
        Some("object" | "array") => json_item(schema),
        _ => "PARAM_TEXT".into(),
    }
}

/// `<function=NAME>` body of Qwen3-Coder: required parameters in schema order, optional ones may follow.
fn xml_function(function: &Value) -> String {
    let schema = params_of(function);
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut body = String::new();
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (key, property) in properties {
            let value = xml_value(property);
            // PARAM_TEXT contains its own closing tag: a separate literal would overlap with the text
            let close = if value == "PARAM_TEXT" { String::new() } else { lit("\n</parameter>\n") };
            let param = format!("{} {} {}", lit(&format!("<parameter={key}>\n")), value, close);
            if required.contains(&key.as_str()) {
                body.push_str(&format!(" {param}"));
            } else {
                body.push_str(&format!(" ({param})?"));
            }
        }
    }
    format!("{} {} {}", lit(&format!("<function={}>\n", name_of(function))), body, lit("</function>"))
}

/// Lark grammar for the tool-call format of `parser`, or `None` when the parser has no grammar.
///
/// `functions` are the declared functions the model may call (already narrowed by
/// `tool_choice`), `parallel` allows several calls in one reply and `allow_text` (false for
/// `tool_choice: required`) also allows a plain text answer.
pub fn lark_for(parser: &str, functions: &[Value], parallel: bool, allow_text: bool) -> Option<String> {
    if functions.is_empty() {
        return None;
    }
    // (call alternatives, separator between calls, markers text must not contain, first-char class of text)
    let (calls, separator, markers, first): (Vec<String>, &str, &[&str], Option<&str>) = match parser {
        "hermes" => (
            vec![format!("{} {} {}", lit("<tool_call>\n"), json_item(&call_object_schema(functions, "arguments")), lit("\n</tool_call>"))],
            r#""\n""#,
            &["<tool_call>"],
            None,
        ),
        "qwen3_coder" => (
            functions
                .iter()
                .map(|f| format!("{} {} {}", lit("<tool_call>\n"), xml_function(f), lit("\n</tool_call>")))
                .collect(),
            r#""\n""#,
            &["<tool_call>"],
            None,
        ),
        "llama3" => (
            vec![json_item(&call_object_schema(functions, "parameters"))],
            r#""; ""#,
            &["<|python_tag|>"],
            Some(r"^{\s"),
        ),
        "mistral" => (
            functions
                .iter()
                .map(|f| format!("{} {} {} {}", lit("[TOOL_CALLS]"), lit(name_of(f)), lit("[ARGS]"), json_item(&params_of(f))))
                .collect(),
            r#""""#,
            &["[TOOL_CALLS]"],
            None,
        ),
        "granite3" => {
            let item = call_object_schema(functions, "arguments");
            let mut array = json!({"type":"array","items":item,"minItems":1});
            if !parallel {
                array["maxItems"] = json!(1);
            }
            (vec![format!("{} {}", lit("<|tool_call|>"), json_item(&array))], r#""""#, &["<|tool_call|>"], None)
        }
        _ => return None,
    };
    let mut out = String::new();
    for (index, call) in calls.iter().enumerate() {
        out.push_str(&format!("call_{index}: {call}\n"));
    }
    let names: Vec<String> = (0..calls.len()).map(|i| format!("call_{i}")).collect();
    out.push_str(&format!("call_one: {}\n", names.join(" | ")));
    // granite3 carries several calls inside one array, so it never repeats the marker
    let repeat = parallel && parser != "granite3";
    let sequence = if repeat {
        if separator == r#""""# {
            "call_one+".to_string()
        } else {
            format!("call_one ({separator} call_one)*")
        }
    } else {
        "call_one".to_string()
    };
    out.push_str(&format!("calls: {sequence}\n"));
    if parser == "qwen3_coder" {
        // value, newline and the closing tag in one lexeme (see `xml_function`); the tag occurs only at the end
        out.push_str(
            "PARAM_TEXT: /(?s:.+)\\n<\\/parameter>\\n/ & ~/(?s:.*)<\\/parameter>(?s:.*)<\\/parameter>(?s:.*)/\n",
        );
    }
    if allow_text {
        out.push_str(&text_terminal("TEXT", markers, first));
        out.push_str("start: TEXT | calls\n");
    } else {
        out.push_str("start: calls\n");
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_parsers;
    use llguidance::{api::TopLevelGrammar, toktrie::ApproximateTokEnv, ParserFactory};

    fn tools() -> Vec<Value> {
        vec![
            json!({"name":"get_weather","parameters":{"type":"object","properties":{
                "city":{"type":"string"},"days":{"type":"integer"},"unit":{"enum":["c","f"]}},
                "required":["city"],"additionalProperties":false}}),
            json!({"name":"ping","parameters":{"type":"object","properties":{},"additionalProperties":false}}),
        ]
    }

    /// `true` when the grammar accepts exactly `text` as a complete output.
    fn accepts(lark: &str, text: &str) -> bool {
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let Ok(mut m) = crate::structured::matcher(&factory, TopLevelGrammar::from_lark(lark.into())) else {
            panic!("grammar does not compile:\n{lark}");
        };
        m.consume_tokens(&env.tokenize(text)).is_ok() && m.is_accepting().unwrap_or(false)
    }

    /// Number of leading bytes of `text` the grammar lets through (for readable failures).
    fn progress(lark: &str, text: &str) -> usize {
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let mut m = crate::structured::matcher(&factory, TopLevelGrammar::from_lark(lark.into())).unwrap();
        for (i, token) in env.tokenize(text).iter().enumerate() {
            if m.consume_tokens(&[*token]).is_err() {
                return i;
            }
        }
        text.len()
    }

    fn parses(parser: &str, text: &str) -> Vec<tool_parsers::RawCall> {
        let parser = tool_parsers::by_name(parser).unwrap();
        tool_parsers::extract(parser.as_ref(), text, true).calls
    }

    #[test]
    fn hermes_accepts_calls_and_text_only() {
        let g = lark_for("hermes", &tools(), true, true).unwrap();
        let call = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}\n</tool_call>";
        assert!(accepts(&g, call));
        assert_eq!(parses("hermes", call)[0].name, "get_weather");
        assert!(accepts(&g, &format!("{call}\n{call}")));
        assert!(accepts(&g, "I will list the files for you."));
        // prose cannot contain the marker, and a call cannot name another tool or break the schema
        assert!(!accepts(&g, "sure <tool_call>"));
        assert!(!accepts(&g, "<tool_call>\n{\"name\":\"rm\",\"arguments\":{}}\n</tool_call>"));
        assert!(!accepts(&g, "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{}}\n</tool_call>"));
        assert!(!accepts(&g, "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":3}}\n</tool_call>"));
        assert!(!accepts(&g, "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"a\""));
        // not parallel: one call only; required: no text
        let single = lark_for("hermes", &tools(), false, true).unwrap();
        assert!(accepts(&single, call) && !accepts(&single, &format!("{call}\n{call}")));
        let required = lark_for("hermes", &tools(), true, false).unwrap();
        assert!(accepts(&required, call) && !accepts(&required, "hello"));
    }

    #[test]
    fn llama_mistral_granite_formats_parse_with_their_parsers() {
        let g = lark_for("llama3", &tools(), true, true).unwrap();
        let call = "{\"name\":\"get_weather\",\"parameters\":{\"city\":\"Oslo\",\"days\":2}}";
        assert!(accepts(&g, call));
        assert_eq!(parses("llama3", call)[0].name, "get_weather");
        assert!(accepts(&g, &format!("{call}; {{\"name\":\"ping\",\"parameters\":{{}}}}")));
        assert!(accepts(&g, "Here is the answer."));
        assert!(!accepts(&g, "{\"foo\":1}"));

        let g = lark_for("mistral", &tools(), true, true).unwrap();
        let call = "[TOOL_CALLS]get_weather[ARGS]{\"city\":\"Oslo\",\"unit\":\"c\"}";
        assert!(accepts(&g, call));
        assert_eq!(parses("mistral", call)[0].name, "get_weather");
        assert!(accepts(&g, &format!("{call}[TOOL_CALLS]ping[ARGS]{{}}")));
        assert!(!accepts(&g, "[TOOL_CALLS]ping[ARGS]{\"x\":1}"));

        let g = lark_for("granite3", &tools(), true, true).unwrap();
        let call = "<|tool_call|>[{\"name\":\"ping\",\"arguments\":{}},{\"name\":\"get_weather\",\"arguments\":{\"city\":\"a\"}}]";
        assert!(accepts(&g, call));
        assert_eq!(parses("granite3", call).len(), 2);
        let g = lark_for("granite3", &tools(), false, true).unwrap();
        assert!(!accepts(&g, call));
    }

    #[test]
    fn qwen3_coder_xml_is_typed_by_the_schema() {
        let g = lark_for("qwen3_coder", &tools(), true, true).unwrap();
        let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=unit>\nf\n</parameter>\n</function>\n</tool_call>";
        let done = progress(&g, call);
        assert!(accepts(&g, call), "failed: stopped after {done} of {} bytes at {:?}; grammar {:?}", call.len(), &call[done.min(call.len())..], g);
        let calls = parses("qwen3_coder", call);
        assert_eq!(calls[0].name, "get_weather");
        let args = tool_parsers::arguments_json(&calls[0].args, Some(&tools()[0]["parameters"])).unwrap();
        assert_eq!(args, json!({"city":"Oslo","days":3,"unit":"f"}));
        assert!(accepts(&g, "<tool_call>\n<function=ping>\n</function>\n</tool_call>"));
        // missing required parameter, wrong type, value outside the enum
        assert!(!accepts(&g, "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"));
        assert!(!accepts(&g, &call.replace("\n3\n", "\nx\n")));
        assert!(!accepts(&g, &call.replace("\nf\n", "\nk\n")));
    }

    #[test]
    fn unsupported_parsers_have_no_grammar() {
        assert!(lark_for("gpt_oss", &tools(), true, true).is_none());
        assert!(lark_for("hermes", &[], true, true).is_none());
    }
}
