//! Grammar masks run directly in fLLM's sampler; they do not rewrite generated text.
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory};
use serde_json::{json, Value};

pub fn request_grammar(body: &Value) -> Result<Option<TopLevelGrammar>, String> {
    if let Some(grammar) = body.get("grammar") {
        let text = grammar.as_str().ok_or("grammar must be a string")?;
        let kind = body
            .get("grammar_type")
            .and_then(Value::as_str)
            .unwrap_or("gbnf");
        return Ok(Some(match kind {
            "lark" => TopLevelGrammar::from_lark(text.into()),
            "regex" => TopLevelGrammar::from_regex(text),
            "gbnf" => TopLevelGrammar::from_lark(gbnf_to_lark(text)?),
            _ => return Err("grammar_type must be gbnf, lark or regex".into()),
        }));
    }
    if let Some(format) = body.get("response_format") {
        let format = format
            .as_object()
            .ok_or("response_format must be an object")?;
        return match format.get("type").and_then(Value::as_str) {
            Some("text") => Ok(None),
            Some("json_object") => Ok(Some(TopLevelGrammar::from_json_schema(
                json!({"type":"object"}),
            ))),
            Some("json_schema") => {
                let schema = format
                    .get("json_schema")
                    .and_then(|v| v.get("schema"))
                    .ok_or("response_format.json_schema.schema is required")?;
                Ok(Some(TopLevelGrammar::from_json_schema(schema.clone())))
            }
            _ => Err("response_format.type must be text, json_object or json_schema".into()),
        };
    }
    if let Some(format) = body.get("format") {
        return match format {
            Value::String(s) if s == "json" => Ok(Some(TopLevelGrammar::from_json_schema(
                json!({"type":"object"}),
            ))),
            Value::String(s) if s.is_empty() => Ok(None),
            Value::Object(_) => Ok(Some(TopLevelGrammar::from_json_schema(format.clone()))),
            _ => Err("format must be 'json' or a JSON schema object".into()),
        };
    }
    Ok(None)
}

pub fn matcher(factory: &ParserFactory, grammar: TopLevelGrammar) -> Result<Matcher, String> {
    let mut matcher = Matcher::new(factory.create_parser(grammar));
    let mask = matcher
        .compute_mask()
        .map_err(|e| format!("invalid output grammar: {e}"))?;
    if mask.num_set() == 0 {
        return Err("output grammar cannot be expressed by this tokenizer".into());
    }
    Ok(matcher)
}

/// Translate the regular CFG subset shared by GBNF and Lark. Unsupported constructs
/// fail at compilation rather than being accepted as an unconstrained request.
fn gbnf_to_lark(text: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut names = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = line.split_once("::=") {
            let name = name.trim();
            if name.is_empty()
                || !name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return Err("invalid GBNF rule name".into());
            }
            names.push(name.to_string());
        }
    }
    if !names.iter().any(|n| n == "root") {
        return Err("GBNF grammar must define root ::= ...".into());
    }
    out.push_str("start: root\n");
    let mut quoted = false;
    let mut in_class = false;
    let mut escaped = false;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if escaped {
            out.push(c);
            escaped = false;
            i += 1;
            continue;
        }
        if c == '\\' {
            out.push(c);
            escaped = true;
            i += 1;
            continue;
        }
        if c == '"' && !in_class {
            quoted = !quoted;
            out.push(c);
            i += 1;
            continue;
        }
        if !quoted && c == '[' {
            in_class = true;
            out.push('/');
            out.push(c);
            i += 1;
            continue;
        }
        if in_class {
            out.push(c);
            if c == ']' {
                in_class = false;
                out.push('/');
            }
            i += 1;
            continue;
        }
        if !quoted && c == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if !quoted && i + 2 < chars.len() && chars[i..i + 3] == [':', ':', '='] {
            out.push(':');
            i += 3;
            continue;
        }
        if !quoted && c == '-' {
            out.push('_');
        } else {
            out.push(c);
        }
        i += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llguidance::toktrie::ApproximateTokEnv;
    #[test]
    fn json_schema_masks_invalid_choices_and_accepts_valid_object() {
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grammar = request_grammar(&json!({"response_format":{"type":"json_schema","json_schema":{"schema":{
            "type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false
        }}}})).unwrap().unwrap();
        let mut m = matcher(&factory, grammar).unwrap();
        assert!(!m.compute_mask().unwrap().is_allowed(b'x' as u32));
        m.consume_tokens(&env.tokenize("{\"ok\":true}")).unwrap();
        assert!(m.is_accepting().unwrap());
    }
    #[test]
    fn gbnf_string_and_character_class_enforce_language() {
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grammar = request_grammar(&json!({"grammar":"root ::= \"yes\" | [0-9]+"}))
            .unwrap()
            .unwrap();
        let mut m = matcher(&factory, grammar).unwrap();
        assert!(!m.compute_mask().unwrap().is_allowed(b'n' as u32));
        m.consume_tokens(&env.tokenize("yes")).unwrap();
        assert!(m.is_accepting().unwrap());
    }
}
