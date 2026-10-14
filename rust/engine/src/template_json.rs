//! Python-compatible `tojson` for chat templates.
//!
//! Hugging Face templates call `x|tojson(ensure_ascii=False, indent=2, sort_keys=True,
//! separators=(',', ':'))` and expect `json.dumps` output (`{"a": 1, "b": [1, 2]}`). minijinja's
//! own filter rejects those keyword arguments, escapes `<`, `>`, `&` and `'` and prints compact
//! JSON, which changes the prompt text for every model that serialises tools.
use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Error, ErrorKind};

struct Style {
    ensure_ascii: bool,
    indent: Option<usize>,
    sort_keys: bool,
    item_separator: String,
    key_separator: String,
}

/// The `tojson` filter: `json.dumps(value, ensure_ascii=False, ...)` semantics.
pub fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let indent = kwargs.get::<Option<usize>>("indent")?;
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let separators = kwargs.get::<Option<Vec<String>>>("separators")?;
    kwargs.assert_all_used()?;
    let (item_separator, key_separator) = match separators {
        Some(pair) if pair.len() == 2 => (pair[0].clone(), pair[1].clone()),
        Some(_) => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "separators must be a pair of strings",
            ))
        }
        None if indent.is_some() => (",".to_owned(), ": ".to_owned()),
        None => (", ".to_owned(), ": ".to_owned()),
    };
    let style = Style {
        ensure_ascii,
        indent,
        sort_keys,
        item_separator,
        key_separator,
    };
    let mut out = String::new();
    write_value(&value, &style, 0, &mut out)?;
    Ok(Value::from_safe_string(out))
}

fn newline(style: &Style, depth: usize, out: &mut String) {
    if let Some(width) = style.indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', width * depth));
    }
}

fn write_value(value: &Value, style: &Style, depth: usize, out: &mut String) -> Result<(), Error> {
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            let text = serde_json::to_string(value)
                .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
            out.push_str(&python_exponent(&text));
        }
        ValueKind::String => write_string(value.as_str().unwrap_or_default(), style, out),
        ValueKind::Map => {
            let mut keys: Vec<Value> = value.try_iter()?.collect();
            if style.sort_keys {
                keys.sort_by_key(|k| k.to_string());
            }
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(&style.item_separator);
                }
                newline(style, depth + 1, out);
                write_string(&key.to_string(), style, out);
                out.push_str(&style.key_separator);
                let item = value.get_item(key)?;
                write_value(&item, style, depth + 1, out)?;
            }
            newline(style, depth, out);
            out.push('}');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = value.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&style.item_separator);
                }
                newline(style, depth + 1, out);
                write_value(item, style, depth + 1, out)?;
            }
            newline(style, depth, out);
            out.push(']');
        }
        _ => write_string(&value.to_string(), style, out),
    }
    Ok(())
}

/// Python prints exponents with at least two digits (`1e-07`), serde_json with as many as needed.
fn python_exponent(text: &str) -> String {
    let Some((mantissa, exponent)) = text.split_once('e') else {
        return text.to_owned();
    };
    let (sign, digits) = match exponent.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("+", exponent.strip_prefix('+').unwrap_or(exponent)),
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

fn write_string(text: &str, style: &Style, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if style.ensure_ascii && (!c.is_ascii() || c == '\u{7f}') => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use minijinja::Environment;

    fn render(template: &str) -> String {
        let mut env = Environment::new();
        env.add_filter("tojson", super::tojson);
        env.render_str(template, minijinja::Value::UNDEFINED).unwrap()
    }

    #[test]
    fn matches_python_json_dumps() {
        assert_eq!(
            render("{{ {'a': 1, 'b': [1, 2.5, true, none], 'c': {}} | tojson }}"),
            "{\"a\": 1, \"b\": [1, 2.5, true, null], \"c\": {}}"
        );
        assert_eq!(render("{{ 'x<y>&\\'z\"' | tojson }}"), "\"x<y>&'z\\\"\"");
    }

    #[test]
    fn request_key_order_is_preserved() {
        let request: serde_json::Value =
            serde_json::from_str(r#"{"type":"function","function":{"name":"f","description":"d"}}"#)
                .unwrap();
        let mut env = Environment::new();
        env.add_filter("tojson", super::tojson);
        let out = env
            .render_str("{{ tool | tojson }}", minijinja::context! { tool => request })
            .unwrap();
        assert_eq!(
            out,
            r#"{"type": "function", "function": {"name": "f", "description": "d"}}"#
        );
    }

    #[test]
    fn exponents_use_python_notation() {
        assert_eq!(super::python_exponent("1e-7"), "1e-07");
        assert_eq!(super::python_exponent("1.5e-10"), "1.5e-10");
        assert_eq!(super::python_exponent("1e20"), "1e+20");
        assert_eq!(super::python_exponent("0.5"), "0.5");
    }

    #[test]
    fn keyword_arguments_are_accepted() {
        assert_eq!(render("{{ 'é日' | tojson(ensure_ascii=False) }}"), "\"é日\"");
        assert_eq!(
            render("{{ 'é日😀' | tojson(ensure_ascii=True) }}"),
            "\"\\u00e9\\u65e5\\ud83d\\ude00\""
        );
        assert_eq!(
            render("{{ {'b': 1, 'a': [1]} | tojson(sort_keys=True, separators=(',', ':')) }}"),
            "{\"a\":[1],\"b\":1}"
        );
        assert_eq!(
            render("{{ {'a': [1, 2], 'b': {}} | tojson(indent=2) }}"),
            "{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": {}\n}"
        );
    }
}
