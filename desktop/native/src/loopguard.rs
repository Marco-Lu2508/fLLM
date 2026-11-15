//! The page's guard against a model that is stuck (`callSignature`, `isRepeating` and
//! `NO_LOOP_GUARD` of `code.js`, used by `runToolCall`): the same tool with the same arguments
//! three times in a row is refused the third time, with words that tell the model to try
//! something else. The conversation and every sub-agent have a history of their own.

use serde_json::Value;

/// The tools that are never counted (`NO_LOOP_GUARD`): polling and bookkeeping are meant to repeat.
pub const NO_LOOP_GUARD: [&str; 11] = [
    "bash_output",
    "todo_read",
    "todo_write",
    "ask_user",
    "task",
    "spawn_agent",
    "memory",
    "computer_screenshot",
    "computer_wait",
    "computer_cursor",
    "computer_displays",
];

/// How many signatures of the history the guard keeps (`history.length > 8`).
pub const HISTORY: usize = 8;

/// The card's words and the model's (`finish('error', …)` of the page).
pub const REPEATED_CARD: &str = "The same call was repeated three times.";

/// `Number.prototype.toString` for the numbers `JSON.stringify` writes: the shortest digits that
/// read back, in the exponent form from 1e21 and below 1e-6.
fn js_number(f: f64) -> String {
    if !f.is_finite() {
        return "null".into();
    }
    if f == 0.0 {
        return "0".into();
    }
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp.parse::<i32>().unwrap_or(0) + 1;
    let sign = if f < 0.0 { "-" } else { "" };
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let e = n - 1;
        let tail = if k == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        format!("{tail}e{}{}", if e < 0 { "-" } else { "+" }, e.abs())
    };
    format!("{sign}{body}")
}

/// Whether a key is an array index to JavaScript: those come first in an object, by value.
fn index_key(key: &str) -> Option<u32> {
    let canonical = key == "0" || (!key.starts_with('0') && !key.is_empty() && key.bytes().all(|b| b.is_ascii_digit()));
    canonical.then(|| key.parse::<u64>().ok()).flatten().filter(|n| *n < u32::MAX as u64).map(|n| n as u32)
}

/// `JSON.stringify(norm(value))`: object keys sorted (by UTF-16 units, array indices first),
/// numbers the way JavaScript writes them.
fn normalized(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&js_number(n.as_f64().unwrap_or(f64::NAN))),
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                normalized(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by_key(|k| k.encode_utf16().collect::<Vec<u16>>());
            let (mut indexed, rest): (Vec<&String>, Vec<&String>) = keys.into_iter().partition(|k| index_key(k).is_some());
            indexed.sort_by_key(|k| index_key(k));
            out.push('{');
            for (i, key) in indexed.into_iter().chain(rest).enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                normalized(&map[key], out);
            }
            out.push('}');
        }
    }
}

/// `callSignature(name, args)`: the same tool with the same arguments, whatever the key order.
pub fn call_signature(name: &str, args: &Value) -> String {
    let mut out = String::new();
    normalized(args, &mut out);
    format!("{name}:{out}")
}

/// `isRepeating(history, signature, times)`: the last `times - 1` calls are this one.
pub fn is_repeating(history: &[String], signature: &str, times: usize) -> bool {
    if history.len() < times.saturating_sub(1) {
        return false;
    }
    history[history.len() - (times - 1)..].iter().all(|s| s == signature)
}

/// What `runToolCall` does for one call: writes it into the history (at most eight kept) and says
/// whether it is the third of the same.
pub fn check(history: &mut Vec<String>, name: &str, args: &Value) -> bool {
    let signature = call_signature(name, args);
    let looping = !NO_LOOP_GUARD.contains(&name) && is_repeating(history, &signature, 3);
    history.push(signature);
    if history.len() > HISTORY {
        history.remove(0);
    }
    looping
}

/// What the model is told.
pub const REPEATED_REPLY: &str = "Error: you have made this exact call three times in a row. Repeating it will not change the result. Try a different approach (another tool, other arguments, a smaller step), or ask the user with ask_user.";
