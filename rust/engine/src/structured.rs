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

/// Jump-forward decoding (the "compressed finite state machine" idea of SGLang).
///
/// Where the grammar allows exactly one continuation for the next several bytes (the fixed keys, quotes and punctuation of a JSON
/// schema, a literal in a Lark rule), the engine does not need the model to choose them. `plan` asks the grammar for those forced
/// bytes, turns them into tokens and consumes them in the grammar; the runtime then appends the tokens and lets the model process
/// all of them in one forward pass (a prefill chunk) instead of one pass per token.
///
/// Rules that keep the output valid and, where possible, identical to token-by-token decoding:
///
/// 1. Nothing is forced when the grammar currently accepts (end of sequence is an alternative) or has stopped, and a jump stops at
///    the first token after which the grammar accepts.
/// 2. Only bytes the grammar itself forces (`Matcher::compute_ff_bytes`) are turned into tokens, the bytes of the chosen tokens
///    must spell exactly those bytes (checked, never assumed), and the tokens are validated against the grammar before use.
/// 3. With a vocabulary of single-byte tokens a byte string has exactly one tokenisation, so a forced byte is also the only token
///    the mask would allow: the result is token for token identical to decoding without jump-forward, and no random number is
///    spent on it (a one-element distribution draws none).
/// 4. With multi-byte tokens the forced text is tokenised canonically together with the token that precedes it (the tokenisation
///    must keep that token as its first token, otherwise the greedy longest match of the vocabulary is used), and the last token of
///    the forced text is not jumped over: it may merge with whatever the model writes next. Its bytes stay forced, so the next
///    sampled token has to start with them (token healing, the way llguidance treats forced bytes). The text is still the same
///    text; the token sequence may differ from what token-by-token sampling would have produced.
pub struct JumpForward {
    env: llguidance::toktrie::TokEnv,
    canonical: Option<CanonicalEncoder>,
    multi_byte: bool,
}

/// Whether jump-forward decoding is on unless a flag says otherwise: it is, except when `FLLM_JUMP_FORWARD` is `off`, `0`, `no` or
/// `false` (`--no-jump-forward` turns it off on the command line).
pub fn jump_forward_default() -> bool {
    !matches!(std::env::var("FLLM_JUMP_FORWARD").unwrap_or_default().trim().to_ascii_lowercase().as_str(), "off" | "0" | "no" | "false")
}

/// Encodes plain text (no BOS, no special tokens) with the model's tokenizer.
pub type CanonicalEncoder = Box<dyn Fn(&str) -> Option<Vec<u32>> + Send + Sync>;

impl JumpForward {
    /// Without `canonical` the greedy longest match over the vocabulary tokenises forced text.
    pub fn new(env: llguidance::toktrie::TokEnv, canonical: Option<CanonicalEncoder>) -> Self {
        let trie = env.tok_trie();
        let multi_byte = (0..trie.vocab_size() as u32).any(|id| !trie.is_special_token(id) && trie.token_len(id) > 1);
        Self { env, canonical, multi_byte }
    }

    /// Whether forced text can be tokenised in more than one way (some token spans several bytes).
    pub fn multi_byte(&self) -> bool {
        self.multi_byte
    }

    /// Number of leading `tokens` that spell the start of `bytes` exactly.
    fn spelled_prefix(&self, tokens: &[u32], bytes: &[u8]) -> usize {
        let trie = self.env.tok_trie();
        let mut at = 0;
        for (i, &t) in tokens.iter().enumerate() {
            let piece = trie.token(t);
            if trie.is_special_token(t) || piece.is_empty() || bytes.get(at..at + piece.len()) != Some(piece) {
                return i;
            }
            at += piece.len();
        }
        tokens.len()
    }

    /// Tokens that spell the start of the forced `bytes` which follow the sampled token `last`.
    fn cover(&self, last: u32, bytes: &[u8]) -> Vec<u32> {
        let trie = self.env.tok_trie();
        if let (true, Some(encode), false) = (self.multi_byte, &self.canonical, trie.is_special_token(last)) {
            let mut context = trie.token(last).to_vec();
            context.extend_from_slice(bytes);
            if let Some(ids) = std::str::from_utf8(&context).ok().and_then(encode) {
                if ids.first() == Some(&last) {
                    let rest = &ids[1..];
                    if self.spelled_prefix(rest, bytes) == rest.len() && rest.iter().map(|&t| trie.token_len(t)).sum::<usize>() == bytes.len() {
                        return rest.to_vec();
                    }
                }
            }
        }
        let greedy = trie.greedy_tokenize(bytes);
        let n = self.spelled_prefix(&greedy, bytes);
        greedy[..n].to_vec()
    }

    /// The tokens the grammar forces next, already consumed by `matcher`, at most `budget` of them. Empty when nothing is forced
    /// or when it is not safe to jump; the matcher is then where plain decoding would find it, except that forced bytes may be
    /// pending as the prefix of the next token (rule 4).
    pub fn plan(&self, matcher: &mut Matcher, last: u32, budget: usize) -> Vec<u32> {
        if budget == 0 || matcher.is_stopped() || !matches!(matcher.is_accepting(), Ok(false)) {
            return Vec::new();
        }
        let forced = matcher.compute_ff_bytes();
        if forced.is_empty() {
            return Vec::new();
        }
        let mut tokens = self.cover(last, &forced);
        if self.multi_byte {
            tokens.pop();
        }
        tokens.truncate(budget);
        let valid = matcher.validate_tokens(&tokens).unwrap_or(0);
        tokens.truncate(valid);
        let mut taken = Vec::with_capacity(tokens.len());
        for t in tokens {
            if matcher.consume_token(t).is_err() {
                break;
            }
            taken.push(t);
            if matcher.is_stopped() || !matches!(matcher.is_accepting(), Ok(false)) {
                break;
            }
        }
        taken
    }
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
