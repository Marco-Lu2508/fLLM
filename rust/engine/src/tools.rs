use crate::tool_parsers::{self, RawCall, ToolParser};
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Clone)]
pub struct ToolSpec {
    functions: Vec<Value>,
    pub forced_schema: Option<Value>,
    pub disabled: bool,
    parallel: bool,
    parser: Arc<dyn ToolParser>,
}

impl ToolSpec {
    pub fn from_request(body: &Value) -> Result<Self, String> {
        let mut functions = Vec::new();
        if let Some(tools) = body.get("tools") {
            for tool in tools.as_array().ok_or("tools must be an array")? {
                if tool.get("type").and_then(Value::as_str) != Some("function") {
                    return Err("only function tools are supported".into());
                }
                let function = tool
                    .get("function")
                    .filter(|f| f.is_object())
                    .ok_or("tool.function must be an object")?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|n| !n.is_empty() && n.len() <= 128)
                    .ok_or("tool function name is required")?;
                if functions.iter().any(|f: &Value| f["name"] == name) {
                    return Err("tool names must be unique".into());
                }
                let schema = function
                    .get("parameters")
                    .cloned()
                    .unwrap_or(json!({"type":"object"}));
                jsonschema::validator_for(&schema)
                    .map_err(|e| format!("invalid parameters schema for {name}: {e}"))?;
                functions.push(function.clone());
            }
        }
        let choice = body
            .get("tool_choice")
            .cloned()
            .unwrap_or(json!(if functions.is_empty() { "none" } else { "auto" }));
        let named = choice
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str);
        if choice.is_object()
            && (choice.get("type").and_then(Value::as_str) != Some("function") || named.is_none())
        {
            return Err("tool_choice object must select a function name".into());
        }
        let disabled = choice == "none";
        let required = choice == "required" || named.is_some();
        if !choice.is_object() && !matches!(choice.as_str(), Some("none" | "auto" | "required")) {
            return Err("tool_choice must be auto, none, required or a function selector".into());
        }
        if required && functions.is_empty() {
            return Err("tool_choice requires at least one tool".into());
        }
        if let Some(name) = named {
            if !functions.iter().any(|f| f["name"] == name) {
                return Err("tool_choice selects an undeclared tool".into());
            }
        }
        let forced_schema = required.then(|| json!({"oneOf":functions.iter().filter(|f| named.is_none_or(|n| f["name"] == n)).map(|f|
            json!({"type":"object","properties":{"name":{"const":f["name"]},"arguments":f.get("parameters").cloned().unwrap_or(json!({"type":"object"}))},"required":["name","arguments"],"additionalProperties":false})
        ).collect::<Vec<_>>()}));
        let parallel = match body.get("parallel_tool_calls") {
            None => true,
            Some(v) => v.as_bool().ok_or("parallel_tool_calls must be a boolean")?,
        };
        Ok(Self {
            functions,
            forced_schema,
            disabled,
            parallel,
            parser: tool_parsers::default_parser(),
        })
    }

    /// Read tool calls with the parser of the loaded model instead of the Hermes default.
    pub fn with_parser(mut self, parser: Arc<dyn ToolParser>) -> Self {
        self.parser = parser;
        self
    }

    /// `true` when the request declares tools the model may call.
    pub fn active(&self) -> bool {
        !self.disabled && !self.functions.is_empty()
    }

    /// Literals of the call format that must survive detokenization (see `ToolParser::markers`).
    pub fn markers(&self) -> Vec<&'static str> {
        if self.active() && self.forced_schema.is_none() {
            self.parser.markers()
        } else {
            Vec::new()
        }
    }

    /// Return visible content and validated API calls; incomplete tool spans stay buffered.
    ///
    /// `text` is everything generated so far. Spans that do not decode, and calls to
    /// functions the request did not declare, stay in the content unchanged; a call of a
    /// declared function whose arguments violate its schema is an error.
    pub fn parse_reply(
        &self,
        text: &str,
        final_chunk: bool,
        prefix: &str,
    ) -> Result<(String, Vec<Value>), String> {
        if self.disabled || self.functions.is_empty() {
            return Ok((text.into(), Vec::new()));
        }
        let mut content = String::new();
        let mut raw_calls: Vec<RawCall> = Vec::new();
        if self.forced_schema.is_some() {
            let payload = match serde_json::from_str::<Value>(text.trim()) {
                Ok(value) => value,
                Err(_) if !final_chunk => return Ok((String::new(), Vec::new())),
                Err(_) => return Err("generation ended before a complete function call was produced; increase max_tokens".into()),
            };
            let items = match payload {
                Value::Array(items) => items,
                item => vec![item],
            };
            for item in items {
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("tool call has no function name")?;
                let arguments = item
                    .get("arguments")
                    .or_else(|| item.get("parameters"))
                    .cloned()
                    .unwrap_or(json!({}));
                if !self.functions.iter().any(|f| f["name"] == name) {
                    return Err(format!("model called undeclared tool {name}"));
                }
                raw_calls.push(RawCall::json(name, arguments));
            }
        } else {
            let declared = |call: &RawCall| {
                self.functions
                    .iter()
                    .any(|f| f["name"] == call.name.as_str())
            };
            let extracted =
                tool_parsers::extract_filtered(self.parser.as_ref(), text, final_chunk, &declared);
            content = extracted.content;
            raw_calls = extracted.calls;
        }
        let mut calls = Vec::new();
        for raw in raw_calls {
            let name = raw.name.as_str();
            let function = self
                .functions
                .iter()
                .find(|f| f["name"] == name)
                .ok_or_else(|| format!("model called undeclared tool {name}"))?;
            let schema = function
                .get("parameters")
                .cloned()
                .unwrap_or(json!({"type":"object"}));
            let arguments = tool_parsers::arguments_json(&raw.args, Some(&schema))?;
            jsonschema::validator_for(&schema)
                .map_err(|e| e.to_string())?
                .validate(&arguments)
                .map_err(|e| format!("tool arguments do not match {name}: {e}"))?;
            calls.push(json!({"id":format!("call_{prefix}_{}",calls.len()),"type":"function","function":{"name":name,"arguments":arguments.to_string()}}));
        }
        if !self.parallel && calls.len() > 1 {
            return Err("model produced multiple calls while parallel_tool_calls is false".into());
        }
        Ok((content, calls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning::ReasoningParser;
    fn request() -> Value {
        json!({"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}]})
    }
    #[test]
    fn buffers_partial_tag_and_extracts_validated_calls() {
        let spec = ToolSpec::from_request(&request()).unwrap();
        assert_eq!(
            spec.parse_reply("hello <tool_ca", false, "r").unwrap().0,
            "hello "
        );
        let (content, calls) = spec
            .parse_reply(
                "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Berlin\"}}</tool_call>",
                true,
                "r",
            )
            .unwrap();
        assert!(content.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "weather");
        assert!(spec
            .parse_reply(
                "<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call>",
                true,
                "r"
            )
            .is_err());
    }
    #[test]
    fn required_and_named_choices_build_a_constraint() {
        let mut body = request();
        body["tool_choice"] = json!("required");
        assert!(ToolSpec::from_request(&body)
            .unwrap()
            .forced_schema
            .is_some());
        body["tool_choice"] = json!({"type":"function","function":{"name":"missing"}});
        assert!(ToolSpec::from_request(&body).is_err());
    }
    fn spec_with(parser: &str, body: Value) -> ToolSpec {
        ToolSpec::from_request(&body)
            .unwrap()
            .with_parser(tool_parsers::by_name(parser).unwrap())
    }

    fn weather_tools() -> Value {
        json!({"tools":[{"type":"function","function":{"name":"get_weather","parameters":{
            "type":"object",
            "properties":{"city":{"type":"string"},"days":{"type":"integer"},"zip":{"type":"string"}},
            "required":["city"]}}}]})
    }

    #[test]
    fn minicpm_reply_becomes_typed_tool_calls() {
        let spec = spec_with("minicpm", weather_tools());
        let reply = "<function name=\"get_weather\"><param name=\"city\">Köln</param><param name=\"days\">3</param><param name=\"zip\">50667</param></function>";
        let (content, calls) = spec.parse_reply(reply, true, "r").unwrap();
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        // The schema decides: `3` is a number, the zip code stays a string.
        let arguments: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(arguments, json!({"city":"Köln","days":3,"zip":"50667"}));
        // Streaming: nothing of the call leaks before it is complete.
        let (partial, none) = spec
            .parse_reply(&reply[..reply.len() - 5], false, "r")
            .unwrap();
        assert_eq!((partial.as_str(), none.len()), ("", 0));
        // Missing required argument is an error, as for every other format.
        assert!(spec
            .parse_reply("<function name=\"get_weather\"></function>", true, "r")
            .is_err());
    }

    #[test]
    fn undeclared_functions_and_prose_stay_content() {
        let spec = spec_with("hermes", weather_tools());
        let text = "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>";
        let (content, calls) = spec.parse_reply(text, true, "r").unwrap();
        assert_eq!(content, text);
        assert!(calls.is_empty());
        assert!(spec.markers().contains(&"<tool_call>"));
        let none = ToolSpec::from_request(&json!({})).unwrap();
        assert!(none.markers().is_empty() && !none.active());
    }

    #[test]
    fn gpt_oss_reply_through_the_reasoning_split() {
        let spec = spec_with("gpt_oss", weather_tools());
        let raw = "<|channel|>analysis<|message|>Weather needed.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Berlin\"}<|call|>";
        let parts = crate::reasoning::HarmonyReasoning.split(raw, true);
        let (content, calls) = spec.parse_reply(&parts.content, true, "r").unwrap();
        assert_eq!(content, "");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(parts.reasoning, "Weather needed.");
    }
}
