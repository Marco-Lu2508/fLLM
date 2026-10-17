#!/usr/bin/env python3
"""Writes rust/engine/testdata/chat_templates/cases.json: the conversation fixtures every chat template is rendered with."""
import json, sys

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. München"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
CALC = {"type": "function", "function": {"name": "calc", "description": "Evaluate an arithmetic expression.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}, "precision": {"type": "integer"}}, "required": ["expression"]}}}

def call(i, name, args):
    return {"id": f"call_{i:04d}", "type": "function", "function": {"name": name, "arguments": args}}

cases = [
 {"name": "user_only", "messages": [{"role": "user", "content": "Hello"}]},
 {"name": "system_user", "messages": [{"role": "system", "content": "You are a terse assistant."}, {"role": "user", "content": "Capital of France?"}]},
 {"name": "multi_turn", "messages": [
    {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user", "content": "Tell me a joke."}, {"role": "assistant", "content": "Why did the chicken cross the road?"}, {"role": "user", "content": "Why?"}]},
 {"name": "multi_turn_no_generation_prompt", "add_generation_prompt": False, "messages": [
    {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},
 {"name": "unicode", "messages": [{"role": "user", "content": "Grüße à tous — 你好 \U0001F600 \"quotes\" <b>&amp;</b> 'x'"}]},
 {"name": "tools_user", "tools": [WEATHER, CALC], "messages": [{"role": "user", "content": "Weather in München?"}]},
 {"name": "tools_system_user", "tools": [WEATHER], "messages": [{"role": "system", "content": "You can call tools."}, {"role": "user", "content": "Weather in Berlin?"}]},
 {"name": "tool_call_roundtrip", "tools": [WEATHER, CALC], "messages": [
    {"role": "user", "content": "Weather in Berlin and 2+2?"},
    {"role": "assistant", "content": "", "tool_calls": [call(1, "get_weather", {"city": "Berlin", "unit": "celsius"}), call(2, "calc", {"expression": "2+2", "precision": 2})]},
    {"role": "tool", "tool_call_id": "call_0001", "name": "get_weather", "content": "{\"temp\": 21.5, \"sky\": \"clear\"}"},
    {"role": "tool", "tool_call_id": "call_0002", "name": "calc", "content": "4"},
 ]},
 {"name": "tool_call_then_answer", "tools": [WEATHER], "messages": [
    {"role": "system", "content": "Use tools when helpful."},
    {"role": "user", "content": "Weather in Berlin?"},
    {"role": "assistant", "content": None, "tool_calls": [call(1, "get_weather", {"city": "Berlin"})]},
    {"role": "tool", "tool_call_id": "call_0001", "content": "21 C, sunny"},
    {"role": "assistant", "content": "It is 21 C and sunny in Berlin."},
    {"role": "user", "content": "And Paris?"}]},
 {"name": "assistant_with_text_and_tool_call", "tools": [CALC], "messages": [
    {"role": "user", "content": "Compute 17*3"},
    {"role": "assistant", "content": "Let me compute that.", "tool_calls": [call(7, "calc", {"expression": "17*3"})]},
    {"role": "tool", "tool_call_id": "call_0007", "content": "51"}]},
 {"name": "image_parts", "messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this picture?"}, {"type": "image"}]}]},
 {"name": "text_parts", "messages": [{"role": "system", "content": [{"type": "text", "text": "Sys part."}]}, {"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]}]},
 {"name": "thinking_disabled", "kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": "2+2?"}]},
 {"name": "thinking_enabled", "kwargs": {"enable_thinking": True}, "messages": [{"role": "user", "content": "2+2?"}]},
 {"name": "reasoning_history", "messages": [
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "<think>\nsimple\n</think>\n\n4", "reasoning_content": "simple"},
    {"role": "user", "content": "And 3+3?"}]},
 {"name": "reasoning_effort", "kwargs": {"reasoning_effort": "high"}, "messages": [{"role": "user", "content": "Think hard: 13*17"}]},
 {"name": "assistant_first", "messages": [{"role": "assistant", "content": "I start."}, {"role": "user", "content": "ok"}]},
 {"name": "two_users_in_a_row", "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]},
 {"name": "system_not_first", "messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "late system"}, {"role": "user", "content": "b"}]},
 {"name": "long_system_whitespace", "messages": [{"role": "system", "content": "  padded system with trailing newline\n\n"}, {"role": "user", "content": "  padded user \n"}]},
 {"name": "documents", "documents": [{"title": "Doc A", "text": "Alpha text."}, {"title": "Doc B", "text": "Beta text."}], "messages": [{"role": "user", "content": "Summarize."}]},
]
out = sys.argv[1] if len(sys.argv) > 1 else "rust/engine/testdata/chat_templates/cases.json"
json.dump(cases, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(cases), "cases")
