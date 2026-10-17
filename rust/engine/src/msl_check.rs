//! Static checks of the Metal Shading Language sources (`rust/engine/metal/*.metal`) that run on every platform.
//!
//! Nobody can compile MSL on the Linux hosts that develop the kernels, and the CI runner that can (Apple silicon) only reports a failure
//! after a push. This module catches the classes of compiler errors that have actually occurred (or that are easy to make) with a small
//! lexical scan of each kernel function:
//!
//! * thread-position attributes of one kernel must have the same type (all scalar, or all `uint3`, ...): MSL rejects a kernel that
//!   mixes `uint gid [[thread_position_in_grid]]` with `uint3 tg [[threadgroup_position_in_grid]]`;
//! * every pointer or reference parameter of a kernel names an address space (`device`, `constant`, `threadgroup`, `thread`);
//! * `[[buffer(n)]]` indices of a kernel are distinct and below 31;
//! * `threadgroup` variables are declared at the top level of a kernel body, and the static threadgroup memory of a kernel stays within
//!   32 KiB (when the array sizes can be evaluated);
//! * program-scope variables live in the `constant` address space;
//! * no `double`, `long double`, `printf` or C++ library names, no `erf` (Metal has no such function: use `o_erf*` of common_ops.metal);
//! * kernel names are unique across the sources.
//!
//! It is a lint, not a compiler: it does not replace the CI run on a Mac, and it will not find type errors.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Attributes whose type must agree within one kernel (position-like built-ins). `thread_index_in_threadgroup` and the simdgroup
/// attributes are always scalar and not part of the rule.
const POSITION_ATTRIBUTES: &[&str] = &[
    "thread_position_in_grid",
    "threads_per_grid",
    "thread_position_in_threadgroup",
    "threads_per_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "dispatch_threads_per_threadgroup",
];

/// Threadgroup memory limit of every Apple GPU family in bytes.
const THREADGROUP_LIMIT: usize = 32 * 1024;

/// `text` without comments and string literals (replaced by blanks, newlines kept so that line numbers stay valid).
fn strip_comments(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            out.push_str("  ");
            while i < b.len() && !(b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/') {
                out.push(if b[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
            i += 2;
            out.push_str("  ");
        } else if b[i] == b'"' {
            // keep the quotes (host_name("...") is read from the original text), blank the content
            out.push('"');
            i += 1;
            while i < b.len() && b[i] != b'"' {
                out.push(' ');
                i += 1;
            }
            out.push('"');
            i += 1;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn line_of(text: &str, at: usize) -> usize {
    text[..at.min(text.len())].bytes().filter(|&c| c == b'\n').count() + 1
}

/// A function definition found at program scope.
struct Function {
    header: String,
    /// byte range of the body without the braces
    body: (usize, usize),
    header_at: usize,
}

impl Function {
    fn is_kernel(&self) -> bool {
        has_word(&self.header, "kernel")
    }

    /// Name: the identifier before the first top-level `(` of the parameter list.
    fn name(&self) -> String {
        let h = self.header.as_str();
        let open = param_open(h).unwrap_or(h.len());
        let before = h[..open].trim_end();
        let start = before
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(0, |i| i + 1);
        before[start..].to_owned()
    }

    fn params(&self) -> Vec<String> {
        let h = self.header.as_str();
        let Some(open) = param_open(h) else { return Vec::new() };
        let mut depth = 0i32;
        let mut end = h.len();
        for (i, c) in h[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        split_top_level(&h[open + 1..end])
    }
}

/// Index of the `(` that opens the parameter list: the first `(` that is not inside `[[...]]` or `<...>` and follows an identifier that is
/// not `decltype`/`template`-like punctuation.
fn param_open(h: &str) -> Option<usize> {
    let b = h.as_bytes();
    let mut square = 0i32;
    let mut angle = 0i32;
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'[' => square += 1,
            b']' => square -= 1,
            b'<' => angle += 1,
            b'>' => angle -= 1,
            b'(' if square == 0 && angle <= 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn split_top_level(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            _ => {}
        }
        if c == ',' && depth == 0 {
            parts.push(cur.trim().to_owned());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur.trim().to_owned());
    }
    parts
}

fn is_word_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Whether `word` occurs in `text` as a whole identifier.
fn has_word(text: &str, word: &str) -> bool {
    find_word(text, word, 0).is_some()
}

fn find_word(text: &str, word: &str, from: usize) -> Option<usize> {
    let b = text.as_bytes();
    let mut at = from;
    while let Some(i) = text[at..].find(word) {
        let s = at + i;
        let e = s + word.len();
        let before_ok = s == 0 || !is_word_char(b[s - 1]);
        let after_ok = e >= b.len() || !is_word_char(b[e]);
        if before_ok && after_ok {
            return Some(s);
        }
        at = s + 1;
    }
    None
}

/// Function definitions at program scope (the header is the text since the previous `;` or `}` at depth 0).
fn functions(text: &str) -> Vec<Function> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut stmt_start = 0usize;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'{' => {
                if depth == 0 {
                    let header = text[stmt_start..i].trim().to_owned();
                    // body end
                    let mut d = 0i32;
                    let mut j = i;
                    while j < b.len() {
                        match b[j] {
                            b'{' => d += 1,
                            b'}' => {
                                d -= 1;
                                if d == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        j += 1;
                    }
                    let is_type = ["struct ", "namespace ", "enum ", "union ", "class ", "typedef "]
                        .iter()
                        .any(|k| header.starts_with(k) || header.contains(&format!(" {k}")));
                    let looks_like_function = header.contains('(') && !is_type && !header.contains('=');
                    if looks_like_function {
                        out.push(Function { header_at: stmt_start, header, body: (i + 1, j) });
                    }
                    i = j + 1;
                    stmt_start = i;
                    continue;
                }
                depth += 1;
            }
            b'}' => depth -= 1,
            b';' if depth == 0 => stmt_start = i + 1,
            b'\n' if depth == 0 => {
                // preprocessor lines end a statement
                let line_start = text[..i].rfind('\n').map_or(0, |p| p + 1);
                if text[line_start..i].trim_start().starts_with('#') {
                    stmt_start = i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// Type text of a parameter: everything before the last identifier and before the `[[...]]` attribute.
fn param_type_and_attrs(p: &str) -> (String, Vec<String>) {
    let (decl, attrs) = match p.find("[[") {
        Some(i) => (p[..i].trim(), &p[i..]),
        None => (p.trim(), ""),
    };
    let mut names = Vec::new();
    let mut rest = attrs;
    while let Some(i) = rest.find("[[") {
        let tail = &rest[i + 2..];
        let end = tail.find("]]").unwrap_or(tail.len());
        names.push(tail[..end].trim().to_owned());
        rest = &tail[end..];
    }
    let ty_end = decl.rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(0, |i| i + 1);
    (decl[..ty_end].trim().replace(['&', '*'], " ").split_whitespace().collect::<Vec<_>>().join(" "), names)
}

fn evaluate(expr: &str, defines: &HashMap<String, i64>) -> Option<i64> {
    // tiny recursive-descent evaluator for + - * / ( ) with integer literals (u suffix allowed) and known names
    struct P<'a> {
        t: Vec<String>,
        at: usize,
        defines: &'a HashMap<String, i64>,
    }
    impl P<'_> {
        fn peek(&self) -> Option<&str> {
            self.t.get(self.at).map(String::as_str)
        }
        fn atom(&mut self) -> Option<i64> {
            let t = self.t.get(self.at)?.clone();
            self.at += 1;
            if t == "(" {
                let v = self.sum()?;
                if self.peek() != Some(")") {
                    return None;
                }
                self.at += 1;
                return Some(v);
            }
            if t == "-" {
                return self.atom().map(|v| -v);
            }
            let digits = t.trim_end_matches(['u', 'U', 'l', 'L']);
            if let Ok(v) = digits.parse::<i64>() {
                return Some(v);
            }
            self.defines.get(&t).copied()
        }
        fn product(&mut self) -> Option<i64> {
            let mut v = self.atom()?;
            while let Some(op) = self.peek() {
                match op {
                    "*" => {
                        self.at += 1;
                        v = v.checked_mul(self.atom()?)?;
                    }
                    "/" => {
                        self.at += 1;
                        let d = self.atom()?;
                        if d == 0 {
                            return None;
                        }
                        v /= d;
                    }
                    _ => break,
                }
            }
            Some(v)
        }
        fn sum(&mut self) -> Option<i64> {
            let mut v = self.product()?;
            while let Some(op) = self.peek() {
                match op {
                    "+" => {
                        self.at += 1;
                        v = v.checked_add(self.product()?)?;
                    }
                    "-" => {
                        self.at += 1;
                        v = v.checked_sub(self.product()?)?;
                    }
                    _ => break,
                }
            }
            Some(v)
        }
    }
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for c in expr.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
            if !c.is_whitespace() {
                tokens.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    let mut p = P { t: tokens, at: 0, defines };
    let v = p.sum()?;
    (p.at == p.t.len()).then_some(v)
}

/// Integer constants of the unit: `#define NAME <expr>`, `constant constexpr uint NAME = <expr>;` and similar.
fn constants(text: &str) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    for _ in 0..3 {
        for line in text.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("#define ") {
                let mut it = rest.splitn(2, char::is_whitespace);
                let name = it.next().unwrap_or("");
                if name.contains('(') {
                    continue;
                }
                if let Some(v) = it.next().and_then(|e| evaluate(e, &map)) {
                    map.insert(name.to_owned(), v);
                }
            } else if (t.contains("constexpr") || t.starts_with("constant ")) && t.contains('=') && t.ends_with(';') && !t.contains('(') {
                let eq = t.find('=').unwrap_or(0);
                let name = t[..eq].split_whitespace().last().unwrap_or("");
                if let Some(v) = evaluate(t[eq + 1..t.len() - 1].trim(), &map) {
                    map.insert(name.to_owned(), v);
                }
            }
        }
    }
    map
}

fn type_bytes(ty: &str) -> Option<usize> {
    Some(match ty {
        "float" | "int" | "uint" | "unsigned" => 4,
        "half" | "short" | "ushort" | "bfloat" => 2,
        "char" | "uchar" | "bool" => 1,
        "ulong" | "long" => 8,
        "float2" | "uint2" | "int2" => 8,
        "float3" | "float4" | "uint3" | "uint4" | "int4" | "int3" => 16,
        "half2" => 4,
        "half4" => 8,
        _ => return None,
    })
}

/// Everything wrong with one source file (`name` is used in the messages). `defines_from` holds shared files that are prepended to the unit by
/// the backend (their constants are visible).
pub(crate) fn check_source(name: &str, shared: &str, source: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let text = strip_comments(source);
    let shared_text = strip_comments(shared);
    let defines = constants(&format!("{shared_text}\n{text}"));
    let mut report = |at: usize, msg: String| problems.push(format!("{name}:{}: {msg}", line_of(&text, at)));

    // forbidden words
    for word in ["double", "printf", "std", "assert", "malloc", "erf", "erfc", "erff"] {
        let mut from = 0;
        while let Some(at) = find_word(&text, word, from) {
            let msg = match word {
                "double" => "`double` does not exist in MSL".to_owned(),
                "erf" | "erfc" | "erff" => format!("`{word}` is not part of the Metal standard library (use the o_erf helpers of common_ops.metal)"),
                _ => format!("`{word}` is not available in MSL"),
            };
            // `std` only matters as `std::`; `erf` as a call
            let after = text[at + word.len()..].trim_start();
            let relevant = match word {
                "std" => after.starts_with("::"),
                "erf" | "erfc" | "erff" | "printf" | "assert" | "malloc" => after.starts_with('('),
                _ => true,
            };
            if relevant {
                report(at, msg);
            }
            from = at + word.len();
        }
    }
    if text.contains("long double") {
        report(text.find("long double").unwrap_or(0), "`long double` does not exist in MSL".into());
    }

    // program-scope variables must be in the constant address space
    {
        let b = text.as_bytes();
        let mut depth = 0i32;
        let mut stmt_start = 0usize;
        for (i, &c) in b.iter().enumerate() {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    // the end of a function or struct body ends the statement; the braces of an initializer (`= { ... };`) do not
                    if depth == 0 && !text[stmt_start..i].contains('=') {
                        stmt_start = i + 1;
                    }
                }
                b';' if depth == 0 => {
                    let stmt = text[stmt_start..i].trim();
                    stmt_start = i + 1;
                    let skip = stmt.is_empty()
                        || stmt.starts_with('#')
                        || stmt.starts_with("template")
                        || stmt.starts_with("typedef")
                        || stmt.starts_with("using")
                        || stmt.starts_with("struct")
                        || stmt.starts_with("static_assert")
                        || stmt.contains("decltype")
                        || !stmt.contains('=');
                    if !skip && !stmt.contains('(') && !has_word(stmt, "constant") {
                        report(stmt_start, format!("program-scope variable `{}` is not in the `constant` address space", stmt.split('=').next().unwrap_or("").trim()));
                    }
                }
                b'\n' if depth == 0 => {
                    let line_start = text[..i].rfind('\n').map_or(0, |p| p + 1);
                    if text[line_start..i].trim_start().starts_with('#') {
                        stmt_start = i + 1;
                    }
                }
                _ => {}
            }
        }
    }

    for f in functions(&text) {
        let fname = f.name();
        if f.is_kernel() {
            // attributes
            let mut position_types: Vec<(String, String)> = Vec::new();
            let mut buffers: BTreeMap<u32, String> = BTreeMap::new();
            for p in f.params() {
                let (ty, attrs) = param_type_and_attrs(&p);
                let is_pointer_or_ref = p.contains('*') || p.contains('&');
                let attr_text = attrs.join(" ");
                if is_pointer_or_ref {
                    let spaced = ["device", "constant", "threadgroup", "thread"].iter().any(|s| has_word(&p, s));
                    if !spaced {
                        report(f.header_at, format!("kernel {fname}: pointer/reference parameter `{p}` has no address space"));
                    }
                }
                for a in &attrs {
                    if POSITION_ATTRIBUTES.contains(&a.as_str()) {
                        position_types.push((a.clone(), ty.clone()));
                    }
                    if let Some(n) = a.strip_prefix("buffer(").and_then(|r| r.strip_suffix(')')) {
                        match n.trim().parse::<u32>() {
                            Ok(n) if n < 31 => {
                                if let Some(prev) = buffers.insert(n, p.clone()) {
                                    report(f.header_at, format!("kernel {fname}: buffer index {n} used twice (`{prev}` and `{p}`)"));
                                }
                            }
                            Ok(n) => report(f.header_at, format!("kernel {fname}: buffer index {n} is above 30")),
                            Err(_) => {}
                        }
                    }
                }
                if !is_pointer_or_ref && attr_text.is_empty() && !ty.is_empty() {
                    // a by-value parameter without attribute is not allowed for kernels
                    if !ty.starts_with("threadgroup") {
                        report(f.header_at, format!("kernel {fname}: parameter `{p}` has no attribute"));
                    }
                }
            }
            if let Some((first_attr, first_ty)) = position_types.first() {
                for (a, t) in &position_types {
                    if t != first_ty {
                        report(
                            f.header_at,
                            format!(
                                "kernel {fname}: thread-position attributes of different types (`{first_attr}` is `{first_ty}`, `{a}` is `{t}`); MSL requires one type for all of them"
                            ),
                        );
                        break;
                    }
                }
            }
            // threadgroup variables
            let body = &text[f.body.0..f.body.1];
            let mut total = 0usize;
            let mut known = true;
            let mut depth = 0i32;
            let mut stmt_start = 0usize;
            let bb = body.as_bytes();
            for (i, &c) in bb.iter().enumerate() {
                match c {
                    b'{' => {
                        depth += 1;
                        stmt_start = i + 1;
                    }
                    b'}' => {
                        depth -= 1;
                        stmt_start = i + 1;
                    }
                    b';' => {
                        let stmt = body[stmt_start..i].trim();
                        stmt_start = i + 1;
                        // a pointer variable (`threadgroup float *qs = stage + ...`) is not memory; a `*` inside the brackets is a product
                        let declares_memory = |d: &str| !d.split('[').next().unwrap_or(d).contains('*') && !d.contains('=') && !d.contains('(');
                        if let Some(decl) = stmt
                            .strip_prefix("threadgroup ")
                            .or_else(|| stmt.strip_prefix("static threadgroup "))
                            .filter(|d| declares_memory(d))
                        {
                            if depth != 0 {
                                report(f.body.0 + i, format!("kernel {fname}: `threadgroup {decl}` is declared inside a nested scope (MSL needs the kernel's top level)"));
                            }
                            let (ty, rest) = decl.split_once(char::is_whitespace).unwrap_or((decl, ""));
                            let elems = match (rest.find('['), rest.rfind(']')) {
                                (Some(a), Some(z)) if a < z => {
                                    let mut n = 1i64;
                                    let mut ok = true;
                                    let mut r = &rest[a..=z];
                                    while let Some(s) = r.find('[') {
                                        let e = r[s..].find(']').map(|e| s + e).unwrap_or(r.len() - 1);
                                        match evaluate(&r[s + 1..e], &defines) {
                                            Some(v) => n = n.saturating_mul(v),
                                            None => ok = false,
                                        }
                                        r = &r[e + 1..];
                                    }
                                    ok.then_some(n as usize)
                                }
                                _ => Some(1),
                            };
                            match (type_bytes(ty), elems) {
                                (Some(sz), Some(n)) => total += sz * n,
                                _ => known = false,
                            }
                        }
                    }
                    _ => {}
                }
            }
            if known && total > THREADGROUP_LIMIT {
                report(f.header_at, format!("kernel {fname}: {total} bytes of static threadgroup memory exceed the {THREADGROUP_LIMIT} byte limit"));
            }
        }
    }
    problems
}

/// `.metal` files of the metal directory, sorted.
fn sources() -> Vec<(String, String)> {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("metal");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "metal"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(&p).unwrap()))
        .collect()
}

/// Names of the kernels a source defines (`kernel void NAME(`, `[[kernel]] void NAME(`, `[[host_name("NAME")]]`).
fn kernel_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let raw = strip_comments_keep_strings(source);
    let mut rest = raw.as_str();
    while let Some(i) = rest.find("host_name(\"") {
        rest = &rest[i + "host_name(\"".len()..];
        let end = rest.find('"').unwrap_or(0);
        names.push(rest[..end].to_owned());
    }
    let text = strip_comments(source);
    for f in functions(&text) {
        if f.is_kernel() && !f.header.contains("template") {
            names.push(f.name());
        }
    }
    names
}

/// Like [`strip_comments`] but string literals are kept.
fn strip_comments_keep_strings(text: &str) -> String {
    text.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared() -> String {
        // the shared units the backend prepends: common*.metal and files without kernels
        sources()
            .into_iter()
            .filter(|(n, _)| n.starts_with("common") || n.starts_with('_'))
            .map(|(_, t)| t)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn metal_sources_pass_the_static_checks() {
        let shared = shared();
        let mut problems = Vec::new();
        for (name, text) in sources() {
            problems.extend(check_source(&name, &shared, &text));
        }
        assert!(problems.is_empty(), "{} problem(s) in the MSL sources:\n{}", problems.len(), problems.join("\n"));
    }

    #[test]
    fn kernel_names_are_unique_across_the_sources() {
        let mut seen: HashMap<String, String> = HashMap::new();
        let mut clashes = Vec::new();
        for (file, text) in sources() {
            for name in kernel_names(&text) {
                if let Some(prev) = seen.insert(name.clone(), file.clone()) {
                    // the same kernel in two units is only a problem when both are compiled into one library
                    clashes.push(format!("kernel {name} defined in {prev} and {file}"));
                }
            }
        }
        assert!(clashes.is_empty(), "{}", clashes.join("\n"));
    }

    // ---- the checker itself ---------------------------------------------------------------------------------------------------

    fn bad(src: &str) -> Vec<String> {
        check_source("t.metal", "", src)
    }

    #[test]
    fn mixed_scalar_and_vector_position_attributes_are_found() {
        let p = bad("kernel void k(device float *x [[buffer(0)]], uint gid [[thread_position_in_grid]], uint3 tg [[threadgroup_position_in_grid]]) { x[gid] = float(tg.x); }");
        assert!(p.iter().any(|m| m.contains("different types")), "{p:?}");
        let ok = bad("kernel void k(device float *x [[buffer(0)]], uint3 gid [[thread_position_in_grid]], uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { x[gid.x] = float(tg.x + lane); }");
        assert!(ok.is_empty(), "{ok:?}");
    }

    #[test]
    fn missing_address_space_is_found() {
        let p = bad("kernel void k(float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = 1.0f; }");
        assert!(p.iter().any(|m| m.contains("no address space")), "{p:?}");
    }

    #[test]
    fn duplicate_and_large_buffer_indices_are_found() {
        let p = bad("kernel void k(device float *x [[buffer(1)]], device float *y [[buffer(1)]], device float *z [[buffer(31)]], uint i [[thread_position_in_grid]]) { }");
        assert!(p.iter().any(|m| m.contains("used twice")), "{p:?}");
        assert!(p.iter().any(|m| m.contains("above 30")), "{p:?}");
    }

    #[test]
    fn nested_threadgroup_declarations_and_too_much_memory_are_found() {
        let p = bad("kernel void k(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) { threadgroup float t[4]; t[0] = 1.0f; } }");
        assert!(p.iter().any(|m| m.contains("nested scope")), "{p:?}");
        let p = bad("#define N 4096\nkernel void k(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { threadgroup float a[N * 2]; threadgroup float b[N]; }");
        assert!(p.iter().any(|m| m.contains("exceed")), "{p:?}");
        let ok = bad("#define N 4096\nkernel void k(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { threadgroup float a[N]; threadgroup float b[N / 2]; }");
        assert!(ok.is_empty(), "{ok:?}");
    }

    #[test]
    fn double_erf_and_program_scope_variables_are_found() {
        let p = bad("kernel void k(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { double d = 1.0; x[i] = erf(x[i]); }");
        assert!(p.iter().any(|m| m.contains("double")), "{p:?}");
        assert!(p.iter().any(|m| m.contains("erf")), "{p:?}");
        let p = bad("static const float tab[2] = {1.0f, 2.0f};\nconstant float ok_tab[2] = {1.0f, 2.0f};\n");
        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("tab"), "{p:?}");
    }
}
