//! A text-level structure check of StableHLO programs in the generic MLIR syntax the emitter writes.
//!
//! It is deliberately independent of the emitter's own bookkeeping: it re-reads the finished text and checks
//!
//! - that brackets (`() [] {} <>`) balance,
//! - that every SSA name is defined exactly once and before its first use,
//! - that every operand has the type the operation's signature states for it,
//! - the shape and element-type rules of the operations the emitter uses (elementwise, `reshape`, `convert`, `broadcast_in_dim`,
//!   `dot_general`, `reduce`, `gather`, …) against the result type written in the signature,
//! - that the function's returned values match its declared result types.
//!
//! It is not a StableHLO verifier (the real one lives in the compiler); it catches the mistakes an emitter makes without needing one.

use std::collections::HashMap;

/// What a successful check saw.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    pub functions: usize,
    pub params: usize,
    pub ops: usize,
}

/// A parsed `tensor<…>` type.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Ty {
    dims: Vec<i64>,
    dtype: String,
}

impl Ty {
    fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let inner = text
            .strip_prefix("tensor<")
            .and_then(|t| t.strip_suffix('>'))
            .ok_or_else(|| format!("`{text}` is not a tensor type"))?;
        let mut parts: Vec<&str> = inner.split('x').collect();
        let dtype = parts.pop().unwrap_or_default().to_owned();
        let mut dims = Vec::new();
        for part in parts {
            dims.push(
                part.parse::<i64>()
                    .map_err(|_| format!("bad dimension `{part}` in `{text}`"))?,
            );
        }
        if !matches!(
            dtype.as_str(),
            "f32" | "f16" | "bf16" | "i1" | "i8" | "ui8" | "i16" | "ui16" | "i32" | "ui32" | "i64"
        ) {
            return Err(format!("unknown element type `{dtype}` in `{text}`"));
        }
        Ok(Self { dims, dtype })
    }

    fn elements(&self) -> i64 {
        self.dims.iter().product()
    }

    fn width(&self) -> i64 {
        match self.dtype.as_str() {
            "i1" | "i8" | "ui8" => 1,
            "f16" | "bf16" | "i16" | "ui16" => 2,
            "i64" => 8,
            _ => 4,
        }
    }

    fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
}

/// One parsed operation line.
#[derive(Debug)]
struct OpLine {
    result: Option<String>,
    name: String,
    operands: Vec<String>,
    attrs: String,
    inputs: Vec<Ty>,
    outputs: Vec<Ty>,
}

/// Split `text` at top-level commas (not inside brackets).
fn split_top(text: &str) -> Vec<&str> {
    let (mut depth, mut start) = (0i32, 0usize);
    let mut parts = Vec::new();
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'>' if i == 0 || bytes[i - 1] != b'-' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(text[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts
}

/// Check bracket balance of the whole text (string literals and the `->` arrow are skipped).
fn check_brackets(text: &str) -> Result<(), String> {
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut in_string = false;
    let mut previous = ' ';
    for (line_no, line) in text.lines().enumerate() {
        for c in line.chars() {
            if c == '"' {
                in_string = !in_string;
            } else if !in_string {
                match c {
                    '(' | '[' | '{' | '<' => stack.push((c, line_no + 1)),
                    '>' if previous == '-' => {}
                    ')' | ']' | '}' | '>' => {
                        let want = match c {
                            ')' => '(',
                            ']' => '[',
                            '}' => '{',
                            _ => '<',
                        };
                        match stack.pop() {
                            Some((open, _)) if open == want => {}
                            Some((open, at)) => {
                                return Err(format!(
                                    "line {}: `{c}` closes `{open}` opened on line {at}",
                                    line_no + 1
                                ))
                            }
                            None => {
                                return Err(format!(
                                    "line {}: `{c}` without an opening bracket",
                                    line_no + 1
                                ))
                            }
                        }
                    }
                    _ => {}
                }
            }
            previous = c;
        }
        if in_string {
            return Err(format!("line {}: unterminated string", line_no + 1));
        }
        previous = ' ';
    }
    match stack.pop() {
        Some((open, at)) => Err(format!("`{open}` opened on line {at} is never closed")),
        None => Ok(()),
    }
}

/// The parts of an operation line before its attributes and signature: result name, operation name, operand names and the rest.
/// Result name, operation name, operand names and the text after the operand list.
type Head<'a> = (Option<String>, String, Vec<String>, &'a str);

fn parse_head(head: &str) -> Result<Head<'_>, String> {
    let head = head.trim();
    let (result, rest) = match head.strip_prefix('%') {
        Some(after) => {
            let (name, rest) = after
                .split_once(" = ")
                .ok_or("missing ` = ` after the result name")?;
            (Some(format!("%{name}")), rest)
        }
        None => (None, head),
    };
    let rest = rest
        .strip_prefix('"')
        .ok_or("operation name is not quoted")?;
    let (name, rest) = rest.split_once('"').ok_or("unterminated operation name")?;
    let rest = rest.strip_prefix('(').ok_or("missing operand list")?;
    let mut depth = 1;
    let mut close = None;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close.ok_or("unterminated operand list")?;
    let operands = split_top(&rest[..close])
        .into_iter()
        .map(str::to_owned)
        .collect();
    Ok((result, name.to_owned(), operands, &rest[close + 1..]))
}

/// Parse `%r = "op"(%a, %b) {attrs} : (T, T) -> R` (result and attributes are optional). For a region operation `head` is the opening
/// line without `({` and `tail` the closing line without `})`; otherwise `tail` is empty.
fn parse_op(head: &str, tail: &str) -> Result<OpLine, String> {
    let (result, name, operands, after_operands) = parse_head(head)?;
    // attributes sit between the operand list and the signature; for a region operation they follow the region
    let attr_source = if tail.is_empty() {
        after_operands
    } else {
        tail
    };
    let marker = attr_source
        .rfind(" : (")
        .ok_or("missing signature ` : (…) -> …`")?;
    let attrs = attr_source[..marker].trim().to_owned();
    let signature = &attr_source[marker + 3..];
    let (inputs_text, outputs_text) = signature
        .split_once(" -> ")
        .ok_or("signature without `->`")?;
    let inputs_text = inputs_text.trim();
    let inputs_text = inputs_text
        .strip_prefix('(')
        .and_then(|t| t.strip_suffix(')'))
        .ok_or("signature inputs are not parenthesised")?;
    let inputs = split_top(inputs_text)
        .into_iter()
        .map(Ty::parse)
        .collect::<Result<Vec<_>, _>>()?;
    let outputs_text = outputs_text.trim();
    let outputs = if outputs_text == "()" {
        Vec::new()
    } else {
        let list = outputs_text
            .strip_prefix('(')
            .and_then(|t| t.strip_suffix(')'))
            .unwrap_or(outputs_text);
        split_top(list)
            .into_iter()
            .map(Ty::parse)
            .collect::<Result<Vec<_>, _>>()?
    };
    if inputs.len() != operands.len() {
        return Err(format!(
            "{} operands but {} input types",
            operands.len(),
            inputs.len()
        ));
    }
    Ok(OpLine {
        result,
        name,
        operands,
        attrs,
        inputs,
        outputs,
    })
}

/// The integers of `key = array<i64: …>` or `key = […]` in `attrs`.
fn int_list(attrs: &str, key: &str) -> Option<Vec<i64>> {
    let at = attrs.find(&format!("{key} = "))? + key.len() + 3;
    let rest = &attrs[at..];
    let (open, close) = if rest.starts_with("array<i64") {
        let close = rest.find('>')?;
        (rest[..close].find(':').map_or(close, |c| c + 1), close)
    } else {
        (rest.find('[')? + 1, rest.find(']')?)
    };
    let body = rest[open..close].trim();
    if body.is_empty() {
        return Some(Vec::new());
    }
    body.split(',')
        .map(|v| v.trim().parse::<i64>().ok())
        .collect()
}

fn int_attr(attrs: &str, key: &str) -> Option<i64> {
    let at = attrs.find(&format!("{key} = "))? + key.len() + 3;
    attrs[at..]
        .split(|c: char| !(c.is_ascii_digit() || c == '-'))
        .next()?
        .parse()
        .ok()
}

fn expect(cond: bool, message: impl FnOnce() -> String) -> Result<(), String> {
    if cond {
        Ok(())
    } else {
        Err(message())
    }
}

/// The shape and element-type rules of one operation against its signature.
fn check_rules(op: &OpLine) -> Result<(), String> {
    let name = op.name.strip_prefix("stablehlo.").unwrap_or(&op.name);
    let (ins, outs) = (&op.inputs, &op.outputs);
    let out = outs.first();
    let n = name.to_owned();
    match name {
        "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum" | "and" | "or"
        | "power" => expect(
            ins.len() == 2 && ins[0] == ins[1] && out == Some(&ins[0]),
            || format!("{n}: operands and result must have one type, got {ins:?} -> {outs:?}"),
        ),
        "negate" | "abs" | "exponential" | "tanh" | "sqrt" | "rsqrt" | "logistic" | "not"
        | "floor" | "sine" | "cosine" | "log" => {
            expect(ins.len() == 1 && out == Some(&ins[0]), || {
                format!("{n}: result must have the operand's type, got {ins:?} -> {outs:?}")
            })
        }
        "compare" => expect(
            ins.len() == 2
                && ins[0] == ins[1]
                && out.is_some_and(|o| o.dims == ins[0].dims && o.dtype == "i1"),
            || {
                format!("{n}: needs equal operands and a boolean result of their shape, got {ins:?} -> {outs:?}")
            },
        ),
        "select" => expect(
            ins.len() == 3
                && ins[0].dtype == "i1"
                && (ins[0].is_scalar() || ins[0].dims == ins[1].dims)
                && ins[1] == ins[2]
                && out == Some(&ins[1]),
            || format!("{n}: bad predicate or branch types {ins:?} -> {outs:?}"),
        ),
        "clamp" => expect(
            ins.len() == 3
                && out == Some(&ins[1])
                && ins[0].dtype == ins[1].dtype
                && ins[2].dtype == ins[1].dtype,
            || format!("{n}: bad types {ins:?}"),
        ),
        "convert" => expect(
            ins.len() == 1 && out.is_some_and(|o| o.dims == ins[0].dims),
            || format!("{n}: shape must not change, got {ins:?} -> {outs:?}"),
        ),
        "bitcast_convert" => {
            let o = out.ok_or("bitcast_convert without result")?;
            let (a, b) = (&ins[0], o);
            let ok = if a.width() == b.width() {
                a.dims == b.dims
            } else if a.width() > b.width() {
                let mut dims = a.dims.clone();
                dims.push(a.width() / b.width());
                dims == b.dims
            } else {
                let mut dims = b.dims.clone();
                dims.push(b.width() / a.width());
                dims == a.dims
            };
            expect(ok, || {
                format!("{n}: {a:?} -> {b:?} does not follow the bitcast rule")
            })
        }
        "reshape" => expect(
            ins.len() == 1
                && out
                    .is_some_and(|o| o.elements() == ins[0].elements() && o.dtype == ins[0].dtype),
            || format!("{n}: element count or type changes: {ins:?} -> {outs:?}"),
        ),
        "broadcast_in_dim" => {
            let map = int_list(&op.attrs, "broadcast_dimensions")
                .ok_or("broadcast_in_dim without broadcast_dimensions")?;
            let o = out.ok_or("broadcast_in_dim without result")?;
            expect(
                map.len() == ins[0].dims.len() && o.dtype == ins[0].dtype,
                || format!("{n}: map {map:?} for {ins:?} -> {outs:?}"),
            )?;
            for (i, &m) in map.iter().enumerate() {
                let target = o
                    .dims
                    .get(m as usize)
                    .copied()
                    .ok_or_else(|| format!("{n}: map {map:?} outside {o:?}"))?;
                expect(ins[0].dims[i] == target || ins[0].dims[i] == 1, || {
                    format!("{n}: dimension {i} of {:?} does not fit {o:?}", ins[0])
                })?;
                expect(i == 0 || map[i - 1] < m, || {
                    format!("{n}: map {map:?} is not increasing")
                })?;
            }
            Ok(())
        }
        "transpose" => {
            let perm = int_list(&op.attrs, "permutation").ok_or("transpose without permutation")?;
            let o = out.ok_or("transpose without result")?;
            let want: Vec<i64> = perm
                .iter()
                .map(|&p| ins[0].dims.get(p as usize).copied().unwrap_or(-1))
                .collect();
            expect(want == o.dims && perm.len() == ins[0].dims.len(), || {
                format!(
                    "{n}: permutation {perm:?} of {:?} gives {want:?}, not {:?}",
                    ins[0], o.dims
                )
            })
        }
        "slice" => {
            let start =
                int_list(&op.attrs, "start_indices").ok_or("slice without start_indices")?;
            let limit =
                int_list(&op.attrs, "limit_indices").ok_or("slice without limit_indices")?;
            let o = out.ok_or("slice without result")?;
            let want: Vec<i64> = start.iter().zip(&limit).map(|(s, l)| l - s).collect();
            expect(
                want == o.dims && limit.iter().zip(&ins[0].dims).all(|(l, d)| l <= d),
                || {
                    format!(
                        "{n}: {start:?}..{limit:?} of {:?} gives {want:?}, not {:?}",
                        ins[0], o.dims
                    )
                },
            )
        }
        "concatenate" => {
            let dim =
                int_attr(&op.attrs, "dimension").ok_or("concatenate without dimension")? as usize;
            let o = out.ok_or("concatenate without result")?;
            let total: i64 = ins.iter().map(|t| t.dims[dim]).sum();
            let mut want = ins[0].dims.clone();
            want[dim] = total;
            expect(
                want == o.dims && ins.iter().all(|t| t.dtype == o.dtype),
                || format!("{n}: {ins:?} does not concatenate to {o:?}"),
            )
        }
        "dynamic_slice" => {
            let sizes =
                int_list(&op.attrs, "slice_sizes").ok_or("dynamic_slice without slice_sizes")?;
            let o = out.ok_or("dynamic_slice without result")?;
            expect(
                sizes == o.dims
                    && ins.len() == 1 + ins[0].dims.len()
                    && ins[1..]
                        .iter()
                        .all(|t| t.is_scalar() && t.dtype.starts_with('i')),
                || {
                    format!(
                        "{n}: sizes {sizes:?}, indices {:?}, result {o:?}",
                        &ins[1..]
                    )
                },
            )
        }
        "dynamic_update_slice" => expect(
            ins.len() == 2 + ins[0].dims.len()
                && out == Some(&ins[0])
                && ins[1].dtype == ins[0].dtype
                && ins[1].dims.len() == ins[0].dims.len()
                && ins[2..]
                    .iter()
                    .all(|t| t.is_scalar() && t.dtype.starts_with('i')),
            || format!("{n}: bad operands {ins:?}"),
        ),
        "reduce" => {
            let dims = int_list(&op.attrs, "dimensions").ok_or("reduce without dimensions")?;
            let o = out.ok_or("reduce without result")?;
            let want: Vec<i64> = ins[0]
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !dims.contains(&(*i as i64)))
                .map(|(_, &d)| d)
                .collect();
            expect(
                want == o.dims
                    && ins[1].is_scalar()
                    && ins[1].dtype == ins[0].dtype
                    && o.dtype == ins[0].dtype,
                || {
                    format!(
                        "{n}: reducing {dims:?} of {:?} gives {want:?}, not {o:?}",
                        ins[0]
                    )
                },
            )
        }
        "dot_general" => {
            let lb = int_list(&op.attrs, "lhs_batching_dimensions")
                .ok_or("dot_general without lhs_batching_dimensions")?;
            let rb = int_list(&op.attrs, "rhs_batching_dimensions")
                .ok_or("dot_general without rhs_batching_dimensions")?;
            let lc = int_list(&op.attrs, "lhs_contracting_dimensions")
                .ok_or("dot_general without lhs_contracting_dimensions")?;
            let rc = int_list(&op.attrs, "rhs_contracting_dimensions")
                .ok_or("dot_general without rhs_contracting_dimensions")?;
            let (l, r) = (&ins[0], &ins[1]);
            let mut want: Vec<i64> = lb.iter().map(|&i| l.dims[i as usize]).collect();
            want.extend(
                l.dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !lb.contains(&(*i as i64)) && !lc.contains(&(*i as i64)))
                    .map(|(_, &d)| d),
            );
            want.extend(
                r.dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !rb.contains(&(*i as i64)) && !rc.contains(&(*i as i64)))
                    .map(|(_, &d)| d),
            );
            let batch_ok = lb
                .iter()
                .zip(&rb)
                .all(|(&a, &b)| l.dims[a as usize] == r.dims[b as usize]);
            let contract_ok = lc
                .iter()
                .zip(&rc)
                .all(|(&a, &b)| l.dims[a as usize] == r.dims[b as usize]);
            let o = out.ok_or("dot_general without result")?;
            expect(
                batch_ok && contract_ok && l.dtype == r.dtype && want == o.dims,
                || format!("{n}: {l:?} x {r:?} gives {want:?}, not {o:?}"),
            )
        }
        "gather" => {
            let sizes = int_list(&op.attrs, "slice_sizes").ok_or("gather without slice_sizes")?;
            let o = out.ok_or("gather without result")?;
            expect(
                ins.len() == 2
                    && sizes.len() == ins[0].dims.len()
                    && o.dtype == ins[0].dtype
                    && o.dims.last() == sizes.last()
                    && ins[1].dims.len() == 2,
                || format!("{n}: {ins:?} with sizes {sizes:?} gives {o:?}"),
            )
        }
        "scatter" => expect(
            ins.len() == 3
                && out == Some(&ins[0])
                && ins[2].dtype == ins[0].dtype
                && ins[2].dims.len() == ins[0].dims.len(),
            || format!("{n}: bad operands {ins:?} -> {outs:?}"),
        ),
        "constant" | "iota" => expect(ins.is_empty() && outs.len() == 1, || {
            format!("{n}: takes no operands and returns one value")
        }),
        "return" => Ok(()),
        other => Err(format!(
            "operation `{other}` is not one the emitter should produce"
        )),
    }
}

/// Check a whole module (see the module comment).
pub fn check(text: &str) -> Result<Report, String> {
    check_brackets(text)?;
    let mut report = Report::default();
    let mut defined: HashMap<String, Ty> = HashMap::new();
    let mut declared_results: Vec<Ty> = Vec::new();
    // region operations: the opening line waits for the closing one
    let mut pending: Vec<(String, usize)> = Vec::new();
    let mut in_func = false;
    let mut saw_module = false;
    for (index, raw) in text.lines().enumerate() {
        let line_no = index + 1;
        let line = raw.trim();
        let fail = |msg: String| format!("line {line_no}: {msg}");
        if line.is_empty() {
            continue;
        }
        if line.starts_with("module @") {
            saw_module = true;
            continue;
        }
        if line.starts_with("func.func ") {
            if in_func {
                return Err(fail("nested function".into()));
            }
            in_func = true;
            report.functions += 1;
            defined.clear();
            let open = line
                .find('(')
                .ok_or_else(|| fail("function without parameter list".into()))?;
            let arrow = line
                .rfind(") -> (")
                .ok_or_else(|| fail("function without result list".into()))?;
            for param in split_top(&line[open + 1..arrow]) {
                let (name, rest) = param
                    .split_once(": ")
                    .ok_or_else(|| fail(format!("bad parameter `{param}`")))?;
                let ty_text = rest.split(" {").next().unwrap_or(rest);
                let ty = Ty::parse(ty_text).map_err(fail)?;
                if defined.insert(name.to_owned(), ty).is_some() {
                    return Err(fail(format!("parameter {name} defined twice")));
                }
                report.params += 1;
            }
            let results_text = line[arrow + 6..]
                .trim_end_matches(" {")
                .trim_end_matches(')');
            declared_results = split_top(results_text)
                .into_iter()
                .map(Ty::parse)
                .collect::<Result<Vec<_>, _>>()
                .map_err(fail)?;
            continue;
        }
        if line == "}" {
            in_func = false;
            continue;
        }
        if !in_func {
            return Err(fail(format!("`{line}` outside a function")));
        }
        if let Some(rest) = line.strip_prefix("func.return") {
            let rest = rest.trim();
            if rest.is_empty() {
                expect(declared_results.is_empty(), || {
                    fail("returns nothing but the function declares results".into())
                })?;
                continue;
            }
            let (names, types) = rest
                .split_once(" : ")
                .ok_or_else(|| fail("return without types".into()))?;
            let names = split_top(names);
            let types = split_top(types)
                .into_iter()
                .map(Ty::parse)
                .collect::<Result<Vec<_>, _>>()
                .map_err(fail)?;
            expect(
                names.len() == types.len() && types == declared_results,
                || {
                    fail(format!(
                        "returns {types:?}, the function declares {declared_results:?}"
                    ))
                },
            )?;
            for (name, ty) in names.iter().zip(&types) {
                match defined.get(*name) {
                    Some(have) if have == ty => {}
                    Some(have) => {
                        return Err(fail(format!("{name} is {have:?}, returned as {ty:?}")))
                    }
                    None => return Err(fail(format!("returned value {name} is not defined"))),
                }
            }
            continue;
        }
        if line.starts_with("^bb") {
            let open = line
                .find('(')
                .ok_or_else(|| fail("block without arguments".into()))?;
            let close = line
                .rfind("):")
                .ok_or_else(|| fail("block header does not end with `):`".into()))?;
            for arg in split_top(&line[open + 1..close]) {
                let (name, ty_text) = arg
                    .split_once(": ")
                    .ok_or_else(|| fail(format!("bad block argument `{arg}`")))?;
                let ty = Ty::parse(ty_text).map_err(fail)?;
                if defined.insert(name.to_owned(), ty).is_some() {
                    return Err(fail(format!("block argument {name} defined twice")));
                }
            }
            continue;
        }
        if line.ends_with("({") {
            let head = line.trim_end_matches("({").trim().to_owned();
            // the operands of a region operation are used at its opening line
            let (_, _, operands, _) = parse_head(&head).map_err(fail)?;
            for operand in &operands {
                expect(defined.contains_key(operand), || {
                    fail(format!("{operand} is used before it is defined"))
                })?;
            }
            pending.push((head, line_no));
            continue;
        }
        let parsed = if let Some(tail) = line.strip_prefix("})") {
            let (head, _) = pending
                .pop()
                .ok_or_else(|| fail("`})` without an opening region".into()))?;
            parse_op(&head, tail.trim()).map_err(fail)?
        } else {
            parse_op(line, "").map_err(fail)?
        };
        report.ops += 1;
        for (operand, ty) in parsed.operands.iter().zip(&parsed.inputs) {
            match defined.get(operand) {
                Some(have) if have == ty => {}
                Some(have) => {
                    return Err(fail(format!(
                        "{operand} has type {have:?} but `{}` expects {ty:?}",
                        parsed.name
                    )))
                }
                None => return Err(fail(format!("{operand} is used before it is defined"))),
            }
        }
        check_rules(&parsed).map_err(fail)?;
        match (&parsed.result, parsed.outputs.first()) {
            (Some(name), Some(ty)) => {
                if defined.insert(name.clone(), ty.clone()).is_some() {
                    return Err(fail(format!("{name} is defined twice")));
                }
            }
            (None, None) => {}
            (Some(name), None) => {
                return Err(fail(format!(
                    "{name} is assigned the result of an operation that returns nothing"
                )))
            }
            (None, Some(_)) => {
                return Err(fail(format!(
                    "`{}` returns a value that is not named",
                    parsed.name
                )))
            }
        }
    }
    expect(saw_module, || "no `module @…` header".to_owned())?;
    expect(!in_func && pending.is_empty(), || {
        "the text ends inside a function or region".to_owned()
    })?;
    expect(report.functions > 0, || "no function".to_owned())?;
    Ok(report)
}
