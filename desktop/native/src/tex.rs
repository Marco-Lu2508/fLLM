//! A small, safe LaTeX subset for answers — the `texToHtml` of the Studio's `markdown.js` without
//! its library: fractions, powers, subscripts, greek letters, roots, sums, integrals and common
//! symbols. Everything is escaped; anything not understood makes the window show the source in a
//! chip (the `mathSpan` of the Studio).

/// The longest source that is still typeset (`s.length > 2000` in the Studio).
pub const MAX_CHARS: usize = 2000;

/// How a piece of math is drawn: the classes of the Studio's CSS become styles the window can
/// paint (`i`, `b`, `.tx-rm`, `.tx-fn`, `.tx-op`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Normal,
    Italic,
    Bold,
    Roman,
    Func,
    Op,
}

/// One piece of the typeset math: the text with its style, or a shape of its own (the spans of the
/// Studio's HTML become nodes the window draws).
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Text(String, Style),
    /// A base with an optional power and subscript (the `<sup>`/`<sub>` of `scripted`).
    Script { base: Box<Node>, sup: Option<Box<Node>>, sub: Option<Box<Node>>, limits: bool },
    Frac(Box<Node>, Box<Node>),
    Sqrt { idx: Option<Box<Node>>, body: Box<Node> },
    /// A big operator with what stands above and below it (the `.tx-lim` of a display answer).
    Limits { top: Box<Node>, big: Box<Node>, bot: Box<Node> },
    Over(Box<Node>),
    Under(Box<Node>),
    /// Bold around pieces that keep their own style (`\mathbf{x}` is `<b><i>x</i></b>`).
    Bold(Box<Node>),
    /// An accent over the next group, which keeps its own style (`\hat{x}` is `<i>x</i>̂`).
    Accent(Box<Node>, String),
    /// The line break of `\`.
    Break,
    /// Several pieces in order (the Studio concatenates their html).
    Seq(Vec<Node>),
}

/// What one atom or command produced: the node, and whether a power over it stands as limits (the
/// `limits` flag of `TEX_BIG` and `TEX_FN` in the Studio).
struct Parsed {
    node: Node,
    limits: bool,
}

impl Parsed {
    fn plain(node: Node) -> Parsed {
        Parsed { node, limits: false }
    }
}

/// The HTML of the Studio for this math — the bridge the fixtures are compared through (the
/// window draws the same nodes as text and shapes). `display` is what makes a power over a big
/// operator stand above and below it (`base.limits && display` of the Studio).
pub fn to_html(node: &Node, display: bool) -> String {
    match node {
        Node::Seq(parts) => parts.iter().map(|p| to_html(p, display)).collect(),
        Node::Text(t, style) => {
            let esc = escape(t);
            match style {
                Style::Normal => esc,
                Style::Italic => format!("<i>{esc}</i>"),
                Style::Bold => format!("<b>{esc}</b>"),
                Style::Roman => format!("<span class=\"tx-rm\">{esc}</span>"),
                Style::Func => format!("<span class=\"tx-fn\">{esc}</span>"),
                Style::Op => format!("<span class=\"tx-op\">{esc}</span>"),
            }
        }
        Node::Script { base, sup, sub, limits } => {
            let base_html = to_html(base, display);
            if *limits && display {
                // a big operator in a display answer puts its power above and its subscript below,
                // each of them there or empty (`base.limits && display` of the Studio)
                let empty = Node::Text(String::new(), Style::Normal);
                let top = sup.as_deref().unwrap_or(&empty);
                let bot = sub.as_deref().unwrap_or(&empty);
                return format!(
                    "<span class=\"tx-lim\"><span class=\"tx-top\">{}</span><span class=\"tx-big\">{base_html}</span><span class=\"tx-bot\">{}</span></span>",
                    to_html(top, display),
                    to_html(bot, display)
                );
            }
            match (sup, sub) {
                (Some(sup), Some(sub)) => format!(
                    "{base_html}<span class=\"tx-ss\"><sup>{}</sup><sub>{}</sub></span>",
                    to_html(sup, display),
                    to_html(sub, display)
                ),
                (Some(sup), None) => format!("{base_html}<sup>{}</sup>", to_html(sup, display)),
                (None, Some(sub)) => format!("{base_html}<sub>{}</sub>", to_html(sub, display)),
                (None, None) => base_html,
            }
        }
        Node::Frac(num, den) => format!(
            "<span class=\"tx-frac\"><span class=\"tx-num\">{}</span><span class=\"tx-den\">{}</span></span>",
            to_html(num, display),
            to_html(den, display)
        ),
        Node::Sqrt { idx, body } => {
            let root = match idx {
                Some(i) => format!("<sup class=\"tx-rootn\">{}</sup>", to_html(i, display)),
                None => String::new(),
            };
            format!(
                "{root}<span class=\"tx-sqrt\"><span class=\"tx-rad\">√</span><span class=\"tx-rcnt\">{}</span></span>",
                to_html(body, display)
            )
        }
        Node::Limits { top, big, bot } => format!(
            "<span class=\"tx-lim\"><span class=\"tx-top\">{}</span><span class=\"tx-big\">{}</span><span class=\"tx-bot\">{}</span></span>",
            to_html(top, display),
            to_html(big, display),
            to_html(bot, display)
        ),
        Node::Bold(inner) => format!("<b>{}</b>", to_html(inner, display)),
        Node::Accent(inner, acc) => format!("{}{acc}", to_html(inner, display)),
        Node::Over(inner) => format!("<span class=\"tx-over\">{}</span>", to_html(inner, display)),
        Node::Under(inner) => format!("<span class=\"tx-under\">{}</span>", to_html(inner, display)),
        Node::Break => "<br>".to_string(),
    }
}

/// The html of the Studio's `mathSpan`: the typeset source in its styled span, the plain source
/// in a chip when it cannot be typeset.
pub fn span_html(src: &str, display: bool) -> String {
    match typeset(src, display) {
        Some(node) => {
            let class = if display { "tex tex-display" } else { "tex" };
            format!("<span class=\"{class}\">{}</span>", to_html(&node, display))
        }
        None => format!(
            "<span class=\"math-inline\" title=\"Shown as source: not a supported LaTeX expression\">{}</span>",
            escape(src)
        ),
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// The greek letters of the Studio (`GREEK`).
fn greek(name: &str) -> Option<&'static str> {
    Some(match name {
        "alpha" => "α",
        "beta" => "β",
        "gamma" => "γ",
        "delta" => "δ",
        "epsilon" | "varepsilon" => "ε",
        "zeta" => "ζ",
        "eta" => "η",
        "theta" => "θ",
        "vartheta" => "ϑ",
        "iota" => "ι",
        "kappa" => "κ",
        "lambda" => "λ",
        "mu" => "μ",
        "nu" => "ν",
        "xi" => "ξ",
        "pi" => "π",
        "varpi" => "ϖ",
        "rho" => "ρ",
        "varrho" => "ϱ",
        "sigma" => "σ",
        "varsigma" => "ς",
        "tau" => "τ",
        "upsilon" => "υ",
        "phi" => "φ",
        "varphi" => "φ",
        "chi" => "χ",
        "psi" => "ψ",
        "omega" => "ω",
        _ => return None,
    })
}

/// The capital greek letters (`GREEK_UP`).
fn greek_up(name: &str) -> Option<&'static str> {
    Some(match name {
        "Gamma" => "Γ",
        "Delta" => "Δ",
        "Theta" => "Θ",
        "Lambda" => "Λ",
        "Xi" => "Ξ",
        "Pi" => "Π",
        "Sigma" => "Σ",
        "Upsilon" => "Υ",
        "Phi" => "Φ",
        "Psi" => "Ψ",
        "Omega" => "Ω",
        _ => return None,
    })
}

/// The symbols with a name of their own (`TEX_SYM`).
fn tex_sym(name: &str) -> Option<&'static str> {
    Some(match name {
        "cdot" => "·",
        "times" => "×",
        "div" => "÷",
        "pm" => "±",
        "mp" => "∓",
        "leq" | "le" => "≤",
        "geq" | "ge" => "≥",
        "neq" | "ne" => "≠",
        "approx" => "≈",
        "equiv" => "≡",
        "sim" => "∼",
        "propto" => "∝",
        "infty" => "∞",
        "partial" => "∂",
        "nabla" => "∇",
        "to" | "rightarrow" => "→",
        "leftarrow" | "gets" => "←",
        "Rightarrow" => "⇒",
        "Leftarrow" => "⇐",
        "Leftrightarrow" | "leftrightarrow" => "⇔",
        "mapsto" => "↦",
        "implies" => "⟹",
        "iff" => "⟺",
        "in" => "∈",
        "notin" => "∉",
        "subset" => "⊂",
        "subseteq" => "⊆",
        "supset" => "⊃",
        "cup" => "∪",
        "cap" => "∩",
        "forall" => "∀",
        "exists" => "∃",
        "ldots" | "dots" => "…",
        "cdots" => "⋯",
        "circ" => "∘",
        "ell" => "ℓ",
        "angle" => "∠",
        "neg" => "¬",
        "land" | "wedge" => "∧",
        "lor" | "vee" => "∨",
        "emptyset" | "varnothing" => "∅",
        "mid" => "|",
        "ast" => "∗",
        "star" => "⋆",
        "oplus" => "⊕",
        "otimes" => "⊗",
        "langle" => "⟨",
        "rangle" => "⟩",
        "lbrace" | "{" => "{",
        "rbrace" | "}" => "}",
        "lVert" | "rVert" => "‖",
        "lvert" | "rvert" => "|",
        "prime" => "′",
        "degree" => "°",
        "," => "\u{2009}",
        ";" => "\u{2004}",
        ":" => "\u{2005}",
        "!" => "",
        "quad" => "\u{2003}",
        "qquad" => "\u{2003}\u{2003}",
        " " => " ",
        "%" => "%",
        "&" => "&",
        "#" => "#",
        "_" => "_",
        "$" => "$",
        "|" => "‖",
        "(" => "(",
        ")" => ")",
        "[" => "[",
        "]" => "]",
        "/" => "/",
        "<" => "<",
        ">" => ">",
        "." => "",
        _ => return None,
    })
}

/// The big operators (`TEX_BIG`).
fn tex_big(name: &str) -> Option<&'static str> {
    Some(match name {
        "sum" => "∑",
        "prod" => "∏",
        "coprod" => "∐",
        "bigcup" => "⋃",
        "bigcap" => "⋂",
        "int" => "∫",
        "iint" => "∬",
        "oint" => "∮",
        _ => return None,
    })
}

/// The functions that stand without an operator (`TEX_FN`).
fn tex_fn(name: &str) -> bool {
    matches!(
        name,
        "sin" | "cos"
            | "tan"
            | "cot"
            | "sec"
            | "csc"
            | "sinh"
            | "cosh"
            | "tanh"
            | "arcsin"
            | "arccos"
            | "arctan"
            | "log"
            | "ln"
            | "lg"
            | "exp"
            | "lim"
            | "max"
            | "min"
            | "sup"
            | "inf"
            | "det"
            | "gcd"
            | "arg"
            | "dim"
            | "ker"
            | "deg"
            | "mod"
            | "Pr"
    )
}

/// The accents that are put over the next group (`TEX_ACCENT`).
fn tex_accent(name: &str) -> Option<&'static str> {
    Some(match name {
        "hat" | "widehat" => "\u{0302}",
        "bar" => "\u{0304}",
        "vec" => "\u{20D7}",
        "dot" => "\u{0307}",
        "ddot" => "\u{0308}",
        "tilde" | "widetilde" => "\u{0303}",
        _ => return None,
    })
}

/// The blackboard letters (`BB`).
fn bb(ch: char) -> char {
    match ch {
        'R' => 'ℝ',
        'N' => 'ℕ',
        'Z' => 'ℤ',
        'Q' => 'ℚ',
        'C' => 'ℂ',
        'P' => 'ℙ',
        'H' => 'ℍ',
        other => other,
    }
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    depth: usize,
}

type PResult<T> = Result<T, ()>;

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_whitespace() {
            self.i += 1;
        }
    }

    /// The raw text of a `{...}` group (`rawGroup`).
    fn raw_group(&mut self) -> PResult<String> {
        self.skip_ws();
        if self.s.get(self.i) != Some(&'{') {
            return Err(());
        }
        let mut level = 0usize;
        self.i += 1;
        let start = self.i;
        while self.i < self.s.len() {
            let c = self.s[self.i];
            if c == '\\' {
                self.i += 2;
                continue;
            }
            if c == '{' {
                level += 1;
            } else if c == '}' {
                if level == 0 {
                    let out: String = self.s[start..self.i].iter().collect();
                    self.i += 1;
                    return Ok(out);
                }
                level -= 1;
            }
            self.i += 1;
        }
        Err(())
    }

    /// Everything up to `end` (`seq`): the pieces in order.
    fn seq(&mut self, end: Option<char>) -> PResult<Vec<Node>> {
        self.depth += 1;
        if self.depth > 30 {
            return Err(());
        }
        let mut parts = Vec::new();
        loop {
            self.skip_ws();
            if self.i >= self.s.len() {
                if end.is_some() {
                    return Err(());
                }
                break;
            }
            if let Some(e) = end {
                if self.s[self.i] == e {
                    self.i += 1;
                    break;
                }
            }
            parts.push(self.scripted()?);
        }
        self.depth -= 1;
        Ok(parts)
    }

    /// One argument: a group, or a single atom (`arg`).
    fn arg(&mut self) -> PResult<Node> {
        self.skip_ws();
        if self.i >= self.s.len() {
            return Err(());
        }
        if self.s[self.i] == '{' {
            self.i += 1;
            let parts = self.seq(Some('}'))?;
            return Ok(one(parts));
        }
        Ok(self.atom()?.node)
    }

    /// An atom with the powers and subscripts that follow it (`scripted`).
    fn scripted(&mut self) -> PResult<Node> {
        let base = if self.i < self.s.len() && matches!(self.s[self.i], '^' | '_') {
            Parsed { node: Node::Text(String::new(), Style::Normal), limits: false }
        } else {
            self.atom()?
        };
        let mut sup: Option<Box<Node>> = None;
        let mut sub: Option<Box<Node>> = None;
        loop {
            self.skip_ws();
            if self.s.get(self.i) == Some(&'^') && sup.is_none() {
                self.i += 1;
                sup = Some(Box::new(self.arg()?));
            } else if self.s.get(self.i) == Some(&'_') && sub.is_none() {
                self.i += 1;
                sub = Some(Box::new(self.arg()?));
            } else {
                break;
            }
        }
        let empty = |n: &Option<Box<Node>>| matches!(n.as_deref(), Some(Node::Text(t, _)) if t.is_empty());
        if empty(&sup) {
            sup = None;
        }
        if empty(&sub) {
            sub = None;
        }
        if sup.is_none() && sub.is_none() {
            return Ok(base.node);
        }
        Ok(Node::Script { base: Box::new(base.node), sup, sub, limits: base.limits })
    }

    /// One atom of the source (`atom`).
    fn atom(&mut self) -> PResult<Parsed> {
        let Some(&c) = self.s.get(self.i) else { return Err(()) };
        self.i += 1;
        match c {
            '{' => {
                let parts = self.seq(Some('}'))?;
                Ok(Parsed::plain(one(parts)))
            }
            '\\' => self.command(),
            '}' | '^' | '_' => Err(()),
            c if c.is_ascii_alphabetic() => Ok(Parsed::plain(Node::Text(c.to_string(), Style::Italic))),
            '-' => Ok(Parsed::plain(Node::Text("−".into(), Style::Normal))),
            '*' => Ok(Parsed::plain(Node::Text("∗".into(), Style::Normal))),
            '\'' => Ok(Parsed::plain(Node::Text("′".into(), Style::Normal))),
            '~' | '&' => Ok(Parsed::plain(Node::Text(" ".into(), Style::Normal))),
            c => Ok(Parsed::plain(Node::Text(c.to_string(), Style::Normal))),
        }
    }

    /// A delimiter of `\left` and friends (`delimiter`).
    fn delimiter(&mut self) -> PResult<Node> {
        self.skip_ws();
        if self.s.get(self.i) == Some(&'\\') {
            self.i += 1;
            let sym = self.command_name()?;
            let sym = tex_sym(&sym).ok_or(())?;
            return Ok(Node::Text(sym.to_string(), Style::Normal));
        }
        let c = *self.s.get(self.i).ok_or(())?;
        self.i += 1;
        Ok(Node::Text(if c == '.' { String::new() } else { c.to_string() }, Style::Normal))
    }

    /// The name of a command: its letters, or the one character after the backslash.
    fn command_name(&mut self) -> PResult<String> {
        let rest: String = self.s[self.i..].iter().collect();
        let letters: String = rest.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        if !letters.is_empty() {
            self.i += letters.chars().count();
            return Ok(letters);
        }
        let one = rest.chars().next().ok_or(())?;
        self.i += 1;
        Ok(one.to_string())
    }

    /// One command (`command`).
    fn command(&mut self) -> PResult<Parsed> {
        let name = self.command_name()?;
        if let Some(g) = greek(&name) {
            return Ok(Parsed::plain(Node::Text(g.into(), Style::Italic)));
        }
        if let Some(g) = greek_up(&name) {
            return Ok(Parsed::plain(Node::Text(g.into(), Style::Normal)));
        }
        if name == "\\" {
            return Ok(Parsed::plain(Node::Break));
        }
        if let Some(big) = tex_big(&name) {
            return Ok(Parsed {
                node: Node::Text(big.into(), Style::Op),
                limits: name != "int" && name != "iint" && name != "oint",
            });
        }
        if tex_fn(&name) {
            return Ok(Parsed {
                node: Node::Text(name.clone(), Style::Func),
                limits: matches!(name.as_str(), "lim" | "max" | "min"),
            });
        }
        if let Some(sym) = tex_sym(&name) {
            return Ok(Parsed::plain(Node::Text(sym.into(), Style::Normal)));
        }
        match name.as_str() {
            "frac" | "dfrac" | "tfrac" => {
                let a = self.arg()?;
                let b = self.arg()?;
                Ok(Parsed::plain(Node::Frac(Box::new(a), Box::new(b))))
            }
            "sqrt" => {
                let idx = if self.s.get(self.i) == Some(&'[') {
                    self.i += 1;
                    let parts = self.seq(Some(']'))?;
                    Some(Box::new(one(parts)))
                } else {
                    None
                };
                let body = self.arg()?;
                Ok(Parsed::plain(Node::Sqrt { idx, body: Box::new(body) }))
            }
            "text" | "mathrm" | "textrm" | "mbox" | "operatorname" => {
                let raw = self.raw_group()?;
                Ok(Parsed::plain(Node::Text(raw, Style::Roman)))
            }
            "textbf" => {
                let raw = self.raw_group()?;
                Ok(Parsed::plain(Node::Text(raw, Style::Bold)))
            }
            "mathbf" | "boldsymbol" | "bm" => {
                let a = self.arg()?;
                Ok(Parsed::plain(Node::Bold(Box::new(a))))
            }
            "mathbb" => {
                let raw = self.raw_group()?;
                let out: String = raw.chars().map(bb).collect();
                Ok(Parsed::plain(Node::Text(out, Style::Normal)))
            }
            "mathcal" | "mathit" | "mathsf" | "mathtt" => {
                let a = self.arg()?;
                Ok(Parsed::plain(a))
            }
            "displaystyle" | "textstyle" | "limits" | "nolimits" => Ok(Parsed::plain(Node::Text(String::new(), Style::Normal))),
            "left" | "right" | "big" | "Big" | "bigg" | "Bigg" => self.delimiter().map(Parsed::plain),
            "overline" => {
                let a = self.arg()?;
                Ok(Parsed::plain(Node::Over(Box::new(a))))
            }
            "underline" => {
                let a = self.arg()?;
                Ok(Parsed::plain(Node::Under(Box::new(a))))
            }
            other => {
                if let Some(acc) = tex_accent(other) {
                    let a = self.arg()?;
                    return Ok(Parsed::plain(Node::Accent(Box::new(a), acc.to_string())));
                }
                Err(())
            }
        }
    }
}

/// One piece, or the sequence of them (the Studio uses the concatenated html of `seq`).
fn one(parts: Vec<Node>) -> Node {
    match parts.len() {
        0 => Node::Text(String::new(), Style::Normal),
        1 => parts.into_iter().next().unwrap(),
        _ => Node::Seq(parts),
    }
}

/// One place where an answer writes math inline (the pattern of the Studio's `inline`).
#[derive(Debug, Clone, PartialEq)]
pub struct MathHit {
    pub start: usize,
    pub end: usize,
    pub display: bool,
    pub body: String,
}

/// The text of a typeset piece: the tags are not drawn, the entities are the signs they stand
/// for (`<span class="tx-rm">\u{3c}</span>` is the less-than sign).
pub fn html_plain(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut chars = html.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' {
            for c in chars.by_ref() {
                if c == '>' {
                    break;
                }
            }
            continue;
        }
        if c == '&' {
            let mut entity = String::new();
            for c in chars.by_ref() {
                if c == ';' {
                    break;
                }
                entity.push(c);
                if entity.len() > 8 {
                    break;
                }
            }
            match entity.as_str() {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "quot" => out.push('"'),
                "apos" => out.push('\''),
                _ if entity.starts_with('#') => {
                    let digits = &entity[1..];
                    let parsed = digits
                        .strip_prefix('x')
                        .or_else(|| digits.strip_prefix('X'))
                        .map(|n| u32::from_str_radix(n, 16).ok())
                        .unwrap_or_else(|| digits.parse::<u32>().ok());
                    if let Some(ch) = parsed.and_then(char::from_u32) {
                        out.push(ch);
                    }
                }
                _ => {
                    out.push('&');
                    out.push_str(&entity);
                    out.push(';');
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// The math of an answer's text: `$$x$$`, `\[x\]`, `\(x\)` and `$x$`, with the lookarounds of
/// the Studio's pattern as character checks (`(?<![\w$\])\$(?!\s)` and the closing one — a
/// Rust regex cannot express them).
pub fn math_hits(text: &str) -> Vec<MathHit> {
    let chars: Vec<char> = text.chars().collect();
    let mut hits = Vec::new();
    let mut i = 0usize;
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    while i < chars.len() {
        let display = if chars[i] == '$' && chars.get(i + 1) == Some(&'$') {
            true
        } else if chars[i] == '$' {
            false
        } else if chars[i] == '\\' && matches!(chars.get(i + 1), Some('[') | Some('(')) {
            chars[i + 1] == '['
        } else {
            i += 1;
            continue;
        };
        // the body: no `$` and no line break inside, the closing sign after it
        // `\[..\]`, `\(..\)` and `$$..$$` are two signs on both ends, `$..$` is one
        let (open, close): (usize, usize) = if chars[i] == '\\' || display { (2, 2) } else { (1, 1) };
        let body_start = i + open;
        let mut j = body_start;
        let mut closed = None;
        while j < chars.len() {
            if chars[j] == '\n' {
                break;
            }
            if chars[j] == '$' && open == 1 {
                closed = Some(j);
                break;
            }
            if chars[j] == '$' && open == 2 && chars.get(j + 1) == Some(&'$') {
                closed = Some(j);
                break;
            }
            if chars[j] == '\\' && open == 2 && matches!(chars.get(j + 1), Some(']') | Some(')')) {
                closed = Some(j);
                break;
            }
            j += 1;
        }
        let Some(close_at) = closed else {
            i += body_start.max(i + 1);
            continue;
        };
        let body: String = chars[body_start..close_at].iter().collect();
        let end = close_at + close;
        // the lookarounds of the pattern: a `$` that stands on its own
        if open == 1 {
            let before = if i == 0 { None } else { Some(chars[i - 1]) };
            let after = chars.get(end).copied();
            let body_first = body.chars().next();
            let body_last = body.chars().last();
            let ok_before = before.is_none_or(|c| !is_word(c) && c != '$' && c != '\\');
            let ok_after_first = !body.chars().next().is_some_and(|c| c.is_whitespace());
            let ok_after_last = body_last.is_some_and(|c| !c.is_whitespace() && c != '$' && c != '\\');
            let ok_end = after.is_none_or(|c| !is_word(c) && c != '$');
            if !ok_before || !ok_after_first || !ok_after_last || !ok_end || body_first.is_none() {
                i += 1;
                continue;
            }
            // `[^\s$\]$` also forbids an empty body
            if body.is_empty() {
                i += 1;
                continue;
            }
        }
        if body.is_empty() {
            i += 1;
            continue;
        }
        hits.push(MathHit { start: i, end, display: open == 2 && display, body });
        i = end;
    }
    hits
}

/// The typeset math of one source, or `None` when something is not understood — the window then
/// shows the source in a chip (the `texToHtml` and `mathSpan` of the Studio).
pub fn typeset(src: &str, _display: bool) -> Option<Node> {
    if src.trim().is_empty() || src.chars().count() > MAX_CHARS {
        return None;
    }
    let chars: Vec<char> = src.chars().collect();
    let mut p = Parser { s: &chars, i: 0, depth: 0 };
    let parts = p.seq(None).ok()?;
    if parts.is_empty() {
        return None;
    }
    let node = one(parts);
    if to_html(&node, false).is_empty() {
        None
    } else {
        Some(node)
    }
}
