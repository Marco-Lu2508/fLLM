//! Pure handling of the embedded Metal sources: which files form the shared prefix, how a compile unit is
//! assembled, and which kernel names a unit defines (found by a small scan, so a function can be looked up in the
//! right library without compiling everything).
//!
//! Layout rule (documented in `API.md`): a source file whose name starts with `common` or `_`, or in which no
//! kernel is found, is shared code (types, helpers, dequantisation functions); it is prepended to every other file,
//! and every other file becomes one compile unit (one `MTLLibrary`, compiled on first use). Kernels must therefore
//! not depend on helper code in a file that has kernels of its own and is not theirs.

/// One library worth of source text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    /// Name of the source file the unit is built around (`"all"` for the single-library mode).
    pub name: String,
    /// Complete text handed to `newLibraryWithSource`.
    pub text: String,
    /// Kernel names found in the unit's own file (may miss macro-generated kernels).
    pub kernels: Vec<String>,
}

/// Files that are prepended to every unit.
pub fn is_shared(file_name: &str) -> bool {
    file_name.starts_with("common") || file_name.starts_with('_')
}

/// `#define NAME VALUE` lines for the device-dependent preprocessor symbols (empty value allowed).
pub fn defines_header(defines: &[(String, String)]) -> String {
    let mut header = String::new();
    for (name, value) in defines {
        header.push_str("#define ");
        header.push_str(name);
        if !value.is_empty() {
            header.push(' ');
            header.push_str(value);
        }
        header.push('\n');
    }
    header
}

fn push_file(text: &mut String, name: &str, source: &str) {
    text.push_str("#line 1 \"");
    text.push_str(name);
    text.push_str("\"\n");
    text.push_str(source);
    if !source.ends_with('\n') {
        text.push('\n');
    }
}

/// Assemble the compile units from the embedded `(file name, source)` table. With `single`, all files go into one
/// unit named `all` (shared files first, then files without kernels, then the rest); otherwise every file with kernels
/// is a unit of its own with the shared prefix in front. Shared are the files named `common*` / `_*` and, because
/// they cannot be run on their own, the files in which no kernel is found (helper-only files).
pub fn build_units(
    sources: &[(&str, &str)],
    defines: &[(String, String)],
    single: bool,
) -> Vec<Unit> {
    let header = defines_header(defines);
    let scanned: Vec<(&str, &str, Vec<String>)> = sources
        .iter()
        .map(|(name, source)| (*name, *source, scan_kernel_names(source)))
        .collect();
    let shared =
        |(name, _, kernels): &(&str, &str, Vec<String>)| is_shared(name) || kernels.is_empty();
    if single {
        let mut text = header;
        let mut kernels = Vec::new();
        let ordered = scanned
            .iter()
            .filter(|f| is_shared(f.0))
            .chain(scanned.iter().filter(|f| !is_shared(f.0) && f.2.is_empty()))
            .chain(
                scanned
                    .iter()
                    .filter(|f| !is_shared(f.0) && !f.2.is_empty()),
            );
        for (name, source, found) in ordered {
            push_file(&mut text, name, source);
            kernels.extend(found.iter().cloned());
        }
        dedup_in_order(&mut kernels);
        return vec![Unit {
            name: "all".to_owned(),
            text,
            kernels,
        }];
    }
    let mut prefix = String::new();
    for (name, source, _) in scanned.iter().filter(|f| shared(f)) {
        push_file(&mut prefix, name, source);
    }
    scanned
        .iter()
        .filter(|f| !shared(f))
        .map(|(name, source, kernels)| {
            let mut text = header.clone();
            text.push_str(&prefix);
            push_file(&mut text, name, source);
            Unit {
                name: (*name).to_owned(),
                text,
                kernels: kernels.clone(),
            }
        })
        .collect()
}
fn dedup_in_order(names: &mut Vec<String>) {
    let mut seen = std::collections::HashSet::new();
    names.retain(|n| seen.insert(n.clone()));
}

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Ident(String),
    Str(String),
    Punct(char),
}

/// Split MSL text into identifiers, string literals and single punctuation characters. Comments and preprocessor
/// directives (with their continuation lines) are dropped.
fn tokenize(source: &str) -> Vec<Token> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut line_start = true;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            line_start = true;
            i += 1;
        } else if c.is_whitespace() {
            i += 1;
        } else if c == '#' && line_start {
            // preprocessor directive: skip to the end of the (continued) line
            while i < chars.len() {
                if chars[i] == '\\' && i + 1 < chars.len() && chars[i + 1] == '\n' {
                    i += 2;
                } else if chars[i] == '\n' {
                    break;
                } else {
                    i += 1;
                }
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(chars.len());
        } else if c == '"' {
            line_start = false;
            i += 1;
            let mut text = String::new();
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 1;
                }
                text.push(chars[i]);
                i += 1;
            }
            i += 1;
            tokens.push(Token::Str(text));
        } else if c.is_ascii_alphabetic() || c == '_' {
            line_start = false;
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(Token::Ident(chars[start..i].iter().collect()));
        } else if c.is_ascii_digit() {
            line_start = false;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || chars[i] == '.' || chars[i] == '_')
            {
                i += 1;
            }
        } else {
            line_start = false;
            tokens.push(Token::Punct(c));
            i += 1;
        }
    }
    tokens
}

/// Names of the kernels a source defines: `kernel void NAME(` / `[[kernel]] void NAME(` and the names given to
/// template instantiations by `[[host_name("NAME")]]`. Kernels generated inside preprocessor macros are not seen;
/// the pipeline cache falls back to asking the compiled libraries for such names.
pub fn scan_kernel_names(source: &str) -> Vec<String> {
    let tokens = tokenize(source);
    let mut names = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        let Token::Ident(word) = token else { continue };
        if word == "host_name" {
            if let (Some(Token::Punct('(')), Some(Token::Str(name))) =
                (tokens.get(i + 1), tokens.get(i + 2))
            {
                names.push(name.clone());
            }
        } else if word == "kernel" {
            let mut j = i + 1;
            while matches!(tokens.get(j), Some(Token::Punct(']'))) {
                j += 1;
            }
            if let (Some(Token::Ident(ret)), Some(Token::Ident(name))) =
                (tokens.get(j), tokens.get(j + 1))
            {
                if ret == "void" {
                    names.push(name.clone());
                }
            }
        }
    }
    dedup_in_order(&mut names);
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_plain_and_attribute_kernels() {
        let src = r#"
            kernel void rms_norm_f32(device const float *x [[buffer(0)]]) {}
            [[kernel]] void swiglu_f16 (device half *x [[buffer(0)]]) {}
            static inline float helper(float x) { return x; }
        "#;
        assert_eq!(scan_kernel_names(src), vec!["rms_norm_f32", "swiglu_f16"]);
    }

    #[test]
    fn finds_template_instantiations_by_host_name() {
        let src = r#"
            template <typename T> kernel void copy_t(device T *a [[buffer(0)]]) {}
            typedef decltype(copy_t<half>) copy_t_type;
            template [[host_name("copy_f16")]] kernel copy_t_type copy_t<half>;
            template [[host_name("copy_f32")]] kernel copy_t_type copy_t<float>;
        "#;
        let names = scan_kernel_names(src);
        assert!(names.contains(&"copy_f16".to_owned()) && names.contains(&"copy_f32".to_owned()));
        assert!(
            names.contains(&"copy_t".to_owned()),
            "the template itself is listed too, the lookup copes"
        );
    }

    #[test]
    fn ignores_comments_strings_and_directives() {
        let src = "// kernel void fake_a()\n/* kernel void fake_b() */\n#define K kernel void fake_c\nconst char *s = \"kernel void fake_d\";\nkernel void real(){}\n";
        assert_eq!(scan_kernel_names(src), vec!["real"]);
    }

    #[test]
    fn dedups_names() {
        let src = "kernel void a(){}\nkernel void a(){}\n";
        assert_eq!(scan_kernel_names(src), vec!["a"]);
    }

    #[test]
    fn shared_files_are_prefixed_to_every_unit() {
        let sources = [
            (
                "common.metal",
                "#include <metal_stdlib>\nstruct P { uint n; };\n",
            ),
            ("norm.metal", "kernel void rms(){}"),
            ("rope.metal", "kernel void rope(){}\n"),
        ];
        let defines = vec![
            ("FLLM_SIMD_WIDTH".to_owned(), "32".to_owned()),
            ("FLLM_FLAG".to_owned(), String::new()),
        ];
        let units = build_units(&sources, &defines, false);
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].name, "norm.metal");
        assert!(units[0].text.starts_with(
            "#define FLLM_SIMD_WIDTH 32\n#define FLLM_FLAG\n#line 1 \"common.metal\"\n"
        ));
        let common_at = units[0].text.find("struct P").unwrap();
        let own_at = units[0].text.find("kernel void rms").unwrap();
        assert!(common_at < own_at);
        assert!(!units[0].text.contains("rope"));
        assert_eq!(units[1].kernels, vec!["rope"]);
    }

    #[test]
    fn single_mode_has_one_unit_with_everything() {
        let sources = [
            ("common.metal", "struct P{};\n"),
            ("a.metal", "kernel void a(){}\n"),
            ("b.metal", "kernel void b(){}\n"),
        ];
        let units = build_units(&sources, &[], true);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].kernels, vec!["a", "b"]);
        assert!(units[0].text.contains("struct P") && units[0].text.contains("kernel void b"));
    }

    #[test]
    fn helper_only_files_are_shared_too() {
        let sources = [
            ("a_kernels.metal", "kernel void a(){ help(); }\n"),
            ("quant_helpers.metal", "inline void help() {}\n"),
        ];
        let units = build_units(&sources, &[], false);
        assert_eq!(units.len(), 1);
        assert!(
            units[0].text.find("inline void help").unwrap()
                < units[0].text.find("kernel void a").unwrap()
        );
        let single = build_units(&sources, &[], true);
        assert!(
            single[0].text.find("inline void help").unwrap()
                < single[0].text.find("kernel void a").unwrap()
        );
    }

    #[test]
    fn shared_name_rule() {
        assert!(
            is_shared("common.metal")
                && is_shared("common_quants.metal")
                && is_shared("_helpers.metal")
        );
        assert!(!is_shared("matmul_q4.metal"));
    }
}
