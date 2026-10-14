//! Writer for OpenVINO IR v11 (`.xml` graph + `.bin` constants).
//!
//! Only what the NPU linear models need: single-output layers (`Parameter`, `Const`, `Convert`, `Multiply`, `Reshape`, `MatMul`,
//! `Result`, ...), explicit element types and static shapes. The format follows the OpenVINO "Deep Learning Network Intermediate
//! Representation" documentation and the reader's own tests (`src/frontends/ir/tests/frontend_test_basic.cpp`): every layer has an
//! `id`, `name`, `type` and `version` (the opset), input ports are numbered from 0, the output port of a layer continues the
//! numbering after its inputs, edges name `from-layer`/`from-port`/`to-layer`/`to-port`, a `Const` points into the `.bin` with
//! `offset` and `size` in bytes. 4-bit constants are packed little-nibble first (element `2j` in bits 0..3 of byte `j`), as
//! `ov::element::Iterator` reads them (`is_lsb_packed`).
//!
//! The text is only produced here; whether a given OpenVINO version accepts it is checked by `read_model` at run time (and by the
//! Python parser check described in OPENVINO.md).
//!
//! Verification level: L0 (unit tests check structure only; the generated files have not been run on an NPU).

use std::fmt::Write as _;

/// Element types the writer knows (`element_type` attribute in lower case, `precision` of ports in upper case).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F16,
    F32,
    I4,
    U4,
    I8,
    U8,
    I32,
    I64,
}

impl Elem {
    /// Value of the `element_type` / `destination_type` attributes.
    pub fn name(self) -> &'static str {
        match self {
            Elem::F16 => "f16",
            Elem::F32 => "f32",
            Elem::I4 => "i4",
            Elem::U4 => "u4",
            Elem::I8 => "i8",
            Elem::U8 => "u8",
            Elem::I32 => "i32",
            Elem::I64 => "i64",
        }
    }

    /// Value of the `precision` attribute of ports.
    pub fn precision(self) -> &'static str {
        match self {
            Elem::F16 => "FP16",
            Elem::F32 => "FP32",
            Elem::I4 => "I4",
            Elem::U4 => "U4",
            Elem::I8 => "I8",
            Elem::U8 => "U8",
            Elem::I32 => "I32",
            Elem::I64 => "I64",
        }
    }

    pub fn bits(self) -> usize {
        match self {
            Elem::I4 | Elem::U4 => 4,
            Elem::I8 | Elem::U8 => 8,
            Elem::F16 => 16,
            Elem::F32 | Elem::I32 => 32,
            Elem::I64 => 64,
        }
    }
}

/// Bytes `count` elements of `elem` occupy in a `.bin` (sub-byte types are packed).
pub fn byte_len(elem: Elem, count: usize) -> usize {
    (count * elem.bits()).div_ceil(8)
}

/// Output of a layer, used as an input of the next.
#[derive(Clone, Debug)]
pub struct Node {
    layer: usize,
    port: usize,
    pub elem: Elem,
    pub dims: Vec<usize>,
}

struct PortDef {
    id: usize,
    elem: Elem,
    dims: Vec<usize>,
    /// `names` attribute (tensor name), if any.
    name: Option<String>,
}

struct Layer {
    id: usize,
    name: String,
    kind: &'static str,
    attrs: Vec<(&'static str, String)>,
    inputs: Vec<PortDef>,
    output: Option<PortDef>,
}

struct Edge {
    from_layer: usize,
    from_port: usize,
    to_layer: usize,
    to_port: usize,
}

/// Incrementally built graph; `finish` yields the two files.
pub struct IrBuilder {
    name: String,
    layers: Vec<Layer>,
    edges: Vec<Edge>,
    bin: Vec<u8>,
}

/// Alignment of constants inside the `.bin` (the runtime may map the file and read the constants in place).
const BIN_ALIGN: usize = 64;

fn dims_text(dims: &[usize]) -> String {
    dims.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\n' | '\r' | '\t' => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

impl IrBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            layers: Vec::new(),
            edges: Vec::new(),
            bin: Vec::new(),
        }
    }

    fn push_layer(
        &mut self,
        name: &str,
        kind: &'static str,
        attrs: Vec<(&'static str, String)>,
        inputs: &[&Node],
        out: Option<(Elem, Vec<usize>, Option<&str>)>,
    ) -> Option<Node> {
        let id = self.layers.len();
        let mut in_ports = Vec::with_capacity(inputs.len());
        for (index, node) in inputs.iter().enumerate() {
            in_ports.push(PortDef {
                id: index,
                elem: node.elem,
                dims: node.dims.clone(),
                name: None,
            });
            self.edges.push(Edge {
                from_layer: node.layer,
                from_port: node.port,
                to_layer: id,
                to_port: index,
            });
        }
        let node = out.map(|(elem, dims, tensor)| {
            let port = inputs.len();
            let def = PortDef {
                id: port,
                elem,
                dims: dims.clone(),
                name: tensor.map(str::to_owned),
            };
            (
                def,
                Node {
                    layer: id,
                    port,
                    elem,
                    dims,
                },
            )
        });
        let (output, handle) = match node {
            Some((def, node)) => (Some(def), Some(node)),
            None => (None, None),
        };
        self.layers.push(Layer {
            id,
            name: name.to_owned(),
            kind,
            attrs,
            inputs: in_ports,
            output,
        });
        handle
    }

    /// A model input of the given type and static shape (`tensor_name` names the tensor).
    pub fn parameter(&mut self, name: &str, elem: Elem, dims: &[usize]) -> Node {
        let attrs = vec![
            ("shape", dims_text(dims)),
            ("element_type", elem.name().to_owned()),
        ];
        self.push_layer(
            name,
            "Parameter",
            attrs,
            &[],
            Some((elem, dims.to_vec(), Some(name))),
        )
        .expect("parameter has an output")
    }

    /// A constant whose bytes are appended to the `.bin`. `data` must have exactly `byte_len(elem, product(dims))` bytes.
    pub fn constant(
        &mut self,
        name: &str,
        elem: Elem,
        dims: &[usize],
        data: &[u8],
    ) -> Result<Node, String> {
        let count: usize = dims.iter().product();
        let expected = byte_len(elem, count);
        if data.len() != expected {
            return Err(format!(
                "constant {name}: {} bytes for {count} {} elements, expected {expected}",
                data.len(),
                elem.name()
            ));
        }
        while !self.bin.len().is_multiple_of(BIN_ALIGN) {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(data);
        let attrs = vec![
            ("element_type", elem.name().to_owned()),
            ("shape", dims_text(dims)),
            ("offset", offset.to_string()),
            ("size", data.len().to_string()),
        ];
        Ok(self
            .push_layer(name, "Const", attrs, &[], Some((elem, dims.to_vec(), None)))
            .expect("const has an output"))
    }

    /// An operation of opset1 with one output. `attrs` are the entries of its `<data>` element.
    pub fn op(
        &mut self,
        kind: &'static str,
        name: &str,
        attrs: &[(&'static str, &str)],
        inputs: &[&Node],
        out_elem: Elem,
        out_dims: &[usize],
    ) -> Node {
        let attrs = attrs.iter().map(|(k, v)| (*k, (*v).to_owned())).collect();
        self.push_layer(
            name,
            kind,
            attrs,
            inputs,
            Some((out_elem, out_dims.to_vec(), None)),
        )
        .expect("op has an output")
    }

    /// Mark `input` as the model output.
    pub fn result(&mut self, name: &str, input: &Node) {
        self.push_layer(name, "Result", Vec::new(), &[input], None);
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn bin(&self) -> &[u8] {
        &self.bin
    }

    /// The graph as IR v11 XML text.
    pub fn to_xml(&self) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("<?xml version=\"1.0\"?>\n");
        let _ = writeln!(s, "<net name=\"{}\" version=\"11\">", escape(&self.name));
        s.push_str("\t<layers>\n");
        for layer in &self.layers {
            let _ = writeln!(
                s,
                "\t\t<layer id=\"{}\" name=\"{}\" type=\"{}\" version=\"opset1\">",
                layer.id,
                escape(&layer.name),
                layer.kind
            );
            if !layer.attrs.is_empty() {
                s.push_str("\t\t\t<data");
                for (key, value) in &layer.attrs {
                    let _ = write!(s, " {key}=\"{}\"", escape(value));
                }
                s.push_str("/>\n");
            }
            if !layer.inputs.is_empty() {
                s.push_str("\t\t\t<input>\n");
                for port in &layer.inputs {
                    write_port(&mut s, port, "\t\t\t\t");
                }
                s.push_str("\t\t\t</input>\n");
            }
            if let Some(port) = &layer.output {
                s.push_str("\t\t\t<output>\n");
                write_port(&mut s, port, "\t\t\t\t");
                s.push_str("\t\t\t</output>\n");
            }
            s.push_str("\t\t</layer>\n");
        }
        s.push_str("\t</layers>\n\t<edges>\n");
        for e in &self.edges {
            let _ = writeln!(
                s,
                "\t\t<edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{}\" to-port=\"{}\"/>",
                e.from_layer, e.from_port, e.to_layer, e.to_port
            );
        }
        s.push_str("\t</edges>\n\t<rt_info/>\n</net>\n");
        s
    }

    /// The two files: XML text and `.bin` bytes.
    pub fn finish(self) -> (String, Vec<u8>) {
        (self.to_xml(), self.bin)
    }
}

fn write_port(s: &mut String, port: &PortDef, indent: &str) {
    let _ = write!(
        s,
        "{indent}<port id=\"{}\" precision=\"{}\"",
        port.id,
        port.elem.precision()
    );
    if let Some(name) = &port.name {
        let _ = write!(s, " names=\"{}\"", escape(name));
    }
    if port.dims.is_empty() {
        s.push_str("/>\n");
        return;
    }
    s.push_str(">\n");
    for d in &port.dims {
        let _ = writeln!(s, "{indent}\t<dim>{d}</dim>");
    }
    let _ = writeln!(s, "{indent}</port>");
}

/// A minimal XML reader for the unit tests: element names, attributes and nesting; checks that tags are balanced.
#[cfg(test)]
pub(crate) mod test_xml {
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub struct Element {
        pub tag: String,
        pub attrs: HashMap<String, String>,
        pub text: String,
        pub parent: Option<usize>,
    }

    fn unescape(v: &str) -> String {
        v.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&amp;", "&")
    }

    /// All elements in document order; `Err` for unbalanced or malformed markup.
    pub fn parse(text: &str) -> Result<Vec<Element>, String> {
        let mut out: Vec<Element> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        let mut seen_root = false;
        while i < bytes.len() {
            if bytes[i] != b'<' {
                let end = text[i..].find('<').map_or(text.len(), |p| i + p);
                if let Some(&top) = stack.last() {
                    out[top].text.push_str(text[i..end].trim());
                } else if !text[i..end].trim().is_empty() {
                    return Err("text outside the root element".into());
                }
                i = end;
                continue;
            }
            let close = text[i..].find('>').ok_or("unterminated tag")? + i;
            let inner = &text[i + 1..close];
            i = close + 1;
            if inner.starts_with('?') {
                if !inner.ends_with('?') {
                    return Err("bad processing instruction".into());
                }
                continue;
            }
            if let Some(name) = inner.strip_prefix('/') {
                let top = stack.pop().ok_or("closing tag without opening tag")?;
                if out[top].tag != name.trim() {
                    return Err(format!(
                        "mismatched tag: <{}> closed by </{name}>",
                        out[top].tag
                    ));
                }
                continue;
            }
            let (inner, self_closing) = match inner.strip_suffix('/') {
                Some(rest) => (rest, true),
                None => (inner, false),
            };
            let name_end = inner.find(char::is_whitespace).unwrap_or(inner.len());
            let tag = inner[..name_end].to_owned();
            if tag.is_empty() {
                return Err("empty tag name".into());
            }
            let mut attrs = HashMap::new();
            let mut rest = inner[name_end..].trim_start();
            while !rest.is_empty() {
                let eq = rest
                    .find('=')
                    .ok_or_else(|| format!("attribute without value in <{tag}>"))?;
                let key = rest[..eq].trim().to_owned();
                let after = rest[eq + 1..].trim_start();
                if !after.starts_with('"') {
                    return Err(format!("unquoted attribute {key} in <{tag}>"));
                }
                let end = after[1..].find('"').ok_or("unterminated attribute value")? + 1;
                if attrs
                    .insert(key.clone(), unescape(&after[1..end]))
                    .is_some()
                {
                    return Err(format!("duplicate attribute {key} in <{tag}>"));
                }
                rest = after[end + 1..].trim_start();
            }
            if stack.is_empty() {
                if seen_root {
                    return Err("more than one root element".into());
                }
                seen_root = true;
            }
            let index = out.len();
            out.push(Element {
                tag,
                attrs,
                text: String::new(),
                parent: stack.last().copied(),
            });
            if !self_closing {
                stack.push(index);
            }
        }
        if !stack.is_empty() {
            return Err("unclosed element".into());
        }
        if !seen_root {
            return Err("no root element".into());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::test_xml::parse;
    use super::*;

    fn sample() -> (IrBuilder, Node) {
        let mut b = IrBuilder::new("sample <&> \"net\"");
        let x = b.parameter("x", Elem::F16, &[2, 4]);
        let w = b.constant("w", Elem::F16, &[3, 4], &[0u8; 24]).unwrap();
        b.constant(
            "q",
            Elem::I4,
            &[3, 1, 4],
            &[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB],
        )
        .unwrap();
        let y = b.op(
            "MatMul",
            "mm",
            &[("transpose_a", "false"), ("transpose_b", "true")],
            &[&x, &w],
            Elem::F16,
            &[2, 3],
        );
        b.result("out", &y);
        (b, y)
    }

    #[test]
    fn xml_is_well_formed_and_escaped() {
        let (b, _) = sample();
        let xml = b.to_xml();
        let elements = parse(&xml).expect("well-formed");
        assert_eq!(elements[0].tag, "net");
        assert_eq!(elements[0].attrs["version"], "11");
        assert_eq!(elements[0].attrs["name"], "sample <&> \"net\"");
        assert!(xml.starts_with("<?xml version=\"1.0\"?>"));
        assert!(!xml.contains("<&>"));
    }

    #[test]
    fn layers_edges_and_ports_are_consistent() {
        let (b, _) = sample();
        let elements = parse(&b.to_xml()).unwrap();
        let layers: Vec<_> = elements.iter().filter(|e| e.tag == "layer").collect();
        assert_eq!(layers.len(), 5);
        for (i, l) in layers.iter().enumerate() {
            assert_eq!(l.attrs["id"], i.to_string(), "ids are sequential");
            assert_eq!(l.attrs["version"], "opset1");
        }
        let edges: Vec<_> = elements.iter().filter(|e| e.tag == "edge").collect();
        assert_eq!(edges.len(), 3);
        // MatMul (layer 3) reads x (0) at port 0 and w (1) at port 1; Result (4) reads the MatMul output, whose port id follows the inputs
        let find = |to: &str, port: &str| {
            edges
                .iter()
                .find(|e| e.attrs["to-layer"] == to && e.attrs["to-port"] == port)
                .unwrap()
        };
        assert_eq!(find("3", "0").attrs["from-layer"], "0");
        assert_eq!(find("3", "1").attrs["from-layer"], "1");
        assert_eq!(find("4", "0").attrs["from-layer"], "3");
        assert_eq!(find("4", "0").attrs["from-port"], "2");
        // the MatMul output port is numbered 2
        let mm = elements
            .iter()
            .position(|e| e.tag == "layer" && e.attrs["type"] == "MatMul")
            .unwrap();
        let out_port = elements.iter().enumerate().find(|(i, e)| {
            *i > mm && e.tag == "port" && e.attrs["precision"] == "FP16" && e.attrs["id"] == "2"
        });
        assert!(out_port.is_some());
    }

    #[test]
    fn constants_point_into_the_bin() {
        let (b, _) = sample();
        let elements = parse(&b.to_xml()).unwrap();
        let consts: Vec<_> = elements
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tag == "layer" && e.attrs["type"] == "Const")
            .map(|(index, _)| {
                elements
                    .iter()
                    .find(|d| d.tag == "data" && d.parent == Some(index))
                    .unwrap()
            })
            .collect();
        assert_eq!(consts.len(), 2);
        let mut previous_end = 0;
        for c in consts {
            let offset: usize = c.attrs["offset"].parse().unwrap();
            let size: usize = c.attrs["size"].parse().unwrap();
            assert_eq!(offset % BIN_ALIGN, 0);
            assert!(offset >= previous_end, "constants do not overlap");
            assert!(offset + size <= b.bin().len());
            previous_end = offset + size;
        }
        assert_eq!(
            b.bin().len(),
            64 + 6,
            "f16 3x4 = 24 bytes padded to 64, then the 6 packed bytes of 12 nibbles"
        );
        assert_eq!(&b.bin()[64..70], &[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB]);
    }

    #[test]
    fn constant_size_is_checked() {
        let mut b = IrBuilder::new("n");
        assert!(b.constant("c", Elem::F16, &[2, 2], &[0u8; 7]).is_err());
        assert!(
            b.constant("c", Elem::I4, &[3], &[0u8; 2]).is_ok(),
            "3 nibbles need 2 bytes"
        );
        assert_eq!(byte_len(Elem::U4, 5), 3);
        assert_eq!(byte_len(Elem::F32, 5), 20);
    }

    #[test]
    fn parser_rejects_broken_markup() {
        assert!(parse("<a><b></a></b>").is_err());
        assert!(parse("<a x=1/>").is_err());
        assert!(parse("<a>").is_err());
        assert!(parse("<a/><b/>").is_err());
        assert!(parse("<a x=\"1\" x=\"2\"/>").is_err());
        assert!(parse("<a><b x=\"1\"/>text</a>").is_ok());
    }
}
