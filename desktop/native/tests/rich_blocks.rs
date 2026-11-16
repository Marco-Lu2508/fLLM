//! Rich blocks: charts, diagrams, SVG pictures and typeset math in an answer — what is drawn, what
//! stays code, what is refused, and that nothing the model writes becomes markup.

use fllm_app::app::Message;
use fllm_app::rich::{self, chart};
use fllm_app::theme::Tokens;
use iced::widget::markdown;
use iced::{Element, Length};
use iced_test::Simulator;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn light() -> Tokens {
    Tokens::new(false, "blue")
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

// ---- the spec of a chart -------------------------------------------------------------------------

#[test]
fn the_three_ways_to_write_a_chart_mean_the_same() {
    let chartjs = chart::parse(r#"{"type":"bar","data":{"labels":["A","B"],"datasets":[{"label":"X","data":[1,2]}]}}"#).unwrap();
    let short = chart::parse(r#"{"type":"bar","labels":["A","B"],"series":[{"name":"X","values":[1,2]}]}"#).unwrap();
    let objects = chart::parse(r#"{"type":"bar","data":[{"label":"A","value":1},{"label":"B","value":2}]}"#).unwrap();
    let pairs = chart::parse(r#"{"type":"bar","data":[["A",1],["B",2]]}"#).unwrap();
    let values = chart::parse(r#"{"type":"bar","labels":["A","B"],"values":[1,2]}"#).unwrap();
    for spec in [&chartjs, &short, &objects, &pairs, &values] {
        assert_eq!(spec.labels, ["A", "B"]);
        assert_eq!(spec.series[0].values, [Some(1.0), Some(2.0)]);
        assert_eq!(spec.kind, chart::Kind::Bar);
    }
    assert_eq!(chartjs.series[0].name, "X");
    assert_eq!(objects.series[0].name, "", "a single series has no name unless it is given one");
}

#[test]
fn the_names_of_the_types_and_the_options_of_chartjs_are_understood() {
    let kind = |t: &str| chart::parse(&format!(r#"{{"type":"{t}","labels":["a","b"],"values":[1,2]}}"#)).map(|s| (s.kind, s.horizontal, s.stacked, s.area, s.smooth, s.doughnut));
    assert_eq!(kind("column").unwrap().0, chart::Kind::Bar);
    assert_eq!(kind("horizontalBar").unwrap(), (chart::Kind::Bar, true, false, false, false, false));
    assert_eq!(kind("stacked-bar").unwrap(), (chart::Kind::Bar, false, true, false, false, false));
    assert_eq!(kind("area").unwrap(), (chart::Kind::Line, false, false, true, false, false));
    assert!(kind("Spline").unwrap().4);
    assert_eq!(kind("donut").unwrap(), (chart::Kind::Pie, false, false, false, false, true));
    assert_eq!(kind("radar").unwrap().0, chart::Kind::Radar);
    assert!(kind("treemap").unwrap_err().contains("not known"));
    let s = chart::parse(r#"{"type":"bar","options":{"indexAxis":"y","scales":{"x":{"stacked":true,"title":{"text":"Units"}},"y":{"title":{"text":"Team"},"min":0,"max":50}},"plugins":{"title":{"text":"Sales"},"legend":{"display":false}}},"labels":["a"],"values":[3],"unit":"€"}"#).unwrap();
    assert!(s.horizontal && s.stacked);
    assert_eq!((s.title.as_str(), s.x_title.as_str(), s.y_title.as_str(), s.unit.as_str()), ("Sales", "Units", "Team", "€"));
    assert_eq!((s.legend, s.min, s.max), (Some(false), Some(0.0), Some(50.0)));
}

#[test]
fn a_chart_that_cannot_be_drawn_says_why_and_a_half_written_one_is_not_an_error_yet() {
    assert!(chart::parse("[1,2]").unwrap_err().contains("JSON object"));
    assert!(chart::parse(r#"{"type":"bar"}"#).unwrap_err().contains("no numbers"));
    assert!(chart::parse(r#"{"type":"bar","values":["x","y"]}"#).unwrap_err().contains("no numbers"));
    assert!(chart::parse("{nope").unwrap_err().contains("not valid JSON"));
    // an answer that is still being written ends in the middle of the JSON
    assert!(chart::unfinished(r#"{"type":"bar","labels":["A","#));
    assert!(chart::unfinished(r#"{"type":"bar""#));
    assert!(!chart::unfinished("{nope"), "a mistake in the middle is a mistake");
    assert!(!chart::unfinished(r#"{"type":"bar","values":[1]}"#), "a whole chart is not unfinished");
    // numbers written as text count, gaps stay gaps, and too much is cut
    let s = chart::parse(r#"{"type":"line","labels":["a","b","c"],"values":["1,5","2",null]}"#).unwrap();
    assert_eq!(s.series[0].values, [Some(1.5), Some(2.0), None]);
    let many: Vec<String> = (0..2000).map(|i| i.to_string()).collect();
    let s = chart::parse(&format!(r#"{{"type":"bar","values":[{}]}}"#, many.join(","))).unwrap();
    assert_eq!(s.series[0].values.len(), chart::MAX_POINTS);
}

#[test]
fn numbers_and_ticks_are_written_the_way_a_chart_writes_them() {
    let f = chart::format_number;
    assert_eq!((f(0.0).as_str(), f(5.0).as_str(), f(0.25).as_str(), f(12.5).as_str(), f(-3.0).as_str()), ("0", "5", "0.25", "12.5", "-3"));
    assert_eq!((f(12_500.0).as_str(), f(3_400_000.0).as_str(), f(2.5e9).as_str(), f(999.0).as_str()), ("12.5k", "3.4M", "2.5B", "999"));
    assert_eq!(chart::nice_ticks(0.0, 300.0, 5), vec![0.0, 100.0, 200.0, 300.0]);
    let t = chart::nice_ticks(0.0, 97.0, 5);
    assert_eq!(t.first(), Some(&0.0));
    assert!(*t.last().unwrap() >= 97.0 && t.windows(2).all(|w| w[1] > w[0]), "{t:?}");
    let t = chart::nice_ticks(-40.0, 25.0, 5);
    assert!(t.contains(&0.0) && t.first().unwrap() <= &-40.0 && t.last().unwrap() >= &25.0, "{t:?}");
    // a flat range still has a scale
    assert!(chart::nice_ticks(5.0, 5.0, 5).len() >= 2);
}

// ---- what the picture holds ----------------------------------------------------------------------

#[test]
fn a_bar_chart_has_a_bar_for_every_value_and_the_words_are_outlines() {
    let art = chart::render(r#"{"type":"bar","title":"T","labels":["A","B","C"],"series":[{"name":"x","values":[1,2,3]},{"name":"y","values":[3,2,1]}]}"#, light()).unwrap();
    let svg = art.source.to_string();
    // six bars with rounded tops and two legend swatches
    assert_eq!(count(&svg, "<rect"), 2, "the legend");
    assert!(count(&svg, "Q") >= 6, "the rounded corners of six bars");
    assert!(!svg.contains("<text") && !svg.contains("font-family"), "words are outlines: no font is needed");
    assert!(art.width > 0.0 && art.height > 100.0);
    assert!(rich::from_svg(&svg).is_ok(), "and it is an SVG the window can draw");
}

#[test]
fn every_kind_of_chart_is_drawn_in_both_themes() {
    let specs = [
        r#"{"type":"line","labels":["a","b","c","d"],"series":[{"name":"s","values":[1,3,2,4]}]}"#,
        r#"{"type":"line","smooth":true,"labels":["a","b","c"],"values":[1,null,3]}"#,
        r#"{"type":"area","labels":["a","b","c"],"series":[{"name":"s","values":[1,3,2]},{"name":"t","values":[2,2,2]}]}"#,
        r#"{"type":"stackedarea","labels":["a","b","c"],"series":[{"name":"s","values":[1,3,2]},{"name":"t","values":[2,2,2]}]}"#,
        r#"{"type":"stackedbar","labels":["a","b"],"series":[{"name":"s","values":[1,3]},{"name":"t","values":[2,-2]}]}"#,
        r#"{"type":"hbar","labels":["a","b","c"],"values":[1,2,3]}"#,
        r#"{"type":"pie","data":[["a",1],["b",2],["c",0],["d",-1]]}"#,
        r#"{"type":"doughnut","data":[["a",1],["b",2]]}"#,
        r#"{"type":"scatter","data":{"datasets":[{"label":"p","data":[[1,2],[3,4],{"x":5,"y":1}]}]}}"#,
        r#"{"type":"radar","labels":["a","b","c","d"],"series":[{"name":"s","values":[1,2,3,4]}]}"#,
        r#"{"type":"bar","labels":["a long long long label","another very long label","third one is long as well","fourth"],"values":[1,2,3,4]}"#,
        r#"{"type":"bar","labels":["a"],"values":[0]}"#,
        r#"{"type":"bar","labels":["a","b"],"values":[-5,-2]}"#,
    ];
    for dark in [false, true] {
        let t = Tokens::new(dark, "blue");
        for s in specs {
            let art = chart::render(s, t).unwrap_or_else(|e| panic!("{s}: {e}"));
            assert!(rich::from_svg(&art.source).is_ok(), "{s}");
            assert!(art.height >= 100.0 && art.height <= 1000.0, "{s}: {}", art.height);
            assert!(!art.source.contains("NaN") && !art.source.contains("inf"), "{s}: a number went wrong");
        }
    }
    // the dark theme draws in other colors than the light one
    let s = specs[0];
    assert_ne!(chart::render(s, Tokens::new(true, "blue")).unwrap().source, chart::render(s, light()).unwrap().source);
}

#[test]
fn a_pie_has_a_slice_for_every_positive_value_and_says_how_big_it_is() {
    let art = chart::render(r#"{"type":"pie","data":[["a",50],["b",25],["c",25],["gone",0]]}"#, light()).unwrap();
    let svg = art.source.to_string();
    assert!(count(&svg, "A") >= 3);
    assert!(count(&svg, "stroke-linejoin=\"round\"/>") >= 3);
    // the percent signs are glyphs: three labels of 25-50 percent
    assert!(count(&svg, "<path") >= 3 + 3);
}

#[test]
fn nothing_the_model_writes_becomes_markup() {
    // a color that tries to leave its attribute is dropped; the series gets the color of the theme
    let evil = r#"{"type":"bar","labels":["a"],"series":[{"name":"<script>alert(1)</script>","values":[1],"color":"red\" onload=\"x"}]}"#;
    let art = chart::render(evil, light()).unwrap();
    let svg = art.source.to_string();
    assert!(!svg.contains("onload") && !svg.contains("<script") && !svg.contains("alert"), "{svg}");
    assert_eq!(chart::clean_color("#ff8800").as_deref(), Some("#ff8800"));
    assert_eq!(chart::clean_color("rgba(1, 2, 3, 0.5)").as_deref(), Some("rgba(1, 2, 3, 0.5)"));
    assert_eq!(chart::clean_color("tomato").as_deref(), Some("tomato"));
    for bad in ["red\" x=\"1", "url(#a)", "#12", "#gggggg", "<b>", "", "javascript:1"] {
        assert_eq!(chart::clean_color(bad), None, "{bad}");
    }
    // a title with markup is outlines of its characters, not markup
    let art = chart::render(r#"{"type":"bar","title":"</svg><script>x</script>","labels":["a"],"values":[1]}"#, light()).unwrap();
    assert_eq!(count(&art.source, "<svg"), 1);
    assert!(!art.source.contains("<script"));
}

// ---- math, diagrams, pictures --------------------------------------------------------------------

#[test]
fn math_is_typeset_sized_by_the_text_and_remembered() {
    let small = rich::math(r"\frac{a}{b}", false, 15.0, light().fg).unwrap();
    let big = rich::math(r"\frac{a}{b}", false, 30.0, light().fg).unwrap();
    assert!(big.width > small.width * 1.6 && big.height > small.height * 1.6, "{} {}", small.width, big.width);
    assert!(small.baseline > 0.0 && small.baseline < small.height + 0.01);
    // display style is taller for a sum with limits
    let inline = rich::math(r"\sum_{i=1}^{n} i", false, 16.0, light().fg).unwrap();
    let display = rich::math(r"\sum_{i=1}^{n} i", true, 16.0, light().fg).unwrap();
    assert!(display.height > inline.height, "limits above and below");
    let before = rich::kept();
    let again = rich::math(r"\frac{a}{b}", false, 15.0, light().fg).unwrap();
    assert_eq!(rich::kept(), before, "the same formula is not made twice");
    assert_eq!(again.source, small.source);
    // the color of the text is the color of the formula
    let red = rich::math("x", false, 16.0, iced::Color::from_rgb8(255, 0, 0)).unwrap();
    assert!(red.source.contains("#ff0000"));
    for tex in [r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}", r"\begin{cases} x & x>0 \\ 0 & \text{else} \end{cases}", r"\int_0^\infty e^{-x^2}\,dx", r"\begin{aligned} a &= b \\ c &= d \end{aligned}", r"\left( \frac{1}{2} \right)^n", r"\mathbb{R}^n \to \mathcal{F}", r"\sqrt[3]{x}", r"\hat{x}\ \vec{v}\ \overline{AB}"] {
        assert!(rich::math(tex, true, 18.0, light().fg).is_ok(), "{tex}");
    }
}

#[test]
fn math_that_is_not_math_is_an_error_and_never_a_panic() {
    for bad in ["", "   ", r"\notacommand{x}", r"\frac{1}", r"{{{{", r"\begin{matrix} a", "$$"] {
        assert!(rich::math(bad, false, 15.0, light().fg).is_err(), "{bad:?}");
    }
    assert!(rich::math(&"x+".repeat(20_000), false, 15.0, light().fg).unwrap_err().contains("too long"));
}

#[test]
fn a_mermaid_diagram_is_drawn_in_the_colors_of_the_theme_with_outlined_words() {
    let src = "flowchart TD\n  A[Start] --> B{Choice}\n  B -->|Yes| C[Done]\n  B -->|No| A";
    let l = rich::diagram(src, light(), false).unwrap();
    let d = rich::diagram(src, Tokens::new(true, "blue"), true).unwrap();
    assert!(l.width > 100.0 && l.height > 100.0);
    assert!(!l.source.contains("<text"), "the words are outlines");
    assert_ne!(l.source, d.source);
    for kind in ["sequenceDiagram\n  A->>B: hi", "classDiagram\n  class A\n  A <|-- B", "stateDiagram-v2\n  [*] --> S\n  S --> [*]", "pie title Pets\n  \"Dogs\" : 5\n  \"Cats\" : 3", "gantt\n  title Plan\n  dateFormat YYYY-MM-DD\n  section A\n  Task : 2026-01-01, 3d", "erDiagram\n  CUSTOMER ||--o{ ORDER : places"] {
        assert!(rich::diagram(kind, light(), false).is_ok(), "{kind}");
    }
    // text that is no diagram is an error, whatever the engine does with it
    assert!(rich::diagram("", light(), false).is_err());
    let _ = rich::diagram("flowchart LR\n A --", light(), false);
    let _ = rich::diagram(&"A-->B\n".repeat(50), light(), false);
}

#[test]
fn an_svg_picture_has_the_size_it_names_and_a_broken_one_is_refused() {
    let art = rich::picture(r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60"><rect width="200" height="60" fill="#4f46e5"/></svg>"##).unwrap();
    assert_eq!((art.width, art.height), (200.0, 60.0));
    let boxed = rich::picture(r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80"><circle cx="40" cy="40" r="30"/></svg>"#).unwrap();
    assert_eq!((boxed.width, boxed.height), (120.0, 80.0), "the viewBox is the size when there is no width");
    assert!(rich::picture("<svg").is_err());
    assert!(rich::picture("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"0\" height=\"0\"></svg>").is_err());
    assert!(rich::picture("not svg at all").is_err());
}

// ---- in the answer ---------------------------------------------------------------------------------

fn view_of<'a>(content: &'a markdown::Content, t: Tokens) -> Element<'a, Message> {
    let theme = if t.dark { iced::Theme::Dark } else { iced::Theme::Light };
    fllm_app::md::view(content, t, &theme)
}

fn find_in(source: &str, words: &str) -> bool {
    let content = markdown::Content::parse(source);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(900.0, 3000.0), view_of(&content, light()));
    ui.find(words).is_ok()
}

#[test]
fn a_chart_block_is_drawn_and_its_json_is_not_shown() {
    let json = r#"{"type":"bar","title":"Sales","labels":["Q1","Q2"],"values":[1,2]}"#;
    let source = format!("Here:\n\n```chart\n{json}\n```\n");
    assert!(find_in(&source, "Save SVG") && find_in(&source, "Copy code"), "the block has its buttons");
    assert!(!find_in(&source, json), "the JSON is not shown under the chart");
    assert!(find_in(&source, "chart"), "the label of the block");
}

#[test]
fn a_chart_that_is_still_being_written_stays_code_and_a_wrong_one_says_so() {
    let half = "```chart\n{\"type\":\"bar\",\"labels\":[\"Q1\",";
    assert!(find_in(half, "Copy code") && !find_in(half, "Save SVG"), "half a chart is shown as the code it is");
    assert!(!find_in(half, "could not be drawn"), "and nothing is said while it is still coming");
    let wrong = "```chart\n{\"type\":\"treemap\",\"values\":[1]}\n```";
    assert!(find_in(wrong, "This chart could not be drawn: The chart type \"treemap\" is not known. Use bar, line, area, pie, doughnut, scatter or radar."));
    assert!(find_in(wrong, "Copy code"), "the code stays");
}

#[test]
fn other_code_blocks_are_untouched() {
    for source in ["```rust\nfn main() {}\n```", "```json\n{\"type\":\"bar\"}\n```", "```svg\nnot a picture\n```", "```\nplain\n```"] {
        assert!(find_in(source, "Copy code"), "{source}");
        assert!(!find_in(source, "Save SVG"), "{source}");
    }
}

#[test]
fn math_diagram_and_svg_blocks_are_drawn() {
    for source in ["```math\n\\frac{1}{2}\n```", "```latex\n$$x^2$$\n```", "```mermaid\nflowchart LR\n A-->B\n```", "```svg\n<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\"><rect width=\"20\" height=\"20\"/></svg>\n```"] {
        assert!(find_in(source, "Save SVG"), "{source}");
    }
}

#[test]
fn the_save_button_writes_the_picture_next_to_the_downloads() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("FLLM_EXPORT_DIR", dir.path());
    let art = chart::render(r#"{"type":"bar","labels":["a"],"values":[1]}"#, light()).unwrap();
    let store_dir = tempfile::tempdir().unwrap();
    let path = store_dir.path().join("chats.json");
    fllm_app::store::Store::default().save(&path).unwrap();
    let (mut app, _task) = fllm_app::app::App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::SaveRich { name: "chart.svg".into(), svg: art.source.clone() });
    assert_eq!(std::fs::read_to_string(dir.path().join("chart.svg")).unwrap(), art.source.to_string());
    assert_eq!(app.toast(), Some("Saved to your downloads."));
    // and the button of the block sends exactly that
    let content = markdown::Content::parse("```chart\n{\"type\":\"bar\",\"labels\":[\"a\"],\"values\":[1]}\n```");
    let mut ui = Simulator::with_size(settings(), iced::Size::new(900.0, 1500.0), view_of(&content, light()));
    ui.click("Save SVG").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::SaveRich { name, .. }] if name == "chart.svg"), "{sent:?}");
    let _ = Length::Fill;
}

#[test]
fn the_answer_is_drawn_in_both_themes() {
    let answer = r##"Here is the **quadratic formula**: for $ax^2+bx+c=0$ the roots are

$$x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$$

and $\Delta = b^2 - 4ac$ tells you how many real roots exist, for example $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$ with `code` and a [link](https://example.com).

```chart
{"type":"bar","title":"Revenue per quarter","labels":["Q1","Q2","Q3","Q4"],"series":[{"name":"2025","values":[120,190,150,240]},{"name":"2026","values":[150,210,180,300]}]}
```

```mermaid
sequenceDiagram
  Alice->>Bob: Hello Bob
  Bob-->>Alice: Hi Alice
```
"##;
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    for (dark, name) in [(false, "rich-answer-light.png"), (true, "rich-answer-dark.png")] {
        let t = Tokens::new(dark, "blue");
        let theme = if dark { iced::Theme::Dark } else { iced::Theme::Light };
        let content = markdown::Content::parse(answer);
        let body: Element<Message> = fllm_app::md::view(&content, t, &theme);
        let view: Element<Message> = iced::widget::container(iced::widget::container(body).padding(20).max_width(760))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_| iced::widget::container::Style { background: Some(iced::Background::Color(t.bg)), ..Default::default() })
            .into();
        let mut ui = Simulator::with_size(settings(), iced::Size::new(800.0, 1300.0), view);
        let image = ui.snapshot(&theme).unwrap();
        assert!(image.matches_image(out.join(name)).unwrap(), "the picture of the answer ({name})");
    }
}

#[test]
fn the_model_is_told_about_the_formats_unless_you_switch_it_off_or_replace_the_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    fllm_app::store::Store::default().save(&path).unwrap();
    let (mut app, _task) = fllm_app::app::App::with("http://127.0.0.1:9".into(), path);
    let on = app.system_prompt();
    assert!(on.contains("```chart") && on.contains("```mermaid") && on.contains("$$ ... $$"), "{on}");
    assert!(on.find(fllm_app::rich::INSTRUCTIONS).is_some_and(|at| at > 0), "behind the built-in prompt");
    let _ = app.update(Message::RichAnswers(false));
    assert!(!app.system_prompt().contains("```chart"), "switched off");
    let _ = app.update(Message::RichAnswers(true));
    // your own prompt that replaces the built-in one is left alone
    let _ = app.update(Message::SystemCustom(true));
    assert!(!app.system_prompt().contains("```chart") || !app.system_prompt().starts_with("You are"), "a custom prompt is not the built-in one");
}
