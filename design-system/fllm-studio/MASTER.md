# fLLM Studio design system (source of truth)

The Studio and the desktop app share one page (`rust/engine/src/webui/`). The look is layered, loaded in this order and
embedded in the binary: `style.css`, `chat.css`, `code.css` (the original sheets), `theme.css` (v2 tokens),
`apple.css` (v3, "calm native"). A later sheet refines an earlier one; nothing was rewritten. `FLLM_STUDIO_DEV=<dir>`
serves the files from disk on every request, so a change shows on reload.

## Direction

Calm and native, with one identity. The structure follows Apple's Human Interface Guidelines (foundations: color,
typography, layout, materials, motion, accessibility; components: sidebars, toolbars, buttons, sheets, segmented
controls). The identity is **copper**, the material of a Rust engine: one accent, used for the one primary action of a
view, the selection tint and the focus ring. Green is for "ok" only, red for errors. There is no system blue and no
decorative colour.

Deliberate deviations from "looks like macOS": the accent; the composer keeps its own rounded shape; shortcut hints show
Ctrl on Windows and Linux (the shortcut registry in `boot.js` formats them per platform).

## Tokens

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#f5f5f7` | `#161618` | page |
| `--surface` | `#ffffff` | `#1c1c1e` | cards, composer |
| `--surface-2 / -3` | `#f5f5f7` / `#e8e8ed` | `#232326` / `#2c2c2e` | grouped fills |
| `--fg` / `--fg-soft` | `#1d1d1f` / `#424245` | `#f5f5f7` / `#d1d1d6` | text |
| `--muted-fg` | `#636366` (5.6:1 on white) | `#a1a1a6` (6.9:1 on `#1c1c1e`) | secondary text, never below 4.5:1 |
| `--fill` (+ `-hover`, `-active`) | gray 12 / 18 / 24 % | gray 24 / 32 / 40 % | quiet control fill |
| `--separator` | 16 % gray | 55 % gray | hairlines (0.5 px) |
| `--primary` | `#a64b1a` | `#e08a4b` | copper accent; `--primary-soft` is its 14–16 % tint |
| `--material`, `--material-side`, `--material-thick` | translucent grays | translucent grays | sidebar, toolbar, popovers; solid under `prefers-reduced-transparency` |

Type: system stack (`-apple-system`, `SF Pro`, `Segoe UI Variable`, Inter), 14 px body, 13 px controls, 11 px minimum;
display text tightens (`-0.022em` at 30 px), small text opens (`+0.005em`). Radii: 8 controls, 10 inner surfaces, 14
cards, 22 sheets and the composer. Control height 30 px (26 small, 22 extra small). One chrome height (52 px) for the
sidebar brand row and the page header, one sidebar inset (12 px; text and icons 8 px further in).

Motion: critically damped (`cubic-bezier(0.25, 0.9, 0.3, 1)`, 140 / 220 / 320 ms) by default; a springy `linear()`
curve only for things that arrive (popovers, sheets); press feedback (`scale(0.97)`) starts on pointer down; everything
off under `prefers-reduced-motion`.

## Rules

1. One primary (filled) action per view. In a list only the first, recommended row keeps the filled button; the others
   are tinted. Everything else is a gray fill without a border.
2. Cards are grouped surfaces (a 0.5 px ring, no border box); rows inside are separated by straight hairlines.
3. The sidebar is one flat material. Selection is the accent tint in a rounded row; no bar outside the row.
4. Messages, composer and start screen share one 780 px column.
5. Every material has a fallback: `prefers-reduced-transparency` (solid), `prefers-contrast: more` (stronger lines and
   text), `forced-colors` (system outlines).
6. Numbers, not adjectives: contrast is checked on the real screens in both themes (a script walks every visible text
   node and composites translucent backgrounds); text under 4.5:1 (3:1 for large or bold text) is a bug.

## Checked screens

Chat start, conversation (markdown, table, code), Settings, Models, Code mode, welcome sheet, model popover; light and
dark; 1440 and 820 px; reduced transparency, reduced motion and more contrast emulated. Not checked on a real Mac or
with SF Pro installed (the review machine has neither): type metrics follow the stack, not a measurement of SF.
