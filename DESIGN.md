---
name: Tez
description: Open, local typed decisions from one forward pass, shown as calibrated probabilities.
colors:
  ink: "#0b0d10"
  paper: "#ffffff"
  paper-subtle: "#f6f7f9"
  paper-inset: "#eef0f3"
  hairline: "#e4e7ec"
  hairline-strong: "#cdd2da"
  ink-muted: "#5b6270"
  ink-faint: "#6b7280"
  signal-blue: "#2f5bd8"
  signal-blue-deep: "#2449b8"
  signal-blue-fill: "#3f6ff0"
  signal-blue-wash: "#eaf0fe"
  bar-rest: "#c9ced6"
  act-green: "#166534"
  act-wash: "#e8f5ec"
  escalate-amber: "#9a4a09"
  escalate-wash: "#fdf1e4"
  danger-red: "#b91c1c"
  danger-wash: "#fdecec"
  code-string: "#0e7c66"
  night-bg: "#0b0d10"
  night-subtle: "#12151a"
  night-inset: "#191d24"
  night-hairline: "#232833"
  night-ink: "#e8eaee"
  night-blue: "#7c9dff"
  night-blue-fill: "#6e93ff"
typography:
  display:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "clamp(2.5rem, 1.6rem + 2.3vw, 3.25rem)"
    fontWeight: 640
    lineHeight: 1.05
    letterSpacing: "-0.036em"
  headline:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.875rem"
    fontWeight: 600
    lineHeight: 1.18
    letterSpacing: "-0.025em"
  title:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.015em"
  body:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.6
    fontFeature: "\"ss01\" on, \"cv11\" on"
  lead:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 520
    lineHeight: 1.3
  mono:
    fontFamily: "Geist Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.6
  code:
    fontFamily: "Geist Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "13.5px"
    fontWeight: 400
    lineHeight: 1.65
rounded:
  sm: "6px"
  md: "8px"
  lg: "12px"
  xl: "16px"
  pill: "999px"
spacing:
  s-1: "4px"
  s-2: "8px"
  s-3: "12px"
  s-4: "16px"
  s-5: "24px"
  s-6: "32px"
  s-7: "48px"
  s-8: "64px"
  s-9: "96px"
components:
  button-primary:
    backgroundColor: "{colors.signal-blue}"
    textColor: "{colors.paper}"
    rounded: "{rounded.md}"
    padding: "0 18px"
    height: "42px"
    typography: "{typography.label}"
  button-primary-hover:
    backgroundColor: "{colors.signal-blue-deep}"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "0 18px"
    height: "42px"
  button-secondary-hover:
    backgroundColor: "{colors.paper-subtle}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink-muted}"
    rounded: "{rounded.md}"
    padding: "0 18px"
    height: "42px"
  button-small:
    height: "34px"
    padding: "0 12px"
  pill:
    backgroundColor: "{colors.paper-inset}"
    textColor: "{colors.ink-muted}"
    rounded: "{rounded.pill}"
    padding: "0 10px"
    height: "24px"
  pill-accent:
    backgroundColor: "{colors.signal-blue-wash}"
    textColor: "{colors.signal-blue}"
  pill-act:
    backgroundColor: "{colors.act-wash}"
    textColor: "{colors.act-green}"
  pill-escalate:
    backgroundColor: "{colors.escalate-wash}"
    textColor: "{colors.escalate-amber}"
  card:
    backgroundColor: "{colors.paper}"
    rounded: "{rounded.lg}"
  panel:
    backgroundColor: "{colors.paper}"
    rounded: "{rounded.xl}"
  code-block:
    backgroundColor: "{colors.paper-subtle}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "16px 24px"
    typography: "{typography.code}"
  input-text:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "8px 12px"
    height: "40px"
  segmented:
    backgroundColor: "{colors.paper-inset}"
    rounded: "{rounded.md}"
    padding: "3px"
  segmented-option-on:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    height: "30px"
  prob-bar-track:
    backgroundColor: "{colors.paper-inset}"
    rounded: "{rounded.pill}"
    height: "8px"
  prob-bar-fill-chosen:
    backgroundColor: "{colors.signal-blue-fill}"
  prob-bar-fill-rest:
    backgroundColor: "{colors.bar-rest}"
  nav-link:
    textColor: "{colors.ink-muted}"
    rounded: "{rounded.sm}"
    padding: "6px 10px"
  nav-link-current:
    backgroundColor: "{colors.paper-inset}"
    textColor: "{colors.ink}"
  docs-tab:
    textColor: "{colors.ink-muted}"
    padding: "0 12px"
    height: "48px"
    typography: "{typography.label}"
  docs-tab-current:
    textColor: "{colors.ink}"
  rail-link:
    textColor: "{colors.ink-muted}"
    rounded: "{rounded.sm}"
    padding: "5px 10px"
  rail-link-current:
    backgroundColor: "{colors.signal-blue-wash}"
    textColor: "{colors.signal-blue}"
  pager-link:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "14px 16px"
---

# Design System: Tez

## Overview

**Creative North Star: "The Instrument Readout"**

Tez presents itself the way a well-made measuring instrument does: white paper, near-black ink, hairline rules, and one blue that marks the thing that was chosen. The system is built so that numbers carry the page. Every result is a row of probability bars with a mono value at the end, and the page's visual interest comes from those readouts updating, not from decoration. It sits deliberately inside the category standard for open local-model tools (the register of Ollama, LM Studio and Hugging Face) rather than inventing a private idiom: Geist, pill tags, bordered code blocks, a sticky hairline header.

Density is moderate and tool-like. Sections are separated by a single hairline and generous block padding; inside a section, content sits in a 5/7 split of copy against evidence (a code block, a table, a panel of bars). The only lifted object on a page is the product panel; everything else is flat and bordered. Motion is small and functional: bars ease their fill in 120 ms, words highlight as a recorded run replays, and reduced motion always lands on the final state.

Light is the primary theme. Dark follows the operating system automatically, and a visitor's explicit choice is stored and wins over the system setting. Both themes use the same token names, so no surface defines colours outside the token set.

**Key Characteristics:**
- Ink on white with two quiet grey surface steps; hairline borders instead of fills to separate things.
- One blue, in two steps: a text-and-button blue and a slightly brighter bar blue.
- Probability rows are the recurring motif on every page that shows a result.
- Geist for everything readable, Geist Mono for every number, identifier and command.
- Flat by default; one large soft shadow reserved for the product panel.
- Decision states (act, escalate, danger) have their own hue pairs and never borrow the blue.

## Colors

A neutral, cool-grey palette with a single saturated blue and three semantic state hues, each paired with a pale wash.

### Primary
- **Signal Blue** (signal-blue): links, primary buttons, focus outlines, the selected tab underline, the check on a chosen row, and the accent pill. The text-weight step of the blue, dark enough to read on white.
- **Signal Blue Deep** (signal-blue-deep): hover state of links and primary buttons only.
- **Signal Blue Fill** (signal-blue-fill): large fills only: the chosen probability bar, the Tez row in comparison ladders, the scrubber progress, the brand mark's centre, and the Tez series in charts. Slightly brighter than Signal Blue because it is read as area, not as text.
- **Signal Blue Wash** (signal-blue-wash): the highlighted current word during replay, the final-answer strip, text selection, and the accent pill background.

### Tertiary (decision states)
- **Act Green** with **Act Wash**: the "act" outcome and the "Live" origin label; also the ok tone in status lines.
- **Escalate Amber** with **Escalate Wash**: the "escalate" or abstain outcome and the "Recorded" origin label; also the warn tone.
- **Danger Red** with **Danger Wash**: invalid fields and error status only.
- **Code String Teal** (code-string): string literals in syntax-coloured code. Numbers in code use Signal Blue.

### Neutral
- **Ink** (ink): headings, body text, table values, chosen-row labels.
- **Paper** (paper): page background, cards, panels, inputs, the "on" option of a segmented control.
- **Paper Subtle** (paper-subtle): footer, code blocks, notes, the product panel's toolbar, message boxes, hover fills on rows and ghost buttons.
- **Paper Inset** (paper-inset): pill backgrounds, segmented-control troughs, inline code, the current nav link, and the empty track of every bar.
- **Hairline** (hairline) and **Hairline Strong** (hairline-strong): borders and dividers; the strong step for secondary-button and input strokes and scrollbar thumbs.
- **Ink Muted** (ink-muted): secondary copy, nav links, table headers, bar labels and values at rest.
- **Ink Faint** (ink-faint): source lines, placeholders, separators, comments in code.
- **Bar Rest** (bar-rest): the fill of every probability bar that was not chosen.
- **Night** set (night-bg, night-subtle, night-inset, night-hairline, night-ink, night-blue, night-blue-fill): the dark theme's values for the same roles. The dark blue steps are lighter, and the button text flips to Ink.

### Named Rules
**The One Blue Rule.** Blue means "chosen" or "do this": the chosen option, the primary action, a link, focus. Nothing else on the page is blue, and no second accent hue exists. If a thing is not chosen and not clickable, it is grey.

**The Two-Step Blue Rule.** Text, buttons and outlines use Signal Blue; bars, fills and marks use Signal Blue Fill. Never swap them.

**The States Keep Their Own Hues Rule.** Act is green, escalate is amber, errors are red, each with its own wash. A decision state never renders in blue, and blue never signals a state.

## Typography

**Display Font:** Geist (with ui-sans-serif, system-ui fallbacks), self-hosted variable woff2
**Body Font:** Geist, with stylistic sets ss01 and cv11 on
**Label/Mono Font:** Geist Mono (with ui-monospace, SFMono-Regular, Menlo, Consolas)

**Character:** The category's own typeface, kept on purpose so the product reads as a peer of the tools builders already use. Sans for sentences, mono for anything a machine produced or will consume.

### Hierarchy
- **Display** (640, fluid 2.5 to 3.25rem, 1.05, tight negative tracking): the landing hero headline only; fixed at 2.25rem on phones.
- **Headline** (600, 1.875rem, 1.18): section headings, one per section, balanced wrapping. Page titles use 2.375rem at the same weight.
- **Title** (600, 1.125 to 1.25rem, 1.3): step titles, figure and panel headings, use-case row titles.
- **Body** (400, 1rem, 1.6): running copy, pretty wrapping, measure capped near 70ch; leads are 1.125rem in Ink Muted, capped at 60ch.
- **Label** (520 to 560, 0.875rem): buttons, tabs, form labels, segmented options, chosen-row labels. No uppercase anywhere; tracking stays normal.
- **Mono** (400, 0.75rem, tabular): probability values, milliseconds, counts, table numbers, code-block headers.
- **Code** (400, 13.5px, 1.65; 12.5px inside dense panes): code blocks and commands, ligatures off.

### Named Rules
**The Mono Numbers Rule.** Every measured number (probability, latency, accuracy, count) is Geist Mono with tabular figures, right-aligned in tables. A number in the sans face is prose, not a measurement.

**The Sentence Case Rule.** Headings, labels, pills and buttons are sentence case at normal tracking. There are no uppercase labels and no small overline text above headings.

## Layout

A 1200px container with a fluid gutter (16 to 32px) and a 64px sticky header. Sections have fluid block padding (56 to 104px) and are divided by a single hairline. A section opens with a heading block capped at 720px, then a two-column body split 5:7, copy on the left and evidence (code, table, bars, panel) on the right, with 48px row and 56px column gaps. The landing hero uses a 12-column grid with copy in five columns and the product panel in seven.

### Reading pages
The docs, benchmarks and research pages share one reading shell instead of the landing page's 5:7 split: a single reading column (capped at 780px on the docs, full width on the benchmarks and research pages) beside a 224px page-map rail on the right, 64px apart. The docs are split by job into seven pages (Start, Guides, Recipes, Hooks, Integrations, API, Reference), reached from a 48px section tab strip that sticks directly under the 64px header; the reading column starts at the tab strip's left edge, so text sits under the tabs. The rail sticks under all sticky chrome and scrolls on its own when it is taller than the viewport. Each docs page ends with a previous/next pager. Benchmarks and research use the same shell and rail with no tab strip. Sections inside a reading page are divided by a hairline with 48px above the heading and 64px between sections; running prose is 1rem at 1.65.

Spacing follows a 4px base scale (4, 8, 12, 16, 24, 32, 48, 64, 96). Component interiors use 8 to 24px; section-level separations use 32 to 64px.

Responsive behaviour, as built:
- **Below 1080px** the hero stacks to one column; the copy caps at 640px.
- **Below 960px** every 5:7 split and the use-case grid collapse to one column. On the reading pages the tab strip (44px here) stops sticking and scrolls away with the page, and the rail becomes a 48px disclosure that sticks under the header, naming the current section; it opens to a panel of at most 70% of the viewport and closes on a link, Escape or an outside tap. When the tab strip is wider than the screen it scrolls sideways with no scrollbar, the current tab is centred on load, and the side that has more tabs fades out over 40px.
- **Below 860px** the nav collapses behind a toggle into a full-width dropdown sheet.
- **Below 720px** the hero reorders so the product panel comes before the install block; the head-to-head table reflows into stacked rows with each cell carrying its own column label; command blocks wrap, but only between whole tokens, with continuation lines hanging under the prompt; the footer goes to two columns.
- **Below 560px** every table in a reading column reflows into stacked rows: the header row is visually hidden, each row becomes a block with 12px by 14px padding and a hairline under it, and each cell sits under its own column label (sans, 12px, weight 520, Ink Muted), wrapping anywhere rather than scrolling. A reference row keeps a Paper Subtle fill and a highlighted row a Signal Blue Wash fill across all its stacked cells. The pager and definition lists go to one column at the same width.
- **Below 480px** probability rows put the label on its own line above the bar and value, except inside the hero panel, where rows stay on one line and a long label wraps within its column.
- Any code block that scrolls sideways fades out at its right edge (40px mask) until scrolled to the end.
- Inline code never breaks inside a hyphenated token: a flag such as `--n-ctx` stays on one line, and paths may still break after each slash.

## Elevation & Depth

Flat by default, with borders doing the separating. Surfaces are distinguished by hairlines and by the three paper steps, not by shadow. Shadows exist in three sizes with a cool, low-opacity tint; dark theme uses deeper black shadows under the same names.

### Shadow Vocabulary
- **Small** (`0 1px 2px rgba(16, 24, 40, 0.06)`): primary and secondary buttons, the "on" segment of a segmented control, the scrubber thumb.
- **Medium** (two-layer, 4px/12px and 2px/4px): transient overlays only: the mobile nav sheet, chart tooltips, the skip link, floating research elements.
- **Large** (two-layer, 24px/48px and 4px/12px): the product panel alone.
- **Focus ring** (3px at 32% Signal Blue): text inputs and search fields on focus; other controls use a 2px Signal Blue outline offset by 2px.

### Named Rules
**The One Lifted Object Rule.** Only the product panel carries the large shadow. Cards, tables, code blocks and notes are flat with a hairline border.

## Shapes

Gently rounded, never soft. Corners step with the size of the object: 6px for small controls (nav links, icon buttons, segment options), 8px for buttons, inputs, message boxes and segmented troughs, 12px for cards, code blocks, tables and notes, 16px for the product panel and playground cards. Pills and every bar are fully rounded. Borders are always 1px hairlines; the only 2px strokes are the selected-tab underline (which on the docs tab strip has 2px rounded top corners and sits over the strip's bottom hairline) and the focus outline. Bars are 8px tall with a rounded track; the fill is revealed by a rounded clip, so the bar never changes width, only how much of it shows.

## Components

### Buttons
Compact and confident, the category's standard.
- **Shape:** 8px radius, 42px tall (34px small), 18px side padding, label weight 560, optional 16px leading icon.
- **Primary:** Signal Blue with white label and a small shadow; hover deepens to Signal Blue Deep.
- **Secondary:** Paper with an Ink label and a strong hairline stroke; hover fills with Paper Subtle.
- **Ghost:** transparent, Ink Muted; hover fills Paper Subtle and inks the label.
- **Active / Disabled:** press nudges down 1px; disabled is 50% opacity. Transitions are 120 ms.

### Chips (pills)
- **Style:** 24px tall, fully rounded, 12px text at weight 520, Paper Inset with Ink Muted by default; optional 12px icon.
- **Variants:** accent (blue wash), act (green wash), escalate (amber wash), outline (hairline, no fill). Origin labels ("Recorded", "Live") are pills whose icon takes the state colour while the text stays Ink. A smaller outlined proxy badge (20px) flags evidence measured on a related benchmark.

### Cards / Containers
- **Corner Style:** 12px for cards and notes, 16px for the product panel.
- **Background:** Paper; notes and toolbars use Paper Subtle.
- **Shadow Strategy:** none, except the product panel (see Elevation).
- **Border:** 1px hairline.
- **Internal Padding:** 16 to 24px.

### Inputs / Fields
- **Style:** 40px minimum height, 8px radius, strong hairline stroke on Paper, 14px text; labels above at weight 560; selects carry an inline chevron.
- **Focus:** border turns Signal Blue with the 3px focus ring.
- **Error / Disabled:** Danger Red border and a red-tinted ring; disabled fills Paper Subtle with muted text.

### Navigation
- **Style:** sticky 64px header on Paper with a bottom hairline; brand mark and wordmark left, links in Ink Muted at 14px, 6px radius, actions pushed right (theme toggle, GitHub).
- **States:** hover fills Paper Subtle and inks the label; the current page sits on Paper Inset in Ink.
- **Mobile:** below 860px a toggle opens a full-width sheet with 16px links and the medium shadow.

### Docs Section Tabs
A text tab strip under the header, one tab per docs page, on Paper with a bottom hairline.
- **Style:** 48px tall (44px below 960px), 14px labels at weight 520, 12px side padding, 2px gaps; tabs are plain links, so they work without JavaScript.
- **States:** Ink Muted at rest, Ink on hover; the current page is Ink with a 2px Signal Blue underline inset 12px from each side. Focus uses the standard outline, inset 4px.
- **Motion:** moving between docs pages, the underline slides from the old tab to the new one over 320 ms on the shared ease-out while the page cross-fades over 200 ms; the header and tab strip stay still. Browsers without cross-document view transitions just load the page, and reduced motion turns the whole transition off.

### Page Map Rail
"On this page", built from the page's h2 and h3 headings.
- **Style:** group headings 12.5px at weight 600 in Ink; links 13.5px (13px and further indented for h3 entries) in Ink Muted, 5px by 10px padding, 6px radius, 1px gaps. An "Edit this page" link with a 14px pencil icon closes the rail.
- **States:** hover fills Paper Subtle and inks the label. Scroll-spy marks the section being read in Signal Blue on Signal Blue Wash at weight 540, keeps it in view inside the rail, and at the very bottom of the page picks the last section on screen.
- **Mobile:** see Layout: a sticky disclosure naming the current section, with a chevron that turns over when it opens.

### Pager
Previous and next page at the foot of every docs page, below a hairline.
- **Style:** two equal bordered cards (12px radius, 14px by 16px padding), title at weight 600 with a 16px Signal Blue arrow, a one-line 14px Ink Muted description; the next card aligns right.
- **Hover:** the border strengthens, the card fills Paper Subtle, and the arrow nudges 2px in its direction.

### Tabs and Segmented Controls
- **Tabs:** text tabs on a hairline, 14px at 520, selected tab gets a 2px Signal Blue underline and Ink label. The docs section tabs are the page-level form of the same tab (see above).
- **Segmented:** a Paper Inset trough with 3px padding; the pressed option is a Paper chip with the small shadow. Counts inside options are mono.

### Code Blocks
- **Style:** Paper Subtle, hairline border, 12px radius, a mono 12px header with the title and a quiet copy button. Syntax colour is restrained: keys in Ink, strings teal, numbers blue, punctuation and comments muted.
- **Commands:** one command per line with a non-selectable prompt; long lines scroll on wide screens and wrap between whole tokens on phones.
- **Inline code:** on Paper Inset in running text; hyphenated tokens (flags, model names) never break, so a flag is always read whole.

### Probability Row (signature)
The unit of the whole system: a label, an 8px rounded bar, and a mono percentage, in a three-column grid.
- **At rest:** muted label, Bar Rest fill, muted value.
- **Chosen:** the label inks and goes to weight 560 with a blue check, the fill turns Signal Blue Fill, the value inks.
- **Motion:** the fill is revealed by a rounded clip driven by a percentage custom property, easing over 120 ms; only the chosen row turns blue.
- **Grouping:** rows stack with 8px gaps; the remaining options fold behind a small disclosure.
- **Comparison ladders** reuse the same bar with the label and value above it, the Tez row in blue and all others at rest.

### Product Panel (hero)
A lifted 16px panel that replays recorded runs. A Paper Subtle toolbar holds a segmented picker of four examples (Support, Security, Voice, Injection; Support opens first) and play, pause and restart icon buttons beside a "Recorded" pill. Tabs switch between the decision view and JSON, Python and curl panes. The decision view shows the message revealing word by word at 280 ms per word, the current word on a blue wash, a scrubber with a blue progress track, the live probability rows, a milliseconds footer, and a final-answer strip that turns blue-washed when the run ends. Under reduced motion the panel shows each run's final state and the scrubber still steps through the words.

## Do's and Don'ts

### Do:
- **Do** mark exactly one thing as chosen per question: one blue bar, one checked label.
- **Do** use Signal Blue for text and buttons and Signal Blue Fill for bars and marks, in both themes via the same tokens.
- **Do** set every measured number in Geist Mono with tabular figures, and put a source line (12px, Ink Faint) under any figure or table of results.
- **Do** separate sections with a single hairline and generous padding, and split section bodies 5:7 copy against evidence.
- **Do** keep bar motion at 120 ms on the shared ease-out curve, and land on the final state under reduced motion.
- **Do** give every sideways-scrolling code block and tab strip the edge fade, and let tables reflow into labelled cells on phones rather than scroll: on the reading pages below 560px, with each cell labelled in the sans face and a highlighted row keeping one fill across its cells.
- **Do** put reading-page navigation in the sticky tab strip (between pages) and the right-hand rail (within a page), with the reading column aligned under the tabs.
- **Do** ship light first, follow the system into dark, and respect a stored theme choice.

### Don't:
- **Don't** introduce a second accent hue or use blue for decision states; act, escalate and danger keep their own colours.
- **Don't** lift anything except the product panel; cards, tables and code blocks stay flat with hairlines.
- **Don't** set labels, pills or headings in uppercase or add small overline text above headings.
- **Don't** use glows, gradient fills or gradient text; the only gradients in the build are the scrubber's progress track and the scroll-edge masks on code blocks and the docs tab strip.
- **Don't** break a command, or a flag in inline code, inside a flag or token when it wraps.
- **Don't** give the reading pages a left-hand sidebar tree; page-to-page navigation is the tab strip and in-page navigation is the right rail.
- **Don't** hard-code colours outside the token set; figures that must sit on white (the research PNGs) are framed as paper deliberately.
