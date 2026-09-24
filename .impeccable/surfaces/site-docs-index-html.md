---
version: 1
slug: "site-docs-index-html"
primary_target: "site/docs/index.html"
related_targets: ["site/docs/guides.html","site/docs/recipes.html","site/docs/hooks.html","site/docs/integrations.html","site/docs/api.html","site/docs/reference.html"]
---

## Scope

`site/docs/*.html`, the documentation (Read). One page per tab: Start (`index.html`), Guides, Recipes, Hooks,
Integrations, API, Reference. `site/docs.html` becomes a redirect that keeps old `#anchors` working. Benchmarks and
research keep their single long pages but share the reading shell, so their contents move to the right rail too.

## Audience, job, action

Builders wiring Tez into their own code: install, first decision, then the recipe, hook, integration or reference
entry they came for. Proof: the measured numbers already on the site and in `BENCHMARKS.md`; code examples must run
against the shipped API.

## Constraints

Static GitHub Pages, no docs generator. Works without JavaScript (tabs and rail are plain links; scroll-spy and the
view transition are enhancements). Shared chrome is kept in sync by `scripts/build_docs.py`.

## Unresolved

Search across the docs pages (a later addition, not in this build).

## Direction contract

THESIS: Docs moved by job along a sticky tab strip, read in one column aligned under the tabs, beside a right-hand page map; refuses the generator-default left sidebar tree.

OWN-WORLD: The site's world: white and #F6F7F9, ink #0B0D10, hairlines, Geist; one blue #2F5BD8 marks the current tab's underline and rail entry.

STORY: A builder copies four commands on Start, gets a decision, then tabs straight to the recipe, hook, integration or reference they need.

FIRST VIEWPORT: Tabs Start, Guides, Recipes, Hooks, Integrations, API, Reference with Start underlined; H1 "Get started", lead, version pills, quickstart step one; rail "On this page" with Quickstart lit.

FORM: Section tabs, list position 2 of 6, seed key b32296d8. Signature: a tab switch slides the underline to the new tab while the page cross-fades.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
