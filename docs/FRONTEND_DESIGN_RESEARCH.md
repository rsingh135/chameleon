# Frontend workspace trial research

Researched October 29, 2026. This note concerns the local viewer interface only.
The engine's output composition, renderer, source assets, scores, and exports
remain a separate design-quality surface. These are trial directions for
comparison, not a claim that Adobe or Emil designed Chameleon.

## Current product and useful foundations

`viewer/index.html` is a dependency-free application, backed by the existing
local Python server and `viewer/app.js`. Its workflow is source selection or JSON
import, preset or custom dimensions, resize, original/result comparison, element
inspection, design review, and JSON/PNG export. It already has dark/light modes,
bundled Source Sans 3 fonts, keyboard-accessible native controls, local status,
and collapsible panels. `viewer/README.md` already describes a Spectrum-inspired
frame and Emil-inspired press feedback. The trials should refine this foundation
and compare information architecture, rather than add a framework or imitation
editing tools the product does not support.

## Primary-source findings

### Emil: responsiveness comes from restraint

Emil distinguishes purposeful feedback from decorative delay. Repeated actions
and keyboard navigation benefit from immediate responses; occasional transitions
should generally finish below 300 ms. A small press response can confirm input,
while delays on every selection interfere with work. Tooltips can delay initially
and then appear immediately when moving across adjacent tools.
[You Don't Need Animations](https://emilkowal.ski/ui/you-dont-need-animations).

Entrances should have a spatial origin related to their trigger. Emil recommends
strong custom easing curves and usually ease-out for UI entrances. He also
recommends reviewing motion slowly to find discontinuities that are hard to
notice at full speed.
[Good vs Great Animations](https://emilkowal.ski/ui/good-vs-great-animations).

His Sonner write-up explains why transitions can retarget during rapid changes,
whereas keyframe-driven changes can jump. It also emphasizes edge cases and
directly usable interactive examples.
[Building a toast component](https://emilkowal.ski/ui/building-a-toast-component).
The author's course provides further context, but this research uses his public
articles rather than inferring the contents of paid lessons.
[Animations on the Web](https://animations.dev/).

**Chameleon application, our inference:** preserve immediate preview and element
selection; use restrained transform/opacity feedback only where it communicates
an action or state change; respect reduced motion; keep feedback close to the
canvas instead of introducing a procession of notifications. More animation is
not the objective.

### Adobe: make the canvas the center of the editing workspace

Spectrum explicitly distinguishes editing contexts from browsing contexts. In
editing applications, the canvas is the focus. Neutral background layers
separate frame, pasteboard, and content. Elevated surfaces should be used
sparingly for floating controls and popovers, rather than applying shadows to
every panel.
[Spectrum: background layers](https://spectrum.adobe.com/foundations/color/background-layers).

Spectrum uses color to communicate hierarchy and meaning, avoids unnecessary
color, and pairs color feedback with text or icons. Contrast needs checking in
each appearance.
[Spectrum: colors](https://spectrum.adobe.com/page/color/).

The Spectrum app-frame guidance separates navigation, global actions, and
contextual actions. It is a flexible framework rather than a universal fixed
layout. Its currently documented anatomy primarily covers browsing contexts;
it should not be treated as a complete editing-app prescription.
[Spectrum: app frame](https://spectrum.adobe.com/page/application-frame/).

Photoshop's documented workspace includes application controls, tools, a
document window, an options bar, and grouped panels. Its contextual task bar
surfaces relevant actions for the current selection or task.
[Photoshop workspace overview](https://helpx.adobe.com/in/photoshop/desktop/get-started/learn-the-basics/workspace-overview.html).
The task bar is intended to reduce panel switching and keep focus on the canvas.
[Photoshop contextual task bar](https://helpx.adobe.com/photoshop/desktop/get-started/learn-the-basics/boost-workflows-with-the-contextual-task-bar.html).

**Chameleon application, our inference:** keep imports and target setup together,
put document exports together, distinguish preview controls from engine actions,
and reserve the strongest contrast for the artwork and the next useful action.
Use the existing Chameleon identity and local fonts. Do not copy Adobe branding
or imply unsupported pixel/layer editing capabilities.

## Three trial directions

The names below describe our design proposals, not Adobe product modes. Each
trial must use the same live source, real rendered previews, and export logic.
Differences should remain structural even when appearance is held constant.

| Direction | Structure and emphasis | Best fit | Tradeoff |
| --- | --- | --- | --- |
| A: Docked studio | Persistent source/canvas dock on the left, large central pasteboard, element/properties dock on the right; document actions above | Repeated resize and inspection, with a familiar professional editing workspace | Persistent docks consume space on smaller laptops |
| B: Review desk | Source/setup collected into a horizontal ribbon; source and result presented as peers; element/properties inspection in a bottom tray | Visual quality assessment and choosing adaptations | Setup consumes vertical space and two canvases reduce individual preview size |
| C: Focus canvas | Wide central artwork area; source/canvas setup dock on the right; element/properties inspection in a tray below the artwork | Fast one-design/one-output work and a larger individual preview | Inspection can fall below the fold |

**Recommendation:** start from A, Docked studio, for the main application. It maps to the
existing control groups with the least behavioral disruption, keeps export
identity and element inspection visible, and provides an Adobe-like professional
workspace without pretending to be a general-purpose image editor. Borrow the
Focus canvas layout at laptop widths and retain Review desk as an explicit comparison mode.
Evaluate real screenshots and user interaction before choosing a final layout.

## Review lens for the trials

| Before | After | Why |
| --- | --- | --- |
| Setup and result metadata compete for attention | A clear document/action region and quieter setup groups | The next action and current output become easier to locate |
| A collection of independent sections | A deliberate source/setup, artwork, inspection hierarchy | Supports the actual resize-and-check workflow |
| Dense information always visible | Intentional dock density and progressive disclosure | Creates room for the artwork without removing functionality |
| Motion added as decoration | Immediate repeated interactions and brief purposeful feedback | Applies Emil's responsiveness principles to a work tool |

## Acceptance checks

Compare all directions with the same source and target at laptop and narrow
widths. Exercise real source loading, preset and custom dimensions, import,
invalid input, resize, view switching, element-list selection, review disclosure,
and current-result exports. Source or dimension changes must invalidate stale
exports. Ensure controls have visible focus, status/error text remains announced,
reduced motion is respected, no desktop-only panel hides essential work, and the
trial switcher cannot capture arrow keys inside editable controls.

## Implemented preview and validation

Run the existing viewer server (`python -m viewer.server --port 8765`) and open
[the side-by-side comparison](http://127.0.0.1:8765/trial-comparison.html).
Live trial URLs use `/trial.html?variant=A`, `B`, or `C`. The default `/` studio
and `app.js` are unchanged by this experiment. Layout switching changes only
the frame and appearance; the same live document and export state remain.
The shell is regenerated with `python -m tools.build_frontend_trial`.

Validation completed October 29, 2026:

- Existing viewer server suite: **28 passed**. The initial sandboxed run could
  not open loopback client sockets; the permitted rerun passed.
- Browser checks: live sample loading, preset sizes, custom 800 × 600 resize,
  invalid width blocking, stale export invalidation, switching frames without
  losing input, original/compare/result views, element boxes, element selection
  and geometry, malformed pasted JSON feedback, and valid import recovery.
- Downloaded JSON parsed successfully with an 800 × 600 canvas. The exported PNG
  opened as an 800 × 600 image with rendered content.
- All three frames checked at laptop and narrow widths; 390 px outer viewport
  checks showed no horizontal page overflow. Review desk also checked at an
  intermediate 800 px viewport; a compact ribbon rule covers 761–900 px.
- Trial button and caption colors adjusted after contrast review. Existing
  status/error live regions, focus styles, hidden states, and reduced-motion
  rules remain active.
- Comparison screenshots use the same Level 2 source and 1600 × 400 output.
  Review desk opens paired previews; the other directions open the result.

The screenshots capture the existing engine's output and review notes. Selecting
a frontend direction does not resolve composition warnings in that output.
