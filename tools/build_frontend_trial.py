"""Refresh the opt-in trial shell from the current viewer (no engine changes).

Run: python -m tools.build_frontend_trial
The checked-in trial.html is directly served by the usual local viewer server.
"""

from pathlib import Path


def build() -> None:
    viewer = Path(__file__).resolve().parents[1] / "viewer"
    page = (viewer / "index.html").read_text(encoding="utf-8")
    page = page.replace(
        '<html lang="en" data-theme="dark">',
        '<html lang="en" data-theme="dark" data-variant="A">',
    )
    page = page.replace(
        '<link rel="stylesheet" href="/styles.css">',
        '<link rel="stylesheet" href="/styles.css">\n'
        '<link rel="stylesheet" href="/trial.css">\n'
        '<script>const trialVariant=new URLSearchParams(location.search).get("variant");'
        'document.documentElement.dataset.variant=["A","B","C"].includes(trialVariant)'
        '?trialVariant:"A";document.documentElement.dataset.theme='
        'trialVariant==="B"?"light":"dark";</script>',
    )
    page = page.replace(
        '<script src="/app.js" defer></script>',
        '<script src="/app.js" defer></script>\n<script src="/trial.js" defer></script>',
    )
    page = page.replace("<body>", '<body class="trial-ui">')
    page = page.replace(
        '<span class="app-label">Design studio</span>',
        '<span class="trial-context">Design studio <span aria-hidden="true">/</span> '
        '<span id="trialSourceName">Preparing design</span></span>',
    )
    page = page.replace(
        '<div class="header-right">',
        '<div class="header-right"><span class="trial-local-badge">Local workspace</span>',
        1,
    )
    page = page.replace(
        '<main id="workspace" class="workspace">',
        "<!-- THROWAWAY UI TRIAL: three frames around the same live app.js workflow. -->\n"
        '<main id="workspace" class="workspace">\n'
        '<nav class="trial-rail" aria-label="Workspace shortcuts">'
        '<button class="trial-tool" data-trial-action="source" type="button" '
        'aria-label="Choose source design" title="Choose source design">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="16" '
        'height="16" rx="2"/><path d="M4 15l5-5 4 4 3-3 4 4"/><circle cx="15" '
        'cy="8" r="1"/></svg></button>'
        '<button class="trial-tool" data-trial-action="canvas" type="button" '
        'aria-label="Choose target canvas" title="Choose target canvas">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3v14a2 2 0 002 2h12'
        'M3 7h14a2 2 0 012 2v12"/></svg></button>'
        '<button class="trial-tool" data-trial-action="compare" type="button" '
        'aria-label="Compare original and result" title="Compare original and result">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="7" '
        'height="14" rx="1"/><rect x="14" y="5" width="7" height="14" rx="1"/>'
        "</svg></button>"
        '<button class="trial-tool" data-trial-action="inspect" type="button" '
        'aria-label="Inspect design elements" title="Inspect design elements">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7l8-4 8 4-8 4z'
        'M4 12l8 4 8-4M4 17l8 4 8-4"/></svg></button>'
        '<a class="trial-tool trial-rail-bottom" href="/trial-comparison.html" '
        'aria-label="Compare workspace trials" title="Compare workspace trials">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="18" '
        'height="18" rx="2"/><path d="M3 10h18M10 10v11"/></svg></a></nav>',
    )
    page = page.replace("<h1>Canvas setup</h1>", "<h1>Design setup</h1>")
    page = page.replace("<p>One design. Every format.</p>", "<p>Source and target canvas</p>")
    page = page.replace(
        '<h2 id="canvasHeading">Canvas</h2>', '<h2 id="canvasHeading">Target canvas</h2>'
    )
    page = page.replace(
        "<p>Composition workspace</p>",
        '<p id="trialWorkspaceLabel" class="trial-workspace-label">Docked studio workspace</p>',
    )
    page = page.replace(
        "<h3>Resized</h3>",
        "<h3>Result</h3>",
    )
    page = page.replace(
        '<details class="inspector-section" open>',
        '<div class="trial-inspector-heading">Inspection</div>'
        '<details class="inspector-section" open>',
        1,
    )
    page = page.replace(
        "<span>Adobe University Hackathon</span>",
        "<span>Chameleon · Local design engine</span>",
    )
    page = page.replace(
        "</body>",
        '<nav class="trial-switcher" aria-label="Frontend trial options">'
        '<a href="/trial-comparison.html" class="trial-switcher-label">UI trials</a>'
        '<button id="trialPrevious" type="button" aria-label="Previous workspace trial">←</button>'
        '<div class="trial-switcher-copy" aria-live="polite">'
        '<strong id="trialName">Docked studio</strong>'
        '<span id="trialDescription">Setup · canvas · inspector</span></div>'
        '<div class="trial-variant-buttons" role="group" aria-label="Choose workspace trial">'
        '<button type="button" data-trial-variant="A" aria-pressed="true" '
        'title="Docked studio">A</button>'
        '<button type="button" data-trial-variant="B" aria-pressed="false" '
        'title="Review desk">B</button>'
        '<button type="button" data-trial-variant="C" aria-pressed="false" '
        'title="Focus canvas">C</button>'
        '</div><button id="trialNext" type="button" aria-label="Next workspace trial">→</button>'
        '<a href="/" class="trial-original">Current GUI ↗</a></nav>\n</body>',
    )
    (viewer / "trial.html").write_text(page, encoding="utf-8")
    print("Refreshed viewer/trial.html from viewer/index.html")


if __name__ == "__main__":
    build()
