# Chameleon Design Viewer

The stage-one Adobe Express viewer displays the engine's real source and resized
PNGs, lists the output elements, and exports the exact design JSON and
full-resolution PNG. **Add PNG to Express** places that rendered image on the
current page using Adobe's stable UI SDK. The exported JSON keeps the original
elements; PNG placement does not create editable native text or shapes.

The production manifest has no experimental API declaration or document sandbox.
The separate [native editable development integration](README.native.md) remains
available through `manifest.native.json`; it is excluded from the submission ZIP.

## Build, verify, and package

Use Node.js 22 and Python 3.11 or newer:

```powershell
Set-Location 'D:\My Projects\chameleon\integrations\adobe-express'
npm ci --ignore-scripts
npm test
npm run package
```

The package command writes:

- `submission/chameleon-viewer-1.0.0.zip`: a verified, reproducible private-distribution archive.
- Its SHA-256 receipt, `icon-144.png`, and `release-notes.txt` for the developer console.

To include verified engine-result bundles beside the ZIP, run
`python scripts/package.py --results-dir ../../output/addon-results` after the
exporter below. These go in `submission/engine-results/` with SHA-256 receipts;
they do not change the Adobe ZIP's runtime files.

The ZIP contains a root `manifest.json`, stable viewer code, and the three
supplied task JSONs. It excludes the native sandbox, experimental manifest,
development adapter, dependencies, and source files. The package script verifies
its inventory and the 50 MiB private-upload limit.

## View offline engine results

This workflow does not need browser access to localhost:

1. Run the submitted engine in Docker with `--network none` as described in the
   repository README. Keep the produced JSON and PNG files.
2. From the repository root, create viewer bundles using the same supplied assets
   and real renderer:

   ```powershell
   .\.venv\Scripts\python.exe -m tools.package_addon_results --outputs output/docker-compliance/run-20261030-current/artifacts --docker-report output/docker-compliance/run-20261030-current/report.json --out output/addon-results
   ```

   Set `--outputs` to the directory containing that run's `<task>_output.json` and
   `<task>_output.png` files. The exporter checks the output contract, keeps those
   PNG bytes, and adds a native source render. The Docker report binds the
   imported artifacts to the verified offline run and its Linux renderer.
3. In the Express add-on, choose **Open engine result** and select a generated
   result bundle. Compare both renders, inspect elements, download JSON/PNG, or
   choose **Add PNG to Express**.

Bundles include schema version 1, source, target canvas, request identity, output
design, source/output PNG data, and diagnostic report. The viewer checks element
identity, canvas bounds, request/target agreement, and PNG dimensions. It decodes
both images before enabling exports. Changing source, target, or engine settings
invalidates exports and labels any retained preview as a previous result.

## Generate a new layout from the panel

Start `python -m viewer.server --port 8765` from the repository root. Choose a
bundled sample or open a task JSON, select a format or custom dimensions, and
choose **Generate resized design**. Task imports use the engine's supplied
`brand_kit/assets/` and `brand_kit/fonts/`. The intentional missing Level 3 asset
remains a labeled placeholder.

The panel calls `POST /api/addon/resize` with `{source, target_canvas, request_id}`.
The backend uses strict stage-one bounds and returns both real PNGs and design
JSON. Only explicit add-on origins receive CORS access. The default development
origin is `https://localhost:5241`; configure `CHAMELEON_ADDON_ORIGINS` with the
assigned production HTTPS origin when using a hosted private add-on.

Adobe Express's frame and browser local-network policy can prevent a hosted panel
from reaching a local engine. Allow local-network access if the browser offers
it. If access is unavailable, use **Open engine result**; viewing, downloading,
and stable PNG placement do not require a localhost fetch. The scored engine's
offline execution is independent of Adobe's web application and SDK.

## Load the add-on in Adobe Express

For development, supply an existing trusted localhost TLS certificate and key:

```powershell
$env:CHAMELEON_ADDON_TLS_CERT = 'C:\path\to\localhost-cert.pem'
$env:CHAMELEON_ADDON_TLS_KEY = 'C:\path\to\localhost-key.pem'
npm start
```

Connect `https://localhost:5241` through **Test your local add-on** in Express.
The server binds to loopback and does not create or trust certificates. For
standalone browser QA, `node scripts/serve.mjs --http` serves
`http://127.0.0.1:5242/ui/index.html`; HTTP mode is not an Express sideload origin.
PNG placement stays disabled until Adobe's SDK is ready inside the host.

For private distribution, create an add-on in Adobe's developer console, upload
the ZIP, supply the release notes and 144 × 144 icon, and use the generated
private link. The manifest's `testId` supports local testing; Adobe ignores it for
submitted add-ons. Uploading, accepting developer terms, and publishing are
separate account actions. A local build or SDK compilation alone does not prove
host execution or create a private installation link.

Before reporting host verification, exercise the actual panel in Express: open
an engine-result bundle, compare both renders, download and parse the JSON,
inspect the full PNG dimensions, add that PNG to the document, and check that the
original document remains unchanged. Also exercise changed dimensions, invalid
imports, cancellation, and recovery.

## Official API and distribution references

- [Stable UI SDK document APIs, including `addImage(Blob)`](https://developer.adobe.com/express/add-ons/docs/references/addonsdk/app-document)
- [Image placement guide](https://developer.adobe.com/express/add-ons/docs/guides/learn/how-to/use-images)
- [Manifest v2 and experimental API restrictions](https://developer.adobe.com/express/add-ons/docs/references/manifest/)
- [Private-distribution package and metadata requirements](https://developer.adobe.com/express/add-ons/docs/guides/build/distribute/private-dist)
- [Chrome local-network access and iframe delegation](https://developer.chrome.com/docs/modern-web-guidance/guides/security/local-network-access)

Local checks establish the tested code and package behavior; they are separate
from Adobe's installation, runtime permissions, and official competition scoring.
