# Stage-one release checklist

This is an operator guide for preparing an offline engine release from the full
repository. Run the checks for each new source snapshot; a historical receipt or
successful ZIP creation does not verify a later release.

## Engine contract

- [ ] Keep the public entry point `engine.resize.resize(source, target_canvas)`.
- [ ] Return the supplied v0.9 design schema with the original element IDs and
      count, copy, font families, asset references, colors, opacity, groups, and
      stacking order preserved.
- [ ] Fit output to the requested canvas, keep images proportional, and use the
      supplied fonts and original asset pixels.
- [ ] Produce output JSON and its actual native rendered PNG. Preserve the
      advanced sample's intentional missing-asset placeholder.
- [ ] Run without runtime downloads, API keys, or networking. Package all
      required model weights, assets, fonts, and notices at build time.
- [ ] Confirm that required trained artifacts are publicly accessible and match
      the packaged model identities, including any required base/adapter pins.

The authoritative requirements remain [RULES.md](../RULES.md),
[FAQ.md](../FAQ.md), and `chameleon_design_schema_v0.9.json`.

## Build and verify a release

From the repository root, with a Python 3.11+ virtual environment active and
Docker running in Linux-container mode:

```bash
python -m pip install -r requirements-dev.txt
python -m engine.main
python -m tools.verify_docker --out output/docker-release-check
python -m tools.package_submission --out output/submission/release-check
```

Choose fresh output directories for each verification and packaging invocation.
The Docker verifier builds Linux AMD64, runs all three supplied tasks with
networking disabled and input mounted read-only, and checks the JSON schema,
protected source fields, strict rotated bounds, PNG dimensions, native-render
pixel parity, and packaged file identities.

- [ ] Review the actual source/result PNGs as well as structural checks.
- [ ] Require successful exit codes and a receipt for this exact release's
      image, source bytes, dependencies, resources, and fit-policy weights.
- [ ] Record the generated ZIP and manifest hashes. A packaging receipt records
      archive integrity; Docker verification is a separate check.
- [ ] Extract the ZIP to a clean directory and repeat the offline verifier there.
      Install the host verifier dependency with
      `python -m pip install "jsonschema>=4.26,<5"` if needed, then run
      `python -m tools.verify_docker --out output/package-check`.
- [ ] Retain the release archive and its verification receipts together.

The package contains the offline engine, renderer, brand resources, default fit
policy, schema, three task JSONs, Docker files, run instructions, and host
verification utility. The full repository additionally contains Studio,
contributor tests, and model-development tools.

## Viewer requirement and current scope

The final brief assigns the offline engine to stage one and the full Adobe
Express add-on to the finale. The supplied FAQ also requires an unscored viewer
add-on at stage one. This difference remains unresolved; the current project
scope prioritizes the offline engine and defers Adobe hosting and live host
work at the user's request.

Keep `integrations/adobe-express/submission/chameleon-viewer-1.0.0.zip` available
separately. Its existence does not establish installation, in-host import,
downloads, or image placement. The local Studio is a development tool; Adobe
host verification remains a separate completion check. Retain the FAQ
requirement until organizers clarify it.
