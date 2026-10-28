# Working on this repo

Two people, one `main`, and a Docker image that gets graded. These rules keep
`main` always buildable and keep us from stepping on each other.

## Setup (once)

```bash
git clone https://github.com/rsingh135/chameleon.git
cd chameleon
make setup        # creates .venv, installs runtime + dev deps
make check        # lint + tests — should be green on a fresh clone
make run          # resize the 3 sample tasks -> output/
```

## Daily loop

1. **Start from fresh `main`.** Always, before you write a line.
   ```bash
   git switch main
   git pull --ff-only
   git switch -c <your-name>/<short-topic>     # e.g. ranveer/text-fitting
   ```
2. **Work in small steps.** Commit when one logical thing works, not at the end
   of the day. Run `make check` before each commit.
3. **Stay current.** If `main` moved while you worked, rebase rather than merge
   so history stays linear:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
4. **Push and open a PR.** `git push -u origin HEAD`, then `gh pr create` (or
   the GitHub UI). Fill in the template. CI must be green to merge.
5. **Squash-merge** on GitHub, then delete the branch. Back to step 1.

`main` is protected: no direct pushes, no force pushes, CI required. If you
need to change a rule, change it in the repo settings, don't route around it.

## Commits

- **Subject line ≤ 72 chars, imperative mood, no trailing period.**
  `Fit headline to two lines when target is narrower` — not `fixed stuff`.
- **Say why in the body** when the diff doesn't make it obvious. The diff shows
  *what* changed; the message is the only place *why* survives.
- **One concern per commit.** A refactor and a behaviour change are two commits.
  Formatting-only changes go alone so `git blame` stays useful.
- Don't commit generated files: `output/*`, `.venv/`, caches. `.gitignore`
  already covers these; if `git status` shows one, don't `git add -A` past it.

## Branches

- Name: `<you>/<topic>` — `ranveer/aspect-ratio`, `sam/ci-docker-cache`.
- Short-lived. A branch older than a few days is a branch about to conflict.
- Never commit directly on `main`, even for a "tiny" fix. Open a PR; it takes
  thirty seconds and CI catches the typo you didn't see.

## Tests

- `tests/test_resize.py` encodes the hard requirements from `RULES.md`
  (ids preserved, nothing off-canvas, font sizes ≥ 1, content untouched,
  image aspect ratios kept). It runs every sample task against a catalogue of
  real platform sizes.
- Tests marked `xfail(strict=True)` are requirements the baseline knowingly
  fails. When your `resize()` makes one pass, pytest will fail loudly —
  that's your cue to delete the marker. Treat the xfail list as the to-do list.
- New behaviour gets a test in the same PR. If a bug slips through, the fix
  PR adds the test that would have caught it.
- Install `requirements-dev.txt` before running tests. Typography and evaluation
  tests use the real renderer and bundled fonts; CI installs those dependencies.
  Tests must not fetch anything from the network at runtime. Visual checks are
  done by running `make run` and looking at `output/*_output.png`.

## Lint & format

`ruff` handles both. `make fmt` fixes what it can; `make lint` is what CI
runs. Config lives in `pyproject.toml`. `renderer/` is vendored upstream code
and is excluded — don't edit it to satisfy the linter.

## Pull requests

- Small. One PR per idea. If the description needs "also", split it.
- Describe how you verified it, not just what you did.
- Review each other's PRs when you can, but a green CI is the only hard gate —
  don't block a teammate overnight waiting for a rubber stamp.
- CI runs lint, tests, and a full `docker build`. The Docker job exists
  because that image *is* the submission; a passing test suite with a broken
  image is still a broken submission.

## Don'ts (from RULES.md — these disqualify, not deduct)

- Never regenerate, redraw, or replace asset pixels, including with generative AI.
- Don't rewrite text `content` or change `font_family` for stage one.
- Nothing may fetch from the network at run time; dependencies go in
  `requirements.txt` and install at Docker build time.
