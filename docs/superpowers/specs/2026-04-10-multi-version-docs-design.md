# Multi-Version Documentation Build Redesign

**Date:** 2026-04-10
**Status:** Draft
**Author:** Antoine Richard

## Problem

IsaacLab uses `sphinx-multiversion==0.2.4` (unmaintained since 2020) to build
multi-version documentation. This tool installs a single Sphinx version and
builds all branches/tags against it, causing:

- **No dependency isolation.** All versions share one Sphinx install. When
  Sphinx 9.0 changed `Config.read()` to keyword-only parameters, every version
  broke simultaneously.
- **Slow CI.** Every push rebuilds every whitelisted branch and tag.
- **Fragile version coupling.** Upgrading Sphinx or the theme on one branch
  forces compatibility with every historical branch/tag.

Large projects (NumPy, CPython) solve this with per-version CI builds and a
JSON-driven version switcher. This spec describes migrating IsaacLab to that
pattern.

## Goals

1. Each branch/tag builds with its own `docs/requirements.txt` (full isolation).
2. Only the pushed branch rebuilds on each CI run (fast).
3. Old version HTML stays frozen on gh-pages; no rebuild unless the branch is
   updated.
4. A version switcher dropdown works on all versions, including old tags that
   predate this migration.
5. Remove the `sphinx-multiversion` dependency entirely.

## Non-Goals

- Changing the hosting target (stays on gh-pages).
- Pruning old versions from the published docs (can be done later).
- Changing the Sphinx theme.

## Architecture

### Build Model

Each push to a deploy branch (`main`, `develop`, `release/**`) or creation of a
version tag (`v*.*.*`) triggers a CI job that:

1. Checks out that single ref.
2. Installs dependencies from that ref's `docs/requirements.txt`.
3. Runs `sphinx-build` to produce HTML.
4. Deploys the output to a versioned subdirectory on gh-pages.

```
Push to develop       -> build -> deploy to gh-pages/develop/
Push to main          -> build -> deploy to gh-pages/main/
Push to release/2.3.0 -> build -> deploy to gh-pages/release-2.3.0/
Tag v3.1.0 created    -> build -> deploy to gh-pages/v3.1.0/
```

PRs build docs with `-W` (warnings as errors) for validation but do not deploy.

### Version Slug Mapping

| Git Ref                     | Slug              | URL Path                  |
|-----------------------------|-------------------|---------------------------|
| `refs/heads/main`           | `main`            | `/IsaacLab/main/`         |
| `refs/heads/develop`        | `develop`         | `/IsaacLab/develop/`      |
| `refs/heads/release/2.3.0`  | `release-2.3.0`   | `/IsaacLab/release-2.3.0/`|
| `refs/tags/v2.3.2`          | `v2.3.2`          | `/IsaacLab/v2.3.2/`      |

Slugs are derived by stripping `refs/heads/` or `refs/tags/` and replacing `/`
with `-` for release branches.

### Version Switcher

New versions (main, develop, future releases) use the `pydata-sphinx-theme`
built-in version switcher, which `sphinx-book-theme` inherits. It is configured
via `html_theme_options` in `conf.py` and driven by a `versions.json` file.

Old versions (tags/branches that predate this migration) get a standalone
JavaScript-based version switcher injected during the deploy step.

#### `versions.json`

Hosted at the root of gh-pages (`/IsaacLab/versions.json`). Updated
automatically by each deploy job (fetch current file, upsert entry, write back).

```json
[
  {"name": "main", "version": "main", "url": "/IsaacLab/main/"},
  {"name": "develop", "version": "develop", "url": "/IsaacLab/develop/"},
  {"name": "v2.3.2 (stable)", "version": "v2.3.2", "url": "/IsaacLab/v2.3.2/", "preferred": true},
  {"name": "v2.3.1", "version": "v2.3.1", "url": "/IsaacLab/v2.3.1/"}
]
```

#### `conf.py` Changes (New Versions)

```python
# Determine the version slug for the switcher.
# CI sets DOCS_VERSION_SLUG (e.g. "main", "develop", "v2.3.2").
# Locally, fall back to the semver from the VERSION file.
_version_slug = os.getenv("DOCS_VERSION_SLUG", version)

html_theme_options = {
    # ... existing options ...
    "switcher": {
        "json_url": "https://isaac-sim.github.io/IsaacLab/versions.json",
        "version_match": _version_slug,
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
}
```

The CI workflow sets `DOCS_VERSION_SLUG` to the computed slug (e.g. `main`,
`develop`, `release-2.3.0`, `v2.3.2`) so the switcher highlights the correct
entry. Local builds fall back to the semver from the `VERSION` file.

#### Injected Switcher (Old Versions)

A standalone `version-switcher.js` file hosted at `/IsaacLab/version-switcher.js`.
Maintained in the repo at `docs/_static/version-switcher.js` and deployed to
the gh-pages root on every deploy.

**Behavior:**

- Fetches `/IsaacLab/versions.json` on page load.
- Renders a floating dropdown (fixed position, top-right corner).
- Detects the current version from the URL path.
- Highlights the current version in the dropdown.
- On selection, navigates to the same relative page path under the new version,
  with fallback to `index.html` if the page does not exist in that version.
- Detects if the theme's built-in switcher already exists and skips rendering
  to avoid duplicates.
- Self-contained styling (injects its own `<style>` tag). Neutral dark/light
  appearance that does not clash with the old Sphinx themes.

**Injection mechanism:** During deploy, a script appends a `<script>` tag to
every HTML file in the version directory:

```bash
find "${SLUG}/" -name "*.html" -exec sed -i \
  's|</body>|<script src="/IsaacLab/version-switcher.js"></script></body>|' {} +
```

## CI Workflow

### `docs.yaml` (Regular Builds)

**Triggers:**

```yaml
on:
  push:
    branches:
      - main
      - develop
      - 'release/**'
    tags:
      - 'v[1-9]*.*.*'
  pull_request:
    types: [opened, synchronize, reopened]
```

**Job 1: `build-docs`** (all pushes and PRs)

1. Checkout code.
2. Setup Python 3.12.
3. `pip install -r docs/requirements.txt`.
4. `sphinx-build -W --keep-going docs docs/_build`.
5. Upload artifact.

**Job 2: `deploy-docs`** (deploy branches and tags only)

Condition: `github.repository == env.REPO_NAME` and ref is a deploy branch or
version tag.

1. Download build artifact into `site/<slug>/`.
2. Determine version slug from `github.ref`.
3. Inject `version-switcher.js` script tag into all HTML files in `site/<slug>/`.
4. Fetch current `versions.json` from gh-pages via GitHub Pages URL or the
   `gh-pages` branch. Upsert this version's entry. Write to `site/versions.json`.
5. Copy `docs/_static/version-switcher.js` to `site/version-switcher.js`.
6. Copy `docs/_redirect/index.html` to `site/index.html`.
7. Deploy `site/` to gh-pages using `peaceiris/actions-gh-pages` with
   `keep_files: true`. This preserves all other version subdirectories and only
   overwrites the current slug, `versions.json`, `version-switcher.js`, and
   `index.html`.

### `docs-migration.yaml` (One-Time Bootstrap)

Triggered manually via `workflow_dispatch`. Rebuilds all historical versions
and performs a full deploy.

**Job 1: `build-version`** (matrix strategy)

```yaml
strategy:
  matrix:
    ref:
      - main
      - develop
      - release/2.1.0
      - release/2.2.0
      - release/2.3.0
      - v1.0.0
      - v1.1.0
      - v1.2.0
      - v1.3.0
      - v1.4.0
      - v1.4.1
      - v2.0.0
      - v2.0.1
      - v2.0.2
      - v2.1.0
      - v2.1.1
      - v2.2.0
      - v2.2.1
      - v2.3.0
      - v2.3.1
      - v2.3.2
      - v3.0.0-beta
  fail-fast: false
```

Each matrix job:

1. Checks out `${{ matrix.ref }}`.
2. Installs from that ref's `docs/requirements.txt`.
3. Runs `sphinx-build` (without `-W` for old versions that may have warnings).
4. Uploads artifact named `docs-<slug>`.

**Job 2: `deploy-all`**

1. Downloads all artifacts.
2. Assembles directory structure (`main/`, `develop/`, `v2.3.2/`, etc.).
3. Injects `version-switcher.js` into all HTML files.
4. Generates `versions.json` from the matrix list.
5. Copies `index.html` redirect to root.
6. Copies `version-switcher.js` to root.
7. Deploys to gh-pages as a **full overwrite** (not `keep_files`).

## Files Changed

### Removed

| File | Reason |
|------|--------|
| `docs/_templates/versioning.html` | Replaced by theme switcher + injected JS |
| `sphinx-multiversion==0.2.4` in `docs/requirements.txt` | No longer needed |
| `sphinx_multiversion` in `conf.py` extensions | No longer needed |
| `smv_remote_whitelist`, `smv_branch_whitelist`, `smv_tag_whitelist` in `conf.py` | No longer needed |
| `"versioning.html"` in `html_sidebars` | Replaced by theme switcher |

### Added

| File | Purpose |
|------|---------|
| `docs/_static/version-switcher.js` | Standalone switcher injected into old versions |
| `.github/workflows/docs-migration.yaml` | One-time bootstrap workflow |

### Modified

| File | Changes |
|------|---------|
| `docs/conf.py` | Add `switcher` config to `html_theme_options`, add `navbar_end`, remove `smv_*` config, remove `sphinx_multiversion` extension |
| `docs/requirements.txt` | Remove `sphinx-multiversion==0.2.4` |
| `.github/workflows/docs.yaml` | Replace multi-version build with single-version build + deploy logic |
| `docs/Makefile` | Remove `multi-docs` target (or keep for backwards compat) |

## Rollback

If the migration causes issues:

1. The old gh-pages content is preserved in git history — `git revert` the
   deploy commit on gh-pages to restore the previous state.
2. Re-enable the old `docs.yaml` workflow with `sphinx-multiversion`.
3. Re-add `sphinx-multiversion` to `requirements.txt` and `conf.py`.

## Open Questions

None. All design decisions have been validated with the team.
