# Releasing Tez

A release is a tag. Pushing `v<version>` runs `.github/workflows/release.yml`, which:

1. runs the whole CI workflow (`ci.yml`: tests on Ubuntu and Windows, Python 3.10 to 3.13, lint, the build check, the
   TypeScript client and the Docker image);
2. checks that the tag names the version in `tez/_version.py`, and that `pyproject.toml` still reads its version from
   there (`scripts/check_version.py`);
3. builds the sdist and the wheel, runs `twine check --strict`, and checks that the wheel holds only `tez/` and the
   licence files, with both console scripts (`tez`, `tez-mcp`) and every extra (`scripts/check_dist.py`);
4. publishes both to PyPI as `tez-decisions` with **trusted publishing**: no PyPI token is stored anywhere, GitHub's
   OIDC token is exchanged for a short-lived upload token;
5. creates the GitHub release with both files attached and generated notes (tags ending in `aN`, `bN` or `rcN` are
   marked as pre-releases).

## One-time setup

### 1. PyPI trusted publisher

The name `tez-decisions` does not exist on PyPI until the first upload, so register a **pending** publisher:

1. Sign in at <https://pypi.org>, open *Your account* → *Publishing*.
2. Under *Add a new pending publisher*, choose *GitHub* and fill in exactly:
   - PyPI project name: `tez-decisions`
   - Owner: `Jibalmi` (the account that owns the repository; `github.com/Jibalmi/Tez`)
   - Repository name: `Tez`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. Save. The first successful release creates the project and turns the pending publisher into a normal one. A pending
   publisher does not reserve the name: publish soon after registering it.

Once the project exists, publishers are managed under the project's *Manage* → *Publishing* page. If the repository is
renamed or transferred, update the publisher there, or the upload fails with `invalid-publisher`.

### 2. GitHub environment `pypi`

In the repository: *Settings* → *Environments* → *New environment* → `pypi`. Recommended:

- *Deployment branches and tags*: *Selected branches and tags*, add the tag pattern `v*`, so only release tags can
  deploy to it.
- *Required reviewers*: add a maintainer if every upload should wait for a click.

### 3. Optional: protect release tags

*Settings* → *Rules* → *Rulesets* → *New tag ruleset*, target `v*`, and restrict creation, update and deletion to
maintainers.

## Making a release

1. Choose the version and set it in `tez/_version.py`: `__version__` (and `RELEASE_DATE`, which `/v1/models` reports).
   That is the only place: `pyproject.toml` declares `dynamic = ["version"]` and reads it from there, and CI checks
   that it still does (`scripts/check_version.py`).
2. If the wire format changed, `docs/API.md`, `tests/test_wire.py` and the TypeScript types in `clients/ts/src/types.ts`
   change with it.
3. Commit, push to `main`, and wait for CI to pass. Commit messages carry no AI or assistant attribution (see
   [`AGENTS.md`](../AGENTS.md)).
4. Tag and push the tag:

   ```
   git tag -a v0.2.0 -m "Tez 0.2.0"
   git push origin v0.2.0
   ```

5. Watch the *Release* workflow. When it is green, check the result in a fresh environment:

   ```
   python -m venv /tmp/tez-check && /tmp/tez-check/bin/pip install tez-decisions==0.2.0
   /tmp/tez-check/bin/tez --version
   ```

## When something goes wrong

- **The tag does not match the version.** Nothing was built or published. Delete the tag (`git push --delete origin
  v0.2.0` and `git tag -d v0.2.0`), fix `tez/_version.py`, tag again.
- **PyPI refused the upload.** PyPI never accepts the same file twice, even after a deletion: release the next version
  (for example `0.2.1`) rather than re-tagging.
- **`invalid-publisher`.** The owner, repository, workflow file name or environment name on PyPI does not match this
  repository exactly.
- **A rehearsal first.** Register the same pending publisher on <https://test.pypi.org> and, on a branch, add
  `with: {repository-url: https://test.pypi.org/legacy/}` to the publish step.

## The TypeScript client

`clients/ts` (npm package `tez-client`) is versioned separately and is not published by this workflow. To publish it by
hand: bump `version` in `clients/ts/package.json`, then `cd clients/ts && npm ci && npm test && npm publish` (the
`prepack` script builds `dist/`).
