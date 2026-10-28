# Eigen now comes from a KoliStat mirror, not from GitLab

Date: 2026-10-16. Trigger: CI failures on PR #63 (`lin_hyp`).

## What happened

The "C++ unit tests" job failed three runs in a row. The failure was always the
same step, `Check out Eigen`, and the job never reached the compiler:

```
fatal: remote error: GitLab is currently unable to handle this request due to load (ID a4af9699ba874c92-MSP).
fatal: clone of 'https://gitlab.com/libeigen/eigen.git' into submodule path '.../third_party/eigen' failed
Failed to clone 'third_party/eigen' a second time, aborting
```

Six clone attempts failed in total, counting git's own built-in retry. The
request IDs ended in three different GitLab points of presence: MSP, LAX and
ORD. At the same time gitlab.com served a developer machine normally. So this
was GitLab shedding load from GitHub Actions address ranges, not an outage and
not a problem with the branch.

## Why GitLab was in the picture at all

Eigen is a GitLab project. It has no maintained GitHub mirror: the former
official one, `eigenteam/eigen-git-mirror`, says "THIS MIRROR IS DEPRECATED" in
its own description, was last pushed in April 2022, and stops at tag 3.3.7.
`PX4/eigen` stops at 3.3.4. We pin 3.4.0, so neither carries our commit.

Eigen was the only submodule of the four not hosted on github.com. `duckdb`,
`extension-ci-tools` and `third_party/readstat` all are.

Two workflows fetch it:

- `.github/workflows/lint.yml` runs
  `git submodule update --init --depth 1 third_party/eigen` directly.
- `MainDistributionPipeline.yml` calls
  `duckdb/extension-ci-tools/.github/workflows/_extension_distribution.yml`,
  which checks out submodules recursively across its whole matrix: linux
  amd64 and arm64, osx amd64 and arm64, windows, and three wasm targets. That
  is roughly nine jobs cloning Eigen from GitLab per release run.

## Why the fix had to be in .gitmodules

A CI-side workaround can only reach the first of those two. The release
pipeline's checkout happens inside a reusable workflow we call but do not
control, so we cannot inject a `git config url.<x>.insteadOf` rule, an
`apt-get install libeigen3-dev`, or a retry wrapper into it. Only a change to
`.gitmodules` or to the tree itself fixes both.

## Options weighed

| Option | Covers release builds | Cost |
| --- | --- | --- |
| Mirror under KoliStat | yes | a 138 MB repo to own; one line changed |
| Vendor `Eigen/` into the repo | yes | 337 files, 6.9 MB committed; mixed-licence redistribution; large diffs on every bump |
| `apt-get install libeigen3-dev` | no | fixes one job; drifts from the 3.4.0 pin |
| `actions/cache` plus retry | partly | needs one good clone to seed; caches evict after 7 days unused |

Vendoring is a real option and was close. `src/linalg.cpp` is the only file
that includes Eigen, and it includes only `<Eigen/Dense>`, which pulls in
Core, LU, Cholesky, QR, SVD, Geometry and Eigenvalues. Vendoring therefore
means the whole `Eigen/` tree, not a few headers. It would also mean
redistributing a tree that ships `COPYING.APACHE`, `.BSD`, `.GPL`, `.LGPL`,
`.MINPACK` and `.MPL2`; the clean way to do that is to keep every `COPYING.*`
and build with `-DEIGEN_MPL2_ONLY`, which Eigen defines so that including any
LGPL header becomes a compile error. The one thing vendoring buys that the
mirror does not is a repository that builds straight after `git clone`, with
no submodule step.

The mirror won on two points. The recorded commit stays byte-identical to
upstream's, so provenance is verifiable by anyone. And bumping Eigen stays a
one-SHA diff instead of a 337-file one.

## What was done

- Created `github.com/KoliStat/eigen`, public, with issues, wiki and projects
  disabled so Eigen bugs are not filed there. The description points at
  upstream.
- Mirrored it with `git clone --mirror` from GitLab, then pushed only
  `refs/heads/*` and `refs/tags/*`. GitLab's `refs/merge-requests/*` and
  `refs/pipelines/*` were deliberately left behind. Default branch set to
  `master`, matching upstream.
- Changed one URL in `.gitmodules`. `path` and `shallow = true` are unchanged,
  and so is the recorded commit.

## How it was verified

- Tag `3.4.0` on the mirror resolves to `3147391d946bb4b6c68edd901f2add6ac1f31f8c`,
  the commit the parent repo already recorded.
- A shallow fetch of that bare SHA from GitHub succeeds, which is exactly what
  `git submodule update --init --depth 1` asks for. GitHub allows fetching any
  reachable SHA, so the depth-1 pin keeps working.
- The fetched commit's tree is `228ffa5204a0035e58a3e57798dcf23eda0134f2`, equal
  to the tree of the checkout that came from GitLab. The content is identical,
  not merely similar.

## What to watch

The mirror is now a dependency of every build, so it must stay public and must
not be deleted. It does not track upstream automatically, which is the right
default for a pinned submodule. Before bumping Eigen, refresh the mirror first:

```bash
git clone --mirror https://gitlab.com/libeigen/eigen.git eigen-mirror.git
cd eigen-mirror.git
git push https://github.com/KoliStat/eigen.git "refs/heads/*:refs/heads/*" "refs/tags/*:refs/tags/*"
```

Then update the submodule pin in this repository as usual.
