# PUBLISH.md — how to publish this repository

This repo has been sanitized and validated for public release, and a **clean
local root commit** has been made (no remote, nothing pushed). Publishing is
**not automated** — it requires your GitHub credentials and an explicit
decision. Follow the steps below when you're ready.

## 0. Pre-flight (do this once, on the machine that will push)

```bash
cd aps-rag-public

# (a) Confirm you're on a single clean root commit with no remote yet:
git log --oneline           # expect one commit
git remote -v               # expect no output

# (b) Recommended: run a secret scanner over the working tree before publishing
#     (it was not available on the build host).
gitleaks detect --no-git --source .        # or: trufflehog filesystem .

# (c) Confirm the process artifacts are NOT tracked (they name the origin /
#     list removed PII and are gitignored on purpose):
git ls-files | grep -E 'MIGRATION_PLAN|SANITIZATION_REPORT' || echo "OK — not tracked"

# (d) Sanity: confirm no user data / secrets / models are tracked:
git ls-files | grep -E '(^|/)\.env$|config/config\.yaml|^data/raw/(?!\.gitkeep|README)|^models/' || echo "OK — none tracked"
```

## 1. Create the GitHub repo and push

Using the GitHub CLI (recommended):

```bash
# Create a new public repo under your account/org and push the current branch.
gh repo create <owner>/aps-rag --public --source . --remote origin --push
```

Or manually:

```bash
git branch -M main
git remote add origin git@github.com:<owner>/aps-rag.git
git push -u origin main
```

## 2. Post-publish checklist

- [ ] Fill in `CITATION.cff` (authors, DOI, repo URL) and the citation stub in `README.md`.
- [ ] Confirm the `LICENSE` copyright line and the non-commercial reranker note.
- [ ] Optionally add a GitHub Actions CI (lint + `pytest`) — note some legacy
      unit tests assume the original fixed-collection layout and may need updating
      for the folder-per-database model.
- [ ] Consider enabling GitHub secret scanning + Dependabot on the new repo.

## Notes

- **Process artifacts** `MIGRATION_PLAN.md` and `SANITIZATION_REPORT.md` remain on
  disk for your review but are gitignored. If you want to publish a record of the
  sanitization, add a **redacted** version that does not name the origin
  institution or list the removed names/hosts.
- **`.env`, `config/config.yaml`, `data/raw/*`, and `models/*`** are gitignored —
  users create their own. Only the `.example` templates and `.gitkeep`/README
  placeholders are tracked.
- **Default reranker license:** `jinaai/jina-reranker-v3` is CC BY-NC 4.0
  (non-commercial). This is documented in `README.md`, `LICENSE`, and
  `config/config.example.yaml`; switch to a permissive reranker for commercial use.
