# ADSC Project Template

A starter repository for **project managers (PMs)** in the [TAMU Aggie Data Science Club](https://github.com/TAMU-Aggie-Data-Science-Club). Click **Use this template** to spin up a new project with the club's conventions, workflow, and scaffolding already in place.

## What's in here

| File / folder | Purpose |
|---|---|
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **Start here.** How the team runs the project on GitHub — PM vs. member roles, the Discussion → Issue → PR → `main` flow, branching, worktrees, reviews. |
| [`DELIVERABLES.md`](DELIVERABLES.md) | Suggested deliverables and rough timeline. A living plan, not a contract. |
| [`DATA.md`](DATA.md) | Where the project's data comes from, how to find sources, and how to think about using them. |
| [`CODEOWNERS`](CODEOWNERS) | **Team roster + review policy in one file.** PMs, members, and the code-owner rule that GitHub enforces on PRs into `main`. |
| [`AGENTS.md`](AGENTS.md) | Machine-facing workflow rules for AI coding agents. |
| [`data/`](data/) | Working folder for actual datasets. **Git-ignored** — data is never committed. |
| [`.github/workflows/gitleaks.yml`](.github/workflows/gitleaks.yml) | Secret-scanning check that runs on every push and every PR into `main`. |
| [`.github/dependabot.yml`](.github/dependabot.yml) | Weekly automatic PRs to bump vulnerable dependencies. |
| [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) | Prefilled PR body with the checklist. |
| [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) | Feature and Bug issue forms. |

## How to use this template

1. **Create your repo from it.** On GitHub, click **Use this template → Create a new repository** (private, under `TAMU-Aggie-Data-Science-Club`).
2. **Read [`CONTRIBUTING.md`](CONTRIBUTING.md).** Everyone on the team reads it; it's the operating manual.
3. **Fill in [`DELIVERABLES.md`](DELIVERABLES.md)** with your project's real deliverables and dates.
4. **Fill in [`DATA.md`](DATA.md)** with your actual data sources.
5. **Populate [`CODEOWNERS`](CODEOWNERS)** with the PMs' names/handles/emails, and update the `*` rule line at the bottom to list PMs by `@handle`.
6. **Update the Discussions URL** in [`.github/ISSUE_TEMPLATE/config.yml`](.github/ISSUE_TEMPLATE/config.yml) to point at your repo.
7. **Enable Discussions** on the repo (Settings → Features → Discussions).
8. **Open your first issue** and run the flow.

Branch protection (require PR + Code Owner review, block force-push, gate on Gitleaks) is applied at the org level by officers, not per-repo by PMs.

## Team

The current PMs and members for this project are listed in [`CODEOWNERS`](CODEOWNERS). PMs listed there are the code owners for PRs into `main`.

## The one-paragraph version

Every change starts as either a **Discussion** (for ideation) or directly as a **GitHub Issue** (for something concrete to build). Work happens on a **branch** (organized as a small tree per issue, each slice optionally in its own **worktree**), opened as a **pull request**, reviewed (by a teammate and, ideally, an auto-review agent), and merged **up the tree**. Only a **PM** (via `CODEOWNERS`) approves the merge into `main`. `main` is always in a known-good state. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow.
