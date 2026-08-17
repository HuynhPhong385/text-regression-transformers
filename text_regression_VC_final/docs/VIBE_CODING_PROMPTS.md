# Vibe Coding Prompt Pack

## Prompt 01 — Understand

Read:

```text
AGENTS.md
progress_status.md
knowledge_base/
.ai/rules/
```

Determine current phase and requirement. Do not code yet.

## Prompt 02 — Plan

Inspect relevant files and propose the smallest implementation plan.

## Prompt 03 — Implement

Implement only the approved scope. Follow rules and naming conventions.

## Prompt 04 — Test

Run relevant tests. Fix root causes, not symptoms.

## Prompt 05 — Debug

Reproduce → isolate → hypothesis → smallest fix → regression test.

## Prompt 06 — Audit

Run codebase audit and report CRITICAL/MAJOR/MINOR issues.

## Prompt 07 — Documentation

Synchronize docs with actual code. Never invent behavior.

## Prompt 08 — Experiment

Create config, run experiment, save metrics and metadata. Do not fake missing results.

## Prompt 09 — Full pipeline

Verify:

```text
input → preprocessing → model → inference → API/UI → output
```

## Prompt 10 — Final audit

Verify Definition of Done, tests, reproducibility, docs and report.
