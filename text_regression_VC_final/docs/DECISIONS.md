# Architecture Decisions

## ADR-001 — Normalized target

Use `[0,1]` because it is easy to interpret and compatible with sigmoid output.

## ADR-002 — IMDb first

IMDb is MVP dataset; Yelp is extension.

## ADR-003 — MAE primary

MAE is intuitive for average absolute prediction error.

## ADR-004 — Config-driven

Model and hyperparameters come from configs.

## ADR-005 — AI Agent controlled by docs

Agent must rely on Business Understanding, Knowledge Base, Rules and Skills rather than hidden assumptions.

## ADR-006 — Two-level testing

Model quality and full-system correctness are separate acceptance dimensions.
