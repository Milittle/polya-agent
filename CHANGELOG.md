# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tool schema inference now supports `Literal`, `Enum`, `Annotated[T, "description"]`,
  `dict[str, T]`, and nested `dataclasses`; unknown types get no constraint instead of
  being silently downgraded to `string`.
- Tools carry three separate self-descriptions: `description` (API `tools[]`),
  `snippet` (system-prompt `<tools>` section), and `guidelines` (system-prompt
  `<rules>` section).
- New `polya/prompt.py`: the system prompt is assembled from ordered, named sections
  with a `diff_sections()` helper for incremental updates.
- New `polya/i18n.py`: user- and model-facing strings live in a catalog selected by
  `POLYA_LANG` (`zh` default, `en` available).
- New `polya/subagent.py`: the `task` tool (kind `delegate`) runs a bounded subtask in
  a child agent with its own context and shell session — only the final report returns
  to the parent. Child tool calls pass through the shared `ApprovalGate` one by one;
  depth is 1 (the child has no `task` tool).
- Context compression: `keep_recent_tokens` token budget for the retained region;
  cumulative read/modified file tracking; iterative summarization that feeds the
  previous summary back in; manual `/compact [instructions]`.
- Micro-compaction (ADR 0004): at a lower threshold (0.6) old, large tool results are
  replaced with `history_read` pointers without an LLM call; the `Compaction` event now
  carries `mode` (`full`/`micro`) and `cleared`.
- `cached_tokens` are collected from usage and shown in the status bar and footer.
- `finish_reason == "length"` is handled: a truncated reply triggers compaction and a
  bounded continuation instead of being returned as a finished answer.
- Streaming client retries transient errors before the first token, and falls back to
  non-streaming when an endpoint rejects streaming.
- Type checking: `mypy` configuration and a CI step.
- `examples/` package for runnable demos, moved out of the repository root.

### Changed

- Tool descriptions, snippets, and guidelines are localized: docstrings remain the
  Chinese source, English overrides live in the catalog, and snippets/guidelines are
  selected by `POLYA_LANG`.
- Project memory (`AGENTS.md`) is injected as its own `<project_memory>` section.
- The pre-compaction history archive is now an in-memory, non-destructive snapshot whose
  lifetime matches the session (previously a temporary directory that could expire while
  history still referenced it).
- Invalid tool-argument JSON is reported back to the model instead of being silently
  treated as `{}`.
- Context-size estimation caches the static prompt/tool-schema part and per-message sizes.

### Removed

- `Tool.dangerous` compatibility shim (use `Tool.kind`).
- `demo.py` / `quicksort.py` moved into `examples/`.

## [0.1.0] - 2026-09-26

### Added

- Initial public release: minimal, hackable LLM tool-calling coding agent with a
  generator-based agent loop (ADR 0002), local coding tools, plan mode, approval
  system, skills, model capability profiles, and context compression.
