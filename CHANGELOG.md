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
- Session tree + deterministic projection (ADR 0005): `polya/tree.py` introduces an
  immutable `Entry` with stable ids, a `SessionTree` (append / move_to / rewind /
  branch / override), and a deterministic `project()`. The raw history is never
  rewritten; the model context is a projection of the active branch, so KV-cache
  prefix friendliness is a structural guarantee rather than a discipline.
- entry-id history read-back: `history_read(entry_id, offset)` replaces the snapshot +
  message-number addressing.
- Skills hot reload: `/reload` re-scans skill directories and patches the `<skills>`
  prompt section via a new system entry (a legal restart point).
- Branch navigation and history editing: `/rewind [N]`, `/jump <id>`, `/tree`
  (fork points + all branches), and `/edit <id> <新内容|remove>` (projection-level
  context edit; the original stays readable via `history_read(entry_id)`).
- Session persistence: `/save [名称]`, `/sessions`, and `/load <名称>`; the session
  tree serializes to JSONL under `~/.polya/sessions/`.
- Session lifecycle (`polya/session.py`): sessions get a stable name and metadata
  (title / created / updated / cwd) in the JSONL header, and auto-save after every
  task. New commands: `/resume [名称]` (picker listing name · title · updated),
  `/fork <id>` (new session from an ancestor path), `/clone` (duplicate the current
  session), and `/export [路径]` (Markdown transcript of the active branch).
  Switching sessions resets stats, todos and file tracking so they never bleed
  across conversations. `SessionTree` gains `ancestry` / `copy_branch_upto` / `copy`.
- Session import (`polya/importer.py`) and JSONL export: `/export <path>.jsonl` writes a
  raw session, `/import <path>` loads one from any path. Both polya JSONL and pi's
  session format are accepted; pi's active branch is rebuilt honoring `compaction`
  (`firstKeptEntryId`) and `context_edit` (mapped to polya projection overrides).
- Per-vendor reasoning levels: `/thinking [off|low|medium|high]`; `ModelProfile.reasoning_style`
  selects the request shape (`reasoning_effort` for o-series/gpt-5, `thinking` toggle for
  GLM/DeepSeek, `none` otherwise) via `providers.reasoning_params`.

- `polya/review.py`: a pluggable `Reviewer` seam (`allow`/`deny`) called before every
tool call; the default `AllowAllReviewer` allows everything and only enforces plan
mode's read-only rule.
- `polya/trust.py`: a project trust gate in pi's shape. `~/.polya/trust.json` maps
  canonical paths to `true | false | null`, with **parent-directory inheritance** (the
  closest decision wins; `null` = no decision). Untrusted directories do not load
  `AGENTS.md` or project/ancestor skills; tools still run. The gate only asks when a
  directory actually has protected resources. `/trust` shows the saved decision and its
  inheritance source and saves Trust / Trust parent folder / Do not trust / Clear (next
  start); `--trust` / `--no-trust` cover non-interactive runs. Legacy
  `{"trusted": [...]}` stores migrate on read.
- pi-style input semantics: `Enter` steers (injected before the next model request),
`Alt+Enter` queues a follow-up (runs after the current task), `Alt+Up` pulls queued
messages back into the editor, and Esc-aborting returns them there too.
- `/plan go` (or an exact `批准`/`go`-style message) approves a submitted plan; any
other message keeps revising it read-only.

### Changed

- Tool descriptions, snippets, and guidelines are localized: docstrings remain the
  Chinese source, English overrides live in the catalog, and snippets/guidelines are
  selected by `POLYA_LANG`.
- Project memory (`AGENTS.md`) is injected as its own `<project_memory>` section.
- Invalid tool-argument JSON is reported back to the model instead of being silently
  treated as `{}`.
- Context-size estimation caches the static prompt/tool-schema part and per-message sizes.
- Compaction and micro-compaction now operate on the projection: originals stay in the
  tree and are readable by entry id; the system prompt supports named-section patching
  replayed at projection time.
- `Agent.history` is now a read-only derived view; writes go through tree methods
  (`append_user_message`, `replace_conversation`, `strip_reasoning`, …).
- `web_fetch` validates every redirect hop before following it, and re-validates the
  final URL to close SSRF bypasses.
- The injected status bar no longer carries a timestamp, and it is appended only when
  substantive fields (tool counts / TODO / plan mode) change.
- Interaction v3: tools run by default with the process's permissions (restricted to
  `--root`); the interactive driver no longer preempts the terminal, and `render.py`
  has a single scrollback path (Rich Live removed).
- `Agent` takes `reviewer` instead of `approve` / `approve_plan`; plan state is
  released by the driver via `leave_plan_mode()`.
- Commands are a flat registry; the `busy` three-state and queue pause/resume were
  removed. Session-rewriting commands (`/clear`, `/new`, `/exit`, …) require an idle
  agent and ask for Esc first.

### Removed

- `polya/approval.py` and `polya/permissions.py`: the per-call approval stack,
  authorization rules and high-risk heuristics.
- `/permissions`, `/resume` and `--yes`; session authorization rules.
- `/expand`: merged into `/details [ID]` (no argument shows the last five blocks).
- `Tool.dangerous` compatibility shim (use `Tool.kind`).
- `demo.py` / `quicksort.py` moved into `examples/`.
- `polya/history.py` (`HistoryArchive`): whole-history snapshots replaced by per-entry
  immutability + entry-id read-back.

## [0.1.0] - 2026-09-26

### Added

- Initial public release: minimal, hackable LLM tool-calling coding agent with a
  generator-based agent loop (ADR 0002), local coding tools, plan mode, approval
  system, skills, model capability profiles, and context compression.
