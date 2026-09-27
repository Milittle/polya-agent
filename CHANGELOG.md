# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **The CLI no longer has a step cap** (breaking): `--max-steps` is gone and `steps()`
  runs until the model gives a final answer (pi semantics), with the no-progress breaker
  as the only automatic guard. The engine keeps the soft-checkpoint budget for the
  library API: `Agent(max_steps=…, max_continuations=…)` yields a `BudgetCheckpoint` at
  the budget and auto-continues, then `BudgetExhausted` ends the turn with the full
  history preserved and a resumable message (no `[任务失败]`). The library default stays
  bounded (`max_steps=10`, `max_continuations=0`), and sub-agents remain hard-bounded.
- `-p` now saves the session on every exit path (success, budget-exhausted, interrupt,
  error), so one-shot runs are resumable and auditable; hitting the continuation limit
  prints `[未完成]` and exits 1.

### Added

- No-progress breaker (always on, no CLI switch): repeating the same tool call with
  identical arguments first nudges the model at the third repetition (the nudge is
  appended to the tool result), then stops the turn resumably on the next one. With the
  CLI now uncapped this is the only automatic guard. Waiting tools are exempt via the
  new `Tool.poll` flag (e.g. `bash_output` polling); sub-agents keep their hard
  `max_steps` bound instead. The limit stays tunable through the library API
  (`Agent(loop_guard=…, loop_repeat_limit=…)`).
- `/new`, `/clear` and `/reset` now wipe the screen (erase display + cursor home)
  before reprinting the startup banner, so the terminal reads like a freshly
  launched session instead of stacking on the old scrollback.
- Compaction trigger now guarantees an absolute reserve (`--reserve-tokens`, default
  16384, matching pi's `reserveTokens`): the trigger is
  `min(window × threshold, window − reserve)` floored at half the window, so small
  windows compact earlier instead of heading into the thin 20% tail with ~1.6k of
  headroom; windows ≥ 80k behave exactly as before.
- Provider context-overflow errors (`maximum context length`, `prompt is too long`, …)
  now trigger one compact-and-retry per task (`llm.is_context_overflow`), mirroring
  pi's recovery; the truncated-continue path reuses the same compaction-apply path.
- Summarization calls count toward `total_usage` via an `on_usage` callback on
  `compact_messages` / `compact_restart` (previously invisible to session totals).
- The restart summary follows a fixed nine-section template (goal & acceptance /
  constraints & preferences / progress done-in-progress-blocked / key decisions &
  rationale / files / verification evidence / dead ends / skills / next steps), and
  the per-piece summary prompt asks for explicit progress states.
- `PRICES` is documented as a static table refreshed per release; the comment records
  the future dynamic path (pi's remote catalog: ETag conditional requests,
  lastModified-vs-bundled competition, 4h throttle, silent fallback).

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
- `polya/gitinfo.py`: reads the current git branch straight from `.git/HEAD`
(worktrees and detached HEAD included) with an mtime cache, no subprocess.
- Cost estimation: `providers.estimate_cost` prices cumulative tokens against a prefix
  table `PRICES` (input / output / cacheRead / cacheWrite, USD per 1M) taken from pi's
  generated model catalog for the bundled providers; `SUBSCRIPTION_PROVIDERS`
  (coding / token plans) annotate the figure with `(sub)`. Like pi, the `↑input` count
  excludes cache reads, which are billed at the separate `cacheRead` rate.

### Changed

- Completion menu now renders background-free in dim gray with the selected row in
  bold bright cyan (variant B of the 2026-09 style prototype), replacing the
  prompt_toolkit default light-gray block; every class sets `bg:default` explicitly
  so the defaults cannot bleed through.
- Tool descriptions, snippets, and guidelines are localized: docstrings remain the
  Chinese source, English overrides live in the catalog, and snippets/guidelines are
  selected by `POLYA_LANG`.
- Project memory (`AGENTS.md`) is injected as its own `<project_memory>` section.
- Invalid tool-argument JSON is reported back to the model instead of being silently
  treated as `{}`.
- Context-size estimation caches the static prompt/tool-schema part and per-message sizes.
- `/login` and `/logout`: a provider table (first batch of 8 groups from pi's
  OpenRouter/DeepSeek/z.ai/Moonshot/Groq/Together/NVIDIA/Qwen catalog) logs in by
  name, prefills the base URL (overridable), asks for the key via getpass, then saves
  the credential immediately (pi-aligned): the model catalog is refreshed in a
  background thread (15s timeout, no retries) instead of blocking the wizard on
  `/models`, so login returns at once; failures fall back to the static window table
  and are reported in the scrollback. Re-running `/login` keeps the user's currently
  selected model rather than resetting it to the preset default.
- `/model`: a picker across every logged-in provider and discovered model; `Ctrl+S`
  saves the highlighted model as the default startup model.
- Context windows resolve through discovered endpoint metadata (`context_length` /
  `max_model_len`) → the built-in capability table → 128k; the resolved value drives
  compaction, not just the footer.
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
  removed. Session-rewriting commands (`/new`, `/exit`, …) require an idle
  agent and ask for Esc first.
- The status area under the editor is now three lines aligned with pi's footer:
identity (project with `~` and git branch, topic, and `(provider) model • thinking`
on the right), usage (`↑input ↓output CRcache CHhit% $cost ctx %/window (auto)`), and
actions (mode + queue on the left, context-sensitive hints on the right). The busy
status line above the editor is unchanged. Narrow terminals drop provider → thinking
→ branch → topic → usage details, always keeping the model, project basename, and mode.
- The startup banner now reports identity only (version, tagline, project memory /
  trust), dropping the model, directory and `/help` that the persistent footer already
  shows. The first footer line adds the model's context window next to the model
  (`model · 200k · dir`), appending `ctx N%` after the first request; on narrow
  terminals it drops `ctx%` first, then the window. Both footer lines leave a
  one-column right margin so the last character is no longer clipped at the edge.
- `~/.polya/models.json` now keys credentials by provider id with a discovered model
  catalog (`{active: "provider/model", providers: {id: {base_url, api_key, model,
  models: [{id, context_window}]}}}`); old free-form `profiles` migrate automatically.
  Startup resolution is CLI flags > default model (`active`) > `OPENAI_*` env vars.
  `/models` remains as an alias of `/model`; switching no longer rewrites the default.
- `/new`, `/clear` and `/reset` are now one command (`/new`, with `/clear` and `/reset`
  as aliases): starting a fresh session always clears the context, assigns a new session
  name, resets the topic, drops queued messages and reprints the banner. The previous
  session is preserved on disk and recoverable via `/resume`; the old in-place `/clear`
  (which kept the session name and let the next auto-save overwrite the old transcript)
  is gone.

### Removed

- `polya/approval.py` and `polya/permissions.py`: the per-call approval stack,
  authorization rules and high-risk heuristics.
- `/permissions`, `/resume` and `--yes`; session authorization rules.
- `/expand`: merged into `/details [ID]` (no argument shows the last five blocks).
- `Tool.dangerous` compatibility shim (use `Tool.kind`).
- `demo.py` / `quicksort.py` moved into `examples/`.
- `/models add` / `/models remove` and the profile wizard: replaced by `/login` and
  `/logout`; `/models` is kept only as an alias of `/model`.
- `polya/history.py` (`HistoryArchive`): whole-history snapshots replaced by per-entry
  immutability + entry-id read-back.

### Fixed

- Typing a slash command to its full name (e.g. `/help`) keeps the completion menu
  visible with that command selected and its description, matching Claude Code / Codex;
  previously prompt_toolkit's unique-no-increment reset collapsed the menu exactly when
  the confirmation mattered most. Esc still closes it (without reopening), Enter still
  executes, and the zombie-state guard from the menu-reopen fix is preserved.
- Live thinking is rendered again: the v3 refactor left `_reasoning_buf` collected but
  unrendered, so the preview window stayed hidden while the model reasoned and the UI
  showed only `Thinking · Ns`. The bounded thinking tail (`✻ 思考中 · N 字` + last few
  lines, dim italic) is restored in the preview tail; the full thinking text still only
  lands in scrollback folded, and via `/details`. The buffer is also cleared on tool
  result and on interrupt, so a finished or aborted thinking tail cannot linger.
- The busy line repaints once a second while a task is running, so `Thinking · Ns`
  keeps counting even when the provider sends no events during a long think or tool.
- Completion menu now reopens after deleting back to a matching prefix (prompt_toolkit
  only restarts completion on insertion, not deletion) and auto-popup preselects the
  first candidate without inserting its text; Tab keeps its insert-first behavior and
  typing a full command name still closes the menu.
- The footer keeps showing a command's description (and argument hint) after the name is
  typed in full and the completion menu closes; previously it fell back to the bare
  command name, dropping the description. A duplicate `on_text_changed` handler that ran
  auto-completion twice per keystroke was also removed.
- `/login` no longer freezes the whole TUI. Idle commands were executed on the asyncio
  event-loop thread, so the terminal-handoff handshake in `_borrow_terminal`
  (`call_soon_threadsafe` + `ready.wait()`) waited on the very loop it was blocking on —
  a permanent deadlock that hit every provider (reported with DeepSeek). Commands are now
  dispatched as their own worker task (like model tasks), keeping the loop schedulable;
  `_borrow_terminal` also falls back to running the callback directly if it is ever
  invoked from the loop thread, so the failure mode can no longer be an infinite hang.

## [0.1.0] - 2026-09-26

### Added

- Initial public release: minimal, hackable LLM tool-calling coding agent with a
  generator-based agent loop (ADR 0002), local coding tools, plan mode, approval
  system, skills, model capability profiles, and context compression.
