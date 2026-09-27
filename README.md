# polya

A minimal, hackable LLM tool-calling agent: the model thinks → calls tools → results
are fed back → repeat until it has an answer. Ships with a set of local coding tools,
so it works as a coding agent out of the box.

Named after G. Pólya (*How to Solve It*) — the patron saint of problem-solving agents.

[中文文档](./README.zh-CN.md)

## Install

Python 3.10+, recommended via [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

or pip:

```bash
pip install -e .
```

## Configuration

Any OpenAI-compatible endpoint (`/v1/chat/completions`) works; the example defaults
to DeepSeek:

```bash
cp .env.example .env
# edit .env: OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL)
```

Prompts and user-facing messages are localized via `POLYA_LANG` (`zh` default, `en`
available). The prompts are evaluated at import time, so the choice is stable for the
whole session and the KV-cache prefix stays intact.

For multiple providers or coding plans, log in entirely inside the REPL. `/login`
opens the provider list (first batch: OpenRouter, DeepSeek, z.ai global/CN,
Moonshot global/CN, Groq, Together, NVIDIA, Qwen Token Plan global/CN, plus a
custom OpenAI-compatible endpoint). The base URL is prefilled from the provider
table and can be overridden; the key is entered via getpass while the input box
yields the terminal, so it never reaches the screen or input history. Credentials
live in `~/.polya/models.json` (mode 0600):

```
/login zai
  base_url（回车用 https://api.z.ai/api/coding/paas/v4）:
  api_key（输入不回显）:
已登录 zai：glm-5.3 @ api.z.ai · 窗口 1M，已设为默认启动模型；模型目录后台刷新中。/model 切换，Ctrl+S 设默认。
```

After login, Polya saves the credential right away and refreshes the provider's
`/models` list in a background thread (15s timeout), so the wizard never blocks on
it; context windows are read when the endpoint reports them (`context_length` /
`max_model_len`), otherwise the built-in capability table applies. Resolution:
discovered value > built-in table > 128k. `/model` switches mid-session (no
arguments open a picker across every logged-in model; `Ctrl+S` saves the default
startup model). `/logout <provider>` removes credentials. Startup resolution:
CLI flags > default model (`active`) > `OPENAI_*` env vars.

## CLI

```bash
uv run polya                    # interactive REPL
uv run polya -p "fix the failing tests" --plan   # one-shot mode
```

Useful flags: `--root DIR` (working dir; file tools are jailed inside), `--plan`
(start in plan mode), `--trust` / `--no-trust` (save and apply a project-trust decision;
`--trust` is needed for non-interactive runs in a fresh directory), `--max-steps N` (default 25),
`--model/--base-url/--api-key`, `--no-stream`, `--no-compress`, `--no-microcompact`,
`--context-window N` (default 128000), `--keep-recent N`, `--keep-recent-tokens N`,
`--reserve-tokens N` (absolute compaction reserve, default 16384), `--prefix-check`.

### REPL

| Key / prefix | Action |
|---|---|
| `Enter` | send (steering: injected before the next model request) |
| `Alt+Enter` | queue a follow-up (runs after the current task) · `Ctrl+J` / trailing `\` + Enter: newline |
| `/help` `/todos` `/status` `/plan on\|go\|off` `/login [provider\|custom]` `/logout <provider>` `/model [provider/model]` `/thinking [level]` `/compact [note]` `/details [ID]` `/resume [name]` `/fork <id>` `/clone` `/export [path]` `/import <path>` `/trust [decision]` `/new` `/clear` `/reset` `/exit` | slash commands (`/` completes with descriptions) |
| `@` | file-path completion |
| `!command` | run a shell command locally; output goes into the conversation |
| `#note` | append a line to the project memory file (`AGENTS.md`) |
| `Esc` | close completion, or request task interruption; queued messages return to the editor |
| `Alt+Up` | pull queued messages back into the editor |
| `Ctrl+C` | clears the input; press twice within 2s on an empty box to quit |
| big paste | folds to `[Pasted #1 +200 lines]`, expanded again on submit |

`/plan`, `/login`, `/logout` and `/model` without arguments open options in the
input box; the current value is marked. `Ctrl+S` inside the `/model` picker saves
the highlighted model as the default startup model. Choose with arrows and
Tab/Enter, then Enter to execute; Esc closes the menu. You can also type
`/plan on|go|off`, `/login zai`, `/logout zai` or `/model zai/glm-5.3` directly,
with argument completion. Invalid commands and arguments stay in the
editor with a hint; `/details [ID]` takes an optional positive integer (no argument shows
the last five blocks).
`/help` lists all commands and aliases from the same flat registry. Commands run
immediately; the ones that rewrite the session (`/new`, `/exit`, `/compact`,
`/rewind`, `/jump`, `/edit`, `/load`, `/resume`, `/fork`, `/clone`, `/import`, `/model`, `/reload`, `/save`)
need an idle agent
and ask you to press Esc first when a task is running. In a piped REPL, supply options
explicitly.

**Sessions** get a stable name and metadata (title, created/updated, cwd) and are
auto-saved to `~/.polya/sessions/<name>.jsonl` after every task, so `/resume` lists
conversations you actually had. `/resume` with no argument opens a picker in the input
box (name · title · updated); `/resume <name>` switches directly. `/fork <id>` derives a
new session from the ancestor path up to entry `#id` (`/tree` shows ids); `/clone`
duplicates the current session. Switching resets stats, todos and file tracking so
sessions never bleed into each other. `/export [path]` writes the active branch as
Markdown (default `~/.polya/exports/<name>.md`); `/save` keeps the raw JSONL for `/load`.
`/export <path>.jsonl` writes a raw session for `/import`; `/import <path>` loads a session
from any path, accepting both polya JSONL and pi's session format (the pi active branch is
rebuilt, honoring `compaction` and `context_edit`).

`/thinking [off|low|medium|high]` sets the reasoning level. The request field is
per-vendor (`reasoning_effort` for o-series/gpt-5, `thinking` toggle for GLM/DeepSeek);
models whose profile has no reasoning style report that no level is available.

`/new` (aliases `/clear`, `/reset`) starts a fresh session: the conversation —
history, todos, stats — is replaced, a new session name is assigned, the topic resets
(re-derived from the next task), queued messages are dropped and the banner reprints.
The previous session is kept on disk and can be recovered with `/resume`.

**Project memory**: if the working directory is trusted and has an `AGENTS.md`, it is
read once at startup and appended to the system prompt (quasi-static: it never changes
mid-session, so the KV-cache prefix stays stable). Untrusted directories do not load
it. `#` prefix appends to it — takes effect next session.

### Skills for development

Skills live in `.polya/skills/<name>/SKILL.md` (project) or
`~/.polya/skills/<name>/SKILL.md` (user), discovered recursively — a directory
containing `SKILL.md` is a skill root and is not descended further. Agent Skills
standard locations load too: `~/.agents/skills/` (user) and `.agents/skills/`
found from the working directory through its ancestors, stopping at the git
repository root. Same-name precedence, lowest to highest: user `.agents` →
user `.polya` → ancestor `.agents` (far to near) → project `.polya`. Each file
needs YAML frontmatter with `name` and `description`;
invalid entries are skipped with a warning. Only the catalog enters the startup
prompt. The model loads applicable instructions using `skill_read(name)`; you can
also ask explicitly with `$name`. Resources use paths relative to the skill folder,
for example `skill_read(name="develop", path="references/testing.md")`.
Reads are paginated; skill scripts run with the process's permissions like any bash
command. Project skills load only in a trusted directory.

This repository includes `$develop-polya`: read the ticket and project conventions,
make the change, run checks, repair failures, inspect the diff and report evidence.
For example, in the REPL: `Use $develop-polya to implement .scratch/feature/issues/01-task.md`.
Skill catalogs are refreshed on startup; loading a skill does not change the system
prompt or tool schemas. User skills do not widen the file tools' workspace boundary.

### Trust, permissions & the reviewer seam

Tools declare a `kind`: `read` (no side effects), `write` (files), or `exec` (commands).
There is **no per-call approval**: tools run with the permissions of the Polya process,
restricted to `--root` (`../` and absolute paths are rejected). Every call passes
through a pluggable `Reviewer` (`polya/review.py`) that returns `allow` or `deny`:

- the default `AllowAllReviewer` allows everything, except that in plan mode it denies
every non-`read` tool (delegation is the exception, since its side effects happen in
child calls), keeping plan mode read-only;
- a future **model reviewer** (Jev-style) can implement the same protocol to intercept
commands. A `deny` only becomes a tool result for the model — it never opens a prompt
or takes over the terminal.

**Project trust** is the one gate. It controls whether project resources are loaded:
`AGENTS.md` and project/ancestor skills. Decisions are saved in `~/.polya/trust.json`
as `{ "<abs path>": true | false | null }` and **inherited from parent directories**
(the closest `true`/`false` wins; `null` means “no decision”). Polya only asks when a
directory actually has protected resources; declining leaves tools running but loads no
project instructions. `/trust` shows the saved decision (and where it was inherited
from) and saves Trust / Trust parent folder / Do not trust / Clear — the change takes
effect on the next start. Non-interactive runs never prompt; use `--trust` / `--no-trust`
for an explicit decision (`--trust` also loads project resources when none is saved).
`/permissions` and session authorization rules are gone.

**Plan mode** is a two-phase read-only stance. `exit_plan_mode` prints the plan to
scrollback and ends the turn; `plan_mode` stays on. Reply with `/plan go` (or an exact
`批准`/`go`-style approval) to execute, or send any other message to keep revising the
plan read-only.

**Interrupts**: `Esc` requests a stop at the next event boundary. A running tool or
model request may need to finish first. Completed steps stay in history, pending tool
results are back-filled, and queued messages return to the editor. `Ctrl+C` edits/quits
the input; `-p` retains Ctrl+C interruption.

### Display

The startup header shows the version, tagline and project-level status (AGENTS.md
loaded, or an untrusted-directory warning) once, then scrolls away. A persistent
input area sits below the conversation, with adjoining rules and three footer lines.
It grows to six lines, then scrolls internally. The first footer is identity: the
project (`~`-shortened) with its git branch and the session topic on the left, and
`(provider) model • thinking` on the right. The second is usage: cumulative
`↑input ↓output CRcache CHhit% $cost` followed by `ctx %/window (auto)`. The third
holds mode and queue state on the left and context-sensitive action hints on the
right. Narrow terminals drop provider, thinking, branch, topic and usage details in
that order, always keeping the model, project name and mode.
`/rename <topic>` changes the topic and terminal title (one line, up to 120 characters).
`/new` (and its aliases `/clear`, `/reset`) resets the topic; it is re-derived from
the first task, without an extra model request.

You can keep typing while the agent runs. `Enter` sends a **steering** message: it is
injected before the next model request, after **all** results of the current tool batch
are recorded. `Alt+Enter` queues a **follow-up** that runs after the current task ends.
`Alt+Up` pulls queued messages back into the editor; Esc-aborting does the same.
`/new` (and its aliases `/clear`, `/reset`), `/exit` and `/quit` need an idle agent — press Esc first.
Enter selects a completion when its menu is open; otherwise it
submits at the end of the buffer or inserts a newline inside the text.

Completed messages append once to native terminal scrollback;
reasoning is collapsed and available through `/details`. Tool names use display labels
(`Bash`, `Read File`); identifiers sent to the model stay unchanged. Tools show
`Running`, then `Ran`, `Failed`, or `Denied`. Result previews are limited to three
lines and 400 characters. While running, Bash shows a moving tail above the input;
only its completion summary enters scrollback. `+ Show details: /details ID` opens that exact archived
tool block, including full arguments and the returned result
(the tool itself may cap its output).
While a task runs, its activity and elapsed time stay above the input: `Waiting for
model`, `Thinking`, `Responding`, `Running …`, or `Reviewing`.
Tool changes do not reset the task timer. `Stopping` identifies the activity being
waited on. Each model task leaves a turn-ended, interrupted or failed receipt
with duration; turn-ended does not claim that the requested goal was achieved.
The most recent 20 blocks are retained; expired IDs report unavailable.
With completions open, Esc closes them first. Streaming text appears before a newline
in a live Markdown tail above the input (up to eight content lines, fewer on short
terminals). Completed messages retain full table, list and code-block formatting.
Interrupted partial responses remain in scrollback, labeled as partial.

The interactive UI uses one output proxy, without a full-screen terminal or a
competing Rich Live display. Native scroll and copy remain available. `-p` keeps
its existing final-answer stdout and diagnostic stderr behavior.

## Library usage

Define a tool with a plain decorated function; the JSON schema is inferred:

```python
from polya import Agent, LLM, tool


@tool
def get_weather(city: str, unit: str | None = None) -> str:
    """Look up the weather for a city. unit: 'celsius' or 'fahrenheit'."""
    ...


agent = Agent(llm=LLM(), tools=[get_weather])
print(agent.run("What's the weather in Beijing?"))
```

`Agent.run()` is the **built-in driver** (ADR 0002): the agent itself is a generator —

```python
gen = agent.steps(prompt)
ev = next(gen)  # Text / ReasoningDelta / Usage / Iteration /
# AssistantMessage / ToolCall / Compaction / PlanSubmitted
result = gen.send("...")  # ToolCall / PlanSubmitted expect the result string back
```

so any frontend (REPL, TUI, web) can drive rendering and tool review itself — that's
exactly what `polya/loop.py` does. Event names double as the renderer vocabulary (see
`polya/agent.py`).

Notable knobs:

- `reviewer=Reviewer()`: pluggable allow/deny decision before each tool call (default
  `AllowAllReviewer`); tool exceptions become `Error:` text handed back to the model
  instead of crashing.
- `Agent(plan_mode=True)`: two-phase plan mode. Plan state is owned by the driver
  layer (`leave_plan_mode()` on approval); the tool array never changes mid-session
  (KV-cache prefix stays stable).
- `status_bar=True`: appends an `<agent_status>` snapshot (iteration, per-tool call
  counts, token usage, todos) to the **end** of the context each round — the model
  never has to count from its own trajectory. Append-only, cache-friendly.
- `Agent(compress=True)`: compaction triggers at `min(window × threshold, window −
  reserve_tokens)` (default 80% / 16384, floored at half the window — small windows
  never head into the thin 20% tail) — old tool results
  are batch-replaced by summaries (or the whole history is summarized into a restart
  message for thinking-bound models, per `ModelProfile`). Preflight uses the latest
  prompt usage plus estimated new input, with a UTF-8 size estimate when usage is
  missing; it is approximate, never cumulative usage. Empty summaries preserve the
  original history. Three consecutive failures trip a breaker; at an estimated 95%
  of the window, the task stops locally with history intact instead of sending an
  oversized request, and a provider context-overflow error triggers one
  compact-and-retry per task. Switch to a larger model window to continue.
  Compaction carries active skill references and TODOs, and asks the summary to retain
  constraints, edits, verification evidence and next steps; the restart summary follows
  a fixed nine-section template, and summarization calls count toward `total_usage`. `history_read` retrieves
  pre-compaction messages by snapshot, message number and character offset. These
  temporary archives last for this process/session and are cleared by `/new`,
  `/clear` or `/reset`; they are not conversation persistence. When tool-only compaction has no
  targets, a full summary restart can compact the remaining conversation.
- `ModelProfile` (providers): capability-driven behavior — reasoning passthrough,
  in-place tool edit vs summary restart, and temperature defaults. Its context
  window is the static fallback layer under discovered provider values.
  Unknown models fall back to safe defaults.
- `Agent(prefix_check=True)`: runtime assert that each request strictly extends the
  previous one (append-only between compaction points).

The system prompt stays static during the session; project memory and skill metadata
are injected once at startup. Dynamic information is appended to the conversation.

## Built-in tools

`default_tools(root)` — all file operations are jailed inside `root` (`../` and
absolute paths are rejected):

| Tool | kind | Notes |
|---|---|---|
| `read_file` | read | numbered lines, line ranges, rejects binary/oversized |
| `list_dir` | read | single level, dirs suffixed `/` |
| `glob` | read | recursive filename match, skips `.venv`/`.git` |
| `grep` | read | regex content search, context lines, ignore-case, glob filter |
| `write_file` | write | create/overwrite, makes parent dirs |
| `edit_file` | write | unique-match replace |
| `multi_edit` | write | multiple edits, all-or-nothing |
| `bash` | exec | persistent session: cwd/env survive across calls |
| `bash_output` | read | poll/wait for completion; retrieve full output by command ID and line range |
| `kill_bash` | exec | terminate the session |
| `web_fetch` | read | URL → text, `<external_content>` wrapping, SSRF guard |
| `todo_write` | read | full TODO rewrite (external memory via status bar) |

The CLI/driver additionally registers `task` (kind `delegate`): it delegates a bounded
subtask to a child agent with its own context and shell session, keeping exploration
chains out of the main context. Only the final report returns to the parent; the child's
tool calls still pass through the shared reviewer one by one (depth is 1 — the child
has no `task` tool).

Context is managed in two tiers: **micro-compaction** (no LLM call) replaces large old
tool results with `history_read` pointers once usage passes 60% of the window, and
**compaction** (LLM summary) runs at 80%. `/compact` triggers a full compaction manually.
Usage reports cache hits when the endpoint provides `cached_tokens`.

Most tool output is truncated to 8000 characters. Bash previews keep both the beginning
and end, including the exit code. `bash(timeout=N)` waits up to N seconds (0–300), then
returns a command ID while the command continues. Use `bash_output(timeout=30)` to
wait again; another command cannot start until the current one finishes or is killed.
Use `bash_output(command_id=1, start_line=1, end_line=200)` for complete logs;
long pages expose a character `offset` for further reads. Logs are temporary for the
lifetime of the tool set. Commands must be non-interactive. `kill_bash` terminates the
session (including its process group on POSIX); the next command starts a new shell.
Explicit `&` jobs share shell stdout, so use foreground commands for attributable checks.

CLI assembly also adds read-only `skill_read`; compression adds read-only `history_read`.
Pure Python throughout (Windows-native friendly); only `bash` needs a real bash.

## Project layout

```
polya/
  agent.py       # generator protocol: event union + steps() + built-in run() driver
  loop.py        # interactive driver: render / review / execute / dispatch / queues
  input.py       # InputBox: multiline editing, completion, paste folding, steering keys
  render.py      # scrollback renderer + preview tail for the editor
  review.py      # pluggable tool reviewer (allow/deny); default allows all
  trust.py       # project trust gate for AGENTS.md / project skills
  executor.py    # shared tool executor (loop and run())
  cli.py         # argparse + assembly + one-shot mode
  llm.py         # OpenAI-compatible client; chat_iter streaming
  models.py      # provider config (~/.polya/models.json): load/save/validate/migrate, presets, discovery
  skills.py      # skill discovery, metadata catalog and read-only resource loading
  history.py     # temporary pre-compaction history snapshots and paginated reads
  tools.py       # @tool decorator & registry (kind: read/write/exec)
  builtin.py     # built-in coding tools + coding system prompt
  shell.py web.py status.py todos.py compact.py providers.py
```

Domain glossary: [CONTEXT.md](./CONTEXT.md) · decisions:
[docs/adr/](./docs/adr/).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## License

[MIT](./LICENSE)
