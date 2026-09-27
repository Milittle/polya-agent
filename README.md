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

For multiple providers or coding plans, register named profiles entirely inside
the REPL — `/models add` opens an interactive wizard (stored in
`~/.polya/models.json`, mode 0600; the key is entered via getpass while the input
box yields the terminal, so it never reaches the screen or input history —
listings show only the last four characters):

```
/models add
  可用预设（已知厂商内置，选名字即可）：
    1. z.ai coding plan（国际） · glm-5.3 @ api.z.ai
    2. z.ai coding plan（国内 bigmodel） · glm-5.3 @ open.bigmodel.cn
    3. DeepSeek API · deepseek-flash @ api.deepseek.com
    4. OpenRouter（跨厂商） · （自填模型名） @ openrouter.ai
    5. Moonshot Kimi · （自填模型名） @ api.moonshot.cn
    6. 自定义 OpenAI 兼容端点
/models add ds deepseek            # one-line shortcut: preset + hidden key
/models add box http://localhost:8000/v1 qwen3   # custom endpoint shortcut
/models remove ds
```

Presets are just prefilled `base_url` + suggested model — any OpenAI-compatible
endpoint fits the same triple. Startup resolution: CLI flags > active profile >
`OPENAI_*` env vars. In the REPL, `/models` switches mid-session (no arguments
opens a picker): the conversation is kept, the old model's thinking is stripped,
and the choice is written back as `active`.

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
`--prefix-check`.

### REPL

| Key / prefix | Action |
|---|---|
| `Enter` | send · `Alt+Enter` / `Ctrl+J` / trailing `\` + Enter: newline |
| `/help` `/todos` `/status` `/plan on\|off` `/models [profile]` `/compact [note]` `/expand [N]` `/clear` `/new` `/exit` | slash commands (`/` completes with descriptions) |
| `@` | file-path completion |
| `!command` | run a shell command locally; output goes into the conversation |
| `#note` | append a line to the project memory file (`AGENTS.md`) |
| `Esc` | close completion, or request task interruption; queued messages return to the editor |
| `Alt+Up` | pull queued messages back into the editor |
| `Ctrl+C` | clears the input; press twice within 2s on an empty box to quit |
| big paste | folds to `[Pasted #1 +200 lines]`, expanded again on submit |

`/plan` and `/models` without arguments open options in the input
box; the current value is marked (the `/models` picker also offers `add` for the
setup wizard and `remove`). Choose with arrows and Tab/Enter, then Enter to
execute; Esc closes the menu. You can also type `/plan on|go|off`
or `/models <profile>`
directly, with argument completion. Invalid commands and arguments stay in the
editor with a hint; `/details ID` and `/expand [N]` require positive integers.
`/help` lists all commands, aliases and busy behavior from the same registry.
Queued commands run before the next model request; `/clear`, `/new`, `/exit` and
`/quit` wait until the current task ends, preserving queue order. `/resume` immediately
resumes a queue paused after interruption, rejection or failure; it waits for an active stop to finish. In a piped REPL, supply options explicitly.

`/clear` (alias `/reset`) clears the conversation — history, todos, stats — while
keeping the session identity, topic and queued messages. `/new` starts a
fresh session: it additionally assigns a new session name, resets the topic (re-derived
from the next task), drops queued messages and reprints the banner.

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

The startup header shows the version, project and model once, then scrolls away.
A persistent input area sits below the conversation, with adjoining rules and a
two footer lines. It grows to six lines, then scrolls internally. The first footer
shows model, project directory and optional context usage; the second shows the
session topic, mode, queue state and action hints. Narrow terminals shorten paths
from the left to retain the project name, and shorten the topic before hiding context.
`/rename <topic>` changes the topic and terminal title (one line, up to 120 characters).
`/clear` preserves the topic; `/new` resets it. The topic starts from the first task,
without an extra model request.

You can keep typing while the agent runs. `Enter` sends a **steering** message: it is
injected before the next model request, after **all** results of the current tool batch
are recorded. `Alt+Enter` queues a **follow-up** that runs after the current task ends.
`Alt+Up` pulls queued messages back into the editor; Esc-aborting does the same.
`/clear`, `/new`, `/exit` and `/quit` need an idle agent — press Esc first.
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
- `Agent(compress=True)`: compaction at 80% of the context window — old tool results
  are batch-replaced by summaries (or the whole history is summarized into a restart
  message for thinking-bound models, per `ModelProfile`). Preflight uses the latest
  prompt usage plus estimated new input, with a UTF-8 size estimate when usage is
  missing; it is approximate, never cumulative usage. Empty summaries preserve the
  original history. Three consecutive failures trip a breaker; at an estimated 95%
  of the window, the task stops locally with history intact instead of sending an
  oversized request. Switch to a larger model window to continue.
  Compaction carries active skill references and TODOs, and asks the summary to retain
  constraints, edits, verification evidence and next steps. `history_read` retrieves
  pre-compaction messages by snapshot, message number and character offset. These
  temporary archives last for this process/session and are cleared by `/clear` or
  `/new`; they are not conversation persistence. When tool-only compaction has no
  targets, a full summary restart can compact the remaining conversation.
- `ModelProfile` (providers): capability-driven behavior — reasoning passthrough,
  in-place tool edit vs summary restart, context window and temperature defaults.
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
  models.py      # model profiles (~/.polya/models.json): load/save/validate, presets
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
