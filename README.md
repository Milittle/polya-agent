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

## CLI

```bash
uv run polya                    # interactive REPL
uv run polya -p "fix the failing tests" --plan -y   # one-shot mode
```

Useful flags: `--root DIR` (working dir; file tools are jailed inside), `--plan`
(start in plan mode), `--yes` (auto-approve everything), `--max-steps N` (default 25),
`--model/--base-url/--api-key`, `--no-stream`, `--no-compress`, `--context-window N`
(default 128000), `--keep-recent N`, `--prefix-check`.

### REPL

| Key / prefix | Action |
|---|---|
| `Enter` | send · `Alt+Enter` / `Ctrl+J` / trailing `\` + Enter: newline |
| `/help` `/todos` `/status` `/plan on\|off` `/expand [N]` `/reset` `/exit` | slash commands (`/` completes with descriptions) |
| `@` | file-path completion |
| `!command` | run a shell command locally; output goes into the conversation |
| `#note` | append a line to the project memory file (`AGENTS.md`) |
| `Ctrl+C` | clears the input; press twice within 2s on an empty box to quit |
| big paste | folds to `[Pasted #1 +200 lines]`, expanded again on submit |

**Project memory**: if the working directory has an `AGENTS.md`, it is read once at
startup and appended to the system prompt (quasi-static: it never changes mid-session,
so the KV-cache prefix stays stable). `#` prefix appends to it — takes effect next
session.

### Permissions & approval

Tools declare a `kind`: `read` (no side effects), `write` (files), `exec` (commands).
Every call goes through a six-step `decide()`:

1. `read` → allow
2. plan mode and not `read` → deny ("read-only mode, submit a plan first")
3. high-risk (heuristic match on the bash command string: `rm -rf`, `sudo`,
   `curl | sh`, `push -f`) → forced ask, **no authorization escape hatch**
4. a session rule matches, or `--yes` → allow
5. auto-edit mode and `write` → allow
6. otherwise → ask

The approval prompt shows a **change preview** first — a unified diff against what's
on disk for write tools (red/green, new files all green, folded past 40 lines), the
full command for `bash` — then a four-option list, **cursor on "reject" by default:
Enter alone never approves**:

1. Allow once
2. Allow by prefix for this session (e.g. `bash(pytest tests/test_a.py:*)`; compound
   commands with `&&` `;` `|` never get this option, and existing prefix rules don't
   match them either)
3. Modify then run (bash only: pre-filled command line)
4. Reject, optionally with a reason that goes back to the model

Plan mode works the same way: `exit_plan_mode` becomes a plan-approval prompt; after
approval, write tools run (still subject to the normal approval flow). Non-interactive
environments reject everything dangerous by default.

**Interrupts**: `Ctrl+C` during a task aborts only that task — completed steps stay in
history and pending tool results are back-filled so the conversation can continue.

### Display

Rendering targets Claude Code's inline style (see
[ADR 0002](./docs/adr/0002-agent-as-generator.md)): a permanent scrollback region plus
a small transient live region at the bottom (spinner with iteration/phase/context
usage, streaming tail). No full-screen TUI — native terminal scroll and copy just
work. Piped/CI runs degrade to plain text; `-p` writes only the final answer to
stdout, so `> answer.md` is safe.

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

so any frontend (REPL, TUI, web) can drive rendering, permission decisions and
approval itself — that's exactly what `polya/loop.py` does. Event names double as the
renderer vocabulary (see `polya/agent.py`).

Notable knobs:

- `approve(tool, args) -> bool`: approval hook for the built-in driver; tool
  exceptions become `Error:` text handed back to the model instead of crashing.
- `Agent(plan_mode=True, approve_plan=...)`: two-phase plan mode. Plan state is owned
  by the driver layer; the tool array never changes mid-session (KV-cache prefix
  stays stable).
- `status_bar=True`: appends an `<agent_status>` snapshot (iteration, per-tool call
  counts, token usage, todos) to the **end** of the context each round — the model
  never has to count from its own trajectory. Append-only, cache-friendly.
- `Agent(compress=True)`: compaction at 80% of the context window — old tool results
  are batch-replaced by summaries (or the whole history is summarized into a restart
  message for thinking-bound models, per `ModelProfile`). The trigger is the *latest*
  prompt tokens, never cumulative usage. 3 consecutive failures trip a breaker.
- `ModelProfile` (providers): capability-driven behavior — reasoning passthrough,
  in-place tool edit vs summary restart, context window and temperature defaults.
  Unknown models fall back to safe defaults.
- `Agent(prefix_check=True)`: runtime assert that each request strictly extends the
  previous one (append-only between compaction points).

The system prompt should be 100% static — dynamic info is appended to the end of the
conversation instead. The one sanctioned exception: `AGENTS.md` project memory is
injected once at startup (quasi-static, see above).

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
| `bash_output` | read | non-blocking read of session output |
| `kill_bash` | exec | terminate the session |
| `web_fetch` | read | URL → text, `<external_content>` wrapping, SSRF guard |
| `todo_write` | read | full TODO rewrite (external memory via status bar) |

Tool output is truncated to 8000 chars. Pure Python throughout (Windows-native
friendly); only `bash` needs a real bash.

## Project layout

```
polya/
  agent.py       # generator protocol: event union + steps() + built-in run() driver
  loop.py        # interactive driver: render / decide / approve / execute / dispatch
  input.py       # InputBox: multiline editing, completion, paste folding, Ctrl+C
  render.py      # scrollback + live-region renderer, shared consoles
  permissions.py # decide() six-step ordering, high-risk table, prefix rules
  executor.py    # shared tool executor (loop and run())
  cli.py         # argparse + assembly + one-shot mode
  llm.py         # OpenAI-compatible client; chat_iter streaming
  tools.py       # @tool decorator & registry (kind: read/write/exec)
  builtin.py     # built-in coding tools + coding system prompt
  shell.py web.py status.py todos.py compact.py providers.py
```

Domain glossary: [CONTEXT.md](./CONTEXT.md) · decisions:
[docs/adr/](./docs/adr/).

## Development

```bash
uv run pytest            # 186 tests
uv run ruff check .
uv run ruff format --check .
```

## License

[MIT](./LICENSE)
