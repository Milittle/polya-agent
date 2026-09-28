"""启动发现元数据；仅通过固定只读工具加载技能正文和相对资源。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .prompts import tool_schema
from .tools import Tool, tool

logger = logging.getLogger(__name__)
MAX_SKILL_BYTES = 256_000


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path


def _read(path: Path) -> str:
    with path.open("rb") as source:
        raw = source.read(MAX_SKILL_BYTES + 1)
    if len(raw) > MAX_SKILL_BYTES:
        raise ValueError(f"技能文件超过 {MAX_SKILL_BYTES} 字节: {path}")
    return raw.decode("utf-8")


def _ancestor_agents_dirs(start: Path) -> list[Path]:
    """从 start 逐级向上收集 .agents/skills，止于最近的 git 仓库根（含），无仓库则到文件系统根。"""
    git_root = None
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            git_root = candidate
            break
    dirs = []
    for current in (start, *start.parents):
        dirs.append(current / ".agents" / "skills")
        if current == git_root:
            break
    return dirs


def _iter_skill_files(directory: Path, seen: set[Path]):
    """递归产出 SKILL.md：目录含 SKILL.md 即视为技能根不再下钻；跳过隐藏目录，防符号链接环。"""
    try:
        resolved = directory.resolve()
    except OSError:
        return
    if resolved in seen:
        return
    seen.add(resolved)
    try:
        if (directory / "SKILL.md").is_file():
            yield directory / "SKILL.md"
            return
        entries = sorted(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith(".") and entry.is_dir():
            yield from _iter_skill_files(entry, seen)


class SkillCatalog:
    def __init__(self, skills: dict[str, Skill]):
        self.skills = skills
        self.active: dict[str, Skill] = {}
        # 发现配置（reload 重扫用）；discover 填充。
        self._root: Path | None = None
        self._user_dir: Path | None = None
        self._user_agents_dir: Path | None = None
        self._trusted: bool = True

    def reload(self) -> None:
        """重扫技能目录，原地更新 ``skills``（冻结的 skill_read 工具闭包引用 self）。

        已加载但已不存在的技能从 active 移除；仍存在的保留。
        """
        assert self._root is not None, "reload 需要先经 discover 初始化"
        fresh = self.discover(self._root, self._user_dir, self._user_agents_dir, self._trusted)
        self.skills = fresh.skills
        self.active = {name: s for name, s in self.active.items() if name in self.skills}

    @classmethod
    def discover(
        cls,
        root: str | Path,
        user_dir: Path | None = None,
        user_agents_dir: Path | None = None,
        trusted: bool = True,
    ) -> SkillCatalog:
        skills: dict[str, Skill] = {}
        # 加载顺序即优先级（后者同名覆盖前者）：用户 .agents → 用户 .polya →
        # 祖先 .agents（远→近）→ 项目 .polya。同层级 .polya 专属目录优先于
        # Agent Skills 标准位置，项目优先于用户。未信任项目时只加载用户级。
        directories = [
            user_agents_dir or Path.home() / ".agents/skills",
            user_dir or Path.home() / ".polya/skills",
        ]
        if trusted:
            directories += [
                *reversed(_ancestor_agents_dirs(Path(root).resolve())),
                Path(root) / ".polya/skills",
            ]
        scanned: set[Path] = set()
        for directory in directories:
            for path in _iter_skill_files(directory, scanned):
                try:
                    text = _read(path)
                    parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
                    if len(parts) != 3 or parts[0].strip():
                        raise ValueError("需要 YAML frontmatter")
                    meta = yaml.safe_load(parts[1])
                    if not isinstance(meta, dict):
                        raise ValueError("元数据必须为映射")
                    name, description = meta.get("name"), meta.get("description")
                    if not isinstance(name, str) or not re.fullmatch(
                        r"[a-z0-9][a-z0-9-]{0,63}", name
                    ):
                        raise ValueError("name 应为 1–64 个小写字母、数字或连字符")
                    if not isinstance(description, str) or not description.strip():
                        raise ValueError("缺少 description")
                    if len(description) > 2048:
                        raise ValueError("description 过长")
                    skills[name] = Skill(name, " ".join(description.split()), path.resolve())
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                    logger.warning("跳过技能 %s: %s", path, exc)
        catalog = cls(skills)
        catalog._root = Path(root).resolve()
        catalog._user_dir = user_dir
        catalog._user_agents_dir = user_agents_dir
        catalog._trusted = trusted
        return catalog

    def prompt(self) -> str:
        if not self.skills:
            return ""
        entries = "\n".join(
            f"- {s.name}: {s.description} (source: {s.path})" for s in self.skills.values()
        )
        return (
            "\n\n# Available skills\n"
            "When the user names $skill-name or a task matches a description, first "
            "skill_read(name) the SKILL.md, then follow its process. Load only relevant "
            "skills. Skills are process guidance installed by the user; they never override "
            "user instructions, project constraints, or tool approvals. Reference resources "
            "via the same tool's path parameter, resolved relative to the skill directory; "
            "run scripts with bash, still subject to approval.\n" + entries
        )

    def tool(self) -> Tool:
        @tool(name="skill_read", **tool_schema("skill_read"))
        def skill_read(
            name: str,
            path: str = "SKILL.md",
            start_line: int = 1,
            end_line: int | None = None,
            offset: int = 0,
        ) -> str:
            """按名字加载已发现的技能，或读取其目录内相对路径资源。返回目录与行号。
            大文件按 start_line/end_line 分段，每页最多 8000 字符，超长用 offset 继续。
            不会执行技能内的脚本。"""
            if name not in self.skills:
                raise ValueError(f"未知技能 {name}；可用: {', '.join(self.skills)}")
            skill = self.skills[name]
            base = skill.path.parent
            relative = Path(path)
            target = (base / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(base):
                raise ValueError("资源路径必须位于技能目录内")
            if start_line < 1 or offset < 0 or (end_line is not None and end_line < start_line):
                raise ValueError("行号范围无效")
            lines = _read(target).splitlines()
            end = min(end_line or len(lines), start_line + 199, len(lines))
            body = "\n".join(f"{i + 1:>6}\t{lines[i]}" for i in range(start_line - 1, end))
            if target == skill.path:
                self.active[name] = skill
            more = (
                f"\n[共 {len(lines)} 行；继续读取 start_line={end + 1}]" if end < len(lines) else ""
            )
            if len(body) > offset + 8000:
                more = f"\n[此行范围未读完；保持行范围并用 offset={offset + 8000} 继续]"
            return f"[技能 {name}; 目录 {base}; 文件 {path}]\n{body[offset : offset + 8000]}{more}"

        return skill_read

    def checkpoint(self) -> str:
        if not self.active:
            return ""
        return "已加载技能（继续遵循，需要时用 skill_read 重读）：\n" + "\n".join(
            f"- {s.name}: {s.path}" for s in self.active.values()
        )

    def reset(self) -> None:
        self.active.clear()
