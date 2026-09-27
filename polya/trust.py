"""项目信任门：控制**项目资源**（AGENTS.md / 项目 skills）的加载。

对齐 pi 的 project trust 语义，2026-09-27 按 pi 模型升级：

- 信任门管项目资源，不管工具执行——未信任目录的文件指令（AGENTS.md 会被当指令注入
  系统提示词）不加载，防陌生仓库的提示注入；工具仍以进程权限在 root 内运行。
- 存储 ``~/.polya/trust.json``：``{"<abs path>": true | false | null}``。``true`` =
  信任，``false`` = 不信任，``null`` = 显式「无决定」（查找时跳过、回退父目录）。
  旧格式 ``{"trusted": [abs...]}`` 读取时迁移为 ``true``。
- **父目录继承**：:func:`nearest` 从目录逐级向上取最近的 true/false。
- **撤销**：写 ``false``（不信任）或删键（回退继承），见 :func:`set_decision`。
- 仅在目录或其祖先存在受保护资源时才需要决定（:func:`has_trust_requiring_resources`）。

未来的模型审查器是工具拦截层（``review.py``），与本门正交。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# polya 比 pi 更严：AGENTS.md 也在信任门内（pi 的 context 文件无条件加载）。
TRUST_REQUIRING_FILES = ("AGENTS.md",)
TRUST_REQUIRING_DIRS = (".polya/skills",)
# Agent Skills 标准位置（排除用户级 ~/.agents/skills）。
_ANCESTOR_SKILLS_DIR = ".agents/skills"

Decision = bool | None


def default_path() -> Path:
    return Path.home() / ".polya" / "trust.json"


def _canonical(root: str | os.PathLike[str]) -> str:
    return str(Path(root).resolve())


def load(path: Path | None = None) -> dict[str, Decision]:
    """读取信任表；文件缺失 / 损坏一律当作空表。旧格式迁移为 ``true``。"""
    target = path or default_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    legacy = data.get("trusted")
    if isinstance(legacy, list):  # 旧格式 {"trusted": [abs...]}
        return {str(item): True for item in legacy if isinstance(item, str)}
    store: dict[str, Decision] = {}
    for key, value in data.items():
        if value is True or value is False or value is None:
            store[str(key)] = value
    return store


def save(store: dict[str, Decision], path: Path | None = None) -> None:
    """写信任表（键排序、0600）；``null`` 值保留写盘（查找时跳过）。"""
    target = path or default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: store[key] for key in sorted(store)}
    target.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:  # Windows 等不支持时不强求
        pass


def nearest(root: str | os.PathLike[str], path: Path | None = None) -> tuple[str, bool] | None:
    """从 root 逐级向上找最近的 true/false 决定；``null`` 跳过。返回 (目录, 决定)。"""
    store = load(path)
    current = Path(root).resolve()
    while True:
        decision = store.get(str(current))
        if decision is True or decision is False:
            return str(current), decision
        parent = current.parent
        if parent == current:
            return None
        current = parent


def is_trusted(root: str | os.PathLike[str], path: Path | None = None) -> bool:
    found = nearest(root, path)
    return bool(found and found[1])


def set_decision(
    root: str | os.PathLike[str], decision: Decision, path: Path | None = None
) -> None:
    """写一个信任决定；``None`` 表示删键（回退父目录继承）。"""
    target = path or default_path()
    store = load(target)
    key = _canonical(root)
    if decision is None:
        store.pop(key, None)
    else:
        store[key] = decision
    save(store, target)


def trust(root: str | os.PathLike[str], path: Path | None = None) -> None:
    """信任目录（写 true）。"""
    set_decision(root, True, path)


def untrust(root: str | os.PathLike[str], path: Path | None = None) -> None:
    """标记不信任（写 false）。"""
    set_decision(root, False, path)


def forget(root: str | os.PathLike[str], path: Path | None = None) -> None:
    """删除该目录的决定（回退父目录继承）。"""
    set_decision(root, None, path)


def has_trust_requiring_resources(root: str | os.PathLike[str]) -> bool:
    """目录或其祖先是否存在受信任门保护的项目资源。

    资源 = AGENTS.md、``.polya/skills/``、或（cwd 或祖先的）``.agents/skills/``。
    用户级 ``~/.agents/skills`` 恒为可信用户资源，不计入。
    """
    home_agents = (Path.home() / _ANCESTOR_SKILLS_DIR).resolve()
    current = Path(root).resolve()
    while True:
        if any((current / name).is_file() for name in TRUST_REQUIRING_FILES):
            return True
        if any((current / name).exists() for name in TRUST_REQUIRING_DIRS):
            return True
        ancestor_skills = current / _ANCESTOR_SKILLS_DIR
        if ancestor_skills != home_agents and ancestor_skills.exists():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent
