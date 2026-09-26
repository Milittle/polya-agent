"""Skills 按需加载、覆盖规则、路径边界与压缩后的再发现。"""

from pathlib import Path

import pytest

from polya.builtin import default_tools
from polya.cli import build_agent, parse_args
from polya.skills import SkillCatalog
from polya.tools import ToolRegistry


def make_skill(directory, name="develop", body="Read tests before editing.", description="Develop"):
    path = directory / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    return path


def test_agents_locations_recursive_discovery(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    make_skill(home / ".agents/skills", "global-helper")
    make_skill(root / ".agents/skills/docs/guides", "nested")  # 分类嵌套，递归发现
    packaged = make_skill(root / ".agents/skills", "packaged")
    inner = packaged.parent / "child"  # 技能根内不再下钻
    inner.mkdir()
    (inner / "SKILL.md").write_text("---\nname: inner\ndescription: x\n---\n")
    catalog = SkillCatalog.discover(root, home / ".polya/skills", home / ".agents/skills")
    assert set(catalog.skills) == {"global-helper", "nested", "packaged"}


def test_agents_ancestor_walk_stops_at_git_root(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    make_skill(tmp_path / ".agents/skills", "outside")  # 仓库根之上，不加载
    make_skill(repo / ".agents/skills", "far")
    make_skill(nested / ".agents/skills", "develop", body="NEAR")  # 近处同名覆盖远处
    catalog = SkillCatalog.discover(nested, home / ".polya/skills", home / ".agents/skills")
    assert set(catalog.skills) == {"far", "develop"}
    assert "NEAR" in catalog.tool().run({"name": "develop"})


def test_priority_project_polya_overrides_agents(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    make_skill(home / ".agents/skills", "develop", body="USER AGENTS")
    make_skill(home / ".polya/skills", "develop", body="USER POLYA")
    make_skill(root / ".agents/skills", "develop", body="PROJECT AGENTS")
    make_skill(root / ".polya/skills", "develop", body="PROJECT POLYA")
    catalog = SkillCatalog.discover(root, home / ".polya/skills", home / ".agents/skills")
    assert "PROJECT POLYA" in catalog.tool().run({"name": "develop"})
    catalog = SkillCatalog.discover(root / "sub", home / ".polya/skills", home / ".agents/skills")
    assert "PROJECT AGENTS" in catalog.tool().run({"name": "develop"})
    catalog = SkillCatalog.discover(
        tmp_path / "empty", home / ".polya/skills", home / ".agents/skills"
    )
    assert "USER POLYA" in catalog.tool().run({"name": "develop"})


def test_discovery_metadata_only_and_project_override(tmp_path):
    user = tmp_path / "user"
    root = tmp_path / "project"
    make_skill(user, body="USER BODY")
    make_skill(user, "review", description="'Review: code'")
    project = make_skill(
        root / ".polya/skills", body="PROJECT BODY", description=">\n  Project\n  development"
    )
    (root / ".git").mkdir()
    invalid = user / "broken/SKILL.md"
    invalid.parent.mkdir()
    invalid.write_text("---\nname: []\n---\nbroken")
    catalog = SkillCatalog.discover(root, user, tmp_path / "none")
    assert list(catalog.skills) == ["develop", "review"]
    assert catalog.skills["develop"].path == project.resolve()
    assert "Project development" in catalog.prompt()
    assert "PROJECT BODY" not in catalog.prompt()
    assert "USER BODY" not in catalog.prompt()
    assert "PROJECT BODY" in catalog.tool().run({"name": "develop"})
    assert "develop" in catalog.checkpoint()


def test_user_skill_resources_do_not_widen_project_read_access(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    user = tmp_path / "user"
    skill = make_skill(user)
    resource = skill.parent / "references/check.md"
    resource.parent.mkdir()
    resource.write_text("run related tests")
    (root / ".git").mkdir()
    catalog = SkillCatalog.discover(root, user, tmp_path / "none")
    reader = catalog.tool()
    assert "run related tests" in reader.run({"name": "develop", "path": "references/check.md"})
    with pytest.raises(ValueError):
        reader.run({"name": "develop", "path": "../../secret"})
    with pytest.raises(ValueError):
        reader.run({"name": "develop", "path": str(resource)})
    secret = tmp_path / "secret"
    secret.write_text("private")
    (skill.parent / "link").symlink_to(secret)
    with pytest.raises(ValueError):
        reader.run({"name": "develop", "path": "link"})
    registry = ToolRegistry(default_tools(root))
    assert registry.call("read_file", {"path": str(skill)}).startswith("Error:")


def test_skill_pagination_and_reset(tmp_path):
    make_skill(tmp_path / ".polya/skills", body="\n".join(f"step{i}" for i in range(250)))
    (tmp_path / ".git").mkdir()
    catalog = SkillCatalog.discover(tmp_path, tmp_path / "absent", tmp_path / "none")
    result = catalog.tool().run({"name": "develop"})
    assert "step249" not in result and "start_line=201" in result
    assert "step249" in catalog.tool().run({"name": "develop", "start_line": 201})
    catalog.reset()
    assert catalog.checkpoint() == ""


def test_cli_assembles_skills_and_static_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    make_skill(tmp_path / ".polya/skills")
    agent = build_agent(parse_args(["--root", str(tmp_path)]), llm=object())
    assert "develop" in agent.system_prompt
    prompt = agent.system_prompt
    schemas = agent.tools.schemas()
    agent.tools.call("skill_read", {"name": "develop"})
    assert agent.system_prompt == prompt
    assert agent.tools.schemas() == schemas
    assert agent.tools.get("history_read") is not None


def test_long_skill_line_can_be_read_without_silent_loss(tmp_path):
    make_skill(tmp_path / ".polya/skills", body="x" * 12000 + "END")
    (tmp_path / ".git").mkdir()
    reader = SkillCatalog.discover(tmp_path, tmp_path / "empty", tmp_path / "none").tool()
    first = reader.run({"name": "develop"})
    second = reader.run({"name": "develop", "offset": 8000})
    assert "offset=8000" in first and "END" not in first
    assert "END" in second
