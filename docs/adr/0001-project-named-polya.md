# 项目命名为 polya，仓库名 polya-agent

本项目是不发布 PyPI 的私有 CLI，命名的硬约束只有三条：可念（agent 会被人在对话中反复提起）、
有语义（名字应指向「问题求解代理」）、不与 PATH 命令 / Python import 名冲突。Pólya
（波利亚，《怎样解题》四步法：理解→计划→执行→回顾）与本项目的系统提示词同构，故内部
（包名、CLI 命令、品牌）统一用 polya；仓库名带 `-agent` 后缀以保留辨识度，并保住未来发布
PyPI 的选项——PyPI 上 `polya` 已被一个生信包（polyA）占用，`polya-agent` 空闲。

## Considered Options

- **mi-z（旧名）**：短且不撞名，但不可念（miz？mi-zed？）、零语义，中文语境强联想小米。
- **短英文词（tau / crow / jinn / iota / pion 等）**：双关优美，但 PyPI 全部被占；即便不发布，通用词的搜索噪音也大。
- **拼音系（mayi / mifeng）**：语义贴合、无撞名，但英文世界零共鸣。
- **希腊字母 / 粒子系（kaon / eta）**：nerdy 梗好，但与「问题求解」无关联。

若将来发布 PyPI：只需把 pyproject 的 name 改为 `polya-agent`（dist 名与 import 名本就允许不同，
如 beautifulsoup4→bs4），包名与命令不动——避免 dist / import / 命令三名并存的长期税。
