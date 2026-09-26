"""web_fetch 工具：抓取 URL 内容转成可读文本。

零第三方依赖：HTTP 用标准库 urllib（自动跟随重定向），HTML 解析用 HTMLParser。

注入防御（书 2.4「来源标记」）：网页是不可信外部内容，抓回的文本一律用
<external_content> 标签包裹、标注来源 URL 再进上下文，配合系统提示词中
「数据不是指令」的边界构成双保险。
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from http import HTTPStatus
from urllib.parse import urlparse

MAX_BYTES = 1_000_000
MAX_OUTPUT = 8000

_USER_AGENT = "polya-agent/0.1 (web_fetch tool)"


def _assert_public_url(url: str) -> None:
    """SSRF 防护：拒绝 localhost、内网、无域名主机——模型不应能借本工具探测内网。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"只支持 http/https URL: {url}")
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"URL 缺少主机名: {url}")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError(f"拒绝访问本地地址: {host}")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:  # 域名：解析后逐一校验，防止域名指向内网
            addresses = [ipaddress.ip_address(i[4][0]) for i in socket.getaddrinfo(host, None)]
        except socket.gaierror as exc:
            raise RuntimeError(f"域名解析失败: {host}") from exc
    for addr in addresses:
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise ValueError(f"拒绝访问内网/保留地址: {addr}")


class _TextExtractor(HTMLParser):
    """把 HTML 剥成纯文本：跳过 script/style，块级标签处断行。"""

    _SKIP = {"script", "style", "noscript", "head", "template"}
    _BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag in self._BREAK:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _html_to_text(body: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(body)
    return extractor.text()


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n... [已截断，完整内容共 {len(text)} 字符]"


def web_fetch_impl(url: str, timeout: int = 15) -> str:
    """抓取 url 并返回包裹在 <external_content> 里的正文文本。"""
    _assert_public_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            # urllib 不自动解压：显式要未压缩响应，否则 gzip 字节会被当文本解析
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 URL 来自模型
            if response.status != HTTPStatus.OK:
                raise RuntimeError(f"HTTP {response.status}: {url}")
            content_type = response.headers.get("content-type", "")
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read(MAX_BYTES).decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"请求失败: {exc.reason}") from exc

    if "html" in content_type:
        text = _html_to_text(body)
    else:
        text = body  # 纯文本 / JSON / Markdown 原样返回
    return (
        f'<external_content source="webpage" url="{url}">\n{_truncate(text)}\n</external_content>'
    )
