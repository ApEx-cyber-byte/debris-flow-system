"""控制台编码 / Console encoding helpers.

Windows 控制台默认使用传统代码页（简体中文环境为 GBK/936）。Python 的
``sys.stdout`` 会按该代码页编码，直接 ``print`` 中文会出现乱码，甚至在遇到
代码页无法表示的字符时抛 ``UnicodeEncodeError`` 中断脚本。

本模块提供一个统一的入口，把标准输出/错误流切到 UTF-8。命令行脚本在
``sys.path`` 就绪后立即调用 :func:`use_utf8_console` 即可。

注意：这只是**输出端**的修正。PowerShell 脚本（``scripts/start.ps1``）
还有一个输入端的问题——PS 5.1 对无 BOM 的文件按 ANSI 代码页解码，
因此该文件必须以 **UTF-8 with BOM** 保存。
"""
from __future__ import annotations

import sys

_APPLIED = False


def use_utf8_console() -> bool:
    """Switch stdout/stderr to UTF-8. Returns True when applied.

    Safe to call repeatedly and safe to call on streams that cannot be
    reconfigured (e.g. a detached or already-wrapped stream).
    """
    global _APPLIED
    if _APPLIED:
        return True

    applied = False
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
            applied = True
        except (AttributeError, ValueError, OSError):
            # AttributeError: stream is not a TextIOWrapper (already wrapped,
            # or replaced by a test harness).
            continue

    _APPLIED = applied
    return applied
