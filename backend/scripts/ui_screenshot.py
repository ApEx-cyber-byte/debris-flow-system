"""UI 截图与前端冒烟检查 / Headless-Chrome UI verification.

用 Chrome DevTools Protocol 驱动无头浏览器：加载页面 → 点击「开始仿真」→
等待若干秒 → 截图。用于验证前端确实挂载、WebGL 可用、Three.js 有画面、
ECharts 有曲线，而不是只看构建是否通过。

用法::

    python scripts/ui_screenshot.py --url http://127.0.0.1:5173 \\
        --out ../docs/screenshots --run-seconds 9

依赖：本机已安装 Chrome 或 Edge（脚本自动探测），以及 ``websockets``
（已在后端 requirements 中）。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.console import use_utf8_console  # noqa: E402

use_utf8_console()

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

# SwiftShader gives headless Chrome a working WebGL implementation.
CHROME_FLAGS = [
    "--headless=new",
    "--disable-gpu",
    "--enable-unsafe-swiftshader",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--mute-audio",
    "--window-size=1920,1080",
]


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit("找不到 Chrome / Edge，无法进行 UI 截图验证。")


class CDP:
    """Minimal Chrome DevTools Protocol client."""

    def __init__(self, ws) -> None:  # noqa: ANN001
        self.ws = ws
        self.next_id = 1

    async def call(self, method: str, params: dict | None = None) -> dict:
        message_id = self.next_id
        self.next_id += 1
        await self.ws.send(
            json.dumps({"id": message_id, "method": method, "params": params or {}})
        )
        while True:
            raw = await self.ws.recv()
            payload = json.loads(raw)
            if payload.get("id") == message_id:
                if "error" in payload:
                    raise RuntimeError(f"{method}: {payload['error']}")
                return payload.get("result", {})

    async def evaluate(self, expression: str, await_promise: bool = False):
        result = await self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        return result.get("result", {}).get("value")

    async def screenshot(self, path: Path) -> int:
        result = await self.call(
            "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}
        )
        data = base64.b64decode(result["data"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return len(data)


async def run(args: argparse.Namespace) -> int:
    import websockets

    chrome = find_chrome()
    profile = tempfile.mkdtemp(prefix="dfs-ui-")
    port = args.debug_port
    process = subprocess.Popen(
        [chrome, *CHROME_FLAGS, f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         "about:blank"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        target = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json", timeout=2
                ) as response:
                    targets = json.loads(response.read())
                pages = [t for t in targets if t.get("type") == "page"]
                if pages:
                    target = pages[0]
                    break
            except Exception:  # noqa: BLE001 - Chrome is still starting
                pass
            time.sleep(0.5)

        if target is None:
            print("无法连接到 Chrome 调试端口")
            return 1

        print(f"Chrome 已启动，目标页面 {target['url']}")
        failures: list[str] = []

        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024) as ws:
            cdp = CDP(ws)
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")

            print(f"导航到 {args.url} ...")
            await cdp.call("Page.navigate", {"url": args.url})
            await asyncio.sleep(args.load_seconds)

            # --- structural assertions ------------------------------------
            title = await cdp.evaluate("document.title")
            print(f"  标题: {title}")
            if not title or "泥石流" not in title:
                failures.append("页面标题不正确")

            has_canvas = await cdp.evaluate(
                "!!document.querySelector('.viewport-canvas canvas')"
            )
            print(f"  Three.js canvas: {has_canvas}")
            if not has_canvas:
                failures.append("找不到 Three.js canvas")

            webgl = await cdp.evaluate(
                "(() => { const c = document.querySelector('.viewport-canvas canvas');"
                " if (!c) return null;"
                " const gl = c.getContext('webgl2') || c.getContext('webgl');"
                " return gl ? gl.getParameter(gl.VERSION) : null; })()"
            )
            print(f"  WebGL 版本: {webgl}")
            if not webgl:
                failures.append("WebGL 上下文不可用")

            buttons = await cdp.evaluate(
                "Array.from(document.querySelectorAll('.run-bar button')).map(b => b.textContent.trim())"
            )
            print(f"  运行控制按钮: {buttons}")
            if not buttons or "开始仿真" not in (buttons or []):
                failures.append("找不到「开始仿真」按钮")

            charts = await cdp.evaluate("document.querySelectorAll('.chart-canvas canvas').length")
            print(f"  ECharts 画布数: {charts}")
            if not charts or charts < 3:
                failures.append(f"ECharts 画布数应为 3，实际 {charts}")

            await cdp.screenshot(args.out / "01_initial.png")
            print(f"  已保存初始状态截图 01_initial.png")

            # --- start a run and let it stream -----------------------------
            print("点击「开始仿真」...")
            await cdp.evaluate(
                "(() => { const b = Array.from(document.querySelectorAll('.run-bar button'))"
                ".find(x => x.textContent.includes('开始仿真')); if (b) b.click(); })()"
            )
            await asyncio.sleep(args.run_seconds)

            status = await cdp.evaluate(
                "document.querySelector('.header-right .el-tag')?.textContent?.trim()"
            )
            frames = await cdp.evaluate(
                "document.querySelectorAll('.footer-stat')[1]?.textContent?.trim()"
            )
            print(f"  运行状态: {status} | {frames}")
            if status and status not in ("运行中", "已完成", "创建中"):
                failures.append(f"意外的运行状态: {status}")

            first_stat = await cdp.evaluate(
                "Array.from(document.querySelectorAll('.footer-stat')).map(e => e.textContent.trim())"
            )
            print(f"  状态栏: {first_stat}")

            await cdp.screenshot(args.out / "02_running.png")
            print("  已保存运行中截图 02_running.png")

            # --- switch to the AI tab -------------------------------------
            await cdp.evaluate(
                "(() => { const t = Array.from(document.querySelectorAll('.el-tabs__item'))"
                ".find(x => x.textContent.includes('AI 预测')); if (t) t.click(); })()"
            )
            await asyncio.sleep(1.0)
            await cdp.evaluate(
                "(() => { const b = Array.from(document.querySelectorAll('.ai-header button'))"
                "[0]; if (b) b.click(); })()"
            )
            await asyncio.sleep(2.0)
            prediction = await cdp.evaluate(
                "Array.from(document.querySelectorAll('.ai-metric')).map(e => e.textContent.trim())"
            )
            print(f"  AI 预测结果: {prediction}")
            if not prediction:
                failures.append("AI 预测面板没有结果")

            await cdp.screenshot(args.out / "03_ai.png")
            print("  已保存 AI 面板截图 03_ai.png")

            # --- console errors -------------------------------------------
            await cdp.evaluate(
                "(() => { const t = Array.from(document.querySelectorAll('.el-tabs__item'))"
                ".find(x => x.textContent.includes('日志')); if (t) t.click(); })()"
            )
            await asyncio.sleep(0.5)
            logs = await cdp.evaluate(
                "Array.from(document.querySelectorAll('.log-line')).slice(0, 6)"
                ".map(e => e.textContent.trim())"
            )
            print("  应用日志:")
            for line in logs or []:
                print(f"    {line}")

        if failures:
            print("\n失败项:")
            for item in failures:
                print(f"  - {item}")
            return 1
        print("\nUI 验证全部通过 ✓")
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="无头 Chrome UI 验证与截图")
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    parser.add_argument("--load-seconds", type=float, default=6.0)
    parser.add_argument("--run-seconds", type=float, default=9.0)
    parser.add_argument("--debug-port", type=int, default=9222)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
