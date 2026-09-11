"""端到端冒烟测试 / End-to-end smoke check.

启动一个真实的 uvicorn 服务，走完整条链路：
REST 建任务 → WebSocket 收二进制关键帧 → 解码校验 → 导出 CSV/JSON → AI 预测。

用法::

    cd backend
    python scripts/smoke_e2e.py                # 自动启动/关闭临时服务
    python scripts/smoke_e2e.py --url http://127.0.0.1:8000   # 用已有服务

退出码 0 表示全部通过。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.console import use_utf8_console  # noqa: E402

use_utf8_console()

from app.services.frames import decode_message  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [{PASS}] {name}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  [{FAIL}] {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 60.0):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        return response.status, (json.loads(body) if body else None)


def wait_for_server(base: str, attempts: int = 60) -> bool:
    for _ in range(attempts):
        try:
            status, _ = http_json("GET", f"{base}/health", timeout=2.0)
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(0.5)
    return False


async def read_stream(base: str, run_id: str, expect_frames: int, timeout: float) -> dict:
    """Consume the WebSocket stream and return a summary of what arrived."""
    import websockets

    ws_url = base.replace("http://", "ws://").replace("https://", "wss://")
    summary = {
        "meta": None,
        "frames": 0,
        "bed_length": 0,
        "completed": False,
        "first_frame": None,
        "last_stats": None,
        "binary_sizes": [],
    }

    async with websockets.connect(f"{ws_url}/ws/simulations/{run_id}", max_size=64 * 1024 * 1024) as socket:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                message = await asyncio.wait_for(socket.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            if isinstance(message, str):
                payload = json.loads(message)
                if payload.get("type") == "completed":
                    summary["completed"] = True
                    summary["metrics"] = payload.get("metrics", {})
                    break
                if payload.get("type") == "error":
                    summary["error"] = payload.get("message")
                    break
                continue

            try:
                header, arrays = decode_message(message)
            except Exception as error:  # noqa: BLE001 - report the raw bytes
                raw = bytes(message)
                print(f"        解码失败 {type(error).__name__}: {error}")
                print(f"        长度={len(raw)} 前 48 字节={raw[:48]!r}")
                print(f"        前 4 字节解出的 header_len={struct.unpack_from('<I', raw, 0)[0]}")
                raise
            summary["binary_sizes"].append(len(message))
            if header.get("type") == "meta":
                summary["meta"] = header
                summary["bed_length"] = int(arrays["bed"].size)
                continue

            if header.get("type") == "frame":
                summary["frames"] += 1
                summary["last_stats"] = header.get("stats")
                if summary["first_frame"] is None:
                    summary["first_frame"] = {
                        "pos": int(arrays["pos"].size),
                        "vel": int(arrays["vel"].size),
                        "depth": int(arrays["depth"].size),
                        "vx": int(arrays["vx"].size),
                        "vy": int(arrays["vy"].size),
                        "grid": header.get("grid"),
                        "step": header.get("step"),
                        "time": header.get("time"),
                    }
                if summary["frames"] >= expect_frames and summary["completed"]:
                    break

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端冒烟测试")
    parser.add_argument("--url", default=None, help="使用已运行的服务")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    server: subprocess.Popen | None = None
    base = args.url

    if base is None:
        base = f"http://127.0.0.1:{args.port}"
        print(f"启动临时后端 {base} ...")
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(args.port), "--log-level", "warning"],
            cwd=ROOT,
        )

    try:
        if not wait_for_server(base):
            print(f"{FAIL} 服务未能在 30 秒内就绪")
            return 1

        print("\n[1] 系统探测")
        _status, backends = http_json("GET", f"{base}/api/system/backends")
        check("GET /api/system/backends", isinstance(backends, dict))
        print(f"        PySPH={backends['pysph']['available']} "
              f"SynxFlow={backends['synxflow']['available']} "
              f"PyTorch={backends['torch']['available']}")
        check("回退说明已给出", len(backends["notes"]) >= 1,
              f"{len(backends['notes'])} 条提示")

        print("\n[2] 创建并启动仿真")
        params = {
            "slope_angle_deg": 32.0,
            "slope_length_m": 20.0,
            "rheology": "bingham",
            "yield_stress_pa": 450.0,
            "viscosity_pa_s": 60.0,
            "manning_n": 0.04,
            "density_kg_m3": 2200.0,
            "volume_concentration": 0.45,
            "initial_volume_m3": 150.0,
            "duration_s": args.duration,
            "particle_count": 1500,
            "backend": "auto",
            "frame_stride": 10,
        }
        status, created = http_json("POST", f"{base}/api/simulations", params)
        check("POST /api/simulations 返回 201", status == 201, f"status={status}")
        if status != 201:
            print(json.dumps(created, ensure_ascii=False, indent=2)[:800])
            return 1

        run_id = created["run_id"]
        print(f"        run_id={run_id}  后端={created['backend']['active']}  "
              f"栅格={created['grid']['nx']}x{created['grid']['ny']}  "
              f"坡脚x={created['terrain']['toe_x_m']:.2f}m")

        print("\n[3] WebSocket 关键帧流")
        summary = asyncio.run(read_stream(base, run_id, expect_frames=3, timeout=90.0))

        check("收到 meta 握手帧", summary["meta"] is not None)
        check("meta 含床面高程栅格", summary["bed_length"] > 0, f"{summary['bed_length']} 个格点")
        check("收到关键帧", summary["frames"] > 0, f"{summary['frames']} 帧")
        check("收到 completed 事件", summary["completed"])

        first = summary["first_frame"]
        if first:
            grid = first["grid"]
            cells = grid["nx"] * grid["ny"]
            check("位置数组长度为 3N", first["pos"] % 3 == 0, f"{first['pos']} = 3x{first['pos'] // 3}")
            check("速度数组与位置等长", first["vel"] == first["pos"])
            check("流深栅格尺寸正确", first["depth"] == cells, f"{first['depth']} vs {cells}")
            check("流速矢量栅格尺寸正确", first["vx"] == cells and first["vy"] == cells)
            check("关键帧步长为 frame_stride 的倍数",
                  first["step"] % 10 == 0, f"step={first['step']}")
            avg = sum(summary["binary_sizes"]) / max(len(summary["binary_sizes"]), 1)
            print(f"        平均二进制帧大小 {avg / 1024:.1f} KB")

        stats = summary["last_stats"] or {}
        check("诊断量齐全",
              all(k in stats for k in ("max_speed", "max_depth", "front_x", "impact_force")))
        print(f"        末帧: maxV={stats.get('max_speed', 0):.2f} m/s  "
              f"maxh={stats.get('max_depth', 0):.2f} m  "
              f"front={stats.get('front_x', 0):.1f} m  "
              f"F={stats.get('impact_force', 0) / 1000:.1f} kN")

        metrics = summary.get("metrics", {})
        check("跑距 > 0", metrics.get("runout_distance_m", 0) > 0,
              f"{metrics.get('runout_distance_m', 0):.2f} m")
        check("峰值流速 > 1 m/s", metrics.get("max_speed", 0) > 1.0,
              f"{metrics.get('max_speed', 0):.2f} m/s")

        print("\n[4] 导出")
        for kind, fmt in (("time_series", "csv"), ("time_series", "json"),
                          ("particles", "csv"), ("depth_profile", "csv"), ("full", "json")):
            try:
                with urllib.request.urlopen(
                    f"{base}/api/simulations/{run_id}/export?kind={kind}&fmt={fmt}", timeout=60
                ) as response:
                    body = response.read()
                    disposition = response.headers.get("Content-Disposition", "")
                check(f"导出 {kind}.{fmt}", len(body) > 0,
                      f"{len(body)} 字节, RFC5987={disposition.startswith(chr(97) + 'ttachment')}")
            except urllib.error.HTTPError as error:
                check(f"导出 {kind}.{fmt}", False, f"HTTP {error.code}: {error.read()[:200]!r}")

        print("\n[5] AI 预测")
        _status, models = http_json("GET", f"{base}/api/ai/models")
        check("GET /api/ai/models", isinstance(models.get("models"), list),
              f"{len(models['models'])} 个模型")
        status, prediction = http_json(
            "POST", f"{base}/api/ai/predict", {"params": params, "model_name": "heuristic-v1"}
        )
        check("POST /api/ai/predict", status == 200 and prediction is not None)
        if prediction:
            check("预测跑距 > 0", prediction["runout_distance_m"] > 0,
                  f"{prediction['runout_distance_m']:.1f} m")
            check("置信度在 (0,1]", 0 < prediction["confidence"] <= 1)
            print(f"        模型={prediction['model_name']}  "
                  f"跑距={prediction['runout_distance_m']:.1f} m  "
                  f"峰值流速={prediction['max_velocity_m_s']:.1f} m/s  "
                  f"冲击力={prediction['peak_impact_force_kn']:.1f} kN")

        print("\n[6] 参数校验")
        try:
            http_json("POST", f"{base}/api/simulations", {**params, "slope_angle_deg": 75.0})
            check("坡度角 75° 被拒", False, "竟然接受了")
        except urllib.error.HTTPError as error:
            check("坡度角 75° 被拒（规格上限 60）", error.code == 422, f"HTTP {error.code}")

        try:
            http_json("POST", f"{base}/api/simulations", {**params, "backend": "pysph"})
            check("未安装 PySPH 时显式请求被拒", False, "竟然接受了")
        except urllib.error.HTTPError as error:
            check("未安装 PySPH 时显式请求返回 409", error.code == 409, f"HTTP {error.code}")

    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()

    print("\n" + "=" * 66)
    if failures:
        print(f"\033[31m{len(failures)} 项失败:\033[0m")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("\033[32m全部通过 ✓\033[0m")
    _ = struct
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
