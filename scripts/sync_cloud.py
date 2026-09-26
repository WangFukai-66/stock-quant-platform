"""安全同步训练产物（预测/指标文件）到云端 cloud 分支。

仅做一件事：把白名单内的预测 CSV 与指标 JSON 提交并推送到 cloud 分支，
供 Streamlit Cloud 自动重建后向所有访问者展示。任何情况下都不触碰代码、
依赖、配置等其他文件，也不执行任何危险 git 命令。

用法:
    python -m scripts.sync_cloud 600519,000001
输出: 一行 JSON（ok / error / commit / files / output），供看板后台管理页回显。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SYNC_STATE = RESULTS_DIR / ".sync_state.json"

# 允许提交的文件前缀（与训练落盘命名一致）
_ALLOWED_NAMES = ("fusion", "xgb", "lstm", "transformer")
_ALLOWED_KINDS = ("predictions", "metrics")
_EXT = {"predictions": ".csv", "metrics": ".json"}


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """固定 cwd、无 shell 的子进程调用，输出全量捕获。"""
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _fail(msg: str, **extra) -> None:
    print(json.dumps({"ok": False, "error": msg, **extra}, ensure_ascii=False))
    sys.exit(1)


def collect_files(symbols: list[str]) -> list[Path]:
    """按白名单命名规则枚举存在的预测/指标文件。"""
    files: list[Path] = []
    for s in symbols:
        for name in _ALLOWED_NAMES:
            for kind in _ALLOWED_KINDS:
                p = RESULTS_DIR / f"{name}_{kind}_{s}{_EXT[kind]}"
                if p.exists():
                    files.append(p)
    return files


def main() -> None:
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    symbols = [s for s in re.findall(r"\d{6}", raw)]
    if not symbols:
        _fail("未提供有效的 6 位股票代码")

    files = collect_files(symbols)
    if not files:
        _fail(f"没有找到待同步的预测/指标文件（{', '.join(symbols)}）")
    rel_files = [str(f.relative_to(PROJECT_ROOT)) for f in files]

    # ---- 安全检查 1：当前分支必须是 cloud ----
    r = _run(["git", "symbolic-ref", "--short", "HEAD"], PROJECT_ROOT)
    if r.returncode != 0:
        _fail("无法读取当前分支：" + (r.stderr or r.stdout))
    branch = r.stdout.strip()
    if branch != "cloud":
        _fail(f"当前分支为 {branch}，同步仅允许在 cloud 分支执行")

    # ---- 安全检查 2：工作区只允许存在本次待同步文件的改动 ----
    r = _run(["git", "status", "--porcelain"], PROJECT_ROOT)
    if r.returncode != 0:
        _fail("git status 执行失败：" + (r.stderr or r.stdout))
    dirty = [line[3:] for line in r.stdout.splitlines() if line.strip()]
    others = [p for p in dirty if p not in rel_files]
    if others:
        _fail("工作区存在其他未提交改动，请先处理后再同步：" + ", ".join(others[:10]))

    # ---- 提交（命令白名单，硬编码参数，不拼接用户输入）----
    r = _run(["git", "add", "-f", *rel_files], PROJECT_ROOT)
    if r.returncode != 0:
        _fail("git add 失败：" + (r.stderr or r.stdout), step="add")
    msg = f"sync: 更新股票预测 {','.join(symbols)} ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
    r = _run(["git", "commit", "-m", msg], PROJECT_ROOT)
    if r.returncode != 0:
        _fail("git commit 失败：" + (r.stderr or r.stdout), step="commit")
    r = _run(["git", "push", "origin", "cloud"], PROJECT_ROOT)
    if r.returncode != 0:
        _fail("git push 失败：" + (r.stderr or r.stdout), step="push")

    # ---- 记录同步状态（供后台管理页展示"已同步云端"）----
    r = _run(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
    commit = r.stdout.strip()
    state: dict = {}
    if SYNC_STATE.exists():
        try:
            state = json.loads(SYNC_STATE.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    now = datetime.now()
    for s in symbols:
        s_files = collect_files([s])
        state[s] = {"commit": commit, "time": now.isoformat(timespec="seconds"),
                    "files": [f.name for f in s_files],
                    "mtime": max((f.stat().st_mtime for f in s_files), default=0)}
    SYNC_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "commit": commit, "symbols": symbols,
                      "files": rel_files, "output": r.stdout}, ensure_ascii=False))


if __name__ == "__main__":
    main()
