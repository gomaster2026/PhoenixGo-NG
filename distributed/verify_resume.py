#!/usr/bin/env python3
"""断点续训本地验证入口（一键验证）。

本质是运行正式单元测试 training/tf/test_ckpt_utils.py：
  - 纯函数测试（resolve_start 三分支 / find_latest / prune / meta 等，无 TF 依赖）
  - 断点续训核心机制测试（checkpoint 保存/恢复 global_step 与 learning_rate，
    需要 TensorFlow；无 TF 时自动跳过，不算失败）

用法:
    python distributed/verify_resume.py [--venv-python PATH]

说明:
  - 需要 Python 3.9-3.12 + tensorflow==2.16.x + tf-keras==2.16.0
    （TF 2.16 起 Python 3.13 不受支持；TF 2.16 默认 Keras 3，
      legacy API 需 tf-keras 包，代码里已自动设 TF_USE_LEGACY_KERAS=1）
  - 默认使用当前解释器；Kaggle/远程环境请显式指定 --venv-python
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TF_DIR = ROOT / "training" / "tf"


def main():
    parser = argparse.ArgumentParser(description="断点续训本地验证")
    parser.add_argument("--venv-python", default=sys.executable,
                        help="带 tensorflow==2.16.x 的 Python 解释器路径")
    parser.add_argument("--verbose", action="store_true",
                        help="显示每个用例的详细信息")
    args = parser.parse_args()

    print("=" * 60)
    print("断点续训验证 (training/tf/test_ckpt_utils.py)")
    print("=" * 60, flush=True)

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["TF_USE_LEGACY_KERAS"] = "1"

    # 检查 TF 可用性（仅提示，不阻断）
    probe = subprocess.run([str(args.venv_python), "-c", "import tensorflow"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        print(f"TensorFlow: 可用 ({args.venv_python})\n", flush=True)
    else:
        print("TensorFlow: 不可用 —— 机制测试将自动跳过，仅跑纯函数测试")
        print("  (需 Python 3.9-3.12 安装: pip install tensorflow==2.16.1 tf-keras==2.16.0)\n",
              flush=True)

    cmd = [str(args.venv_python), "-m", "unittest", "test_ckpt_utils"]
    if args.verbose:
        cmd.append("-v")
    proc = subprocess.run(cmd, cwd=str(TF_DIR), env=env,
                          capture_output=True, text=True, timeout=1200)

    # 打印输出（过滤 TF 噪声）
    for line in (proc.stdout or "").splitlines():
        if not line.startswith("2026-"):
            print(line, flush=True)
    if proc.returncode != 0:
        for line in (proc.stderr or "").splitlines():
            if not line.startswith("2026-"):
                print(line, flush=True)

    print("\n" + "=" * 60)
    print("验证结果: {}".format("全部通过" if proc.returncode == 0 else "存在失败项"))
    print("=" * 60, flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
