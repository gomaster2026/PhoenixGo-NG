#!/usr/bin/env python3
"""
Kaggle T4 自对弈 + 训练循环脚本

在 Kaggle Notebook (T4 GPU) 上运行完整的强化学习循环:
  1. leelaz 用当前权重自对弈 → 生成训练数据 (chunk)
  2. TensorFlow 用 chunk 训练 → 输出新权重
  3. 用新权重重复步骤 1-2

全程在 Kaggle 内完成，不需要外部贡献者，不需要 Gitee。

用法 (Kaggle Notebook, 选 GPU T4 x2):
    !git clone --depth 1 https://gitee.com/ABCradio/AeonGo
    %cd AeonGo
    !pip install tensorflow==2.16.1
    !python distributed/kaggle_selfplay_loop.py \\
        --weights /kaggle/input/weights/phoenixgo-v1.txt.gz \\
        --iterations 5 \\
        --games-per-iter 50 \\
        --steps-per-iter 1000

参数说明:
    --weights          初始权重文件路径 (PhoenixGo v3 .txt.gz, ~211MB)
    --iterations       循环次数 (默认 5)
    --games-per-iter   每轮自对弈局数 (默认 50)
    --steps-per-iter   每轮训练步数 (默认 1000)
    --visits           每步 MCTS 访问数 (默认 800, 越大越强越慢)
    --threads          leelaz 搜索线程数 (默认 2)
    --use-cpu-leelaz   强制使用 CPU 版 leelaz (默认尝试 GPU/OpenCL)
    --keep-old-chunks  保留上一轮的训练数据 (默认清除, 每轮用新数据)
    --repo             仓库地址 (默认 gitee ABCradio/AeonGo)

注意:
    - TensorFlow 必须是 2.16.x (最后完整支持 tf.compat.v1 的版本)
    - Notebook 里 ! 命令必须写在一行, 不能用 \\ 换行
    - Kaggle T4 GPU session 限时约 9 小时, 请合理设置参数
    - 每轮迭代结束都会保存权重到 /kaggle/working/weights_iterN.txt.gz
    - 即使中途超时, 已完成的迭代权重不会丢失
"""

import argparse
import gzip
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── 路径常量 (Kaggle 环境) ──
WORK_DIR = Path("/kaggle/working")
SRC_DIR = WORK_DIR / "src"
TF_DIR = WORK_DIR / "tf"
CHUNKS_DIR = WORK_DIR / "chunks"
CKPT_DIR = WORK_DIR / "checkpoint"
LEELAZ_BIN = Path("/tmp/leelaz")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def log(msg):
    print(f"[loop] {msg}", flush=True)


def hr(title=""):
    line = "=" * 50
    log(f"{line} {title}" if title else line)


# ================================================================
# 1. 环境准备
# ================================================================

def clone_repo(repo_url):
    """克隆仓库, 复制训练代码到 TF_DIR"""
    if SRC_DIR.exists():
        shutil.rmtree(SRC_DIR)
    log(f"克隆仓库: {repo_url}")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", repo_url, str(SRC_DIR)],
        timeout=120,
    )
    if TF_DIR.exists():
        shutil.rmtree(TF_DIR)
    shutil.copytree(SRC_DIR / "training" / "tf", TF_DIR)
    log("训练代码已就绪")
    return SRC_DIR


def compile_leelaz(src_dir, use_gpu=True):
    """
    编译 leelaz。
    优先尝试 OpenCL (T4 GPU 推理), 失败则回退 CPU-only。
    编译一次后缓存到 /tmp/leelaz, 后续调用直接复用。
    """
    if LEELAZ_BIN.exists():
        log("leelaz 已编译, 跳过")
        return str(LEELAZ_BIN)

    log("安装编译依赖...")
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    pkgs = [
        "build-essential", "cmake",
        "libboost-all-dev", "zlib1g-dev", "libeigen3-dev",
    ]
    if use_gpu:
        pkgs.append("ocl-icd-opencl-dev")
    subprocess.run(
        ["apt-get", "install", "-y", "-qq"] + pkgs,
        check=False,
    )

    nproc = os.cpu_count() or 2

    # ── 尝试 OpenCL (GPU) 编译 ──
    if use_gpu:
        log("尝试 OpenCL (GPU) 编译...")
        build_gpu = src_dir / "build_gpu"
        build_gpu.mkdir(exist_ok=True)

        # 确保 NVIDIA OpenCL ICD 可被发现
        icd_dir = Path("/etc/OpenCL/vendors")
        icd_dir.mkdir(parents=True, exist_ok=True)
        nvidia_icd = icd_dir / "nvidia.icd"
        if not nvidia_icd.exists():
            # 查找 libnvidia-opencl 并写入 ICD 配置
            try:
                result = subprocess.run(
                    ["find", "/usr", "-name", "libnvidia-opencl.so*"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.stdout.strip():
                    nvidia_icd.write_text(result.stdout.strip().split("\n")[0])
                    log(f"写入 NVIDIA OpenCL ICD: {nvidia_icd}")
            except Exception:
                pass

        cmake = subprocess.run(
            ["cmake", ".."],
            cwd=str(build_gpu),
            capture_output=True, text=True, timeout=60,
        )
        if cmake.returncode == 0:
            make = subprocess.run(
                ["make", f"-j{nproc}"],
                cwd=str(build_gpu),
                capture_output=True, text=True, timeout=600,
            )
            if make.returncode == 0 and (build_gpu / "leelaz").exists():
                shutil.copy2(build_gpu / "leelaz", LEELAZ_BIN)
                log("OpenCL leelaz 编译成功 (GPU 推理)")
                return str(LEELAZ_BIN)
            else:
                log(f"OpenCL make 失败, 回退 CPU-only")
                if make.stderr:
                    log(f"  stderr: {make.stderr[-200:]}")
        else:
            log("OpenCL cmake 失败, 回退 CPU-only")
            if cmake.stderr:
                log(f"  stderr: {cmake.stderr[-200:]}")

    # ── CPU-only 编译 (保底) ──
    log("编译 CPU-only leelaz...")
    build_cpu = src_dir / "build_cpu"
    build_cpu.mkdir(exist_ok=True)

    cmake = subprocess.run(
        ["cmake", "..", "-DUSE_CPU_ONLY=1"],
        cwd=str(build_cpu),
        capture_output=True, text=True, timeout=60,
    )
    if cmake.returncode != 0:
        log(f"cmake 失败: {cmake.stderr[-300:]}")
        return None

    make = subprocess.run(
        ["make", f"-j{nproc}"],
        cwd=str(build_cpu),
        capture_output=True, text=True, timeout=600,
    )
    if make.returncode != 0:
        log(f"make 失败: {make.stderr[-300:]}")
        return None

    built = build_cpu / "leelaz"
    if built.exists():
        shutil.copy2(built, LEELAZ_BIN)
        log("CPU-only leelaz 编译成功")
        return str(LEELAZ_BIN)

    log("编译失败")
    return None


# ================================================================
# 2. 自对弈 — 生成训练数据 (支持多 GPU 并行)
# ================================================================

def _play_games(leelaz_path, weights_path, game_start, game_count,
                visits, threads, output_dir, gpu_id, prefix, results, lock):
    """
    在单个 GPU 上运行 leelaz 自对弈, 生成 game_count 局棋谱。
    leelaz 通过 --gpu 参数绑定到指定 OpenCL 设备。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        leelaz_path,
        "-w", str(weights_path),
        "-g",                  # GTP 模式
        "--noponder",
        "--noise",
        "-t", str(threads),
        "-r", "5",
        "-v", str(visits),
        "--quiet",
    ]
    # 绑定到指定 GPU (leelaz 支持 --gpu 参数)
    if gpu_id is not None:
        args.extend(["--gpu", str(gpu_id)])

    log(f"  [{prefix}] 启动 leelaz: {game_count} 局, GPU={gpu_id}")
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        cwd=str(output_dir),
    )
    time.sleep(8)

    def _readline(timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([proc.stdout], [], [], 0.1)
            if r:
                line = proc.stdout.readline()
                if not line:
                    return None
                return line.rstrip("\n")
        return None

    def send(cmd, timeout=300):
        # 引擎崩溃后 stdin 管道已关闭, write 会抛 BrokenPipeError,
        # 这里统一转为 None, 让上层按"引擎无响应"处理
        try:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return None
        lines = []
        while True:
            t = timeout if not lines else 2
            line = _readline(t)
            if line is None:
                return None
            if line == "":
                break
            lines.append(line)
        if not lines:
            return None
        first = lines[0]
        if first.startswith("?") or not first.startswith("="):
            return None
        result = first[1:].strip()
        if len(lines) > 1:
            rest = "\n".join(lines[1:])
            return result + "\n" + rest if result else rest
        return result

    def stop_engine():
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        except OSError:
            try:
                proc.kill()
                proc.wait()
            except OSError:
                pass

    chunk_files = []
    t0 = time.time()

    for i in range(game_count):
        g = game_start + i
        # clear_board 失败(超时/引擎无响应)时管道里残留旧响应, 若继续跑
        # genmove 会读到错位数据, 整局(含胜负标签)全部错乱 —— 必须重启引擎。
        if send("clear_board", timeout=10) is None:
            log(f"  [{prefix}] clear_board 无响应, 重启引擎后跳过本局 #{g}")
            stop_engine()
            proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                cwd=str(output_dir))
            time.sleep(8)
            continue
        passes = 0
        resigned = False
        last_to_move = "b"
        failed = False

        for j in range(400):
            color = "b" if j % 2 == 0 else "w"
            last_to_move = color
            move = send(f"genmove {color}", timeout=300)
            if move is None:
                # genmove 超时/引擎无响应 → 协议状态不可信, 本局作废,
                # 重启引擎保证下一局从干净的棋盘开始。
                failed = True
                break
            if move == "resign":
                resigned = True
                break
            if move == "pass":
                passes += 1
                if passes >= 2:
                    break
            else:
                passes = 0

        if failed:
            log(f"  [{prefix}] genmove 无响应, 重启引擎后跳过本局 #{g}")
            stop_engine()
            proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                cwd=str(output_dir))
            time.sleep(8)
            continue

        if resigned:
            winner = "w" if last_to_move == "b" else "b"
        else:
            score = send("final_score", timeout=10)
            if score and score.startswith("W+"):
                winner = "w"
            elif score and score.startswith("B+"):
                winner = "b"
            else:
                winner = "w"

        # 文件名带 GPU 前缀, 避免并行时冲突
        basename = f"{prefix}_sp_{g:06d}"
        send(f"dump_training {winner} {basename}", timeout=30)
        chunk_path = output_dir / f"{basename}.0.gz"
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunk_files.append(str(chunk_path))

        if (i + 1) % 10 == 0 or (i + 1) == game_count:
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            remaining = avg * (game_count - i - 1)
            log(f"  [{prefix}] 自对弈 {i+1}/{game_count} 局 "
                f"({elapsed/60:.1f}m, 剩余 {remaining/60:.1f}m)")

    send("quit", timeout=5)
    # terminate 后若进程不退出, wait 超时会抛 TimeoutExpired 导致子进程残留;
    # 兜底 kill 强制结束, 确保不泄漏进程
    stop_engine()

    with lock:
        results.extend(chunk_files)
    log(f"  [{prefix}] 完成: {len(chunk_files)}/{game_count} 局")


def selfplay(leelaz_path, weights_path, num_games, visits, threads,
             output_dir, num_gpus=1):
    """
    驱动 leelaz 自对弈, 支持多 GPU 并行。
    num_gpus > 1 时, 启动多个 leelaz 进程, 各绑一个 GPU, 并行生成棋谱。
    """
    if num_gpus <= 1:
        results = []
        lock = threading.Lock()
        log(f"单 GPU 自对弈: {num_games} 局, visits={visits}")
        _play_games(leelaz_path, weights_path, 1, num_games,
                    visits, threads, output_dir, None, "GPU0",
                    results, lock)
        log(f"自对弈完成: {len(results)}/{num_games} 个训练文件")
        return results

    # 多 GPU 并行: 每个 GPU 跑一部分局数
    log(f"多 GPU 并行自对弈: {num_games} 局, {num_gpus} GPU, visits={visits}")
    games_per_gpu = num_games // num_gpus
    remainder = num_games % num_gpus

    threads_list = []
    all_results = []
    lock = threading.Lock()

    current_start = 1
    for gpu_id in range(num_gpus):
        count = games_per_gpu + (1 if gpu_id < remainder else 0)
        if count == 0:
            continue
        prefix = f"GPU{gpu_id}"
        t = threading.Thread(
            target=_play_games,
            args=(leelaz_path, weights_path, current_start, count,
                  visits, threads, output_dir, gpu_id, prefix,
                  all_results, lock),
        )
        threads_list.append(t)
        current_start += count

    for t in threads_list:
        t.start()
    for t in threads_list:
        t.join()

    log(f"自对弈完成: {len(all_results)}/{num_games} 个训练文件")
    return all_results


# ================================================================
# 3. 权重格式转换 (.txt.gz → TF checkpoint)
# ================================================================

def detect_architecture(lines):
    """
    从权重文件识别网络结构, 返回 (blocks, filters) 或 (None, None)。

    v3 (PhoenixGo) 格式:
      每卷积块 6 行 (conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var)
      + trunk BN 4 行 + policy head 8 行 + value head 10 行
      总张量数 = 6 + 12*blocks + 4 + 8 + 10 = 28 + 12*blocks
      blocks = (total - 28) / 12
    """
    if not lines:
        return None, None
    version = lines[0].strip()
    tensors = [l for l in lines[1:] if l.strip()]
    if len(tensors) < 2:
        return None, None

    filters = len(tensors[1].split())
    if filters <= 0:
        return None, None

    input_conv = tensors[0].split()
    # 本 fork 输入为 17 通道 (PhoenixGo: 16 手历史 + 1 颜色)
    if len(input_conv) != filters * 17 * 9:
        return None, None

    if version == "3":
        if len(tensors) < 28:
            return None, None
        blocks = (len(tensors) - 28) // 12
        if blocks < 1 or (len(tensors) - 28) % 12 != 0:
            return None, None
        return blocks, filters

    return None, None


def convert_weights_to_checkpoint(tf_dir, weights_gz_path, blocks, filters, ckpt_dir):
    """
    将 .txt.gz 权重文件转为 TF checkpoint (供 parse.py --restore)。
    返回 (checkpoint_path, blocks, filters)。
    """
    tf_dir = Path(tf_dir)
    if str(tf_dir) not in sys.path:
        sys.path.insert(0, str(tf_dir))

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tfprocess import TFProcess

    weights_gz_path = Path(weights_gz_path)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(weights_gz_path, "rt") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    version = lines[0].strip() if lines else "0"
    log(f"权重版本: {version}")

    if version != "3":
        raise ValueError(
            f"不支持的权重版本: '{version}'。"
            f"训练端仅支持 version=3 (PhoenixGo 预激活结构)。"
        )

    detected_blocks, detected_filters = detect_architecture(lines)
    if blocks is None and detected_blocks is not None:
        blocks, filters = detected_blocks, detected_filters
        log(f"自动识别网络结构: blocks={blocks}, filters={filters}")
    if blocks is None or filters is None:
        raise ValueError("无法识别网络结构, 请用 --blocks/--filters 手动指定")

    # 解析权重数值
    new_weights = []
    for line in lines[1:]:
        if line:
            new_weights.append(list(map(float, line.split(" "))))

    log(f"加载 {len(new_weights)} 个权重张量, "
        f"创建 TF 图 (blocks={blocks}, filters={filters})...")

    tfprocess = TFProcess(blocks, filters)
    tfprocess.init(batch_size=1, gpus_num=1)
    tfprocess.replace_weights(new_weights)

    ckpt_path = str(ckpt_dir / "initial")
    saved = tfprocess.saver.save(tfprocess.session, ckpt_path, global_step=0)
    log(f"TF checkpoint: {saved}")
    tfprocess.session.close()
    return saved, blocks, filters


# ================================================================
# 4. 训练 — 调用 parse.py
# ================================================================

def run_training(tf_dir, chunks_dir, checkpoint_path, output_path,
                 blocks, filters, max_steps):
    """
    调用 parse.py 训练, 输出 .txt.gz 权重。
    返回 True/False。
    """
    parse_script = Path(tf_dir) / "parse.py"
    if not parse_script.exists():
        log(f"parse.py 不存在: {parse_script}")
        return False

    chunk_dir = Path(chunks_dir)
    gz_files = sorted(chunk_dir.glob("*.gz"))
    if not gz_files:
        log(f"错误: {chunks_dir} 中没有 .gz 训练数据")
        return False

    # 统一重命名 chunk 文件, 确保 get_chunks 能读到
    for i, f in enumerate(gz_files):
        new_name = chunk_dir / f"t{i:06d}.gz"
        if new_name != f:
            shutil.move(str(f), str(new_name))

    train_prefix = str(chunk_dir) + "/t"
    cmd = [
        sys.executable, str(parse_script),
        "--blocks", str(blocks),
        "--filters", str(filters),
        "--train", train_prefix,
        "--max-steps", str(max_steps),
    ]
    if checkpoint_path:
        cmd.extend(["--restore", str(checkpoint_path)])

    log(f"开始训练: blocks={blocks}, filters={filters}, "
        f"steps={max_steps}, chunks={len(gz_files)}")

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    result = subprocess.run(cmd, cwd=str(tf_dir), env=env)
    if result.returncode != 0:
        log("训练进程返回非零退出码")
        return False

    # parse.py 在 cwd 生成 leelaz-model-final.txt
    final_txt = Path(tf_dir) / "leelaz-model-final.txt"
    if not final_txt.exists():
        log(f"训练完成但未找到输出: {final_txt}")
        return False

    # 压缩为 .txt.gz
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with final_txt.open("rb") as src, gzip.open(str(output_path), "wb") as dst:
        shutil.copyfileobj(src, dst)
    log(f"输出权重: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return True


# ================================================================
# 5. 主循环
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kaggle T4 自对弈 + 训练循环 (强化学习闭环)")
    parser.add_argument("--weights", required=True,
                        help="初始权重文件路径 (PhoenixGo v3 .txt.gz)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="循环次数 (默认 5)")
    parser.add_argument("--games-per-iter", type=int, default=50,
                        help="每轮自对弈局数 (默认 50)")
    parser.add_argument("--steps-per-iter", type=int, default=1000,
                        help="每轮训练步数 (默认 1000)")
    parser.add_argument("--visits", type=int, default=1200,
                        help="每步 MCTS 访问数 (默认 1200)")
    parser.add_argument("--threads", type=int, default=2,
                        help="leelaz 搜索线程数 (默认 2)")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="并行 GPU 数 (默认 2, Kaggle T4 x2)")
    parser.add_argument("--use-cpu-leelaz", action="store_true",
                        help="强制使用 CPU 版 leelaz (默认尝试 GPU/OpenCL)")
    parser.add_argument("--keep-old-chunks", action="store_true",
                        help="保留上一轮训练数据 (默认每轮清除)")
    parser.add_argument("--blocks", type=int, default=None,
                        help="残差块数 (默认自动从权重识别)")
    parser.add_argument("--filters", type=int, default=None,
                        help="卷积滤波器数 (默认自动从权重识别)")
    parser.add_argument("--repo",
                        default="https://gitee.com/ABCradio/AeonGo",
                        help="仓库地址")
    args = parser.parse_args()

    # ── 检查初始权重 ──
    weights_path = Path(args.weights)
    if not weights_path.exists():
        log(f"错误: 初始权重不存在: {weights_path}")
        sys.exit(1)
    log(f"初始权重: {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")

    # ── 1. 环境准备 ──
    hr("环境准备")
    clone_repo(args.repo)
    leelaz = compile_leelaz(SRC_DIR, use_gpu=not args.use_cpu_leelaz)
    if not leelaz:
        log("错误: leelaz 编译失败, 无法自对弈")
        sys.exit(1)

    # ── 初始化路径 ──
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    current_weights = weights_path
    blocks = args.blocks
    filters = args.filters

    total_start = time.time()

    # ── 2. 强化学习循环 ──
    for iteration in range(1, args.iterations + 1):
        hr(f"迭代 {iteration}/{args.iterations}")
        iter_start = time.time()

        # ── 2a. 自对弈生成训练数据 ──
        log(f"阶段 1/2: 自对弈 ({args.games_per_iter} 局)")

        # 清除上一轮的 chunk (除非指定保留)
        if not args.keep_old_chunks and iteration > 1:
            old_chunks = list(CHUNKS_DIR.glob("*.gz"))
            for f in old_chunks:
                f.unlink()
            log(f"已清除 {len(old_chunks)} 个旧 chunk")

        chunks = selfplay(
            leelaz, str(current_weights),
            args.games_per_iter, args.visits, args.threads,
            CHUNKS_DIR, num_gpus=args.num_gpus,
        )

        if not chunks:
            log("错误: 本轮未生成任何训练数据, 跳过训练")
            continue

        log(f"本轮共 {len(chunks)} 个训练文件")

        # ── 2b. 训练新权重 ──
        log(f"阶段 2/2: 训练 ({args.steps_per_iter} 步)")

        # 转换当前权重为 checkpoint (每次都重新转换, 确保从最新权重继续)
        try:
            ckpt_path, blocks, filters = convert_weights_to_checkpoint(
                TF_DIR, current_weights, blocks, filters, CKPT_DIR)
            log(f"从基础权重继续训练 (blocks={blocks}, filters={filters})")
        except Exception as e:
            # 权重转换失败是致命错误: current_weights 不变, continue 会导致
            # 每轮都自对弈后再转换同一份权重、再失败, 在 Kaggle 上白烧 GPU 时长。
            log(f"致命错误: 权重转换失败: {e}")
            sys.exit(1)

        # 训练
        output_path = WORK_DIR / f"weights_iter{iteration}.txt.gz"
        success = run_training(
            tf_dir=TF_DIR,
            chunks_dir=CHUNKS_DIR,
            checkpoint_path=ckpt_path,
            output_path=output_path,
            blocks=blocks,
            filters=filters,
            max_steps=args.steps_per_iter,
        )

        if not success or not output_path.exists():
            log(f"迭代 {iteration} 训练失败, 继续下一轮")
            continue

        # 更新当前权重
        current_weights = output_path
        iter_time = (time.time() - iter_start) / 60
        total_time = (time.time() - total_start) / 60
        log(f"迭代 {iteration} 完成: {iter_time:.1f}m "
            f"(累计 {total_time:.1f}m)")
        log(f"新权重: {current_weights}")

    # ── 3. 最终输出 ──
    hr("全部完成")
    total_time = (time.time() - total_start) / 60
    log(f"总耗时: {total_time:.1f} 分钟")
    log(f"最终权重: {current_weights}")

    # 复制最终权重到标准路径 (方便 Kaggle Output 下载)
    final_output = WORK_DIR / "weights_final.txt.gz"
    if current_weights != final_output:
        shutil.copy2(str(current_weights), str(final_output))
        log(f"最终权重已复制到: {final_output}")
        log(f"可从 Notebook Output 面板下载")


if __name__ == "__main__":
    main()
