#!/usr/bin/env python3
"""
Leela Zero Kaggle 训练脚本（简洁版）
=====================================

专为 Kaggle Notebook (T4 GPU) 设计的一键训练脚本。
从基础权重继续训练，用 Gitee 数据仓库的 chunk 或自对弈数据。

用法（在 Kaggle Notebook 中，选 GPU T4 x2）:

    !git clone --depth 1 https://gitee.com/ABCradio/AeonGo
    %cd AeonGo
    !pip install tensorflow==2.16.1
    !python distributed/kaggle_train_simple.py \
        --weights-path /kaggle/input/weights/phoenixgo-v1.txt.gz \
        --data-owner ABCradio --data-repo shuju \
        --data-token 你的Gitee令牌 \
        --max-steps 1000

流程:
  1. 从 Gitee 数据仓库下载训练 chunk（贡献者上传的自对弈数据）
  2. （可选）编译 leelaz 自对弈补充训练数据
  3. 基础权重 .txt.gz → TF checkpoint（自动识别网络结构）
  4. parse.py 训练（PhoenixGo v3 预激活残差网络）
  5. 输出新权重到 /kaggle/working/weights_new.txt.gz
  6. （可选）上传新权重到 Gitee Release + 清理旧 chunk
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "training", "tf"))

import gitee_data
import weight_release
import ckpt_utils


# ============================================================
#  工具函数
# ============================================================

def log(msg):
    """带时间戳的日志输出"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def elapsed(start, msg=""):
    """打印耗时"""
    sec = time.time() - start
    if sec < 60:
        log(f"{msg} ({sec:.1f}s)")
    else:
        log(f"{msg} ({sec/60:.1f}min)")
    return time.time()


# ============================================================
#  第1步: 克隆仓库 + 准备训练代码
# ============================================================

def clone_repo(repo_url, dest_dir):
    """克隆仓库（浅克隆），返回仓库路径"""
    dest = Path(dest_dir)
    if dest.exists():
        shutil.rmtree(dest)
    log(f"克隆仓库: {repo_url}")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        timeout=120,
    )
    return dest


def setup_tf_dir(repo_dir, tf_dest):
    """把 training/tf 目录复制到工作区"""
    src = Path(repo_dir) / "training" / "tf"
    if not src.exists():
        log(f"错误: 训练代码目录不存在: {src}")
        sys.exit(1)
    tf_dest = Path(tf_dest)
    if tf_dest.exists():
        shutil.rmtree(tf_dest)
    shutil.copytree(src, tf_dest)
    log(f"训练代码已复制到: {tf_dest}")
    return tf_dest


# ============================================================
#  第2步: 准备基础权重
# ============================================================

def prepare_weights(args, weights_dir):
    """
    确定基础权重文件路径。
    优先级: --weights-path 本地文件 > Gitee/GitHub Release 下载
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 方式1: 使用 Kaggle Input 中的本地权重文件
    if args.weights_path and Path(args.weights_path).exists():
        log(f"使用本地基础权重: {args.weights_path}")
        return Path(args.weights_path)

    # 方式2: 从 Gitee/GitHub Release 下载最新权重
    if args.weight_owner and args.weight_repo:
        latest = weights_dir / "current.txt.gz"
        log(f"从 {args.weight_platform} 下载最新权重: {args.weight_owner}/{args.weight_repo}")
        if args.weight_platform == "gitee":
            ok = weight_release.download_gitee(
                args.weight_owner, args.weight_repo, latest)
        else:
            ok = weight_release.download_github(
                args.weight_owner, args.weight_repo, latest)
        if ok and latest.stat().st_size > 100_000:
            log(f"下载成功: {latest} ({latest.stat().st_size/1e6:.1f} MB)")
            return latest
        log("下载失败")
        return None

    log("错误: 没有指定基础权重")
    log("  首次训练用 --weights-path 指定初始权重（如 phoenixgo-v1.txt.gz）")
    log("  后续训练用 --weight-owner/--weight-repo 从 Release 下载")
    return None


# ============================================================
#  第3步: 收集训练数据
# ============================================================

def download_chunks(data_owner, data_repo, dest_dir, token="", limit=None):
    """从 Gitee 数据仓库下载所有 chunk，返回文件路径列表"""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    log(f"列出 Gitee 数据仓库 {data_owner}/{data_repo} 的 chunks/ ...")
    chunks = gitee_data.list_chunks(data_owner, data_repo, token=token)
    if not chunks:
        log("数据仓库暂无 chunk（还没有贡献者上传数据）")
        return []

    log(f"共 {len(chunks)} 个 chunk")
    if limit:
        chunks = chunks[:limit]
        log(f"限制下载前 {limit} 个")

    downloaded = []
    failed = 0
    t0 = time.time()
    for i, c in enumerate(chunks, 1):
        name = c.get("name")
        dst = dest / name
        # 跳过已下载的文件（断点续传）
        if dst.exists() and dst.stat().st_size > 0:
            downloaded.append(str(dst))
            continue
        try:
            if gitee_data.download_chunk(data_owner, data_repo, name, dst, token=token):
                if dst.stat().st_size > 0:
                    downloaded.append(str(dst))
                else:
                    dst.unlink(missing_ok=True)
                    failed += 1
            else:
                failed += 1
        except Exception as e:
            log(f"  下载失败 {name}: {e}")
            failed += 1

        if i % 50 == 0 or i == len(chunks):
            log(f"  下载进度: {i}/{len(chunks)} (成功 {len(downloaded)}, 失败 {failed})")

    elapsed(t0, f"下载完成: {len(downloaded)} 成功, {failed} 失败")
    return downloaded


def compile_leelaz(src_dir):
    """在 Kaggle Linux 环境编译 leelaz（CPU-only，Kaggle 无 OpenCL 驱动）"""
    leelaz_path = Path("/tmp/leelaz")
    if leelaz_path.exists():
        log("leelaz 已编译，跳过")
        return str(leelaz_path)

    log("安装编译依赖...")
    subprocess.run(["apt-get", "update", "-qq"], check=False, capture_output=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq",
         "build-essential", "cmake",
         "libboost-all-dev", "zlib1g-dev", "libeigen3-dev"],
        check=False, capture_output=True,
    )

    log("编译 leelaz (CPU-only)...")
    build_dir = Path(src_dir) / "build"
    build_dir.mkdir(exist_ok=True)
    nproc = os.cpu_count() or 2

    cmake = subprocess.run(
        ["cmake", "..", "-DUSE_CPU_ONLY=1"],
        cwd=str(build_dir), timeout=60, capture_output=True, text=True,
    )
    if cmake.returncode != 0:
        log(f"cmake 失败: {cmake.stderr[-300:]}")
        return None

    make = subprocess.run(
        ["make", f"-j{nproc}"],
        cwd=str(build_dir), timeout=600, capture_output=True, text=True,
    )
    if make.returncode != 0:
        log(f"make 失败: {make.stderr[-300:]}")
        return None

    built = build_dir / "leelaz"
    if built.exists():
        shutil.copy2(built, leelaz_path)
        log("leelaz 编译成功")
        return str(leelaz_path)
    log("编译产物未找到")
    return None


def selfplay(leelaz_path, weights_path, num_games, visits, output_dir):
    """用 leelaz 自对弈生成训练数据（含 MCTS 搜索概率）"""
    import select
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        leelaz_path, "-w", str(weights_path),
        "-g", "--noponder", "--noise",
        "-t", "1", "-r", "5",
        "-v", str(visits), "--quiet",
    ]

    log(f"自对弈 {num_games} 局 (visits={visits})...")
    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
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

    def send(cmd, timeout=120):
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
        # terminate 后若进程不退出, wait 超时会抛 TimeoutExpired 导致子进程残留;
        # 兜底 kill 强制结束, 确保不泄漏进程
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
    for g in range(1, num_games + 1):
        # clear_board 失败(超时/引擎无响应)时管道里残留旧响应, 若继续跑
        # genmove 会读到错位数据, 整局(含胜负标签)全部错乱 —— 必须重启引擎。
        if send("clear_board", timeout=10) is None:
            log(f"clear_board 无响应, 重启引擎后跳过本局 #{g}")
            stop_engine()
            proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                cwd=str(output_dir))
            time.sleep(8)
            continue
        passes = 0
        resigned = False
        last_color = "b"
        failed = False
        for i in range(400):
            color = "b" if i % 2 == 0 else "w"
            last_color = color
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
            log(f"genmove 无响应, 重启引擎后跳过本局 #{g}")
            stop_engine()
            proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                cwd=str(output_dir))
            time.sleep(8)
            continue

        # 判断胜负
        if resigned:
            winner = "w" if last_color == "b" else "b"
        else:
            score = send("final_score", timeout=10)
            if score and score.startswith("W+"):
                winner = "w"
            elif score and score.startswith("B+"):
                winner = "b"
            else:
                winner = "w"

        # 导出训练数据
        basename = f"sp_{g:06d}"
        send(f"dump_training {winner} {basename}", timeout=30)
        chunk_path = output_dir / f"{basename}.0.gz"
        if chunk_path.exists():
            chunk_files.append(str(chunk_path))

        if g % 5 == 0 or g == num_games:
            elapsed(t0, f"  自对弈进度: {g}/{num_games} 局")

    send("quit", timeout=5)
    stop_engine()
    log(f"自对弈完成: 生成 {len(chunk_files)} 个训练文件")
    return chunk_files


# ============================================================
#  第4步: 权重 → TF checkpoint
# ============================================================

def detect_architecture(lines):
    """
    从权重文件识别网络结构，返回 (blocks, filters)。
    仅支持 v3 (PhoenixGo) 格式:
      每卷积块 6 行 (conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var)
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
    # 17 通道 (PhoenixGo: 16 手历史 + 1 颜色)
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
    将 .txt.gz 权重文件转为 TF checkpoint（供 parse.py --restore）。
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
            f"不支持的权重版本: '{version}'。训练端仅支持 v3 (PhoenixGo 预激活) 结构。"
            f"请使用 PhoenixGo v3 格式权重。")

    detected_blocks, detected_filters = detect_architecture(lines)
    if blocks is None and detected_blocks is not None:
        blocks, filters = detected_blocks, detected_filters
        log(f"自动识别网络结构: blocks={blocks}, filters={filters}")
    if blocks is None or filters is None:
        raise ValueError("无法识别网络结构，请用 --blocks/--filters 手动指定")

    # 解析权重数值
    new_weights = []
    for line in lines[1:]:
        if line:
            new_weights.append(list(map(float, line.split(" "))))

    log(f"加载 {len(new_weights)} 个权重张量, 构建 TF 图 (blocks={blocks}, filters={filters})...")
    tfprocess = TFProcess(blocks, filters)
    tfprocess.init(batch_size=1, gpus_num=1)
    try:
        tfprocess.replace_weights(new_weights)

        ckpt_path = str(ckpt_dir / "leelaz-model")
        saved = tfprocess.saver.save(tfprocess.session, ckpt_path, global_step=0)
        log(f"TF checkpoint 已保存: {saved}")
    finally:
        # 异常时也关闭 session，避免 TF 图/显存泄漏
        tfprocess.session.close()
    return saved, blocks, filters


# ============================================================
#  第5步: 训练
# ============================================================

def run_training(tf_dir, chunks_dir, checkpoint_path, output_path,
                 blocks, filters, max_steps, ckpt_dir=None,
                 save_every=8000, keep_n=5, lr_schedule=None):
    """调用 parse.py 训练，输出新权重"""
    parse_script = Path(tf_dir) / "parse.py"
    if not parse_script.exists():
        log(f"错误: parse.py 不存在: {parse_script}")
        return False

    # 统一 chunk 文件名前缀（parse.py 用 glob 匹配）
    chunk_dir = Path(chunks_dir)
    gz_files = sorted(chunk_dir.glob("*.gz"))
    if not gz_files:
        log(f"错误: {chunks_dir} 中没有训练数据")
        return False

    log(f"重命名 {len(gz_files)} 个 chunk 文件...")
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
        "--save-every", str(save_every),
        "--keep-n", str(keep_n),
    ]
    if ckpt_dir:
        cmd.extend(["--ckpt-dir", str(ckpt_dir)])
    if checkpoint_path:
        cmd.extend(["--restore", str(checkpoint_path)])
    if lr_schedule:
        cmd.extend(["--lr-schedule",
                    ",".join("{}:{}".format(s, lr) for s, l in lr_schedule)])

    log(f"开始训练 (blocks={blocks}, filters={filters}, steps={max_steps})")
    log(f"训练命令: {' '.join(cmd)}")

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(tf_dir), env=env,
                                timeout=8 * 3600)
    except subprocess.TimeoutExpired:
        log("训练超时（8 小时上限），提前结束")
        return False
    elapsed(t0, "训练完成")

    if result.returncode != 0:
        log(f"训练失败 (返回码 {result.returncode})")
        return False

    # parse.py 在 ckpt_dir（或 cwd）生成 leelaz-model-final.txt
    final_dir = Path(ckpt_dir) if ckpt_dir else Path(tf_dir)
    final_txt = final_dir / "leelaz-model-final.txt"
    if not final_txt.exists():
        log(f"错误: 训练输出未找到: {final_txt}")
        return False

    # 压缩为 .txt.gz
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with final_txt.open("rb") as src, gzip.open(str(output_path), "wb") as dst:
        shutil.copyfileobj(src, dst)
    log(f"新权重已保存: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
    return True


# ============================================================
#  第6步: 上传新权重 + 清理旧 chunk
# ============================================================

def upload_weights(args, output_path):
    """上传新权重到 Gitee/GitHub Release"""
    if not args.weight_token:
        log("跳过上传: 没有 --weight-token")
        log(f"新权重保存在: {output_path}")
        log("可从 Kaggle Notebook Output 面板手动下载")
        return False

    log(f"上传新权重到 {args.weight_platform}...")
    if args.weight_platform == "gitee":
        ok = weight_release.upload_gitee(
            args.weight_owner, args.weight_repo, args.weight_token, output_path)
    else:
        ok = weight_release.upload_github(
            args.weight_owner, args.weight_repo, args.weight_token, output_path)

    if ok:
        log("新权重已发布! 贡献者下次会自动下载到新权重")
    else:
        log("上传失败，请从 Output 面板手动下载")
    return ok


def purge_old_chunks(data_owner, data_repo, data_token):
    """训练完成后删除 Gitee 数据仓库里的旧 chunk，释放空间"""
    if not data_token:
        log("跳过清理: 没有 --data-token")
        return
    log("清理 Gitee 数据仓库的旧 chunk...")
    chunks = gitee_data.list_chunks(data_owner, data_repo, token=data_token)
    if not chunks:
        log("数据仓库已空")
        return
    ok = 0
    for c in chunks:
        if gitee_data.delete_chunk(data_owner, data_repo, data_token,
                                   c["name"], c["sha"]):
            ok += 1
    log(f"已清理 {ok}/{len(chunks)} 个 chunk")


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Leela Zero Kaggle 训练脚本（简洁版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 首次训练（用本地初始权重）
  python distributed/kaggle_train_simple.py \\
      --weights-path /kaggle/input/weights/phoenixgo-v1.txt.gz \\
      --data-owner ABCradio --data-repo shuju \\
      --data-token 你的令牌 \\
      --max-steps 1000

  # 后续训练（从 Release 下载最新权重 + 上传新权重）
  python distributed/kaggle_train_simple.py \\
      --weight-platform gitee --weight-owner ABCradio --weight-repo lz-weights \\
      --weight-token 你的令牌 \\
      --data-owner ABCradio --data-repo shuju --data-token 你的令牌 \\
      --max-steps 1000
        """)

    # --- 基础权重 ---
    g_w = parser.add_argument_group("基础权重（二选一）")
    g_w.add_argument("--weights-path", default=os.environ.get("LZ_WEIGHTS_PATH", ""),
                     help="本地权重文件路径（首次训练用，如 /kaggle/input/weights/xxx.txt.gz）")
    g_w.add_argument("--weight-platform", choices=["gitee", "github"],
                     default=os.environ.get("LZ_WEIGHT_PLATFORM", "gitee"))
    g_w.add_argument("--weight-owner", default=os.environ.get("LZ_WEIGHT_OWNER", ""),
                     help="权重仓库所有者（Gitee 登录名）")
    g_w.add_argument("--weight-repo", default=os.environ.get("LZ_WEIGHT_REPO", ""),
                     help="权重仓库名")
    g_w.add_argument("--weight-token", default=os.environ.get("LZ_WEIGHT_TOKEN", ""),
                     help="权重仓库私人令牌（上传新权重用）")

    # --- 训练数据 ---
    g_d = parser.add_argument_group("训练数据")
    g_d.add_argument("--data-owner", default=os.environ.get("LZ_DATA_OWNER", "ABCradio"),
                     help="数据仓库所有者（Gitee 登录名）")
    g_d.add_argument("--data-repo", default=os.environ.get("LZ_DATA_REPO", "shuju"),
                     help="数据仓库名（默认 shuju）")
    g_d.add_argument("--data-token", default=os.environ.get("LZ_DATA_TOKEN", ""),
                     help="Gitee 私人令牌（私密仓库下载 + 训练后清理 chunk）")
    g_d.add_argument("--crawl-limit", type=int, default=0,
                     help="最多下载多少个 chunk（0=全部）")

    # --- 自对弈 ---
    g_s = parser.add_argument_group("自对弈（可选，补充训练数据）")
    g_s.add_argument("--selfplay-games", type=int, default=0,
                     help="自对弈局数（默认 0=不自对弈，只用 Gitee chunk）")
    g_s.add_argument("--selfplay-visits", type=int, default=800,
                     help="自对弈每步搜索量（默认 800）")

    # --- 训练参数 ---
    g_t = parser.add_argument_group("训练参数")
    g_t.add_argument("--blocks", type=int, default=None,
                     help="残差块数（默认自动从权重文件识别）")
    g_t.add_argument("--filters", type=int, default=None,
                     help="卷积滤波器数（默认自动识别）")
    g_t.add_argument("--max-steps", type=int, default=1000,
                     help="最大训练步数（默认 1000；续训时=本次新增步数）")
    g_t.add_argument("--ckpt-dir", default=os.environ.get("LZ_CKPT_DIR", ""),
                     help="断点续训 checkpoint 目录（默认 /kaggle/working/checkpoint）")
    g_t.add_argument("--save-every", type=int, default=8000,
                     help="每 N 步保存一次 checkpoint（默认 8000，短会话建议 1000）")
    g_t.add_argument("--keep-n", type=int, default=5,
                     help="只保留最近 N 个 checkpoint/权重文件（默认 5）")
    g_t.add_argument("--lr-schedule", type=str, default=None,
                     help="学习率调度 'step:lr,step:lr'（默认内置调度）")

    # --- 其他 ---
    g_o = parser.add_argument_group("其他")
    g_o.add_argument("--repo", default="https://gitee.com/ABCradio/AeonGo",
                     help="仓库地址（用于克隆训练代码）")
    g_o.add_argument("--no-upload", action="store_true",
                     help="训练完成后不上传权重")
    g_o.add_argument("--no-cleanup", action="store_true",
                     help="训练完成后不清理 Gitee chunk")

    args = parser.parse_args()

    # --- 校验 ---
    if not args.weights_path and not (args.weight_owner and args.weight_repo):
        print("错误: 需要指定基础权重")
        print("  方式1: --weights-path 本地权重文件")
        print("  方式2: --weight-owner + --weight-repo 从 Release 下载")
        sys.exit(1)

    if not (args.data_owner and args.data_repo):
        print("错误: 需要 --data-owner 和 --data-repo")
        sys.exit(1)

    # ============================================================
    #  开始训练流程
    # ============================================================
    total_start = time.time()
    log("=" * 60)
    log("Leela Zero Kaggle 训练（简洁版）")
    log("=" * 60)

    work_dir = Path("/kaggle/working")
    # 非 Kaggle 环境用当前目录
    if not work_dir.exists():
        work_dir = Path("./kaggle_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    chunks_dir = work_dir / "chunks"
    weights_dir = work_dir / "weights"
    tf_dir = work_dir / "tf"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else work_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "weights_new.txt.gz"

    # --- 断点续训检测：有历史 checkpoint 直接续训，跳过权重下载/转换 ---
    hist_ckpt, hist_step = ckpt_utils.find_latest_checkpoint(str(ckpt_dir))
    resumed = hist_ckpt is not None
    if resumed:
        log(f"发现历史 checkpoint (step={hist_step})，将自动恢复续训")
    else:
        log("无历史 checkpoint，将从基础权重开始")

    # --- 第1步: 克隆仓库 + 准备训练代码 ---
    log("[1/6] 准备训练代码...")
    t0 = time.time()
    repo_dir = clone_repo(args.repo, work_dir / "src")
    setup_tf_dir(repo_dir, tf_dir)
    elapsed(t0)

    # --- 第2步: 准备基础权重（续训时跳过） ---
    base_net = None
    if not resumed:
        log("[2/6] 准备基础权重...")
        t0 = time.time()
        base_net = prepare_weights(args, weights_dir)
        if not base_net:
            sys.exit(1)
        elapsed(t0)
    else:
        log("[2/6] 续训模式，跳过基础权重下载/转换")

    # --- 第3步: 收集训练数据 ---
    log("[3/6] 收集训练数据...")
    t0 = time.time()
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # 3a. 从 Gitee 下载贡献者上传的 chunk
    crawled = download_chunks(
        args.data_owner, args.data_repo, chunks_dir,
        token=args.data_token,
        limit=args.crawl_limit or None,
    )
    log(f"从 Gitee 下载了 {len(crawled)} 个 chunk")

    # 3b. 自对弈补充训练数据（可选；续训时跳过——无基础权重，且主消费贡献者 chunk）
    if args.selfplay_games > 0 and not resumed:
        leelaz_path = compile_leelaz(repo_dir)
        if leelaz_path:
            selfplay(
                leelaz_path, str(base_net),
                args.selfplay_games, args.selfplay_visits,
                chunks_dir,
            )
            total = len(list(chunks_dir.glob("*.gz")))
            log(f"自对弈后总计: {total} 个训练文件")
        else:
            log("leelaz 编译失败，跳过自对弈")
    elif args.selfplay_games > 0 and resumed:
        log("续训模式，跳过自对弈（checkpoint 目录中的权重即当前模型）")

    total_chunks = list(chunks_dir.glob("*.gz"))
    if not total_chunks:
        log("错误: 没有训练数据!")
        log("  确保有贡献者上传了 chunk，或启用 --selfplay-games")
        sys.exit(1)
    log(f"共 {len(total_chunks)} 个 chunk 用于训练")
    elapsed(t0)

    # --- 第4步: 权重 → TF checkpoint / 续训结构决策 ---
    blocks, filters = args.blocks, args.filters
    lr_schedule = None
    ckpt_path = None
    if resumed:
        log("[4/6] 续训模式，读取网络结构...")
        meta = ckpt_utils.checkpoint_architecture(str(ckpt_dir))
        user_lr = ckpt_utils.parse_lr_schedule(args.lr_schedule)
        if user_lr is not None:
            lr_schedule = user_lr
        elif meta is not None:
            lr_schedule = meta.get("lr_schedule")
        if meta is not None:
            blocks = int(meta.get("blocks", blocks))
            filters = int(meta.get("filters", filters))
        elif args.blocks is None or args.filters is None:
            log("错误: 找到历史 checkpoint 但 meta.json 缺失，无法确定网络结构")
            log("请用 --blocks/--filters 显式指定，或清空 checkpoint 目录重新从基础权重开始")
            sys.exit(1)
        # 用户本次显式指定了新调度: 在 blocks/filters 确定后写回 meta.json,
        # 否则下次续训会读到旧的调度, 本次指定被静默丢弃。
        if user_lr is not None:
            ckpt_utils.save_meta(str(ckpt_dir), blocks, filters, user_lr)
        ckpt_path = hist_ckpt
        max_steps = hist_step + args.max_steps  # 续训：本次新增步数
        log(f"将从 checkpoint 续训 (blocks={blocks}, filters={filters}, "
            f"step={hist_step}, 目标 {max_steps})")
    else:
        log("[4/6] 转换权重为 TF checkpoint...")
        t0 = time.time()
        try:
            ckpt_path, blocks, filters = convert_weights_to_checkpoint(
                tf_dir, base_net, args.blocks, args.filters, ckpt_dir)
            ckpt_utils.save_meta(str(ckpt_dir), blocks, filters,
                                 ckpt_utils.parse_lr_schedule(args.lr_schedule))
            log(f"将从基础权重继续训练 (blocks={blocks}, filters={filters})")
        except Exception as e:
            log(f"权重转换失败: {e}")
            sys.exit(1)
        max_steps = args.max_steps
        elapsed(t0)

    # --- 第5步: 训练 ---
    log("[5/6] 开始训练...")
    t0 = time.time()
    success = run_training(
        tf_dir=tf_dir,
        chunks_dir=chunks_dir,
        checkpoint_path=ckpt_path,
        output_path=output_path,
        blocks=blocks,
        filters=filters,
        max_steps=max_steps,
        ckpt_dir=str(ckpt_dir),
        save_every=args.save_every,
        keep_n=args.keep_n,
        lr_schedule=lr_schedule,
    )

    if not success or not output_path.exists():
        log("训练失败!")
        sys.exit(1)
    elapsed(t0)

    log(f"新权重: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")

    # --- 第6步: 上传 + 清理 ---
    log("[6/6] 后处理...")
    t0 = time.time()
    uploaded = False
    if not args.no_upload:
        uploaded = upload_weights(args, output_path)

    if uploaded and not args.no_cleanup:
        purge_old_chunks(args.data_owner, args.data_repo, args.data_token)
    else:
        log("跳过清理 (--no-cleanup 或未上传)")

    elapsed(t0)

    log("=" * 60)
    log(f"训练完成! 新权重: {output_path}")
    total_time = time.time() - total_start
    log(f"总耗时: {total_time/60:.1f} 分钟")
    log("=" * 60)


if __name__ == "__main__":
    main()
