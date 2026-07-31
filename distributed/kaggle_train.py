#!/usr/bin/env python3
"""
Leela Zero 训练脚本 — 在 Kaggle Notebook (T4 GPU) 上运行

架构: Gitee 数据仓库（chunks，API 读写） + Gitee/GitHub Release（权重）

用法 (Kaggle Notebook):
    !git clone --depth 1 https://gitee.com/leela-zero-next/leela-zero-next
    %cd leela-zero-next
    !pip install tensorflow==2.15.0
    !python distributed/kaggle_train.py \\
        --data-owner 数据仓库所有者 --data-repo lz-data \\
        --data-token 组织者的Gitee令牌（训练完删除chunk） \\
        --weight-owner 权重仓库所有者 --weight-repo 权重仓库名 \\
        --weight-token 组织者的Gitee令牌（上传新权重） \\
        --weights-path /kaggle/input/weights/phoenixgo-v1.txt.gz \\
        --selfplay-games 20

流程:
  1. 从 Gitee 数据仓库 chunks/ 下载贡献者上传的 chunk
  2. 用 leelaz 自对弈补充训练数据（可选）
  3. 权重 .txt.gz → TF checkpoint（从基础权重继续训练）
  4. parse.py 训练
  5. 新权重输出到 /kaggle/working/weights_new.txt.gz
  6. 上传新权重到 Gitee/GitHub Release
  7. 训练成功后删除 Gitee 数据仓库里的旧 chunk（释放免费空间）
"""

import argparse
import gzip
import os
import select
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gitee_data
import weight_release


def log(msg):
    print(f"[train] {msg}", flush=True)


def http_get(url, dest_path, timeout=600):
    """下载文件到本地"""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    return True


def clone_repo(repo_url, dest_dir):
    log(f"克隆仓库: {repo_url}")
    dest = Path(dest_dir)
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.check_call(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        timeout=120,
    )
    return dest


def compile_leelaz(src_dir):
    """在 Kaggle Linux 环境下编译 leelaz（CPU-only，Kaggle 无 OpenCL 驱动）"""
    leelaz_path = Path("/tmp/leelaz")
    if leelaz_path.exists():
        log("leelaz 已编译，跳过")
        return str(leelaz_path)

    log("编译 leelaz（约 5 分钟，CPU-only）...")
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq",
         "build-essential", "cmake",
         "libboost-all-dev", "zlib1g-dev", "libeigen3-dev"],
        check=False,
    )

    build_dir = src_dir / "build"
    build_dir.mkdir(exist_ok=True)

    nproc = os.cpu_count() or 2
    try:
        cmake = subprocess.run(
            ["cmake", "..", "-DUSE_CPU_ONLY=1"],
            cwd=str(build_dir),
            timeout=60, capture_output=True, text=True,
        )
        if cmake.returncode != 0:
            log(f"cmake 失败: {cmake.stderr[-200:]}")
            return None

        make = subprocess.run(
            ["make", f"-j{nproc}"],
            cwd=str(build_dir),
            timeout=600, capture_output=True, text=True,
        )
        if make.returncode != 0:
            log(f"make 失败: {make.stderr[-200:]}")
            return None
    except Exception as e:
        log(f"编译异常: {e}")
        return None

    built = build_dir / "leelaz"
    if built.exists():
        shutil.copy2(built, leelaz_path)
        log("leelaz 编译成功")
        return str(leelaz_path)
    log("编译产物未找到")
    return None


def crawl_chunks(data_owner, data_repo, dest_dir, token="", limit=None):
    """爬虫: 从 Gitee 数据仓库下载所有 chunk 到 dest_dir，返回文件列表"""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    log(f"列出 Gitee 数据仓库 {data_owner}/{data_repo} chunks/ ...")
    chunks = gitee_data.list_chunks(data_owner, data_repo, token=token)
    if not chunks:
        log("数据仓库暂无 chunk")
        return []

    log(f"共 {len(chunks)} 个 chunk，开始下载...")
    if limit:
        chunks = chunks[:limit]

    downloaded = []
    failed = 0
    for i, c in enumerate(chunks, 1):
        name = c.get("name")
        dst = dest / name
        if dst.exists() and dst.stat().st_size == c.get("size", 0) and c.get("size", 0) > 0:
            downloaded.append(str(dst))
            continue
        try:
            if gitee_data.download_chunk(data_owner, data_repo, name, dst, token=token):
                if dst.stat().st_size > 0:
                    downloaded.append(str(dst))
                    if i % 20 == 0 or i == len(chunks):
                        log(f"  下载 {i}/{len(chunks)}")
                else:
                    dst.unlink(missing_ok=True)
                    failed += 1
            else:
                failed += 1
        except Exception as e:
            log(f"  下载失败 {name}: {e}")
            failed += 1

    log(f"下载完成: {len(downloaded)} 成功, {failed} 失败")
    return downloaded


def purge_chunks(data_owner, data_repo, data_token):
    """训练完成并上传新权重后，删除 Gitee 数据仓库里的旧 chunk 释放空间"""
    if not data_token:
        log("跳过清理: 没有 --data-token")
        return
    log("列出 chunks 准备清理...")
    chunks = gitee_data.list_chunks(data_owner, data_repo, token=data_token)
    if not chunks:
        log("数据仓库已空，无需清理")
        return
    ok = 0
    for c in chunks:
        if gitee_data.delete_chunk(data_owner, data_repo, data_token,
                                   c["name"], c["sha"]):
            ok += 1
    log(f"已清理 {ok}/{len(chunks)} 个 chunk")


def selfplay(leelaz_path, weights_path, num_games, visits, output_dir):
    """自对弈 — 生成真实训练数据（含 MCTS 搜索概率）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        leelaz_path, "-w", str(weights_path),
        "-g",           # GTP 模式
        "--noponder", "--noise",
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
                if not line and proc.poll() is not None:
                    return None
                return line.rstrip("\n")
        return None

    def send(cmd, timeout=120):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
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
        if first.startswith("?"):
            return None
        if not first.startswith("="):
            return None
        result = first[1:].strip()
        if len(lines) > 1:
            rest = "\n".join(lines[1:])
            if result:
                return result + "\n" + rest
            return rest
        return result

    chunk_files = []
    for g in range(1, num_games + 1):
        send("clear_board", timeout=10)
        passes = 0
        resigned = False
        last_to_move = "b"
        for i in range(400):
            color = "b" if i % 2 == 0 else "w"
            last_to_move = color
            move = send(f"genmove {color}", timeout=300)
            if move is None:
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

        train_basename = f"sp_{g:06d}"
        send(f"dump_training {winner} {train_basename}", timeout=30)
        chunk_path = output_dir / f"{train_basename}.0.gz"
        if chunk_path.exists():
            chunk_files.append(str(chunk_path))

        if g % 5 == 0:
            log(f"  完成 {g}/{num_games} 局")

    send("quit", timeout=5)
    proc.terminate()
    proc.wait(timeout=10)

    log(f"自对弈完成: {len(chunk_files)} 个训练文件")
    return chunk_files


def convert_weights_to_checkpoint(tf_dir, weights_gz_path, blocks, filters, ckpt_dir):
    """将 .txt.gz 权重文件转为 TF checkpoint（供 parse.py --restore）"""
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

    version = lines[0] if lines else "0"
    log(f"权重版本: {version}")

    new_weights = []
    for line in lines[1:]:
        if line:
            new_weights.append(list(map(float, line.split(" "))))

    log(f"加载 {len(new_weights)} 个权重张量, 创建 TF 图 (blocks={blocks}, filters={filters})...")
    tfprocess = TFProcess(blocks, filters)
    tfprocess.init(batch_size=1, gpus_num=1)
    tfprocess.replace_weights(new_weights)

    ckpt_path = str(ckpt_dir / "initial")
    saved = tfprocess.saver.save(tfprocess.session, ckpt_path, global_step=0)
    log(f"TF checkpoint: {saved}")
    tfprocess.session.close()
    return saved


def run_training(tf_dir, chunks_dir, checkpoint_path, output_path,
                 blocks, filters, max_steps):
    """直接调用 parse.py 训练"""
    parse_script = Path(tf_dir) / "parse.py"
    if not parse_script.exists():
        log(f"parse.py 不存在: {parse_script}")
        return False

    # 给 chunk 文件统一前缀，确保 get_chunks 能读到
    chunk_dir = Path(chunks_dir)
    gz_files = sorted(chunk_dir.glob("*.gz"))
    if not gz_files:
        log(f"错误: {chunks_dir} 中没有 .gz 训练数据")
        return False
    for i, f in enumerate(gz_files):
        new_name = chunk_dir / f"t{i:06d}.gz"
        if new_name != f:
            shutil.copy2(f, new_name)

    # 前缀必须是目录路径 + 文件名前缀 → glob 为 {dir}/t*.gz
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

    log(f"开始训练 (blocks={blocks}, filters={filters}, steps={max_steps})")

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    result = subprocess.run(cmd, cwd=str(tf_dir), env=env)
    if result.returncode != 0:
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


def download_latest_weight(platform, owner, repo, dest_path):
    """从 Gitee/GitHub Release 下载最新权重，返回 bool"""
    log(f"下载最新权重: {platform} {owner}/{repo}")
    if platform == "gitee":
        return weight_release.download_gitee(owner, repo, dest_path)
    return weight_release.download_github(owner, repo, dest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Leela Zero Kaggle 训练脚本（爬虫+训练）")
    parser.add_argument("--data-owner",
                        default=os.environ.get("LZ_DATA_OWNER", "ABCradio"),
                        help="数据仓库所有者（Gitee 登录名，放训练 chunk 的仓库）")
    parser.add_argument("--data-repo",
                        default=os.environ.get("LZ_DATA_REPO", "shuju"),
                        help="数据仓库名（默认 shuju）")
    parser.add_argument("--data-token",
                        default=os.environ.get("LZ_DATA_TOKEN", ""),
                        help="组织者的 Gitee 私人令牌（私密仓库下载 + 训练成功后删除旧 chunk）")
    parser.add_argument("--weight-platform", choices=["gitee", "github"],
                        default=os.environ.get("LZ_WEIGHT_PLATFORM", "gitee"),
                        help="权重托管平台（默认 gitee）")
    parser.add_argument("--weight-owner",
                        default=os.environ.get("LZ_WEIGHT_OWNER", ""),
                        help="权重仓库所有者")
    parser.add_argument("--weight-repo",
                        default=os.environ.get("LZ_WEIGHT_REPO", ""),
                        help="权重仓库名")
    parser.add_argument("--weight-token",
                        default=os.environ.get("LZ_WEIGHT_TOKEN", ""),
                        help="权重仓库私人令牌（上传新权重用）")
    parser.add_argument("--weights-path",
                        default=os.environ.get("LZ_WEIGHTS_PATH", ""),
                        help="基础权重 .txt.gz 路径（首次用，如 /kaggle/input/weights/phoenixgo-v1.txt.gz）")
    parser.add_argument("--repo",
                        default="https://gitee.com/leela-zero-next/leela-zero-next",
                        help="仓库地址")
    parser.add_argument("--blocks", type=int, default=19,
                        help="残差块数（默认 19）")
    parser.add_argument("--filters", type=int, default=256,
                        help="卷积滤波器数（默认 256）")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="最大训练步数（默认 500）")
    parser.add_argument("--selfplay-games", type=int, default=20,
                        help="自对弈局数（默认 20）")
    parser.add_argument("--selfplay-visits", type=int, default=800,
                        help="自对弈访问数（默认 800）")
    parser.add_argument("--crawl-limit", type=int, default=0,
                        help="最多爬取多少个 chunk（0=全部）")
    args = parser.parse_args()

    if not (args.data_owner and args.data_repo):
        print("错误: 需要 --data-owner 和 --data-repo（数据仓库位置）")
        sys.exit(1)

    work_dir = Path("/kaggle/working")
    chunks_dir = work_dir / "chunks"
    weights_dir = work_dir / "weights"
    tf_dir = work_dir / "tf"

    # 1. 拉取仓库获取训练脚本
    repo_dir = clone_repo(args.repo, work_dir / "src")
    shutil.copytree(repo_dir / "training" / "tf", tf_dir, dirs_exist_ok=True)

    # 2. 确定基础权重
    weights_dir.mkdir(parents=True, exist_ok=True)
    base_net = None

    # 2a. 从本地文件读取（首次训练用，Kaggle Input）
    if args.weights_path and Path(args.weights_path).exists():
        base_net = Path(args.weights_path)
        log(f"使用本地基础权重: {base_net}")

    # 2b. 从 Gitee/GitHub Release 下载最新权重
    if not base_net:
        if not (args.weight_owner and args.weight_repo):
            log("错误: 没有本地权重，且缺少 --weight-owner/--weight-repo")
            sys.exit(1)
        latest = weights_dir / "current.txt.gz"
        if download_latest_weight(args.weight_platform, args.weight_owner,
                                  args.weight_repo, latest):
            if latest.stat().st_size > 100_000:
                base_net = latest
                log(f"使用最新权重: {latest}")
        else:
            log("下载最新权重失败")

    if not base_net:
        log("错误: 没有基础权重")
        log("首次运行请用 --weights-path 指定初始权重（PhoenixGo v1 等）")
        sys.exit(1)

    # 3. 爬虫: 收集训练数据
    chunks_dir.mkdir(parents=True, exist_ok=True)
    crawled = crawl_chunks(args.data_owner, args.data_repo, chunks_dir,
                           token=args.data_token,
                           limit=args.crawl_limit or None)
    log(f"爬虫共下载 {len(crawled)} 个 chunk")

    # 4. 自对弈补充训练数据
    if args.selfplay_games > 0:
        leelaz_path = compile_leelaz(repo_dir)
        if leelaz_path:
            selfplay(
                leelaz_path, str(base_net),
                args.selfplay_games, args.selfplay_visits,
                chunks_dir,
            )
            log(f"自对弈后总计: {len(list(chunks_dir.glob('*.gz')))} 个训练文件")

    total_chunks = list(chunks_dir.glob("*.gz"))
    if not total_chunks:
        log("错误: 没有训练数据，确保有贡献者上传了 chunk 或启用 --selfplay-games")
        sys.exit(1)
    log(f"共 {len(total_chunks)} 个 chunk 用于训练")

    # 5. 权重 → TF checkpoint
    ckpt_dir = work_dir / "checkpoint"
    try:
        ckpt_path = convert_weights_to_checkpoint(
            tf_dir, base_net, args.blocks, args.filters, ckpt_dir)
    except Exception as e:
        log(f"权重转换失败: {e}，将从头训练")
        ckpt_path = None

    # 6. 训练
    output_path = work_dir / "weights_new.txt.gz"
    success = run_training(
        tf_dir=tf_dir,
        chunks_dir=chunks_dir,
        checkpoint_path=ckpt_path,
        output_path=output_path,
        blocks=args.blocks,
        filters=args.filters,
        max_steps=args.max_steps,
    )

    if not success or not output_path.exists():
        log("训练失败")
        sys.exit(1)

    log(f"新权重: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # 7. 上传新权重到 Gitee/GitHub Release
    uploaded = False
    if args.weight_token:
        log(f"上传新权重到 {args.weight_platform} ({args.weight_owner}/{args.weight_repo})...")
        if args.weight_platform == "gitee":
            uploaded = weight_release.upload_gitee(
                args.weight_owner, args.weight_repo, args.weight_token, output_path)
        else:
            uploaded = weight_release.upload_github(
                args.weight_owner, args.weight_repo, args.weight_token, output_path)
        if uploaded:
            log("新权重已发布! 贡献者下次会自动下载到新权重")
        else:
            log("权重上传失败，文件保存在 /kaggle/working/weights_new.txt.gz")
            log("可以从 Notebook Output 面板手动下载，再手动上传")
    else:
        log("跳过上传: 没有 --weight-token")
        log("新权重保存在 /kaggle/working/weights_new.txt.gz")

    # 8. 上传成功后删除 Gitee 数据仓库里的旧 chunk，释放免费空间
    if uploaded:
        purge_chunks(args.data_owner, args.data_repo, args.data_token)

    log("=" * 40)
    log(f"训练文件数: {len(total_chunks)}")
    log("=" * 40)


if __name__ == "__main__":
    main()
