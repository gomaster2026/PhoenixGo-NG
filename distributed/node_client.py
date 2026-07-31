#!/usr/bin/env python3
"""
Leela Zero 节点客户端 — 贡献算力

通过 GTP 协议驱动 leelaz 自对弈，生成 V1 格式训练数据（.gz），
自动上传到 Gitee 数据仓库（chunks/ 目录）。

权重文件优先级:
  1. 本地已有权重（组织者手动发给你，放 --weights 指定的路径）
  2. Gitee/GitHub Release 下载（自动合并分片，需 --weight-owner/--weight-repo）

用法:
  python node_client.py \\
      --token 你的Gitee私人令牌 \\
      --data-owner 组织者登录名 --data-repo shuju \\
      --weights ./权重文件.txt.gz \\
      --leelaz ./leelaz.exe --games 20 --node-name alice

Gitee 私人令牌: gitee.com → 设置 → 安全设置 → 私人令牌 → 勾选 projects 权限
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gitee_data
import weight_release


class GTPEngine:
    """GTP 协议驱动 leelaz 自对弈引擎"""

    def __init__(self, leelaz_path, weights_path, playouts, seed, work_dir):
        leelaz_path = Path(leelaz_path)
        if not leelaz_path.exists():
            raise FileNotFoundError(f"leelaz 不存在: {leelaz_path}")
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(leelaz_path),
            "-w", str(weights_path),
            "-g",           # GTP 模式，stdout 只输出 GTP 响应
            "-t", "1",
            "-p", str(playouts),
            "-q",           # 静默
            "-n",           # 策略网络噪声（探索用）
            "-m", "30",     # 前 30 手随机
            "--noponder",
            "-r", "5",      # 胜率 < 5% 时认输
            "-s", str(seed),
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, bufsize=1,
            cwd=str(work_dir),
        )

        resp = self._command("protocol_version", timeout=30)
        if resp is None:
            self.close()
            stderr_tail = ""
            try:
                stderr_tail = self.proc.stderr.read()[-500:]
            except Exception:
                pass
            raise RuntimeError(
                f"leelaz 启动失败 (GTP 无响应)\n"
                f"  命令: {' '.join(cmd)}\n"
                f"  可能原因: 路径不对、缺少 DLL、权重格式错误\n"
                f"  stderr: ...{stderr_tail}"
            )

    def _command(self, cmd_text, timeout=60):
        """发送 GTP 命令 → 返回响应行列表"""
        try:
            self.proc.stdin.write(cmd_text + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return None

        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    return None
                time.sleep(0.05)
                continue
            line = line.rstrip("\n\r")
            if line == "":
                break
            lines.append(line)

        if not lines:
            return None
        first = lines[0]
        if first.startswith("?"):
            return None
        if first.startswith("= "):
            lines[0] = first[2:]
        elif first.startswith("="):
            lines[0] = first[1:]
        return lines

    def clear_board(self):
        return self._command("clear_board", timeout=5)

    def genmove(self, color):
        lines = self._command(f"genmove {color}", timeout=600)
        if lines is None:
            return None
        move = lines[0].strip()
        return move

    def final_score(self):
        lines = self._command("final_score", timeout=10)
        if lines is None:
            return None
        return lines[0].strip() if lines else None

    def dump_training(self, winner, basename):
        self._command(f"dump_training {winner} {basename}", timeout=30)

    def printsgf(self, filename):
        self._command(f"printsgf {filename}", timeout=10)

    def close(self):
        try:
            self._command("quit", timeout=5)
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=3)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def play_one_game(engine, game_id, work_dir):
    """
    用 GTP 驱动一局自对弈。
    返回 (winner, num_moves, train_file_path, sgf_file_path, error)
    """
    engine.clear_board()
    time.sleep(0.2)

    moves = 0
    passes = 0
    resigned = False
    winner = None

    for i in range(500):
        color = "b" if i % 2 == 0 else "w"
        move = engine.genmove(color)
        if move is None:
            return "?", moves, None, None, "genmove timeout"
        if move == "resign":
            resigned = True
            winner = "w" if color == "b" else "b"
            break
        moves += 1
        if move == "pass":
            passes += 1
            if passes >= 2:
                break
        else:
            passes = 0

    if not resigned:
        score = engine.final_score()
        if score is None:
            return "?", moves, None, None, "final_score failed"
        if score.startswith("W+"):
            winner = "w"
        elif score.startswith("B+"):
            winner = "b"
        else:
            winner = "w"

    if winner is None:
        winner = "w"

    train_basename = f"train_{game_id:06d}"
    engine.dump_training(winner, train_basename)

    train_file = work_dir / f"{train_basename}.0.gz"
    sgf_name = f"game_{game_id:06d}.sgf"
    engine.printsgf(sgf_name)
    sgf_file = work_dir / sgf_name

    return winner, moves, train_file, sgf_file, None


# ---------- 权重下载 ----------

def download_weights(weight_platform, weight_owner, weight_repo, dest_path):
    """从 Gitee/GitHub Release 下载最新权重，返回 bool"""
    print(f"[node] 从 {weight_platform} ({weight_owner}/{weight_repo}) 下载权重...")
    try:
        if weight_platform == "gitee":
            ok = weight_release.download_gitee(weight_owner, weight_repo, dest_path)
        else:
            ok = weight_release.download_github(weight_owner, weight_repo, dest_path)
        if not ok:
            return False
        size = Path(dest_path).stat().st_size
        if size < 100_000:
            print(f"[node] 权重文件过小 ({size} bytes)，可能下载到错误内容")
            return False
        print(f"[node] 权重已下载: {dest_path} ({size/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"[node] 下载权重失败: {e}")
        return False


def upload_chunk(data_owner, data_repo, token, file_path, node_name="", retries=3):
    """上传一个训练文件到 Gitee 数据仓库 chunks/，返回 bool"""
    remote_name, ok = gitee_data.upload_chunk(
        data_owner, data_repo, token, file_path,
        node_name=node_name, retries=retries,
    )
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Leela Zero 自对弈节点客户端 — 贡献算力")
    parser.add_argument("--data-owner",
                        default=os.environ.get("LZ_DATA_OWNER", "ABCradio"),
                        help="数据仓库所有者（Gitee 登录名，放训练 chunk 的仓库）")
    parser.add_argument("--data-repo",
                        default=os.environ.get("LZ_DATA_REPO", "shuju"),
                        help="数据仓库名（默认 shuju）")
    parser.add_argument("--token",
                        default=os.environ.get("LZ_TOKEN", ""),
                        help="你的 Gitee 私人令牌（需 projects 权限，用于上传 chunk）")
    parser.add_argument("--weight-platform", choices=["gitee", "github"],
                        default=os.environ.get("LZ_WEIGHT_PLATFORM", "gitee"),
                        help="权重托管平台（默认 gitee）")
    parser.add_argument("--weight-owner",
                        default=os.environ.get("LZ_WEIGHT_OWNER", ""),
                        help="权重仓库所有者（如 gitee 用户名）")
    parser.add_argument("--weight-repo",
                        default=os.environ.get("LZ_WEIGHT_REPO", ""),
                        help="权重仓库名")
    parser.add_argument("--node-name",
                        default=os.environ.get("LZ_NODE_NAME", ""),
                        help="(可选) 你的昵称，排行榜会显示")
    parser.add_argument("--leelaz",
                        default=os.environ.get("LZ_LEELAZ", "./leelaz.exe"),
                        help="leelaz 可执行文件路径")
    parser.add_argument("--weights", default="./weights_current.txt.gz",
                        help="权重缓存路径")
    parser.add_argument("--games", type=int, default=10,
                        help="每轮自对弈局数")
    parser.add_argument("--playouts", type=int, default=800,
                        help="每步搜索 playout 数")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（0=自动）")
    parser.add_argument("--keep-sgf", action="store_true",
                        help="同时保留棋谱文件到本地")
    args = parser.parse_args()

    if not (args.data_owner and args.data_repo):
        print("错误: 需要 --data-owner 和 --data-repo（数据仓库位置）")
        sys.exit(1)
    if not args.token:
        print("错误: 需要 --token（你的 Gitee 私人令牌，用于上传 chunk）")
        sys.exit(1)

    seed = args.seed or int(time.time() * 1000000) % (2 ** 31)
    weights_path = Path(args.weights)

    work_dir = Path(tempfile.mkdtemp(prefix="lz_selfplay_"))
    print(f"[node] 工作目录: {work_dir}")

    # 权重获取: 优先用本地已有的权重文件（组织者手动分发）
    if weights_path.exists() and weights_path.stat().st_size > 100_000:
        print(f"[node] 使用本地权重: {weights_path} "
              f"({weights_path.stat().st_size / 1e6:.1f} MB)")
    elif args.weight_owner and args.weight_repo:
        if not download_weights(args.weight_platform, args.weight_owner,
                                args.weight_repo, weights_path):
            print("提示: 请确认组织者已经把初始权重上传到权重仓库（weight_release.py upload）")
            sys.exit(1)
    else:
        print("错误: 没有本地权重文件，且缺少 --weight-owner/--weight-repo")
        print("提示: 请向组织者要权重文件，放到: " + str(weights_path))
        sys.exit(1)

    print(f"[node] 启动引擎: {args.leelaz}")
    print(f"[node] playouts={args.playouts}, games={args.games}, seed={seed}")
    try:
        engine = GTPEngine(args.leelaz, str(weights_path),
                           args.playouts, seed, work_dir)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    train_files = []
    sgf_files = []
    total_moves = 0
    failed = 0
    try:
        print(f"[node] 自对弈 {args.games} 局...")
        for g in range(1, args.games + 1):
            start = time.time()
            winner, moves_cnt, tf_path, sf_path, err = play_one_game(
                engine, g, work_dir)
            elapsed = time.time() - start
            if err:
                print(f"  #{g}/{args.games} 失败: {err}")
                failed += 1
                continue
            total_moves += moves_cnt
            print(f"  #{g}/{args.games} {moves_cnt}手 胜者={winner} {elapsed:.0f}s")
            if tf_path and tf_path.exists():
                train_files.append(tf_path)
            if sf_path and sf_path.exists():
                sgf_files.append(sf_path)
    except KeyboardInterrupt:
        print("\n[node] 用户中断")
    finally:
        engine.close()

    print(f"\n[node] 完成! {len(train_files)} 个训练文件 ({total_moves} 手)")
    if failed:
        print(f"[node] 失败: {failed} 局")

    if args.keep_sgf and sgf_files:
        out_dir = Path("./sgf_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sgf_files:
            shutil.copy2(f, out_dir / f.name)
        print(f"[node] 棋谱保存到 {out_dir}")

    # 上传所有训练文件到 Gitee 数据仓库
    uploaded = 0
    for tf in train_files:
        if upload_chunk(args.data_owner, args.data_repo, args.token,
                        tf, args.node_name):
            uploaded += 1

    print(f"\n[node] 上传完成: {uploaded}/{len(train_files)} 个文件")
    if uploaded < len(train_files):
        print("[node] 部分上传失败，文件还在工作目录里，可以稍后重跑或手动上传:")
        print(f"  {work_dir}")
    else:
        print("[node] 全部上传成功! 组织者下次训练时会自动用到你的棋谱")
    print(f"[node] 下次继续贡献: python {__file__} "
          f"--data-owner {args.data_owner} --data-repo {args.data_repo} "
          f"--weight-owner {args.weight_owner} --weight-repo {args.weight_repo} "
          f"--leelaz {args.leelaz} --node-name {args.node_name or '你的名字'} "
          f"--games {args.games}")


if __name__ == "__main__":
    main()
