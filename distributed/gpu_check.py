#!/usr/bin/env python3
"""GPU 版云验证脚本 — 在任意有 N 卡的机器上跑 (本机 GT 730 会卡死, 别跑).

用法:
  python gpu_check.py --gpu /path/to/leelaz_gpu.exe --cpu /path/to/leelaz_cpu.exe \
                      -w /path/to/PhoenixGo-NG-v1.txt.gz

它做两件事:
  1. 正确性: 空棋盘 + 3 个局面, GPU 与 CPU 的 NN eval 逐局面对比
     (CPU 版是已实测正确的地面真值; 偏差 > 0.02 说明 GPU 内核有问题)。
  2. 速度: 双方 netbench 200 次, 报 n/s。现代 N 卡 (3060/4060) 期望远超 CPU。
建议 GPU 用 -t 16 -b 16 (默认已是 16), CPU 用 -t 8。
"""
import argparse, re, subprocess, time

SITUATIONS = [
    [],                              # 空棋盘
    ["play b D4"],                   # 黑D4 白先
    ["play b D4", "play w Q16"],     # 黑D4白Q16 黑先
    ["play b D4", "play w Q16", "play b C3"],  # 3手
]


def run_one(exe, weights, threads, batch, plays, timeout=300):
    cmd = [exe, "-w", weights, "-g", "--noponder", "-t", str(threads),
           "-b", str(batch), "-p", str(plays), "-q"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    def send(c, to=timeout):
        p.stdin.write(c + "\n"); p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line:
                return None
            out.append(line.rstrip())
            if line.startswith("=") or line.startswith("?"):
                return line.rstrip()
    send("boardsize 19"); send("komi 7.5")
    evals = []
    for moves in SITUATIONS:
        send("clear_board")
        for m in moves:
            send(m)
        r = send("genmove b")
        if r is None:
            break
        m = re.search(r"NN eval=([0-9.]+)", "\n".join(out))
        if m:
            evals.append(float(m.group(1)))
        else:
            evals.append(None)
    try: p.kill()
    except Exception: pass
    return evals


def bench(exe, weights, threads, batch, n=200, timeout=1200):
    p = subprocess.Popen([exe, "-w", weights, "-g", "--noponder",
                          "-t", str(threads), "-b", str(batch)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    def send(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line:
                return None
            out.append(line.rstrip())
            if line.startswith("=") or line.startswith("?"):
                return line.rstrip()
    send("boardsize 19"); send("komi 7.5")
    t0 = time.time()
    send("netbench %d" % n)
    dt = time.time() - t0
    try: p.kill()
    except Exception: pass
    m = re.search(r"(\d+) evaluations in [\d.]+ seconds -> (\d+) n/s", "\n".join(out))
    return int(m.group(2)) if m else None, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("-w", "--weights", required=True)
    a = ap.parse_args()

    print("=== 1) 正确性: GPU vs CPU 的 NN eval 对比 ===")
    g = run_one(a.gpu, a.weights, 16, 16, 3000)
    c = run_one(a.cpu, a.weights, 8, 16, 3000)
    ok = True
    for i, (gv, cv) in enumerate(zip(g, c)):
        if gv is None or cv is None:
            print("局面 %d: 无法获取 eval" % i); ok = False; continue
        d = abs(gv - cv)
        tag = "OK ✓" if d < 0.02 else "!!! 偏差过大 !!!"
        print("局面 %d: GPU=%.4f CPU=%.4f 偏差=%.4f %s" % (i, gv, cv, d, tag))
        if d >= 0.02: ok = False

    print("\n=== 2) 速度 netbench ===")
    gs, _ = bench(a.gpu, a.weights, 16, 16)
    cs, _ = bench(a.cpu, a.weights, 8, 16)
    print("GPU: %s n/s | CPU: %s n/s" % (gs, cs))
    if gs and cs:
        print("GPU/CPU 比: %.1fx" % (gs / cs))

    print("\n结论:", "全部通过, 可以放心用" if ok else "存在偏差, 内核有问题, 先别用!")
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
