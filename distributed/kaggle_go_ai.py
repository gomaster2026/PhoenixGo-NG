"""
围棋 AI 训练 — 复制到 Kaggle Notebook 使用 (双 T4 并行版)
跑起来就能训，不用传任何权重文件（第一次随机初始化）。
每周自动续训：Kaggle 跑完后下载 model.pt，下周新建 Notebook 时拖回去即可。

改进点:
  1. 双 T4 并行自对弈 (2 个 GPU 同时生成棋谱, 速度翻倍)
  2. 修复 _legal 方法修改棋盘的 bug (try-finally 保护)
  3. 修复 ko 检测不完整的问题
  4. PhoenixGo 兼容网络 (20 blocks, 64 channels)
  5. 梯度裁剪 + 学习率衰减 (Cosine Annealing)
  6. L2 正则化 (weight_decay, AlphaGo Zero 标准)
  7. 8 种对称性数据增强 (AlphaGo Zero/Leela Zero 标准, 8 倍数据效率)
  8. _liberties 递归改迭代 (防止大棋串栈溢出)
  9. 温度调度 (前 30 步 temp=1, 之后 temp→0, AlphaGo Zero 标准)
  10. 统一 policy softmax (361 棋盘 + 1 pass, 不分开 sigmoid)
  11. buffer 持久化 (跨 session 续训, 防止断线丢数据)
  12. 训练 loss 分解显示 (策略 loss + 价值 loss)
  13. PhoenixGo 17 通道输入 (8 步历史 × 己方/对方 + 颜色通道)
  14. 修复 MCTS UCB 符号 + sim.done 检查
  15. 预激活残差块 (BN+ReLU→Conv→BN+ReLU→Conv→+residual, 无最后 ReLU)
  16. MCTS 已知终局节点跳过搜索 (避免浪费模拟次数)
  17. MCTS 叶节点无合法落子时仍添加 PASS (PASS 总是合法的)
  18. 多线程随机数种子独立化 (防止竞态)
 19. PhoenixGo 策略头 (2 通道 conv→BN→ReLU→FC 722→362)
 20. PhoenixGo 输入卷积 post-activation (conv→BN→ReLU) + trunk BN+ReLU
    (与引擎 CPUPipe::forward / training/tf/tfprocess.py 的 conv_block 一致)
"""
import os, time, math, copy, threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ========== 配置 ==========
BOARD = 19
NUM_GPUS = 2 if torch.cuda.device_count() >= 2 else 1
BUDGET_HOURS = 20.0       # 控制在 30h/周 以内
SIMS_PER_MOVE = 1200      # MCTS 模拟次数 (AlphaGo Zero 标准, 不降)
GAMES_PER_BATCH = 5       # 每批自对弈局数 (用于进度显示和存档)
GAMES_BEFORE_TRAIN = 1200 # 攒满 1200 局再训练 (跨 session 续训, 累计到就训练)
TRAIN_EPOCHS = 3          # 训练轮数 (遍历全部数据 3 遍)
BATCH = 256               # 训练 batch size
LR = 1e-3                 # 初始学习率
WEIGHT_DECAY = 1e-4       # L2 正则化 (AlphaGo Zero 标准)
GRAD_CLIP = 1.0           # 梯度裁剪
BUFFER_MAX = 200000       # replay buffer 最大容量
BUFFER_PATH = '/kaggle/working/buffer.npz'  # buffer 持久化路径
PROGRESS_PATH = '/kaggle/working/progress.npz'  # 自对弈进度 (games_done 持久化)

# 网络 (匹配 PhoenixGo 权重: 20 blocks, 64 channels)
NET_CHANNELS = 64
NET_BLOCKS = 20

def find_model():
    """自动搜索 .pt 权重文件, 支持 3 层深度"""
    candidates = ['model.pt']
    # 1. 固定路径
    for c in candidates:
        p = os.path.join('/kaggle/input/model', c)
        if os.path.exists(p):
            return p
    # 2. 遍历 Kaggle Input (3层深度)
    input_root = '/kaggle/input'
    if os.path.isdir(input_root):
        for d1 in os.listdir(input_root):
            p1 = os.path.join(input_root, d1)
            if os.path.isfile(p1) and p1 in candidates:
                return p1
            if not os.path.isdir(p1):
                continue
            for c in candidates:
                p = os.path.join(p1, c)
                if os.path.exists(p):
                    return p
            for d2 in os.listdir(p1):
                p2 = os.path.join(p1, d2)
                if os.path.isfile(p2) and p2 in candidates:
                    return p2
                if not os.path.isdir(p2):
                    continue
                for c in candidates:
                    p = os.path.join(p2, c)
                    if os.path.exists(p):
                        return p
                for d3 in os.listdir(p2):
                    p3 = os.path.join(p2, d3)
                    for c in candidates:
                        p = os.path.join(p3, c)
                        if os.path.exists(p):
                            return p
    # 找不到, 打印目录结构
    print('未找到 model.pt, Kaggle Input 目录:')
    if os.path.isdir(input_root):
        for d in os.listdir(input_root):
            dp = os.path.join(input_root, d)
            print(f'  {d}/{" [文件]" if os.path.isfile(dp) else ""}')
            if os.path.isdir(dp):
                for f in os.listdir(dp):
                    print(f'    {f}/{" [文件]" if os.path.isfile(os.path.join(dp,f)) else ""}')
    return None


MODEL_IN = '/kaggle/input/model/model.pt'
MODEL_OUT = '/kaggle/working/model.pt'

print(f'GPU 数量: {torch.cuda.device_count()} | 使用 {NUM_GPUS} GPU')
print(f'网络: {NET_BLOCKS} blocks, {NET_CHANNELS} channels')

# ========== 围棋环境 ==========
NONE, BLACK, WHITE = 0, 1, 2
PASS = BOARD * BOARD

def other(c):
    return BLACK if c == WHITE else WHITE

class GoEnv:
    def __init__(self, komi=7.5):
        self.N = BOARD
        self.komi = komi
        self.reset()

    def reset(self):
        self.board = np.zeros((self.N, self.N), dtype=np.int8)
        self.current = BLACK
        self.ko = None
        self.last_move = None
        self.passes = 0
        self.done = False
        self.winner = None
        self.move_history = []  # 记录所有落子
        self.board_history = [self.board.copy()]  # 最近 8 步棋盘 (PhoenixGo 17 通道)

    def _push_history(self):
        """落子后保存棋盘快照 (最多保留 8 步)"""
        self.board_history.append(self.board.copy())
        if len(self.board_history) > 8:
            self.board_history.pop(0)

    def _group(self, r, c):
        """找到 (r,c) 所在的棋串"""
        color = self.board[r, c]
        stack = [(r, c)]
        seen = set()
        grp = []
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in seen:
                continue
            seen.add((cr, cc))
            grp.append((cr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.N and 0 <= nc < self.N and self.board[nr, nc] == color:
                    stack.append((nr, nc))
        return grp

    def _liberties(self, r, c, color, vis):
        """计算 (r,c) 所在棋串的气数 (迭代式, 防止大棋串栈溢出)

        vis 只记录同色子；气（空点）必须按坐标去重，否则同一个空点邻接
        棋串多个子（如 L 形棋串的凹角）时会被重复计数，导致气数虚高、
        自杀/提子/劫判定错误。
        """
        if (r, c) in vis:
            return 0
        libs = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in vis:
                continue
            vis.add((cr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.N and 0 <= nc < self.N:
                    v = self.board[nr, nc]
                    if v == NONE:
                        libs.add((nr, nc))
                    elif v == color and (nr, nc) not in vis:
                        stack.append((nr, nc))
        return len(libs)

    def _is_legal(self, r, c):
        """检查落子是否合法 (临时修改棋盘, 用 try-finally 确保恢复)"""
        color = self.current
        opp = other(color)
        self.board[r, c] = color
        try:
            captures = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N and self.board[nr, nc] == opp:
                    if self._liberties(nr, nc, opp, set()) == 0:
                        captures = True
                        break
            legal = captures or self._liberties(r, c, color, set()) > 0
        finally:
            self.board[r, c] = NONE  # 无论是否异常都恢复棋盘
        return legal

    def legal_moves(self):
        moves = []
        for r in range(self.N):
            for c in range(self.N):
                if self.board[r, c] == NONE and self.ko != (r, c) and self._is_legal(r, c):
                    moves.append(r * self.N + c)
        return moves

    def play(self, move):
        if self.done:
            raise ValueError('game over')
        if move == PASS:
            self.passes += 1
            self.ko = None
            self.last_move = PASS
            self.move_history.append(PASS)
            self._push_history()  # PhoenixGo: PASS 也推进时间步
            if self.passes >= 2:
                self._finish()
            self.current = other(self.current)
            return
        r, c = divmod(move, self.N)
        if self.board[r, c] != NONE or self.ko == (r, c) or not self._is_legal(r, c):
            raise ValueError(f'illegal move {move}')
        color = self.current
        opp = other(color)
        self.board[r, c] = color
        # 提子
        cap_pts = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.N and 0 <= nc < self.N and self.board[nr, nc] == opp:
                if self._liberties(nr, nc, opp, set()) == 0:
                    grp = self._group(nr, nc)
                    for (gr, gc) in grp:
                        self.board[gr, gc] = NONE
                    cap_pts.extend(grp)
        # ko 检测: 只提一子且自己只有一气
        self.ko = None
        if len(cap_pts) == 1 and self._liberties(r, c, color, set()) == 1:
            self.ko = cap_pts[0]
        self.last_move = move
        self.passes = 0
        self._push_history()  # PhoenixGo: 保存棋盘快照
        self.current = opp
        self.move_history.append(move)

    def _finish(self):
        self.done = True
        bs, ws = self._score()
        self.winner = BLACK if bs > ws else WHITE

    def _score(self):
        terr = {BLACK: 0, WHITE: 0}
        vis = set()
        for r in range(self.N):
            for c in range(self.N):
                if self.board[r, c] == NONE and (r, c) not in vis:
                    region = []
                    stack = [(r, c)]
                    cols = set()
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in vis:
                            continue
                        vis.add((cr, cc))
                        region.append((cr, cc))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < self.N and 0 <= nc < self.N:
                                if self.board[nr, nc] == NONE and (nr, nc) not in vis:
                                    stack.append((nr, nc))
                                elif self.board[nr, nc] != NONE:
                                    cols.add(self.board[nr, nc])
                    if len(cols) == 1:
                        terr[next(iter(cols))] += len(region)
        bs = int(np.sum(self.board == BLACK)) + terr[BLACK]
        ws = int(np.sum(self.board == WHITE)) + terr[WHITE] + self.komi
        return bs, ws

    def get_features(self):
        """17 通道输入 (PhoenixGo): 8 步历史 × (己方/对方棋子) + 1 颜色通道
        布局: [me_t0, opp_t0, me_t1, opp_t1, ..., me_t7, opp_t7, color]
        己方视角: 黑方行棋时 me=黑, opp=白; 白方行棋时 me=白, opp=黑
        颜色通道: 黑方行棋=1.0, 白方行棋=0.0
        """
        f = np.zeros((17, self.N, self.N), dtype=np.float32)
        color = self.current
        opp = other(color)
        for h in range(8):
            if h == 0:
                b = self.board  # h=0: 当前棋盘
            else:
                idx = -(h + 1)  # -2, -3, ..., -8
                if abs(idx) <= len(self.board_history):
                    b = self.board_history[idx]
                else:
                    b = self.board_history[0]  # 最早的棋盘 (通常为空)
            f[2 * h] = (b == color)       # 己方棋子
            f[2 * h + 1] = (b == opp)     # 对方棋子
        # Channel 16: 颜色 (黑方行棋=1.0, 白方行棋=0.0)
        f[16] = 1.0 if color == BLACK else 0.0
        return f

    def clone(self):
        """快速克隆 (比 deepcopy 快)"""
        new = GoEnv.__new__(GoEnv)
        new.N = self.N
        new.komi = self.komi
        new.board = self.board.copy()
        new.current = self.current
        new.ko = self.ko
        new.last_move = self.last_move
        new.passes = self.passes
        new.done = self.done
        new.winner = self.winner
        new.move_history = list(self.move_history)
        new.board_history = [b.copy() for b in self.board_history]
        return new


# ========== 网络 ==========
class ResBlock(nn.Module):
    """预激活残差块 (PhoenixGo): BN+ReLU→Conv→BN+ReLU→Conv→+residual (无最后ReLU)
    最后一个残差块的 ReLU 由 GoNet 的 trunk BN+ReLU 统一处理"""
    def __init__(self, ch):
        super().__init__()
        self.b1 = nn.BatchNorm2d(ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)

    def forward(self, x):
        y = self.c1(torch.relu(self.b1(x)))
        y = self.c2(torch.relu(self.b2(y)))
        return x + y  # 无 ReLU (trunk BN+ReLU 统一处理)


class GoNet(nn.Module):
    """PhoenixGo 兼容网络: 17 输入通道, 预激活残差塔, 2 通道策略头 + FC"""
    def __init__(self, in_channels=17, channels=NET_CHANNELS, blocks=NET_BLOCKS):
        super().__init__()
        # 输入卷积 (post-activation): conv → BN → ReLU。
        # 与引擎 CPUPipe::forward 及 tfprocess.py conv_block 一致
        # (v3 权重文件的输入单元 6 行含输入 BN 的 gamma/beta/mean/var)。
        self.inp = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.ibn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(blocks)])
        # Trunk BN+ReLU (PhoenixGo: layer_final/batch_norm)
        self.bn_trunk = nn.BatchNorm2d(channels)
        # 策略头: 1x1 conv (channels→2) → BN → ReLU → flatten → FC(722→362)
        self.policy = nn.Conv2d(channels, 2, 1, bias=False)
        self.pbn = nn.BatchNorm2d(2)
        self.pfc = nn.Linear(2 * BOARD * BOARD, BOARD * BOARD + 1)
        # 价值头: 1x1 conv (channels→1) → BN → ReLU → FC(361→256) → ReLU → FC(256→1) → tanh
        self.vconv = nn.Conv2d(channels, 1, 1, bias=False)
        self.vbn = nn.BatchNorm2d(1)
        self.vfc1 = nn.Linear(BOARD * BOARD, 256)
        self.vfc2 = nn.Linear(256, 1)

    def forward(self, x):
        y = torch.relu(self.ibn(self.inp(x)))  # 输入卷积 post-activation
        for b in self.blocks:
            y = b(y)
        y = torch.relu(self.bn_trunk(y))  # Trunk BN+ReLU
        # 策略: conv → BN → ReLU → flatten → FC
        p = torch.relu(self.pbn(self.policy(y))).view(x.size(0), -1)
        logits = self.pfc(p)
        # 价值: conv → BN → ReLU → flatten → FC → ReLU → FC → tanh
        v = torch.relu(self.vbn(self.vconv(y))).view(x.size(0), -1)
        v = torch.tanh(self.vfc2(torch.relu(self.vfc1(v))))
        return logits, v


def policy_weight_chw_to_nhwc(w):
    """把策略头 FC 权重的输入列从 CHW 展平重排为 NHWC (position-major) 展平。

    GoNet.forward 用 .view(B, -1) 展平策略头卷积输出 (CHW: 列索引 c*361 + y*19 + x),
    而引擎 src/Network.cpp get_output_internal 对 v3 网络先做 CHW->NHWC 转换
    (列索引 y*38 + x*2 + c) 再乘 dense 权重。直接导出 PyTorch 权重会导致
    引擎端策略输出与棋盘位置错位 (黑白通道交错), 必须重排列序。
    价值头输出仅 1 通道, CHW 与 NHWC 等价, 无需处理。
    """
    w = np.asarray(w)
    out = np.zeros_like(w)
    for c in range(2):
        for y in range(BOARD):
            for x in range(BOARD):
                chw_col = c * BOARD * BOARD + y * BOARD + x
                nhwc_col = y * BOARD * 2 + x * 2 + c
                out[:, nhwc_col] = w[:, chw_col]
    return out


def export_leela_v3(net, path):
    """导出 PhoenixGo v3 格式权重, 与引擎 src/Network.cpp load_v3_network 互操作。

    文件布局 (行数 = 29 + 12*blocks):
      1  版本号 "3"
      6  输入卷积单元 (conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var)
      12*blocks  残差块 (每块 2 个 conv+BN 单元)
      4  trunk BN (gamma, beta, mean, var)
      8  策略头 (conv_w, conv_b, bn 4 行, ip_w, ip_b)
      10 价值头 (conv_w, conv_b, bn 4 行, ip1_w, ip1_b, ip2_w, ip2_b)
    PyTorch conv 权重 [out,in,kh,kw] 与 v3 的 [out,in,kh,kw] 一致, FC [out,in] 一致。
    注意: 策略头 FC 的输入列必须按 position-major (NHWC) 重排 (见
    policy_weight_chw_to_nhwc), 以匹配引擎 CHW->NHWC 展平后的 dense 读取顺序。
    """
    import gzip as _gzip

    def bn_params(bn):
        return (bn.weight.detach().cpu().numpy(),
                bn.bias.detach().cpu().numpy(),
                bn.running_mean.detach().cpu().numpy(),
                bn.running_var.detach().cpu().numpy())

    def line(arr):
        return " ".join(repr(float(x)) for x in np.ravel(arr))

    def conv_unit(conv, bn):
        g, b, m, v = bn_params(bn)
        out = [line(conv.weight.detach().cpu().numpy()),
               line(np.zeros(conv.weight.shape[0]))]
        out.extend(line(x) for x in (g, b, m, v))
        return out

    lines = ["3"]
    lines.extend(conv_unit(net.inp, net.ibn))
    for blk in net.blocks:
        lines.extend(conv_unit(blk.c1, blk.b1))
        lines.extend(conv_unit(blk.c2, blk.b2))
    lines.extend(line(x) for x in bn_params(net.bn_trunk))
    lines.extend(conv_unit(net.policy, net.pbn))
    lines.append(line(policy_weight_chw_to_nhwc(
        net.pfc.weight.detach().cpu().numpy())))
    lines.append(line(net.pfc.bias.detach().cpu().numpy()))
    lines.extend(conv_unit(net.vconv, net.vbn))
    lines.append(line(net.vfc1.weight.detach().cpu().numpy()))
    lines.append(line(net.vfc1.bias.detach().cpu().numpy()))
    lines.append(line(net.vfc2.weight.detach().cpu().numpy()))
    lines.append(line(net.vfc2.bias.detach().cpu().numpy()))

    out = "\n".join(lines) + "\n"
    if str(path).endswith(".gz"):
        with _gzip.open(path, "wt") as f:
            f.write(out)
    else:
        with open(path, "w") as f:
            f.write(out)
    expected = 29 + 12 * len(net.blocks)
    ok = "OK" if len(lines) == expected else f"MISMATCH(期望 {expected})"
    print(f"[export] v3 权重已导出: {path} ({len(lines)} 行, {ok})")


# ========== MCTS ==========
class Node:
    __slots__ = ('prior', 'to_move', 'children', 'visits', 'value', 'outcome')

    def __init__(self, prior, to_move):
        self.prior = prior
        self.to_move = to_move
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.outcome = None


def run_mcts_single(model, env, device, num_sim=SIMS_PER_MOVE, cpuct=1.0, dirichlet=0.03):
    model.eval()
    root = Node(0.0, env.current)
    legal = env.legal_moves()
    if not legal:
        return np.zeros(BOARD * BOARD + 1, dtype=np.float32), 0.0

    with torch.no_grad():
        logits, _ = model(torch.from_numpy(env.get_features()[None]).to(device))
    # 修复: 统一用 softmax (361棋盘 + 1 pass), 不能分开用 softmax + sigmoid
    pri_full = torch.softmax(logits[0], 0).cpu().numpy().astype(np.float64)
    pri = pri_full[:-1]
    pp = pri_full[-1]
    d = np.random.dirichlet(np.full(len(legal), dirichlet))
    next_player = other(env.current)  # 落子后轮到对手
    for i, m in enumerate(legal):
        root.children[m] = Node(0.75 * pri[m] + 0.25 * d[i], next_player)
    root.children[PASS] = Node(pp, next_player)
    s = sum(c.prior for c in root.children.values())
    for c in root.children.values():
        c.prior /= s

    for _ in range(num_sim):
        node = root
        sim = env.clone()  # 用 clone 代替 deepcopy
        path = [node]
        while node.children and not sim.done:
            if node.outcome is not None:  # 已知终局结果, 不再搜索子节点
                break
            best, ba = -1e9, None
            for a, ch in node.children.items():
                # 修复: ch.value 是子节点视角(对手), 需取反得到己方视角
                u = -ch.value + cpuct * ch.prior * math.sqrt(node.visits + 1) / (1 + ch.visits)
                if u > best:
                    best, ba = u, a
            node = node.children[ba]
            path.append(node)
            sim.play(ba)
            if node.outcome is not None:
                break

        if node.outcome is not None:
            v = 1.0 if node.outcome == node.to_move else -1.0
        elif sim.done:
            # 游戏在模拟中结束 (如连续 pass), 记录终局结果
            node.outcome = sim.winner
            v = 1.0 if node.outcome == node.to_move else -1.0
        else:
            lgl = sim.legal_moves()
            # 无论是否有合法落子, PASS 总是合法的, 用网络估值扩展叶节点
            with torch.no_grad():
                ll, val = model(torch.from_numpy(sim.get_features()[None]).to(device))
                pr_full = torch.softmax(ll[0], 0).cpu().numpy().astype(np.float64)
                pr = pr_full[:-1]
                ppp = pr_full[-1]
            if lgl:
                for m in lgl:
                    node.children[m] = Node(pr[m], sim.current)
            node.children[PASS] = Node(ppp, sim.current)
            s2 = sum(c.prior for c in node.children.values())
            for c in node.children.values():
                c.prior /= s2
            v = float(val.item())

        tmp = v
        for n in reversed(path):
            n.visits += 1
            n.value += (tmp - n.value) / n.visits
            tmp = -tmp

    visits = np.zeros(BOARD * BOARD + 1, dtype=np.float64)
    for a, ch in root.children.items():
        visits[a] = ch.visits
    if visits.sum() == 0:
        for m in legal:
            visits[m] = pri[m]
        visits[PASS] = pp
    return visits.astype(np.float32), 0.0


def run_mcts_parallel(models, env, devices, num_sim=SIMS_PER_MOVE, cpuct=1.0, dirichlet=0.03):
    """多GPU并行MCTS: 每个GPU跑 num_sim/N 次独立搜索, 合并访问次数
    棋谱质量与单GPU跑num_sim次基本一致 (两棵独立树合并≈一棵大树)"""
    n = len(devices)
    per_gpu = num_sim // n
    remainder = num_sim % n
    results = [None] * n

    def worker(idx, model, device, sims):
        np.random.seed((int(time.time() * 1e6) ^ threading.get_ident()) % (2**32))
        env_copy = env.clone()  # 每个线程独立 env, 防止 _is_legal 竞态
        results[idx] = run_mcts_single(model, env_copy, device, num_sim=sims,
                                        cpuct=cpuct, dirichlet=dirichlet)

    threads = []
    for i, (model, device) in enumerate(zip(models, devices)):
        sims = per_gpu + (1 if i < remainder else 0)
        t = threading.Thread(target=worker, args=(i, model, device, sims))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 合并访问次数
    total_visits = np.zeros(BOARD * BOARD + 1, dtype=np.float64)
    for v, _ in results:
        total_visits += v
    return total_visits.astype(np.float32), 0.0


# ========== 自对弈 ==========
def generate_game(models, devices, num_sim=SIMS_PER_MOVE, temp=1.0, temp_th=30,
                  game_idx=0):
    env = GoEnv()
    samples = []
    mc = 0
    max_moves = BOARD * BOARD * 2  # 安全上限, 防止无限对局
    t_game = time.time()
    gpu_label = f"{len(devices)}xGPU"
    while not env.done and mc < max_moves:
        t_move = time.time()
        visits, _ = run_mcts_parallel(models, env, devices, num_sim=num_sim)
        # AlphaGo Zero 温度调度: 前 30 步 temp=1 (比例选子), 之后 temp→0 (贪心选子)
        if mc < temp_th:
            probs = visits.astype(np.float64)  # temp=1, 按访问次数比例
        else:
            # temp→0, 贪心: 选访问次数最多的落子
            probs = np.zeros_like(visits, dtype=np.float64)
            probs[np.argmax(visits)] = 1.0
        s = probs.sum()
        if s <= 0:
            probs = np.zeros_like(visits)
            probs[PASS] = 1.0
            s = 1.0
        probs /= s
        move = int(np.random.choice(len(probs), p=probs))
        feats = env.get_features()
        # 策略目标: 与 MCTS 访问次数成正比 (AlphaGo Zero 标准)
        pol = visits / (visits.sum() + 1e-8)
        samples.append((feats, pol.astype(np.float32), env.current))
        try:
            env.play(PASS) if move == PASS else env.play(move)
        except ValueError:
            break  # 安全退出, 保留已收集的样本
        mc += 1
        # 每 10 手打印一次进度, 让用户看到脚本在正常运行
        if mc % 10 == 0:
            move_time = time.time() - t_move
            print(f'  [{gpu_label}] 第{game_idx}局 第{mc}手 | '
                  f'每手{move_time:.1f}s | 已{time.time()-t_game:.0f}s | '
                  f'{"PASS" if move==PASS else f"({move//BOARD},{move%BOARD})"}',
                  flush=True)
    winner = env.winner
    if winner is None or not samples:
        return []  # 异常对局, 跳过
    return [(f, p, np.float32(1.0 if mv == winner else -1.0)) for f, p, mv in samples]


def generate_games_sequential(model, num_games, devices, num_sim=SIMS_PER_MOVE):
    """单局自对弈 (多GPU并行MCTS), 逐局运行"""
    # 创建每个GPU的模型副本
    model_copies = [copy.deepcopy(model).to(dev) for dev in devices]
    for mc in model_copies:
        mc.eval()

    all_samples = []
    t0 = time.time()
    for g in range(num_games):
        samples = generate_game(model_copies, devices, num_sim=num_sim, game_idx=g + 1)
        all_samples.extend(samples)
        elapsed = time.time() - t0
        print(f'  自对弈 {g+1}/{num_games} 局 | '
              f'累计 {len(all_samples)} 样本 | {elapsed:.0f}s', flush=True)
    return all_samples


# ========== 训练 ==========
def augment_sample(feats, pol):
    """随机应用 8 种对称性变换之一 (AlphaGo Zero/Leela Zero 标准数据增强, 8 倍数据效率)"""
    sym = np.random.randint(8)
    if sym == 0:
        return feats, pol
    board_pol = pol[:-1].reshape(BOARD, BOARD).copy()
    pass_pol = pol[-1]
    if sym == 1:      # 水平翻转
        feats, board_pol = feats[:, :, ::-1], board_pol[:, ::-1]
    elif sym == 2:    # 垂直翻转
        feats, board_pol = feats[:, ::-1, :], board_pol[::-1, :]
    elif sym == 3:    # 旋转 90°
        feats, board_pol = np.rot90(feats, 1, axes=(1, 2)), np.rot90(board_pol, 1)
    elif sym == 4:    # 旋转 180°
        feats, board_pol = np.rot90(feats, 2, axes=(1, 2)), np.rot90(board_pol, 2)
    elif sym == 5:    # 旋转 270°
        feats, board_pol = np.rot90(feats, 3, axes=(1, 2)), np.rot90(board_pol, 3)
    elif sym == 6:    # 主对角线翻转
        feats, board_pol = np.swapaxes(feats, 1, 2), board_pol.T
    elif sym == 7:    # 反对角线翻转
        feats, board_pol = np.swapaxes(feats, 1, 2)[:, ::-1, :][:, :, ::-1], board_pol[::-1, ::-1].T
    pol_new = np.empty_like(pol)
    pol_new[:-1] = board_pol.reshape(-1)
    pol_new[-1] = pass_pol
    return np.ascontiguousarray(feats), pol_new


def save_buffer(buffer, path):
    """保存 buffer 到磁盘 (防止 Kaggle 断线丢数据). 空 buffer 时删除旧文件."""
    if not buffer:
        if os.path.exists(path):
            os.remove(path)
            print(f'  buffer 已清空: 删除 {path}', flush=True)
        return
    feats = np.stack([s[0] for s in buffer])
    pols = np.stack([s[1] for s in buffer])
    vals = np.array([s[2] for s in buffer], dtype=np.float32)
    np.savez_compressed(path, feats=feats, pols=pols, vals=vals)
    print(f'  buffer 已保存: {len(buffer)} 样本 -> {path}', flush=True)


def load_buffer(path):
    """从磁盘加载 buffer"""
    if not os.path.exists(path):
        return []
    data = np.load(path)
    return list(zip(data['feats'], data['pols'], data['vals']))


def save_progress(games_done):
    """保存自对弈进度 (跨 session 续训用)"""
    np.savez_compressed(PROGRESS_PATH, games_done=games_done)


def load_progress():
    """加载自对弈进度, 返回已累计局数"""
    if not os.path.exists(PROGRESS_PATH):
        return 0
    prog = np.load(PROGRESS_PATH, allow_pickle=True)
    return int(prog['games_done'])


def train_batch(model, optimizer, scheduler, buffer, batch_size, iters, device):
    """训练, 返回 (avg_loss, avg_policy_loss, avg_value_loss). 每隔 100 步打印进度."""
    model.train()
    total_loss = 0.0
    total_ploss = 0.0
    total_vloss = 0.0
    actual_iters = 0
    for step in range(iters):
        if len(buffer) < batch_size:
            break
        idx = np.random.choice(len(buffer), batch_size, replace=False)
        # 对称性增强: 每个样本随机变换, 8 倍数据效率
        feats_list, pol_list, val_list = [], [], []
        for i in idx:
            f, p, v = buffer[i]
            f_aug, p_aug = augment_sample(f, p)
            feats_list.append(f_aug)
            pol_list.append(p_aug)
            val_list.append(v)
        feats = torch.tensor(np.stack(feats_list), dtype=torch.float32).to(device)
        pol = torch.tensor(np.stack(pol_list), dtype=torch.float32).to(device)
        val = torch.tensor(np.array(val_list, dtype=np.float32),
                           dtype=torch.float32).unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits, v = model(feats)
        policy_loss = -(pol * F.log_softmax(logits, 1)).sum(1).mean()
        value_loss = F.mse_loss(v, val)
        loss = policy_loss + value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total_ploss += policy_loss.item()
        total_vloss += value_loss.item()
        actual_iters += 1

        if (step + 1) % 100 == 0:
            print(f'  训练 {step+1}/{iters} | loss {total_loss/actual_iters:.4f} '
                  f'(策略 {total_ploss/actual_iters:.4f} + 价值 {total_vloss/actual_iters:.4f}) | '
                  f'lr {optimizer.param_groups[0]["lr"]:.6f}', flush=True)

    return total_loss / max(actual_iters, 1), \
           total_ploss / max(actual_iters, 1), \
           total_vloss / max(actual_iters, 1)


# ========== 主循环 ==========
def main():
    # 设备
    if NUM_GPUS >= 2:
        devices = [torch.device(f'cuda:{i}') for i in range(NUM_GPUS)]
    elif torch.cuda.is_available():
        devices = [torch.device('cuda:0')]
    else:
        devices = [torch.device('cpu')]
    train_device = devices[0]

    # 网络
    net = GoNet(in_channels=17, channels=NET_CHANNELS, blocks=NET_BLOCKS).to(train_device)
    model_path = find_model()
    if model_path:
        # strict=False: 兼容旧版本 checkpoint(没有输入卷积 BN/ibn)。
        # 缺失的 ibn 保持身份初始化(gamma=1,beta=0,mean=0,var=1),
        # 与旧架构数学等价, 训练会自动调整。
        missing, unexpected = net.load_state_dict(
            torch.load(model_path, map_location=train_device), strict=False)
        if 'ibn.weight' in missing:
            print('旧权重无输入 BN, 已按身份 BN 初始化(与旧架构数学等价)')
        print(f'已载入权重: {model_path}')
    else:
        print('无历史权重, 随机初始化')

    optimizer = Adam(net.parameters(), LR, weight_decay=WEIGHT_DECAY)
    # 估算训练步数: GAMES_BEFORE_TRAIN 局 × ~100样本 / BATCH × TRAIN_EPOCHS
    est_train_steps = GAMES_BEFORE_TRAIN * 100 // BATCH * TRAIN_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(est_train_steps, 500), eta_min=1e-5)

    # 尝试加载上次保存的 buffer 和进度 (跨 session 续训)
    buffer = load_buffer(BUFFER_PATH)
    if buffer:
        print(f'已加载上次 buffer: {len(buffer)} 样本')
    else:
        buffer = []
    # 加载已累计的局数 (上次跑到哪)
    games_done = load_progress()
    if games_done > 0:
        print(f'已加载上次进度: {games_done}/{GAMES_BEFORE_TRAIN} 局')
    start = time.time()
    budget = BUDGET_HOURS * 3600

    print(f'开始 | GPU: {NUM_GPUS} | 预算: {BUDGET_HOURS}h | '
          f'目标: {GAMES_BEFORE_TRAIN} 局 | visits: {SIMS_PER_MOVE}')
    print(f'流程: 自对弈 {GAMES_BEFORE_TRAIN} 局 → 训练 {TRAIN_EPOCHS} epochs → 保存')
    if games_done > 0:
        print(f'续训: 已累计 {games_done} 局, 还需 {GAMES_BEFORE_TRAIN - games_done} 局触发训练')

    # ════════════════════════════════════════════════════
    # 主循环: 自对弈 → 攒满1200局 → 训练 → 重置 → 继续
    # ════════════════════════════════════════════════════
    train_reserve = 1800  # 预留 30 分钟给训练
    total_sessions = 0    # 已完成的训练轮次数

    while time.time() - start < budget - train_reserve:
        # ── 阶段 1: 自对弈 ──
        print(f'\n=== 自对弈 (累计 {games_done}/{GAMES_BEFORE_TRAIN} 局) ===')
        while (games_done < GAMES_BEFORE_TRAIN and
               time.time() - start < budget - train_reserve):
            t0 = time.time()
            samples = generate_games_sequential(
                net, GAMES_PER_BATCH, devices, num_sim=SIMS_PER_MOVE)
            buffer.extend(samples)
            if len(buffer) > BUFFER_MAX:
                buffer = buffer[-BUFFER_MAX:]
            selfplay_time = time.time() - t0
            games_done += GAMES_PER_BATCH

            # 每批都保存 (防止断线丢数据)
            torch.save(net.state_dict(), MODEL_OUT)
            save_buffer(buffer, BUFFER_PATH)
            save_progress(games_done)

            elapsed = time.time() - start
            remaining = (budget - elapsed) / 60
            need = GAMES_BEFORE_TRAIN - games_done
            print(f'  自对弈 {games_done}/{GAMES_BEFORE_TRAIN} 局'
                  f' (还差{max(need,0)}局训练) | '
                  f'buffer {len(buffer)} 样本 | '
                  f'本批 {selfplay_time:.0f}s | '
                  f'已用 {elapsed/60:.1f}m | 剩余 {remaining:.1f}m',
                  flush=True)

        # ── 阶段 2: 训练 ──
        if games_done >= GAMES_BEFORE_TRAIN:
            total_sessions += 1
            print(f'\n=== 训练第 {total_sessions} 轮 ({len(buffer)} 样本, {TRAIN_EPOCHS} epochs) ===')

            if len(buffer) >= BATCH:
                train_steps = max(len(buffer) // BATCH * TRAIN_EPOCHS, 100)
                t1 = time.time()
                avg_loss, avg_ploss, avg_vloss = train_batch(
                    net, optimizer, scheduler, buffer, BATCH, train_steps, train_device)
                train_time = time.time() - t1

                torch.save(net.state_dict(), MODEL_OUT)
                lr_now = optimizer.param_groups[0]['lr']
                print(f'训练完成 | 步数 {train_steps} | loss {avg_loss:.4f} '
                      f'(策略 {avg_ploss:.4f} + 价值 {avg_vloss:.4f}) | '
                      f'lr {lr_now:.6f} | 训练时间 {train_time:.0f}s',
                      flush=True)
            else:
                print(f'数据不足 ({len(buffer)} < {BATCH}), 跳过训练')

            # 重置: 清空 buffer + 归零进度, 开始下一轮 1200 局
            buffer.clear()
            games_done = 0
            save_buffer([], BUFFER_PATH)
            save_progress(0)
            torch.save(net.state_dict(), MODEL_OUT)
            print('已重置进度, 开始下一轮 1200 局自对弈')
        else:
            # 预算用完, 跑不满
            break

    # ════════════════════════════════════════════════════
    # 最终保存
    # ════════════════════════════════════════════════════
    torch.save(net.state_dict(), MODEL_OUT)
    # 导出引擎可读的 v3 权重(与 leelaz.exe -w 互操作)
    export_leela_v3(net, MODEL_OUT.replace(".pt", ".v3.txt.gz"))
    save_buffer(buffer, BUFFER_PATH)
    save_progress(games_done)
    elapsed = time.time() - start
    print(f'\n预算结束 | 完成训练 {total_sessions} 轮 | '
          f'当前进度 {games_done}/{GAMES_BEFORE_TRAIN} 局 | '
          f'buffer {len(buffer)} 样本 | '
          f'总用时 {elapsed/60:.1f}m | 权重已存到 {MODEL_OUT}')


if __name__ == '__main__':
    main()
