# PhoenixGo-NG

基于AlphaGo Zero论文，leela zero和PhoenixGo的分布式围棋引擎与训练框架。
通过社区算力贡献，复现PhoenixGo

**English** | [中文说明](README.zh.md)

## 什么是PhoenixGo-NG？

PhoenixGo-NG是一个没有人类知识先验的围棋程序，复现DeepMind论文
[Mastering the Game of Go without Human Knowledge]

### 核心技术

- MCTS（蒙特卡洛树搜索），但不使用蒙特卡洛 rollout
- 深度残差卷积神经网络（ResNet）同时输出策略（policy）和价值（value）
- 通过自我对弈 + 强化学习训练，不依赖人类棋谱

### 在Leela Zero源码基础上增加或修改了：

- PhoenixGo权重格式兼容（V3）
- Trunk / Policy / Value Head 的 BatchNorm 扩展
- 分布式训练基础设施
- 社区算力贡献客户端

## 特性

- 复现 AlphaGo Zero 论文架构
- OpenCL GPU 加速（NVIDIA / AMD）+ CPU 回退模式
- Winograd 卷积加速（3×3 filter 专用）
- FP16 / FP32 自动精度选择
- 多线程 MCTS（支持 pondering）
- GTP 2.0 协议兼容，可与 Lizzie / Sabaki / GoReviewPartner 等 GUI 联用
- 支持 Leela Zero V1/V3 / PhoenixGo 权重格式
- 分布式训练：贡献者自对弈 → 自动训练 → 权重自动发布

## 编译

### Linux（Ubuntu / Debian）

```bash
sudo apt install cmake g++ libboost-dev libboost-program-options-dev \
    libboost-filesystem-dev opencl-headers ocl-icd-libopencl1 \
    ocl-icd-opencl-dev zlib1g-dev

mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS

```bash
brew install boost cmake zlib

mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Windows

打开 `msvc/leela-zero2015.sln` 或 `msvc/leela-zero2017.sln` 编译。

### CPU only

```bash
cmake .. -DUSE_CPU_ONLY=1
```

### 运行测试

```bash
./tests
```

## 使用方式

### 棋力分析 / 对弈

1.选择适合你的版本，下载你喜欢的GUI，在网上搜索该GUI如何使用即可

2.GTP 命令示例：
   ```
   komi 7.5
   play b q16
   play w d4
   genmove b
   ```

## 网络架构

### 输入

18 个 19×19 的二值平面：

```
平面  1-8 : 当前行棋方在 T=0, -1, ..., -7 的棋子
平面  9-16: 对方在 T=0, -1, ..., -7 的棋子
平面    17: 黑方行棋 = 1，否则 0
平面    18: 白方行棋 = 1，否则 0
```

### 网络结构

```
输入 (18 × 19 × 19)
    │
    ▼ 卷积 (3×3, same padding)
    │
    ▼ Trunk BatchNorm (PhoenixGo 扩展)
    │
    ▼ 残差塔 (N × [Conv3×3 → BN → ReLU → Conv3×3 → BN] + skip)
    │
    ├─── Policy Head ── Conv 1×1 → BN → ReLU → FC(362) ──► 落子概率
    │
    └─── Value Head ── Conv 1×1 → BN → ReLU → FC(256) → ReLU → FC(1) ──► 胜率
```

- 残差块数量可配置（默认 20 块）
- 卷积层后接 BatchNorm
- PhoenixGo-NG权重扩展了 trunk / policy / value head 的 BN gamma/beta 参数

## 权重格式

文本文件，每行一个数值，gzip 压缩。

- 卷积层：2 行（权重 + bias），权重按 `[output, input, 3, 3]` 排列
- BN 层：2 行（mean + variance），PhoenixGo 额外 2 行（gamma + beta）
- 全连接层：2 行（权重 + bias），按 `[output, input]` 排列

顺序：残差塔 → Policy Head → Value Head

## 训练

### 训练数据格式

```
# 16 行十六进制字符串，每行 361 bit = 前 16 个输入平面
# 1 行：行棋方 (0=黑, 1=白)
# 1 行：362 个浮点数（361 个交叉点 + pass）的 MCTS 搜索概率
# 1 行：胜负结果 (1 = 行棋方胜, -1 = 行棋方负)
```

### 使用 TensorFlow 训练

```bash
# 从自对弈数据生成训练样本
src/leelaz -w weights.txt.gz
dump_supervised game.sgf train.out
exit

# 解析并开始训练
training/tf/parse.py 20 256 train.out
```

- `20` = 残差块数量，`256` = 滤波器数量
- 使用 `tf.compat.v1` 兼容 TensorFlow 2.16+
- 详见 `training/tf/` 目录

## 贡献算力

如需贡献算力，请联系 phoenixgo_zero@163.com，预计一天内即可回复。

详细说明见 [CONTRIBUTING_GUIDE.md](CONTRIBUTING_GUIDE.md)。

## 目录结构

```
PhoenixGo/
├── src/                    # C++ 引擎核心
│   ├── Leela.cpp           # 主入口
│   ├── config.h            # 编译配置
│   ├── Network.cpp/h       # 神经网络推理 + 权重加载
│   ├── UCTSearch.cpp/h     # MCTS 搜索
│   ├── UCTNode.cpp/h       # MCTS 节点
│   ├── FastBoard.cpp/h     # 棋盘表示
│   ├── GameState.cpp/h     # 游戏状态管理
│   ├── GTP.cpp/h           # GTP 协议实现
│   ├── OpenCL.cpp/h        # GPU 加速后端
│   ├── CPUPipe.cpp/h       # CPU 神经网络管道
│   ├── Training.cpp/h      # 训练数据导出
│   ├── NNCache.cpp/h       # 神经网络缓存
│   ├── SGFParser.cpp/h     # SGF 棋谱解析
│   ├── Zobrist.cpp/h       # Zobrist 哈希
│   ├── Tuner.cpp/h         # 搜索参数调优
│   └── tests/              # 单元测试
├── training/tf/            # TensorFlow 训练脚本
│   ├── tfprocess.py        # 网络构建 + 训练循环
│   ├── parse.py            # 训练数据解析
│   ├── chunkparser.py      # chunk 数据加载器
│   └── mixprec.py          # 混合精度训练
├── distributed/            # 分布式训练框架
│   ├── node_client.py      # 贡献者客户端
│   ├── kaggle_train.py     # 训练主脚本
│   ├── gitee_data.py       # 数据仓库 API
│   ├── weight_release.py   # 权重分片发布工具
│   └── make_dummy_weights.py # 生成初始权重
├── CMakeLists.txt          # 构建配置
├── COPYING                 # GPLv3 许可证
├── CONTRIBUTING_GUIDE.md   # 贡献算力教程
└── README.md               # 本文件
```

## 核心数据流

```
Leela.cpp (main)
    │
    ├── GTP::execute() ──► GameState
    │                           │
    ├── UCTSearch::think() ◄───┤
    │       │                   │
    │       ├── UCTWorker[] ◄──┘ (多线程搜索)
    │       │       │
    │       │   ┌───▼────┐
    │       │   │ 模拟   │
    │       │   │ selection│
    │       │   │ expansion│
    │       │   │ evaluation ◄── Network::get_output()
    │       │   │ backprop │           │
    │       │   └─────────┘           ▼
    │       │                   ForwardPipe (OpenCL / CPU)
    │       │                       │
    │       └───────────────────────┘
    │
    └── Network::initialize() ──► 加载权重
            │
            └── load_v1_network() / load_v3_network()
                    │
                    └── Winograd 卷积变换 + 权重解析
```

## 编译依赖

- C++14/17 编译器（GCC ≥ 5, Clang, MSVC 2015+）
- CMake ≥ 3.5
- Boost ≥ 1.58（program_options, filesystem, system）
- zlib
- OpenCL 头文件 + ICD Loader（GPU 模式，可选）
- OpenBLAS / Intel MKL（可选，CPU 加速）
- Qt5（可选，validation 工具）

## 相关项目

- `https://github.com/leela-zero/leela-zero`
- `https://github.com/Tencent/PhoenixGo` — 腾讯开源改进版

## 关于招募

我们正在寻找：

**程序员**：参与引擎优化、训练框架开发

**有 GPU 的贡献者**：提供算力进行自对弈训练

如果你有兴趣，请通过 **phoenixgo_zero@163.com** 联系我们。

## License

GPLv3 or later. 部分第三方库使用其各自许可证，兼容 GPLv3。

详见 [COPYING](COPYING) 文件。

## 致谢

本项目基于 Leela Zero by Gian-Carlo Pascutto and contributors。

感谢所有贡献算力的社区成员,本项目全程都有ai辅助，感谢！
