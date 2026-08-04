PhoenixGo-NG

A distributed Go engine and training framework based on the AlphaGo Zero paper, Leela Zero, and PhoenixGo.
Reproducing PhoenixGo through community computing power contributions.

What is PhoenixGo-NG?

PhoenixGo-NG is a Go program without human knowledge priors, reproducing the DeepMind paper
[Mastering the Game of Go without Human Knowledge]

Core Technology

- MCTS (Monte Carlo Tree Search), but without Monte Carlo rollout
- Deep residual convolutional neural network (ResNet) outputting both policy and value simultaneously
- Trained through self-play + reinforcement learning, without relying on human game records

Added or modified on top of Leela Zero source code:

- PhoenixGo weight format compatibility (V3)
- BatchNorm extensions for Trunk / Policy / Value Head
- Distributed training infrastructure
- Community computing power contribution client

Features

- Reproduces AlphaGo Zero paper architecture
- OpenCL GPU acceleration (NVIDIA / AMD) + CPU fallback mode
- Winograd convolution acceleration (for 3x3 filters)
- FP16 / FP32 automatic precision selection
- Multi-threaded MCTS (with pondering support)
- GTP 2.0 protocol compatible, works with GUIs such as Lizzie / Sabaki / GoReviewPartner
- Supports Leela Zero V1/V3 / PhoenixGo weight formats
- Distributed training: contributor self-play -> automatic training -> automatic weight publication

Build

Linux (Ubuntu / Debian)

sudo apt install cmake g++ libboost-dev libboost-program-options-dev \
    libboost-filesystem-dev opencl-headers ocl-icd-libopencl1 \
    ocl-icd-opencl-dev zlib1g-dev

mkdir build && cd build
cmake ..
make -j$(nproc)

macOS

brew install boost cmake zlib

mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

Windows

Open msvc/leela-zero2015.sln or msvc/leela-zero2017.sln to build.

CPU only

cmake .. -DUSE_CPU_ONLY=1

Run Tests

./tests

Usage

Analysis / Play

1. Choose the version that suits you, download your preferred GUI, and search online for how to use that GUI.

2. GTP command example:

   komi 7.5
   play b q16
   play w d4
   genmove b

Network Architecture

Input

18 binary planes of size 19x19:

Planes  1-8 : Current player's stones at T=0, -1, ..., -7
Planes  9-16: Opponent's stones at T=0, -1, ..., -7
Plane    17 : 1 if Black to play, 0 otherwise
Plane    18 : 1 if White to play, 0 otherwise

Network Structure

Input (18 x 19 x 19)
    |
    v Convolution (3x3, same padding)
    |
    v Trunk BatchNorm (PhoenixGo extension)
    |
    v Residual Tower (N x [Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN] + skip)
    |
    +--- Policy Head -- Conv 1x1 -> BN -> ReLU -> FC(362) --> Move probabilities
    |
    +--- Value Head -- Conv 1x1 -> BN -> ReLU -> FC(256) -> ReLU -> FC(1) --> Win rate

- Number of residual blocks is configurable (default 20 blocks)
- BatchNorm follows convolution layers
- PhoenixGo-NG weights extend the BN gamma/beta parameters for trunk / policy / value heads

Weight Format

Text file, one value per line, gzip compressed.

- Convolution layer: 2 lines (weights + bias), weights arranged as [output, input, 3, 3]
- BN layer: 2 lines (mean + variance), PhoenixGo adds 2 extra lines (gamma + beta)
- Fully connected layer: 2 lines (weights + bias), arranged as [output, input]

Order: Residual tower -> Policy Head -> Value Head

Training

Training Data Format

# 16 lines of hexadecimal strings, each line 361 bits = first 16 input planes
# 1 line: player to move (0=Black, 1=White)
# 1 line: 362 floating-point values (361 intersections + pass) of MCTS search probabilities
# 1 line: win/loss result (1 = player to move wins, -1 = player to move loses)

Training with TensorFlow

# Generate training samples from self-play data
src/leelaz -w weights.txt.gz
dump_supervised game.sgf train.out
exit

# Parse and start training
training/tf/parse.py 20 256 train.out

- 20 = number of residual blocks, 256 = number of filters
- Uses tf.compat.v1 for TensorFlow 2.16+ compatibility
- See training/tf/ directory for details

Contributing Computing Power

To contribute computing power, please contact phoenixgo_zero@163.com. Expect a reply within one day.

See CONTRIBUTING_GUIDE.md for detailed instructions.

Directory Structure

PhoenixGo/
+-- src/                    # C++ engine core
|   +-- Leela.cpp           # Main entry point
|   +-- config.h            # Build configuration
|   +-- Network.cpp/h       # Neural network inference + weight loading
|   +-- UCTSearch.cpp/h     # MCTS search
|   +-- UCTNode.cpp/h       # MCTS node
|   +-- FastBoard.cpp/h     # Board representation
|   +-- GameState.cpp/h     # Game state management
|   +-- GTP.cpp/h           # GTP protocol implementation
|   +-- OpenCL.cpp/h        # GPU acceleration backend
|   +-- CPUPipe.cpp/h       # CPU neural network pipeline
|   +-- Training.cpp/h      # Training data export
|   +-- NNCache.cpp/h       # Neural network cache
|   +-- SGFParser.cpp/h     # SGF game record parser
|   +-- Zobrist.cpp/h       # Zobrist hashing
|   +-- Tuner.cpp/h         # Search parameter tuning
|   +-- tests/              # Unit tests
+-- training/tf/            # TensorFlow training scripts
|   +-- tfprocess.py        # Network construction + training loop
|   +-- parse.py            # Training data parser
|   +-- chunkparser.py      # Chunk data loader
|   +-- mixprec.py          # Mixed precision training
+-- distributed/            # Distributed training framework
|   +-- node_client.py      # Contributor client
|   +-- kaggle_train.py     # Main training script
|   +-- gitee_data.py       # Data repository API
|   +-- weight_release.py   # Weight shard publication tool
|   +-- make_dummy_weights.py # Generate initial weights
+-- CMakeLists.txt          # Build configuration
+-- COPYING                 # GPLv3 License
+-- CONTRIBUTING_GUIDE.md   # Computing power contribution guide
+-- README.md               # This file

Core Data Flow

Leela.cpp (main)
    |
    +-- GTP::execute() --> GameState
    |                           |
    +-- UCTSearch::think() <---|
    |       |                   |
    |       +-- UCTWorker[] <--+ (multi-threaded search)
    |       |       |
    |       |   +---v----+
    |       |   | Simulation |
    |       |   | selection  |
    |       |   | expansion  |
    |       |   | evaluation <-- Network::get_output()
    |       |   | backprop   |           |
    |       |   +------------+           v
    |       |                   ForwardPipe (OpenCL / CPU)
    |       |                       |
    |       +-----------------------+
    |
    +-- Network::initialize() --> Load weights
            |
            +-- load_v1_network() / load_v3_network()
                    |
                    +-- Winograd convolution transform + weight parsing

Build Dependencies

- C++14/17 compiler (GCC >= 5, Clang, MSVC 2015+)
- CMake >= 3.5
- Boost >= 1.58 (program_options, filesystem, system)
- zlib
- OpenCL headers + ICD Loader (GPU mode, optional)
- OpenBLAS / Intel MKL (optional, CPU acceleration)
- Qt5 (optional, validation tool)

Related Projects

- https://github.com/leela-zero/leela-zero
- https://github.com/Tencent/PhoenixGo -- Tencent's open-source improved version

Recruitment

We are looking for:

Programmers: participate in engine optimization and training framework development

Contributors with GPUs: provide computing power for self-play training

If you are interested, please contact us at phoenixgo_zero@163.com.

License

GPLv3 or later. Some third-party libraries use their respective licenses, compatible with GPLv3.

See the COPYING file for details.

Acknowledgments

This project is based on Leela Zero by Gian-Carlo Pascutto and contributors.

Thanks to all community members who contributed computing power. This project was assisted by AI throughout. Thank you!

