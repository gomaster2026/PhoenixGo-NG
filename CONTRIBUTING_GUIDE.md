# 算力贡献教程

感谢你想为这个项目贡献算力！你 GPU 自对弈产生的**训练数据**（含 MCTS 搜索概率）是每次强化学习迭代的基础。

**总耗时**: 约 10 分钟（首次）+ 之后每次 10-30 分钟（取决于 GPU）
**你需要**: 一台有 NVIDIA GPU 的电脑（CPU 也能跑），Python 3.8+，以及组织者给你的信息:
- **Gitee 私人令牌**（`--token`，用于上传 chunk，需 `projects` 权限；因为数据仓库是私密的，下载也需要它）
- **数据仓库位置**（`--data-owner` / `--data-repo`，即 `gitee.com/组织者登录名/shuju`）
- **权重仓库位置**（`--weight-owner` / `--weight-repo`）

> 注意: 所有仓库地址里的 `--data-owner` / `--weight-owner` 填 **Gitee 登录名**（URL 里那个英文），不是中文昵称。
> 不需要 Kaggle 账号，不需要 kaggle CLI，不需要建 Dataset。找组织者要令牌和仓库位置即可。

---

## 第一步：准备 Gitee 私人令牌

1. 注册 [Gitee](https://gitee.com)（如果没有账号）
2. **设置** → **安全设置** → **私人令牌** → 生成私人令牌
3. 权限勾选 **projects**（项目相关）
4. 复制令牌，后面 `--token` 用

> 令牌就是你的"上传钥匙"，只在本机使用，不要发给别人。

---

## 第二步：准备 leelaz 引擎

方法 A（推荐）: 从组织者的 Releases 下载编译好的 `leelaz.exe`（Windows）或 `leelaz`（Linux）

方法 B: 自己编译（Linux 一条命令）:

```bash
apt install -y build-essential cmake libboost-all-dev zlib1g-dev
git clone --depth 1 https://gitee.com/组织者用户名/leela-zero-next
cd leela-zero-next
mkdir build && cd build
cmake .. -DUSE_CPU_ONLY=1
make -j$(nproc)
```

> 注意: `node_client.py` 现在**用 CPU 也能跑**，只是慢。GPU 版 leelaz 需要 OpenCL。

---

## 第三步：下载仓库

```bash
git clone --depth 1 https://gitee.com/组织者用户名/leela-zero-next
cd leela-zero-next
```

---

## 第四步：跑自对弈 + 自动上传（一条命令）

```bash
python distributed/node_client.py \
    --token 你的Gitee私人令牌 \
    --data-owner 组织者的gitee登录名 \
    --data-repo shuju \
    --weight-owner 组织者的gitee登录名 \
    --weight-repo lz-weights \
    --node-name 你的昵称 \
    --leelaz C:\path\to\leelaz.exe \
    --games 20 \
    --playouts 800
```

脚本自动完成: **从 Gitee 下载最新权重（自动合并分片）→ 自对弈 → 每局 `dump_training` 生成训练数据 → 上传到 Gitee 数据仓库**。

参数说明:

| 参数 | 必须 | 说明 |
|------|------|------|
| `--token` | 是 | 你的 Gitee 私人令牌（`projects` 权限；数据仓库私密，下载也需要） |
| `--data-owner` | 是 | 数据仓库所有者（组织者的 gitee 登录名） |
| `--data-repo` | 是 | 数据仓库名（默认 `shuju`） |
| `--weight-owner` | 是 | 权重仓库所有者（组织者的 gitee 用户名） |
| `--weight-repo` | 是 | 权重仓库名（如 `lz-weights`） |
| `--weight-platform` | 否 | `gitee`（默认）或 `github` |
| `--node-name` | 否 | 你的昵称，上传的文件会标注来源 |
| `--leelaz` | 是 | leelaz 引擎路径 |
| `--games` | 否 | 每轮局数（默认 10） |
| `--playouts` | 否 | 每步搜索量（默认 800，越大越强越慢） |
| `--seed` | 否 | 随机种子（默认自动） |
| `--keep-sgf` | 否 | 同时把棋谱保存到本地 `sgf_output/` |

> 运行前请先确认 `phoenixgo` 权重: 下载最新权重后会在终端显示大小，确认 >100MB 即为正常。
> 大权重（>100MB）在 Gitee 是分片存储的，脚本会自动下载并合并，无需手动处理。

### 周期贡献（定时任务）

Windows 任务计划程序 / Linux cron 每周执行一次即可:

```bash
python distributed/node_client.py --token xxx --data-owner 组织者的登录名 --data-repo shuju --weight-owner xxx --weight-repo lz-weights --leelaz ./leelaz.exe --games 20
```

---

## 上传了什么？

每次运行，Gitee 数据仓库 `shuju` 的 `chunks/` 目录会新增 N 个训练文件:

```
chunks/
  alice_1690000000_train_000001.0.gz   ← 第1局训练数据（含 MCTS 概率，你的昵称前缀）
  alice_1690000000_train_000002.0.gz   ← 第2局
  ...
```

每个 `.gz` 文件是 Leela Zero V1 格式的训练数据（gzip 压缩），包含每步棋的 **19×19 棋盘输入 + 362 维 MCTS 搜索概率 + 胜负结果**。组织者的 Kaggle 爬虫会自动下载这些文件进行训练。

上传后**告诉组织者你的昵称**，他就能看到你的贡献。

---

## 常见问题

### Q: leelaz 启动报错 "缺少 DLL"

Windows 上需要这些 DLL 文件（放在 leelaz 同目录）:

- `libgcc_s_seh-1.dll`
- `libstdc++-6.dll`
- `libwinpthread-1.dll`
- `zlib1.dll`

从 Releases 下载完整包。

### Q: 上传失败 "token 不正确" / 403

检查 `--token` 是否是你的 Gitee 私人令牌，且勾选了 `projects` 权限。确认 `--data-owner` / `--data-repo` 拼写正确。

### Q: 没有 NVIDIA GPU，只有 CPU 能跑吗？

能。`-DUSE_CPU_ONLY=1` 编译的 leelaz 用 CPU 计算，只是每步搜索慢，`--playouts 200` 也够产生训练数据。

### Q: GPU 不够好，跑得慢

- `--playouts 200` 减少搜索量，每局更快
- `--games 10` 减少局数
- 即使 GTX 960 也能跑，只是慢点

### Q: 我关机了，训练数据会丢吗？

只要上传成功了就不会丢。已上传的文件在 Gitee 数据仓库，本机可以删除。

### Q: 数据仓库满了怎么办？

chunk 是临时数据，组织者每次训练完会自动删除旧 chunk 释放空间。如果上传时报错，多半是仓库快满了，等下一轮训练清空后再传。

---

## 我能贡献多少？

| 显卡 | 20 局耗时 | 每周能贡献 |
|------|-----------|-----------|
| RTX 4090 | ~3 分钟 | 无限 |
| RTX 3060 | ~8 分钟 | 随便跑 |
| GTX 1060 | ~15 分钟 | 有空就开 |
| GTX 960 | ~30 分钟 | 挂机跑 |

每局产生的训练数据约 10KB，1000 局才 10MB，不影响硬盘。

**谢谢你的贡献！**
