# Leela Zero 分布式训练

**免费方案**: 全部数据走 Gitee API（chunks 数据仓库 + Release 权重仓库）+ Kaggle 免费 T4 GPU 训练。全程不需要信用卡，不需要任何域名/服务器。

```
数据仓库 (Gitee 仓库 lz-data, 公开, chunks/ 目录)
  ├── chunks/*.gz  ← 贡献者上传的训练数据 (chunk, ~0.5MB/局)
  └── Gitee API: 列出 / 下载 / 删除
        ▲                                  │
        │ 上传chunk (Gitee API + 令牌)       │ 列出+下载所有chunk
        │                                  ▼
  贡献者 GPU ───── node_client.py 自对弈 ── Kaggle Notebook (T4)
        ▲                                  │
        │ 下载最新权重                        │ 训练出新权重
        │                                  ▼
  Gitee Release (权重仓库 lz-weights) ← weight_release.py 上传
     权重 >100MB 自动分片, 贡献者自动合并下载
```

整个循环全走 Gitee，**不需要 Cloudflare、不需要域名、不需要 Kaggle Dataset、不需要 kaggle/kagglehub 命令行工具**。

为什么不用 Cloudflare Worker？
- `*.workers.dev` 在国内被 DNS 污染 + TCP 阻断，从国内完全访问不了
- Gitee 国内访问稳定、免费、无需绑卡，API 齐全（contents 读写 + Release 附件 + raw 直链）

## 第一步: 准备数据仓库（存训练 chunk）

1. 注册 [Gitee](https://gitee.com)，新建一个仓库 `shuju`（可以是私密的；私密仓库读写都需要令牌）
2. 获取私人令牌: Gitee → **设置** → **安全设置** → **私人令牌** → 生成（勾选 `projects` 权限）
   - 这个令牌就是"上传 token"，**分发给所有贡献者**（贡献者用它上传 chunk，私密仓库下载也需要它）
   - 注意: 命令里填的是 **Gitee 登录名**（URL 里那个），不是中文显示名

> Gitee 免费仓库总容量有限（约几百 MB）。chunk 是临时数据，训练完会被删除，容量循环使用。

## 第二步: 准备权重仓库（Gitee Release）

1. 在 Gitee 新建仓库 `lz-weights`（公开）
2. 上传初始权重（在本地，或 Kaggle Input 上传后执行）:

```
python distributed/weight_release.py upload --platform gitee \
    --owner 你的用户名 --repo lz-weights \
    --token Gitee私人令牌 \
    --weights phoenixgo-v1.txt.gz
```

- phoenixgo 权重 ~211MB，会自动切成 3 片（每片 <100MB）+ 1 个 meta 清单上传
- 每次训练完，`kaggle_train.py` 会自动调用它发布新权重（覆盖同一个 `lz-latest` tag）

## 第三步: 组织者（你）

### 在 Kaggle Notebook 训练

创建 Notebook → 选 **GPU T4 x2** → 输入代码:

```
!git clone --depth 1 https://gitee.com/ABCradio/AeonGo
%cd AeonGo
!pip install tensorflow==2.16.1
!python distributed/kaggle_train.py --data-owner ABCradio --data-repo shuju --data-token Gitee私人令牌 --weights-path /kaggle/input/你的数据集名/phoenixgo-v1.txt.gz --selfplay-games 10
```

> 注意: 
> - TensorFlow 必须是 **2.16.x**（训练代码用 TF1 风格 API，2.16 是最后完整支持 `tf.compat.v1` 的版本，且 Kaggle Python 3.12 能装）。不要装 2.17+。
> - Notebook 里 `!` 命令**必须写在一行**，不能用 `\` 换行。

- chunk 从 Gitee 数据仓库 `shuju` 下载（私密仓库需要 `--data-token`）
- 权重用 `--weights-path` 指定 Kaggle Input 里的本地文件（首次用 phoenixgo；之后用上一轮训练出的新权重）
- 训练完新权重保存在 `/kaggle/working/weights_new.txt.gz`，从 Notebook Output 下载后发给贡献者
- **上传成功后自动删除数据仓库里的旧 chunk**（释放 Gitee 免费容量，给下一轮贡献者腾地方）
- 想让循环全自动？把权重放到 Gitee Release 后，改用 `--weight-owner`/`--weight-repo`/`--weight-token`（`kaggle_train.py` 会自动发布新权重并供贡献者下载）
- **建议加定时任务**: 每天自动跑一次 Notebook（右侧菜单 → Scheduling）

## 第四步: 贡献者（一条命令）

```
python distributed/node_client.py \
    --token 你的Gitee私人令牌 \
    --data-owner 你的登录名 --data-repo shuju \
    --weights ./current.txt.gz \
    --leelaz ./leelaz.exe --games 20 --playouts 800
```

会自动: 用你发的权重自对弈 → 每局 `dump_training` 生成训练数据 → 自动上传到 Gitee 数据仓库。

不需要 kagglehub、不需要 kaggle 账号、不需要建 Dataset，只要有 leelaz 就行。

## 权重格式说明

- 权重是 Leela Zero V1/V3 格式 `.txt.gz`（PhoenixGo 权重可直接用）
- `kaggle_train.py` 内置 `convert_weights_to_checkpoint()` 把 `.txt.gz` 转成 TF checkpoint
- 训练用 `parse.py`（跳过已废弃的 `train_windows.py`）

## Gitee 容量与限速

| 项目 | 限制 | 对策 |
|------|------|------|
| 单文件（contents API） | ≤100MB | chunk ~0.5MB，远低于上限 |
| 仓库总容量（免费） | 约几百 MB | 训练完自动删除旧 chunk |
| API 调用频率 | 5000 次/小时 | 每局 1 次上传；每轮训练 <2000 次调用 |
| Release 单附件 | ≤100MB | 权重自动分片（weight_release.py） |

## 文件说明

| 文件 | 用途 |
|------|------|
| `gitee_data.py` | 数据仓库模块: Gitee API 上传/列出/下载/删除 chunk |
| `weight_release.py` | 权重发布工具: Gitee/GitHub 分片上传/下载合并 |
| `node_client.py` | 贡献者: 自对弈 + 自动上传 chunk 到 Gitee |
| `kaggle_train.py` | 组织者: 爬取 chunk + 训练 + 发布权重 + 清理旧 chunk |
| `CONTRIBUTING_GUIDE.md` | 贡献者教程 |
| `../training/tf/parse.py` | 核心训练脚本（TF v1, chunkparser + tfprocess） |

> 提示: 本机账号登录名为 `ABCradio`（显示名"甜不辣的神"），命令和文档中所有 `--data-owner` / `--weight-owner` 都用登录名。
