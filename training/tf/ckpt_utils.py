#!/usr/bin/env python3
"""Checkpoint 断点续训工具模块（纯 stdlib，无 TensorFlow 依赖）。

设计借鉴 betago 项目的 TrainingRun 断点设计（进度元数据与模型同存、原子保存），
本项目以 global_step 替代 epoch/chunk 计数，用 meta.json 记录网络结构信息。

checkpoint 文件布局（由 tf.train.Saver 生成）:
    <ckpt_dir>/leelaz-model-<step>.data-00000-of-00001
    <ckpt_dir>/leelaz-model-<step>.index
    <ckpt_dir>/leelaz-model-<step>.meta
    <ckpt_dir>/meta.json          <- 网络结构元数据（本模块维护）

--restore 恢复时传给 parse.py 的是不带后缀的路径前缀:
    <ckpt_dir>/leelaz-model-<step>
"""

import glob
import json
import os
import re
import shutil

CKPT_RE = re.compile(r"^leelaz-model-(\d+)\.index$")
CKPT_PREFIX = "leelaz-model"
META_FILE = "meta.json"

DEFAULT_LR_SCHEDULE = [
    (0, 0.05),
    (50000, 0.02),
    (150000, 0.01),
    (300000, 0.005),
    (600000, 0.002),
]


def parse_lr_schedule(spec):
    """解析 --lr-schedule 参数。

    支持:
      - 空字符串 / "default"  -> 返回 None（用默认调度）
      - "step:lr,step:lr,..." 如 "0:0.05,50000:0.02"
      - 逗号分隔元组列表字符串 "[(0,0.05),(50000,0.02)]"
    返回 list[(step, lr)] 或 None。
    """
    if not spec or spec.strip().lower() in ("default", "none"):
        return None
    text = spec.strip()
    # "step:lr,step:lr" 形式
    if ":" in text and not text.lstrip().startswith("["):
        out = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            step, lr = part.split(":")
            out.append((int(step), float(lr)))
        return out or None
    # "[(0,0.05),...]" 形式
    if text.startswith("["):
        import ast
        pairs = ast.literal_eval(text)
        return [(int(s), float(l)) for s, l in pairs]
    raise ValueError(
        "无法解析 lr schedule: {!r}（支持 'step:lr,step:lr' 或 '[(0,0.05),...]'）".format(spec))


def find_latest_checkpoint(ckpt_dir):
    """扫描目录，返回 (checkpoint_prefix, step) 即最新的 checkpoint。

    以 .index 文件为准（Saver 最后写 .index，写入即代表该 checkpoint 完整）。
    返回的 checkpoint_prefix 不含后缀，可直接传给 parse.py --restore。
    没有任何 checkpoint 时返回 (None, None)。
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return None, None
    best = None
    best_step = -1
    for index_file in glob.glob(os.path.join(ckpt_dir, CKPT_PREFIX + "-*.index")):
        m = CKPT_RE.match(os.path.basename(index_file))
        if not m:
            continue
        step = int(m.group(1))
        prefix = os.path.join(ckpt_dir, CKPT_PREFIX + "-" + str(step))
        # 校验配套数据文件存在，避免选中写了一半的 checkpoint
        if not glob.glob(prefix + ".data-*"):
            continue
        if step > best_step:
            best_step = step
            best = prefix
    return best, best_step if best_step >= 0 else None


def checkpoint_architecture(ckpt_dir):
    """读取 meta.json，返回 dict 或 None（文件不存在/损坏）。"""
    meta_path = os.path.join(ckpt_dir, META_FILE)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_meta(ckpt_dir, blocks, filters, lr_schedule=None):
    """原子写入 meta.json，记录网络结构与学习率调度。"""
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {
        "blocks": int(blocks),
        "filters": int(filters),
        "lr_schedule": [(int(s), float(l)) for s, l in (lr_schedule or DEFAULT_LR_SCHEDULE)],
    }
    tmp_path = os.path.join(ckpt_dir, META_FILE + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, os.path.join(ckpt_dir, META_FILE))


def resolve_start(ckpt_dir, blocks, filters, lr_schedule, convert_fn):
    """断点续训启动决策。

    参数:
      ckpt_dir:    checkpoint 目录
      blocks/filters: 期望的网络结构（无 checkpoint 或 meta 缺失时用于转换基础权重）
      lr_schedule: 期望的学习率调度 (list[(step, lr)])，写入 meta.json
      convert_fn:  回调 fn(ckpt_dir) -> checkpoint_prefix，
                   无 checkpoint 时由调用方从基础权重转换。返回转换后的 checkpoint 前缀。

    返回:
      (restore_prefix, blocks, filters, lr_schedule, resumed)
        - resumed=True  找到历史 checkpoint，直接续训
        - resumed=False 从基础权重转换的新 checkpoint
    """
    prefix, step = find_latest_checkpoint(ckpt_dir)
    if prefix is not None:
        meta = checkpoint_architecture(ckpt_dir)
        if meta is not None:
            blocks = int(meta.get("blocks", blocks))
            filters = int(meta.get("filters", filters))
            lr_schedule = [(int(s), float(l)) for s, l in meta.get(
                "lr_schedule", lr_schedule or DEFAULT_LR_SCHEDULE)]
        return prefix, blocks, filters, lr_schedule, True

    # 没有历史 checkpoint，从基础权重转换（转换函数负责写 meta.json）
    prefix = convert_fn(ckpt_dir)
    if prefix is None:
        return None, blocks, filters, lr_schedule, False
    save_meta(ckpt_dir, blocks, filters, lr_schedule)
    return prefix, blocks, filters, lr_schedule, False


def prune_checkpoints(ckpt_dir, keep_n=5):
    """只保留最近 keep_n 个 checkpoint，删除更旧的（连同 .data/.index/.meta）。

    同时删除 TF 生成的 checkpoint 文本指针文件（下次 Saver.save 会重建）。
    返回删除的文件数。
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return 0
    index_files = [f for f in glob.glob(os.path.join(ckpt_dir, CKPT_PREFIX + "-*.index"))
                   if CKPT_RE.match(os.path.basename(f))]
    index_files.sort(key=lambda f: int(CKPT_RE.match(os.path.basename(f)).group(1)))
    if keep_n <= 0:
        keep_n = 1
    removed = 0
    for old in index_files[:-keep_n]:
        prefix = os.path.splitext(old)[0]
        for f in glob.glob(prefix + ".*"):
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
    # checkpoint 文本指针文件由 Saver 维护，删除旧 checkpoint 后一并清掉，
    # 避免它指向已被删除的 checkpoint（本模块恢复不依赖它）。
    pointer = os.path.join(ckpt_dir, "checkpoint")
    if os.path.isfile(pointer):
        try:
            os.remove(pointer)
        except OSError:
            pass
    return removed


def copy_atomic(src, dst):
    """原子拷贝文件或目录到 dst：先写临时路径，成功后 rename。

    适用于把训练产生的 checkpoint 同步到 --ckpt-dir，
    避免拷贝到一半留下损坏文件（借鉴 betago 的备份-写-删模式）。
    """
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    tmp = dst + ".tmp-{}".format(os.getpid())
    try:
        if os.path.isdir(src):
            if os.path.exists(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(src, tmp)
        else:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
