#!/usr/bin/env python3
"""生成一份随机的微型权重文件（v3 PhoenixGo 格式），用于端到端验证整个训练管线。

用法 (Kaggle Notebook, 需先装 tensorflow==2.16.x):
    !python distributed/make_dummy_weights.py --blocks 2 --filters 16 \
        --output /kaggle/working/dummy.txt.gz

生成出的权重是一份随机初始化的极小网络（v3 预激活结构），
leelaz 和 TFProcess 都能加载。
用它跑通 kaggle_train.py，即可证明整条链路可用，且完全不依赖真实权重。
"""

import argparse
import gzip
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../training/tf")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    parser = argparse.ArgumentParser(description="生成随机微型权重")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--output", default="/kaggle/working/dummy.txt.gz")
    args = parser.parse_args()

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tfprocess import TFProcess

    print(f"构建随机网络 blocks={args.blocks} filters={args.filters} ...", flush=True)
    tp = TFProcess(args.blocks, args.filters)
    tp.init(batch_size=1, gpus_num=1)

    txt_path = os.path.splitext(args.output)[0] + ".txt"
    tp.save_leelaz_weights(txt_path)
    tp.session.close()

    with open(txt_path, "rb") as src, gzip.open(args.output, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(txt_path)

    size = os.path.getsize(args.output)
    print(f"随机权重已生成: {args.output} ({size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
