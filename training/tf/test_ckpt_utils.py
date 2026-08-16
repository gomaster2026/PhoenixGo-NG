#!/usr/bin/env python3
"""ckpt_utils / 断点续训核心机制的单元测试。

运行:
    python -m unittest test_ckpt_utils          # 全部（含 TF 机制测试）
    python -m unittest test_ckpt_utils.TestPure # 仅纯函数（无 TF 依赖）

设计说明: 借鉴 betago 项目的断点续训"思想"（原子保存 + 进度元数据），
实现完全基于现代 TF Saver checkpoint + meta.json，无 betago 代码。
"""
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_utils


def _mk_ckpt(directory, step):
    """在目录里伪造一个完整 checkpoint（.index + .data + .meta）。"""
    prefix = os.path.join(directory, "leelaz-model-%d" % step)
    for suffix in (".index", ".data-00000-of-00001", ".meta"):
        with open(prefix + suffix, "wb") as f:
            f.write(b"x")
    return prefix


class TestPure(unittest.TestCase):
    """纯函数测试（不需要 TensorFlow）。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_ckpt_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse_lr_schedule(self):
        self.assertEqual(ckpt_utils.parse_lr_schedule("0:0.05,50000:0.02"),
                         [(0, 0.05), (50000, 0.02)])
        self.assertEqual(ckpt_utils.parse_lr_schedule("[(0,0.05),(50000,0.02)]"),
                         [(0, 0.05), (50000, 0.02)])
        self.assertIsNone(ckpt_utils.parse_lr_schedule(""))
        self.assertIsNone(ckpt_utils.parse_lr_schedule("default"))

    def test_find_latest_empty(self):
        p, s = ckpt_utils.find_latest_checkpoint(self.tmp)
        self.assertIsNone(p)
        self.assertIsNone(s)

    def test_find_latest_picks_max_step(self):
        _mk_ckpt(self.tmp, 1000)
        _mk_ckpt(self.tmp, 5000)
        _mk_ckpt(self.tmp, 2000)
        p, s = ckpt_utils.find_latest_checkpoint(self.tmp)
        self.assertEqual(s, 5000)
        self.assertTrue(p.endswith("leelaz-model-5000"))

    def test_find_latest_skips_broken(self):
        # 只有 .index 没有 .data -> 视为损坏，跳过
        _mk_ckpt(self.tmp, 1000)
        with open(os.path.join(self.tmp, "leelaz-model-9999.index"), "wb") as f:
            f.write(b"x")
        p, s = ckpt_utils.find_latest_checkpoint(self.tmp)
        self.assertEqual(s, 1000)

    def test_save_meta_roundtrip(self):
        ckpt_utils.save_meta(self.tmp, blocks=2, filters=16,
                             lr_schedule=[(0, 0.05), (100, 0.01)])
        meta = ckpt_utils.checkpoint_architecture(self.tmp)
        self.assertEqual(meta["blocks"], 2)
        self.assertEqual(meta["filters"], 16)
        self.assertEqual(meta["lr_schedule"], [[0, 0.05], [100, 0.01]])

    def test_prune_checkpoints(self):
        _mk_ckpt(self.tmp, 1000)
        _mk_ckpt(self.tmp, 2000)
        _mk_ckpt(self.tmp, 3000)
        ckpt_utils.prune_checkpoints(self.tmp, keep_n=2)
        left = [f for f in os.listdir(self.tmp) if f.startswith("leelaz-model")]
        self.assertIn("leelaz-model-3000.index", left)
        self.assertIn("leelaz-model-2000.index", left)
        self.assertNotIn("leelaz-model-1000.index", left)

    def test_resolve_start_resume(self):
        """有 checkpoint -> resumed=True，结构信息从 meta.json 读取。"""
        _mk_ckpt(self.tmp, 1000)
        ckpt_utils.save_meta(self.tmp, blocks=4, filters=32,
                             lr_schedule=[(0, 0.05), (100, 0.01)])
        called = []
        res = ckpt_utils.resolve_start(
            self.tmp, 2, 16, None, lambda d: called.append(d) or None)
        prefix, blocks, filters, lr, resumed = res
        self.assertTrue(resumed)
        self.assertEqual(blocks, 4)
        self.assertEqual(filters, 32)
        self.assertEqual(lr, [(0, 0.05), (100, 0.01)])
        self.assertEqual(called, [])

    def test_resolve_start_fresh(self):
        """无 checkpoint -> 调用 convert_fn 并写 meta.json。"""
        called = []

        def convert(d):
            called.append(d)
            return _mk_ckpt(d, 0)

        res = ckpt_utils.resolve_start(
            self.tmp, 8, 64, ckpt_utils.DEFAULT_LR_SCHEDULE, convert)
        self.assertFalse(res[4])
        self.assertEqual(called, [self.tmp])
        meta = ckpt_utils.checkpoint_architecture(self.tmp)
        self.assertEqual(meta["blocks"], 8)

    def test_resolve_start_broken_ckpt(self):
        """只有 .index 无 .data -> 视为无 checkpoint，走 convert_fn。"""
        with open(os.path.join(self.tmp, "leelaz-model-5000.index"), "wb") as f:
            f.write(b"x")
        called = []
        res = ckpt_utils.resolve_start(
            self.tmp, 2, 16, None,
            lambda d: called.append(d) or None)
        self.assertFalse(res[4])
        self.assertEqual(len(called), 1)

    def test_copy_atomic(self):
        src = os.path.join(self.tmp, "src.txt")
        with open(src, "w") as f:
            f.write("hello")
        dst = os.path.join(self.tmp, "sub", "dst.txt")
        ckpt_utils.copy_atomic(src, dst)
        with open(dst) as f:
            self.assertEqual(f.read(), "hello")


def _tf_available():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_tf_available(), "TensorFlow 不可用，跳过机制测试")
class TestResumeMechanism(unittest.TestCase):
    """断点续训核心机制：checkpoint 保存/恢复 global_step 与 learning_rate。

    只测关键机制，不跑完整训练循环（避免 CPU 图构建耗时）。
    """

    def test_lr_schedule_and_resume(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        from tfprocess import TFProcess, optimistic_restore

        lr_schedule = [(0, 0.05), (15, 0.01)]  # step>=15 后 lr=0.01
        tmpdir = tempfile.mkdtemp(prefix="resume_mech_")
        try:
            # 阶段1: 构造网络，模拟训练到 step=20，保存 checkpoint
            tp = TFProcess(2, 16, lr_schedule=lr_schedule)
            tp.init(batch_size=1, gpus_num=1)
            tp.session.run(tf.assign(tp.global_step, 20))
            tp.session.run(tp.update_lr_op)
            lr_now = tp.session.run(tp.learning_rate)
            self.assertAlmostEqual(lr_now, 0.01, places=6)
            prefix = os.path.join(tmpdir, "leelaz-model-20")
            tp.saver.save(tp.session, prefix)  # 不传 global_step，避免 -20 后缀
            tp.session.close()

            # 阶段2: 重新构造同结构网络，从 checkpoint 恢复
            tf.reset_default_graph()
            tp2 = TFProcess(2, 16, lr_schedule=lr_schedule)
            tp2.init(batch_size=1, gpus_num=1)
            optimistic_restore(tp2.session, prefix)
            gs = tf.train.global_step(tp2.session, tp2.global_step)
            lr2 = tp2.session.run(tp2.learning_rate)
            self.assertEqual(gs, 20)          # global_step 连续
            self.assertAlmostEqual(lr2, 0.01, places=6)  # lr 自动对齐

            # 恢复后再跑 update_lr_op 无冲突
            tp2.session.run(tp2.update_lr_op)
            self.assertAlmostEqual(
                tp2.session.run(tp2.learning_rate), 0.01, places=6)

            # 步数回退到 <15 时 lr 正确回落
            tp2.session.run(tf.assign(tp2.global_step, 10))
            tp2.session.run(tp2.update_lr_op)
            self.assertAlmostEqual(
                tp2.session.run(tp2.learning_rate), 0.05, places=6)
            tp2.session.close()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
