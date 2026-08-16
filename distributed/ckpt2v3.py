import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

BASE = r"C:/Users/admin/Desktop/新建文件夹 (3)/新建文件夹 (2)/PhoenixGo-win-x64-cpuonly-v1/ckpt"
reader = tf.train.NewCheckpointReader(BASE + "/zero.ckpt-20b-v1")
G = reader.get_tensor

def get(name):
    # name without OptimizeLoss prefix
    return reader.get_tensor(name)

def line(arr):
    return " ".join(str(float(x)) for x in np.ravel(arr))

# NHWC kernel [h,w,in,out] -> LZ [out,in,h,w]
def conv_w(name):
    k = get(name)  # [3,3,in,out] or [1,1,in,out]
    return np.transpose(k, (3, 2, 0, 1))

def conv_unit(kname, bname, bn_prefix, cout):
    w = conv_w(kname)
    b = get(bname)
    g = get(bn_prefix + "/gamma")
    beta = get(bn_prefix + "/beta")
    mean = get(bn_prefix + "/moving_mean")
    var = get(bn_prefix + "/moving_var")
    return [line(w), line(b), line(g), line(beta), line(mean), line(var)]

layers = sorted(set(int(n.split("/layer_")[1].split("/")[0]) for n in reader.get_variable_to_shape_map() if "/zero/layer_" in n and "/Momentum" not in n))
blocks = [l for l in layers if l != 0]
print("layers:", layers, "num residual blocks:", len(blocks))

out = ["3"]
# input conv: kernel [3,3,17,256] + bias, NO BN -> synthesize identity BN
w = conv_w("zero/layer_0/conv_block/conv/kernel")
b = get("zero/layer_0/conv_block/conv/bias")
ch = b.shape[0]
out.append(line(w))
out.append(line(b))
out.append(line(np.ones(ch)))      # gamma = 1
out.append(line(np.zeros(ch)))     # beta = 0
out.append(line(np.zeros(ch)))     # mean = 0
out.append(line(np.ones(ch)))      # var = 1 (identity)
# residual blocks (pre-activation: input_batch_norm)
for blk in sorted(blocks):
    for cb in (0, 1):
        pre = f"zero/layer_{blk}/residual_block/conv_block_{cb}"
        out += conv_unit(pre + "/conv/kernel", pre + "/conv/bias", pre + "/input_batch_norm", ch)
# trunk BN (layer_final)
pre = "zero/layer_final/batch_norm"
out.append(line(get(pre + "/gamma")))
out.append(line(get(pre + "/beta")))
out.append(line(get(pre + "/moving_mean")))
out.append(line(get(pre + "/moving_var")))
# policy head
pre = "zero/policy_head/conv_block"
out += conv_unit(pre + "/conv/kernel", pre + "/conv/bias", pre + "/output_batch_norm", 2)
# dense kernel [722,362] TF -> LZ [362,722]
dk = get("zero/policy_head/dense/kernel")
out.append(line(np.transpose(dk, (1, 0))))
out.append(line(get("zero/policy_head/dense/bias")))
# value head
pre = "zero/value_head/conv_block"
out += conv_unit(pre + "/conv/kernel", pre + "/conv/bias", pre + "/output_batch_norm", 1)
v1 = get("zero/value_head/dense/kernel")     # [361,256]
out.append(line(np.transpose(v1, (1, 0))))
out.append(line(get("zero/value_head/dense/bias")))
v2 = get("zero/value_head/dense_1/kernel")   # [256,1]
out.append(line(np.transpose(v2, (1, 0))))
out.append(line(get("zero/value_head/dense_1/bias")))

with open("D:/tmp/pg_20b_v1.txt", "w") as f:
    f.write("\n".join(out) + "\n")
print("wrote D:/tmp/pg_20b_v1.txt lines:", len(out), "expected:", 29 + 12 * len(blocks))
