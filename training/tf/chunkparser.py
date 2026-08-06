import random
import numpy as np

# PhoenixGo/AeonGo 输入: 17 通道 = 16 手历史 + 1 颜色
INPUT_CHANNELS = 17
HISTORY_PLANES = 16
NUM_INTERSECTIONS = 19 * 19
POTENTIAL_MOVES = NUM_INTERSECTIONS + 1
# 每个局面在 chunk 中占 19 行: 16 平面 + to_move + 概率 + 结果
PLANES_PER_POS = HISTORY_PLANES + 3


class ChunkParser:
    def __init__(self, chunk_iterable, shuffle_size=1 << 20, sample=1, batch_size=256):
        self.chunk_iterable = chunk_iterable
        self.shuffle_size = shuffle_size
        self.sample = sample
        self.batch_size = batch_size
        self.buffer = []

    def parse(self):
        while True:
            chunkdata = self.chunk_iterable.next()
            if chunkdata is None:
                break
            for parsed in self.parse_chunk(chunkdata):
                self.buffer.append(parsed)
                if len(self.buffer) >= self.shuffle_size:
                    random.shuffle(self.buffer)
                    while len(self.buffer) >= self.batch_size:
                        yield self._pack(self.buffer[:self.batch_size])
                        del self.buffer[:self.batch_size]
        random.shuffle(self.buffer)
        while len(self.buffer) >= self.batch_size:
            yield self._pack(self.buffer[:self.batch_size])
            del self.buffer[:self.batch_size]

    def parse_chunk(self, chunkdata):
        lines = chunkdata.splitlines()
        for i in range(0, len(lines), PLANES_PER_POS):
            pos = lines[i:i + PLANES_PER_POS]
            if len(pos) < PLANES_PER_POS:
                break
            if random.randrange(self.sample) != 0:
                continue
            yield self.parse_position(pos)

    def parse_position(self, pos):
        # 16 个平面, 每行 90 个十六进制字符(每组 4 位) + 最后 1 位
        planes = np.zeros((INPUT_CHANNELS, NUM_INTERSECTIONS), dtype=np.uint8)
        for c in range(HISTORY_PLANES):
            hexstr = pos[c].decode("utf-8")
            for i in range(90):
                nibble = int(hexstr[i], 16)
                base = i * 4
                planes[c, base] = (nibble >> 3) & 1
                planes[c, base + 1] = (nibble >> 2) & 1
                planes[c, base + 2] = (nibble >> 1) & 1
                planes[c, base + 3] = nibble & 1
            planes[c, NUM_INTERSECTIONS - 1] = int(hexstr[90])

        # to_move 行: "0" = 黑先, "1" = 白先
        # 颜色通道: 黑先 = 1.0, 白先 = 0.0 (与 gather_features 一致)
        to_move = pos[HISTORY_PLANES].decode("utf-8").strip()
        if to_move == "0":
            planes[INPUT_CHANNELS - 1, :] = 1

        probs = np.array(
            [float(x) for x in pos[HISTORY_PLANES + 1].split()],
            dtype=np.float32)
        winner = np.array([float(pos[HISTORY_PLANES + 2])], dtype=np.float32)

        # 随机对称变换做数据增强
        self.transform(planes, probs, random.randrange(8))
        return planes, probs, winner

    def transform(self, planes, probs, sym):
        rot = sym % 4
        flip = sym // 4
        for c in range(INPUT_CHANNELS):
            board = planes[c].reshape(19, 19)
            if flip:
                board = np.fliplr(board)
            board = np.rot90(board, rot)
            planes[c] = board.reshape(-1)
        p = probs[:NUM_INTERSECTIONS].reshape(19, 19)
        if flip:
            p = np.fliplr(p)
        p = np.rot90(p, rot)
        probs[:NUM_INTERSECTIONS] = p.reshape(-1)

    def _pack(self, batch):
        planes = np.concatenate([b[0] for b in batch]).reshape(-1)
        probs = np.concatenate([b[1] for b in batch]).reshape(-1)
        winner = np.concatenate([b[2] for b in batch]).reshape(-1)
        # planes 按 uint8、probs/winner 按 float32 打包成原始字节
        return planes.tobytes(), probs.tobytes(), winner.tobytes()
