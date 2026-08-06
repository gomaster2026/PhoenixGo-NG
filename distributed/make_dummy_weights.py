"""Generate initial dummy weights for PhoenixGo-NG."""

import gzip
import random

def make_weights(blocks=20, channels=256, filename='dummy_weights.txt.gz'):
    lines = []
    # Input conv
    for _ in range(channels * 18 * 3 * 3):
        lines.append(f'{random.gauss(0, 0.1):.6e}')
    lines.append('0.000000e+00')
    # Trunk BN
    for _ in range(channels):
        lines.append('1.000000e+00')
    for _ in range(channels):
        lines.append('0.000000e+00')
    for _ in range(channels):
        lines.append('0.000000e+00')
    for _ in range(channels):
        lines.append('1.000000e+00')
    # Residual blocks
    for _ in range(blocks):
        for __ in range(2):
            for ___ in range(channels * channels * 3 * 3):
                lines.append(f'{random.gauss(0, 0.1):.6e}')
            for ___ in range(channels):
                lines.append('0.000000e+00')
            for ___ in range(channels):
                lines.append('1.000000e+00')
            for ___ in range(channels):
                lines.append('0.000000e+00')
            for ___ in range(channels):
                lines.append('0.000000e+00')
            for ___ in range(channels):
                lines.append('1.000000e+00')
    # Policy head
    for _ in range(2 * channels):
        for __ in range(3 * 3):
            lines.append(f'{random.gauss(0, 0.1):.6e}')
    for _ in range(2):
        for __ in range(channels):
            lines.append('0.000000e+00')
    for _ in range(2):
        for __ in range(channels):
            lines.append('1.000000e+00')
    for _ in range(2):
        for __ in range(channels):
            lines.append('0.000000e+00')
    for _ in range(2):
        for __ in range(channels):
            lines.append('0.000000e+00')
    for _ in range(2):
        for __ in range(channels):
            lines.append('1.000000e+00')
    for _ in range(362 * channels):
        lines.append(f'{random.gauss(0, 0.01):.6e}')
    for _ in range(362):
        lines.append('0.000000e+00')
    # Value head
    for _ in range(channels):
        for __ in range(3 * 3):
            lines.append(f'{random.gauss(0, 0.1):.6e}')
    for _ in range(channels):
        lines.append('0.000000e+00')
    for _ in range(channels):
        lines.append('1.000000e+00')
    for _ in range(channels):
        lines.append('0.000000e+00')
    for _ in range(channels):
        lines.append('0.000000e+00')
    for _ in range(channels):
        lines.append('1.000000e+00')
    for _ in range(256 * channels):
        lines.append(f'{random.gauss(0, 0.01):.6e}')
    for _ in range(256):
        lines.append('0.000000e+00')
    for _ in range(256):
        lines.append('1.000000e+00')
    for _ in range(256):
        lines.append('0.000000e+00')
    for _ in range(256):
        lines.append('0.000000e+00')
    for _ in range(256):
        lines.append('1.000000e+00')
    for _ in range(1 * 256):
        lines.append(f'{random.gauss(0, 0.01):.6e}')
    for _ in range(1):
        lines.append('0.000000e+00')
    with gzip.open(filename, 'wt') as f:
        for line in lines:
            f.write(line + '\n')
    print(f'Written {len(lines)} lines to {filename}')

if __name__ == '__main__':
    make_weights()