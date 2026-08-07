"""
Training data parser for PhoenixGo-NG.
Parses Leela Zero V1 format training data.
"""

import sys
import gzip
import struct

def parse_data(input_file, output_file):
    """Parse training data from gzipped file."""
    count = 0
    with gzip.open(input_file, 'rt') as fin:
        with open(output_file, 'w') as fout:
            # Read in blocks of 19 lines per position
            while True:
                lines = []
                for _ in range(19):
                    line = fin.readline()
                    if not line:
                        break
                    lines.append(line.strip())
                if len(lines) < 19:
                    break
                count += 1
                fout.write('\n'.join(lines) + '\n')
                if count % 1000 == 0:
                    print(f'Processed {count} positions')
    print(f'Total: {count} positions written to {output_file}')

def main():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <num_blocks> <num_filters> <input_file>')
        sys.exit(1)
    num_blocks = int(sys.argv[1])
    num_filters = int(sys.argv[2])
    input_file = sys.argv[3]
    print(f'Parsing with {num_blocks} blocks, {num_filters} filters')
    output_file = input_file + '.parsed'
    parse_data(input_file, output_file)

if __name__ == '__main__':
    main()