"""
Chunk data loader for distributed training.
Handles compressed training data chunks.
"""

import gzip
import os
import struct

class ChunkParser:
    def __init__(self, chunk_files):
        self.chunk_files = chunk_files
        self.current_file = 0
    
    def __iter__(self):
        for filepath in self.chunk_files:
            with gzip.open(filepath, 'rt') as f:
                while True:
                    lines = []
                    for _ in range(19):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.strip())
                    if len(lines) < 19:
                        break
                    yield lines
    
    def get_chunks(self):
        return self.chunk_files

def find_chunks(directory):
    """Find all .gz chunk files in directory."""
    chunks = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith('.gz'):
                chunks.append(os.path.join(root, f))
    return chunks