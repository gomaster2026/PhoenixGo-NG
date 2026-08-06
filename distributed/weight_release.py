"""Weight shard publication tool for PhoenixGo-NG."""

import os, sys, json, gzip, hashlib

def publish_weights(weights_file, version='1.0'):
    with gzip.open(weights_file, 'rt') as f:
        content = f.read()
    sha = hashlib.sha256(content.encode()).hexdigest()
    meta = {
        'version': version,
        'sha256': sha,
        'size': len(content),
        'lines': content.count('\n'),
    }
    with open(f'{weights_file}.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Published weights v{version}')
    print(f'SHA256: {sha}')
    return meta

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: weight_release.py <weights_file.gz>')
        sys.exit(1)
    publish_weights(sys.argv[1])