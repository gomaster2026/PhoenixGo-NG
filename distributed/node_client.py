import os, sys, json, time, hashlib, requests

SERVER_URL = os.environ.get('PHOENIX_SERVER', 'https://phoenixgo-zero.cn')


def get_machine_id():
    mid = hashlib.md5(str(os.getpid()).encode()).hexdigest()[:12]
    return mid


def register():
    r = requests.post(f'{SERVER_URL}/api/register', json={'machine_id': get_machine_id()})
    return r.json()


def get_chunk():
    r = requests.get(f'{SERVER_URL}/api/chunk', params={'machine_id': get_machine_id()})
    return r.json()


def submit_result(chunk_id, result):
    r = requests.post(f'{SERVER_URL}/api/submit', json={'chunk_id': chunk_id, 'result': result})
    return r.json()


def run_client():
    print('Registering...')
    info = register()
    print(f'Registered: {info}')
    while True:
        print('Getting chunk...')
        chunk = get_chunk()
        if not chunk.get('chunk_id'):
            print('No chunks available, waiting...')
            time.sleep(60)
            continue
        print(f'Got chunk: {chunk["chunk_id"]}')
        # Process chunk (self-play)
        result = process_chunk(chunk)
        submit_result(chunk['chunk_id'], result)
        print('Chunk submitted!')


def process_chunk(chunk):
    # Run self-play games using leelaz
    # This is a placeholder - actual implementation uses GTP protocol
    return {'status': 'completed', 'games': []}


if __name__ == '__main__':
    run_client()