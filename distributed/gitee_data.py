"""Gitee data repository API for storing training data."""

import os, json, requests

GITEE_API = 'https://gitee.com/api/v5'

def get_file(owner, repo, path, access_token):
    url = f'{GITEE_API}/repos/{owner}/{repo}/contents/{path}'
    r = requests.get(url, params={'access_token': access_token})
    return r.json()

def create_file(owner, repo, path, content, message, access_token):
    url = f'{GITEE_API}/repos/{owner}/{repo}/contents/{path}'
    r = requests.post(url, json={
        'access_token': access_token,
        'content': content,
        'message': message
    })
    return r.json()