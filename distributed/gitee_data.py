#!/usr/bin/env python3
"""
Gitee 训练数据仓库 — 存储/下载训练 chunk

数据仓库是一个公开的 Gitee 仓库（如 lz-data），chunk 文件放在 chunks/ 目录。
用 Gitee API 读写：上传/删除需要私人令牌，下载对公开仓库无需令牌。

Gitee 免费仓库限制: 单文件 ≤100MB，仓库总容量 ≤500MB。
每个 chunk 约 0.5MB，容量足够；Kaggle 训练完成后会用组织者令牌删除已用 chunk，
释放空间给下一轮贡献者上传。

私人令牌: gitee.com → 设置 → 私人令牌 → 勾选 projects 权限
"""

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GITEE_API = "https://gitee.com/api/v5"
CHUNK_DIR = "chunks"
BRANCH = "master"


def _log(msg):
    print(f"[gitee] {msg}", flush=True)


def _q(seg):
    """URL 编码单个路径段（中文用户名/仓库名/文件名）"""
    return urllib.parse.quote(str(seg), safe="-_.")


def _api(owner, repo, path, method="GET", params=None, token=None, timeout=120):
    """Gitee API 请求（仅 urllib，无第三方依赖），返回 (status, json)"""
    url = f"{GITEE_API}/repos/{_q(owner)}/{_q(repo)}/contents/{path}"
    payload = dict(params or {})
    if token:
        payload["access_token"] = token
    if method in ("POST", "PUT", "DELETE"):
        data = urllib.parse.urlencode(payload).encode()
    else:
        qs = urllib.parse.urlencode(payload)
        if qs:
            url += "?" + qs
        data = None
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except Exception:
                return resp.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


def upload_chunk(owner, repo, token, file_path, node_name="", retries=3):
    """
    上传一个 chunk 到数据仓库 chunks/ 目录。
    文件名自动加时间戳+随机数防止冲突。返回 (remote_name, ok)
    """
    file_path = Path(file_path)
    raw = file_path.read_bytes()
    if len(raw) > 100 * 1024 * 1024:
        _log(f"错误: {file_path.name} 超过 100MB，Gitee 单文件上限")
        return None, False

    base_name = file_path.name.replace(" ", "_")
    remote_name = f"{node_name or 'anon'}_{int(time.time())}_{base_name}"
    # 只保留安全字符
    remote_name = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in remote_name
    )

    b64 = base64.b64encode(raw).decode("ascii")
    for attempt in range(1, retries + 1):
        status, body = _api(
            owner, repo, f"{CHUNK_DIR}/{remote_name}",
            method="POST",
            params={
                "content": b64,
                "message": f"upload training chunk {remote_name}",
                "branch": BRANCH,
            },
            token=token,
            timeout=180,
        )
        if status in (200, 201):
            _log(f"上传成功: {remote_name} ({len(raw) / 1e6:.2f} MB)")
            return remote_name, True
        if status in (400, 401, 403, 404):
            _log(f"上传失败: HTTP {status} {body}")
            return remote_name, False
        _log(f"上传失败 (HTTP {status})，重试 {attempt}/{retries}...")
        time.sleep(3 * attempt)
    return remote_name, False


def list_chunks(owner, repo, token="", branch=BRANCH):
    """
    列出数据仓库里所有 chunk，返回 [{name, path, sha, size}]。
    公开仓库不需要 token。
    """
    status, body = _api(owner, repo, CHUNK_DIR, token=token)
    if status == 404:
        _log("chunks/ 目录还不存在（还没有人上传）")
        return []
    if status != 200:
        _log(f"列出 chunk 失败: HTTP {status} {body}")
        return []
    files = []
    for item in body:
        if isinstance(item, dict) and item.get("type") == "file":
            files.append({
                "name": item.get("name"),
                "path": item.get("path"),
                "sha": item.get("sha"),
                "size": item.get("size") or 0,
            })
    return files


def _api_raw(owner, repo, path, token="", timeout=300):
    """Gitee API raw 端点下载二进制内容，返回 bytes 或 None"""
    url = f"{GITEE_API}/repos/{_q(owner)}/{_q(repo)}/raw/{path}"
    if token:
        url += "?" + urllib.parse.urlencode({"access_token": token})
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def download_chunk(owner, repo, name, dest_path, branch=BRANCH, token="", retries=3):
    """
    从数据仓库下载一个 chunk。
    私密仓库必须传 token（走 API raw 端点），公开仓库可不传。
    """
    dest_path = Path(dest_path)
    for attempt in range(1, retries + 1):
        try:
            if token:
                data = _api_raw(
                    owner, repo,
                    f"{CHUNK_DIR}/{urllib.parse.quote(name)}",
                    token=token,
                )
            else:
                url = f"https://gitee.com/{_q(owner)}/{_q(repo)}/raw/{branch}/{CHUNK_DIR}/{urllib.parse.quote(name)}"
                with urllib.request.urlopen(url, timeout=300) as resp:
                    data = resp.read()
            dest_path.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                _log(f"404: {name} 不存在")
                return False
            _log(f"下载 {name} 失败 (HTTP {e.code})，重试 {attempt}/{retries}...")
        except Exception as e:
            _log(f"下载 {name} 失败: {e}，重试 {attempt}/{retries}...")
        time.sleep(3 * attempt)
    return False


def delete_chunk(owner, repo, token, name, sha, retries=3):
    """从数据仓库删除一个 chunk（训练完成后清理空间），返回 bool"""
    for attempt in range(1, retries + 1):
        status, body = _api(
            owner, repo, f"{CHUNK_DIR}/{name}",
            method="DELETE",
            params={
                "message": f"delete {name}",
                "sha": sha,
                "branch": BRANCH,
            },
            token=token,
        )
        if status == 200:
            return True
        if status in (400, 401, 403, 404):
            _log(f"删除 {name} 失败: HTTP {status} {body}")
            return False
        _log(f"删除 {name} 失败 (HTTP {status})，重试 {attempt}/{retries}...")
        time.sleep(3 * attempt)
    return False
