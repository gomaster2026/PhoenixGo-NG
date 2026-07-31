#!/usr/bin/env python3
"""
权重发布工具 — 把训练出的新权重存到 Gitee/GitHub Release

为什么权重不存 Worker？
  D1 单行上限 2MB、KV 上限 25MB，放不下 ~200MB 的权重文件。
  Gitee/GitHub Release 免费、无需绑卡、国内可用。
  Gitee 单附件上限 100MB → 权重自动分片上传（part0/1/2 + meta 清单），下载端自动合并。

用法:
  # 上传权重（首次上传初始权重 + 每次训练完成后）
  python weight_release.py upload --platform gitee \\
      --owner 你的用户名 --repo 权重仓库名 --token Gitee私人令牌 \\
      --weights phoenixgo-v1.txt.gz

  # 下载最新权重（贡献者节点 / Kaggle 训练脚本自动调用）
  python weight_release.py download --platform gitee \\
      --owner 用户名 --repo 权重仓库名 --dest current.txt.gz

支持平台:
  gitee  国内快，权重自动分片（单附件 ≤100MB）
  github 单文件 ≤2GB 无需分片，但国内访问可能较慢

Gitee 私人令牌: gitee.com → 设置 → 私人令牌，勾选 projects/releases 权限
GitHub Token:  Personal access tokens (classic)，勾选 repo 权限
"""

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RELEASE_TAG = "lz-latest"          # 固定 tag，便于下载方拿到"最新"
WEIGHT_BASE = "current.txt.gz"     # 权重固定文件名（可加日期信息在 release 描述里）
GITEE_API = "https://gitee.com/api/v5"
GITHUB_API = "https://api.github.com"
PART_SIZE = 90 * 1024 * 1024       # Gitee 附件 100MB 上限，留余量用 90MB/片


def log(msg):
    print(f"[weight] {msg}", flush=True)


def split_file(path, part_size=PART_SIZE):
    """把权重切成 part 字节数一致的字节列表"""
    parts = []
    with open(path, "rb") as f:
        while True:
            data = f.read(part_size)
            if not data:
                break
            parts.append(data)
    if not parts:
        parts.append(b"")
    return parts


def download_url(url, dest_path, timeout=600, retries=4):
    """下载 URL 到文件（自动重试），返回 bool"""
    dest_path = Path(dest_path)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log(f"404: {url}")
                return False
            last_err = f"HTTP {e.code}"
        except Exception as e:
            last_err = str(e)
        log(f"下载失败 ({last_err})，重试 {attempt}/{retries}...")
        time.sleep(3 * attempt)
    return False


# ---------------- 下载（公开仓库，无需 token） ----------------

def part_names(parts_count):
    return [f"{WEIGHT_BASE}.part{i}" for i in range(parts_count)]


def download_gitee(owner, repo, dest_path):
    """从 Gitee 下载最新权重（读 meta → 逐个下分片 → 合并）"""
    base = f"https://gitee.com/{owner}/{repo}/releases/download/{RELEASE_TAG}"
    meta_url = f"{base}/{WEIGHT_BASE}.meta"
    meta_tmp = Path(dest_path).with_suffix(".meta.tmp")
    log(f"读取分片清单: {meta_url}")
    if not download_url(meta_url, meta_tmp, timeout=120, retries=3):
        log("错误: 找不到分片清单，请先上传权重")
        return False
    try:
        parts_count = int(meta_tmp.read_text().strip())
    except ValueError:
        log("错误: 分片清单格式不对")
        return False
    if parts_count <= 0 or parts_count > 100:
        log(f"错误: 分片数量异常 ({parts_count})")
        return False

    log(f"共 {parts_count} 个分片，开始下载...")
    with open(dest_path, "wb") as f:
        for i, name in enumerate(part_names(parts_count)):
            url = f"{base}/{name}"
            tmp = Path(dest_path).with_suffix(f".part{i}.tmp")
            if not download_url(url, tmp, timeout=600, retries=4):
                log(f"错误: 分片 {name} 下载失败")
                return False
            with open(tmp, "rb") as p:
                while True:
                    chunk = p.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.unlink(missing_ok=True)
            log(f"  分片 {i + 1}/{parts_count} 完成")
    meta_tmp.unlink(missing_ok=True)
    log(f"权重合并完成: {dest_path} ({Path(dest_path).stat().st_size / 1e6:.1f} MB)")
    return True


def download_github(owner, repo, dest_path):
    url = f"https://github.com/{owner}/{repo}/releases/latest/download/{WEIGHT_BASE}"
    log(f"从 GitHub 下载: {url}")
    if not download_url(url, dest_path, timeout=600, retries=4):
        return False
    log(f"权重下载完成: {dest_path} ({Path(dest_path).stat().st_size / 1e6:.1f} MB)")
    return True


# ---------------- 上传（需要 token） ----------------

def _gitee_request(method, path, token=None, **params):
    """Gitee API 请求（用 requests，上传/管理需要）"""
    import requests
    url = GITEE_API + path
    params = {k: v for k, v in params.items() if v is not None}
    if token:
        params["access_token"] = token
    return requests.request(method, url, params=params, timeout=60)


def upload_gitee(owner, repo, token, weight_path, retries=3):
    import requests
    weight_path = Path(weight_path)
    base = f"/repos/{owner}/{repo}"
    weight_size = weight_path.stat().st_size
    log(f"上传权重到 Gitee: {weight_path} ({weight_size / 1e6:.1f} MB)")

    # 1. 查默认分支（建 release 需要 target_commitish）
    branch = "master"
    try:
        r = _gitee_request("GET", base, token)
        r.raise_for_status()
        branch = r.json().get("default_branch", "master")
    except Exception as e:
        log(f"获取仓库信息失败: {e}，使用 master")

    # 2. 删除旧的 lz-latest release（保持固定 tag）
    try:
        r = _gitee_request("GET", f"{base}/releases/tags/{RELEASE_TAG}", token)
        if r.status_code == 200:
            rel_id = r.json().get("id")
            if rel_id:
                _gitee_request("DELETE", f"{base}/releases/{rel_id}", token)
                log(f"已删除旧 release #{rel_id}")
    except Exception as e:
        log(f"查询旧 release 失败: {e}")

    # 3. 创建新 release
    body = (f"Leela Zero 权重自动发布 {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"文件: {WEIGHT_BASE} ({weight_size / 1e6:.1f} MB)\n"
            f"平台: {('Gitee 分片 ' + str((weight_size + PART_SIZE - 1) // PART_SIZE) + ' 片') if weight_size > PART_SIZE else 'Gitee 单文件'}")
    r = _gitee_request("POST", f"{base}/releases", token,
                       tag_name=RELEASE_TAG, target_commitish=branch,
                       name=RELEASE_TAG, body=body)
    if r.status_code not in (200, 201):
        log(f"创建 release 失败: HTTP {r.status_code} {r.text[:300]}")
        return False
    release_id = r.json().get("id")
    log(f"已创建 release #{release_id}")

    # 4. 分片并上传
    parts = split_file(weight_path)
    need_split = weight_size > PART_SIZE
    if need_split:
        meta = str(len(parts))
        files = [(f"{WEIGHT_BASE}.meta", meta.encode())] + [
            (f"{WEIGHT_BASE}.part{i}", data) for i, data in enumerate(parts)
        ]
    else:
        files = [(WEIGHT_BASE, parts[0])]

    for attempt in range(1, retries + 1):
        ok = True
        for name, data in files:
            url = f"{GITEE_API}{base}/releases/{release_id}/upload?name={name}&access_token={token}"
            try:
                r = requests.post(
                    url,
                    files={"file": (name, data, "application/octet-stream")},
                    timeout=600,
                )
                if r.status_code not in (200, 201):
                    log(f"上传 {name} 失败: HTTP {r.status_code} {r.text[:200]}")
                    ok = False
                    break
                log(f"  上传完成: {name} ({len(data) / 1e6:.1f} MB)")
            except Exception as e:
                log(f"上传 {name} 异常: {e}")
                ok = False
                break
        if ok:
            log("全部上传成功!")
            return True
        log(f"重试 {attempt}/{retries}...")
        time.sleep(5)
    return False


def _github_headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def upload_github(owner, repo, token, weight_path, retries=3):
    import requests
    weight_path = Path(weight_path)
    api = f"{GITHUB_API}/repos/{owner}/{repo}"
    headers = _github_headers(token)
    weight_size = weight_path.stat().st_size
    log(f"上传权重到 GitHub: {weight_path} ({weight_size / 1e6:.1f} MB)")

    # 1. 删旧 release
    r = requests.get(f"{api}/releases/tags/{RELEASE_TAG}", headers=headers, timeout=60)
    if r.status_code == 200:
        rel_id = r.json().get("id")
        if rel_id:
            requests.delete(f"{api}/releases/{rel_id}", headers=headers, timeout=60)
            log(f"已删除旧 release #{rel_id}")

    # 2. 建新 release
    r = requests.post(
        f"{api}/releases", headers=headers, timeout=60,
        json={
            "tag_name": RELEASE_TAG,
            "name": RELEASE_TAG,
            "body": f"Leela Zero 权重自动发布 {time.strftime('%Y-%m-%d %H:%M')}",
        },
    )
    if r.status_code not in (200, 201):
        log(f"创建 release 失败: HTTP {r.status_code} {r.text[:300]}")
        return False
    release_id = r.json().get("id")
    log(f"已创建 release #{release_id}")

    # 3. 上传单个 asset（GitHub 单文件上限 2GB）
    upload_url = f"https://uploads.github.com/repos/{owner}/{repo}/releases/{release_id}/assets?name={WEIGHT_BASE}"
    upload_headers = {**headers, "Content-Type": "application/gzip"}
    for attempt in range(1, retries + 1):
        try:
            with open(weight_path, "rb") as f:
                r = requests.post(
                    upload_url, headers=upload_headers,
                    data=f, timeout=1800,
                )
            if r.status_code in (200, 201):
                log(f"权重上传成功: {WEIGHT_BASE}")
                return True
            log(f"上传失败: HTTP {r.status_code} {r.text[:200]}")
        except Exception as e:
            log(f"上传异常: {e}")
        log(f"重试 {attempt}/{retries}...")
        time.sleep(5)
    return False


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Leela Zero 权重发布工具（Gitee/GitHub Release）")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="上传权重")
    up.add_argument("--platform", choices=["gitee", "github"], default="gitee")
    up.add_argument("--owner", required=True, help="用户名/组织名")
    up.add_argument("--repo", required=True, help="仓库名")
    up.add_argument("--token", required=True, help="私人令牌")
    up.add_argument("--weights", required=True, help="权重 .txt.gz 文件路径")

    dl = sub.add_parser("download", help="下载最新权重")
    dl.add_argument("--platform", choices=["gitee", "github"], default="gitee")
    dl.add_argument("--owner", required=True, help="用户名/组织名")
    dl.add_argument("--repo", required=True, help="仓库名")
    dl.add_argument("--dest", default=WEIGHT_BASE, help="保存路径")

    args = parser.parse_args()

    if args.command == "upload":
        if args.platform == "gitee":
            ok = upload_gitee(args.owner, args.repo, args.token, args.weights)
        else:
            ok = upload_github(args.owner, args.repo, args.token, args.weights)
    else:
        if args.platform == "gitee":
            ok = download_gitee(args.owner, args.repo, args.dest)
        else:
            ok = download_github(args.owner, args.repo, args.dest)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
