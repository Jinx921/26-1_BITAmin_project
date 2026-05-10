# 스트림릿 배포 시 구글 드라이브에서 데이터 로딩
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import streamlit as st


def _to_dict(obj):
    try:
        return dict(obj)
    except Exception:
        return {}


def _get_secret_url(namespace: str, key: str) -> str | None:
    # Preferred format:
    # [gdrive_urls.<namespace>]
    # <key> = "https://drive.google.com/..."
    try:
        g = st.secrets.get("gdrive_urls", {})
    except Exception:
        return None

    g = _to_dict(g)

    ns = _to_dict(g.get(namespace, {}))
    if key in ns:
        return str(ns[key])

    flat_key = f"{namespace}_{key}"
    if flat_key in g:
        return str(g[flat_key])

    return None


def _get_env_url(namespace: str, key: str) -> str | None:
    env_key = f"GDRIVE_{namespace.upper()}_{key.upper()}_URL"
    v = os.getenv(env_key)
    return v.strip() if isinstance(v, str) and v.strip() else None


def _download_from_gdrive(url: str, target: Path) -> Tuple[bool, str]:
    try:
        import gdown  # type: ignore
    except Exception:
        return False, "gdown 패키지가 필요합니다. `pip install gdown`"

    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        # fuzzy=True: 다양한 구글드라이브 URL 형식 자동 처리
        out = gdown.download(url=url, output=str(target), quiet=True, fuzzy=True)
        if out is None:
            return False, "다운로드 실패(None 반환)"
        if not target.exists() or target.stat().st_size == 0:
            return False, "다운로드 후 파일이 비어있거나 생성되지 않았습니다."
        return True, "ok"
    except Exception as e:  # pragma: no cover
        return False, str(e)


def ensure_page_files(namespace: str, required_files: Dict[str, Path]) -> Dict[str, Path]:
    """
    namespace 예시: hotspot / repair / weather
    required_files: {"logical_key": Path(...)}

    URL 조회 우선순위:
    1) 환경변수: GDRIVE_<NAMESPACE>_<KEY>_URL
    2) st.secrets[gdrive_urls][namespace][key]
    3) st.secrets[gdrive_urls]["<namespace>_<key>"]
    """
    missing: list[Tuple[str, Path]] = []

    for key, path in required_files.items():
        if path.exists():
            continue

        url = _get_env_url(namespace, key) or _get_secret_url(namespace, key)
        if not url:
            missing.append((key, path))
            continue

        ok, msg = _download_from_gdrive(url, path)
        if not ok:
            st.error(f"[{namespace}] `{key}` 다운로드 실패: {msg}")
            st.stop()

    # Re-check
    not_ready = [(k, p) for k, p in required_files.items() if not p.exists()]
    if not_ready:
        lines = [f"- {k}: `{p}`" for k, p in not_ready]
        st.error(
            "필수 파일이 없습니다. 구글드라이브 URL을 설정해 주세요.\n\n"
            "설정 키 형식:\n"
            f"- 환경변수: `GDRIVE_{namespace.upper()}_<KEY>_URL`\n"
            f"- secrets: `[gdrive_urls.{namespace}] <key> = \"...\"`\n\n"
            "누락 파일:\n" + "\n".join(lines)
        )
        st.stop()

    return required_files


def render_data_help(namespace: str, keys: Iterable[str]) -> None:
    key_lines = "\n".join([f"- `{k}`" for k in keys])
    st.info(
        f"배포 시 `{namespace}` 페이지는 아래 키 URL이 필요합니다(환경변수 또는 st.secrets).\n\n{key_lines}"
    )
