# 스트림릿 배포 시 구글 드라이브에서 데이터 로딩
from __future__ import annotations

import os
import json
import pickle
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


def _get_secret_folder_url(namespace: str) -> str | None:
    try:
        g = st.secrets.get("gdrive_urls", {})
    except Exception:
        return None

    g = _to_dict(g)
    ns = _to_dict(g.get(namespace, {}))

    # 1) [gdrive_urls.<namespace>] folder_url = "..."
    if "folder_url" in ns and str(ns["folder_url"]).strip():
        return str(ns["folder_url"]).strip()

    # 2) [gdrive_urls] folder_url = "..."
    if "folder_url" in g and str(g["folder_url"]).strip():
        return str(g["folder_url"]).strip()

    # 3) [gdrive_urls] <namespace>_folder_url = "..."
    flat_key = f"{namespace}_folder_url"
    if flat_key in g and str(g[flat_key]).strip():
        return str(g[flat_key]).strip()

    return None


def _get_env_folder_url(namespace: str) -> str | None:
    keys = [
        f"GDRIVE_{namespace.upper()}_FOLDER_URL",
        "GDRIVE_FOLDER_URL",
    ]
    for k in keys:
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _download_from_gdrive(url: str, target: Path) -> Tuple[bool, str]:
    if "drive.google.com/drive/folders/" in url:
        return False, (
            "파일 URL 자리에 폴더 링크가 입력되었습니다. "
            "파일별 키에는 file 링크를 넣거나, "
            "[gdrive_urls.<namespace>] folder_url 키를 사용하세요."
        )

    try:
        import gdown  # type: ignore
    except Exception:
        return False, "gdown 패키지가 필요합니다. `pip install gdown`"

    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 신버전 gdown: fuzzy 지원
        try:
            out = gdown.download(url=url, output=str(target), quiet=True, fuzzy=True)
        except TypeError:
            # 구버전 gdown: fuzzy 미지원
            out = gdown.download(url=url, output=str(target), quiet=True)
        if out is None:
            return False, "다운로드 실패(None 반환)"
        if not target.exists() or target.stat().st_size == 0:
            return False, "다운로드 후 파일이 비어있거나 생성되지 않았습니다."
        return True, "ok"
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _download_folder_from_gdrive(url: str, output_dir: Path) -> Tuple[bool, str]:
    try:
        import gdown  # type: ignore
    except Exception:
        return False, "gdown 패키지가 필요합니다. `pip install gdown`"

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = gdown.download_folder(url=url, output=str(output_dir), quiet=True)
        if files is None:
            return False, "폴더 다운로드 실패(None 반환)"
        return True, "ok"
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _is_valid_parquet(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 12:
            return False
        with path.open("rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            tail = f.read(4)
        if not (head == b"PAR1" and tail == b"PAR1"):
            return False

        # 실제 parquet 메타데이터까지 확인
        import pyarrow.parquet as pq  # type: ignore
        _ = pq.read_schema(path)
        return True
    except Exception:
        return False


def _looks_like_html(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open("rb") as f:
            head = f.read(1024).lower()
        return (
            b"<!doctype html" in head
            or b"<html" in head
            or b"<head" in head
            or b"<body" in head
        )
    except Exception:
        return False


def _is_valid_json(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def _is_valid_pickle_like(path: Path) -> bool:
    # .pkl / .joblib 공통 검증: HTML 잘못 저장 방지 + 경량 헤더 체크
    # 주의: 대용량 pickle을 실제로 load하면 배포 메모리 급증으로 프로세스가 죽을 수 있음.
    try:
        if _looks_like_html(path):
            return False
        if not path.exists() or path.stat().st_size < 2:
            return False
        with path.open("rb") as f:
            head = f.read(2)
        # 일반 pickle 프로토콜 시작 바이트(0x80) 기준 경량 판별
        # joblib도 pickle 기반이므로 대부분 동일하게 통과.
        return head[:1] == b"\x80"
    except Exception:
        return False


def _is_usable_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if _looks_like_html(path):
        return False
    if path.suffix.lower() == ".parquet":
        return _is_valid_parquet(path)
    if path.suffix.lower() == ".json":
        return _is_valid_json(path)
    if path.suffix.lower() in {".pkl", ".joblib"}:
        return _is_valid_pickle_like(path)
    return True


def ensure_page_files(
    namespace: str,
    required_files: Dict[str, Path],
    optional_files: Dict[str, Path] | None = None,
) -> Dict[str, Path]:
    """
    namespace 예시: hotspot / repair / weather
    required_files: {"logical_key": Path(...)}

    URL 조회 우선순위:
    1) 환경변수: GDRIVE_<NAMESPACE>_<KEY>_URL
    2) st.secrets[gdrive_urls][namespace][key]
    3) st.secrets[gdrive_urls]["<namespace>_<key>"]
    """
    missing: list[Tuple[str, Path]] = []
    optional_files = optional_files or {}

    for key, path in required_files.items():
        if _is_usable_file(path):
            continue

        url = _get_env_url(namespace, key) or _get_secret_url(namespace, key)
        if not url:
            missing.append((key, path))
            continue

        # 깨진 파일은 삭제 후 재다운로드
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

        ok, msg = _download_from_gdrive(url, path)
        if not ok or not _is_usable_file(path):
            st.error(f"[{namespace}] `{key}` 다운로드 실패: {msg}")
            st.stop()

    # optional 파일: URL이 있으면 내려받고, 없으면 건너뜀
    for key, path in optional_files.items():
        if _is_usable_file(path):
            continue

        url = _get_env_url(namespace, key) or _get_secret_url(namespace, key)
        if not url:
            continue

        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

        ok, msg = _download_from_gdrive(url, path)
        if not ok or not _is_usable_file(path):
            st.error(f"[{namespace}] `{key}` 다운로드 실패: {msg}")
            st.stop()

    # per-file URL이 없으면, folder_url로 한 번에 내려받기 시도
    if missing:
        folder_url = _get_env_folder_url(namespace) or _get_secret_folder_url(namespace)
        if folder_url:
            # required_files가 /.../processed_data/... 라고 가정하고
            # 상위 프로젝트 루트에 폴더를 풀어서 구조를 살립니다.
            sample_path = next(iter(required_files.values()))
            processed_dir = sample_path
            while processed_dir.name != "processed_data" and processed_dir.parent != processed_dir:
                processed_dir = processed_dir.parent
            output_root = processed_dir.parent if processed_dir.name == "processed_data" else sample_path.parent

            ok, msg = _download_folder_from_gdrive(folder_url, output_root)
            if not ok:
                st.error(f"[{namespace}] folder_url 다운로드 실패: {msg}")
                st.stop()

    # Re-check
    not_ready = [(k, p) for k, p in required_files.items() if not _is_usable_file(p)]
    if not_ready:
        lines = [f"- {k}: `{p}`" for k, p in not_ready]
        st.error(
            "필수 파일이 없습니다. 구글드라이브 URL을 설정해 주세요.\n\n"
            "설정 키 형식:\n"
            f"- 환경변수: `GDRIVE_{namespace.upper()}_<KEY>_URL`\n"
            f"- 환경변수(폴더): `GDRIVE_{namespace.upper()}_FOLDER_URL` 또는 `GDRIVE_FOLDER_URL`\n"
            f"- secrets: `[gdrive_urls.{namespace}] <key> = \"...\"`\n\n"
            f"- secrets(폴더): `[gdrive_urls.{namespace}] folder_url = \"...\"` 또는 `[gdrive_urls] folder_url = \"...\"`\n\n"
            "누락 파일:\n" + "\n".join(lines)
        )
        st.stop()

    return required_files


def render_data_help(namespace: str, keys: Iterable[str]) -> None:
    key_lines = "\n".join([f"- `{k}`" for k in keys])
    st.info(
        f"배포 시 `{namespace}` 페이지는 아래 키 URL이 필요합니다(환경변수 또는 st.secrets).\n\n{key_lines}"
    )
