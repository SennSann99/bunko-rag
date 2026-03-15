"""
Ollama API クライアント
LLM の生成と埋め込みベクトル取得を担当
"""

import os
import json
import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")


def embed(texts: list[str]) -> list[list[float]]:
    """
    テキストのリストを埋め込みベクトルに変換する。

    Ollama の /api/embed エンドポイントを使用。
    BGE-M3 は 100+ 言語対応で、日本語のセマンティック検索に最適。
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def generate(prompt: str, system: str = "", stream: bool = False) -> str:
    """
    LLM にプロンプトを送って回答を生成する。

    Qwen 2.5 は日本語を含む 29 言語をサポート。
    128K コンテキストで、RAG の長いプロンプトも安定。
    """
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 1024,
        },
    }
    if system:
        payload["system"] = system

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def is_available() -> bool:
    """Ollama サーバーが利用可能か確認する"""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def list_models() -> list[str]:
    """利用可能なモデル一覧を取得する"""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []
