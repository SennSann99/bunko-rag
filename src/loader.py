"""
青空文庫テキストのローダーとチャンク分割
"""

import re
import os
import glob


def load_aozora_text(filepath: str) -> dict:
    """
    青空文庫形式のテキストファイルを読み込み、
    メタデータと本文に分割する。

    青空文庫のテキスト構造:
      1行目: タイトル
      2行目: 著者名
      3行目以降: 本文
      末尾: 底本情報
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.strip().split("\n")
    title = lines[0].strip()
    author = lines[1].strip()
    body = "\n".join(lines[2:]).strip()

    # 青空文庫のルビ記法を除去: 《...》
    body = re.sub(r"《[^》]+》", "", body)
    # 注記を除去: ［＃...］
    body = re.sub(r"［＃[^］]+］", "", body)
    # 末尾の底本情報を除去
    body = re.sub(r"\n底本.*$", "", body, flags=re.DOTALL)
    body = re.sub(r"（[大昭]正.*$", "", body, flags=re.DOTALL)

    return {"title": title, "author": author, "body": body.strip(), "path": filepath}


def load_all_texts(directory: str = "texts") -> list[dict]:
    """texts/ ディレクトリ内の全テキストを読み込む"""
    docs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        try:
            doc = load_aozora_text(path)
            docs.append(doc)
            print(f"  📖 {doc['title']} ({doc['author']}) — {len(doc['body'])}文字")
        except Exception as e:
            print(f"  ⚠️  {path}: {e}")
    return docs


# ── チャンク分割 ──────────────────────────────────


def chunk_by_section(text: str, doc_meta: dict = None) -> list[dict]:
    """
    漢数字のセクション見出しで分割する。
    「蜘蛛の糸」のような章構成の作品に最適。
    """
    parts = re.split(r"\n(一|二|三|四|五|六|七|八|九|十)\n", text)
    chunks = []
    section = "序"

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in "一二三四五六七八九十":
            section = part
            continue
        chunks.append({
            "id": len(chunks),
            "text": part,
            "section": section,
            "length": len(part),
            **(doc_meta or {}),
        })
    return chunks


def chunk_by_paragraph(text: str, max_size: int = 400, overlap: int = 80,
                       doc_meta: dict = None) -> list[dict]:
    """
    段落ベースの固定サイズ分割（オーバーラップ付き）。

    段落の境界を尊重しつつ、max_size 以内に収める。
    overlap で前のチャンクの末尾を次のチャンクに引き継ぐ。
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_size:
            current += ("\n" + para if current else para)
        else:
            if current:
                chunks.append({
                    "id": len(chunks),
                    "text": current.strip(),
                    "length": len(current.strip()),
                    **(doc_meta or {}),
                })
            if overlap > 0 and current:
                current = current[-overlap:] + "\n" + para
            else:
                current = para

    if current.strip():
        chunks.append({
            "id": len(chunks),
            "text": current.strip(),
            "length": len(current.strip()),
            **(doc_meta or {}),
        })

    return chunks


def chunk_document(doc: dict, strategy: str = "paragraph",
                   max_size: int = 400, overlap: int = 80) -> list[dict]:
    """
    ドキュメントをチャンクに分割する。

    strategy:
      - "section": 章で分割（短編小説向き）
      - "paragraph": 段落ベースの固定サイズ分割
    """
    meta = {"title": doc["title"], "author": doc["author"]}

    if strategy == "section":
        return chunk_by_section(doc["body"], meta)
    else:
        return chunk_by_paragraph(doc["body"], max_size, overlap, meta)
