"""
RAG パイプライン: Retrieval-Augmented Generation

青空文庫の日本文学作品に対する質問応答システム。
Ollama (Qwen 2.5 + BGE-M3) を使用。
"""

import os
import sys

# src/ ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(__file__))

from loader import load_all_texts, chunk_document
from vectorstore import VectorStore
from ollama_client import generate, is_available, list_models, LLM_MODEL, EMBED_MODEL


SYSTEM_PROMPT = """あなたは日本文学の専門家です。
与えられたテキストの抜粋（コンテキスト）を参考にして、
ユーザーの質問に正確かつ簡潔に日本語で回答してください。

ルール:
- コンテキストに含まれる情報のみに基づいて回答すること
- コンテキストに答えがない場合は「テキストからは分かりません」と回答すること
- 原文を引用する場合は「」で囲むこと
- 回答は3〜5文程度に簡潔にまとめること"""


class BunkoRAG:
    """青空文庫 RAG システム"""

    def __init__(self, text_dir: str = "texts", chunk_strategy: str = "paragraph"):
        self.store = VectorStore()
        self.use_llm = False
        self.chunk_strategy = chunk_strategy

        # Ollama の接続を確認
        print("\n🔌 Ollama サーバーを確認中...")
        if is_available():
            models = list_models()
            print(f"  ✅ 接続成功 — モデル: {', '.join(models) or '(なし)'}")

            has_llm = any(LLM_MODEL.split(":")[0] in m for m in models)
            has_embed = any(EMBED_MODEL.split(":")[0] in m for m in models)

            if has_llm:
                self.use_llm = True
                print(f"  🤖 LLM: {LLM_MODEL}")
            else:
                print(f"  ⚠️  LLM モデル {LLM_MODEL} 未検出 → 検索のみモード")
                print(f"     docker exec bunko-ollama ollama pull {LLM_MODEL}")

            use_neural = has_embed
            if has_embed:
                print(f"  🧠 Embedding: {EMBED_MODEL}")
            else:
                print(f"  ⚠️  Embedding モデル {EMBED_MODEL} 未検出 → TF-IDF 使用")
                print(f"     docker exec bunko-ollama ollama pull {EMBED_MODEL}")
        else:
            print("  ⚠️  Ollama に接続できません → オフラインモード (TF-IDF)")
            print("     docker compose up ollama -d")
            use_neural = False

        # テキストの読み込みとインデックス構築
        print(f"\n📚 テキストを読み込み中 ({text_dir}/)...")
        docs = load_all_texts(text_dir)

        if not docs:
            print("  ❌ テキストが見つかりません")
            print("     texts/ ディレクトリに青空文庫のテキストを配置してください")
            sys.exit(1)

        print(f"\n✂️  チャンク分割中 (strategy={chunk_strategy})...")
        all_chunks = []
        for doc in docs:
            chunks = chunk_document(doc, strategy=chunk_strategy)
            all_chunks.extend(chunks)
            print(f"  {doc['title']}: {len(chunks)} チャンク")

        print(f"\n📐 ベクトルストア構築中...")
        self.store.add_chunks(all_chunks, use_neural=use_neural)

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        RAG パイプラインを実行する。

        1. Retrieval: 質問に関連するチャンクを検索
        2. Augmentation: チャンクをコンテキストとしてプロンプトに組み込む
        3. Generation: LLM が回答を生成
        """
        print(f"\n{'─' * 56}")
        print(f"❓ {question}")
        print(f"{'─' * 56}")

        # ── 1. Retrieval ──
        results = self.store.search(question, top_k=top_k)

        print(f"\n📚 検索結果 (top-{top_k}, mode={self.store.mode}):")
        for i, r in enumerate(results, 1):
            c = r["chunk"]
            src = f"{c.get('title', '?')} / {c.get('section', '?')}"
            preview = c["text"][:40].replace("\n", " ")
            print(f"  [{i}] {r['score']:.3f} — {src}")
            print(f"      \"{preview}...\"")

        # ── 2. Augmentation ──
        context_parts = []
        for i, r in enumerate(results, 1):
            c = r["chunk"]
            header = f"【{c.get('title', '?')} / 第{c.get('section', '?')}章】"
            context_parts.append(f"{header}\n{c['text']}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""以下は日本文学作品からの抜粋です:

{context}

---

質問: {question}

上記のコンテキストに基づいて回答してください。"""

        # ── 3. Generation ──
        if self.use_llm:
            print(f"\n💬 {LLM_MODEL} で回答生成中...")
            answer = generate(prompt, system=SYSTEM_PROMPT)
        else:
            answer = (
                "[オフラインモード — Ollama に接続すると AI 回答が生成されます]\n\n"
                f"検索結果のコンテキスト:\n{context[:500]}..."
            )

        print(f"\n💬 回答:\n{answer}")

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "title": r["chunk"].get("title"),
                    "section": r["chunk"].get("section"),
                    "score": r["score"],
                    "text": r["chunk"]["text"][:100],
                }
                for r in results
            ],
        }

    def interactive(self):
        """対話型の質問応答ループ"""
        print(f"""
╔══════════════════════════════════════════════════════╗
║   文庫 RAG — 日本文学 質問応答システム              ║
║   Powered by Ollama + {LLM_MODEL:<26s}  ║
╚══════════════════════════════════════════════════════╝

  チャンク数: {self.store.size}
  検索モード: {self.store.mode}
  LLM 生成:  {'有効' if self.use_llm else '無効 (検索のみ)'}

  質問を入力してください (終了: quit / exit)
""")

        while True:
            try:
                q = input("🔍 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 終了します")
                break

            if not q:
                continue
            if q.lower() in ("quit", "exit", "q", "終了"):
                print("👋 終了します")
                break

            self.query(q)


def main():
    text_dir = os.environ.get("TEXT_DIR", "texts")
    strategy = os.environ.get("CHUNK_STRATEGY", "paragraph")

    rag = BunkoRAG(text_dir=text_dir, chunk_strategy=strategy)

    # 引数があればワンショット、なければ対話モード
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        rag.query(question)
    else:
        rag.interactive()


if __name__ == "__main__":
    main()
