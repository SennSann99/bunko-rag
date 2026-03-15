"""
ベクトルストア: チャンクの埋め込みと類似検索

2つのモード:
  - Neural: Ollama の BGE-M3 を使う高精度モード
  - TF-IDF: ライブラリ不要のフォールバックモード
"""

import numpy as np
from typing import Optional


class VectorStore:
    """
    シンプルなインメモリベクトルストア。

    本番環境では ChromaDB, Weaviate, Qdrant 等に置き換える。
    学習目的ではこのシンプルな実装で十分。
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self.vectors: Optional[np.ndarray] = None
        self.mode: str = "none"

    def add_chunks(self, chunks: list[dict], use_neural: bool = True):
        """チャンクを追加してベクトル化する"""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        if use_neural:
            self._build_neural(texts)
        else:
            self._build_tfidf(texts)

    def _build_neural(self, texts: list[str]):
        """Ollama BGE-M3 でニューラル埋め込みを構築"""
        try:
            from ollama_client import embed

            print("  🧠 BGE-M3 で埋め込みベクトルを生成中...")

            # バッチ処理（Ollama は一度に複数テキストを処理可能）
            embeddings = embed(texts)
            self.vectors = np.array(embeddings, dtype=np.float32)
            self.mode = "neural"
            print(f"  ✅ {len(texts)} チャンク × {self.vectors.shape[1]}次元")

        except Exception as e:
            print(f"  ⚠️  Neural embedding 失敗: {e}")
            print("  → TF-IDF フォールバックに切り替えます")
            self._build_tfidf(texts)

    def _build_tfidf(self, texts: list[str]):
        """TF-IDF フォールバック（Ollama不要）"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("  📊 TF-IDF で埋め込みベクトルを生成中...")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=5000,
        )
        self.vectors = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        self._tfidf_vectorizer = vectorizer
        self.mode = "tfidf"
        print(f"  ✅ {len(texts)} チャンク × {self.vectors.shape[1]}次元 (TF-IDF)")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """コサイン類似度で上位 k 件のチャンクを検索する"""
        if self.vectors is None:
            return []

        # クエリをベクトル化
        if self.mode == "neural":
            from ollama_client import embed
            q_vec = np.array(embed([query])[0], dtype=np.float32)
        else:
            q_vec = self._tfidf_vectorizer.transform([query]).toarray()[0].astype(np.float32)

        # コサイン類似度を計算
        # cos_sim = (q · v) / (|q| × |v|)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scores = []
        for i, v in enumerate(self.vectors):
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(q_vec, v) / (q_norm * v_norm)))

        # 上位 k 件を返す
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk": self.chunks[i], "score": scores[i]}
            for i in top_idx
        ]

    @property
    def size(self) -> int:
        return len(self.chunks)
