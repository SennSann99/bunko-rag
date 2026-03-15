# 文庫 RAG (bunko-rag)

日本文学の名作を読み解く RAG（検索拡張生成）システム。  
[青空文庫](https://www.aozora.gr.jp/)のテキストを使って、作品の内容について質問応答ができます。

**Docker + Ollama で完全ローカル動作。API キー不要。**

## 構成

```
User Query → Embedding (BGE-M3) → Vector Search → Context → LLM (Qwen 2.5) → Answer
```

| コンポーネント | 技術 | 役割 |
|---|---|---|
| LLM | Qwen 2.5 7B (Ollama) | 日本語回答生成 |
| Embedding | BGE-M3 (Ollama) | 多言語ベクトル化 |
| Vector Store | NumPy (in-memory) | コサイン類似度検索 |
| テキスト | 青空文庫 | 著作権切れの日本文学 |

## クイックスタート

### 1. 起動

```bash
git clone https://github.com/SennSann99/bunko-rag.git
cd bunko-rag

# Ollama + モデルのダウンロード + RAG アプリを一括起動
docker compose up
```

初回はモデルのダウンロード（約 5GB）に時間がかかります。

### 2. 質問する

```bash
# 対話モード
docker compose run rag-app

# ワンショット
docker compose run rag-app python src/main.py "蜘蛛の糸が切れた理由は？"
```

### Docker なしで使う場合

```bash
# Ollama をインストール (https://ollama.com)
ollama pull qwen2.5:7b
ollama pull bge-m3

pip install -r requirements.txt
python src/main.py
```

## 収録作品

`texts/` ディレクトリに青空文庫のテキストファイルを配置してください。

サンプルとして芥川龍之介「蜘蛛の糸」が含まれています。  
作品の追加は [青空文庫](https://www.aozora.gr.jp/) からダウンロードするか、
[GitHub リポジトリ](https://github.com/aozorahack/aozorabunko_text) から一括取得できます。

```bash
# 例: 芥川龍之介の他の作品を追加
# 青空文庫の図書カードからテキスト版の zip をダウンロード → 解凍 → texts/ に配置
```

## プロジェクト構成

```
bunko-rag/
├── docker-compose.yml    # Docker 構成 (Ollama + RAG app)
├── Dockerfile            # RAG アプリのイメージ
├── requirements.txt      # Python 依存パッケージ
├── texts/                # 青空文庫テキスト
│   └── kumo_no_ito.txt   # 芥川龍之介「蜘蛛の糸」
└── src/
    ├── main.py           # エントリポイント & RAG パイプライン
    ├── loader.py          # 青空文庫テキストのローダー & チャンク分割
    ├── vectorstore.py     # ベクトルストア (Neural / TF-IDF)
    └── ollama_client.py   # Ollama API クライアント
```

## 動作モード

| モード | 条件 | 検索 | 回答生成 |
|---|---|---|---|
| Full | Ollama + 両モデル | BGE-M3 (Neural) | Qwen 2.5 |
| Search-only | Ollama + BGE-M3のみ | BGE-M3 (Neural) | 無効 |
| Offline | Ollama なし | TF-IDF | 無効 |

Ollama に接続できない場合でも TF-IDF フォールバックで検索機能は動作します。

## 環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama サーバーの URL |
| `LLM_MODEL` | `qwen2.5:7b` | 回答生成に使う LLM |
| `EMBED_MODEL` | `bge-m3` | 埋め込みに使うモデル |
| `TEXT_DIR` | `texts` | テキストファイルのディレクトリ |
| `CHUNK_STRATEGY` | `paragraph` | チャンク分割戦略 (`paragraph` / `section`) |

## GPU を使う

`docker-compose.yml` の GPU セクションのコメントを外してください:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## ライセンス

MIT

青空文庫のテキストは[収録ファイルの取り扱い規準](https://www.aozora.gr.jp/guide/kijyunn.html)に従ってください。

## 参考

- [W&B RAG Course](https://wandb.ai/rag-course/rag-course/reports/How-to-build-a-RAG-system--Vmlldzo5NjY0NDUw) — RAG の基本を学ぶコース
- [Ollama](https://ollama.com) — ローカル LLM ランナー
- [Qwen 2.5](https://qwen2.org) — 日本語対応 LLM
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — 多言語埋め込みモデル
- [青空文庫](https://www.aozora.gr.jp/) — 日本語テキストアーカイブ
- [awesome-japanese-llm](https://github.com/llm-jp/awesome-japanese-llm) — 日本語 LLM まとめ
