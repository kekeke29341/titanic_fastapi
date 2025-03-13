# Titanic Prediction Service

このプロジェクトは、タイタニック乗客データを使用して生存予測を行うウェブサービスを提供します。LightGBMモデルを使用し、同期APIと非同期APIの両方をサポートしています。

## 機能

- **同期API**: リクエストを送信し、即時に予測結果を取得
- **非同期API**: バックグラウンドでの処理をサポートし、長時間実行されるタスクに適しています
- **Dockerを使用した簡単なデプロイ**
- **LightGBMを使用した高精度な予測モデル**

## 前提条件

- Python 3.8以上
- Docker (コンテナ化実行用)
- pip

## セットアップと実行方法

### ローカル環境での実行

1. リポジトリをクローン

```bash
git clone https://github.com/yourusername/titanic-prediction-service.git
cd titanic-prediction-service
```

2. 仮想環境を作成し、依存パッケージをインストール

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
pip install -r requirements.txt
```

3. モデルをトレーニング

```bash
python ml/model.py
```

4. サービスを起動

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. ブラウザで `http://localhost:8000/docs` にアクセスして、Swagger UIからAPIをテスト

### Dockerを使用した実行

1. Dockerイメージをビルド

```bash
docker build -t titanic-prediction-service .
```

2. コンテナを実行

```bash
docker run -d -p 8000:8000 --name titanic-service titanic-prediction-service
```

3. ブラウザで `http://localhost:8000/docs` にアクセスしてAPIをテスト

### Docker Composeを使用した実行

1. Docker Composeを使ってサービスを起動

```bash
docker-compose up -d
```

2. ブラウザで `http://localhost:8000/docs` にアクセスしてAPIをテスト

## APIの使用方法

### 同期API

#### リクエスト

```bash
curl -X 'POST' \
  'http://localhost:8000/titanic_sync' \
  -H 'Content-Type: application/json' \
  -d '{
  "pclass": 3,
  "sex": "male",
  "age": 22.0,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S",
  "name": "John Doe"
}'
```

#### レスポンス

```json
{
  "survived": 0,
  "probability": 0.127,
  "processing_time_seconds": 0.043
}
```

### 非同期API

#### リクエスト

```bash
curl -X 'POST' \
  'http://localhost:8000/titanic_async' \
  -H 'Content-Type: application/json' \
  -d '{
  "pclass": 1,
  "sex": "female",
  "age": 38.0,
  "sibsp": 1,
  "parch": 0,
  "fare": 71.25,
  "embarked": "C",
  "name": "Jane Doe"
}'
```

#### レスポンス

```json
{
  "job_id": "5f4e7a1c-5b5a-4c9e-9c9f-c9c9c9c9c9c9"
}
```

#### ジョブステータスの確認

```bash
curl -X 'GET' \
  'http://localhost:8000/jobs/5f4e7a1c-5b5a-4c9e-9c9f-c9c9c9c9c9c9' \
  -H 'accept: application/json'
```

#### ジョブステータスのレスポンス例

```json
{
  "id": "5f4e7a1c-5b5a-4c9e-9c9f-c9c9c9c9c9c9",
  "status": "completed",
  "result": {
    "survived": 1,
    "probability": 0.913
  },
  "error": null,
  "created_at": "2025-03-13T12:34:56.789012",
  "updated_at": "2025-03-13T12:34:58.123456"
}
```

## 入力データ形式

予測APIは以下のパラメータを受け付けます：

| パラメータ | タイプ | 説明 | 必須 |
|------------|--------|------|------|
| pclass | 整数 | 乗客クラス (1, 2, or 3) | はい |
| sex | 文字列 | 性別 ("male" or "female") | はい |
| age | 浮動小数点 | 年齢 | いいえ |
| sibsp | 整数 | 同乗している兄弟/配偶者の数 | はい |
| parch | 整数 | 同乗している親/子供の数 | はい |
| fare | 浮動小数点 | 乗船料金 | いいえ |
| embarked | 文字列 | 乗船港 ("C", "Q", or "S") | いいえ |
| name | 文字列 | 乗客名 | いいえ |
| cabin | 文字列 | 客室番号 | いいえ |
| ticket | 文字列 | チケット番号 | いいえ |

## プロジェクト構造

```
titanic-prediction-service/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI メインアプリケーション
│   ├── models/
│   │   ├── __init__.py
│   │   ├── prediction.py   # 予測モデルとスキーマ
│   │   └── job.py          # 非同期処理用ジョブモデル
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ml_model.py     # MLモデルのロードと予測
│   │   └── job_manager.py  # バックグラウンドジョブ管理
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py # データ前処理ユーティリティ
│   └── config.py           # アプリケーション設定
├── data/
│   └── titanic.csv         # トレーニング用サンプルデータセット
├── ml/
│   ├── model.py            # モデルトレーニングスクリプト
│   ├── model_artifacts/    # 保存されたモデルと前処理オブジェクト
│   │   ├── model.pkl       # トレーニング済みLightGBMモデル
│   │   └── preprocessor.pkl # 適合済み前処理パイプライン
│   └── notebooks/          # 探索的ノートブック
│       └── model_development.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_sync_api.py
│   └── test_async_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 技術スタック

- **FastAPI**: 高性能APIフレームワーク
- **LightGBM**: 高速・高精度の勾配ブースティングフレームワーク
- **Pandas & Scikit-learn**: データ処理と前処理
- **Docker**: コンテナ化と配布
- **Uvicorn**: ASGIサーバー

## 将来の改善点

- Redis/Celeryを使用した本格的な非同期ジョブキュー
- ログ収集と監視の強化
- スケーラビリティの向上
- モデルのA/Bテスト機能
- モデルのドリフト検出
