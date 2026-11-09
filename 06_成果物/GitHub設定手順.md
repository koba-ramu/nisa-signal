# GitHub の設定手順(段階2:毎朝の自動取得)

このパソコンには git が入っていないため、**GitHub の Web 画面だけで**設定します。所要時間は15分ほどです。

## 先に知っておくこと
- リポジトリ(ファイルの置き場所)は **公開(Public)** にします。無料で GitHub Pages(段階5)を使うために必要です
- 公開されるのは、アプリの画面・株価データ・監視銘柄の一覧だけです。**保有株・資金額・売買記録はスマホの中だけ**に保存されるので、公開されません
- GeminiのAPIキーは、ここではまだ使いません(段階4で、Secretsに登録します)

## 手順

### 1. リポジトリを作る
1. https://github.com にログインする
2. 右上の「+」→「New repository」
3. Repository name:`nisa-signal`
4. 「Public」を選ぶ
5. 「Create repository」を押す

### 2. ファイルをアップロードする
1. 作ったリポジトリの画面で「uploading an existing file」のリンクを押す
2. パソコンで `AI開発作業場\投資アプリ\06_成果物\app` フォルダを開き、**中身**(下記)をまとめてブラウザにドラッグする
   - `index.html` / `使い方.md` / `requirements.txt`
   - `data` フォルダ / `scripts` フォルダ
3. 下の「Commit changes」を押す

※ `.github` フォルダは、ドラッグでは上がらないことがあるため、次の手順3で作ります。

### 3. 自動実行の設定ファイルを作る
1. リポジトリの画面で「Add file」→「Create new file」
2. ファイル名の欄に、次のとおり入力する(`/` を打つと自動でフォルダになります)
   ```
   .github/workflows/fetch-data.yml
   ```
3. 本文に、`app\.github\workflows\fetch-data.yml` の中身をそのまま貼り付ける(メモ帳で開いてコピー)
4. 「Commit changes」を押す

### 4. 試しに動かす
1. リポジトリ上部の「Actions」タブを開く
2. 左の「株価データの取得」→ 右の「Run workflow」→ 緑の「Run workflow」
3. 1〜3分待って、緑のチェックが付けば成功です
4. 「Code」タブに戻り、`data/market.json` の更新時刻が新しくなっていれば完了です

### うまくいかないとき
| 症状 | 対処 |
|---|---|
| 赤い × が付き、「Permission denied」「403」と出る | 「Settings」→「Actions」→「General」→ 一番下の「Workflow permissions」で「Read and write permissions」を選んで「Save」→ もう一度実行 |
| 「Actions」タブに何も出ない | 手順3のファイル名(`.github/workflows/fetch-data.yml`)が正しいか確認 |
| 「Too Many Requests」「429」と出る | Yahoo Finance 側の一時的な制限です。時間をおいて再実行。続くようなら CEO ラムに伝える |

## 終わったら
CEO ラムに「GitHubの設定が終わった」と伝えてください。段階2の完成条件は「**7日連続で取得に成功すること**」です。毎朝7時30分に自動で動くので、1週間ようすを見ます。
