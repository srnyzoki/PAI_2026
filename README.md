# グレアの再現とピックアンドプレースの実装

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<GitHubユーザー名>/<リポジトリ名>/blob/main/glare_aware_pick_and_place.ipynb)

## 1. 概要と目的
カメラを用いた視覚フィードバックによるロボットアーム（CRANE+ V2想定）のPick & Placeタスクにおいて、照明環境による**鏡面反射（グレア・白飛び）**は物体重心推論の大きな誤差要因となる。
画像からグレアの深刻度を3段階に区別し、**「ソフトウェア画像修復（Inpaint + CLAHE）」**と**「ロボット自身の身体性を活用した物理的遮光（Active Shadowing）」**を適応的に切り替えられることを確認できる。

## 2. 実行方法
### Google Colab で実行してください
上部の **「Open in Colab」** をクリックし、セルを実行してください。コンソールにプレビュー画像が表示され、対話形式でシナリオの実行・比較が可能です。実行完了後、`demo_output.mp4` と `dashboard_summary.png` がColab上に生成されるのでローカルにダウンロードして確認してください。