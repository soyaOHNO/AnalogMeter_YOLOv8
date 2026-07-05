# AnalogMeter_YOLOv8 読み取りシステム

YOLOv8の姿勢推定（Pose Estimation / Keypoint Detection）を用いて、アナログ計器（圧力計など）の数値を自動で読み取るAIシステムです。

少量の実写画像から学習を始める限界を突破するため、**Blenderを用いた合成データセットの自動生成パイプライン（Sim2Real）**を構築し、極めて高い精度でのキーポイント検出と数値算出を実現しています。

## 🌟 主な機能と特徴

* **Sim2Real アプローチ:** BlenderのPython API (`bpy`) を活用し、カメラアングル・照明・針の角度をランダム化した「完璧なアノテーション付き合成データ」を数千枚規模で自動生成。実写データのアノテーションコストを劇的に削減しています。
* **6点キーポイント検出:** 計器の構造を `[Pivot, Tip, Min, Mid, Max, Center]` の6点で捉え、斜めから撮影された画像でも射影変換（Perspective Transform）により正面からの正円空間に補正して計算します。
* **高度な数値計算（外挿対応）:** 針が目盛りの「Min」を下回る場合や「Max」を振り切っている場合でも、角度から近い方の目盛り幅を基準にしてマイナス値やオーバー値を正確に算出（Extrapolation）します。
* **専用メタデータエディタ同梱:** 実写画像のメタデータ（計器ごとの最小値・最大値・単位）を爆速で入力するためのGUIツールを備えています。

---

## 📂 ディレクトリ構成（例）

```text
AnalogMeter_YOLOv8/
├── yolo_dataset/          # YOLO学習用データセット (train/val)
├── yolo_synth_data/       # Blenderから出力された合成データ
├── TargetImages/          # テスト・検証用の実写画像
├── meter_meta.json        # 計器ごとのMin/Max/Unitを記録したメタデータ
├── edit_meta.py           # メタデータ入力用GUIツール
├── merge_synth_data.py    # 合成データをYOLOデータセットに結合するスクリプト
├── train_yolo_pose.py     # YOLOv8-pose 学習実行スクリプト
├── test_yolo_model.py     # 推論および数値読み取りテスト用スクリプト
└── Blender_Script/
    └── generate_dataset.py # Blender内で実行するデータ量産スクリプト
```

## 🚀 使い方（ワークフロー）
本システムは以下のステップで運用します。

1. メタデータの準備 (GUIツール)
計器ごとに異なるスケール（0〜1 MPa など）を定義します。

```Bash
python edit_meta.py
```

画面左側に画像、右側にフォームが表示されます。テンキーと矢印キーでサクサク入力でき、自動で meter_meta.json に保存されます。1枚の画像に複数の計器がある場合にも対応しています。

2. Blenderによる合成データの量産 (Sim2Real)
実写データのアノテーション不足を補うため、Blenderでデジタルツインを作成し学習データを量産します。

Blenderで計器の簡易モデルと6つのエンプティ（Pivot, Tip, Min, Mid, Max, Center）を配置します。

BlenderのScriptingタブに generate_dataset.py を貼り付けて実行します。

カメラが半球面上をランダムに移動し、照明と針の角度を変えながら、完璧なYOLOフォーマットのラベル（.txt）と画像（.jpg）が yolo_synth_data フォルダに数千枚出力されます。

3. データセットの統合
生成した合成データを、YOLOの学習用(train)フォルダに統合します。検証用(val)には実写データを残すことで、Sim2Realの適応力をテストします。

Bash
python merge_synth_data.py
4. AIモデルの学習
YOLOv8s-pose を使用して学習を行います。合成データ特有の学習に最適化するため、空間的な歪み（perspective）のオーギュメンテーションはOFFにし、色相や照明（HSV）、ノイズ（erasing）に対する耐性を強化する設定になっています。

Bash
python train_yolo_pose.py
5. 推論と読み取りテスト
学習したモデル（best.pt）を用いて、未知の実写画像から数値を読み取ります。

```Bash
python test_yolo_model.py
```

## 🧮 読み取りアルゴリズムの解説
* 推論スクリプト (test_yolo_model.py) では、単にキーポイントの角度を測るだけでなく、以下の幾何学的な補正を行っています。

* 平行移動 (Shift): 針のベクトル (Tip - Pivot) を計算し、始点を計器の中心 (Center) に移動させた仮想の針 (Shifted Tip) を生成します。

* 射影変換 (Perspective Transform): 斜めから撮影された計器の歪みを直すため、計器を囲むバウンディングボックスを真四角に引き伸ばし、「真円空間」に変換します。

* 極座標系の角度計算: Centerを基準として、Min, Mid, Max, Shifted Tip の角度を時計回りに算出します。

* 比率算出と外挿: Min〜Mid と Mid〜Max のそれぞれの角度幅を基準に、針が何%の位置にあるかを算出します。針が範囲外にある場合は、近い方の基準幅を利用してマイナスや100%超えの比率を割り出します。

## 動作環境
* Python 3.8+
* ultralytics (YOLOv8)
* opencv-python
* numpy
* pillow, tkinter (GUIツール用)
* Blender 3.x 以上 (合成データ生成用)
---