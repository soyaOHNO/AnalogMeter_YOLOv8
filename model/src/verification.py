import cv2
import os
import glob
import random

# ==========================================
# 設定項目
# ==========================================
# ★ 確認したい画像のフォルダと、ラベルのフォルダを指定してください
IMAGE_DIR = r"..\yolo_train_data_all\images"
LABEL_DIR = r"..\yolo_train_data_all\labels"

# 統合した14クラスの名前
CLASS_NAMES = [
    'Pressure0to1', 'hPa970to1050', 'hPa970to1060', 'mmHg730to790',
    'Thermo-30to50', 'Thermo-20to50', 'Thermo-10to40', 'Thermo0to120',
    'Thermo10to40', 'Thermo20to110', 'Hygro0to100', 'Hygro10to100',
    'Hygro40to100', 'rpm400to1800'
]

# キーポイントを見分けやすくするための色設定 (B, G, R)
KPT_COLORS = [
    (255, 0, 0),    # 0: Pivot (青)
    (0, 0, 255),    # 1: Tip   (赤)
    (0, 255, 255),  # 2: Min   (黄)
    (0, 255, 0),    # 3: Mid   (緑)
    (255, 0, 255),  # 4: Max   (ピンク)
    (255, 255, 255) # 5: Center(白)
]
KPT_NAMES = ["Pivot", "Tip", "Min", "Mid", "Max", "Center"]

# 1モデルあたりに確認する枚数
NUM_SAMPLES_PER_MODEL = 5

# ==========================================
# メイン処理
# ==========================================
def main():
    print("=== YOLOデータセット検証ツール（ランダムサンプリング版） ===")
    
    # フォルダ内の全画像を取得
    all_image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    if not all_image_paths:
        print(f"エラー: {IMAGE_DIR} に画像が見つかりません。")
        return

    # モデル番号(n)ごとに画像をグループ化する辞書
    model_groups = {}
    for img_path in all_image_paths:
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        # ファイル名が "model_n_a.jpg" の形式であることを確認
        if len(parts) >= 2 and parts[0] == 'model':
            model_n = parts[1]
            if model_n not in model_groups:
                model_groups[model_n] = []
            model_groups[model_n].append(img_path)

    # 各モデルから指定枚数（5枚）をランダムに抽出
    selected_image_paths = []
    
    # モデル番号順（1, 2, 3...10）にソートして確認しやすくする
    sorted_model_ids = sorted(model_groups.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    for model_n in sorted_model_ids:
        paths = model_groups[model_n]
        if len(paths) > NUM_SAMPLES_PER_MODEL:
            sampled_paths = random.sample(paths, NUM_SAMPLES_PER_MODEL)
        else:
            sampled_paths = paths
        selected_image_paths.extend(sampled_paths)

    total_selected = len(selected_image_paths)
    print(f"全 {len(all_image_paths)} 枚の中から、各モデル最大 {NUM_SAMPLES_PER_MODEL} 枚ずつ、計 {total_selected} 枚を抽出しました。")
    print("操作方法: [Enter] 次の画像へ / [q] 終了")

    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)

    for idx, img_path in enumerate(selected_image_paths):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(LABEL_DIR, f"{base_name}.txt")

        img = cv2.imread(img_path)
        if img is None: continue
        
        img_h, img_w = img.shape[:2]

        # 画面左上に進捗（例：[1/50] model_1_0342.jpg）を描画
        progress_text = f"[{idx + 1}/{total_selected}] {img_name}"
        cv2.putText(img, progress_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, progress_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ラベルファイルが存在するか確認
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 23: 
                    continue
                
                # YOLOフォーマットのパース
                cls_id = int(parts[0])
                cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])
                
                # BBoxの座標計算
                x1 = int((cx_norm - w_norm / 2) * img_w)
                y1 = int((cy_norm - h_norm / 2) * img_h)
                x2 = int((cx_norm + w_norm / 2) * img_w)
                y2 = int((cy_norm + h_norm / 2) * img_h)
                
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
                
                # BBoxとクラス名の描画
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 0), 2)
                cv2.putText(img, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
                
                # キーポイントのパースと描画
                kpt_data = list(map(float, parts[5:]))
                for i in range(6):
                    kx_norm = kpt_data[i*3]
                    ky_norm = kpt_data[i*3 + 1]
                    
                    kx = int(kx_norm * img_w)
                    ky = int(ky_norm * img_h)
                    
                    color = KPT_COLORS[i]
                    cv2.circle(img, (kx, ky), 5, color, -1)
                    cv2.putText(img, KPT_NAMES[i], (kx + 8, ky + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            cv2.putText(img, "No Label Found", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Dataset Viewer", img)
        
        # キー入力待機
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("検証を終了します。")
            break

    cv2.destroyAllWindows()
    print("すべての確認が完了しました。")

if __name__ == "__main__":
    main()