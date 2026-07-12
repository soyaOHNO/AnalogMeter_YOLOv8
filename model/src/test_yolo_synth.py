import cv2
import os
import glob
import math
import numpy as np
from ultralytics import YOLO

# ==========================================
# 設定パス
# ==========================================
# ★ テストしたい画像のフォルダパス（統合テストデータ等）を指定してください
IMAGE_DIR = r"..\yolo_test_data_all\images" 
# ★ ダウンロードした72エポック目の best.pt のパスを指定してください
MODEL_PATH = r"best.pt"  

from meter_config import META_DATA # メタデータのパス

# 描画用の色設定 (B, G, R)
COLOR_MAP = {
    0: (255, 150, 0),  # Atmos: 青系
    1: (0, 100, 255),  # Thermo: オレンジ系
    2: (0, 255, 100)   # Hygro: 緑系
}

# ==========================================
# 計算ロジック (変更なし)
# ==========================================
def apply_perspective_transform(bbox, kpts):
    x1, y1, x2, y2 = bbox
    src_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    side = max(x2-x1, y2-y1)
    dst_pts = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed_kpts = cv2.perspectiveTransform(kpts.reshape(-1, 1, 2), M).reshape(-1, 2)
    return transformed_kpts

def calculate_angle_clockwise(p1, p2):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    return angle if angle >= 0 else angle + 360

def calculate_normalized_ratio(ang_min, ang_mid, ang_max, ang_tip):
    def diff(start, end): return (end - start) % 360
    
    arc1 = diff(ang_min, ang_mid)
    arc2 = diff(ang_mid, ang_max)
    dist_from_min = diff(ang_min, ang_tip)
    
    if dist_from_min <= arc1:
        return 0.5 * (dist_from_min / arc1) if arc1 > 0 else 0.0
    elif dist_from_min <= arc1 + arc2:
        dist_from_mid = diff(ang_mid, ang_tip)
        return 0.5 + 0.5 * (dist_from_mid / arc2) if arc2 > 0 else 0.5
    else:
        d_max_to_tip = diff(ang_max, ang_tip)
        d_tip_to_min = diff(ang_tip, ang_min)
        if d_max_to_tip <= d_tip_to_min:
            return 1.0 + 0.5 * (d_max_to_tip / arc2) if arc2 > 0 else 1.0
        else:
            return 0.0 - 0.5 * (d_tip_to_min / arc1) if arc1 > 0 else 0.0

# ==========================================
# メイン推論処理
# ==========================================
def main():
    print("=== YOLOv8 万能モデル 推論テスト開始 ===")
    
    model = YOLO(MODEL_PATH)
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        
    if not image_paths:
        print(f"エラー: {IMAGE_DIR} に画像が見つかりません。パスを確認してください。")
        return

    cv2.namedWindow("YOLO Synth Test", cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
            
        results = model.predict(source=img, conf=0.5, save=False)
        result = results[0]

        if result.boxes and result.keypoints:
            boxes = result.boxes.xyxy.cpu().numpy() 
            classes = result.boxes.cls.cpu().numpy()  # ★AIが予測したクラスIDを取得
            kpts_all = result.keypoints.xy.cpu().numpy() 
            
            for i in range(len(boxes)):
                bbox = boxes[i]
                cls_id = int(classes[i])  # ★クラスID (0, 1, 2) を整数化
                p_kpts = kpts_all[i] 
                
                # 未知のクラスIDやキーポイント不足の場合はスキップ
                if cls_id not in META_DATA:
                    continue
                if len(p_kpts) < 6 or np.all(p_kpts == 0):
                    print(f"{img_name} - {META_DATA[cls_id]['name']}: キーポイント検出失敗")
                    continue

                meta = META_DATA[cls_id]  # ★該当クラスのメタデータを引き出す
                color = COLOR_MAP.get(cls_id, (0, 255, 0))

                pivot, tip, min_pt, mid_pt, max_pt, center = p_kpts

                # ベクトル計算と平行移動
                needle_vector = tip - pivot
                shifted_tip = center + needle_vector
                
                target_pts = np.array([center, shifted_tip, min_pt, mid_pt, max_pt], dtype=np.float32)
                t_kpts = apply_perspective_transform(bbox, target_pts)
                t_center, t_tip, t_min, t_mid, t_max = t_kpts

                ang_min = calculate_angle_clockwise(t_center, t_min)
                ang_mid = calculate_angle_clockwise(t_center, t_mid)
                ang_max = calculate_angle_clockwise(t_center, t_max)
                ang_tip = calculate_angle_clockwise(t_center, t_tip)
                
                ratio = calculate_normalized_ratio(ang_min, ang_mid, ang_max, ang_tip)
                
                # ★引き出したメタデータの min と max を使って実際の数値を計算
                val = meta["min"] + (meta["max"] - meta["min"]) * ratio
                
                print(f"{img_name} - {meta['name']}検出: {val:.1f} {meta['unit']} (Ratio: {ratio*100:.1f}%)")
                
                # 描画
                raw_center = tuple(center.astype(int))
                raw_shifted_tip = tuple(shifted_tip.astype(int))
                
                # バウンディングボックスと線を描画（計器の種類ごとに色分け）
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.line(img, tuple(pivot.astype(int)), tuple(tip.astype(int)), (255, 255, 255), 1)
                cv2.line(img, raw_center, raw_shifted_tip, color, 2)
                
                for pt in p_kpts:
                    cv2.circle(img, tuple(pt.astype(int)), 4, (0, 0, 255), -1)
                
                # テキスト情報（計器名と数値）を画面上に描画
                display_text = f"{meta['name']}: {val:.1f}{meta['unit']}"
                cv2.putText(img, display_text, (raw_shifted_tip[0] - 30, raw_shifted_tip[1] - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            print(f"{img_name}: 計器が検出されませんでした。")

        cv2.imshow("YOLO Synth Test", img)
        # qキーで終了、それ以外のキーで次の画像へ
        if cv2.waitKey(0) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()