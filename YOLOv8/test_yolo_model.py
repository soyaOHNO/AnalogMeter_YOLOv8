import cv2
import os
import glob
import math
import numpy as np
import json
from ultralytics import YOLO

# ==========================================
# 設定パス (環境に合わせて微調整してください)
# ==========================================
# ★学習で生成されたベストな重みファイルのパス
# ※何回か学習を回している場合は train2, train3 などになることがあります
MODEL_PATH = r"runs\pose\train-2\weights\best.pt"

# テスト対象の画像フォルダ (学習に使わなかった検証用データ5枚)
IMAGE_DIR = r"..\yolo_dataset\images\val"

# メタデータ (目盛りのMin/Max値など)
META_PATH = "meter_meta.json"

# ==========================================
# 幾何学計算ロジック (マイナス/オーバー対応の完全版)
# ==========================================
def apply_perspective_transform(bbox, kpts):
    """BBoxを正方形(真円空間)に引き伸ばす射影変換"""
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
    """Min〜Maxの範囲外にある場合でも、近い方の振れ幅を元に外挿(マイナス/オーバー)計算を行う"""
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

def main():
    print("=== YOLOv8 推論 ＆ 読み取りテスト ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません -> {MODEL_PATH}")
        print("※パスが正しいか(train2やtrain3になっていないか)確認してください。")
        return

    # YOLOモデルの読み込み
    model = YOLO(MODEL_PATH)
    
    # メタデータの読み込み
    meta_data = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding='utf-8') as f:
            meta_data = json.load(f)

    # テスト画像の取得
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        
    if not image_paths:
        print(f"エラー: テスト画像が見つかりません -> {IMAGE_DIR}")
        return

    cv2.namedWindow("AI Inference Result", cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 1. AIによる推論の実行
        # (conf=0.5 は、自信度50%以上の結果のみを採用するという設定)
        results = model(img, conf=0.5)
        result = results[0]
        
        img_meta = meta_data.get(img_name, [])
        
        # AIが計器を検出できた場合
        if result.keypoints is not None and result.boxes is not None:
            # 検出されたすべての計器(インスタンス)に対してループ
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].cpu().numpy() # [x1, y1, x2, y2]
                kpts = result.keypoints.xy[i].cpu().numpy() # [6, 2]
                
                # 6点すべて検出できていない場合はスキップ
                if len(kpts) < 6 or np.all(kpts == 0):
                    continue
                    
                pivot, tip, min_pt, mid_pt, max_pt, center = kpts

                # ====================================================
                # AIが予測した座標を使って、値を計算する
                # ====================================================
                needle_vector = tip - pivot
                shifted_tip = center + needle_vector
                
                target_pts = np.array([center, shifted_tip, min_pt, mid_pt, max_pt], dtype=np.float32)
                t_kpts = apply_perspective_transform(box, target_pts)
                t_center, t_tip, t_min, t_mid, t_max = t_kpts

                ang_min = calculate_angle_clockwise(t_center, t_min)
                ang_mid = calculate_angle_clockwise(t_center, t_mid)
                ang_max = calculate_angle_clockwise(t_center, t_max)
                ang_tip = calculate_angle_clockwise(t_center, t_tip)
                
                ratio = calculate_normalized_ratio(ang_min, ang_mid, ang_max, ang_tip)
                
                meta = next((item for item in img_meta if item["index"] == i), {"min": 0.0, "max": 1.0, "unit": ""})
                val = meta["min"] + (meta["max"] - meta["min"]) * ratio
                
                conf = float(result.boxes.conf[i].cpu().numpy())
                print(f"{img_name} - 計器{i+1}: {val:.2f} {meta['unit']} (AI自信度: {conf*100:.1f}%)")

                # ====================================================
                # 描画処理 (AIがどこを見ているか可視化)
                # ====================================================
                x1, y1, x2, y2 = map(int, box)
                raw_center = tuple(center.astype(int))
                raw_shifted_tip = tuple(shifted_tip.astype(int))
                
                # AIが見つけた計器の枠 (緑)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 計算された針のベクトル (シアン)
                cv2.line(img, raw_center, raw_shifted_tip, (255, 255, 0), 2)
                
                # AIが予測した6つのキーポイント (赤丸)
                for pt in kpts:
                    cv2.circle(img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
                
                # 読み取り値のテキスト表示
                cv2.putText(img, f"{val:.2f}{meta['unit']}", (raw_shifted_tip[0], raw_shifted_tip[1]-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("AI Inference Result", img)
        # 何かキーを押すと次の画像へ
        if cv2.waitKey(0) & 0xFF == ord('q'): 
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()