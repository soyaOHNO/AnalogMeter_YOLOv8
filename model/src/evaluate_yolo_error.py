import cv2
import os
import glob
import math
import numpy as np
from ultralytics import YOLO

# ==========================================
# 設定パス (Blenderで出力したテスト用データ)
# ==========================================
IMAGE_DIR = r"..\yolo_test_data\images"
LABEL_DIR = r"..\yolo_test_data\labels"
MODEL_PATH = r"..\yolo_best\best.pt"  # 学習済みモデル

# 評価用の固定メタデータ (テストなので0.0〜1.0で計算)
TEST_META = {"min": 0.0, "max": 1.0, "unit": "MPa"}


def read_yolo_format(txt_path, img_w, img_h):
    """正解ラベル(.txt)から座標を読み込む"""
    loaded_instances = []
    if not os.path.exists(txt_path): return loaded_instances
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 23:
                xc, yc, w, h = map(float, parts[1:5])
                bbox = (int((xc - w/2)*img_w), int((yc - h/2)*img_h), 
                        int((xc + w/2)*img_w), int((yc + h/2)*img_h))
                kpts = []
                for i in range(6):
                    px = int(float(parts[5 + i*3]) * img_w)
                    py = int(float(parts[6 + i*3]) * img_h)
                    kpts.append([px, py])
                loaded_instances.append({'bbox': bbox, 'kpts': np.array(kpts, dtype=np.float32)})
    return loaded_instances

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

def get_meter_value(bbox, kpts):
    """BBoxと6点から最終的な数値を計算するラッパー関数"""
    pivot, tip, min_pt, mid_pt, max_pt, center = kpts
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
    return TEST_META["min"] + (TEST_META["max"] - TEST_META["min"]) * ratio

# ==========================================
# 評価実行
# ==========================================
def main():
    print("=== YOLOv8 誤差(Error)検証開始 ===")
    
    model = YOLO(MODEL_PATH)
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    
    if not image_paths:
        print(f"エラー: 画像が見つかりません -> {IMAGE_DIR}")
        return

    errors = []

    print("-" * 60)
    print(f"{'ファイル名':<15} | {'絶対正解(GT)':<8} | {'AI予測(Pred)':<10} | {'誤差(Error)':<10}")
    print("-" * 60)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        # 1. 絶対正解 (Ground Truth) の計算
        gt_instances = read_yolo_format(txt_path, img_w, img_h)
        if not gt_instances:
            continue
        gt_val = get_meter_value(gt_instances[0]['bbox'], gt_instances[0]['kpts'])

        # 2. AI予測 (Prediction) の計算
        results = model.predict(source=img, conf=0.5, verbose=False) # ログ抑制
        result = results[0]
        
        if result.boxes and result.keypoints:
            pred_bbox = result.boxes.xyxy[0].cpu().numpy()
            pred_kpts = result.keypoints.xy[0].cpu().numpy()
            
            if len(pred_kpts) == 6 and not np.all(pred_kpts == 0):
                pred_val = get_meter_value(pred_bbox, pred_kpts)
                
                # 3. 誤差の計算
                error = abs(gt_val - pred_val)
                errors.append(error)
                
                print(f"{img_name:<20} | {gt_val:>8.3f} MPa | {pred_val:>8.3f} MPa | {error:>8.3f} MPa")
            else:
                print(f"{img_name:<20} | {gt_val:>8.3f} MPa | {'[キーポイント欠損]':>12} | {'-':>10}")
        else:
            print(f"{img_name:<20} | {gt_val:>8.3f} MPa | {'[計器未検出]':>12} | {'-':>10}")

    print("-" * 60)
    if errors:
        mae = sum(errors) / len(errors)
        max_err = max(errors)
        print(f"✅ 検証完了: {len(errors)} 枚の画像をテストしました。")
        print(f"📊 平均誤差 (MAE) : {mae:.4f} MPa")
        print(f"📈 最大誤差 (Max) : {max_err:.4f} MPa")
    else:
        print("計算できるデータがありませんでした。")

if __name__ == "__main__":
    main()