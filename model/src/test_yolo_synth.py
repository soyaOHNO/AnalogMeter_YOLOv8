import cv2
import os
import glob
import math
import numpy as np
from ultralytics import YOLO

# ==========================================
# 設定パス (合成データテスト用に変更)
# ==========================================
# ★ Blenderで出力したCG画像のフォルダパスを指定してください
IMAGE_DIR = r"..\yolo_test_data\images" 
MODEL_PATH = r"..\yolo_best\best.pt"  # ダウンロードした学習済みモデル

# CGデータ用の固定メタデータ
TEST_META = {"min": 0.0, "max": 1.0, "unit": "MPa"}

# ==========================================
# 計算ロジック (完全同一)
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
    print("=== YOLOv8 CGデータ推論テスト開始 ===")
    
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
            kpts_all = result.keypoints.xy.cpu().numpy() 
            
            for i in range(len(boxes)):
                bbox = boxes[i] 
                p_kpts = kpts_all[i] 
                
                if len(p_kpts) < 6 or np.all(p_kpts == 0):
                    print(f"{img_name} - Inst {i}: キーポイントが正しく検出されませんでした。")
                    continue

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
                
                val = TEST_META["min"] + (TEST_META["max"] - TEST_META["min"]) * ratio
                
                print(f"{img_name} - Inst {i}: {val:.3f} {TEST_META['unit']} (Ratio: {ratio*100:.1f}%)")
                
                # 描画
                raw_center = tuple(center.astype(int))
                raw_shifted_tip = tuple(shifted_tip.astype(int))
                
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.line(img, tuple(pivot.astype(int)), tuple(tip.astype(int)), (255, 100, 0), 2)
                cv2.line(img, raw_center, raw_shifted_tip, (255, 255, 0), 2)
                
                for pt in p_kpts:
                    cv2.circle(img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
                
                cv2.putText(img, f"{val:.3f}{TEST_META['unit']}", (raw_shifted_tip[0], raw_shifted_tip[1]-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            print(f"{img_name}: 計器が検出されませんでした。")

        cv2.imshow("YOLO Synth Test", img)
        if cv2.waitKey(0) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()