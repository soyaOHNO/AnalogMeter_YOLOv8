import cv2
import os
import glob
import math
import numpy as np
import json

IMAGE_DIR = r"..\TargetImages"
LABEL_DIR = r"..\TargetLabels"

# メタデータの読み込み
with open("meter_meta.json", "r") as f:
    meta_data = json.load(f)

def read_yolo_format(txt_path, img_w, img_h):
    """6点構成(Pivot, Tip, Min, Mid, Max, Center)を読み込む"""
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
    """
    Min〜Maxの範囲外にある場合でも、近い方の振れ幅を元に外挿(マイナス/オーバー)計算を行う
    """
    def diff(start, end): return (end - start) % 360
    
    arc1 = diff(ang_min, ang_mid) # 前半の角度幅 (Min -> Mid)
    arc2 = diff(ang_mid, ang_max) # 後半の角度幅 (Mid -> Max)
    
    dist_from_min = diff(ang_min, ang_tip)
    
    if dist_from_min <= arc1:
        # 1. Min 〜 Mid の間 (0% 〜 50%)
        return 0.5 * (dist_from_min / arc1) if arc1 > 0 else 0.0
        
    elif dist_from_min <= arc1 + arc2:
        # 2. Mid 〜 Max の間 (50% 〜 100%)
        dist_from_mid = diff(ang_mid, ang_tip)
        return 0.5 + 0.5 * (dist_from_mid / arc2) if arc2 > 0 else 0.5
        
    else:
        # 3. 範囲外 (Minを下回っている or Maxをオーバーしている)
        d_max_to_tip = diff(ang_max, ang_tip)
        d_tip_to_min = diff(ang_tip, ang_min)
        
        # 針が「Maxの先」と「Minの手前」のどちらに近いかを比較
        if d_max_to_tip <= d_tip_to_min:
            # 【Max側をオーバーしている場合】
            # 後半(Mid〜Max)の角度幅(arc2)を基準にして、どれくらいオーバーしたかを足す (100% + α)
            return 1.0 + 0.5 * (d_max_to_tip / arc2) if arc2 > 0 else 1.0
        else:
            # 【Min側を下回っている場合】
            # 前半(Min〜Mid)の角度幅(arc1)を基準にして、どれくらい下回ったかを引く (0% - α)
            return 0.0 - 0.5 * (d_tip_to_min / arc1) if arc1 > 0 else 0.0

def main():
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        
    cv2.namedWindow("Vector Validation", cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        txt_path = os.path.join(LABEL_DIR, f"{os.path.splitext(img_name)[0]}.txt")
        img = cv2.imread(img_path)
        if img is None: continue
            
        img_h, img_w = img.shape[:2]
        instances = read_yolo_format(txt_path, img_w, img_h)
        img_meta = meta_data.get(img_name, [])

        for i, inst in enumerate(instances):
            p_kpts = inst['kpts'] # [Pivot, Tip, Min, Mid, Max, Center]
            pivot, tip, min_pt, mid_pt, max_pt, center = p_kpts

            # ====================================================
            # 究極の計算ロジック
            # ====================================================
            # 1. 針のベクトルを計算し、Centerを始点に平行移動する
            needle_vector = tip - pivot
            shifted_tip = center + needle_vector
            
            # 2. 変換対象のポイントを作成 (新しいTipを使用し、Pivotは捨てる)
            # 構成: [Center(新Pivot), Shifted_Tip, Min, Mid, Max]
            target_pts = np.array([center, shifted_tip, min_pt, mid_pt, max_pt], dtype=np.float32)
            
            # 3. 射影変換にかけて「真円空間」へ歪みを直す
            t_kpts = apply_perspective_transform(inst['bbox'], target_pts)
            t_center, t_tip, t_min, t_mid, t_max = t_kpts

            # 4. 真円空間で、Centerを基準に角度（極座標）を計算する
            ang_min = calculate_angle_clockwise(t_center, t_min)
            ang_mid = calculate_angle_clockwise(t_center, t_mid)
            ang_max = calculate_angle_clockwise(t_center, t_max)
            ang_tip = calculate_angle_clockwise(t_center, t_tip)
            
            # 5. 比率の計算 (0%未満、100%超えにも対応)
            ratio = calculate_normalized_ratio(ang_min, ang_mid, ang_max, ang_tip)
            
            # ====================================================
            
            meta = next((item for item in img_meta if item["index"] == i), {"min": 0.0, "max": 1.0, "unit": ""})
            
            # 最終的な値の計算 (比率がマイナスならMinを下回り、1.0を超えればMaxを上回る)
            val = meta["min"] + (meta["max"] - meta["min"]) * ratio
            
            print(f"{img_name} - Inst {i+1}: {val:.2f} {meta['unit']} (Ratio: {ratio*100:.1f}%)")
            
            # --- 描画 ---
            raw_center = tuple(center.astype(int))
            raw_shifted_tip = tuple(shifted_tip.astype(int))
            
            # ① 実際の物理的な針 (青色)
            cv2.line(img, tuple(pivot.astype(int)), tuple(tip.astype(int)), (255, 100, 0), 2)
            
            # ② 計算に使った「Centerに平行移動した仮想の針」 (シアン)
            cv2.line(img, raw_center, raw_shifted_tip, (255, 255, 0), 2)
            
            # Centerポイント (赤丸)
            cv2.circle(img, raw_center, 5, (0, 0, 255), -1)
            
            # 値のテキスト表示
            cv2.putText(img, f"{val:.2f}{meta['unit']}", (raw_shifted_tip[0], raw_shifted_tip[1]-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Vector Validation", img)
        if cv2.waitKey(0) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()