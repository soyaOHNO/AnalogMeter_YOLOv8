import cv2
import os
import glob
import math
import numpy as np

# ==========================================
# 設定パス
# ==========================================
# Blenderで生成したテスト用10枚のデータがあるフォルダ
SYNTH_DIR = r"..\yolo_synth_data"
IMAGE_DIR = os.path.join(SYNTH_DIR, "images")
LABEL_DIR = os.path.join(SYNTH_DIR, "labels")

# ※このテストでは固定のメタデータを使用します
# （実際の計器に合わせてMin/Maxの数値を変更してください）
TEST_META = {"min": 0.0, "max": 1.0, "unit": "MPa"}

# ==========================================
# 幾何学計算ロジック (外挿・マイナス対応版)
# ==========================================
def read_yolo_format(txt_path, img_w, img_h):
    """合成データのテキストファイルからBBoxと6点を読み込む"""
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
                # 順序: [Pivot, Tip, Min, Mid, Max, Center]
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
    print("=== 合成データ(Sim2Real) 計算ロジック検証ツール ===")
    
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not image_paths:
        print(f"エラー: 画像が見つかりません -> {IMAGE_DIR}")
        return

    cv2.namedWindow("Synth Data Validation", cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        img_h, img_w = img.shape[:2]
        
        # テキストファイルから「完璧な」座標を読み込む
        instances = read_yolo_format(txt_path, img_w, img_h)

        if not instances:
            print(f"{img_name}: ラベルデータがありません。")
            continue

        for i, inst in enumerate(instances):
            p_kpts = inst['kpts'] # [Pivot, Tip, Min, Mid, Max, Center]
            pivot, tip, min_pt, mid_pt, max_pt, center = p_kpts

            # ====================================================
            # 値の計算
            # ====================================================
            needle_vector = tip - pivot
            shifted_tip = center + needle_vector
            
            target_pts = np.array([center, shifted_tip, min_pt, mid_pt, max_pt], dtype=np.float32)
            t_kpts = apply_perspective_transform(inst['bbox'], target_pts)
            t_center, t_tip, t_min, t_mid, t_max = t_kpts

            ang_min = calculate_angle_clockwise(t_center, t_min)
            ang_mid = calculate_angle_clockwise(t_center, t_mid)
            ang_max = calculate_angle_clockwise(t_center, t_max)
            ang_tip = calculate_angle_clockwise(t_center, t_tip)
            
            ratio = calculate_normalized_ratio(ang_min, ang_mid, ang_max, ang_tip)
            
            val = TEST_META["min"] + (TEST_META["max"] - TEST_META["min"]) * ratio
            
            print(f"{img_name}: 計算値 = {val:.3f} {TEST_META['unit']} (Ratio: {ratio*100:.1f}%)")

            # ====================================================
            # 描画処理
            # ====================================================
            x1, y1, x2, y2 = inst['bbox']
            raw_center = tuple(center.astype(int))
            raw_shifted_tip = tuple(shifted_tip.astype(int))
            
            # バウンディングボックス (緑)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 実際の針 (青)
            cv2.line(img, tuple(pivot.astype(int)), tuple(tip.astype(int)), (255, 100, 0), 2)
            
            # 平行移動した仮想の針 (シアン)
            cv2.line(img, raw_center, raw_shifted_tip, (255, 255, 0), 2)
            
            # 6つのキーポイント (赤丸)
            for pt in p_kpts:
                cv2.circle(img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
            
            # 計算値のテキスト表示
            cv2.putText(img, f"{val:.3f}{TEST_META['unit']}", (raw_shifted_tip[0], raw_shifted_tip[1]-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Synth Data Validation", img)
        # 何かキーを押すと次の画像へ
        if cv2.waitKey(0) & 0xFF == ord('q'): 
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()