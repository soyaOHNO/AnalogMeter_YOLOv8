import os
import glob
import math
from ultralytics import YOLO

# ==========================================
# 設定項目
# ==========================================
IMAGE_DIR = r"..\yolo_test_data_all\images"
LABEL_DIR = r"..\yolo_test_data_all\labels"
MODEL_PATH = r"best.pt"  # 学習済みの最強モデル

# 秘伝のたれ
from meter_config import META_DATA # メタデータのパス

# ==========================================
# 角度・数値計算ロジック（バグ修正済）
# ==========================================
def calculate_angle(pt, center):
    """2点間の角度を計算する（画像座標系）"""
    return math.degrees(math.atan2(pt[1] - center[1], pt[0] - center[0]))

def calculate_meter_value(kpts, meta):
    """6つのキーポイントから最終的な数値を計算する"""
    # kpts: [Pivot(0), Tip(1), Min(2), Mid(3), Max(4), Center(5)]
    pivot = kpts[0]
    tip = kpts[1]
    min_pt = kpts[2]
    max_pt = kpts[4]

    angle_min = calculate_angle(min_pt, pivot)
    angle_max = calculate_angle(max_pt, pivot)
    angle_tip = calculate_angle(tip, pivot)

    # 【重要】0度またぎのバグを防ぐため、Minを0度とした相対角度（0〜360）に正規化する
    norm_max = (angle_max - angle_min) % 360.0
    norm_tip = (angle_tip - angle_min) % 360.0

    # 針がMin以下、またはMax以上にはみ出た場合のクリッピング処理
    if norm_tip > norm_max:
        if 360.0 - norm_tip < norm_tip - norm_max:
            norm_tip = 0.0  # Min寄りの場合は0にクリップ
        else:
            norm_tip = norm_max  # Max寄りの場合はMaxにクリップ
            
    ratio = norm_tip / norm_max if norm_max > 0 else 0
    return meta["min"] + ratio * (meta["max"] - meta["min"])

# ==========================================
# GT（正解）データの読み込み
# ==========================================
def read_gt_labels(txt_path):
    gt_data = []
    if not os.path.exists(txt_path):
        return gt_data
        
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 23: continue
            
            cls_id = int(parts[0])
            # kptsは [x, y, visibility] の並び
            kpt_raw = list(map(float, parts[5:]))
            kpts = [(kpt_raw[i*3], kpt_raw[i*3+1]) for i in range(6)]
            gt_data.append({"class_id": cls_id, "kpts": kpts})
    return gt_data

# ==========================================
# メイン処理
# ==========================================
def main():
    print("=== YOLOv8 誤差(Error)検証開始 ===")
    
    model = YOLO(MODEL_PATH)
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    
    if not image_paths:
        print("画像が見つかりません。")
        return

    # 50枚だけでテスト
    image_paths = image_paths[:50]
    
    total_error = 0.0
    total_count = 0
    max_error = 0.0
    max_error_file = ""

    print("-" * 75)
    print(f"{'ファイル名':<20} | {'計器名':<15} | {'正解(GT)':<10} | {'予測(Pred)':<10} | {'誤差(Error)':<10}")
    print("-" * 75)

    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        txt_path = os.path.join(LABEL_DIR, base_name.replace(".jpg", ".txt"))
        
        gt_list = read_gt_labels(txt_path)
        if not gt_list:
            continue
            
        results = model.predict(img_path, verbose=False)
        pred_list = []
        
        if results[0].keypoints is not None:
            # 予測結果をリスト化
            cls_ids = results[0].boxes.cls.cpu().numpy()
            kpts_array = results[0].keypoints.xyn.cpu().numpy() # 正規化座標 (xyn) を使用
            
            for cls_id, kpts in zip(cls_ids, kpts_array):
                pred_list.append({"class_id": int(cls_id), "kpts": kpts})

        # 【重要】クラスIDが一致するもの同士だけを比較する
        for gt in gt_list:
            cls_id = gt["class_id"]
            if cls_id not in META_DATA: continue
            
            meta = META_DATA[cls_id]
            
            # 同じクラスIDの予測を探す
            matching_preds = [p for p in pred_list if p["class_id"] == cls_id]
            
            if not matching_preds:
                continue # AIが見逃した場合はスキップ
                
            pred = matching_preds[0] # 見つかった予測の1つ目を使用
            
            # 数値計算
            gt_val = calculate_meter_value(gt["kpts"], meta)
            pred_val = calculate_meter_value(pred["kpts"], meta)
            
            error = abs(gt_val - pred_val)
            
            total_error += error
            total_count += 1
            if error > max_error:
                max_error = error
                max_error_file = f"{base_name} ({meta['name']})"
                
            unit = meta["unit"]
            print(f"{base_name:<20} | {meta['name']:<15} | {gt_val:>6.2f} {unit:<3} | {pred_val:>6.2f} {unit:<3} | {error:>6.3f} {unit}")

    print("-" * 75)
    if total_count > 0:
        avg_error = total_error / total_count
        print(f"✅ 検証完了: {total_count} 個のメーターを照合しました。")
        print(f"📊 平均誤差 (MAE) : {avg_error:.4f} (単位は各計器に依存)")
        print(f"📈 最大誤差 (Max) : {max_error:.4f} -> 発生箇所: {max_error_file}")
    else:
        print("比較できるデータがありませんでした。")

if __name__ == "__main__":
    main()