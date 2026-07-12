import cv2
import os
import glob
import math
import numpy as np
from ultralytics import YOLO
from meter_config import META_DATA  # 秘伝のたれを読み込み

# ==========================================
# 設定項目
# ==========================================
IMAGE_DIR = r"..\real_images"  # 実写画像を入れたフォルダ
MODEL_PATH = r"best.pt"        # 学習済みの最強モデル

# ==========================================
# 角度・数値計算ロジック（バグ修正済みの最強版）
# ==========================================
def calculate_angle(pt, center):
    """2点間の角度を計算する"""
    return math.degrees(math.atan2(pt[1] - center[1], pt[0] - center[0]))

def calculate_meter_value(kpts, meta):
    """6つのキーポイントから数値を計算する"""
    pivot = kpts[0]
    tip = kpts[1]
    min_pt = kpts[2]
    max_pt = kpts[4]

    angle_min = calculate_angle(min_pt, pivot)
    angle_max = calculate_angle(max_pt, pivot)
    angle_tip = calculate_angle(tip, pivot)

    # Minを0度とした相対角度（0〜360）に正規化（バグ回避）
    norm_max = (angle_max - angle_min) % 360.0
    norm_tip = (angle_tip - angle_min) % 360.0

    # 範囲外にはみ出た場合のクリッピング処理
    if norm_tip > norm_max:
        if 360.0 - norm_tip < norm_tip - norm_max:
            norm_tip = 0.0
        else:
            norm_tip = norm_max
            
    ratio = norm_tip / norm_max if norm_max > 0 else 0
    val = meta["min"] + ratio * (meta["max"] - meta["min"])
    return val, ratio

# ==========================================
# メイン処理
# ==========================================
def main():
    print("=== 実環境(Sim2Real) メーター読み取りテスト開始 ===")
    
    model = YOLO(MODEL_PATH)
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    if not image_paths:
        print(f"エラー: {IMAGE_DIR} に画像が見つかりません。")
        return

    cv2.namedWindow("Real Meter Detection", cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue

        print(f"\n--- {img_name} を推論中 ---")
        
        # 推論の実行 (conf=0.5 で自信のある予測だけを残す)
        # ハードルを下げて「25%でも可能性があるなら全部出して！」と指示する
        results = model.predict(img, conf=0.25, verbose=False)
        result = results[0]

        if result.boxes is not None and result.keypoints is not None:
            cls_ids = result.boxes.cls.cpu().numpy()
            bboxes = result.boxes.xyxy.cpu().numpy()
            kpts_array = result.keypoints.xy.cpu().numpy() # ピクセル座標

            for cls_id, bbox, kpts in zip(cls_ids, bboxes, kpts_array):
                cls_id = int(cls_id)
                if cls_id not in META_DATA: continue
                if len(kpts) < 6 or np.all(kpts == 0): continue

                meta = META_DATA[cls_id]
                
                # 数値の計算
                val, ratio = calculate_meter_value(kpts, meta)
                unit = meta["unit"]
                
                print(f"検出: {meta['name']} -> 【 {val:.2f} {unit} 】(針の位置: {ratio*100:.1f}%)")

                # ==========================
                # 画面への描画処理
                # ==========================
                x1, y1, x2, y2 = map(int, bbox)
                
                # BBoxを描画 (緑色)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 背景付きのテキストを描画（見やすくするため）
                text = f"{val:.2f} {unit}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(img, text, (x1, y1 - 5), font, 1.0, (0, 0, 0), 2)
                
                # 針の線を描画 (Pivot -> Tip を青線で)
                pivot = tuple(map(int, kpts[0]))
                tip = tuple(map(int, kpts[1]))
                cv2.line(img, pivot, tip, (255, 100, 0), 3)
                
                # 6つのキーポイントを赤丸で描画
                for pt in kpts:
                    cv2.circle(img, tuple(map(int, pt)), 6, (0, 0, 255), -1)

        else:
            print("メーターが検出されませんでした。")

        # 結果を画面に表示
        cv2.imshow("Real Meter Detection", img)
        
        print("操作: [Enter] 次の画像へ / [q] 終了")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("すべてのテストが完了しました。")

if __name__ == "__main__":
    main()