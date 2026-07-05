import cv2
import os
import glob
import math
import numpy as np

# ディレクトリ設定 (srcディレクトリから見た相対パス)
IMAGE_DIR = r"..\TargetImages"
LABEL_DIR = r"..\TargetLabels"

# 保存先ディレクトリの作成
if not os.path.exists(LABEL_DIR):
    os.makedirs(LABEL_DIR)

# 状態管理用
phase = 0 # 0: BBox調整モード, 1: キーポイント入力モード
bbox = [0, 0, 0, 0] # [x1, y1, x2, y2]
dragging_idx = -1 # ドラッグ中の角 (0:左上, 1:右上, 2:右下, 3:左下)
DRAG_RADIUS = 15 # ドラッグ判定の半径

# ルーペ設定
CROP_SIZE = 30           # 切り取るサイズ (30ピクセル四方)
LOUPE_DISPLAY_SIZE = 180 # 拡大して表示するサイズ (6倍拡大)
mouse_x, mouse_y = 0, 0  # マウス座標のトラッキング用

current_points = []
instances = []

# 6点のラベル定義
point_labels = [
    "1. Pivot (Needle Axis)", 
    "2. Tip (Needle Tip)", 
    "3. Min (Scale 0)", 
    "4. Mid (Scale Middle)",
    "5. Max (Scale MAX)",
    "6. Center (Scale Center)"
]

def get_corner_idx(x, y):
    """クリックされた座標がどの角に近いかを判定"""
    corners = [
        (bbox[0], bbox[1]), # 左上
        (bbox[2], bbox[1]), # 右上
        (bbox[2], bbox[3]), # 右下
        (bbox[0], bbox[3])  # 左下
    ]
    for i, (cx, cy) in enumerate(corners):
        if math.hypot(x - cx, y - cy) < DRAG_RADIUS * 2: # 判定を少し甘めに
            return i
    return -1

def normalize_bbox(b):
    """ドラッグで反転した場合に備えて座標を正規化"""
    return [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]

def mouse_callback(event, x, y, flags, param):
    global bbox, dragging_idx, phase, current_points, mouse_x, mouse_y
    img_h, img_w = param

    # 画面外へのドラッグを防ぐ
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    
    # マウス座標を常に更新（ルーペ表示用）
    mouse_x, mouse_y = x, y

    if phase == 0: # BBox調整モード
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging_idx = get_corner_idx(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_idx != -1:
                if dragging_idx == 0:   # 左上
                    bbox[0], bbox[1] = x, y
                elif dragging_idx == 1: # 右上
                    bbox[2], bbox[1] = x, y
                elif dragging_idx == 2: # 右下
                    bbox[2], bbox[3] = x, y
                elif dragging_idx == 3: # 左下
                    bbox[0], bbox[3] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx = -1

    elif phase == 1: # キーポイント入力モード
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_points) < 6:
                current_points.append((x, y))
                print(f"Point {len(current_points)}: {point_labels[len(current_points)-1]} recorded at ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右クリックで1つ前の点を取り消し (Undo)
            if len(current_points) > 0:
                current_points.pop()
                print("Undo: 最後のポイントを取り消しました。")

def create_loupe_image(img, x, y):
    """指定された座標の周辺30ピクセルを切り取って拡大する"""
    h, w = img.shape[:2]
    half_crop = CROP_SIZE // 2
    
    x_min, x_max = max(0, x - half_crop), min(w, x + half_crop)
    y_min, y_max = max(0, y - half_crop), min(h, y + half_crop)
    crop = img[y_min:y_max, x_min:x_max]
    
    # 画像の端の処理 (足りないピクセルを黒でパディング)
    pad_t, pad_b = half_crop - (y - y_min), half_crop - (y_max - y)
    pad_l, pad_r = half_crop - (x - x_min), half_crop - (x_max - x)
    crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # ぼやけないように、ニアレストネイバー(ピクセル単位)で拡大
    loupe = cv2.resize(crop, (LOUPE_DISPLAY_SIZE, LOUPE_DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # ルーペの中心に照準器（白いクロスヘア）を描画
    center = LOUPE_DISPLAY_SIZE // 2
    cv2.line(loupe, (center, 0), (center, LOUPE_DISPLAY_SIZE), (255, 255, 255), 1)
    cv2.line(loupe, (0, center), (LOUPE_DISPLAY_SIZE, center), (255, 255, 255), 1)
    
    # 外枠
    cv2.rectangle(loupe, (0, 0), (LOUPE_DISPLAY_SIZE-1, LOUPE_DISPLAY_SIZE-1), (0, 255, 255), 2)
    return loupe

def convert_to_yolo_format(img_width, img_height, instances):
    yolo_lines = []
    for inst in instances:
        b = inst['bbox']
        kpts = inst['keypoints']
        
        x_center = ((b[0] + b[2]) / 2) / img_width
        y_center = ((b[1] + b[3]) / 2) / img_height
        bbox_w = (b[2] - b[0]) / img_width
        bbox_h = (b[3] - b[1]) / img_height
        
        line = f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
        
        for pt in kpts:
            px = pt[0] / img_width
            py = pt[1] / img_height
            line += f" {px:.6f} {py:.6f} 2"
            
        yolo_lines.append(line)
    return yolo_lines

def main():
    global current_points, instances, phase, bbox
    
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
        
    if not image_paths:
        print(f"エラー: {IMAGE_DIR} に画像が見つかりません。")
        return

    print("=== アノテーションツール (ルーペの文字映り込み解消版) 起動 ===")
    cv2.namedWindow("Annotation Tool", cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
        
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue
            
        img_h, img_w = original_img.shape[:2]

        # 既存データのチェックとユーザーへの選択プロンプト
        if os.path.exists(txt_path):
            check_img = original_img.copy()
            overlay = check_img.copy()
            cv2.rectangle(overlay, (0, 0), (img_w, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, check_img, 0.4, 0, check_img)
            
            cv2.putText(check_img, "Label already exists!", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(check_img, "Press 'o' to Overwrite (Redo) | 's' to Skip | 'q' to Quit", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(check_img, f"File: {img_name}", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Annotation Tool", check_img)
            
            valid_key_pressed = False
            skip_image = False
            while not valid_key_pressed:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    print(f"スキップしました: {img_name}")
                    skip_image = True
                    valid_key_pressed = True
                elif key == ord('o'):
                    print(f"上書きモードで開きます: {img_name}")
                    valid_key_pressed = True
                elif key == ord('q'):
                    print("終了します。")
                    cv2.destroyAllWindows()
                    return
            
            if skip_image:
                continue 
        
        # 初期状態のリセット
        instances = []
        current_points = []
        phase = 0
        
        # 初期BBoxを画面の80%のサイズで中央に配置
        pad_x = int(img_w * 0.1)
        pad_y = int(img_h * 0.1)
        bbox = [pad_x, pad_y, img_w - pad_x, img_h - pad_y]
        
        cv2.setMouseCallback("Annotation Tool", mouse_callback, param=(img_h, img_w))
        
        while True:
            display_img = original_img.copy()
            
            # 確定済みのインスタンスを描画
            for i, inst in enumerate(instances):
                b = inst['bbox']
                cv2.rectangle(display_img, (b[0], b[1]), (b[2], b[3]), (0, 150, 0), 2)
                for pt in inst['keypoints']:
                    cv2.drawMarker(display_img, pt, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
                cv2.putText(display_img, f"Inst {i+1} Saved", (b[0], b[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

            # ガイドテキストの内容を設定
            guide_text = ""
            status_text = ""
            
            if phase == 0:
                # BBox調整モードの描画
                cv2.rectangle(display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                corners = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
                for pt in corners:
                    cv2.circle(display_img, pt, DRAG_RADIUS, (0, 255, 255), -1)
                    cv2.circle(display_img, pt, DRAG_RADIUS, (0, 0, 0), 1)
                
                guide_text = "[Phase 1] Drag corners to fit -> Press ENTER"
                status_text = "Press 'd' to skip this image, 'q' to quit."
                
            elif phase == 1:
                # キーポイント入力モードの描画
                norm_bbox = normalize_bbox(bbox)
                cv2.rectangle(display_img, (norm_bbox[0], norm_bbox[1]), (norm_bbox[2], norm_bbox[3]), (0, 255, 0), 2)
                
                for i, pt in enumerate(current_points):
                    cv2.drawMarker(display_img, pt, (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(display_img, str(i+1), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if len(current_points) < 6:
                    guide_text = f"[Phase 2] Click: {point_labels[len(current_points)]}"
                    status_text = "R-Click: Undo | 'ESC': Back to BBox"
                else:
                    guide_text = "Instance complete!"
                    status_text = "Press 'a' to add needle | 'd' to save & next"
            
            # -----------------------------------------------------------------
            # 【重要】テキストを描画する「前」にルーペ画像を切り取る
            # これにより、UIの文字がルーペ内に映り込まなくなります
            # -----------------------------------------------------------------
            loupe = create_loupe_image(display_img, mouse_x, mouse_y)
            
            # --- テキストの右下固定描画処理 ---
            texts = [
                (guide_text, (100, 100, 255)), # 少し明るい赤
                (status_text, (255, 255, 255)), # 白
                (f"File: {img_name} | Instances: {len(instances)}", (255, 255, 255))
            ]
            
            y_offset = img_h - 20
            # 下から上に向かってテキストを描画
            for text, color in reversed(texts):
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # 黒い背景（座布団）を描画して視認性を確保
                cv2.rectangle(display_img, (img_w - tw - 25, y_offset - th - 5), (img_w - 15, y_offset + 5), (0, 0, 0), -1)
                # テキストを描画
                cv2.putText(display_img, text, (img_w - tw - 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= (th + 15)

            # --- ルーペの合成処理（一番上に乗せる） ---
            if img_w >= LOUPE_DISPLAY_SIZE and img_h >= LOUPE_DISPLAY_SIZE:
                # ユーザーのクリックを邪魔しないように、マウス位置の反対側にルーペを逃がす
                if mouse_x > img_w // 2:
                    display_img[0:LOUPE_DISPLAY_SIZE, 0:LOUPE_DISPLAY_SIZE] = loupe
                else:
                    display_img[0:LOUPE_DISPLAY_SIZE, img_w - LOUPE_DISPLAY_SIZE:img_w] = loupe

            cv2.imshow("Annotation Tool", display_img)
            
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                print("終了します。")
                cv2.destroyAllWindows()
                return
            elif phase == 0:
                if key == 13 or key == 32: 
                    bbox = normalize_bbox(bbox)
                    phase = 1
                    current_points = []
                    print("BBoxを確定しました。キーポイントを入力してください。")
                elif key == ord('d'): 
                    print(f"スキップ: {img_name}")
                    break
            elif phase == 1:
                if key == 27: 
                    phase = 0
                    current_points = []
                    print("BBox調整モードに戻りました。")
                elif key == ord('r'):
                    current_points = []
                    print("ポイントをリセットしました。")
                elif key == ord('a'):
                    if len(current_points) == 6:
                        instances.append({'bbox': bbox.copy(), 'keypoints': current_points.copy()})
                        current_points = []
                        phase = 0 
                        print("インスタンスを追加しました！次の針の枠を調整してください。")
                    else:
                        print("6点すべてクリックされていません。")
                elif key == ord('d'):
                    if len(current_points) == 6:
                        instances.append({'bbox': bbox.copy(), 'keypoints': current_points.copy()})
                    
                    if instances:
                        yolo_data = convert_to_yolo_format(img_w, img_h, instances)
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write("\n".join(yolo_data))
                        print(f"保存完了: {txt_path}")
                    else:
                        print(f"スキップ: {img_name}")
                    break

    cv2.destroyAllWindows()
    print("すべての画像の処理が完了しました。")

if __name__ == "__main__":
    main()