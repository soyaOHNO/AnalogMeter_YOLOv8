import cv2
import os
import glob
import numpy as np

# ディレクトリ設定
IMAGE_DIR = r"..\TargetImages"
LABEL_DIR = r"..\TargetLabels"

# グローバル変数
ellipse_points = []
mouse_x, mouse_y = 0, 0  # マウス座標のトラッキング用

# ルーペ設定
CROP_SIZE = 30           # 切り取るサイズ (30ピクセル四方)
LOUPE_DISPLAY_SIZE = 180 # 拡大して表示するサイズ (6倍拡大)

def mouse_callback(event, x, y, flags, param):
    global ellipse_points, mouse_x, mouse_y
    img_h, img_w = param
    
    # 画面外へのドラッグ・移動を防ぐ
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    
    # マウス座標を常に更新（ルーペ表示用）
    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左クリックで点を追加
        ellipse_points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右クリックで最後の点を削除（Undo）
        if len(ellipse_points) > 0:
            ellipse_points.pop()

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

def read_yolo_format(txt_path):
    instances = []
    if not os.path.exists(txt_path): return instances
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # 最低でも5点(20要素)あるデータのみ対象
            if len(parts) >= 20:
                instances.append({'line_parts': parts})
    return instances

def save_yolo_format(txt_path, instances):
    lines = [" ".join(inst['line_parts']) for inst in instances]
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def main():
    global ellipse_points, mouse_x, mouse_y
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
    
    image_paths = sorted(list(set(image_paths)))
    
    print("=== 手動楕円フィッティングツール (スキップ確認機能付き) ===")
    cv2.namedWindow("Manual Ellipse Fitter", cv2.WINDOW_NORMAL)

    img_idx = 0
    while 0 <= img_idx < len(image_paths):
        img_path = image_paths[img_idx]
        img_name = os.path.basename(img_path)
        txt_path = os.path.join(LABEL_DIR, f"{os.path.splitext(img_name)[0]}.txt")
        
        original_img = cv2.imread(img_path)
        if original_img is None:
            img_idx += 1
            continue
            
        # まずテキストファイルを読み込む
        instances = read_yolo_format(txt_path)
        if not instances:
            # 5点のデータすら無い画像はスキップ
            img_idx += 1
            continue

        img_h, img_w = original_img.shape[:2]

        # ==========================================
        # 既存Centerのチェックとユーザーへの選択プロンプト
        # ==========================================
        # インスタンスの中に1つでも「23要素以上（＝Center計算済み）」のものがあるかチェック
        already_has_center = any(len(inst['line_parts']) >= 23 for inst in instances)
        
        if already_has_center:
            check_img = original_img.copy()
            overlay = check_img.copy()
            cv2.rectangle(overlay, (0, 0), (img_w, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, check_img, 0.4, 0, check_img)
            
            cv2.putText(check_img, "Center already exists!", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(check_img, "Press 'o' to Overwrite | 's' to Skip | 'q' to Quit", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(check_img, f"File: {img_name}", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Manual Ellipse Fitter", check_img)
            
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
                img_idx += 1
                continue 
        # ==========================================

        # マウスコールバックの登録（画像サイズを渡す）
        cv2.setMouseCallback("Manual Ellipse Fitter", mouse_callback, param=(img_h, img_w))
        
        # 1画像内の各計器(インスタンス)ごとに処理
        inst_idx = 0
        while inst_idx < len(instances):
            inst = instances[inst_idx]
            parts = inst['line_parts']
            
            # 既に保存されている中心点があれば取得
            cx_exist, cy_exist = None, None
            if len(parts) >= 23:
                cx_exist = int(float(parts[20]) * img_w)
                cy_exist = int(float(parts[21]) * img_h)

            ellipse_points = [] # 次の計器のためにリセット
            
            while True:
                display_img = original_img.copy()
                
                # 1. 既存の5点(Pivot, Tip, Min, Mid, Max)を控えめに描画 (薄いグレーのクロス)
                for i in range(5):
                    px = int(float(parts[5 + i*3]) * img_w)
                    py = int(float(parts[6 + i*3]) * img_h)
                    cv2.drawMarker(display_img, (px, py), (150, 150, 150), cv2.MARKER_CROSS, 10, 1)

                # 既存の中心点があれば白のクロスで描画
                if cx_exist is not None:
                    cv2.drawMarker(display_img, (cx_exist, cy_exist), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(display_img, "Saved Center", (cx_exist+10, cy_exist-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 2. ユーザーが打った点を緑色のクロスで描画
                for pt in ellipse_points:
                    cv2.drawMarker(display_img, tuple(pt), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

                # 3. 5点以上打たれたら、楕円を計算してリアルタイム描画
                fitted_center = None
                if len(ellipse_points) >= 5:
                    pts_array = np.array(ellipse_points, dtype=np.float32)
                    ellipse = cv2.fitEllipse(pts_array)
                    
                    # 楕円を黄色で描画
                    cv2.ellipse(display_img, ellipse, (0, 255, 255), 2)
                    
                    # 楕円の中心を赤色のクロスで描画
                    (cx, cy), axes, angle = ellipse
                    fitted_center = (int(cx), int(cy))
                    cv2.drawMarker(display_img, fitted_center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(display_img, "New Center", (fitted_center[0]+15, fitted_center[1]-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # -----------------------------------------------------------------
                # テキストを描画する「前」にルーペ画像を切り取る
                # -----------------------------------------------------------------
                loupe = create_loupe_image(display_img, mouse_x, mouse_y)

                # 4. 右下固定テキストの描画処理
                guide_text = "L-Click: Add | R-Click: Undo | [d]: Save | [c]: Clear"
                status_text = f"Image [{img_idx+1}/{len(image_paths)}] - Inst [{inst_idx+1}/{len(instances)}]"
                
                texts = [
                    (guide_text, (255, 255, 255)),   # 白
                    (status_text, (100, 100, 255)),  # 少し明るい赤
                ]
                
                y_offset = img_h - 20
                for text, color in reversed(texts):
                    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_img, (img_w - tw - 25, y_offset - th - 5), (img_w - 15, y_offset + 5), (0, 0, 0), -1)
                    cv2.putText(display_img, text, (img_w - tw - 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset -= (th + 15)

                # 5. ルーペの合成処理（一番上に乗せる）
                if img_w >= LOUPE_DISPLAY_SIZE and img_h >= LOUPE_DISPLAY_SIZE:
                    if mouse_x > img_w // 2:
                        display_img[0:LOUPE_DISPLAY_SIZE, 0:LOUPE_DISPLAY_SIZE] = loupe
                    else:
                        display_img[0:LOUPE_DISPLAY_SIZE, img_w - LOUPE_DISPLAY_SIZE:img_w] = loupe

                cv2.imshow("Manual Ellipse Fitter", display_img)
                key = cv2.waitKey(15) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('c'):
                    ellipse_points = [] # クリア
                elif key == ord('a'):
                    # 前に戻る処理
                    if inst_idx > 0:
                        inst_idx -= 1
                    else:
                        img_idx -= 1
                        inst_idx = -1
                    break
                elif key == ord('d') or key == 13: # 'd' または Enter
                    if fitted_center is not None:
                        # 新しい中心を正規化して保存
                        cx_norm = fitted_center[0] / img_w
                        cy_norm = fitted_center[1] / img_h
                        
                        if len(parts) >= 23:
                            # 既に6点目がある場合は上書き
                            parts[20] = f"{cx_norm:.6f}"
                            parts[21] = f"{cy_norm:.6f}"
                            parts[22] = "2"
                        else:
                            # 6点目として追加
                            parts.extend([f"{cx_norm:.6f}", f"{cy_norm:.6f}", "2"])
                        inst['line_parts'] = parts
                        
                    inst_idx += 1
                    break

            if inst_idx == -1:
                break # 前の画像へ戻るためのループ抜け

        # この画像の全計器の処理が終わったらファイルに書き込み
        if inst_idx == len(instances):
            save_yolo_format(txt_path, instances)
            img_idx += 1
        elif img_idx < 0:
            img_idx = 0

    cv2.destroyAllWindows()
    print("すべての作業が完了しました！")

if __name__ == "__main__":
    main()