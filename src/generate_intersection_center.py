import cv2
import os
import glob
import numpy as np

# ディレクトリ設定
IMAGE_DIR = r"..\TargetImages"
LABEL_DIR = r"..\TargetLabels"

# グローバル変数
line_points = []         # クリックした点のリスト（2点で1本の線を定義）
mouse_x, mouse_y = 0, 0  # マウス座標のトラッキング用

# ルーペ設定
CROP_SIZE = 30           # 切り取るサイズ (30ピクセル四方)
LOUPE_DISPLAY_SIZE = 180 # 拡大して表示するサイズ (6倍拡大)

def calculate_intersection(points):
    """
    複数の線分（2点のペア）を延長し、それらが最も交わる点（最小二乗法）を計算する
    """
    lines = []
    for i in range(0, len(points) - 1, 2):
        p1 = np.array(points[i], dtype=np.float64)
        p2 = np.array(points[i+1], dtype=np.float64)
        
        # 点が同じ場所にある場合はスキップ
        if np.linalg.norm(p1 - p2) < 1e-5:
            continue
            
        # 直線の方程式: ax + by + c = 0
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]
        
        # 距離計算の精度を上げるために正規化 (a^2 + b^2 = 1)
        length = np.hypot(a, b)
        lines.append([a / length, b / length, c / length])

    if len(lines) < 2:
        return None

    # 行列を作成して最小二乗法で交点 (X, Y) を解く
    # A * [X, Y]^T = B  =>  a*X + b*Y = -c
    A = []
    B = []
    for a, b, c in lines:
        A.append([a, b])
        B.append([-c])

    A = np.array(A)
    B = np.array(B)
    
    # lstsq (Least Squares) を使用
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    intersect_x = int(round(res[0][0]))
    intersect_y = int(round(res[1][0]))
    
    return (intersect_x, intersect_y)

def mouse_callback(event, x, y, flags, param):
    global line_points, mouse_x, mouse_y
    img_h, img_w = param
    
    # 画面外へのドラッグ・移動を防ぐ
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    
    # マウス座標を常に更新（ルーペ表示用）
    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左クリックで点を追加
        line_points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右クリックで最後の点を削除（Undo）
        if len(line_points) > 0:
            line_points.pop()

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
    global line_points, mouse_x, mouse_y
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
    
    image_paths = sorted(list(set(image_paths)))
    
    print("=== 手動・目盛り延長線交点ツール (スキップ確認機能付き) ===")
    cv2.namedWindow("Intersection Center Tool", cv2.WINDOW_NORMAL)

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
            
            cv2.imshow("Intersection Center Tool", check_img)
            
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
        cv2.setMouseCallback("Intersection Center Tool", mouse_callback, param=(img_h, img_w))
        
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

            line_points = [] # 次の計器のためにリセット
            
            while True:
                display_img = original_img.copy()
                
                # 1. 既存の5点(Pivot, Tip, Min, Mid, Max)を控えめに描画
                for i in range(5):
                    px = int(float(parts[5 + i*3]) * img_w)
                    py = int(float(parts[6 + i*3]) * img_h)
                    cv2.drawMarker(display_img, (px, py), (150, 150, 150), cv2.MARKER_CROSS, 10, 1)

                # 既存の中心点があれば白のクロスで描画
                if cx_exist is not None:
                    cv2.drawMarker(display_img, (cx_exist, cy_exist), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(display_img, "Saved Center", (cx_exist+10, cy_exist-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 2. ユーザーが打った線を描画
                for i in range(0, len(line_points), 2):
                    if i + 1 < len(line_points):
                        # 2点揃っている場合（確定した線）
                        p1 = tuple(line_points[i])
                        p2 = tuple(line_points[i+1])
                        cv2.line(display_img, p1, p2, (0, 255, 0), 2)
                        cv2.drawMarker(display_img, p1, (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
                        cv2.drawMarker(display_img, p2, (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
                    else:
                        # 1点目だけ打って、2点目を待っている状態（マウスに追従するガイド線）
                        p1 = tuple(line_points[i])
                        cv2.drawMarker(display_img, p1, (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
                        cv2.line(display_img, p1, (mouse_x, mouse_y), (0, 255, 255), 1)

                # 3. 直線が2本以上（点が4個以上）あれば交点を計算して描画
                fitted_center = None
                if len(line_points) >= 4:
                    fitted_center = calculate_intersection(line_points)
                    
                    if fitted_center is not None:
                        # 各線分から交点に向かう「延長線」をシアン色で可視化
                        for i in range(0, len(line_points) - 1, 2):
                            p1 = line_points[i]
                            p2 = line_points[i+1]
                            mid_x = int((p1[0] + p2[0]) / 2)
                            mid_y = int((p1[1] + p2[1]) / 2)
                            cv2.line(display_img, (mid_x, mid_y), fitted_center, (255, 255, 0), 1, cv2.LINE_AA)

                        # 交点(Center)を赤色のクロスで描画
                        cv2.drawMarker(display_img, fitted_center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        cv2.putText(display_img, "New Center", (fitted_center[0]+15, fitted_center[1]-15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # -----------------------------------------------------------------
                # テキストを描画する「前」にルーペ画像を切り取る
                # -----------------------------------------------------------------
                loupe = create_loupe_image(display_img, mouse_x, mouse_y)

                # 4. 右下固定テキストの描画処理
                guide_text = "L-Click: Draw Lines (2 pts/line) | R-Click: Undo | [d]: Save"
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

                cv2.imshow("Intersection Center Tool", display_img)
                key = cv2.waitKey(15) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('c'):
                    line_points = [] # クリア
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