import cv2
import os
import glob
import numpy as np
import math

# ディレクトリ設定
IMAGE_DIR = r"..\TargetImages"
LABEL_DIR = r"..\TargetLabels"

# 状態管理用
instances = []
dragging_info = None 
phase = 0            
calibration_target = None
is_drawing_roi = False
roi_start = (0, 0)
roi_end = (0, 0)
mouse_x, mouse_y = 0, 0
DRAG_RADIUS = 10 

# ルーペ設定
ZOOM_FACTOR = 3
LOUPE_SIZE = 240 

def read_yolo_format(txt_path, img_w, img_h):
    loaded_instances = []
    if not os.path.exists(txt_path): return loaded_instances
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 20: continue 
            
            xc, yc, w, h = map(float, parts[1:5])
            x1, y1 = int((xc - w / 2) * img_w), int((yc - h / 2) * img_h)
            x2, y2 = int((xc + w / 2) * img_w), int((yc + h / 2) * img_h)
            
            kpts = []
            num_points = (len(parts) - 5) // 3
            for i in range(min(num_points, 6)): 
                px = int(float(parts[5 + i*3]) * img_w)
                py = int(float(parts[6 + i*3]) * img_h)
                kpts.append([px, py])
                
            if len(kpts) == 5:
                kpts.append([int(xc * img_w), int(yc * img_h)])
                
            loaded_instances.append({'bbox': [x1, y1, x2, y2], 'keypoints': kpts})
    return loaded_instances

def save_yolo_format(txt_path, img_w, img_h, instances_data):
    yolo_lines = []
    for inst in instances_data:
        b = inst['bbox']
        kpts = inst['keypoints']
        x_center = ((b[0] + b[2]) / 2) / img_w
        y_center = ((b[1] + b[3]) / 2) / img_h
        bbox_w = (b[2] - b[0]) / img_w
        bbox_h = (b[3] - b[1]) / img_h
        
        line = f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
        for pt in kpts: 
            px = pt[0] / img_w
            py = pt[1] / img_h
            line += f" {px:.6f} {py:.6f} 2"
        yolo_lines.append(line)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines))

def get_closest_keypoint(x, y):
    min_dist = float('inf')
    target = None
    for i, inst in enumerate(instances):
        for j, pt in enumerate(inst['keypoints']):
            if j == 5: continue # Center(5) は保護
            dist = math.hypot(x - pt[0], y - pt[1])
            if dist < DRAG_RADIUS and dist < min_dist:
                min_dist = dist
                target = (i, j)
    return target

def auto_calibrate(img, roi_rect, kpt_idx):
    x1, y1 = min(roi_rect[0][0], roi_rect[1][0]), min(roi_rect[0][1], roi_rect[1][1])
    x2, y2 = max(roi_rect[0][0], roi_rect[1][0]), max(roi_rect[0][1], roi_rect[1][1])
    
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
        
    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if kpt_idx == 0: 
        # === Pivot: 円検出 または 全体重心 ===
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=50, param2=20, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            roi_cx, roi_cy = (x2-x1)//2, (y2-y1)//2
            best_circle = min(circles, key=lambda c: (c[0]-roi_cx)**2 + (c[1]-roi_cy)**2)
            return (x1 + best_circle[0], y1 + best_circle[1])
            
        M = cv2.moments(mask)
        if M["m00"] != 0:
            return (x1 + int(M["m10"] / M["m00"]), y1 + int(M["m01"] / M["m00"]))
            
    else: 
        # === Tip, Min, Mid, Max: 凸包(Convex Hull)による先端/エッジ角検出 ===
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea) # 最大の領域(針や目盛り)を取得
            hull = cv2.convexHull(c)               # 外枠の頂点(角)を取得
            
            valid_pts = []
            h_roi, w_roi = mask.shape
            margin = 4 # ROIの境界から何ピクセル以内を「枠に触れている根元」とみなすか
            
            for pt in hull:
                cx, cy = pt[0]
                # ROIの境界付近にある点（切り取られた根元）は除外する
                if margin < cx < w_roi - margin and margin < cy < h_roi - margin:
                    valid_pts.append([cx, cy])
            
            if len(valid_pts) > 0:
                # 境界に触れていない「内部の先端の角」の平均をとる
                valid_pts = np.array(valid_pts)
                avg_x = np.mean(valid_pts[:, 0])
                avg_y = np.mean(valid_pts[:, 1])
                return (x1 + int(avg_x), y1 + int(avg_y))
        
        # フォールバック（輪郭が取れなかった場合は重心）
        M = cv2.moments(mask)
        if M["m00"] != 0:
            return (x1 + int(M["m10"] / M["m00"]), y1 + int(M["m01"] / M["m00"]))
            
    return None

def mouse_callback(event, x, y, flags, param):
    global dragging_info, phase, calibration_target, is_drawing_roi, roi_start, roi_end, mouse_x, mouse_y, instances
    img_h, img_w, orig_img = param
    mouse_x, mouse_y = max(0, min(x, img_w - 1)), max(0, min(y, img_h - 1))
    
    target_names = ["Pivot", "Tip", "Min", "Mid", "Max", "Center"]

    if event == cv2.EVENT_RBUTTONDOWN:
        if phase == 0:
            target = get_closest_keypoint(mouse_x, mouse_y)
            if target is not None:
                calibration_target = target
                phase = 1
                is_drawing_roi = False
                name = target_names[target[1]]
                print(f" -> [{name}] を選択しました。左ドラッグで補正範囲(ROI)を囲んでください。 (右クリックでキャンセル)")
        elif phase == 1:
            phase = 0
            calibration_target = None
            is_drawing_roi = False
            print(" -> 補正選択をキャンセルしました。")

    elif event == cv2.EVENT_LBUTTONDOWN:
        if phase == 0:
            dragging_info = get_closest_keypoint(mouse_x, mouse_y)
        elif phase == 1:
            is_drawing_roi = True
            roi_start = (mouse_x, mouse_y)
            roi_end = (mouse_x, mouse_y)
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_info is not None:
            instances[dragging_info[0]]['keypoints'][dragging_info[1]] = [mouse_x, mouse_y]
        elif is_drawing_roi:
            roi_end = (mouse_x, mouse_y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        if dragging_info is not None:
            dragging_info = None
        elif is_drawing_roi:
            is_drawing_roi = False
            if calibration_target is not None:
                inst_idx, kpt_idx = calibration_target
                new_pt = auto_calibrate(orig_img, (roi_start, roi_end), kpt_idx)
                if new_pt is not None:
                    instances[inst_idx]['keypoints'][kpt_idx] = list(new_pt)
                    print(f"    => {target_names[kpt_idx]} をエッジ先端に自動補正しました: {new_pt}")
                else:
                    print(f"    => 範囲が狭すぎるか、対象が見つかりませんでした。")
            
            phase = 0
            calibration_target = None

def create_loupe_image(img, x, y):
    h, w = img.shape[:2]
    crop_size = LOUPE_SIZE // ZOOM_FACTOR
    half_crop = crop_size // 2
    x_min, x_max = max(0, x - half_crop), min(w, x + half_crop)
    y_min, y_max = max(0, y - half_crop), min(h, y + half_crop)
    crop = img[y_min:y_max, x_min:x_max]
    
    pad_t, pad_b = half_crop - (y - y_min), half_crop - (y_max - y)
    pad_l, pad_r = half_crop - (x - x_min), half_crop - (x_max - x)
    crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    loupe = cv2.resize(crop, (LOUPE_SIZE, LOUPE_SIZE), interpolation=cv2.INTER_NEAREST)
    center = LOUPE_SIZE // 2
    cv2.line(loupe, (center, 0), (center, LOUPE_SIZE), (255, 255, 255), 1)
    cv2.line(loupe, (0, center), (LOUPE_SIZE, center), (255, 255, 255), 1)
    cv2.rectangle(loupe, (0, 0), (LOUPE_SIZE-1, LOUPE_SIZE-1), (0, 255, 255), 2)
    return loupe

def main():
    global instances
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    print("\n=== キャリブレーションツール (先端エッジ検出UI版) ===")
    cv2.namedWindow("Calibrate Tool", cv2.WINDOW_NORMAL)
    
    img_idx = 0
    while img_idx < len(image_paths):
        img_path = image_paths[img_idx]
        img_name = os.path.basename(img_path)
        txt_path = os.path.join(LABEL_DIR, f"{os.path.splitext(img_name)[0]}.txt")
        original_img = cv2.imread(img_path)
        if original_img is None: 
            img_idx += 1
            continue
            
        img_h, img_w = original_img.shape[:2]
        cv2.setMouseCallback("Calibrate Tool", mouse_callback, param=(img_h, img_w, original_img))
        instances = read_yolo_format(txt_path, img_w, img_h)
        
        print(f"\n==================================================")
        print(f" 画像 [{img_idx+1}/{len(image_paths)}] : {img_name}")
        print(f" 凡例 : [青]Pivot | [水色]Tip | [緑]Min | [オレンジ]Mid | [紫]Max | [赤]Center(固定)")
        print(f" 操作 : [L-Drag]手動移動 | [R-Click]自動補正対象の選択 | [d]保存 | [a]戻る | [q]終了")
        print(f"==================================================")
        
        while True:
            display_img = original_img.copy()
            for inst_idx, inst in enumerate(instances):
                kpts = inst['keypoints']
                if len(kpts) == 6: 
                    # 結線描画
                    cv2.line(display_img, tuple(kpts[0]), tuple(kpts[1]), (255, 100, 0), 2)
                    cv2.line(display_img, tuple(kpts[5]), tuple(kpts[2]), (0, 255, 0), 1)
                    cv2.line(display_img, tuple(kpts[5]), tuple(kpts[4]), (0, 255, 0), 1)
                    cv2.line(display_img, tuple(kpts[2]), tuple(kpts[3]), (0, 255, 255), 1)
                    cv2.line(display_img, tuple(kpts[3]), tuple(kpts[4]), (0, 255, 255), 1)
                    
                    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 165, 255), (255, 0, 255), (0, 0, 255)]
                    
                    for i, pt in enumerate(kpts):
                        cv2.drawMarker(display_img, tuple(pt), colors[i], cv2.MARKER_CROSS, 12, 2)
                        # 選択中の対象を黄色い枠で目立たせる
                        if phase == 1 and calibration_target == (inst_idx, i):
                            cv2.circle(display_img, tuple(pt), 14, (0, 255, 255), 2)
                            cv2.circle(display_img, tuple(pt), 18, (0, 0, 0), 1)

            # ROI描画中の白枠
            if is_drawing_roi:
                x1, y1 = roi_start
                x2, y2 = mouse_x, mouse_y
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # ルーペ描画
            loupe = create_loupe_image(display_img, mouse_x, mouse_y)
            if img_w >= LOUPE_SIZE and img_h >= LOUPE_SIZE:
                if mouse_x > img_w // 2:
                    display_img[0:LOUPE_SIZE, 0:LOUPE_SIZE] = loupe
                else:
                    display_img[0:LOUPE_SIZE, img_w - LOUPE_SIZE:img_w] = loupe
                
            cv2.imshow("Calibrate Tool", display_img)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): return
            elif key == ord('d') or key == 13:
                save_yolo_format(txt_path, img_w, img_h, instances)
                print(" -> 保存しました。")
                img_idx += 1
                break
            elif key == ord('a'):
                if img_idx > 0: img_idx -= 1
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()