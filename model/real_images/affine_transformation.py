import cv2
import numpy as np
import sys
import math

# ==========================================
# 設定項目
# ==========================================
INPUT_IMAGE = "7.jpg"        # 読み込む画像ファイル名
OUTPUT_IMAGE = "flat.jpg"    # 保存する画像ファイル名
OUTPUT_SIZE = 800            # 出力する画像のサイズ（正方形のピクセル数）
# ==========================================

# グローバル変数
pts = []
dragging_idx = -1
img_disp = None
img_original_resized = None

def mouse_callback(event, x, y, flags, param):
    global pts, dragging_idx

    # 左クリックを押したとき（近くの点を探して掴む）
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(pts):
            dist = math.hypot(pt[0] - x, pt[1] - y)
            if dist < 20:  # 20ピクセル以内なら吸い付くように掴む
                dragging_idx = i
                break

    # ドラッグ中（掴んでいる点の座標を更新）
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_idx != -1:
            pts[dragging_idx] = [x, y]

    # 左クリックを離したとき（掴むのをやめる）
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = -1

def main():
    global pts, img_disp, img_original_resized

    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"エラー: {INPUT_IMAGE} が見つかりません。")
        sys.exit()

    # 画面に収まるようにリサイズ
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    img_original_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    dh, dw = img_original_resized.shape[:2]

    # 初期状態の4点（計器の「上・右・下・左」を想定した十字配置）
    cx, cy = dw // 2, dh // 2
    r = min(dw, dh) // 3
    pts = [
        [cx, cy - r],  # 0: 上 (Top)
        [cx + r, cy],  # 1: 右 (Right)
        [cx, cy + r],  # 2: 下 (Bottom)
        [cx - r, cy]   # 3: 左 (Left)
    ]

    # 変換後の真円が内接する正方形における「上・右・下・左」のターゲット座標
    W = OUTPUT_SIZE
    dst_pts = np.array([
        [W / 2, 0],         # 上
        [W, W / 2],         # 右
        [W / 2, W],         # 下
        [0, W / 2]          # 左
    ], dtype=np.float32)

    # リアルタイムで楕円を描画するための真円ポイント群（100角形）
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_pts = np.column_stack((W / 2 + (W / 2) * np.cos(theta), W / 2 + (W / 2) * np.sin(theta))).astype(np.float32)
    circle_pts = np.array([circle_pts]) # perspectiveTransform用の配列形状に変換

    cv2.namedWindow("Drag & Drop - Ellipse Fit", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Drag & Drop - Ellipse Fit", mouse_callback)

    print("--- 使い方 ---")
    print("1. 点をドラッグし、計器盤の【12時・3時・6時・9時】のフチに合わせます。")
    print("2. リアルタイムで『緑色の楕円』が描画されるので、計器の輪郭にピッタリ合うように微調整してください。")
    print("3. 位置が完璧に決まったら『Enter』キーを押すと変換・保存されます。")
    print("4. 『q』キーを押すとキャンセルして終了します。")

    while True:
        img_disp = img_original_resized.copy()

        src_pts = np.array(pts, dtype=np.float32)
        
        try:
            # 4点から逆変換行列（出力用の真円 -> 入力画像の楕円）を計算
            M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
            
            # 真円の座標群を、現在の4点で作られる楕円の座標群に変換
            ellipse_pts = cv2.perspectiveTransform(circle_pts, M_inv)
            
            # リアルタイムで楕円のガイドラインを描画（緑色）
            cv2.polylines(img_disp, np.int32(ellipse_pts), isClosed=True, color=(0, 255, 0), thickness=2)
            
            # 十字の補助線（上と下、左と右を結ぶ線）
            cv2.line(img_disp, tuple(pts[0]), tuple(pts[2]), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img_disp, tuple(pts[1]), tuple(pts[3]), (255, 255, 255), 1, cv2.LINE_AA)

        except cv2.error:
            # 4点が一直線上にあるなど、行列計算が一時的に破綻した場合のクラッシュ回避
            pass

        # 4つの操作点（ドラッグ中は赤色になる）を描画
        labels = ["12 o'clock", "3 o'clock", "6 o'clock", "9 o'clock"]
        for i, pt in enumerate(pts):
            color = (0, 0, 255) if i == dragging_idx else (0, 255, 255)
            cv2.circle(img_disp, (pt[0], pt[1]), 8, color, -1)
            cv2.putText(img_disp, labels[i], (pt[0] + 15, pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Drag & Drop - Ellipse Fit", img_disp)

        # 20ミリ秒ごとにキー入力を待つ（画面を更新し続けるため）
        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # Enterキー
            break
        elif key == ord('q'):  # qキー
            print("キャンセルされました。")
            cv2.destroyAllWindows()
            sys.exit()

    # Enterが押されたら、楕円 -> 真円への射影変換を実行
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_original_resized, M, (OUTPUT_SIZE, OUTPUT_SIZE))

    # 結果の表示と保存
    cv2.imshow("Warped", warped)
    cv2.imwrite(OUTPUT_IMAGE, warped)
    print(f"変換完了: {OUTPUT_IMAGE} に保存しました。")
    print("画像ウィンドウを選択した状態で、任意のキーを押して終了してください。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()