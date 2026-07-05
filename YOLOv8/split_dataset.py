import os
import glob
import random
import shutil

# パス設定 (環境に合わせてください)
SRC_IMAGE_DIR = r"..\TargetImages"
SRC_LABEL_DIR = r"..\TargetLabels"
DEST_DIR = r"..\yolo_dataset"

# 振り分ける枚数の設定
TRAIN_COUNT = 15

def main():
    print("=== YOLOデータセット分割・準備ツール ===")
    
    # コピー先のフォルダを作成（実行するたびに中身をリセットして綺麗にします）
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        path = os.path.join(DEST_DIR, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        
    # 画像とラベルのペアを取得
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(SRC_IMAGE_DIR, ext)))
        
    paired_data = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(SRC_LABEL_DIR, f"{base_name}.txt")
        
        # ラベルファイル(テキスト)が存在するものだけを対象にする
        if os.path.exists(txt_path):
            paired_data.append((img_path, txt_path))
            
    total_data = len(paired_data)
    print(f"有効なデータペア（画像＋ラベル）: {total_data}件")
    
    if total_data < TRAIN_COUNT:
        print(f"エラー: データが {TRAIN_COUNT}件未満です。アノテーションを確認してください。")
        return
        
    # データをランダムにシャッフル
    random.seed(42) # いつも同じ分け方にしたい場合はシードを固定
    random.shuffle(paired_data)
    
    train_data = paired_data[:TRAIN_COUNT]
    val_data = paired_data[TRAIN_COUNT:]
    
    print(f"Train用(学習): {len(train_data)}件 / Val用(テスト): {len(val_data)}件 に分割します。")
    
    # データを指定のフォルダにコピー
    def copy_data(data_list, subset):
        for img_p, txt_p in data_list:
            shutil.copy(img_p, os.path.join(DEST_DIR, f'images/{subset}', os.path.basename(img_p)))
            shutil.copy(txt_p, os.path.join(DEST_DIR, f'labels/{subset}', os.path.basename(txt_p)))
            
    copy_data(train_data, 'train')
    copy_data(val_data, 'val')
    
    print("データセットの配置が完了しました！")

if __name__ == "__main__":
    main()