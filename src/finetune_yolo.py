import os
import glob
import shutil
import random
import yaml
from ultralytics import YOLO

# ==========================================
# 設定パス
# ==========================================
SOURCE_IMG_DIR = r"..\TargetImages"
SOURCE_LBL_DIR = r"..\TargetLabels"
REAL_DATASET_DIR = os.path.abspath(r"..\yolo_real_dataset")
CG_MODEL_PATH = r"best.pt"  # 先ほどColabから持ち帰った「CGで鍛えた最強モデル」
YAML_PATH = "real_meter.yaml"

def setup_dataset():
    """実写データを train (8割) と val (2割) に分割して配置する"""
    print("=== 実写データセットの準備 ===")
    
    # フォルダの初期化
    for split in ["train", "val"]:
        os.makedirs(os.path.join(REAL_DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(REAL_DATASET_DIR, "labels", split), exist_ok=True)

    # 画像リストの取得
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(SOURCE_IMG_DIR, ext)))
        
    if not image_paths:
        print(f"エラー: {SOURCE_IMG_DIR} に実写画像が見つかりません。")
        return False

    # ランダムにシャッフルして8:2に分割
    random.seed(42)
    random.shuffle(image_paths)
    split_idx = max(1, int(len(image_paths) * 0.8)) # 最低1枚はvalに残す
    
    train_imgs = image_paths[:split_idx]
    val_imgs = image_paths[split_idx:]

    def copy_files(img_list, split_name):
        count = 0
        for img_p in img_list:
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            lbl_p = os.path.join(SOURCE_LBL_DIR, f"{base_name}.txt")
            
            # ラベルが存在する画像だけをコピー
            if os.path.exists(lbl_p):
                shutil.copy(img_p, os.path.join(REAL_DATASET_DIR, "images", split_name, os.path.basename(img_p)))
                shutil.copy(lbl_p, os.path.join(REAL_DATASET_DIR, "labels", split_name, os.path.basename(lbl_p)))
                count += 1
        return count

    train_count = copy_files(train_imgs, "train")
    val_count = copy_files(val_imgs, "val")
    
    print(f"データセット構築完了: Train {train_count} 枚, Val {val_count} 枚")
    return True

def create_yaml():
    """YOLO用の設定ファイルを作成"""
    yaml_data = {
        'path': REAL_DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['meter'],
        'kpt_shape': [6, 3] # [Pivot, Tip, Min, Mid, Max, Center]
    }
    with open(YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"設定ファイル {YAML_PATH} を作成しました。")

def main():
    # 1. データの準備
    if not setup_dataset():
        return
    create_yaml()

    # 2. CGで鍛えたモデルの読み込み
    if not os.path.exists(CG_MODEL_PATH):
        print(f"エラー: ベースとなるモデル {CG_MODEL_PATH} が見つかりません。")
        return
        
    print(f"=== ファインチューニング開始 (ベースモデル: {CG_MODEL_PATH}) ===")
    model = YOLO(CG_MODEL_PATH)

    # 3. 追加学習の実行
    # 実写データは数が少ないため、過学習を防ぐ設定にします
    results = model.train(
        data=YAML_PATH,
        epochs=100,        # 枚数が少ないので100周で十分
        patience=20,       # 20周改善がなければストップ
        imgsz=640,
        batch=4,           # ローカルPC向けにバッチサイズを小さく
        
        # --- ファインチューニング特有の設定 ---
        lr0=0.001,         # 学習率を小さく（CGで学んだ知識を破壊しないようにそっと教える）
        freeze=10,         # ネットワークの前半10層(特徴抽出の基礎)を凍結し、CGの記憶を保持する
        
        # 実写特有のノイズ耐性をつける
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.1
    )
    
    print("実写データでのファインチューニングが完了しました！")
    print("新しいモデルは 'runs/pose/train.../weights/best.pt' に保存されています。")

if __name__ == "__main__":
    main()