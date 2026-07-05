import os
import yaml
from ultralytics import YOLO

# データセットの絶対パスを取得（YOLOは絶対パスを推奨します）
DATASET_DIR = os.path.abspath(r"..\yolo_dataset")
YAML_PATH = "meter_keypoints.yaml"

def create_yaml():
    """YOLO学習用の設定ファイル(yaml)を自動生成する"""
    yaml_data = {
        'path': DATASET_DIR,
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
    # 1. YAMLファイルの生成
    create_yaml()

    # 2. YOLOv8-poseモデルの読み込み
    print("=== YOLOv8 Pose Sim2Real 学習開始 ===")
    model = YOLO("yolov8s-pose.pt")

    # 3. 学習の実行
    results = model.train(
        data=YAML_PATH,
        epochs=300,        # データが大量にあるので300周でも十分に収束します
        patience=50,       # 50回改善がなければ自動完了
        imgsz=640,         
        batch=16,           # メモリに余裕があれば 16 などに上げてください
        
        # --- データ水増し（Sim2Real最適化） ---
        # 3D的な歪みはBlenderがやってくれているので、照明や色味のノイズに集中させます
        degrees=10.0,      
        translate=0.1,     
        scale=0.2,         
        shear=0.0,
        perspective=0.0,   
        hsv_h=0.02,        # 現実の様々な照明色（蛍光灯や夕日など）に対応
        hsv_s=0.7,         
        hsv_v=0.6,         # 暗い工場や明るい屋外に対応
        erasing=0.2,       # ガラスの反射などで一部が隠れていても推論できるようにする
        
        fliplr=0.0,        # 左右反転はOFFのまま
        mosaic=0.5,        # 複数のCG画像を繋ぎ合わせて学習効率を上げる
    )
    
    print("新データセットでの学習が完了しました！")

if __name__ == "__main__":
    main()