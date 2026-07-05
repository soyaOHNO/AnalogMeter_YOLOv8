import os
import glob
import shutil

# パス設定（環境に合わせて微調整してください）
SYNTH_DIR = r"..\yolo_synth_data" # Blenderが生成したフォルダ
YOLO_DIR = r"..\..\yolo_dataset"     # YOLOのデータセットフォルダ

def main():
    print("=== 合成データをYOLO学習用(Train)に統合するツール ===")
    
    # 移行先のフォルダがあるか確認
    train_img_dir = os.path.join(YOLO_DIR, "images", "train")
    train_lbl_dir = os.path.join(YOLO_DIR, "labels", "train")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    
    synth_imgs = glob.glob(os.path.join(SYNTH_DIR, "images", "*.jpg"))
    
    if not synth_imgs:
        print(f"エラー: {SYNTH_DIR} に画像が見つかりません。パスを確認してください。")
        return

    count = 0
    for img_p in synth_imgs:
        img_name = os.path.basename(img_p)
        base_name = os.path.splitext(img_name)[0]
        txt_p = os.path.join(SYNTH_DIR, "labels", f"{base_name}.txt")
        
        if os.path.exists(txt_p):
            shutil.copy(img_p, os.path.join(train_img_dir, img_name))
            shutil.copy(txt_p, os.path.join(train_lbl_dir, os.path.basename(txt_p)))
            count += 1
            
    print(f"成功！ 合計 {count} 枚の合成データを学習用(train)に追加しました。")
    print("※ テスト(val)には現実の画像がそのまま残っているため、AIの『現実への適応力』をテストできます。")

if __name__ == "__main__":
    main()