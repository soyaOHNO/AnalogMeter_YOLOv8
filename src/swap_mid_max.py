import os
import glob

# ディレクトリ設定
LABEL_DIR = r"..\TargetLabels"

def swap_mid_max():
    txt_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    count = 0
    
    for txt_path in txt_files:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # 5点データ(1+4+15 = 20パート)であることを確認
            if len(parts) >= 20:
                # 0:class, 1-4:bbox
                # 5-7:Pivot, 8-10:Tip, 11-13:Min, 14-16:Mid, 17-19:Max
                
                mid_pt = parts[14:17] # Mid
                max_pt = parts[17:20] # Max
                
                # 入れ替え: 14-16にMaxを、17-19にMidを配置
                parts[14:17] = max_pt
                parts[17:20] = mid_pt
                
                new_lines.append(" ".join(parts))
                count += 1
            else:
                new_lines.append(line.strip())
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(new_lines))
            
    print(f"{count} 個のラベルファイルを処理しました。[Mid]と[Max]の座標を入れ替えました。")

if __name__ == "__main__":
    swap_mid_max()