import os
import glob

# ディレクトリ設定
LABEL_DIR = r"..\TargetLabels"

def migrate():
    txt_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    processed_count = 0
    
    for txt_path in txt_files:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False
        for line in lines:
            parts = line.strip().split()
            
            # 4点データ (17パート: class + bbox(4) + 4pts*3) の場合
            if len(parts) == 17:
                xc, yc = parts[1], parts[2]
                min_pt = parts[11:14]
                max_pt = parts[14:17]
                
                # Mid点の計算 (MinとMaxの中間)
                mid_x = (float(min_pt[0]) + float(max_pt[0])) / 2
                mid_y = (float(min_pt[1]) + float(max_pt[1])) / 2
                mid_pt = [f"{mid_x:.6f}", f"{mid_y:.6f}", "2"]
                
                # Center点の計算 (BBoxの中心を仮置き)
                center_pt = [f"{float(xc):.6f}", f"{float(yc):.6f}", "2"]
                
                # 新しい順序: Pivot(5:8), Tip(8:11), Min, Mid, Max, Center
                new_line_parts = parts[:11] + min_pt + mid_pt + max_pt + center_pt
                new_lines.append(" ".join(new_line_parts))
                modified = True
                
            # 5点データ (20パート: class + bbox(4) + 5pts*3) の場合
            elif len(parts) == 20:
                xc, yc = parts[1], parts[2]
                
                # Center点の計算 (BBoxの中心を仮置き)
                center_pt = [f"{float(xc):.6f}", f"{float(yc):.6f}", "2"]
                
                # 既存の5点の後ろにCenterを追加
                new_line_parts = parts + center_pt
                new_lines.append(" ".join(new_line_parts))
                modified = True
                
            else:
                # 既に6点以上の場合はそのまま
                new_lines.append(line.strip())
        
        with open(txt_path, 'w') as f:
            f.write("\n".join(new_lines))
            
        if modified:
            processed_count += 1
            
    print(f"完了: {processed_count} 個のラベルファイルを [Pivot, Tip, Min, Mid, Max, Center] の6点形式にアップグレードしました！")

if __name__ == "__main__":
    migrate()