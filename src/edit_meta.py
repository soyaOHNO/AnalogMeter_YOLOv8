import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import glob
from PIL import Image, ImageTk

# ==========================================
# 設定
# ==========================================
IMAGE_DIR = r"..\TargetImages"
JSON_PATH = "meter_meta.json"

class MetaEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meter Meta Editor")
        self.root.geometry("1200x700")
        
        # データの読み込み
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
            self.image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
        self.image_paths = sorted(list(set(self.image_paths)))
        
        self.meta_data = {}
        if os.path.exists(JSON_PATH):
            try:
                with open(JSON_PATH, 'r', encoding='utf-8') as f:
                    self.meta_data = json.load(f)
            except Exception as e:
                messagebox.showerror("エラー", f"JSONの読み込みに失敗しました:\n{e}")
                
        self.current_idx = 0
        self.current_vars = [] # 現在表示中のフォームの変数を保持
        
        self.setup_ui()
        
        if not self.image_paths:
            messagebox.showwarning("警告", f"{IMAGE_DIR} に画像が見つかりません。")
        else:
            self.load_image()
            
        # ショートカットキーの設定
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Control-s>", lambda e: self.save_json())

    def setup_ui(self):
        # 画面を左右に分割するPanedWindow
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- 左側：画像表示エリア ---
        self.left_frame = ttk.Frame(self.paned_window, relief=tk.SUNKEN)
        self.paned_window.add(self.left_frame, weight=3) # 幅の比率
        
        self.image_label = ttk.Label(self.left_frame, text="画像がありません", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # --- 右側：操作・入力エリア ---
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=1)
        
        # 情報表示部
        self.info_label = ttk.Label(self.right_frame, text="File: --- (0/0)", font=("Helvetica", 14, "bold"))
        self.info_label.pack(pady=(0, 10), fill=tk.X)
        
        # フォームを配置するスクロール可能なフレーム（計器が多い場合用）
        self.canvas = tk.Canvas(self.right_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="top", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # 計器追加ボタン
        self.add_btn = ttk.Button(self.right_frame, text="＋ 計器を追加", command=self.add_instance)
        self.add_btn.pack(pady=10, fill=tk.X)
        
        # コントロール部（下部）
        self.control_frame = ttk.Frame(self.right_frame)
        self.control_frame.pack(side="bottom", fill=tk.X, pady=10)
        
        self.prev_btn = ttk.Button(self.control_frame, text="< 前へ", command=self.prev_image)
        self.prev_btn.pack(side="left", expand=True, fill=tk.X, padx=2)
        
        self.next_btn = ttk.Button(self.control_frame, text="次へ >", command=self.next_image)
        self.next_btn.pack(side="left", expand=True, fill=tk.X, padx=2)
        
        self.save_btn = ttk.Button(self.control_frame, text="💾 JSONを保存", command=self.save_json)
        self.save_btn.pack(side="left", expand=True, fill=tk.X, padx=2)

    def load_image(self):
        if not self.image_paths: return
        
        # 1. 画像の表示
        img_path = self.image_paths[self.current_idx]
        img_name = os.path.basename(img_path)
        
        try:
            pil_image = Image.open(img_path)
            # 左フレームのサイズに合わせてリサイズ (雑に800x800の枠に収める)
            pil_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=self.tk_image, text="")
        except Exception as e:
            self.image_label.config(image='', text=f"画像読み込みエラー:\n{e}")

        self.info_label.config(text=f"File: {img_name} ({self.current_idx + 1}/{len(self.image_paths)})")
        
        # 2. フォームの再構築
        # 既存のフォームをクリア
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.current_vars = []
        
        # JSONにデータがあれば展開、なければ空のリスト
        instances = self.meta_data.get(img_name, [])
        if not instances:
            # デフォルトで1つ空のフォームを用意しておく
            instances = [{"index": 0, "min": "", "max": "", "unit": ""}]
            
        for inst in instances:
            self.create_instance_form(inst)

    def create_instance_form(self, inst_data):
        idx = len(self.current_vars)
        
        frame = ttk.LabelFrame(self.scrollable_frame, text=f"計器 {idx}")
        frame.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        # 変数の準備
        var_min = tk.StringVar(value=str(inst_data.get("min", "")))
        var_max = tk.StringVar(value=str(inst_data.get("max", "")))
        var_unit = tk.StringVar(value=str(inst_data.get("unit", "")))
        
        # Min
        ttk.Label(frame, text="Min:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(frame, textvariable=var_min, width=10).grid(row=0, column=1, sticky="w", pady=2)
        
        # Max
        ttk.Label(frame, text="Max:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(frame, textvariable=var_max, width=10).grid(row=1, column=1, sticky="w", pady=2)
        
        # Unit
        ttk.Label(frame, text="Unit:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(frame, textvariable=var_unit, width=15).grid(row=2, column=1, sticky="w", pady=2)
        
        # 削除ボタン
        del_btn = ttk.Button(frame, text="削除", width=5, command=lambda f=frame, i=idx: self.remove_instance(f, i))
        del_btn.grid(row=0, column=2, rowspan=3, padx=10)
        
        self.current_vars.append({
            "frame": frame,
            "min": var_min,
            "max": var_max,
            "unit": var_unit
        })

    def add_instance(self):
        """新しい計器入力フォームを追加"""
        idx = len(self.current_vars)
        new_inst = {"index": idx, "min": "", "max": "", "unit": ""}
        self.create_instance_form(new_inst)

    def remove_instance(self, frame, idx):
        """指定した計器入力フォームを削除し、UIを詰め直す"""
        self.save_current_to_memory() # 一旦今の状態を保存
        
        img_name = os.path.basename(self.image_paths[self.current_idx])
        instances = self.meta_data.get(img_name, [])
        
        if 0 <= idx < len(instances):
            instances.pop(idx)
            # indexを振り直す
            for i, inst in enumerate(instances):
                inst["index"] = i
            self.meta_data[img_name] = instances
            
        self.load_image() # 再描画

    def save_current_to_memory(self):
        """現在入力されているフォームの値を辞書（メモリ）に保存する"""
        if not self.image_paths: return
        
        img_name = os.path.basename(self.image_paths[self.current_idx])
        new_instances = []
        
        for i, vars_dict in enumerate(self.current_vars):
            # 空欄の場合はそのまま空文字列として扱うか、パースするか
            val_min = vars_dict["min"].get().strip()
            val_max = vars_dict["max"].get().strip()
            val_unit = vars_dict["unit"].get().strip()
            
            # 数値変換（失敗した場合は文字列のまま保存して後で手直せるようにする）
            try:
                val_min = float(val_min) if '.' in val_min else int(val_min)
            except ValueError:
                pass
                
            try:
                val_max = float(val_max) if '.' in val_max else int(val_max)
            except ValueError:
                pass
                
            new_instances.append({
                "index": i,
                "min": val_min,
                "max": val_max,
                "unit": val_unit
            })
            
        if new_instances:
            self.meta_data[img_name] = new_instances
        elif img_name in self.meta_data:
            # 全て削除された場合
            del self.meta_data[img_name]

    def save_json(self):
        """メモリ上の辞書をJSONファイルに書き出す"""
        self.save_current_to_memory() # 書き出す前に今の画面も確定させる
        try:
            with open(JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.meta_data, f, indent=2, ensure_ascii=False)
            
            # ターミナルへ通知 (画面の邪魔にならないように)
            print(f"[{JSON_PATH}] を保存しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.save_current_to_memory() # 移動前に自動保存
            self.current_idx += 1
            self.load_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.save_current_to_memory() # 移動前に自動保存
            self.current_idx -= 1
            self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    
    # 見た目を少しモダンにする(OSによって使えるテーマが違うため例外処理)
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass

    app = MetaEditorApp(root)
    root.mainloop()