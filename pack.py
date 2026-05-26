"""
跨平台打包脚本
- 生成 Linux/Kaggle 兼容的 project.zip
- 文件名统一使用正斜杠
- 排除 __pycache__、.git、实验结果、旧 zip
"""
import zipfile
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, "project.zip")

# 要包含的顶层条目
TOP_FILES = ["app.py", "config.yaml", "requirements.txt"]
TOP_DIRS = ["src", "experiments", "notebooks", "lora"]
EXCLUDE_DIRS = {"__pycache__", ".git", "results"}

count = 0
with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for top in TOP_FILES:
        path = os.path.join(ROOT, top)
        if os.path.exists(path):
            zf.write(path, top)
            print(f"  + {top}")
            count += 1

    for top in TOP_DIRS:
        top_path = os.path.join(ROOT, top)
        if not os.path.exists(top_path):
            continue
        for root, dirs, files in os.walk(top_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                full = os.path.join(root, f)
                arc = full.replace(ROOT + os.sep, "").replace("\\", "/")
                zf.write(full, arc)
                print(f"  + {arc}")
                count += 1

size_kb = os.path.getsize(OUTPUT) / 1024
print(f"\n[OK] project.zip 已生成 ({count} 文件, {size_kb:.0f} KB)")
print(f"[OK] 路径: {OUTPUT}")
