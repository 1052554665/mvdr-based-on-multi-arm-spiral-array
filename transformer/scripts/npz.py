# usage: 
# cd transformer/scripts
# python npz.py 

import numpy as np
import torch
from pathlib import Path

# 加载 .npz 文件（路径相对于脚本自身位置）
script_dir = Path(__file__).resolve().parent
data_path = script_dir / '..' / 'experiments' / 'train' / 'enhanced_outputs' / 'enhanced_0001.npz'
data = np.load(str(data_path))

# 查看里面包含哪些数组（键名）
print(data.files)  # 假设输出: ['features', 'labels']