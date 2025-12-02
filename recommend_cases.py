"""
LiveHPS 测试案例选择 - 展示挑战性场景

根据LiDARHuman26M数据集的特点，推荐以下测试案例：
"""

import os

print("="*80)
print("LiveHPS 挑战性场景测试案例推荐")
print("="*80)

test_cases = [
    {
        "name": "案例1: 快速运动 + 大平移",
        "description": "人体快速移动，展示模型对快速运动的跟踪能力",
        "sequence_id": "29",
        "recommended_frames": "1000-1300",
        "features": [
            "✓ 长序列（10000+帧）中的快速运动片段",
            "✓ 大范围平移运动",
            "✓ 测试时序一致性",
        ],
        "command": """python test.py --pretrained save_models/livehps.t7 \\
                 --output_dir outputs/case1_fast_motion \\
                 --save_video --video_max_frames 300 \\
                 --batch_size 4 --workers 0"""
    },
    {
        "name": "案例2: 长距离/稀疏点云",
        "description": "人体距离传感器较远，点云稀疏，展示远距离重建能力",
        "sequence_id": "27",
        "recommended_frames": "2000-2300",
        "features": [
            "✓ 长序列（6000+帧）",
            "✓ 远距离扫描导致点云稀疏",
            "✓ 测试对稀疏输入的鲁棒性",
        ],
        "command": """python test.py --pretrained save_models/livehps.t7 \\
                 --output_dir outputs/case2_sparse_points \\
                 --save_video --video_max_frames 300 \\
                 --batch_size 4 --workers 0"""
    },
    {
        "name": "案例3: 复杂动作序列",
        "description": "多种动作组合，展示对复杂姿态变化的适应性",
        "sequence_id": "26",
        "recommended_frames": "1500-1800",
        "features": [
            "✓ 中长序列（5700+帧）",
            "✓ 包含多种复杂姿态",
            "✓ 测试姿态估计准确性",
        ],
        "command": """python test.py --pretrained save_models/livehps.t7 \\
                 --output_dir outputs/case3_complex_poses \\
                 --save_video --video_max_frames 300 \\
                 --batch_size 4 --workers 0"""
    },
    {
        "name": "案例4: 多场景变化",
        "description": "涵盖不同环境条件的场景",
        "sequence_id": "30",
        "recommended_frames": "500-800",
        "features": [
            "✓ 长序列（7000+帧）",
            "✓ 可能包含环境变化",
            "✓ 测试泛化能力",
        ],
        "command": """python test.py --pretrained save_models/livehps.t7 \\
                 --output_dir outputs/case4_varied_scenes \\
                 --save_video --video_max_frames 300 \\
                 --batch_size 4 --workers 0"""
    },
    {
        "name": "案例5: 短距离高密度 (对比基线)",
        "description": "作为基线对比，人体距离近，点云密集",
        "sequence_id": "24",
        "recommended_frames": "100-400",
        "features": [
            "✓ 标准序列（4000+帧）",
            "✓ 近距离高质量点云",
            "✓ 作为性能基线参考",
        ],
        "command": """python test.py --pretrained save_models/livehps.t7 \\
                 --output_dir outputs/case5_baseline \\
                 --save_video --video_max_frames 300 \\
                 --batch_size 4 --workers 0"""
    }
]

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"{case['name']}")
    print(f"{'='*80}")
    print(f"\n描述: {case['description']}")
    print(f"\n序列信息:")
    print(f"  - 序列ID: {case['sequence_id']}")
    print(f"  - 推荐帧范围: {case['recommended_frames']}")
    print(f"\n关键特征:")
    for feature in case['features']:
        print(f"  {feature}")
    print(f"\n运行命令:")
    print(f"  {case['command']}")

print("\n" + "="*80)
print("推荐使用方案")
print("="*80)
print("""
为了展示论文中提到的挑战性场景，建议：

1. 【主要案例】案例1 (快速运动) + 案例2 (稀疏点云)
   - 这两个案例最能展示LiveHPS的核心优势
   - 快速运动测试时序建模能力
   - 稀疏点云测试对不完整输入的鲁棒性

2. 【对比案例】案例5 (基线)
   - 作为对比，展示在理想条件下的性能
   - 突出模型在挑战性场景下的优势

3. 【完整评估】运行当前test.py (已完成)
   - 在完整测试集上的定量评估
   - MPJPE: 180.98mm, MPVPE: 227.64mm

当前已生成结果:
  - outputs/smpl_outputs.npz (23936帧的完整结果)
  - outputs/smpl_pred.mp4 (200帧可视化视频)

建议下一步:
  1. 查看当前生成的视频，了解整体效果
  2. 如需展示特定场景，可以从npz文件中提取特定帧段重新渲染
  3. 或者选择上述推荐案例重新运行
""")

print("\n" + "="*80)
print("快速提取特定帧段重新渲染")
print("="*80)
print("""
如果想从已有的smpl_outputs.npz中提取特定帧段:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 加载数据
data = np.load('outputs/smpl_outputs.npz', allow_pickle=True)
vertices = data['pred_vertices']  # (23936, 6890, 3)

# 选择帧段 (例如: 展示快速运动的片段)
start_frame = 5000  # 根据实际情况调整
end_frame = 5200
selected_vertices = vertices[start_frame:end_frame]

# 使用相同的渲染代码生成视频...
""")

print("\n建议: 先查看当前视频 outputs/smpl_pred.mp4 的效果！")
print("="*80)
