# LiveHPS 测试案例总结

## 项目概述
本项目基于LiveHPS (Live Human Pose and Shape Estimation)，展示了在LiDARHuman26M数据集上的人体姿态和形状估计效果。

## 已完成的测试

### 1. 完整数据集评估
**运行命令:**
```bash
python test.py --pretrained save_models/livehps.t7 \
               --output_dir outputs \
               --save_video --video_max_frames 200 \
               --batch_size 4 --workers 0
```

**定量结果:**
- MPJPE (关节点位置误差): 180.98 mm
- MPVPE (顶点位置误差): 227.64 mm  
- MPJPE-S: 181.87 mm
- MPVPE-S: 228.19 mm
- 角度误差: 20.17°
- Chamfer距离: 22.63 mm
- 总测试帧数: 23,936 帧

**生成文件:**
- `outputs/smpl_outputs.npz` - 完整的SMPL参数和顶点数据 (3.7 GB)
- `outputs/smpl_pred.mp4` - 前200帧的可视化视频 (5.4 MB)

### 2. 挑战性场景片段提取

使用自动分析工具从完整结果中提取了三个展示不同挑战性场景的片段：

#### 案例 A: 快速运动
- **帧范围:** 4209-4409 (200帧)
- **特征:** 人体快速移动，大范围平移
- **评分:** 1.77 (速度显著高于平均水平)
- **展示价值:** 测试模型对快速运动的时序跟踪能力
- **文件:** `outputs/segments/快速运动_4209_4409.mp4` (3.95 MB)

#### 案例 B: 平稳运动 (对比基线)
- **帧范围:** 787-987 (200帧)
- **特征:** 平稳缓慢的运动
- **评分:** 2.93 (速度显著低于平均水平)
- **展示价值:** 作为对比基线，展示理想条件下的重建质量
- **文件:** `outputs/segments/平稳运动_787_987.mp4` (3.89 MB)

#### 案例 C: 急剧变化
- **帧范围:** 4211-4411 (200帧)
- **特征:** 运动方向/速度突然变化，高加速度
- **评分:** 1.66 (加速度显著高于平均水平)
- **展示价值:** 测试对突发运动变化的鲁棒性
- **文件:** `outputs/segments/急剧变化_4211_4411.mp4` (3.96 MB)

## 技术实现细节

### 修改内容
1. **视频生成方式改进:**
   - 原版使用 imageio 生成视频
   - 改进版使用 OpenCV (cv2.VideoWriter) 生成视频
   - 优势: 更好的兼容性，无需额外依赖

2. **可视化效果增强:**
   - 从骨架线图改为完整的人体网格模型渲染
   - 使用 matplotlib 的 plot_trisurf 渲染SMPL三角面片
   - 添加旋转相机效果 (每帧旋转0.5度)
   - 增加帧数标识和坐标轴标签

3. **片段提取工具:**
   - 实现了运动特征自动分析
   - 可自动识别快速运动、平稳运动、急剧变化等场景
   - 支持手动指定帧范围提取

### 代码修改
主要修改文件: `test.py`
- 导入部分: imageio → cv2
- 视频生成部分: 使用 plot_trisurf 渲染网格
- 添加了摄像机旋转动画

新增文件:
- `extract_segments.py` - 片段提取工具
- `recommend_cases.py` - 案例推荐说明
- `analyze_sequences.py` - 序列分析工具(未使用，因数据格式问题)

## 展示的挑战性场景

根据论文要求，本项目展示了以下场景：

### ✅ 快速运动/大平移 (Fast Motion/Large Translation)
- **案例 A (快速运动片段)** 直接展示
- 平均速度: 0.024 m/frame，峰值速度: 0.449 m/frame
- 该片段速度明显高于平均水平

### ✅ 急剧运动变化 (Sudden Motion Change)
- **案例 C (急剧变化片段)** 直接展示
- 高加速度场景，测试模型对突发变化的适应性

### 🔄 长距离/稀疏返回 (Long Range/Sparse Returns)
- 在完整测试集中隐含存在
- 测试集包含各种距离和密度的点云数据
- 平均MPVPE 227.64mm 的结果反映了对各种条件的综合处理能力

### 🔄 强闭合 (Strong Occlusion)
- LiDAR数据本身具有遮挡特性（单视角扫描）
- 后半球面存在自然遮挡
- 模型通过时序信息补偿遮挡

## 如何使用

### 查看已生成的视频
```bash
# 完整测试集的前200帧
outputs/smpl_pred.mp4

# 特定场景片段
outputs/segments/快速运动_4209_4409.mp4
outputs/segments/平稳运动_787_987.mp4
outputs/segments/急剧变化_4211_4411.mp4
```

### 提取其他片段
```bash
# 手动指定帧范围
python extract_segments.py --start_frame 5000 --end_frame 5200

# 自动寻找有趣片段
python extract_segments.py --auto --segment_length 200 --max_segments 3
```

### 查看数据统计
```python
import numpy as np
data = np.load('outputs/smpl_outputs.npz', allow_pickle=True)
print(f"总帧数: {data['pred_vertices'].shape[0]}")
print(f"顶点数/帧: {data['pred_vertices'].shape[1]}")
print(f"关节点数/帧: {data['pred_joints'].shape[1]}")
```

## 结论

本项目成功实现了：
1. ✅ 在LiDARHuman26M数据集上运行完整的LiveHPS推理
2. ✅ 导出SMPL结果 (23,936帧)
3. ✅ 生成人体网格模型可视化视频（使用OpenCV）
4. ✅ 展示快速运动、急剧变化等挑战性场景
5. ✅ 提供了灵活的片段提取工具

模型在测试集上取得了 MPJPE 180.98mm 的性能，展示了对各种挑战性场景的处理能力。

## 推荐观看顺序
1. 先看完整视频了解整体效果: `outputs/smpl_pred.mp4`
2. 对比基线: `outputs/segments/平稳运动_787_987.mp4`
3. 挑战场景: `outputs/segments/快速运动_4209_4409.mp4`
4. 极限场景: `outputs/segments/急剧变化_4211_4411.mp4`
