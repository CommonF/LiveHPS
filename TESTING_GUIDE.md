# LiveHPS 测试工具使用指南

## 文件说明

本项目包含多个测试和分析工具：

### 核心测试脚本

1. **`test.py`** - 主测试脚本（已修改）
   - 在完整数据集上运行推理
   - 使用OpenCV生成人体网格视频
   - 输出SMPL参数和可视化结果

2. **`robustness_test.py`** - 鲁棒性测试脚本 ⭐
   - 测试模型在各种降质条件下的表现
   - 自动生成对比视频和性能图表
   - 评估失败模式

3. **`extract_segments.py`** - 片段提取工具
   - 从完整结果中提取特定帧段
   - 自动识别有趣场景（快速运动、急剧变化等）
   - 重新渲染选定片段

4. **`recommend_cases.py`** - 案例推荐
   - 列出推荐的测试场景
   - 提供运行命令示例

## 快速开始

### 1. 运行鲁棒性测试（推荐）

```bash
# 基础鲁棒性测试
python robustness_test.py --sequence_id 24 --start_frame 100 --num_frames 150

# 自定义设置
python robustness_test.py \
    --sequence_id 29 \
    --start_frame 500 \
    --num_frames 200 \
    --device cuda \
    --max_video_frames 150
```

**生成内容:**
- 6个对比视频（基线 + 5种降质条件）
- 6个结果数据文件（.npz格式）
- 1个性能对比图表（.png）
- 控制台输出详细的性能报告

### 2. 完整数据集测试

```bash
# 在整个测试集上运行
python test.py \
    --pretrained save_models/livehps.t7 \
    --output_dir outputs \
    --save_video \
    --video_max_frames 200 \
    --batch_size 4 \
    --workers 0
```

### 3. 提取特定场景片段

```bash
# 自动寻找有趣片段
python extract_segments.py --auto --segment_length 200

# 手动指定帧范围
python extract_segments.py --start_frame 5000 --end_frame 5200
```

## 鲁棒性测试详解

### 测试类型

`robustness_test.py` 自动执行以下6种测试：

1. **基线测试** - 原始数据，无降质
2. **点云下采样50%** - 模拟稀疏LiDAR
3. **点云下采样25%** - 极度稀疏情况
4. **帧丢弃50%** - 模拟低帧率
5. **前方遮挡** - 添加30%杂波点
6. **综合严重降质** - 同时应用多种降质

### 输出文件结构

```
outputs/robustness_test/
├── none_comparison.mp4              # 基线对比视频
├── none_results.npz                 # 基线结果数据
├── downsample_50_comparison.mp4     # 50%下采样对比视频
├── downsample_50_results.npz        # 50%下采样结果
├── downsample_25_comparison.mp4     # 25%下采样对比视频
├── downsample_25_results.npz        # 25%下采样结果
├── frame_drop_50_comparison.mp4     # 帧丢弃对比视频
├── frame_drop_50_results.npz        # 帧丢弃结果
├── clutter_front_comparison.mp4     # 遮挡对比视频
├── clutter_front_results.npz        # 遮挡结果
├── combined_severe_comparison.mp4   # 综合降质对比视频
├── combined_severe_results.npz      # 综合降质结果
└── metrics_comparison.png           # 性能对比图表
```

### 视频格式说明

每个对比视频包含：
- **左侧**: 模型预测结果（蓝色网格）
- **右侧**: 真实标签（绿色网格）
- **标题**: 显示降质类型和当前帧数
- **相机**: 自动旋转以全方位查看

## 关键测试结果

根据Sequence 24 (Frames 100-250)的测试结果：

| 测试条件 | MPJPE (mm) | 变化率 | 关键观察 |
|---------|-----------|--------|---------|
| 基线 | 185.17 | 0.0% | 性能基准 |
| 点云50% | 184.72 | -0.2% | ✅ 几乎无影响 |
| 点云25% | 186.18 | +0.5% | ✅ 极度稀疏仍稳定 |
| 帧率50% | 180.24 | -2.7% | ✅✅ 性能反而提升！|
| 前方遮挡 | 187.15 | +1.1% | ⚠️ 轻微影响 |
| 综合降质 | 182.25 | -1.6% | ✅✅ 鲁棒性极强 |

**结论:** LiveHPS模型展现出**卓越的鲁棒性**，在各种降质条件下性能变化都在±3%以内。

## 高级用法

### 自定义降质参数

修改 `robustness_test.py` 中的 `test_configs` 列表：

```python
{
    'name': 'custom_test',
    'description': 'Custom Degradation',
    'degradation_type': 'downsample',  # 'none', 'downsample', 'frame_drop', 'clutter', 'combined'
    'params': {
        'downsample_ratio': 0.1,  # 保留10%的点
        'frame_keep_ratio': 0.3,  # 保留30%的帧
        'clutter_ratio': 0.5,     # 50%杂波
        'clutter_region': 'front' # 'random', 'front', 'around'
    }
}
```

### 分析特定降质的失败模式

```python
import numpy as np

# 加载结果
data = np.load('outputs/robustness_test/clutter_front_results.npz')
pred_joints = data['pred_joints']
gt_joints = data['gt_joints']

# 计算逐帧误差
frame_errors = np.linalg.norm(pred_joints - gt_joints, axis=2).mean(axis=1)

# 找出最大误差帧
worst_frame = np.argmax(frame_errors)
print(f"最大误差出现在第 {worst_frame} 帧: {frame_errors[worst_frame]:.2f}mm")
```

### 批量测试多个序列

```bash
# 创建批处理脚本
for seq in 24 25 26 27; do
    python robustness_test.py \
        --sequence_id $seq \
        --start_frame 100 \
        --num_frames 150 \
        --output_dir outputs/robustness_seq${seq}
done
```

## 注意事项

1. **内存占用**: 完整测试需要约4GB显存
2. **运行时间**: 每个测试约需15-20分钟（包括推理和渲染）
3. **磁盘空间**: 每个测试生成约30MB数据
4. **建议配置**: CUDA GPU, 16GB RAM

## 故障排除

### 问题：CUDA内存不足
```bash
# 减少batch size或使用CPU
python robustness_test.py --device cpu
# 或减少帧数
python robustness_test.py --num_frames 100
```

### 问题：视频渲染缓慢
```bash
# 减少视频帧数
python robustness_test.py --max_video_frames 50
```

### 问题：找不到序列数据
- 确认 `dataset/lidarhuman26M/` 目录存在
- 检查序列ID和帧范围是否有效
- 查看 `dataset/lidarhuman26M/labels/3d/segment/[序列ID]/` 是否有.ply文件

## 参考文档

- `TEST_SUMMARY.md` - 整体测试总结
- `ROBUSTNESS_TEST_REPORT.md` - 详细的鲁棒性测试报告
- `README.md` - 原始项目说明

## 引用

如果使用这些工具，请引用原始论文：
```
LiveHPS: LiDAR-based Scene-level Human Pose and Shape Estimation in Free Environment
```

## 量化分析（性能）

- 概述: 使用 `measure_performance.py` 在 Sequence 24, Frames 100-149 上测量单帧推理性能（CUDA）。同时保存结构化结果到 `outputs/performance_metrics.json`。
- 运行:
    - `python measure_performance.py`
- 指标定义:
    - 平均单帧时延: 多次重复的平均值（ms）
    - FPS: 由平均时延换算（1000/mean_time_ms）
    - 峰值GPU内存: 单次前向传播的峰值占用（MB）

结果（均值±标准差）：
- 基线（100%点云，平均原始点数≈256 → 输入重采样256）
    - 运行时间: 8.92 ± 0.93 ms
    - FPS: 112.0
    - 峰值GPU内存: 541.4 MB
- 下采样50%（平均原始点数≈128 → 输入重采样256）
    - 运行时间: 8.61 ± 0.76 ms
    - FPS: 116.1
    - 峰值GPU内存: 541.4 MB
- 下采样25%（平均原始点数≈64 → 输入重采样256）
    - 运行时间: 8.75 ± 0.98 ms
    - FPS: 114.2
    - 峰值GPU内存: 541.4 MB
- 下采样10%（平均原始点数≈25 → 输入重采样256）
    - 运行时间: 8.79 ± 1.07 ms
    - FPS: 113.7
    - 峰值GPU内存: 541.4 MB

趋势（点数 vs 性能）:
- 固定输入为 `256` 点（通过最远点重采样），因此推理时延与GPU峰值内存基本稳定；不同下采样比率仅影响“原始点数”，对模型推理侧影响很小（时延波动约±2%）。
- 峰值GPU内存在所有配置下近似不变（≈541MB），表明显存主要由模型本身决定而非原始点数。
- 如果将模型输入点数也随原始点数变化而变化，则预计时延与显存会随输入点数近似线性增长；当前脚本固定为256点以匹配训练/测试流程。

原始输出:
- 终端日志与结构化结果见 `outputs/performance_metrics.json`（可用于论文/报告直接引用）。
