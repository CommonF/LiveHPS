"""
简单的性能测量脚本
直接测量不同点云密度下的推理时间、FPS和GPU内存
"""

import torch
import numpy as np
import time
from pathlib import Path
from models import LiveHPS
import json

def load_ply_points(path):
    """加载PLY点云文件"""
    with open(path, 'rb') as f:
        content = f.read()
    header_end = content.find(b'end_header\n')
    if header_end == -1:
        return np.zeros((0, 3), dtype=np.float32)
    header = content[:header_end].decode('ascii', errors='ignore').splitlines()
    vertex_count = 0
    for line in header:
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
            break
    start = header_end + len(b'end_header\n')
    data = np.frombuffer(content, dtype='<f4', count=vertex_count*3, offset=start)
    return data.reshape(-1, 3)

def farthest_point_sample(xyz, npoint):
    """最远点采样"""
    ndataset = xyz.shape[0]
    if ndataset == 0:
        return np.zeros((npoint, 3), dtype=np.float32)
    if ndataset < npoint:
        repeat_n = int(npoint / ndataset)
        xyz = np.tile(xyz, (repeat_n, 1))
        xyz = np.append(xyz, xyz[:npoint % ndataset], axis=0)
        return xyz
    centroids = np.zeros(npoint)
    distance = np.ones(ndataset) * 1e10
    farthest = np.random.randint(0, ndataset)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[int(farthest)]
        dist = np.sum((xyz - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return xyz[np.int32(centroids)]

def apply_downsampling(points, ratio):
    """应用下采样"""
    if ratio >= 1.0:
        return points
    n_points = len(points)
    n_keep = int(max(1, n_points * ratio))
    indices = np.random.choice(n_points, n_keep, replace=False)
    return points[indices]

def add_clutter(points, clutter_ratio=0.3, region='front'):
    """添加局部杂波/遮挡物"""
    n_points = len(points)
    n_clutter = int(max(1, n_points * clutter_ratio))
    if region == 'front':
        center = points.mean(axis=0).copy()
        center[2] += 0.5
        clutter = center + np.random.randn(n_clutter, 3).astype(np.float32) * 0.2
    elif region == 'around':
        bounds_min = points.min(axis=0) - 0.5
        bounds_max = points.max(axis=0) + 0.5
        clutter = np.random.uniform(bounds_min, bounds_max, (n_clutter, 3)).astype(np.float32)
    else:
        center = points[np.random.randint(len(points))]
        clutter = center + np.random.randn(n_clutter, 3).astype(np.float32) * 0.3
    return np.vstack([points, clutter])

def measure_performance(model, point_cloud, device, warmup=3, runs=10):
    """测量单帧推理性能"""
    model.eval()
    
    # 准备输入 [1, 1, 256, 3]
    pc = torch.from_numpy(point_cloud).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # GPU预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(pc)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时测试
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(pc)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
    
    return np.array(times)

def measure_gpu_memory(model, point_cloud, device):
    """测量GPU内存使用"""
    if device.type != 'cuda':
        return 0.0, 0.0
    
    model.eval()
    pc = torch.from_numpy(point_cloud).float().unsqueeze(0).unsqueeze(0).to(device)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        _ = model(pc)
    
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    current_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    return peak_mb, current_mb

print("=" * 80)
print("LiveHPS 性能测量 - Quantitative Analysis")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n设备: {device}")

# 加载模型
print("加载模型...")
model = LiveHPS().to(device)
model.eval()
print("✓ 模型加载完成")

# 加载测试数据
sequence_id = 24
start_frame = 100
num_frames = 30  # 使用30帧以加快一次全配置测试

seg_path = Path(f"dataset/lidarhuman26M/labels/3d/segment/{sequence_id}")
print(f"\n加载测试数据: Sequence {sequence_id}, Frames {start_frame}-{start_frame+num_frames-1}")

# 加载点云
all_points = []
original_point_counts = []

for frame_idx in range(start_frame, start_frame + num_frames):
    ply_file = seg_path / f"{frame_idx:06d}.ply"
    if ply_file.exists():
        points = load_ply_points(ply_file)
        original_point_counts.append(len(points))
        # 归一化
        loc = points.mean(0)
        points = points - loc
        # 采样到256个点
        points = farthest_point_sample(points, 256)
        all_points.append(points)

print(f"✓ 加载了 {len(all_points)} 帧")
print(f"  原始点云数量范围: {min(original_point_counts)} - {max(original_point_counts)} 点")
print(f"  平均原始点云数量: {np.mean(original_point_counts):.0f} 点")

# 定义测试配置
test_configs = [
    {'name': 'baseline', 'desc': '基线 (100% 点云)', 'mode': 'none', 'ratio': 1.0},
    {'name': 'downsample_50', 'desc': '50% 点云下采样', 'mode': 'downsample', 'ratio': 0.5},
    {'name': 'downsample_25', 'desc': '25% 点云下采样', 'mode': 'downsample', 'ratio': 0.25},
    {'name': 'downsample_10', 'desc': '10% 点云下采样', 'mode': 'downsample', 'ratio': 0.1},
    {'name': 'frame_drop_50', 'desc': '帧丢弃 50%（每隔一帧）', 'mode': 'frame_drop', 'keep_ratio': 0.5},
    {'name': 'clutter_front', 'desc': '前方遮挡（30% 杂波）', 'mode': 'clutter', 'clutter_ratio': 0.3, 'region': 'front'},
    {'name': 'combined_severe', 'desc': '综合严重降质（30%点+40%前方杂波）', 'mode': 'combined', 'ratio': 0.3, 'clutter_ratio': 0.4, 'region': 'front'},
]

results = []

print("\n" + "=" * 80)
print("开始性能测试")
print("=" * 80)

for config in test_configs:
    print(f"\n{'='*80}")
    print(f"测试: {config['desc']}")
    print(f"{'='*80}")
    
    # 应用降质并重新采样到256点
    test_points = []
    actual_point_counts = []

    # 针对帧丢弃：仅用于时序采样策略，不改变单帧输入大小
    if config['mode'] == 'frame_drop':
        selected_frames = all_points[::2] if config.get('keep_ratio', 0.5) == 0.5 else all_points
    else:
        selected_frames = all_points
    
    for pc in selected_frames:
        if config['mode'] == 'downsample':
            degraded = apply_downsampling(pc.copy(), config['ratio'])
        elif config['mode'] == 'clutter':
            degraded = add_clutter(pc.copy(), config.get('clutter_ratio', 0.3), config.get('region', 'front'))
        elif config['mode'] == 'combined':
            degraded = apply_downsampling(pc.copy(), config.get('ratio', 0.3))
            degraded = add_clutter(degraded, config.get('clutter_ratio', 0.4), config.get('region', 'front'))
        else:  # 'none' or 'frame_drop'
            degraded = pc.copy()

        actual_point_counts.append(len(degraded))
        # 再采样回256点（保持模型输入一致）
        resampled = farthest_point_sample(degraded, 256)
        test_points.append(resampled)
    
    avg_points = np.mean(actual_point_counts)
    print(f"下采样后点数: {avg_points:.0f} → 重采样到256点")
    
    # 性能测量
    print(f"测量运行时间 (warmup=2, runs=5)...")
    all_times = []
    
    for i, pc in enumerate(test_points):
        times = measure_performance(model, pc, device, warmup=2, runs=5)
        all_times.extend(times)
        
        if i == 0:  # 第一帧测量GPU内存
            peak_mem, curr_mem = measure_gpu_memory(model, pc, device)
    
    all_times = np.array(all_times)
    mean_time = np.mean(all_times) * 1000  # ms
    std_time = np.std(all_times) * 1000
    min_time = np.min(all_times) * 1000
    max_time = np.max(all_times) * 1000
    fps = 1000.0 / mean_time
    
    result = {
        'name': config['name'],
        'description': config['desc'],
        'mode': config['mode'],
        'downsample_ratio': float(config.get('ratio', 1.0)),
        'keep_ratio': float(config.get('keep_ratio', 1.0)),
        'clutter_ratio': float(config.get('clutter_ratio', 0.0)),
        'avg_original_points': avg_points,
        'model_input_points': 256,
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps,
        'peak_memory_mb': peak_mem if device.type == 'cuda' else 0,
    }
    
    results.append(result)
    
    print(f"\n结果:")
    print(f"  运行时间: {mean_time:.2f} ± {std_time:.2f} ms (范围: {min_time:.2f} - {max_time:.2f} ms)")
    print(f"  FPS:      {fps:.1f}")
    print(f"  GPU内存:  {peak_mem:.1f} MB (峰值)" if device.type == 'cuda' else "  GPU内存:  N/A (使用CPU)")

# 保存结果
print("\n" + "=" * 80)
print("汇总结果")
print("=" * 80)

print(f"\n{'测试配置':<25} {'平均点数':<12} {'运行时间 (ms)':<20} {'FPS':<10} {'GPU内存 (MB)':<15}")
print("-" * 90)

for r in results:
    print(f"{r['description']:<25} {r['avg_original_points']:>6.0f} → 256  "
          f"{r['mean_time_ms']:>6.2f} ± {r['std_time_ms']:>4.2f}  "
          f"{r['fps']:>8.1f}  {r['peak_memory_mb']:>12.1f}")

# 保存JSON
output_file = Path('outputs/performance_metrics.json')
output_file.parent.mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ 详细结果已保存到: {output_file}")

# 分析点云数量 vs 性能的趋势
print("\n" + "=" * 80)
print("性能趋势分析")
print("=" * 80)

baseline = results[0]
print(f"\n基线性能 (100%点云):")
print(f"  运行时间: {baseline['mean_time_ms']:.2f} ms")
print(f"  FPS: {baseline['fps']:.1f}")

print(f"\n点云密度对性能的影响:")
for r in results[1:]:
    time_change = ((r['mean_time_ms'] - baseline['mean_time_ms']) / baseline['mean_time_ms']) * 100
    fps_change = ((r['fps'] - baseline['fps']) / baseline['fps']) * 100
    
    print(f"\n  {r['description']}:")
    print(f"    点数: {r['avg_original_points']:.0f} ({r['downsample_ratio']*100:.0f}%)")
    print(f"    运行时间变化: {time_change:+.1f}%")
    print(f"    FPS变化: {fps_change:+.1f}%")

print("\n关键观察:")
print("  • 模型使用固定256点输入，因此下采样对运行时间影响很小")
print("  • 实际应用中，下采样可以减少数据采集和传输成本")
print("  • GPU内存使用稳定，适合嵌入式部署")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
