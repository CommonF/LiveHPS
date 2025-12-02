"""
Temporal Window Length Analysis for LiveHPS
测试不同时间窗口长度对准确性、平滑度和运行时/内存的影响
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from models import LiveHPS
from smpl import SMPL


def rotation_6d_to_matrix(d6):
    """将6D旋转表示转为旋转矩阵"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_axis_angle(matrix):
    """旋转矩阵转轴角"""
    return torch.from_numpy(
        R.from_matrix(matrix.cpu().numpy()).as_rotvec().astype(np.float32)
    )


def gen_smpl(smpl, rot, shape, device):
    """生成SMPL网格"""
    num = int(rot.shape[0] / shape.shape[0])
    rot = matrix_to_axis_angle(rotation_6d_to_matrix(rot).view(-1, 3, 3)).reshape(-1, 72)
    pose_b = rot[:, 3:].float()
    g_r = rot[:, :3].float()
    shape = shape.reshape(-1, 1, 10).repeat([1, num, 1]).reshape(-1, 10).float()
    zeros = np.zeros((g_r.shape[0], 3))
    transl_blob = torch.from_numpy(zeros).float().to(device)
    mesh = smpl(betas=shape.to(device), body_pose=pose_b.to(device),
                global_orient=g_r.to(device), transl=transl_blob)
    joints = mesh.joints[:, :24, :]
    v = mesh.vertices - joints[:, 0, :].unsqueeze(1)
    j = joints - joints[:, 0, :].unsqueeze(1)
    return v, j


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


def matrix_to_rotation_6d(matrix):
    """将旋转矩阵转为6D表示"""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def load_sequence_data(sequence_id, start_frame, num_frames, num_points=256):
    """加载序列数据"""
    base_path = Path(f"dataset/lidarhuman26M")
    seg_path = base_path / "labels" / "3d" / "segment" / str(sequence_id)
    pose_path = base_path / "labels" / "3d" / "pose" / str(sequence_id)
    
    all_points = []
    all_gt_poses = []
    all_gt_shapes = []
    all_gt_trans = []
    
    for frame_idx in range(start_frame, start_frame + num_frames):
        # 加载点云
        ply_file = seg_path / f"{frame_idx:06d}.ply"
        if not ply_file.exists():
            continue
        
        points = load_ply_points(ply_file)
        loc = points.mean(0)
        points = points - loc
        points = farthest_point_sample(points, num_points)
        
        # 加载GT pose
        pose_file = pose_path / f"{frame_idx:06d}.json"
        if not pose_file.exists():
            continue
        
        with open(pose_file, 'r') as f:
            meta = json.load(f)
        
        pose = np.array(meta.get('pose', np.zeros(72)), dtype=np.float32)
        shape = np.array(meta.get('shape', np.zeros(10)), dtype=np.float32)
        trans = np.array(meta.get('trans', [0, 0, 0]), dtype=np.float32)
        
        all_points.append(points)
        all_gt_poses.append(pose)
        all_gt_shapes.append(shape)
        all_gt_trans.append(trans - loc)
    
    return {
        'points': np.array(all_points),
        'poses': np.array(all_gt_poses),
        'shapes': np.array(all_gt_shapes),
        'trans': np.array(all_gt_trans)
    }


def run_inference_with_window(model, smpl_model, points, window_size, device):
    """使用指定窗口大小运行推理"""
    model.eval()
    smpl = smpl_model  # 简化变量名
    
    num_frames = len(points)
    all_pred_joints = []
    all_pred_verts = []
    inference_times = []
    peak_memories = []
    
    # 滑窗推理
    for i in range(0, num_frames, window_size):
        window_points = points[i:i + window_size]
        actual_window = len(window_points)
        
        # 准备输入 [1, T, N, 3]
        batch_points = torch.from_numpy(window_points).float().unsqueeze(0).to(device)
        
        # 重置内存统计
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        
        # 计时推理
        start = time.perf_counter()
        
        with torch.no_grad():
            pred_kp, pred_rot, pred_shape, pred_trans = model(batch_points)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)
        
        # 记录内存
        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_memories.append(peak_mem)
        
        # SMPL解码
        B, T = pred_rot.shape[0], pred_rot.shape[1]
        pred_rot_flat = pred_rot.reshape(B * T, 24, 6)
        pred_shape_flat = pred_shape.unsqueeze(1).repeat(1, T, 1).reshape(B * T, -1)
        
        with torch.no_grad():
            pred_v, pred_j = gen_smpl(smpl, pred_rot_flat, pred_shape_flat, device)
        
        pred_j = pred_j.cpu().numpy().reshape(B, T, 24, 3)
        pred_v = pred_v.cpu().numpy().reshape(B, T, -1, 3)
        pred_trans_np = pred_trans.cpu().numpy()
        
        # 应用平移
        pred_j += pred_trans_np
        pred_v += pred_trans_np
        
        all_pred_joints.append(pred_j[0])
        all_pred_verts.append(pred_v[0])
    
    pred_joints = np.concatenate(all_pred_joints, axis=0)
    pred_verts = np.concatenate(all_pred_verts, axis=0)
    
    return {
        'pred_joints': pred_joints,
        'pred_verts': pred_verts,
        'inference_times': inference_times,
        'peak_memories': peak_memories
    }


def compute_metrics(pred_joints, pred_verts, gt_joints, gt_verts):
    """计算MPJPE和MPVPE"""
    # 对齐长度
    min_len = min(len(pred_joints), len(gt_joints))
    pred_joints = pred_joints[:min_len]
    pred_verts = pred_verts[:min_len]
    gt_joints = gt_joints[:min_len]
    gt_verts = gt_verts[:min_len]
    
    # MPJPE
    mpjpe = np.linalg.norm(pred_joints - gt_joints, axis=-1).mean() * 1000  # mm
    
    # MPVPE
    mpvpe = np.linalg.norm(pred_verts - gt_verts, axis=-1).mean() * 1000  # mm
    
    return mpjpe, mpvpe


def compute_smoothness_metrics(joints):
    """计算平滑度指标（加速度的方差）"""
    # 计算速度（一阶差分）
    velocity = np.diff(joints, axis=0)  # [T-1, 24, 3]
    
    # 计算加速度（二阶差分）
    acceleration = np.diff(velocity, axis=0)  # [T-2, 24, 3]
    
    # 加速度的L2范数
    accel_magnitude = np.linalg.norm(acceleration, axis=-1)  # [T-2, 24]
    
    # 平均加速度方差（越小越平滑）
    accel_variance = np.var(accel_magnitude)
    accel_mean = np.mean(accel_magnitude)
    
    return {
        'accel_variance': accel_variance,
        'accel_mean': accel_mean,
        'jerk_metric': accel_variance / (accel_mean + 1e-8)  # 归一化平滑度
    }


def generate_gt_smpl(data, smpl_model, device):
    """生成GT的SMPL网格和关节"""
    poses = data['poses']
    shapes = data['shapes']
    trans = data['trans']
    
    num_frames = len(poses)
    
    # 转换pose为旋转矩阵再转6D
    pose_rotvec = poses.reshape(-1, 3)
    pose_mat = R.from_rotvec(pose_rotvec).as_matrix()
    pose_mat = torch.from_numpy(pose_mat).float().to(device).reshape(num_frames, 24, 3, 3)
    pose_6d = matrix_to_rotation_6d(pose_mat).reshape(num_frames, 24, 6)
    
    shapes_tensor = torch.from_numpy(shapes).float().to(device)
    
    # 使用gen_smpl函数
    gt_verts, gt_joints = gen_smpl(smpl_model, pose_6d, shapes_tensor, device)
    
    gt_joints = gt_joints.cpu().numpy()
    gt_verts = gt_verts.cpu().numpy()
    
    # 应用平移
    gt_joints += trans[:, None, :]
    gt_verts += trans[:, None, :]
    
    return gt_joints, gt_verts


def main():
    print("=" * 80)
    print("时间窗口长度分析 - Temporal Window Length Analysis")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model = LiveHPS().to(device)
    model.eval()
    
    smpl_model = SMPL('./smpl/models/SMPL_NEUTRAL.pkl').to(device)
    
    print("✓ 模型加载完成")
    
    # 测试配置
    sequence_id = 24
    start_frame = 100
    num_frames = 96  # 使用96帧（可以被8,16,32整除）
    
    window_sizes = [8, 16, 32]
    
    print(f"\n加载测试序列: Sequence {sequence_id}, Frames {start_frame}-{start_frame+num_frames-1}")
    data = load_sequence_data(sequence_id, start_frame, num_frames)
    print(f"✓ 加载了 {len(data['points'])} 帧")
    
    # 生成GT
    print("生成GT SMPL...")
    gt_joints, gt_verts = generate_gt_smpl(data, smpl_model, device)
    print("✓ GT生成完成")
    
    results = []
    
    print("\n" + "=" * 80)
    print("开始窗口大小测试")
    print("=" * 80)
    
    for window_size in window_sizes:
        print(f"\n{'='*80}")
        print(f"测试窗口大小: {window_size} 帧")
        print(f"{'='*80}")
        
        # 运行推理
        print(f"运行推理...")
        inference_result = run_inference_with_window(
            model, smpl_model, data['points'], window_size, device
        )
        
        pred_joints = inference_result['pred_joints']
        pred_verts = inference_result['pred_verts']
        
        # 计算准确性指标
        mpjpe, mpvpe = compute_metrics(pred_joints, pred_verts, gt_joints, gt_verts)
        
        # 计算平滑度指标
        smoothness = compute_smoothness_metrics(pred_joints)
        
        # 计算运行时和内存
        inference_times = inference_result['inference_times']
        total_time = sum(inference_times)
        avg_time_per_window = np.mean(inference_times) * 1000  # ms
        avg_time_per_frame = (total_time / len(pred_joints)) * 1000  # ms
        fps = 1000.0 / avg_time_per_frame
        
        peak_memories = inference_result['peak_memories']
        avg_peak_memory = np.mean(peak_memories) if peak_memories else 0
        max_peak_memory = np.max(peak_memories) if peak_memories else 0
        
        result = {
            'window_size': window_size,
            'num_windows': len(inference_times),
            'mpjpe': float(mpjpe),
            'mpvpe': float(mpvpe),
            'accel_variance': float(smoothness['accel_variance']),
            'accel_mean': float(smoothness['accel_mean']),
            'jerk_metric': float(smoothness['jerk_metric']),
            'total_time_s': float(total_time),
            'avg_time_per_window_ms': float(avg_time_per_window),
            'avg_time_per_frame_ms': float(avg_time_per_frame),
            'fps': float(fps),
            'avg_peak_memory_mb': float(avg_peak_memory),
            'max_peak_memory_mb': float(max_peak_memory)
        }
        
        results.append(result)
        
        print(f"\n结果:")
        print(f"  准确性:")
        print(f"    MPJPE: {mpjpe:.2f} mm")
        print(f"    MPVPE: {mpvpe:.2f} mm")
        print(f"  平滑度:")
        print(f"    加速度方差: {smoothness['accel_variance']:.6f}")
        print(f"    加速度均值: {smoothness['accel_mean']:.6f}")
        print(f"    Jerk指标: {smoothness['jerk_metric']:.6f} (越小越平滑)")
        print(f"  性能:")
        print(f"    每窗口时间: {avg_time_per_window:.2f} ms")
        print(f"    每帧时间: {avg_time_per_frame:.2f} ms")
        print(f"    FPS: {fps:.1f}")
        print(f"    峰值GPU内存: {max_peak_memory:.1f} MB")
    
    # 保存结果
    output_dir = Path('outputs/temporal_window_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / 'window_size_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 详细结果已保存到: {json_path}")
    
    # 生成对比图表
    generate_comparison_plots(results, output_dir)
    
    # 生成汇总表格
    print("\n" + "=" * 80)
    print("汇总对比")
    print("=" * 80)
    
    print(f"\n{'窗口大小':<10} {'MPJPE(mm)':<12} {'MPVPE(mm)':<12} {'Jerk指标':<14} "
          f"{'每帧时延(ms)':<15} {'FPS':<8} {'峰值内存(MB)':<15}")
    print("-" * 95)
    
    baseline = results[0]  # 以第一个为基准
    
    for r in results:
        mpjpe_change = ((r['mpjpe'] - baseline['mpjpe']) / baseline['mpjpe']) * 100
        jerk_change = ((r['jerk_metric'] - baseline['jerk_metric']) / baseline['jerk_metric']) * 100
        time_change = ((r['avg_time_per_frame_ms'] - baseline['avg_time_per_frame_ms']) / 
                      baseline['avg_time_per_frame_ms']) * 100
        
        print(f"{r['window_size']:<10} "
              f"{r['mpjpe']:>6.2f} ({mpjpe_change:+5.1f}%)  "
              f"{r['mpvpe']:>6.2f}      "
              f"{r['jerk_metric']:>8.6f} ({jerk_change:+5.1f}%)  "
              f"{r['avg_time_per_frame_ms']:>7.2f} ({time_change:+5.1f}%)  "
              f"{r['fps']:>6.1f}  "
              f"{r['max_peak_memory_mb']:>13.1f}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


def generate_comparison_plots(results, output_dir):
    """生成对比图表"""
    window_sizes = [r['window_size'] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. MPJPE
    ax = axes[0, 0]
    mpjpe_vals = [r['mpjpe'] for r in results]
    bars = ax.bar(window_sizes, mpjpe_vals, color='steelblue', alpha=0.7)
    ax.set_xlabel('Window Size (frames)', fontweight='bold')
    ax.set_ylabel('MPJPE (mm)', fontweight='bold')
    ax.set_title('Accuracy: MPJPE vs Window Size', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 2. MPVPE
    ax = axes[0, 1]
    mpvpe_vals = [r['mpvpe'] for r in results]
    bars = ax.bar(window_sizes, mpvpe_vals, color='coral', alpha=0.7)
    ax.set_xlabel('Window Size (frames)', fontweight='bold')
    ax.set_ylabel('MPVPE (mm)', fontweight='bold')
    ax.set_title('Accuracy: MPVPE vs Window Size', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 3. Jerk指标（平滑度）
    ax = axes[0, 2]
    jerk_vals = [r['jerk_metric'] for r in results]
    bars = ax.bar(window_sizes, jerk_vals, color='lightgreen', alpha=0.7)
    ax.set_xlabel('Window Size (frames)', fontweight='bold')
    ax.set_ylabel('Jerk Metric (lower=smoother)', fontweight='bold')
    ax.set_title('Smoothness: Jerk Metric vs Window Size', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 每帧时延
    ax = axes[1, 0]
    time_vals = [r['avg_time_per_frame_ms'] for r in results]
    bars = ax.bar(window_sizes, time_vals, color='orange', alpha=0.7)
    ax.set_xlabel('Window Size (frames)', fontweight='bold')
    ax.set_ylabel('Time per Frame (ms)', fontweight='bold')
    ax.set_title('Runtime: Per-Frame Latency vs Window Size', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 5. FPS
    ax = axes[1, 1]
    fps_vals = [r['fps'] for r in results]
    bars = ax.bar(window_sizes, fps_vals, color='purple', alpha=0.7)
    ax.set_xlabel('Window Size (frames)', fontweight='bold')
    ax.set_ylabel('FPS', fontweight='bold')
    ax.set_title('Throughput: FPS vs Window Size', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 6. 峰值内存
    ax = axes[1, 2]
    mem_vals = [r['max_peak_memory_mb'] for r in results]
    bars = ax.bar(window_sizes, mem_vals, color='crimson', alpha=0.7)
    ax.set_xlabel('Window Size (frames)', fontweight='bold')
    ax.set_ylabel('Peak GPU Memory (MB)', fontweight='bold')
    ax.set_title('Memory: Peak Usage vs Window Size', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'window_size_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图表已保存到: {plot_path}")
    
    plt.close()


if __name__ == '__main__':
    main()
