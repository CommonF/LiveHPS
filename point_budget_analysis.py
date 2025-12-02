"""
Point Budget Analysis for LiveHPS
分析不同点云密度预算对模型性能的影响

测试场景:
1. 人体区域点云数量变化: 64, 128, 256, 512, 1024, 2048 points
2. 分析准确性、推理速度、GPU内存的权衡关系
"""

import os
import json
import numpy as np
import torch
import cv2
import argparse
from tqdm import tqdm
from pytorch3d.transforms import *
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys

sys.path.append("./smpl")
from smpl import SMPL, SMPL_MODEL_DIR
from models import LiveHPS

def farthest_point_sample(xyz, npoint):
    """FPS采样"""
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

def random_downsample(xyz, npoint):
    """随机下采样"""
    ndataset = xyz.shape[0]
    if ndataset == 0:
        return np.zeros((npoint, 3), dtype=np.float32)
    if ndataset <= npoint:
        return farthest_point_sample(xyz, npoint)
    indices = np.random.choice(ndataset, npoint, replace=False)
    return xyz[indices]

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

def load_sequence_data(sequence_id, start_frame, num_frames, num_points=256, sampling_method='fps'):
    """
    加载序列数据
    sampling_method: 'fps' (Farthest Point Sampling) or 'random' (Random Downsampling)
    """
    base_path = f"./dataset/lidarhuman26M"
    seg_path = f"{base_path}/labels/3d/segment/{sequence_id}"
    pose_path = f"{base_path}/labels/3d/pose/{sequence_id}"
    
    point_clouds = []
    poses = []
    shapes = []
    trans_list = []
    frame_ids = []
    
    print(f"Loading {num_frames} frames with {num_points} points per frame ({sampling_method} sampling)...")
    
    for i in tqdm(range(start_frame, start_frame + num_frames), desc="Loading frames"):
        frame_name = f"{i:06d}"
        ply_file = f"{seg_path}/{frame_name}.ply"
        json_file = f"{pose_path}/{frame_name}.json"
        
        if not os.path.exists(ply_file) or not os.path.exists(json_file):
            continue
        
        # 加载点云
        points = load_ply_points(ply_file)
        
        if len(points) == 0:
            continue
        
        # 先中心化（与训练时一致）
        centroid = points.mean(axis=0)
        points = points - centroid
        
        # 根据采样方法处理
        if sampling_method == 'fps':
            points = farthest_point_sample(points, num_points)
        elif sampling_method == 'random':
            points = random_downsample(points, num_points)
        
        point_clouds.append(points)
        
        # 加载SMPL参数
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        pose = np.array(data.get('pose', np.zeros(72)), dtype=np.float32)
        shape = np.array(data.get('beta', np.zeros(10)), dtype=np.float32)
        trans = np.array(data.get('trans', np.zeros(3)), dtype=np.float32)
        
        poses.append(pose)
        shapes.append(shape)
        trans_list.append(trans)
        frame_ids.append(f"{sequence_id}/{frame_name}.ply")
    
    if len(point_clouds) == 0:
        return None
    
    return {
        'point_clouds': np.array(point_clouds),
        'poses': np.array(poses),
        'shapes': np.array(shapes),
        'trans': np.array(trans_list),
        'frame_ids': frame_ids,
        'num_points': num_points,
        'sampling_method': sampling_method
    }

def gen_smpl(smpl, rot, shape, device, transl=None):
    """生成SMPL网格"""
    num = int(rot.shape[0] / shape.shape[0])
    rot = matrix_to_axis_angle(rotation_6d_to_matrix(rot).view(-1, 3, 3)).reshape(-1, 72)
    pose_b = rot[:, 3:].float()
    g_r = rot[:, :3].float()
    shape = shape.reshape(-1, 1, 10).repeat([1, num, 1]).reshape(-1, 10).float()
    
    if transl is None:
        zeros = np.zeros((g_r.shape[0], 3))
        transl_blob = torch.from_numpy(zeros).float().to(device)
    else:
        transl_blob = transl.float().to(device)
    
    mesh = smpl(betas=shape.to(device), body_pose=pose_b.to(device),
                global_orient=g_r.to(device), transl=transl_blob)
    joints = mesh.joints[:, :24, :]
    v = mesh.vertices - joints[:, 0, :].unsqueeze(1)
    j = joints - joints[:, 0, :].unsqueeze(1)
    return v, j

def run_inference_with_timing(model, smpl, seq_data, device, temporal_window=32, warmup_runs=3):
    """运行推理并测量性能"""
    point_clouds = seq_data['point_clouds']
    poses_gt = seq_data['poses']
    shapes_gt = seq_data['shapes']
    original_num_points = seq_data['num_points']
    
    num_frames = len(point_clouds)
    pred_vertices_all = []
    pred_joints_all = []
    gt_vertices_all = []
    gt_joints_all = []
    
    inference_times = []
    
    model.eval()
    
    # 模型要求输入必须是256个点，所以需要重采样
    MODEL_INPUT_POINTS = 256
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            pc_window = point_clouds[:temporal_window]
            if len(pc_window) < temporal_window:
                pad_size = temporal_window - len(pc_window)
                pc_window = np.concatenate([pc_window, 
                                          np.tile(pc_window[-1:], (pad_size, 1, 1))], axis=0)
            # 如果点数不是256，重采样到256
            if original_num_points != MODEL_INPUT_POINTS:
                pc_window_resampled = np.array([farthest_point_sample(pc, MODEL_INPUT_POINTS) for pc in pc_window])
            else:
                pc_window_resampled = pc_window
            pc_tensor = torch.from_numpy(pc_window_resampled).unsqueeze(0).to(device).float()
            _ = model(pc_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print("Running inference...")
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_frames, temporal_window), desc="Inference"):
            end_idx = min(start_idx + temporal_window, num_frames)
            window_size = end_idx - start_idx
            
            # 准备输入
            pc_window = point_clouds[start_idx:end_idx]
            if window_size < temporal_window:
                pad_size = temporal_window - window_size
                pc_window = np.concatenate([pc_window, 
                                          np.tile(pc_window[-1:], (pad_size, 1, 1))], axis=0)
            
            # 如果点数不是256，重采样到256
            if original_num_points != MODEL_INPUT_POINTS:
                pc_window_resampled = np.array([farthest_point_sample(pc, MODEL_INPUT_POINTS) for pc in pc_window])
            else:
                pc_window_resampled = pc_window
            
            pc_tensor = torch.from_numpy(pc_window_resampled).unsqueeze(0).to(device).float()
            
            # 计时推理
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            _, rot, shape, pre_trans = model(pc_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            # 生成SMPL
            B = 1
            T = temporal_window
            pre_v, pre_j = gen_smpl(smpl, rot.reshape(B*T, -1, 6), shape, device)
            
            pred_vertices_all.append(pre_v[:window_size].cpu().numpy())
            pred_joints_all.append(pre_j[:window_size].cpu().numpy())
            
            # 生成GT SMPL - 使用每帧的shape参数
            poses_window = poses_gt[start_idx:end_idx]
            shapes_window = shapes_gt[start_idx:end_idx]
            
            poses_np = poses_window.reshape(-1, 3)
            gt_pose_mat = torch.from_numpy(R.from_rotvec(poses_np).as_matrix()).to(device).view(window_size, 24, 3, 3)
            gt_pose_6d = matrix_to_rotation_6d(gt_pose_mat).reshape(window_size, 24, 6)
            
            gt_shape_tensor = torch.from_numpy(shapes_window).to(device)
            
            gt_v, gt_j = gen_smpl(smpl, gt_pose_6d, gt_shape_tensor, device)
            gt_vertices_all.append(gt_v.cpu().numpy())
            gt_joints_all.append(gt_j.cpu().numpy())
    
    # 合并结果
    pred_vertices = np.concatenate(pred_vertices_all, axis=0)
    pred_joints = np.concatenate(pred_joints_all, axis=0)
    gt_vertices = np.concatenate(gt_vertices_all, axis=0)
    gt_joints = np.concatenate(gt_joints_all, axis=0)
    
    # 计算性能指标
    avg_inference_time = np.mean(inference_times)
    fps = temporal_window / avg_inference_time
    latency_per_frame = (avg_inference_time / temporal_window) * 1000  # ms
    
    return pred_vertices, pred_joints, gt_vertices, gt_joints, {
        'avg_inference_time': avg_inference_time,
        'fps': fps,
        'latency_per_frame': latency_per_frame,
        'inference_times': inference_times
    }

def calculate_metrics(pred_joints, pred_vertices, gt_joints, gt_vertices):
    """计算评估指标"""
    mpjpe = np.linalg.norm(pred_joints - gt_joints, axis=2).mean()
    mpvpe = np.linalg.norm(pred_vertices - gt_vertices, axis=2).mean()
    
    # 计算加速度（平滑度）
    def calculate_acceleration(joints):
        velocity = np.diff(joints, axis=0)
        acceleration = np.diff(velocity, axis=0)
        return np.linalg.norm(acceleration, axis=2).mean()
    
    pred_accel = calculate_acceleration(pred_joints)
    gt_accel = calculate_acceleration(gt_joints)
    
    return {
        'mpjpe': mpjpe * 1000,
        'mpvpe': mpvpe * 1000,
        'pred_acceleration': pred_accel * 1000,
        'gt_acceleration': gt_accel * 1000,
        'mpjpe_per_frame': np.linalg.norm(pred_joints - gt_joints, axis=2).mean(axis=1) * 1000,
    }

def render_comparison_frame(pred_vertices, gt_vertices, frame_idx, title=""):
    """渲染单帧对比图"""
    fig = plt.figure(figsize=(16, 8))
    
    # 计算统一的边界
    all_verts = np.vstack([pred_vertices.reshape(-1, 3), gt_vertices.reshape(-1, 3)])
    xmin, xmax = all_verts[:, 0].min() - 0.1, all_verts[:, 0].max() + 0.1
    ymin, ymax = all_verts[:, 1].min() - 0.1, all_verts[:, 1].max() + 0.1
    zmin, zmax = all_verts[:, 2].min() - 0.1, all_verts[:, 2].max() + 0.1
    
    # 加载SMPL面片
    try:
        smpl = SMPL(SMPL_MODEL_DIR, create_transl=False)
        smpl_faces = smpl.faces if hasattr(smpl, 'faces') else None
    except:
        smpl_faces = None
    
    # 左图：预测
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    ax1.set_zlim([zmin, zmax])
    ax1.view_init(elev=15, azim=-60)
    ax1.set_title(f'Prediction {title}', fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    pred_v = pred_vertices[frame_idx]
    if smpl_faces is not None:
        ax1.plot_trisurf(pred_v[:, 0], pred_v[:, 1], pred_v[:, 2],
                       triangles=smpl_faces, color='lightblue',
                       edgecolor='none', alpha=0.9, shade=True)
    else:
        ax1.scatter(pred_v[::5, 0], pred_v[::5, 1], pred_v[::5, 2],
                   c='lightblue', s=3, alpha=0.8)
    
    # 右图：GT
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([ymin, ymax])
    ax2.set_zlim([zmin, zmax])
    ax2.view_init(elev=15, azim=-60)
    ax2.set_title(f'Ground Truth (Frame {frame_idx})', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    gt_v = gt_vertices[frame_idx]
    if smpl_faces is not None:
        ax2.plot_trisurf(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2],
                       triangles=smpl_faces, color='lightgreen',
                       edgecolor='none', alpha=0.9, shade=True)
    else:
        ax2.scatter(gt_v[::5, 0], gt_v[::5, 1], gt_v[::5, 2],
                   c='lightgreen', s=3, alpha=0.8)
    
    return fig

def plot_results_comparison(results_dict, output_dir):
    """绘制结果对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    point_budgets = sorted([int(k.split('_')[0]) for k in results_dict.keys()])
    
    # 提取数据
    mpjpe_values = [results_dict[f'{n}_points']['mpjpe'] for n in point_budgets]
    mpvpe_values = [results_dict[f'{n}_points']['mpvpe'] for n in point_budgets]
    fps_values = [results_dict[f'{n}_points']['fps'] for n in point_budgets]
    latency_values = [results_dict[f'{n}_points']['latency_per_frame'] for n in point_budgets]
    accel_values = [results_dict[f'{n}_points']['pred_acceleration'] for n in point_budgets]
    
    # 1. MPJPE vs Point Budget
    ax = axes[0, 0]
    ax.plot(point_budgets, mpjpe_values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Points per Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('MPJPE (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Joint Position Error vs Point Budget', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    for i, (x, y) in enumerate(zip(point_budgets, mpjpe_values)):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # 2. MPVPE vs Point Budget
    ax = axes[0, 1]
    ax.plot(point_budgets, mpvpe_values, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Points per Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('MPVPE (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Vertex Position Error vs Point Budget', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    for i, (x, y) in enumerate(zip(point_budgets, mpvpe_values)):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # 3. FPS vs Point Budget
    ax = axes[0, 2]
    ax.plot(point_budgets, fps_values, 's-', linewidth=2, markersize=8, color='#F18F01')
    ax.set_xlabel('Points per Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax.set_title('Throughput vs Point Budget', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    for i, (x, y) in enumerate(zip(point_budgets, fps_values)):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # 4. Latency vs Point Budget
    ax = axes[1, 0]
    ax.plot(point_budgets, latency_values, 'd-', linewidth=2, markersize=8, color='#C73E1D')
    ax.set_xlabel('Points per Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms/frame)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Frame Latency vs Point Budget', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    for i, (x, y) in enumerate(zip(point_budgets, latency_values)):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # 5. Acceleration (Smoothness) vs Point Budget
    ax = axes[1, 1]
    ax.plot(point_budgets, accel_values, '^-', linewidth=2, markersize=8, color='#6A994E')
    ax.set_xlabel('Points per Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acceleration (mm/frame²)', fontsize=12, fontweight='bold')
    ax.set_title('Motion Smoothness vs Point Budget', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    for i, (x, y) in enumerate(zip(point_budgets, accel_values)):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    # 6. Accuracy-Efficiency Trade-off
    ax = axes[1, 2]
    scatter = ax.scatter(latency_values, mpjpe_values, s=[n*0.5 for n in point_budgets], 
                        c=point_budgets, cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1.5)
    for i, n in enumerate(point_budgets):
        ax.annotate(f'{n}', (latency_values[i], mpjpe_values[i]), 
                   textcoords="offset points", xytext=(5,5), ha='left', fontsize=9, fontweight='bold')
    ax.set_xlabel('Latency (ms/frame)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MPJPE (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy-Efficiency Trade-off', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Points per Frame', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'point_budget_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Analysis plot -> {os.path.join(output_dir, 'point_budget_analysis.png')}")

def main():
    parser = argparse.ArgumentParser(description='LiveHPS Point Budget Analysis')
    parser.add_argument('--sequence_id', type=str, default='24', help='Sequence ID')
    parser.add_argument('--start_frame', type=int, default=100, help='Starting frame')
    parser.add_argument('--num_frames', type=int, default=50, help='Number of frames')
    parser.add_argument('--point_budgets', type=int, nargs='+', 
                       default=[64, 128, 256, 512, 1024],
                       help='Point budgets to test')
    parser.add_argument('--sampling_method', type=str, default='fps', 
                       choices=['fps', 'random'], help='Sampling method')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--pretrained', type=str, default='save_models/livehps.t7')
    parser.add_argument('--output_dir', type=str, default='outputs/point_budget_analysis')
    parser.add_argument('--temporal_window', type=int, default=32)
    parser.add_argument('--screenshot_frame', type=int, default=25, 
                       help='Frame index for screenshot comparison')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("LiveHPS Point Budget Analysis")
    print("="*80)
    print(f"Point budgets to test: {args.point_budgets}")
    print(f"Sampling method: {args.sampling_method}")
    print(f"Device: {device}")
    
    # 加载模型
    print("\nLoading SMPL model...")
    smpl = SMPL(SMPL_MODEL_DIR, create_transl=False).to(device)
    
    print("Loading LiveHPS model...")
    model = LiveHPS()
    if os.path.exists(args.pretrained):
        checkpoint = torch.load(args.pretrained, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {args.pretrained}")
    model.to(device)
    model.eval()
    
    results = {}
    screenshot_data = {}
    
    # 对每个点云预算运行测试
    for num_points in args.point_budgets:
        print("\n" + "="*80)
        print(f"Testing with {num_points} points per frame")
        print("="*80)
        
        # 加载数据
        seq_data = load_sequence_data(
            args.sequence_id,
            args.start_frame,
            args.num_frames,
            num_points,
            args.sampling_method
        )
        
        if seq_data is None:
            print(f"Failed to load data for {num_points} points")
            continue
        
        print(f"Loaded {len(seq_data['point_clouds'])} frames")
        
        # 运行推理
        pred_v, pred_j, gt_v, gt_j, timing_info = run_inference_with_timing(
            model, smpl, seq_data, device, args.temporal_window
        )
        
        # 计算指标
        metrics = calculate_metrics(pred_j, pred_v, gt_j, gt_v)
        
        # 合并结果
        results[f'{num_points}_points'] = {
            **metrics,
            **timing_info
        }
        
        print(f"\nResults for {num_points} points:")
        print(f"  MPJPE: {metrics['mpjpe']:.2f} mm")
        print(f"  MPVPE: {metrics['mpvpe']:.2f} mm")
        print(f"  FPS: {timing_info['fps']:.2f}")
        print(f"  Latency: {timing_info['latency_per_frame']:.2f} ms/frame")
        print(f"  Acceleration: {metrics['pred_acceleration']:.2f} mm/frame²")
        
        # 保存结果
        output_npz = os.path.join(args.output_dir, f'{num_points}_points_results.npz')
        np.savez(output_npz,
                pred_vertices=pred_v,
                pred_joints=pred_j,
                gt_vertices=gt_v,
                gt_joints=gt_j,
                **metrics,
                **timing_info)
        print(f"Saved results to {output_npz}")
        
        # 保存截图数据
        screenshot_data[num_points] = {
            'pred_v': pred_v,
            'gt_v': gt_v,
            'mpjpe': metrics['mpjpe']
        }
    
    # 生成对比图表
    print("\n" + "="*80)
    print("Generating comparison plots...")
    print("="*80)
    plot_results_comparison(results, args.output_dir)
    
    # 生成截图对比
    print("\nGenerating screenshot comparisons...")
    screenshot_dir = os.path.join(args.output_dir, 'screenshots')
    os.makedirs(screenshot_dir, exist_ok=True)
    
    for num_points in args.point_budgets:
        if num_points not in screenshot_data:
            continue
        data = screenshot_data[num_points]
        frame_idx = min(args.screenshot_frame, len(data['pred_v']) - 1)
        
        fig = render_comparison_frame(
            data['pred_v'], 
            data['gt_v'], 
            frame_idx,
            title=f"({num_points} points, MPJPE={data['mpjpe']:.1f}mm)"
        )
        
        output_path = os.path.join(screenshot_dir, f'{num_points}_points_frame_{frame_idx}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved] Screenshot -> {output_path}")
    
    # 保存JSON结果
    results_json = {}
    for key, value in results.items():
        results_json[key] = {
            'mpjpe': float(value['mpjpe']),
            'mpvpe': float(value['mpvpe']),
            'fps': float(value['fps']),
            'latency_per_frame': float(value['latency_per_frame']),
            'pred_acceleration': float(value['pred_acceleration']),
            'gt_acceleration': float(value['gt_acceleration']),
        }
    
    json_path = os.path.join(args.output_dir, 'point_budget_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[Saved] JSON results -> {json_path}")
    
    # 输出汇总报告
    print("\n" + "="*80)
    print("POINT BUDGET ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nSequence: {args.sequence_id}, Frames: {args.start_frame}-{args.start_frame+args.num_frames}")
    print(f"Sampling: {args.sampling_method.upper()}")
    print(f"\n{'Points':<10} {'MPJPE (mm)':<15} {'MPVPE (mm)':<15} {'FPS':<10} {'Latency (ms)':<15} {'Accel (mm/f²)'}")
    print("-" * 95)
    
    for num_points in args.point_budgets:
        key = f'{num_points}_points'
        if key in results:
            r = results[key]
            print(f"{num_points:<10} {r['mpjpe']:<15.2f} {r['mpvpe']:<15.2f} "
                  f"{r['fps']:<10.2f} {r['latency_per_frame']:<15.2f} {r['pred_acceleration']:<15.2f}")
    
    print("\n" + "="*80)
    print("Recommendations:")
    print("-" * 80)
    
    # 找到最佳权衡点
    min_mpjpe = min([results[f'{n}_points']['mpjpe'] for n in args.point_budgets])
    max_fps = max([results[f'{n}_points']['fps'] for n in args.point_budgets])
    
    # 归一化分数（越低越好）
    scores = {}
    for num_points in args.point_budgets:
        key = f'{num_points}_points'
        mpjpe_score = results[key]['mpjpe'] / min_mpjpe
        fps_score = max_fps / results[key]['fps']
        scores[num_points] = (mpjpe_score + fps_score) / 2
    
    best_tradeoff = min(scores, key=scores.get)
    
    print(f"• Best accuracy: {args.point_budgets[-1]} points "
          f"(MPJPE={results[f'{args.point_budgets[-1]}_points']['mpjpe']:.2f}mm)")
    print(f"• Best speed: {args.point_budgets[0]} points "
          f"(FPS={results[f'{args.point_budgets[0]}_points']['fps']:.2f})")
    print(f"• Best trade-off: {best_tradeoff} points "
          f"(MPJPE={results[f'{best_tradeoff}_points']['mpjpe']:.2f}mm, "
          f"FPS={results[f'{best_tradeoff}_points']['fps']:.2f})")
    
    print("\n" + "="*80)
    print("Files generated:")
    print(f"  - point_budget_analysis.png")
    print(f"  - point_budget_results.json")
    for num_points in args.point_budgets:
        print(f"  - {num_points}_points_results.npz")
        print(f"  - screenshots/{num_points}_points_frame_{args.screenshot_frame}.png")
    print("="*80)

if __name__ == "__main__":
    main()
