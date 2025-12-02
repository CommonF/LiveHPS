"""
Robustness/Failure Test for LiveHPS
对序列进行控制性降质处理以测试模型鲁棒性和失败行为

测试类型:
1. 点云下采样 (Random Point Downsampling) - 模拟更稀疏的LiDAR
2. 帧丢弃 (Frame Dropping) - 模拟更低的帧率
3. 局部杂波/遮挡 (Localized Clutter/Occluders) - 模拟环境干扰

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

def apply_random_downsampling(points, ratio=0.5):
    """随机下采样点云"""
    n_points = len(points)
    n_keep = int(n_points * ratio)
    if n_keep == 0:
        return np.zeros((0, 3), dtype=np.float32)
    indices = np.random.choice(n_points, n_keep, replace=False)
    return points[indices]

def apply_frame_dropping(frame_list, keep_ratio=0.5):
    """帧丢弃 - 保留每隔N帧"""
    step = int(1.0 / keep_ratio)
    return frame_list[::step]

def add_localized_clutter(points, clutter_ratio=0.3, clutter_region='random'):
    """添加局部杂波/遮挡物"""
    n_points = len(points)
    n_clutter = int(n_points * clutter_ratio)
    
    if clutter_region == 'random':
        # 在随机位置添加杂波
        center = points[np.random.randint(len(points))]
        radius = 0.3
        clutter = center + np.random.randn(n_clutter, 3) * radius
    elif clutter_region == 'front':
        # 在前方添加遮挡物（模拟物体遮挡）
        center = points.mean(axis=0)
        center[2] += 0.5  # Z方向前移
        clutter = center + np.random.randn(n_clutter, 3) * 0.2
    elif clutter_region == 'around':
        # 周围添加环境杂波
        bounds_min = points.min(axis=0) - 0.5
        bounds_max = points.max(axis=0) + 0.5
        clutter = np.random.uniform(bounds_min, bounds_max, (n_clutter, 3))
    
    return np.vstack([points, clutter])

def load_sequence_data(sequence_id, start_frame, num_frames, num_points=256,
                      degradation_type='none', degradation_params=None):
    """
    加载序列数据并应用降质
    
    degradation_type: 'none', 'downsample', 'frame_drop', 'clutter', 'combined'
    degradation_params: dict with specific parameters for each degradation
    """
    base_path = f"./dataset/lidarhuman26M"
    seg_path = f"{base_path}/labels/3d/segment/{sequence_id}"
    pose_path = f"{base_path}/labels/3d/pose/{sequence_id}"
    
    if degradation_params is None:
        degradation_params = {}
    
    # 收集所有帧
    all_frames = []
    for i in range(start_frame, start_frame + num_frames * 3):  # 读取更多帧以防frame_drop
        frame_name = f"{i:06d}"
        ply_file = f"{seg_path}/{frame_name}.ply"
        json_file = f"{pose_path}/{frame_name}.json"
        
        if not os.path.exists(ply_file) or not os.path.exists(json_file):
            continue
        
        all_frames.append((i, ply_file, json_file))
        if len(all_frames) >= num_frames * 3:
            break
    
    # 应用帧丢弃
    if degradation_type in ['frame_drop', 'combined']:
        frame_ratio = degradation_params.get('frame_keep_ratio', 0.5)
        all_frames = apply_frame_dropping(all_frames, frame_ratio)
    
    # 限制到需要的帧数
    all_frames = all_frames[:num_frames]
    
    point_clouds = []
    poses = []
    shapes = []
    trans_list = []
    frame_ids = []
    
    print(f"Processing {len(all_frames)} frames with degradation: {degradation_type}")
    
    for frame_num, ply_file, json_file in tqdm(all_frames, desc="Loading frames"):
        # 加载点云
        points = load_ply_points(ply_file)
        
        if len(points) == 0:
            continue
        
        # 应用点云降质
        if degradation_type in ['downsample', 'combined']:
            downsample_ratio = degradation_params.get('downsample_ratio', 0.5)
            points = apply_random_downsampling(points, downsample_ratio)
        
        # 添加杂波
        if degradation_type in ['clutter', 'combined']:
            clutter_ratio = degradation_params.get('clutter_ratio', 0.3)
            clutter_region = degradation_params.get('clutter_region', 'random')
            points = add_localized_clutter(points, clutter_ratio, clutter_region)
        
        # 采样到固定数量，并按训练时的方式中心化
        if len(points) == 0:
            points = np.zeros((num_points, 3), dtype=np.float32)
        else:
            # 先中心化（与训练时一致）
            centroid = points.mean(axis=0)
            points = points - centroid
            # 然后FPS采样
            points = farthest_point_sample(points, num_points)
        
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
        frame_ids.append(f"{sequence_id}/{frame_num:06d}.ply")
    
    if len(point_clouds) == 0:
        return None
    
    return {
        'point_clouds': np.array(point_clouds),
        'poses': np.array(poses),
        'shapes': np.array(shapes),
        'trans': np.array(trans_list),
        'frame_ids': frame_ids,
        'degradation': degradation_type,
        'params': degradation_params
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

def run_inference(model, smpl, seq_data, device, temporal_window=32):
    """运行推理"""
    point_clouds = seq_data['point_clouds']
    poses_gt = seq_data['poses']
    shapes_gt = seq_data['shapes']
    
    num_frames = len(point_clouds)
    pred_vertices_all = []
    pred_joints_all = []
    gt_vertices_all = []
    gt_joints_all = []
    
    model.eval()
    
    with torch.no_grad():
        # 滑动窗口处理
        for start_idx in tqdm(range(0, num_frames, temporal_window), desc="Inference"):
            end_idx = min(start_idx + temporal_window, num_frames)
            window_size = end_idx - start_idx
            
            # 准备输入
            pc_window = point_clouds[start_idx:end_idx]
            if window_size < temporal_window:
                # 填充
                pad_size = temporal_window - window_size
                pc_window = np.concatenate([pc_window, 
                                          np.tile(pc_window[-1:], (pad_size, 1, 1))], axis=0)
            
            pc_tensor = torch.from_numpy(pc_window).unsqueeze(0).to(device).float()
            
            # 推理
            _, rot, shape, pre_trans = model(pc_tensor)
            
            # 生成SMPL
            B = 1
            T = temporal_window
            pre_v, pre_j = gen_smpl(smpl, rot.reshape(B*T, -1, 6), shape, device)
            
            # 存储预测（仅有效帧）
            pred_vertices_all.append(pre_v[:window_size].cpu().numpy())
            pred_joints_all.append(pre_j[:window_size].cpu().numpy())
            
            # 生成GT SMPL - 使用每帧的shape参数
            poses_window = poses_gt[start_idx:end_idx]
            shapes_window = shapes_gt[start_idx:end_idx]
            
            # 将 GT pose (axis-angle) 转换为 6D rotation
            poses_np = poses_window.reshape(-1, 3)
            gt_pose_mat = torch.from_numpy(R.from_rotvec(poses_np).as_matrix()).to(device).view(window_size, 24, 3, 3)
            gt_pose_6d = matrix_to_rotation_6d(gt_pose_mat).reshape(window_size, 24, 6)
            
            # 为每帧使用对应的shape参数
            gt_shape_tensor = torch.from_numpy(shapes_window).to(device)
            
            gt_v, gt_j = gen_smpl(smpl, gt_pose_6d, gt_shape_tensor, device)
            gt_vertices_all.append(gt_v.cpu().numpy())
            gt_joints_all.append(gt_j.cpu().numpy())
    
    # 合并结果
    pred_vertices = np.concatenate(pred_vertices_all, axis=0)
    pred_joints = np.concatenate(pred_joints_all, axis=0)
    gt_vertices = np.concatenate(gt_vertices_all, axis=0)
    gt_joints = np.concatenate(gt_joints_all, axis=0)
    
    return pred_vertices, pred_joints, gt_vertices, gt_joints

def calculate_metrics(pred_joints, pred_vertices, gt_joints, gt_vertices):
    """计算评估指标"""
    mpjpe = np.linalg.norm(pred_joints - gt_joints, axis=2).mean()
    mpvpe = np.linalg.norm(pred_vertices - gt_vertices, axis=2).mean()
    
    return {
        'mpjpe': mpjpe * 1000,  # 转换为mm
        'mpvpe': mpvpe * 1000,
        'mpjpe_per_frame': np.linalg.norm(pred_joints - gt_joints, axis=2).mean(axis=1) * 1000,
        'mpvpe_per_frame': np.linalg.norm(pred_vertices - gt_vertices, axis=2).mean(axis=1) * 1000,
    }

def render_comparison_video(pred_vertices, gt_vertices, output_path, 
                            degradation_info="", fps=10, max_frames=200):
    """渲染对比视频"""
    if len(pred_vertices) > max_frames:
        pred_vertices = pred_vertices[:max_frames]
        gt_vertices = gt_vertices[:max_frames]
    
    print(f"Rendering comparison video: {output_path}")
    
    width, height = 1600, 800  # 更宽以容纳两个视图
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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
    
    for f in tqdm(range(len(pred_vertices)), desc="Rendering frames"):
        fig.clf()
        
        # 左图：预测结果
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([ymin, ymax])
        ax1.set_zlim([zmin, zmax])
        ax1.view_init(elev=20, azim=-60 + f * 0.5)
        ax1.set_title(f'Prediction {degradation_info}', fontsize=12)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        pred_v = pred_vertices[f]
        if smpl_faces is not None:
            ax1.plot_trisurf(pred_v[:, 0], pred_v[:, 1], pred_v[:, 2],
                           triangles=smpl_faces, color='lightblue',
                           edgecolor='none', alpha=0.9, shade=True)
        else:
            ax1.scatter(pred_v[::5, 0], pred_v[::5, 1], pred_v[::5, 2],
                       c='lightblue', s=2, alpha=0.8)
        
        # 右图：真实标签
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_xlim([xmin, xmax])
        ax2.set_ylim([ymin, ymax])
        ax2.set_zlim([zmin, zmax])
        ax2.view_init(elev=20, azim=-60 + f * 0.5)
        ax2.set_title(f'Ground Truth (Frame {f+1}/{len(pred_vertices)})', fontsize=12)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        gt_v = gt_vertices[f]
        if smpl_faces is not None:
            ax2.plot_trisurf(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2],
                           triangles=smpl_faces, color='lightgreen',
                           edgecolor='none', alpha=0.9, shade=True)
        else:
            ax2.scatter(gt_v[::5, 0], gt_v[::5, 1], gt_v[::5, 2],
                       c='lightgreen', s=2, alpha=0.8)
        
        # 转换为图像
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (width, height))
        out.write(img)
    
    plt.close(fig)
    out.release()
    print(f"[Saved] Video -> {output_path}")

def plot_metrics_comparison(results_dict, output_path):
    """绘制不同降质条件下的指标对比"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    degradation_types = list(results_dict.keys())
    mpjpe_values = [results_dict[d]['mpjpe'] for d in degradation_types]
    mpvpe_values = [results_dict[d]['mpvpe'] for d in degradation_types]
    
    # 1. MPJPE对比
    ax = axes[0, 0]
    bars = ax.bar(degradation_types, mpjpe_values, color=['green', 'yellow', 'orange', 'red', 'purple'])
    ax.set_ylabel('MPJPE (mm)', fontsize=12)
    ax.set_title('Joint Position Error Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mpjpe_values[i]:.1f}', ha='center', va='bottom', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. MPVPE对比
    ax = axes[0, 1]
    bars = ax.bar(degradation_types, mpvpe_values, color=['green', 'yellow', 'orange', 'red', 'purple'])
    ax.set_ylabel('MPVPE (mm)', fontsize=12)
    ax.set_title('Vertex Position Error Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mpvpe_values[i]:.1f}', ha='center', va='bottom', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. MPJPE逐帧变化
    ax = axes[1, 0]
    for deg_type in degradation_types:
        mpjpe_frames = results_dict[deg_type]['mpjpe_per_frame']
        ax.plot(mpjpe_frames, label=deg_type, linewidth=2, alpha=0.7)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('MPJPE (mm)', fontsize=12)
    ax.set_title('Per-Frame Joint Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 相对性能降级
    ax = axes[1, 1]
    baseline_mpjpe = results_dict['none']['mpjpe']
    relative_degradation = [(results_dict[d]['mpjpe'] - baseline_mpjpe) / baseline_mpjpe * 100 
                           for d in degradation_types[1:]]
    bars = ax.bar(degradation_types[1:], relative_degradation, 
                  color=['yellow', 'orange', 'red', 'purple'])
    ax.set_ylabel('Performance Degradation (%)', fontsize=12)
    ax.set_title('Relative Performance Loss vs Baseline', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Baseline')
    ax.grid(True, alpha=0.3)
    ax.legend()
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{relative_degradation[i]:.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Metrics plot -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description='LiveHPS Robustness Testing')
    parser.add_argument('--sequence_id', type=str, default='24', help='Sequence ID')
    parser.add_argument('--start_frame', type=int, default=100, help='Starting frame')
    parser.add_argument('--num_frames', type=int, default=150, help='Number of frames')
    parser.add_argument('--num_points', type=int, default=256, help='Points per frame')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--pretrained', type=str, default='save_models/livehps.t7')
    parser.add_argument('--output_dir', type=str, default='outputs/robustness_test')
    parser.add_argument('--temporal_window', type=int, default=32)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--max_video_frames', type=int, default=200)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("LiveHPS Robustness / Failure Testing")
    print("="*80)
    
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
    
    # 定义测试配置
    test_configs = [
        {
            'name': 'none',
            'description': 'Baseline (No Degradation)',
            'degradation_type': 'none',
            'params': {}
        },
        {
            'name': 'downsample_50',
            'description': 'Point Downsampling 50%',
            'degradation_type': 'downsample',
            'params': {'downsample_ratio': 0.5}
        },
        {
            'name': 'downsample_25',
            'description': 'Point Downsampling 25%',
            'degradation_type': 'downsample',
            'params': {'downsample_ratio': 0.25}
        },
        {
            'name': 'frame_drop_50',
            'description': 'Frame Dropping (50% kept)',
            'degradation_type': 'frame_drop',
            'params': {'frame_keep_ratio': 0.5}
        },
        {
            'name': 'clutter_front',
            'description': 'Front Occlusion (30% clutter)',
            'degradation_type': 'clutter',
            'params': {'clutter_ratio': 0.3, 'clutter_region': 'front'}
        },
        {
            'name': 'combined_severe',
            'description': 'Combined Degradation (Severe)',
            'degradation_type': 'combined',
            'params': {
                'downsample_ratio': 0.3,
                'frame_keep_ratio': 0.5,
                'clutter_ratio': 0.4,
                'clutter_region': 'front'
            }
        }
    ]
    
    results = {}
    
    # 对每种配置运行测试
    for config in test_configs:
        print("\n" + "="*80)
        print(f"Testing: {config['description']}")
        print("="*80)
        
        # 加载数据
        seq_data = load_sequence_data(
            args.sequence_id,
            args.start_frame,
            args.num_frames,
            args.num_points,
            config['degradation_type'],
            config['params']
        )
        
        if seq_data is None:
            print(f"Failed to load data for {config['name']}")
            continue
        
        print(f"Loaded {len(seq_data['point_clouds'])} frames")
        
        # 运行推理
        pred_v, pred_j, gt_v, gt_j = run_inference(
            model, smpl, seq_data, device, args.temporal_window
        )
        
        # 计算指标
        metrics = calculate_metrics(pred_j, pred_v, gt_j, gt_v)
        results[config['name']] = metrics
        
        print(f"\nResults for {config['description']}:")
        print(f"  MPJPE: {metrics['mpjpe']:.2f} mm")
        print(f"  MPVPE: {metrics['mpvpe']:.2f} mm")
        
        # 保存结果
        output_npz = os.path.join(args.output_dir, f"{config['name']}_results.npz")
        np.savez(output_npz,
                pred_vertices=pred_v,
                pred_joints=pred_j,
                gt_vertices=gt_v,
                gt_joints=gt_j,
                mpjpe=metrics['mpjpe'],
                mpvpe=metrics['mpvpe'])
        print(f"Saved results to {output_npz}")
        
        # 渲染对比视频
        video_path = os.path.join(args.output_dir, f"{config['name']}_comparison.mp4")
        render_comparison_video(
            pred_v, gt_v, video_path,
            degradation_info=f"({config['description']})",
            fps=args.fps,
            max_frames=args.max_video_frames
        )
    
    # 生成对比图表
    print("\n" + "="*80)
    print("Generating comparison plots...")
    print("="*80)
    plot_path = os.path.join(args.output_dir, "metrics_comparison.png")
    plot_metrics_comparison(results, plot_path)
    
    # 输出汇总报告
    print("\n" + "="*80)
    print("ROBUSTNESS TEST SUMMARY")
    print("="*80)
    print(f"\nSequence: {args.sequence_id}, Frames: {args.start_frame}-{args.start_frame+args.num_frames}")
    print(f"\n{'Test Condition':<40} {'MPJPE (mm)':<15} {'MPVPE (mm)':<15} {'Degradation'}")
    print("-" * 85)
    
    baseline_mpjpe = results['none']['mpjpe']
    for config in test_configs:
        name = config['name']
        if name in results:
            mpjpe = results[name]['mpjpe']
            mpvpe = results[name]['mpvpe']
            degradation = ((mpjpe - baseline_mpjpe) / baseline_mpjpe * 100) if name != 'none' else 0
            print(f"{config['description']:<40} {mpjpe:<15.2f} {mpvpe:<15.2f} {degradation:+.1f}%")
    
    print("\n" + "="*80)
    print("Files generated:")
    print(f"  - Comparison plot: {plot_path}")
    for config in test_configs:
        if config['name'] in results:
            print(f"  - {config['name']}_comparison.mp4")
            print(f"  - {config['name']}_results.npz")
    print("="*80)

if __name__ == "__main__":
    main()
