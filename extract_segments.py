"""
从已生成的SMPL结果中提取特定帧段并重新渲染

用途: 快速从完整测试结果中提取展示特定场景的片段
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
import os

def load_smpl_faces():
    """加载SMPL面片信息"""
    try:
        import sys
        sys.path.append("./smpl")
        from smpl import SMPL, SMPL_MODEL_DIR
        import torch
        smpl = SMPL(SMPL_MODEL_DIR, create_transl=False)
        if hasattr(smpl, 'faces'):
            return smpl.faces
    except:
        pass
    return None

def render_mesh_video(vertices, output_path, fps=10, title_prefix=""):
    """渲染人体网格视频"""
    print(f"Creating video with {vertices.shape[0]} frames...")
    
    width, height = 800, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算边界
    xs, ys, zs = vertices[:,:,0], vertices[:,:,1], vertices[:,:,2]
    xmin, xmax = xs.min() - 0.1, xs.max() + 0.1
    ymin, ymax = ys.min() - 0.1, ys.max() + 0.1
    zmin, zmax = zs.min() - 0.1, zs.max() + 0.1
    
    # 获取SMPL面片
    smpl_faces = load_smpl_faces()
    
    for f in tqdm(range(vertices.shape[0]), desc="Rendering frames"):
        ax.clear()
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])
        ax.view_init(elev=20, azim=-60 + f * 0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title_prefix}Frame {f+1}/{vertices.shape[0]}')
        
        verts = vertices[f]
        
        # 渲染网格
        if smpl_faces is not None:
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                           triangles=smpl_faces,
                           color='lightblue',
                           edgecolor='none',
                           alpha=0.9,
                           shade=True,
                           linewidth=0)
        else:
            # 后备: 渲染密集点云
            ax.scatter(verts[::5, 0], verts[::5, 1], verts[::5, 2],
                     c='lightblue', s=2, alpha=0.8)
        
        # 转换为OpenCV图像
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (width, height))
        out.write(img)
    
    plt.close(fig)
    out.release()
    print(f"[Saved] Video -> {output_path}")

def analyze_motion_characteristics(vertices):
    """分析运动特征"""
    # 计算质心轨迹
    centroids = vertices.mean(axis=1)  # (N, 3)
    
    # 计算运动速度
    velocities = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    
    # 计算点云密度变化 (使用边界框体积作为代理)
    volumes = []
    for v in vertices:
        ranges = v.max(axis=0) - v.min(axis=0)
        volume = np.prod(ranges)
        volumes.append(volume)
    
    return {
        'velocities': velocities,
        'avg_velocity': velocities.mean(),
        'max_velocity': velocities.max(),
        'volumes': np.array(volumes),
        'centroids': centroids
    }

def find_interesting_segments(analysis, min_length=100):
    """自动找出有趣的片段"""
    segments = []
    velocities = analysis['velocities']
    
    # 1. 快速运动片段
    fast_threshold = np.percentile(velocities, 80)
    for i in range(len(velocities) - min_length):
        window = velocities[i:i+min_length]
        if window.mean() > fast_threshold:
            segments.append({
                'type': '快速运动',
                'start': i,
                'end': i + min_length,
                'score': window.mean() / fast_threshold
            })
    
    # 2. 平稳运动片段 (作为对比)
    slow_threshold = np.percentile(velocities, 20)
    for i in range(len(velocities) - min_length):
        window = velocities[i:i+min_length]
        if window.mean() < slow_threshold:
            segments.append({
                'type': '平稳运动',
                'start': i,
                'end': i + min_length,
                'score': slow_threshold / (window.mean() + 1e-6)
            })
    
    # 3. 急剧变化片段
    accelerations = np.abs(np.diff(velocities))
    accel_threshold = np.percentile(accelerations, 80)
    for i in range(len(accelerations) - min_length):
        window = accelerations[i:i+min_length]
        if window.mean() > accel_threshold:
            segments.append({
                'type': '急剧变化',
                'start': i,
                'end': i + min_length,
                'score': window.mean() / accel_threshold
            })
    
    return segments

def main():
    parser = argparse.ArgumentParser(description='从SMPL结果中提取并渲染特定片段')
    parser.add_argument('--input', type=str, default='outputs/smpl_outputs.npz',
                       help='输入npz文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs/segments',
                       help='输出目录')
    parser.add_argument('--start_frame', type=int, default=None,
                       help='起始帧 (如果指定则手动选择片段)')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='结束帧 (如果指定则手动选择片段)')
    parser.add_argument('--auto', action='store_true',
                       help='自动寻找并渲染有趣的片段')
    parser.add_argument('--fps', type=int, default=10,
                       help='视频帧率')
    parser.add_argument('--segment_length', type=int, default=200,
                       help='自动模式下的片段长度')
    parser.add_argument('--max_segments', type=int, default=3,
                       help='自动模式下最多渲染的片段数')
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.input}...")
    data = np.load(args.input, allow_pickle=True)
    vertices = data['pred_vertices']
    print(f"Loaded {vertices.shape[0]} frames")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 手动模式
    if args.start_frame is not None and args.end_frame is not None:
        print(f"\n手动模式: 提取帧 {args.start_frame}-{args.end_frame}")
        selected_vertices = vertices[args.start_frame:args.end_frame]
        output_path = os.path.join(args.output_dir, 
                                   f'segment_{args.start_frame}_{args.end_frame}.mp4')
        render_mesh_video(selected_vertices, output_path, args.fps,
                         title_prefix=f"Frames {args.start_frame}-{args.end_frame} | ")
    
    # 自动模式
    elif args.auto:
        print("\n自动模式: 分析运动特征...")
        analysis = analyze_motion_characteristics(vertices)
        
        print(f"\n运动统计:")
        print(f"  平均速度: {analysis['avg_velocity']:.4f}")
        print(f"  最大速度: {analysis['max_velocity']:.4f}")
        
        print(f"\n寻找有趣片段 (长度={args.segment_length})...")
        segments = find_interesting_segments(analysis, args.segment_length)
        
        # 按类型分组
        segments_by_type = {}
        for seg in segments:
            t = seg['type']
            if t not in segments_by_type:
                segments_by_type[t] = []
            segments_by_type[t].append(seg)
        
        # 每种类型选最高分的一个
        selected_segments = []
        for seg_type, segs in segments_by_type.items():
            best = max(segs, key=lambda x: x['score'])
            selected_segments.append(best)
            print(f"\n找到{seg_type}片段:")
            print(f"  帧范围: {best['start']}-{best['end']}")
            print(f"  评分: {best['score']:.2f}")
        
        # 渲染选定的片段
        for i, seg in enumerate(selected_segments[:args.max_segments], 1):
            print(f"\n渲染片段 {i}/{min(len(selected_segments), args.max_segments)}: {seg['type']}")
            selected_vertices = vertices[seg['start']:seg['end']]
            output_path = os.path.join(args.output_dir,
                                       f"{seg['type'].replace('/', '_')}_{seg['start']}_{seg['end']}.mp4")
            render_mesh_video(selected_vertices, output_path, args.fps,
                             title_prefix=f"{seg['type']} | ")
    
    else:
        print("\n请指定 --start_frame 和 --end_frame (手动模式) 或 --auto (自动模式)")
        print("\n示例:")
        print("  # 手动提取特定片段")
        print("  python extract_segments.py --start_frame 5000 --end_frame 5200")
        print("\n  # 自动寻找有趣片段")
        print("  python extract_segments.py --auto --segment_length 200 --max_segments 3")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
