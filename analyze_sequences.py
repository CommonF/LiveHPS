"""
分析LiDARHuman26M序列，找出展示以下情况的片段：
1. 强闭合 (Strong Occlusion)
2. 长距离/稀疏返回 (Long Range/Sparse Returns)
3. 快速运动/大平移 (Fast Motion/Large Translation)
4. 动态背景 (Dynamic Background)
"""

import os
import json
import numpy as np
import argparse
from tqdm import tqdm

def load_frame_data(sequence_id, frame_num):
    """加载单帧数据"""
    json_path = f"dataset/lidarhuman26M/labels/3d/pose/{sequence_id}/{frame_num:06d}.json"
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return {
        'points': np.array(data['lidar_3d_points'], dtype=np.float32),
        'pose': np.array(data['smpl_param']['pose'], dtype=np.float32),
        'trans': np.array(data['smpl_param']['trans'], dtype=np.float32),
    }

def analyze_sequence(sequence_id, start_frame, end_frame, stride=10):
    """分析序列特征"""
    print(f"\n分析序列 {sequence_id}, 帧 {start_frame}-{end_frame}...")
    
    results = {
        'point_counts': [],
        'translations': [],
        'velocities': [],
        'accelerations': [],
        'distances': [],
    }
    
    prev_trans = None
    prev_vel = None
    
    for frame_num in tqdm(range(start_frame, end_frame + 1, stride)):
        data = load_frame_data(sequence_id, frame_num)
        if data is None:
            continue
        
        # 点云密度 (检测稀疏返回)
        point_count = len(data['points'])
        results['point_counts'].append(point_count)
        
        # 人体到传感器的距离 (检测长距离)
        if len(data['points']) > 0:
            distance = np.linalg.norm(data['trans'])
            results['distances'].append(distance)
        
        # 平移变化 (检测快速运动)
        trans = data['trans']
        results['translations'].append(trans)
        
        if prev_trans is not None:
            velocity = np.linalg.norm(trans - prev_trans)
            results['velocities'].append(velocity)
            
            if prev_vel is not None:
                acceleration = abs(velocity - prev_vel)
                results['accelerations'].append(acceleration)
            
            prev_vel = velocity
        
        prev_trans = trans
    
    return results

def find_interesting_segments(results, window_size=50):
    """找出有趣的片段"""
    segments = []
    
    # 1. 稀疏点云片段 (长距离/稀疏返回)
    point_counts = np.array(results['point_counts'])
    sparse_threshold = np.percentile(point_counts, 20)  # 最稀疏的20%
    for i in range(len(point_counts) - window_size):
        window = point_counts[i:i+window_size]
        if np.mean(window) < sparse_threshold:
            segments.append({
                'type': '稀疏点云/长距离',
                'start': i,
                'length': window_size,
                'avg_points': np.mean(window),
                'score': sparse_threshold / (np.mean(window) + 1)
            })
    
    # 2. 快速运动片段
    if len(results['velocities']) > 0:
        velocities = np.array(results['velocities'])
        fast_threshold = np.percentile(velocities, 80)  # 最快的20%
        for i in range(len(velocities) - window_size):
            window = velocities[i:i+window_size]
            if np.mean(window) > fast_threshold:
                segments.append({
                    'type': '快速运动',
                    'start': i,
                    'length': window_size,
                    'avg_velocity': np.mean(window),
                    'score': np.mean(window) / fast_threshold
                })
    
    # 3. 大加速度片段 (突然变化)
    if len(results['accelerations']) > 0:
        accelerations = np.array(results['accelerations'])
        accel_threshold = np.percentile(accelerations, 80)
        for i in range(len(accelerations) - window_size):
            window = accelerations[i:i+window_size]
            if np.mean(window) > accel_threshold:
                segments.append({
                    'type': '急剧运动变化',
                    'start': i,
                    'length': window_size,
                    'avg_acceleration': np.mean(window),
                    'score': np.mean(window) / accel_threshold
                })
    
    # 4. 距离变化大的片段 (可能包含遮挡)
    if len(results['distances']) > 0:
        distances = np.array(results['distances'])
        for i in range(len(distances) - window_size):
            window = distances[i:i+window_size]
            distance_variance = np.var(window)
            if distance_variance > np.percentile(np.var([distances[j:j+window_size] 
                                                         for j in range(len(distances)-window_size)], axis=1), 70):
                segments.append({
                    'type': '距离变化大/可能遮挡',
                    'start': i,
                    'length': window_size,
                    'distance_variance': distance_variance,
                    'score': distance_variance
                })
    
    return segments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_id', type=str, default='29', help='序列ID')
    parser.add_argument('--start_frame', type=int, default=1, help='起始帧')
    parser.add_argument('--end_frame', type=int, default=1000, help='结束帧')
    parser.add_argument('--stride', type=int, default=10, help='采样步长')
    parser.add_argument('--window_size', type=int, default=50, help='分析窗口大小')
    args = parser.parse_args()
    
    # 分析序列
    results = analyze_sequence(args.sequence_id, args.start_frame, args.end_frame, args.stride)
    
    # 打印统计信息
    print("\n" + "="*70)
    print("序列统计信息")
    print("="*70)
    
    if results['point_counts']:
        print(f"\n点云密度:")
        print(f"  平均点数: {np.mean(results['point_counts']):.1f}")
        print(f"  最小点数: {np.min(results['point_counts'])}")
        print(f"  最大点数: {np.max(results['point_counts'])}")
        print(f"  标准差: {np.std(results['point_counts']):.1f}")
    
    if results['distances']:
        print(f"\n距离传感器:")
        print(f"  平均距离: {np.mean(results['distances']):.2f}m")
        print(f"  最远距离: {np.max(results['distances']):.2f}m")
        print(f"  最近距离: {np.min(results['distances']):.2f}m")
    
    if results['velocities']:
        print(f"\n运动速度:")
        print(f"  平均速度: {np.mean(results['velocities']):.4f}m/frame")
        print(f"  最大速度: {np.max(results['velocities']):.4f}m/frame")
        print(f"  标准差: {np.std(results['velocities']):.4f}")
    
    if results['accelerations']:
        print(f"\n加速度:")
        print(f"  平均加速度: {np.mean(results['accelerations']):.4f}")
        print(f"  最大加速度: {np.max(results['accelerations']):.4f}")
    
    # 找出有趣的片段
    segments = find_interesting_segments(results, args.window_size)
    
    print("\n" + "="*70)
    print("发现的有趣片段 (按类型排序)")
    print("="*70)
    
    # 按类型分组并排序
    segments_by_type = {}
    for seg in segments:
        seg_type = seg['type']
        if seg_type not in segments_by_type:
            segments_by_type[seg_type] = []
        segments_by_type[seg_type].append(seg)
    
    for seg_type, segs in segments_by_type.items():
        print(f"\n【{seg_type}】")
        # 按分数排序，取前3个
        segs_sorted = sorted(segs, key=lambda x: x['score'], reverse=True)[:3]
        for i, seg in enumerate(segs_sorted, 1):
            frame_start = args.start_frame + seg['start'] * args.stride
            frame_end = frame_start + seg['length'] * args.stride
            print(f"  {i}. 帧 {frame_start}-{frame_end}")
            for key, value in seg.items():
                if key not in ['type', 'start', 'length', 'score']:
                    print(f"     {key}: {value:.4f}")
            print(f"     评分: {seg['score']:.4f}")
    
    # 推荐最佳片段
    print("\n" + "="*70)
    print("推荐的测试案例")
    print("="*70)
    
    if segments:
        # 选择分数最高的片段
        best_segment = max(segments, key=lambda x: x['score'])
        frame_start = args.start_frame + best_segment['start'] * args.stride
        frame_end = frame_start + best_segment['length'] * args.stride
        
        print(f"\n最佳案例:")
        print(f"  序列ID: {args.sequence_id}")
        print(f"  帧范围: {frame_start}-{frame_end}")
        print(f"  类型: {best_segment['type']}")
        print(f"  帧数: {best_segment['length'] * args.stride}")
        print(f"\n运行命令:")
        print(f"  python test.py --pretrained save_models/livehps.t7 \\")
        print(f"                 --output_dir outputs/case_{best_segment['type'].replace('/', '_')} \\")
        print(f"                 --save_video --video_max_frames 200 \\")
        print(f"                 --batch_size 4 --workers 0")
        print(f"\n注意: 需要修改数据集加载代码来指定帧范围 {frame_start}-{frame_end}")

if __name__ == "__main__":
    main()
