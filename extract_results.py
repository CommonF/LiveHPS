"""
读取最新的鲁棒性测试结果并生成报告数据
"""
import numpy as np
import json

def load_result(path):
    """加载NPZ结果文件"""
    data = np.load(path)
    return {
        'mpjpe': float(data['mpjpe']),
        'mpvpe': float(data['mpvpe'])
    }

# 加载所有测试结果
results = {
    'none': load_result('outputs/robustness_test/none_results.npz'),
    'downsample_50': load_result('outputs/robustness_test/downsample_50_results.npz'),
    'downsample_25': load_result('outputs/robustness_test/downsample_25_results.npz'),
    'frame_drop_50': load_result('outputs/robustness_test/frame_drop_50_results.npz'),
    'clutter_front': load_result('outputs/robustness_test/clutter_front_results.npz'),
    'combined_severe': load_result('outputs/robustness_test/combined_severe_results.npz')
}

# 计算相对变化
baseline = results['none']

print("="*80)
print("鲁棒性测试结果摘要")
print("="*80)
print(f"\n基准 (Baseline):")
print(f"  MPJPE: {baseline['mpjpe']:.2f} mm")
print(f"  MPVPE: {baseline['mpvpe']:.2f} mm")
print()

test_names = {
    'downsample_50': '点云下采样 50%',
    'downsample_25': '点云下采样 25%', 
    'frame_drop_50': '帧丢弃 50%',
    'clutter_front': '前方遮挡',
    'combined_severe': '综合严重降质'
}

print("各测试场景结果:")
print("-"*80)
for key, name in test_names.items():
    r = results[key]
    mpjpe_change = ((r['mpjpe'] - baseline['mpjpe']) / baseline['mpjpe']) * 100
    mpvpe_change = ((r['mpvpe'] - baseline['mpvpe']) / baseline['mpvpe']) * 100
    
    print(f"\n{name}:")
    print(f"  MPJPE: {r['mpjpe']:.2f} mm ({mpjpe_change:+.1f}%)")
    print(f"  MPVPE: {r['mpvpe']:.2f} mm ({mpvpe_change:+.1f}%)")

print("\n" + "="*80)

# 保存为JSON以便后续使用
with open('outputs/robustness_test/results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("结果已保存到: outputs/robustness_test/results_summary.json")
