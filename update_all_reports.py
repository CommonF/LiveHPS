"""
更新所有报告和分析结果
在修改robustness_test.py后运行此脚本以更新所有内容
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(script_path, description):
    """运行Python脚本并显示进度"""
    print("="*80)
    print(f"⏳ {description}")
    print(f"📝 脚本: {script_path}")
    print("="*80)
    
    # 直接使用当前Python解释器运行脚本
    try:
        # 使用exec在同一进程中运行脚本
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        
        # 在独立的命名空间中执行
        namespace = {'__name__': '__main__', '__file__': script_path}
        exec(code, namespace)
        
        # 恢复工作目录
        os.chdir(original_cwd)
        
        print(f"✅ {description} - 成功")
        return True
        
    except Exception as e:
        print(f"❌ {description} - 失败")
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("🔄 LiveHPS 报告更新流程")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查测试结果是否存在
    results_exist = os.path.exists('outputs/robustness_test/none_results.npz')
    
    if not results_exist:
        print("❌ 错误: 找不到鲁棒性测试结果文件")
        print("请先手动运行: python robustness_test.py")
        return
    
    print("✅ 检测到鲁棒性测试结果文件")
    print("📁 使用现有测试结果生成分析报告")
    print()
    
    steps = []
    
    # 跳过鲁棒性测试步骤（假设用户已手动完成）
    
    # Step 1: 生成挑战性测试分析
    steps.append({
        'script': 'analyze_challenging_test.py',
        'desc': 'Step 1/3: 生成挑战性测试分析报告',
        'required': True
    })
    
    # Step 2: 生成点云预算报告
    steps.append({
        'script': 'generate_point_budget_report.py',
        'desc': 'Step 2/3: 生成点云预算分析报告',
        'required': False
    })
    
    # Step 3: 提醒更新综合报告
    print("\n" + "="*80)
    print("📋 Step 3/3: 更新综合报告")
    print("="*80)
    print("注意: COMPREHENSIVE_ANALYSIS_REPORT.md 需要手动检查和更新")
    print("主要更新内容:")
    print("  - 第1-100行: 鲁棒性测试结果")
    print("  - MPJPE/MPVPE数值")
    print("  - 性能退化百分比")
    print()
    
    # 执行所有步骤
    failed_steps = []
    for i, step in enumerate(steps):
        success = run_command(step['script'], step['desc'])
        if not success:
            failed_steps.append(step['desc'])
            if step['required']:
                print(f"\n❌ 关键步骤失败，停止执行")
                break
        print()
    
    # 总结
    print("\n" + "="*80)
    print("📊 更新完成总结")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_steps:
        print(f"\n⚠️  以下步骤失败:")
        for step in failed_steps:
            print(f"  - {step}")
    else:
        print("\n✅ 所有自动化步骤成功完成！")
    
    print("\n📝 后续手动步骤:")
    print("  1. 检查 outputs/robustness_test/CHALLENGING_TEST_ANALYSIS_REPORT.md")
    print("  2. 根据新的测试结果更新 COMPREHENSIVE_ANALYSIS_REPORT.md")
    print("  3. 验证所有视频和图表是否正确生成")
    print("  4. 检查 none_comparison.mp4 中的mesh对齐问题是否修复")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
