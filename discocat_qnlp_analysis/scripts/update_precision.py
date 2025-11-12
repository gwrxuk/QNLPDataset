#!/usr/bin/env python3
"""
更新MD报告中的数值精度
"""

import re

def update_precision_in_file():
    """更新文件中的数值精度"""
    
    file_path = '../analysis_reports/ai_vs_journalist_quantum_comparison.md'
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔍 检查需要更新的数值...")
    
    # 定义需要精确更新的数值映射
    precise_updates = {
        # 语法叠加强度相关
        '1.0000': '1.000000',
        '0.0000': '0.000000',
        
        # 其他常见的4位小数格式
        '0.9175': '0.917508',
        '0.9985': '0.998497',
        '0.8059': '0.805879',
        '0.7477': '0.747725',
        '0.1771': '0.177073',
        '0.2878': '0.287786',
        '0.3089': '0.308865',
        '0.0008': '0.000816',
        '0.9109': '0.910895',
        '3.4378': '3.437841',
        '0.1407': '0.140719',
        '0.3607': '0.360721',
        '0.1033': '0.103291',
        '0.5458': '0.545765',
    }
    
    # 标准差的更新
    std_updates = {
        '0.0191': '0.019116',
        '0.0038': '0.003813',
        '0.0221': '0.022065',
        '0.0057': '0.005674',
        '0.0145': '0.014460',
        '0.0653': '0.065286',
        '0.0751': '0.075128',
        '0.0022': '0.002162',
        '0.0238': '0.023768',
        '0.4385': '0.438545',
        '0.0621': '0.062060',
        '0.1024': '0.102401',
        '0.0299': '0.029875',
        '0.1502': '0.150186',
    }
    
    # 合并所有更新
    all_updates = {**precise_updates, **std_updates}
    
    # 执行替换
    updated_count = 0
    for old_val, new_val in all_updates.items():
        if old_val in content:
            content = content.replace(old_val, new_val)
            updated_count += 1
            print(f"✅ 更新: {old_val} → {new_val}")
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n📝 完成更新: {updated_count} 个数值已更新")
    print(f"📄 文件已保存: {file_path}")

def verify_updates():
    """验证更新结果"""
    
    file_path = '../analysis_reports/ai_vs_journalist_quantum_comparison.md'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n🔍 验证更新结果...")
    
    # 检查是否还有4位小数的数值（排除年份等）
    pattern = r'(?<!\d)0\.\d{4}(?!\d)'
    matches = re.findall(pattern, content)
    
    if matches:
        print(f"⚠️  发现 {len(matches)} 个可能需要更新的4位小数:")
        for match in set(matches):
            print(f"   - {match}")
    else:
        print("✅ 未发现需要进一步更新的4位小数")
    
    # 检查关键指标
    key_indicators = ['1.000000', '0.308865', '0.000816', '0.917508', '0.998497']
    found_indicators = []
    for indicator in key_indicators:
        if indicator in content:
            found_indicators.append(indicator)
    
    print(f"\n📊 关键指标验证:")
    print(f"   发现 {len(found_indicators)}/{len(key_indicators)} 个关键指标使用6位小数")
    for indicator in found_indicators:
        print(f"   ✅ {indicator}")

def main():
    """主函数"""
    
    print("🚀 开始更新MD报告数值精度")
    print("=" * 50)
    
    # 更新精度
    update_precision_in_file()
    
    # 验证结果
    verify_updates()
    
    print("\n✅ 数值精度更新完成!")

if __name__ == "__main__":
    main()
