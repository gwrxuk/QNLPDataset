#!/usr/bin/env python3
"""
主要分析流程管道
Main Analysis Pipeline for Real QNLP Analysis
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import pandas as pd

class QNLPAnalysisPipeline:
    """QNLP分析流程管道"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.scripts_dir = self.base_dir / "scripts"
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        self.viz_dir = self.base_dir / "visualizations"
        
        # 確保目錄存在
        for dir_path in [self.data_dir, self.results_dir, self.viz_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def check_prerequisites(self) -> bool:
        """檢查先決條件"""
        print("🔍 檢查分析先決條件...")
        
        # 檢查API密鑰
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ 請設定OPENAI_API_KEY環境變數")
            return False
        print(f"✅ OpenAI API密鑰已設定: {api_key[:10]}...")
        
        # 檢查數據檔案
        dataset_path = self.base_dir.parent / "dataseet.xlsx"
        if not dataset_path.exists():
            print(f"❌ 未找到數據檔案: {dataset_path}")
            return False
        print(f"✅ 數據檔案存在: {dataset_path}")
        
        # 檢查jieba結果
        jieba_results_path = self.base_dir.parent / "jieba_segmentation_results.csv"
        if not jieba_results_path.exists():
            print(f"❌ 未找到jieba斷詞結果: {jieba_results_path}")
            print("請先運行jieba斷詞分析")
            return False
        print(f"✅ jieba結果存在: {jieba_results_path}")
        
        return True
    
    def copy_existing_data(self):
        """複製現有數據到分析目錄"""
        print("📋 複製現有數據...")
        
        # 複製數據檔案
        source_dataset = self.base_dir.parent / "dataseet.xlsx"
        target_dataset = self.data_dir / "dataseet.xlsx"
        
        if source_dataset.exists():
            import shutil
            shutil.copy2(source_dataset, target_dataset)
            print(f"✅ 複製數據集: {target_dataset}")
        
        # 複製jieba結果
        jieba_files = [
            "jieba_segmentation_results.csv",
            "jieba_vocabulary_stats.csv", 
            "jieba_field_vocabulary.csv",
            "jieba_summary_stats.csv"
        ]
        
        for filename in jieba_files:
            source_file = self.base_dir.parent / filename
            target_file = self.data_dir / filename
            
            if source_file.exists():
                import shutil
                shutil.copy2(source_file, target_file)
                print(f"✅ 複製jieba結果: {filename}")
    
    def run_chatgpt_segmentation(self, max_records: int = None) -> bool:
        """運行ChatGPT斷詞分析"""
        print("\n🤖 運行ChatGPT斷詞分析...")
        
        script_path = self.scripts_dir / "real_chatgpt_segmentation.py"
        
        try:
            # 切換到腳本目錄
            original_dir = os.getcwd()
            os.chdir(self.scripts_dir)
            
            # 運行腳本
            if max_records:
                # 模擬用戶輸入限制記錄數
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=str(max_records))
            else:
                # 運行全部記錄
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input="\\n")  # 按Enter使用默認設定
            
            os.chdir(original_dir)
            
            if process.returncode == 0:
                print("✅ ChatGPT斷詞分析完成")
                print(stdout)
                return True
            else:
                print(f"❌ ChatGPT斷詞分析失敗: {stderr}")
                return False
                
        except Exception as e:
            os.chdir(original_dir)
            print(f"❌ 運行ChatGPT分析時發生錯誤: {e}")
            return False
    
    def run_qnlp_analysis(self) -> bool:
        """運行QNLP比較分析"""
        print("\n🔬 運行QNLP比較分析...")
        
        script_path = self.scripts_dir / "enhanced_qnlp_analyzer.py"
        
        try:
            original_dir = os.getcwd()
            os.chdir(self.scripts_dir)
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True
            )
            
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("✅ QNLP分析完成")
                print(result.stdout)
                return True
            else:
                print(f"❌ QNLP分析失敗: {result.stderr}")
                return False
                
        except Exception as e:
            os.chdir(original_dir)
            print(f"❌ 運行QNLP分析時發生錯誤: {e}")
            return False
    
    def run_visualization(self) -> bool:
        """運行視覺化分析"""
        print("\n🎨 運行視覺化分析...")
        
        script_path = self.scripts_dir / "comprehensive_visualizer.py"
        
        try:
            original_dir = os.getcwd()
            os.chdir(self.scripts_dir)
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True
            )
            
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("✅ 視覺化分析完成")
                print(result.stdout)
                return True
            else:
                print(f"❌ 視覺化分析失敗: {result.stderr}")
                return False
                
        except Exception as e:
            os.chdir(original_dir)
            print(f"❌ 運行視覺化分析時發生錯誤: {e}")
            return False
    
    def generate_final_report(self):
        """生成最終報告"""
        print("\n📄 生成最終報告...")
        
        report_content = f"""# 真實QNLP分析報告
# Real QNLP Analysis Report

## 分析概述
本報告基於jieba和ChatGPT兩種中文斷詞方法，進行量子自然語言處理(QNLP)比較分析。

## 分析時間
{time.strftime('%Y-%m-%d %H:%M:%S')}

## 目錄結構
```
real_qnlp_analysis/
├── data/                    # 數據檔案
│   ├── dataseet.xlsx       # 原始數據
│   ├── jieba_*.csv         # jieba斷詞結果
│   └── real_chatgpt_*.csv  # ChatGPT斷詞結果
├── results/                # 分析結果
│   ├── qnlp_comparative_analysis.json
│   └── statistical_summary.json
├── visualizations/         # 視覺化圖表
│   ├── quantum_metrics_comparison.png
│   ├── word_count_analysis.png
│   ├── radar_chart_comparison.png
│   └── insights_summary.png
└── scripts/               # 分析腳本
    ├── real_chatgpt_segmentation.py
    ├── enhanced_qnlp_analyzer.py
    ├── comprehensive_visualizer.py
    └── main_pipeline.py
```

## 主要發現
詳細的分析結果請查看：
- `results/qnlp_comparative_analysis.json` - 完整分析結果
- `results/statistical_summary.json` - 統計摘要
- `visualizations/` 目錄下的各種圖表

## 使用方法
1. 設定OpenAI API密鑰: `export OPENAI_API_KEY="your-key"`
2. 運行完整分析: `python scripts/main_pipeline.py`
3. 查看結果和圖表

## 注意事項
- ChatGPT分析需要消耗API tokens，請注意成本
- 分析結果基於量子計算原理，具有理論探索性質
- 建議在理解量子語言學理論的基礎上解讀結果

---
生成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report_path = self.base_dir / "README.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 最終報告已生成: {report_path}")
    
    def run_full_pipeline(self, max_records: int = None):
        """運行完整分析流程"""
        print("🚀 開始真實QNLP分析完整流程")
        print("=" * 50)
        
        # 檢查先決條件
        if not self.check_prerequisites():
            print("❌ 先決條件檢查失敗，分析終止")
            return False
        
        # 複製現有數據
        self.copy_existing_data()
        
        # 詢問分析範圍
        if max_records is None:
            print(f"\\n⚙️  分析設定:")
            user_input = input("每個欄位分析多少筆記錄？(Enter=全部, 數字=限制筆數): ").strip()
            if user_input.isdigit():
                max_records = int(user_input)
                print(f"📝 將分析每個欄位的前 {max_records} 筆記錄")
            else:
                print("📝 將分析全部記錄")
        
        # 估算成本
        if max_records:
            estimated_records = max_records * 3  # 3個欄位
        else:
            # 讀取數據估算
            try:
                df = pd.read_excel(self.data_dir / "dataseet.xlsx")
                estimated_records = len(df) * 3
            except:
                estimated_records = 299 * 3  # 預設值
        
        estimated_cost = (estimated_records * 400 / 1000) * 0.002  # 估算tokens和成本
        print(f"\\n💰 估算成本: ${estimated_cost:.3f} ({estimated_records} 筆記錄)")
        
        confirm = input("確認繼續？(y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ 用戶取消分析")
            return False
        
        # 開始分析流程
        start_time = time.time()
        
        # 1. ChatGPT斷詞分析
        if not self.run_chatgpt_segmentation(max_records):
            print("❌ ChatGPT分析失敗，流程終止")
            return False
        
        # 2. QNLP比較分析
        if not self.run_qnlp_analysis():
            print("❌ QNLP分析失敗，流程終止")
            return False
        
        # 3. 視覺化分析
        if not self.run_visualization():
            print("⚠️  視覺化分析失敗，但繼續流程")
        
        # 4. 生成最終報告
        self.generate_final_report()
        
        # 分析完成
        elapsed_time = time.time() - start_time
        print(f"\\n🎉 完整QNLP分析流程完成！")
        print(f"⏱️  總耗時: {elapsed_time/60:.1f} 分鐘")
        print(f"📁 結果目錄: {self.base_dir}")
        print(f"\\n📊 主要輸出檔案:")
        print(f"  - data/real_chatgpt_segmentation_complete.csv")
        print(f"  - results/qnlp_comparative_analysis.json")
        print(f"  - visualizations/*.png")
        print(f"  - README.md")
        
        return True

def main():
    """主函數"""
    pipeline = QNLPAnalysisPipeline()
    
    if len(sys.argv) > 1:
        try:
            max_records = int(sys.argv[1])
            pipeline.run_full_pipeline(max_records)
        except ValueError:
            print("❌ 請提供有效的記錄數量")
    else:
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
