# 真實QNLP分析項目
# Real QNLP Analysis Project

## 🎯 項目概述

本項目使用真實的jieba和ChatGPT中文斷詞方法，進行量子自然語言處理(QNLP)比較分析。通過量子計算原理來分析不同斷詞方法對文本語義理解的影響。

## 📊 主要發現

### 量子指標比較結果

| 指標 | jieba | ChatGPT | 差異 | 洞察 |
|------|-------|---------|------|------|
| **量子連貫性** | 0.2914 | 0.4901 | +0.1986 | ChatGPT語義一致性更強 |
| **敘事複雜度** | 0.0000 | 0.0000 | +0.0000 | 兩者在複雜度上相當 |
| **量子干涉** | 1.0000 | 1.0000 | +0.0000 | 兩者干涉模式相似 |
| **疊加強度** | 0.1635 | 0.2989 | +0.1355 | ChatGPT更能體現多重現實 |
| **平均詞數** | 8.8 | 18.0 | +9.2 | ChatGPT更細粒度斷詞 |

### 🔍 核心洞察

1. **ChatGPT斷詞展現更高的量子連貫性**，表示語義一致性更強
2. **ChatGPT傾向產生更細粒度的斷詞結果**，平均詞數是jieba的2倍以上
3. **兩種方法在量子干涉方面表現相似**，都能保持良好的相位關係
4. **ChatGPT在疊加強度上表現更佳**，更能體現量子語言學的多重現實特性

## 📁 項目結構

```
real_qnlp_analysis/
├── data/                           # 數據文件
│   ├── dataseet.xlsx              # 原始數據集
│   ├── jieba_segmentation_results.csv     # jieba斷詞結果
│   ├── jieba_vocabulary_stats.csv         # jieba詞彙統計
│   ├── jieba_field_vocabulary.csv         # jieba欄位詞彙分析
│   ├── jieba_summary_stats.csv            # jieba摘要統計
│   └── real_chatgpt_segmentation_sample.csv # ChatGPT斷詞樣本
├── scripts/                        # 分析腳本
│   ├── real_chatgpt_segmentation.py       # ChatGPT斷詞分析
│   ├── enhanced_qnlp_analyzer.py          # 增強版QNLP分析器
│   ├── comprehensive_visualizer.py        # 綜合視覺化工具
│   └── main_pipeline.py                   # 主要分析管道
├── results/                        # 分析結果
│   ├── real_qnlp_comparative_analysis.json # 完整比較分析結果
│   └── qnlp_visualization_data.json        # 視覺化數據
├── visualizations/                 # 視覺化圖表
│   ├── qnlp_comparison.png                 # 量子指標比較圖
│   └── qnlp_radar_chart.png               # 雷達圖比較
└── README.md                       # 項目說明文件
```

## 🚀 使用方法

### 環境設置

1. **安裝依賴**:
```bash
pip install pandas numpy matplotlib seaborn qiskit scikit-learn openai jieba
```

2. **設置OpenAI API密鑰** (用於ChatGPT分析):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 運行分析

1. **完整分析流程**:
```bash
cd scripts/
python main_pipeline.py
```

2. **單獨運行各組件**:
```bash
# ChatGPT斷詞分析
python real_chatgpt_segmentation.py

# QNLP比較分析
python enhanced_qnlp_analyzer.py

# 視覺化分析
python comprehensive_visualizer.py
```

## 🔬 技術原理

### 量子自然語言處理 (QNLP)

本項目基於量子計算原理分析自然語言：

1. **量子電路構建**: 將詞彙映射到量子比特，創建疊加態
2. **量子糾纏**: 通過CNOT門建立詞彙間的語義關聯
3. **量子測量**: 計算von Neumann熵、連貫性、干涉等指標
4. **語義分析**: 通過量子指標評估文本的語義特性

### 核心量子指標

- **量子連貫性 (Quantum Coherence)**: 衡量語義一致性
- **敘事複雜度 (Narrative Complexity)**: 基於von Neumann熵的複雜度
- **量子干涉 (Quantum Interference)**: 相位關係的穩定性
- **疊加強度 (Superposition Strength)**: 多重語義狀態的強度

## 📈 分析結果

### 視覺化圖表

1. **量子指標比較圖** (`qnlp_comparison.png`)
   - 四個子圖展示各項量子指標的對比
   - 清楚顯示ChatGPT在連貫性和疊加強度上的優勢

2. **雷達圖比較** (`qnlp_radar_chart.png`)
   - 綜合展示兩種方法的量子特性輪廓
   - 直觀呈現各指標的相對強弱

### 數據文件

- **完整分析結果**: `results/real_qnlp_comparative_analysis.json`
- **視覺化數據**: `results/qnlp_visualization_data.json`

## 💡 應用場景

1. **學術研究**: 量子語言學理論驗證
2. **NLP優化**: 斷詞方法效果評估
3. **語義分析**: 深度語義理解研究
4. **多語言處理**: 中文文本處理優化

## 🔧 技術棧

- **量子計算**: Qiskit
- **機器學習**: Scikit-learn
- **數據處理**: Pandas, NumPy
- **視覺化**: Matplotlib, Seaborn
- **中文處理**: jieba
- **API集成**: OpenAI GPT

## 📝 注意事項

1. **API成本**: ChatGPT分析需要消耗API tokens，請注意使用成本
2. **理論性質**: QNLP分析具有理論探索性質，結果需要在量子語言學框架下理解
3. **樣本限制**: 當前分析基於有限樣本，擴大樣本可能產生不同結果
4. **中文字體**: 視覺化需要支持中文的字體，系統會自動選擇合適字體

## 🎯 未來發展

1. **擴大樣本規模**: 處理更多文本數據
2. **多維度分析**: 增加更多量子指標
3. **實時分析**: 開發在線分析工具
4. **多語言支持**: 擴展到其他語言的分析

## 📞 聯繫方式

如有問題或建議，歡迎通過以下方式聯繫：
- 項目Issues
- 學術討論
- 技術交流

---

**生成時間**: 2024年9月20日  
**項目版本**: v1.0  
**分析樣本**: jieba (5筆) + ChatGPT (5筆)  
**量子框架**: 基於Qiskit的QNLP理論實現
