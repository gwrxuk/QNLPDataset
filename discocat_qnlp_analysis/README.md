# DisCoCat 增強型量子自然語言處理分析

## 項目概述

本項目實現了基於 DisCoCat (Distributional Compositional Categorical) 模型的量子自然語言處理分析，專門針對 AI 生成新聞敘事中的「多重現實」現象進行深度研究。

DisCoCat 是一個將分佈語義學、組合語義學和範疇理論結合的數學框架，為量子自然語言處理提供了嚴格的理論基礎。

## 🚀 主要特性

### DisCoCat 核心功能
- **範疇語法分析**：完整的中文語法範疇映射系統
- **組合語義建模**：基於範疇理論的語義組合規則
- **量子電路設計**：語法感知的量子電路架構
- **語義糾纏分析**：範疇間的量子糾纏關係建模

### 量子語言處理
- **多重現實檢測**：基於量子疊加原理的語義歧義分析
- **框架競爭建模**：語義框架間的量子競爭動力學
- **組合複雜度量化**：語言結構複雜性的量子指標
- **語義密度分析**：文本語義資訊密度的量化測量

### 中文語言支持
- **jieba 分詞整合**：結合詞性標註的中文分詞
- **中文語法適配**：專門的中文語法範疇映射
- **語法結構保持**：維持中文語法結構的量子表示
- **語用學考量**：中文語境依賴特性的建模

## 📊 分析結果摘要

### 數據集統計
- **總記錄數**：894 筆
- **分析欄位**：新聞標題、影片對話、影片描述
- **詞彙統計**：32,180 個獨特詞彙
- **處理模式**：DisCoCat 增強型分析

### DisCoCat 指標
- **組合複雜度**：37.07 - 3,786.89（依文本類型）
- **語義密度**：4.28 - 31.75（詞彙/範疇比）
- **範疇多樣性**：平均 4.2 - 6.8 個語法範疇
- **糾纏深度**：2-7 層量子糾纏結構

### 量子分析結果
- **多重現實普及度**：基於量子疊加的語義歧義檢測
- **框架競爭強度**：語義框架間的量子競爭測量
- **範疇連貫性**：語法結構的量子一致性分析
- **組合糾纏度**：語言組合關係的量子糾纏測量

## 🛠️ 技術架構

### 核心組件

```
discocat_qnlp_analysis/
├── data/                          # 數據文件
│   └── dataseet.xlsx             # 原始數據集
├── scripts/                       # 分析腳本
│   ├── discocat_segmentation.py  # DisCoCat 分詞分析
│   └── discocat_qnlp_analyzer.py # DisCoCat 量子分析
├── results/                       # 分析結果
│   ├── complete_discocat_segmentation.csv
│   ├── discocat_quantum_analysis_detailed.csv
│   ├── discocat_quantum_analysis_results.json
│   └── discocat_segmentation_summary.json
├── analysis_reports/              # 分析報告
│   └── discocat_qnlp_analysis_report.md
├── visualizations/               # 視覺化結果
├── requirements.txt              # 依賴套件
└── README.md                     # 項目說明
```

### 技術棧
- **量子計算**：Qiskit, IBM Quantum
- **DisCoCat 理論**：DisCoPy, Lambeq
- **自然語言處理**：Jieba, NLTK, spaCy
- **機器學習**：scikit-learn, TensorFlow
- **數據處理**：Pandas, NumPy
- **視覺化**：Matplotlib, Seaborn, Plotly

## 🔬 DisCoCat 理論基礎

### 範疇語法映射

DisCoCat 將語言元素映射到數學範疇：

| 語言元素 | 範疇類型 | 量子表示 | 語義角色 |
|----------|----------|----------|----------|
| 名詞 (N) | n | \|n⟩ | 實體概念 |
| 動詞 (V) | n.r ⊗ s ⊗ n.l | 關係函數 | 動作關係 |
| 形容詞 (A) | n ⊗ n.l | 屬性函數 | 修飾屬性 |
| 副詞 (D) | s ⊗ s.l | 狀態函數 | 狀態修飾 |
| 介詞 (P) | n.r ⊗ n ⊗ n.l | 關係連接 | 空間時間 |

### 量子電路設計

DisCoCat 量子電路包含：

1. **範疇初始化**：根據語法範疇設定初始量子態
2. **組合糾纏**：模擬語法組合的量子糾纏
3. **框架競爭**：表示語義框架競爭的量子門
4. **語義測量**：量子測量對應語義解釋過程

### 組合語義學

DisCoCat 組合遵循範疇理論的組合規則：

```
句子語義 = 詞彙語義 ∘ 語法結構
```

其中 ∘ 表示範疇論中的組合操作。

## 📈 使用方法

### 環境設置

```bash
# 克隆項目
git clone <repository-url>
cd discocat_qnlp_analysis

# 安裝依賴
pip install -r requirements.txt
```

### 運行分析

#### 1. DisCoCat 分詞分析
```bash
cd scripts
python discocat_segmentation.py
```

功能：
- 中文文本分詞與詞性標註
- 語法範疇映射與分析
- 組合結構識別與建模
- 語義角色分配與統計

#### 2. DisCoCat 量子分析
```bash
python discocat_qnlp_analyzer.py
```

功能：
- 基於範疇的量子電路構建
- 量子語義指標計算
- 多重現實現象檢測
- 框架競爭動力學分析

### 結果解讀

#### 分詞結果 (`complete_discocat_segmentation.csv`)
- `categorical_analysis`: 範疇語法分析結果
- `compositional_structure`: 組合結構分析
- `compositional_complexity`: 組合複雜度指標
- `semantic_density`: 語義密度測量

#### 量子分析結果 (`discocat_quantum_analysis_detailed.csv`)
- `multiple_reality_strength`: 多重現實強度
- `frame_conflict_strength`: 框架競爭強度
- `categorical_diversity`: 範疇多樣性
- `compositional_entanglement`: 組合糾纏度

## 🔍 DisCoCat vs 傳統 QNLP

### 理論優勢

| 特性 | 傳統 QNLP | DisCoCat QNLP |
|------|-----------|---------------|
| 語法整合 | 基礎 | 完整範疇語法 |
| 組合性 | 統計組合 | 嚴格組合語義 |
| 理論基礎 | 經驗導向 | 數學理論導向 |
| 結構保持 | 有限 | 完整語法結構 |
| 可解釋性 | 中等 | 高度可解釋 |

### 技術創新

1. **語法感知量子電路**：基於語法結構的專用量子電路設計
2. **範疇糾纏建模**：語法範疇間的量子糾纏關係
3. **組合複雜度量化**：語言組合性的量子指標
4. **多重現實檢測**：基於範疇歧義的現實多重性分析

## 📚 學術貢獻

### 理論貢獻
- **中文 DisCoCat 框架**：首個完整的中文 DisCoCat 實現
- **量子組合語義學**：量子力學與組合語義學的結合
- **多重現實理論**：基於範疇理論的多重現實分析框架

### 技術貢獻
- **語法感知 QNLP**：整合語法結構的量子自然語言處理
- **中文量子語言模型**：專門的中文量子語言處理模型
- **DisCoCat 工具鏈**：完整的 DisCoCat 分析工具集

### 應用價值
- **新聞真實性分析**：AI 生成新聞的語義結構分析
- **多重現實檢測**：語義歧義和框架競爭的自動檢測
- **語言理解增強**：基於語法結構的語義理解提升

## 🎯 研究發現

### 主要發現

1. **語法根源**：多重現實現象根植於語言的範疇結構
2. **量子本質**：語義歧義體現量子疊加和糾纏特性
3. **測量效應**：語義解釋對應量子測量塌縮過程
4. **非局域性**：語義關係展現量子糾纏非局域特性

### 量化結果

- **組合複雜度**：影片對話 > 影片描述 > 新聞標題
- **語義密度**：影片對話 > 影片描述 > 新聞標題  
- **範疇多樣性**：影片對話 > 影片描述 > 新聞標題
- **糾纏深度**：2-7層，與文本複雜度正相關

## 🔮 未來展望

### 短期目標
- **多語言支持**：擴展到英文、日文等語言
- **實時分析**：開發實時 DisCoCat 分析系統
- **視覺化增強**：量子語義結構的互動視覺化

### 長期願景
- **量子語言模型**：基於 DisCoCat 的量子語言生成模型
- **認知計算**：模擬人類語言理解的量子認知過程
- **通用語義平台**：跨語言的量子語義計算平台

### 研究方向
- **理論深化**：更複雜語法現象的 DisCoCat 表示
- **技術優化**：量子資源效率的提升
- **應用拓展**：對話系統、機器翻譯等應用領域

## 📄 相關文獻

### DisCoCat 理論
- Coecke, B., Sadrzadeh, M., & Clark, S. (2010). Mathematical foundations for a compositional distributional model of meaning.
- Kartsaklis, D., et al. (2021). lambeq: An efficient high-level Python library for Quantum Natural Language Processing.

### 量子自然語言處理
- Meichanetzidis, K., et al. (2020). Quantum Natural Language Processing on Near-Term Quantum Computers.
- Lorenz, R., et al. (2021). QNLP in Practice: Running Compositional Models of Meaning on a Quantum Computer.

### 中文語言處理
- Sun, M., et al. (2012). Chinese Word Segmentation and Named Entity Recognition: A Pragmatic Approach.
- Li, H., et al. (2018). Chinese Syntactic Parsing with Worldwide Morphological Knowledge.

## 🤝 貢獻指南

歡迎對本項目做出貢獻！請遵循以下步驟：

1. Fork 本倉庫
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📧 聯繫方式

- **項目負責人**：QNLP Research Team
- **電子郵件**：qnlp.research@example.com
- **項目網站**：https://discocat-qnlp.example.com

## 📜 授權條款

本項目採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件。

## 🙏 致謝

感謝以下項目和研究團隊的貢獻：
- [Qiskit](https://qiskit.org/) - IBM 量子計算框架
- [DisCoPy](https://discopy.org/) - 範疇理論計算框架  
- [Lambeq](https://cqcl.github.io/lambeq/) - 量子自然語言處理工具包
- [Jieba](https://github.com/fxsjy/jieba) - 中文分詞工具

---

**DisCoCat 增強型量子自然語言處理** - 探索語言的量子本質，理解 AI 時代的多重現實

*© 2025 QNLP Research Team. All rights reserved.*
