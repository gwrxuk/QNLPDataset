# Visual Asset Plan for 30-Slide Presentation

| Slide(s) | Visual Asset | Path | Notes |
|---|---|---|---|
| 1 | Project title card (optional logo) | *(Add conference branding)* | Use vector/logo from host institution if available. |
| 8–10 | DisCoCat segmentation overview | `results/news標題_discocat_segmentation.csv` (table) | Convert to screenshot/table of category counts. |
| 9 | Qubit mapping table | `20250927-image/pos_to_qubit_mapping_table.md` | Render as slide graphic showing category→qubit indices. |
| 10 | Quantum circuit layout | `paper/quantum_circuit_diagram.png` | Generated via `paper/quantum_circuit_visualization.py`. |
| 11 | Circuit flow diagram | `paper/quantum_circuit_flow.png` | Illustrates preprocessing → circuit steps. |
| 12 | Density matrix workflow | `paper/density_matrix_complete_workflow.png` | Pair with step-by-step text. |
| 13 | Metric formulas | `paper/frame_competition_illustration_fixed.py` output (`frame_competition_main.png`) | Shows normalized entropy definition. |
| 14–17 | AI vs CNA comparison table | Recreate from `paper/technical_methodology_detailed_explanation.md` §6.11 | Embed as formatted table. |
| 18 | Qubit count example diagram | `paper/qubit_calculation_example.png` | Visualize the 麥當勞 case mapping. |
| 19 | Sentiment visualization | `sentiment_prediction_from_paper/sentiment_analysis_visualizations.png` | Six-panel figure summarizing sentiment results. |
| 20 | Emotion visualization | `sentiment_prediction_from_paper/emotion_analysis_visualizations.png` | Four-class emotion heatmap & metrics. |
| 21–24 | Radar or bar comparisons | `20250927-image/quantum_radar_comparison.png`, `20250927-image/distribution_analysis.png` | Highlight major metric deltas. |
| 25 | Frame competition illustration | `paper/frame_competition_illustration_fixed.png` | Use simplified ASCII-safe variant. |
| 26 | Workflow summary | `paper/density_matrix_complete_workflow.png` (or `visualizations/analysis_pipeline.png` if created) | Shows reproducibility pipeline. |
| 27 | Key findings infographic | `visualizations/field_level_heatmap.png` | Reinforces dataset-wide insights. |
| 29 | Future work roadmap | *(Create simple icon layout)* | Optional new slide diagram (PowerPoint SmartArt). |
| 30 | Contact/Thank you | *(Custom)* | Include QR code to repository if desired. |

**Implementation Tips**
- Verify all PNGs render Chinese characters without missing glyphs (use fixed scripts with fallback fonts).
- Prioritize 16:9 aspect ratio exports at ≥1920×1080 for clarity on large screens.
- When embedding tables, convert CSV snippets into clean slide-native tables rather than screenshots where possible.
- For animated explanations, consider sequencing `quantum_circuit_flow.png` elements over multiple builds.
