# BRIDGE 代码发布包

本包将服务器上的分层图模型及后续 C2F 细化代码整理为可安装的 Python 包，统一了数据格式、训练、预测、评价和旧 checkpoint 导入入口。版本为 `0.1.0rc1`；最终论文代码版本及许可证仍待作者确认，详见英文 README、`LICENSING.md` 和来源记录。

## 快速使用

```bash
python -m pip install .
bridge-st make-demo --out data/demo
bridge-st train --data data/demo/synthetic.npz --config configs/demo.json --out outputs/demo
bridge-st predict --data data/demo/synthetic_inputs.npz --checkpoint outputs/demo/model.pt --out outputs/demo_predictions.npz
bridge-st evaluate --reference data/demo/synthetic.npz --predictions outputs/demo_predictions.npz --out outputs/demo_metrics
```

示例完全由合成数据生成，不含真实组织图像、样本数据或论文模型权重。原始研究数据转换需安装 `python -m pip install '.[hd]'`。

## 包内内容

- `src/bridge_st/`：原模型定义、数据读写、训练/预测、指标及 CLI。
- `configs/`：C2F 基线、细胞类型先验＋分位数损失、分层图模型及短演示配置。
- `docs/`：输入格式、数据准备、指标定义及验证记录。
- `provenance/`：原文件路径的项目内相对表示、SHA256、恢复的原训练脚本文本和整理差异。
- `tests/`：训练预测全流程、无标签推理、留出标签隔离、checkpoint 兼容和 HD 格式转换测试。

训练使用细胞真实表达监督；预测只读取粗 spot 表达、图像特征、位置、spot 归属和基因面板，不读取细胞表达标签。原始有向量缓存的 HD 输入可通过 `bridge-st prepare-legacy` 转换，数据规范见 `docs/data-format.md`。

分层图和 C2F 是恢复的两套入口；C2F 入口仅包含实际参与前向计算的模块。旧 C2F 训练脚本中的测试 spot 标签使用、重复遍历邻居导致的采样差异等均在发布整理说明中列出；本版使用明确的留出隔离和每 epoch 一次的 spot 遍历。重新训练后的数值不能直接视为原论文表格的复现。

本包未捆绑 CellViT/图像特征提取器、真实权重和公开数据副本；也没有将原作图中依赖 GT 的阈值/均值调整放入预测。模型输入需要与训练时的特征提取器、表达尺度和基因顺序一致。

正式公开前需要作者确认最终代码版本、许可证及是否提供相应真实权重/数据访问方式。
