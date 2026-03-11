# Medical AutoML

**Automated Medical Research with LLM-powered Architecture Search**

This project enables autonomous AI-driven experimentation for cardiovascular disease diagnosis using transformer architectures. The system automatically explores optimal model configurations through iterative experimentation, guided by clinical metrics (AUC, Sensitivity, Specificity).

## 🎯 Key Features

- **Autonomous Architecture Search**: AI agents automatically modify model architecture and hyperparameters
- **Clinical-Focused Evaluation**: Optimized for real-world medical metrics (AUC, Sensitivity, Specificity) rather than just accuracy
- **Structured-to-Text Pipeline**: Novel approach converting structured patient data into natural language for transformer processing
- **Rapid Prototyping**: 5-minute training cycles enable 100+ experiments overnight
- **Cross-Platform**: Supports Apple Silicon (MPS), NVIDIA GPUs, and CPU environments

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Apple Silicon Mac or NVIDIA GPU
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Zhanbingli/medical-automl.git
cd medical-automl

# Install dependencies
uv sync

# Prepare data and train tokenizer (~2 min)
uv run prepare.py

# Run a single training experiment (~5 min)
uv run train.py
```

## 📊 Project Structure

```
medical-automl/
├── prepare.py          # Data preprocessing, tokenization, and evaluation metrics
├── train.py            # Model architecture, training loop, and hyperparameters
├── program.md          # Agent instructions for autonomous experimentation
├── patients.csv        # Cardiovascular patient dataset (303 samples)
├── data/               # Generated binary data and tokenizer
└── results_clinical.tsv # Experiment tracking
```

## 🔬 How It Works

### 1. Data Textualization
Structured patient records are converted into natural language:
```
患者特征：年龄63，性别1，胸痛类型1，静息血压145，胆固醇233，...
最终诊断结果为：0
```

### 2. Tokenization
Custom BPE tokenizer trained on medical Chinese text with 8,192 vocabulary size.

### 3. Autonomous Experimentation
AI agents iterate on `train.py` to optimize:
- Model architecture (depth, width, attention patterns)
- Hyperparameters (learning rates, dropout, batch size)
- Optimization strategies (Muon + AdamW)

### 4. Clinical Evaluation
Reports comprehensive clinical metrics:
- **AUC**: Area Under ROC Curve (primary metric)
- **Sensitivity**: True Positive Rate (minimize false negatives)
- **Specificity**: True Negative Rate (minimize false positives)

## 📈 Current Best Results

| Metric | Value | Description |
|--------|-------|-------------|
| AUC | 0.941 | ROC curve area |
| Accuracy | 0.828 | Overall correctness |
| Sensitivity | 0.824 | True positive rate |
| Specificity | 1.000 | True negative rate |

**Configuration**: ASPECT_RATIO=48, DROPOUT=0.2, DEPTH=3

## 🧪 Running Autonomous Experiments

1. **Read agent instructions**:
   ```bash
   cat program.md
   ```

2. **Start AI agent** (Claude/Codex/etc.):
   ```
   "Please read program.md and help me optimize the cardiovascular diagnosis model."
   ```

3. **Monitor progress**:
   ```bash
   tail -f run.log
   ```

4. **Track results**:
   ```bash
   cat results_clinical.tsv
   ```

## 🏗️ Architecture Highlights

- **GPT-style Transformer**: Decoder-only architecture with rotary positional embeddings
- **Muon Optimizer**: Advanced second-order optimization for 2D parameters
- **Value Embeddings**: Alternating layer enhancement mechanism
- **Sliding Window Attention**: Efficient attention patterns (SSSL configuration)

## 📚 Dataset

Based on the UCI Heart Disease dataset:
- 303 patient records
- 13 clinical features (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- Binary classification task (presence/absence of heart disease)

## 🤝 Contributing

This project welcomes contributions! Areas for improvement:
- Multi-dataset validation
- Cross-validation implementation
- Additional medical domains
- Interpretability tools (SHAP, attention visualization)

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{medical_automl,
  author = {Zhanbingli},
  title = {Medical AutoML: Autonomous LLM-powered Medical Research},
  url = {https://github.com/Zhanbingli/medical-automl},
  year = {2024}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

This project is inspired by autoresearch concepts but represents independent development focused on medical applications.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- PyTorch team for the deep learning framework
- rustbpe for high-performance tokenization

## 📧 Contact

For questions or collaboration:
- GitHub Issues: https://github.com/Zhanbingli/medical-automl/issues
- Author: https://github.com/Zhanbingli

---

**Disclaimer**: This project is for research and educational purposes only. Not intended for clinical use without proper validation and regulatory approval.
