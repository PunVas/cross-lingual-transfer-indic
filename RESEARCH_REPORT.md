# Research Report: Zero-Shot Cross-Lingual Transfer (English  Hindi)

**Project:** Cross-Lingual Natural Language Inference (XNLI)
**Focus:** Zero-Shot Transfer, Model Comparison, and Edge Optimization


---

## 1. Executive Summary

The primary goal of this experiment was to solve a data scarcity problem: **How can we perform Natural Language Inference (NLI) in Hindi without having a large, labeled Hindi training set?**

By leveraging the **XNLI dataset**, I investigated "Zero-Shot Transfer"—training a model exclusively on English data and evaluating its ability to generalize to Hindi. This report details the comparative performance of **mBERT** vs. **XLM-RoBERTa**, followed by an engineering focus on optimizing the winning model for production using **ONNX quantization**.

**Key Findings:**

* **XLM-RoBERTa** outperformed mBERT by **+10.04%** in accuracy.
* **ONNX Optimization** reduced inference latency by **6x** (down to ~24ms) and reduced model size by **73%**.

---

## 2. Methodology

### The Hypothesis

Multilingual models (like mBERT and XLM-R) share a vector space across languages. The hypothesis is that if the model learns the *logic* of entailment/contradiction in English, it can apply that logic to Hindi text if the cross-lingual alignment is strong enough.

### Experimental Setup

* **Task:** Natural Language Inference (Classifying pairs as *Entailment, Neutral, or Contradiction*).
* **Training Data:** English portion of the XNLI dataset (Sub-sampled to 20% for efficiency).
* **Evaluation Data:** Hindi validation set (Never seen during training).
* **Hardware:** Standard GPU for fine-tuning; CPU for inference benchmarks.

---

## 3. Results & Analysis

### A. Model Comparison (The "Battle of the Transformers")

I fine-tuned both models under identical hyperparameters to ensure a fair comparison. The results on the Hindi validation set were decisive.

| Model | Accuracy | F1 Score (Weighted) | Performance Gap |
| --- | --- | --- | --- |
| **XLM-RoBERTa (Base)** | **67.19%** | **0.6720** | **WINNER** |
| mBERT (Base) | 57.15% | 0.5672 | +10.04% |

**Analysis:**
The **10% gap** highlights the architectural differences. While mBERT maps languages to a shared space, XLM-RoBERTa was pre-trained on significantly more data (CommonCrawl) with a training objective specifically designed to encourage better cross-lingual representation. For Hindi, which is linguistically distant from English, XLM-R's deeper alignment proved superior.

### B. Deployment Optimization (ONNX)

High accuracy is useless if the model is too heavy for deployment. I took the winning model (XLM-R) and applied **ONNX Runtime** conversion with **Dynamic Int8 Quantization**.

**Benchmarking Results:**

| Metric | PyTorch (Original) | ONNX (Optimized) | Impact |
| --- | --- | --- | --- |
| **Inference Time** | 143.38 ms | **23.93 ms** | **6.0x FASTER** |
| **Model Size** | 1081.82 MB | **287.08 MB** | **73% SMALLER** |

**Implication:**
The quantized model is now small enough to fit on edge devices or cheap CPU instances without noticeable lag for the end-user. The 143ms latency of the original model would have felt sluggish in a real-time web app; 24ms is imperceptible.

---

## 4. Workflow Overview

The project was executed in three distinct phases:

1. **Training & Validation (`train_xnli.ipynb`):**
* Loaded the XNLI dataset.
* Fine-tuned `xlm-roberta-base` and `bert-base-multilingual-cased`.
* Logged metrics (Accuracy, F1) to compare zero-shot performance.


2. **Optimization:**
* Exported the trained PyTorch model to `.onnx` format.
* Applied quantization to map 32-bit floats to 8-bit integers.


3. **Application Development (`app.py`):**
* Built a minimal Streamlit interface.
* Allows users to type a Premise and Hypothesis in Hindi (or English) and see the real-time classification.



---

## 5. Conclusion & Next Steps

This project successfully demonstrated that we do not need massive Hindi datasets to build a competent Hindi NLI system. XLM-RoBERTa serves as a powerful backbone for zero-shot transfer.

**Future Improvements:**

* **Few-Shot Learning:** Adding just 1% of Hindi data to the training set could likely boost accuracy to >75%.
* **Broader Language Support:** Since XLM-R is multilingual, this same pipeline could be extended to Swahili, Urdu, or Arabic without retraining the model structure.

---

## 6. Project Artifacts

The following files are included in this repository:

* `train_xnli.ipynb`: Complete training pipeline and analysis code.
* `app.py`: Streamlit web application for testing the model.
* `requirements.txt`: Python dependencies.
* `README.md`: Project overview and setup instructions.