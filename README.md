# Cross-Lingual NLI Project (English -> Hindi)

Experimenting with **Zero-Shot Transfer** using XLM-RoBERTa. The goal was to see if a model trained *only* on English NLI data could correctly classify Hindi sentences.

Creating a multilingual NLI system usually requires expensive labeled data in every target language. I wanted to see if we could skip that by leveraging cross-lingual transfer.

## Results

I compared my fine-tuned **XLM-RoBERTa** against a standard **mBERT** baseline. Tested on the Hindi validation set (zero-shot).

| Model | Hindi Accuracy |
|-------|----------------|
| **My Model (XLM-R)** | **67.2%** |
| Baseline (mBERT) | 57.1% |

I also optimized it using **ONNX quantization** to make it faster for deployment.
- **Speedup:** 6x faster (24ms vs 143ms)
- **Size:** Reduced by ~73% (from 1GB to ~287MB)

## Model Weights
[Download Fine-Tuned Model (Google Drive)](https://drive.google.com/file/d/1XtzrB0JjxHkC612r7ASdMYIPOK-7K3eD/view?usp=sharing)

## Files
- `train_xnli.ipynb`: Main notebook. Handles data loading, training, comparison with mBERT, and ONNX conversion.
- `app.py`: Simple Streamlit app to try the model.
- `RESEARCH_REPORT.md`: My full analysis and observations.

## How to Run
1. Install requirements: `pip install -r requirements.txt` (or just run the first cell in the notebook).
2. Run the notebook `train_xnli.ipynb` in Colab or locally.
3. To try the app: `streamlit run app.py`

## What I Learned
- **Shared Vocabulary Matters:** XLM-R's larger vocab (250k) seems to help a lot with cross-lingual alignment compared to mBERT.
- **Quantization is powerful:** Did not expect a 4x size reduction with almost no accuracy drop.
