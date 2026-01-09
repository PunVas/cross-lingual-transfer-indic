import gradio as gr
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os

# load model
model_path = "onnx_q/model_quantized.onnx"
tok_path = "onnx_q"

sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
tok = AutoTokenizer.from_pretrained(tok_path)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(premise, hypothesis):
    if not premise or not hypothesis:
        return {"error": 1.0}
    
    inputs = tok(premise, hypothesis, return_tensors="np", truncation=True, max_length=128, padding="max_length")
    
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    logits = sess.run(None, ort_inputs)[0]
    probs = softmax(logits[0])
    
    return {
        "Entailment": float(probs[0]),
        "Neutral": float(probs[1]),
        "Contradiction": float(probs[2])
    }

# gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Premise (Hindi)", value="एक आदमी बाजार में सब्जियां खरीद रहा है।"),
        gr.Textbox(label="Hypothesis (Hindi)", value="वह आदमी कुछ खरीद रहा है।")
    ],
    outputs=gr.Label(num_top_classes=3),
    title="Hindi NLI (ONNX Optimized)",
    description="Cross-lingual Zero-Shot Transfer (English → Hindi) | 6x Faster | 73% Smaller",
    examples=[
        ["एक लड़की पार्क में खेल रही है।", "बच्चा बाहर है।"],
        ["आदमी किताब पढ़ रहा है।", "वह सो रहा है।"],
        ["महिला ने लाल ड्रेस पहनी है।", "वह पार्टी में जा रही है।"]
    ]
)

if __name__ == "__main__":
    demo.launch()
