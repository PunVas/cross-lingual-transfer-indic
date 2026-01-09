import streamlit as st
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os

st.set_page_config(page_title="Hindi NLI (ONNX)")

@st.cache_resource
def load():
    # path to quantized onnx model
    model_path = "onnx_q/model_quantized.onnx"
    tok_path = "onnx_q"
    
    if not os.path.exists(model_path):
        st.error(f"ONNX model not found at '{model_path}'")
        return None, None
    
    # load onnx session
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # load tokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    
    return sess, tok

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def pred(p, h, sess, tok):
    # tokenize
    inputs = tok(p, h, return_tensors="np", truncation=True, max_length=128, padding="max_length")
    
    # run inference
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    
    logits = sess.run(None, ort_inputs)[0]
    probs = softmax(logits[0])
    
    labels = ["entailment", "neutral", "contradiction"]
    return {labels[i]: float(probs[i]) for i in range(3)}

def main():
    st.title("Hindi NLI (ONNX Optimized)")
    st.write("6x Faster Inference | 73% Smaller Model")
    
    sess, tok = load()
    
    if sess is None:
        return

    p = st.text_area("Premise (Hindi)", "एक आदमी बाजार में सब्जियां खरीद रहा है।")
    h = st.text_area("Hypothesis (Hindi)", "वह आदमी कुछ खरीद रहा है।")
    
    if st.button("Check"):
        if p and h:
            res = pred(p, h, sess, tok)
            
            best = max(res, key=res.get)
            st.write(f"Prediction: **{best.upper()}** ({res[best]:.2%})")
            
            st.bar_chart(res)
        else:
            st.error("Enter text")

if __name__ == "__main__":
    main()
