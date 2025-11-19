from binoculars import Binoculars
from pathlib import Path
from llamainteract.llamapythonapi import LlamaInterface
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
import numpy as np
import re

# Global model/ tokenizer cache
MODEL_CACHE = {}
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.set_grad_enabled(False)


# --- Dispatcher ---
def run_ai_detector(detector_name, df, answer_name):
    name = detector_name.lower()
    if name == "binoculars":
        return run_binoculars(df, answer_name)
    elif name == "roberta":
        return run_roberta(df, answer_name)
    elif name == "detectgpt":
        return run_detectgpt(df, answer_name)
    elif name == "llama":
        return run_llama_as_detector(df, answer_name)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


# --- Binoculars ---
def run_binoculars(df, answer_name):
    if "binoculars" not in MODEL_CACHE:
        print(f"[Binoculars] Loading models on {DEVICE}...")
        
        # Initialize Binoculars with model name/path strings
        # The library handles loading internally
        bino = Binoculars(
            observer_name_or_path="tiiuae/falcon-7b",
            performer_name_or_path="tiiuae/falcon-7b-instruct",
            use_bfloat16=True, # Use bfloat16 for memory efficiency
            max_token_observed=512
        )

        MODEL_CACHE["binoculars"] = bino
    else:
        bino = MODEL_CACHE["binoculars"]

    scores, preds = [], []
    for text in df[answer_name]:
        score = bino.compute_score(text)
        pred = bino.predict(text)
        scores.append(score)
        preds.append(pred)

    df[answer_name + "_detection_score"] = scores
    df[answer_name + "_detection_prediction"] = preds
    return df

# --- Llama ---
def run_llama_as_detector(df, answer_name):
    if "llama" not in MODEL_CACHE:
        print(f"[Llama] Loading models on {DEVICE}...")
        llama = LlamaInterface()
        MODEL_CACHE["llama"] = llama
    else:
        llama = MODEL_CACHE["llama"]

    scores, preds = [], []
    for text in df[answer_name]:
        score_prompt = f"Reply ONLY with a probability (e.g: '0.651') that the following text was AI generated:\n\n\n{text}"
        score_resp = llama.prompt(score_prompt)
        print(f"Score: {score_resp}")

        score_text = score_resp["choices"][0]["text"].strip()

        matches = re.findall(r"(\d*\.\d+|\d+)", score_text.replace(",", "."))

        if matches:
            numeric_vals = [float(m) for m in matches]
            score_val = float(np.mean(numeric_vals))  # average of all generated nums
        else:
            score_val = 0.5
            print("|NO ANSWER FOUND|")

        # Clamp if necessary
        # score_val = max(0.0, min(1.0, score_val))

        pred_prompt = (
            f"A detection model has assigned this answer a {score_val * 100:.2f}% chance of being AI generated. "
            f"Based on this score and your own judgement, evaluate whether the following text is AI generated. "
            f"Reply ONLY with 'human' or 'ai':\n\n\n{text[:1500]}"
        )

        pred_resp = llama.prompt(pred_prompt)
        pred_text = pred_resp["choices"][0]["text"].strip()

        scores.append(score_val)
        preds.append(pred_text)

    df[answer_name + "_detection_score"] = scores
    df[answer_name + "_detection_prediction"] = preds
    return df


# --- RoBERTa ---
def run_roberta(df, answer_name,batch_size=16):
    if "roberta" not in MODEL_CACHE:
        print(f"[RoBERTa] Using device: {DEVICE}")
        model_name = "roberta-base-openai-detector"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        model.eval()
        MODEL_CACHE['roberta'] = (model, tokenizer)
    
    model, tokenizer = MODEL_CACHE["roberta"]

    texts = [t for t in df[answer_name] if isinstance(t, str)]
    scores, preds = [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().numpy()

        scores.extend(probs.tolist())
        preds.extend(["AI-generated" if p > 0.5 else "Human-written" for p in probs])

    df[f"{answer_name}_detection_score"] = scores
    df[f"{answer_name}_detection_prediction"] = preds
    return df


# --- DetectGPT ---
def run_detectgpt(df, answer_name):
    if "detectgpt" not in MODEL_CACHE:
        print(f"[DetectGPT] Using device: {DEVICE}")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        model.eval()
        MODEL_CACHE['detectgpt'] = (model, tokenizer)
        
    model, tokenizer = MODEL_CACHE["detectgpt"]

    scores, preds = [], []

    for text in df[answer_name]:
        if not isinstance(text, str):
            continue
        
        # tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            # original log probability
            outputs = model(**inputs, labels=inputs["input_ids"])
            logp = -outputs.loss.item()

        # create a perturbed version of text (simple synonym shuffle or mask)
        words = text.split()
        if len(words) > 5:
            words[len(words)//2] = "the"  # simple perturbation
        perturbed = " ".join(words)

        inputs_pert = tokenizer(perturbed, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs_pert = model(**inputs_pert, labels=inputs_pert["input_ids"])
            logp_pert = -outputs_pert.loss.item()

        curvature = logp - logp_pert
        score = np.tanh(curvature * 10)  # normalize to [-1, 1]
        pred = "AI-generated" if score > 0 else "Human-written"

        scores.append(score)
        preds.append(pred)

    df[answer_name + "_detection_score"] = scores
    df[answer_name + "_detection_prediction"] = preds
    return df

# # --- Quick test block ---
# if __name__ == "__main__":
#     # Load only the first two rows to avoid long runtime
#     base_dir = Path(__file__).resolve().parent.parent  # go up from modules/ to project root
#     data_path = base_dir / "datasets" / "eli5_QA.parquet"

#     # Load only the first 2 rows for testing
#     df = pd.read_parquet(data_path).head(2)
#     print("Loaded DataFrame:")
#     print(df)

#     # Make copies so each detector doesn’t overwrite columns
#     df_bino = df.copy()
#     df_roberta = df.copy()
#     df_detect = df.copy()

#     # Run each detector on its DataFrame
#     #df_bino = run_ai_detector("binoculars", df_bino, "answer")
#     #print("Binoculars done")
#     df_roberta = run_ai_detector("roberta", df_roberta, "answer")
#     print("Roberta done")
#     df_detect = run_ai_detector("detectgpt", df_detect, "answer")
#     print("DetectGPt done")

#     # Combine the results into a single DataFrame view
#     combined = pd.DataFrame({
#         "question": df["question"],
#         "answer": df["answer"],
#         "roberta_score": df_roberta["answer_ai_detection_score"],
#         "roberta_pred": df_roberta["answer_ai_detection_prediction"],
#         "detectgpt_score": df_detect["answer_ai_detection_score"],
#         "detectgpt_pred": df_detect["answer_ai_detection_prediction"],
#     })

#     print("\n=== Combined Detection Results ===")
#     print(combined)
