import os
import json
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline
import openai
from typing import List, Dict, Any

# main setup loading .env files and config file
load_dotenv()

API_KEYS = {
    "gpt-3.5-turbo": os.getenv("OPENAI_API_KEY"),
    "gpt-4": os.getenv("OPENAI_API_KEY"),
    "claude-3-sonnet": os.getenv("ANTHROPIC_API_KEY"),
    "gemini-1.5": os.getenv("GOOGLE_API_KEY"),
    "mistral-7b": os.getenv("MISTRAL_API_KEY"),
    "deepseek-coder": os.getenv("DEEPSEEK_API_KEY"),
    "grok": os.getenv("GROK_API_KEY"),
    "glm-4.5": os.getenv("ZHIPU_API_KEY"),
    "qwen": os.getenv("QWEN_API_KEY")
}

CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

INTENT_CANDIDATES = CONFIG.get("intent_candidates", [])
INTENT_TO_MODEL = CONFIG.get("intent_to_model", {})
INTENT_TO_TEMPLATE = CONFIG.get("templates", {})
TOP_K = CONFIG.get("top_k", 5)
OPTIMIZER_CFG = CONFIG.get("optimizer", {"always_optimize": True, "glm_model_name": "glm-4"})
TIEBREAKER = CONFIG.get("tiebreaker", "highest_single_intent_score")

# HF zero-shot classifier (pehli vakhat download thase model so time lagse wait karvu padse)
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# intent ahiya detect thase jetla bhi intent config ma hase ae badha
def detect_intents_topk(user_input: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    try:
        res = intent_classifier(user_input, INTENT_CANDIDATES, multi_label=False)
        labels, scores = res["labels"], res["scores"]
    except Exception as e:
        print("HF classifier failed:", e)
        return fallback_detect_with_glm(user_input)

    intents = []
    for label, score in zip(labels, scores):
        scaled = round(score * 10, 4)  # 0..10 scale
        intents.append({"intent": label, "raw_score": score, "score": scaled})
    topk = intents[:k]

    # GLM model fallback if no strong intent detected
    if topk and topk[0]["score"] < 7.0:
        glm_intent = classify_with_zai(user_input)
        return [{"intent": glm_intent, "raw_score": 1.0, "score": 10.0}]

    return topk

# GLM fallback classifier using z api
def classify_with_zai(user_input: str) -> str:
    glm_api_key = API_KEYS.get("glm-4.5")
    if not glm_api_key:
        raise RuntimeError("GLM API key (ZHIPU_API_KEY) missing for fallback classifier.")
    openai.api_key = glm_api_key
    system_prompt = (
        "Classify the user's input into one of the following intents (reply with one label exactly):\n"
        + ", ".join(INTENT_CANDIDATES)
    )
    try:
        response = openai.ChatCompletion.create(
            model=OPTIMIZER_CFG.get("glm_model_name", "glm-4"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=64,
            temperature=0.0
        )
        out = response.choices[0].message["content"].strip().lower()
        # normalization: try exact match or contains
        for c in INTENT_CANDIDATES:
            if out == c.lower():
                return c
        for c in INTENT_CANDIDATES:
            if c.lower() in out:
                return c
        return "general"
    except Exception as e:
        print("GLM fallback classifier failed:", e)
        return "general"

def fallback_detect_with_glm(user_input: str) -> List[Dict[str, Any]]:
    intent = classify_with_zai(user_input)
    return [{"intent": intent, "raw_score": 1.0, "score": 10.0}]

# map intent - model and aggregate scores
def map_intents_to_models(top_intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapped = []
    for it in top_intents:
        model = INTENT_TO_MODEL.get(it["intent"], "glm-4.5")
        mapped.append({
            "intent": it["intent"],
            "score": it["score"],
            "raw_score": it.get("raw_score"),
            "model": model
        })
    return mapped

def aggregate_scores_by_model(mapped_list: List[Dict[str, Any]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for item in mapped_list:
        m = item["model"]
        s = float(item["score"])
        agg[m] = agg.get(m, 0.0) + s
    return agg

def select_best_model(agg: Dict[str, float], mapped_list: List[Dict[str, Any]]) -> str:
    if not agg:
        return "glm-4.5"
    best_model = max(agg.items(), key=lambda kv: kv[1])[0]
    max_value = agg[best_model]
    tied = [m for m, v in agg.items() if abs(v - max_value) < 1e-9]
    if len(tied) == 1:
        return best_model
    #highest single intent score owner
    highest_intent_score = -1.0
    chosen = tied[0]
    for item in mapped_list:
        if item["model"] in tied and item["score"] > highest_intent_score:
            highest_intent_score = item["score"]
            chosen = item["model"]
    return chosen

#restructuirng
def restructure_prompt(top_intents: List[Dict[str, Any]], user_input: str) -> str:
    top_intent_label = top_intents[0]["intent"] if top_intents else "general"
    template = INTENT_TO_TEMPLATE.get(top_intent_label, "{input}")
    detected = ", ".join([f"{i['intent']}({i['score']:.2f})" for i in top_intents])
    context = f"Detected intents (top {len(top_intents)}): {detected}.\nUse role/template for: {top_intent_label}.\n\n"
    body = template.replace("{input}", user_input.strip())
    return context + body

# model calls
def simulate_model_response(model_name: str, prompt: str) -> str:
    return f"[SIMULATED {model_name}] Replace with vendor SDK. Prompt preview:\n{prompt[:400]}"

def call_model(model_name: str, prompt: str) -> str:
    api_key = API_KEYS.get(model_name)
    if not api_key:
        return f"[ERROR] API key for model '{model_name}' missing."
    # mostly badha gpt use thay sake and if want specific then naam add kari de 
    if model_name.startswith("gpt-"):
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"[ERROR calling {model_name}] {e}"
    # GLM via openai wrapper
    if model_name == "glm-4.5":
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=OPTIMIZER_CFG.get("glm_model_name", "glm-4"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"[ERROR calling {model_name}] {e}"
    return simulate_model_response(model_name, prompt)

#optimization
EXPLAIN_KEYWORDS = ["explain", "how does", "how do", "how to", "why", "describe", "what is", "illustrate", "example"]

def is_explanation_request(user_input: str, top_intents: List[Dict[str, Any]]) -> bool:
    lower = user_input.lower()
    if any(kw in lower for kw in EXPLAIN_KEYWORDS):
        return True
    for it in top_intents:
        if it["intent"].lower() in {"explanation", "explain"}:
            return True
    return False

def optimize_response_with_zai(raw_output: str, top_intents: List[Dict[str, Any]], selected_model: str, user_input: str) -> str:
    glm_api_key = API_KEYS.get("glm-4.5")
    if not glm_api_key:
        return raw_output + "\n\n[Note: GLM optimizer not available - ZHIPU_API_KEY missing.]"

    openai.api_key = glm_api_key
    detected = ", ".join([f"{i['intent']}({i['score']:.2f})" for i in top_intents])
    explanation_needed = is_explanation_request(user_input, top_intents)

    system_prompt = (
        "You are an optimization agent. Polishing assistant output for clarity, conciseness and usefulness.\n"
        "Produce a polished answer between 300 and 450 words. Do NOT append a separate summary line.\n"
    )

    user_msg = (
        f"Context: Detected intents: {detected}. Selected LLM used: {selected_model}.\n\n"
        f"Original user input: {user_input}\n\n"
        f"Raw model output:\n{raw_output}\n\n"
    )
    if explanation_needed:
        user_msg += (
            "The user is asking for an explanation. Include a short, concrete example (2-3 sentences) that illustrates the explanation in simple layman terms. "
            "Ensure the example is contextually relevant and easy to understand.\n\n"
        )
    user_msg += "Please produce a polished final response of about 300-450 words. Do not add any extra summary lines."

    try:
        response = openai.ChatCompletion.create(
            model=OPTIMIZER_CFG.get("glm_model_name", "glm-4"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        polished = response.choices[0].message["content"].strip()
        return polished
    except Exception as e:
        return raw_output + f"\n\n[Optimizer failed: {e}]"

# history 
def append_to_history(entry: Dict[str, Any]):
    file = "autoflow_history.json"
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            json.dump([], f)
    with open(file, "r", encoding="utf-8") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = []
    history.append(entry)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

#main processing
def process_input(user_input: str) -> Dict[str, Any]:
    top_intents = detect_intents_topk(user_input, k=TOP_K)
    mapped = map_intents_to_models(top_intents)
    agg = aggregate_scores_by_model(mapped)
    selected_model = select_best_model(agg, mapped)

    # api there if not fallback to glm-4.5
    if not API_KEYS.get(selected_model):
        sorted_models = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        fallback_model = None
        for m, _ in sorted_models:
            if API_KEYS.get(m):
                fallback_model = m
                break
        if fallback_model:
            selected_model = fallback_model
        else:
            if API_KEYS.get("glm-4.5"):
                selected_model = "glm-4.5"
            else:
                return {"error": "No available API keys for any candidate models."}

    structured_prompt = restructure_prompt(top_intents, user_input)
    raw_output = call_model(selected_model, structured_prompt)

    if isinstance(raw_output, str) and raw_output.startswith("[ERROR"):
        return {"error": raw_output}

    # optimize agar configured (and GLM key present)
    optimized_output = raw_output
    if OPTIMIZER_CFG.get("always_optimize", True) and API_KEYS.get("glm-4.5"):
        optimized_output = optimize_response_with_zai(raw_output, top_intents, selected_model, user_input)

    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": user_input,
        "top_intents": top_intents,
        "mapped_intents": mapped,
        "aggregated_model_scores": agg,
        "selected_model": selected_model,
        "structured_prompt": structured_prompt,
        "raw_output": raw_output,
        "optimized_output": optimized_output
    }
    append_to_history(log)

    result = {
        **log,
        "intent": top_intents[0]["intent"] if top_intents else "general",
        "confidence": top_intents[0]["score"] if top_intents else 0.0
    }
    return result

#main function 
def main():
    print("AutoFlow | Single-run CLI\n")
    user_input = input("Your input: ").strip()
    if not user_input:
        print("No input provided. Exiting.")
        return

    result = process_input(user_input)
    if "error" in result:
        print("Error:", result["error"])
        return

    print("\n--- Results ---")
    print("Intent:", result["intent"])
    print("Confidence:", f"{result['confidence']}/10")
    print("Model Used:", result["selected_model"])
    print("\nStructured Prompt (preview):\n", result["structured_prompt"][:1000])
    print("\nOptimized Output:\n", result["optimized_output"][:5000])
    print("\nHistory saved to autoflow_history.json\n")

if __name__ == "__main__":
    main()




'''1. no hardcoded not even intents not prompts.   done
   2. langchain model intents mate and confidentiality vadu bhi jovanu.      done
   3. bahdu aapvanu optimization mate and agar koi explaination hot toh ene ek example sathe explain karvannu che '''