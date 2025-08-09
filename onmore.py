import os
import json
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline
import openai 

#badhi api keys load kari
load_dotenv()

API_KEYS = {
    "gpt-3.5-turbo": os.getenv("OPENAI_API_KEY"),
    "claude-3-sonnet": os.getenv("ANTHROPIC_API_KEY"),
    "gemini-1.5": os.getenv("GOOGLE_API_KEY"),
    "mistral-7b": os.getenv("MISTRAL_API_KEY"),
    "deepseek-coder": os.getenv("DEEPSEEK_API_KEY"),
    "grok": os.getenv("GROK_API_KEY"),
    "glm-4.5": os.getenv("ZHIPU_API_KEY"),
    "qwen": os.getenv("QWEN_API_KEY")
}

# intent classifier setup
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#total intents
INTENT_CANDIDATES = [
    "code", "creative", "general", "math", "research", "humor", "translation", "finance", "medical",
    "legal", "education", "science", "history", "sports", "business", "marketing", "healthcare",
    "music", "news", "tech", "travel", "gaming", "shopping", "interview", "resume", "presentation",
    "document_summary", "chat", "counseling", "startup", "analytics", "email", "productivity", "story"
]

# intent - model mapping
INTENT_TO_MODEL = {
    "code": "deepseek-coder",
    "creative": "glm-4.5",
    "general": "glm-4.5",
    "math": "glm-4.5",
    "research": "glm-4.5",
    "humor": "grok",
    "translation": "glm-4.5",
    "finance": "glm-4.5",
    "medical": "glm-4.5",
    "legal": "glm-4.5",
    "education": "glm-4.5",
    "science": "glm-4.5",
    "history": "glm-4.5",
    "sports": "gpt-3.5-turbo",
    "business": "glm-4.5",
    "marketing": "gpt-3.5-turbo",
    "healthcare": "glm-4.5",
    "music": "grok",
    "news": "gpt-3.5-turbo",
    "tech": "glm-4.5",
    "travel": "glm-4.5",
    "gaming": "grok",
    "shopping": "gpt-3.5-turbo",
    "interview": "glm-4.5",
    "resume": "glm-4.5",
    "presentation": "glm-4.5",
    "document_summary": "glm-4.5",
    "chat": "glm-4.5",
    "counseling": "glm-4.5",
    "startup": "glm-4.5",
    "analytics": "glm-4.5",
    "email": "glm-4.5",
    "productivity": "glm-4.5",
    "story": "glm-4.5"
}

# templete for structured prompts based on intent
INTENT_TO_TEMPLATE = {
    "code": "You are a coding expert. Solve the programming task below:\n\n{input}",
    "creative": "You are a creative writer. Create something imaginative based on:\n\n{input}",
    "general": "You are a helpful assistant. Provide a thoughtful response to:\n\n{input}",
    "math": "You are a math expert. Solve and explain:\n\n{input}",
    "research": "You are a research assistant. Provide insights about:\n\n{input}",
    "humor": "Add humor or make this funnier:\n\n{input}",
    "translation": "Translate this content into another language:\n\n{input}",
    "finance": "You are a financial analyst. Provide advice or analysis for:\n\n{input}",
    "medical": "You are a healthcare expert. Provide a medical perspective on:\n\n{input}",
    "legal": "You are a legal advisor. Address this issue:\n\n{input}",
    "education": "You are an educator. Explain or teach:\n\n{input}",
    "science": "You are a science expert. Discuss:\n\n{input}",
    "history": "You are a historian. Give context for:\n\n{input}",
    "sports": "You are a sports analyst. Comment on:\n\n{input}",
    "business": "You are a business consultant. Advise on:\n\n{input}",
    "marketing": "You are a marketing strategist. Provide insights on:\n\n{input}",
    "healthcare": "You are a health advisor. Explain:\n\n{input}",
    "music": "You are a music expert. Analyze or respond to:\n\n{input}",
    "news": "You are a journalist. Summarize or comment on:\n\n{input}",
    "tech": "You are a tech specialist. Explain:\n\n{input}",
    "travel": "You are a travel guide. Plan or suggest:\n\n{input}",
    "gaming": "You are a gaming expert. Review or advise on:\n\n{input}",
    "shopping": "You are a shopping assistant. Recommend or suggest:\n\n{input}",
    "interview": "You are an interview coach. Help prepare for:\n\n{input}",
    "resume": "You are a resume expert. Improve or review:\n\n{input}",
    "presentation": "You are a presentation coach. Assist with:\n\n{input}",
    "document_summary": "You are a summarization assistant. Summarize:\n\n{input}",
    "chat": "You are a friendly chatbot. Reply to:\n\n{input}",
    "counseling": "You are a compassionate counselor. Supportively respond to:\n\n{input}",
    "startup": "You are a startup mentor. Help with:\n\n{input}",
    "analytics": "You are a data analyst. Analyze this:\n\n{input}",
    "email": "You are an email assistant. Draft or enhance this:\n\n{input}",
    "productivity": "You are a productivity coach. Suggest improvements for:\n\n{input}",
    "story": "You are a storyteller. Create or continue a story from:\n\n{input}"
}

# z.ai if detecct intent confidence socre is low
def classify_with_zai(user_input):
    openai.api_key = API_KEYS["glm-4.5"]
    system_prompt = (
        "Classify the user's input into one of the following intents:\n" +
        ", ".join(INTENT_CANDIDATES) +
        "\nReply with just the best matching intent."
    )
    response = openai.ChatCompletion.create(
        model="glm-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message["content"].strip().lower()

# intent detection 
def detect_intent(user_input):
    result = intent_classifier(user_input, INTENT_CANDIDATES)
    top_intent = result["labels"][0]
    confidence = result["scores"][0] * 10

    if confidence >= 7:
        return top_intent, confidence
    else:
        fallback_intent = classify_with_zai(user_input)
        return fallback_intent, confidence

# prompt restructuring based on intent
def restructure_prompt(intent, user_input):
    template = INTENT_TO_TEMPLATE.get(intent, "{input}")
    return template.replace("{input}", user_input.strip().capitalize())

# simulated model response (for demonstration purposes only)
def simulate_model_response(model_name, prompt):
    return f"[{model_name}] would respond to:\n{prompt}"

# output optimization using Z.AI
def optimize_response_with_zai(full_output):
    openai.api_key = API_KEYS["glm-4.5"]
    system_prompt = (
        "You are an optimization agent. find the best fit the following content "
        "into a polished, high-quality, coherent response of about 300 to 400 words."
    )
    response = openai.ChatCompletion.create(
        model="glm-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_output}
        ],
        temperature=0.7
    )
    return response.choices[0].message["content"].strip()

# saving history to a JSON file
def append_to_history(entry):
    file = "autoflow_history.json"
    if not os.path.exists(file):
        with open(file, "w") as f:
            json.dump([], f)
    with open(file, "r") as f:
        history = json.load(f)
    history.append(entry)
    with open(file, "w") as f:
        json.dump(history, f, indent=4)

# main processing function
def process_input(user_input):
    intent, confidence = detect_intent(user_input)
    model = INTENT_TO_MODEL.get(intent, "glm-4.5")
    api_key = API_KEYS.get(model)

    if not api_key:
        return {"error": f"API key for {model} is missing."}

    structured_prompt = restructure_prompt(intent, user_input)

    # Simulated output 
    full_output = simulate_model_response(model, structured_prompt)

    # optimize output using GLM-4.5
    optimized_output = optimize_response_with_zai(full_output)

    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "intent": intent,
        "confidence": round(confidence, 2),
        "model": model,
        "input": user_input,
        "structured_prompt": structured_prompt,
        "raw_output": full_output,
        "optimized_output": optimized_output
    }

    append_to_history(log)
    return log

# main function to run the script
def main():
    print("🧠 AutoFlow | Optimized Output via Z.AI\n")
    while True:
        user_input = input("Your input: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Session ended.")
            break

        result = process_input(user_input)
        if "error" in result:
            print("Error:", result["error"])
        else:
            print("\nIntent:", result["intent"])
            print("Confidence:", result["confidence"])
            print("Model Used:", result["model"])
            print("Structured Prompt:\n", result["structured_prompt"])
            print("\nOptimized Output:\n", result["optimized_output"])
            print("History saved.\n")

if __name__ == "__main__":
    main()




'''1. no hardcoded not even intents not prompts
   2. langchain model intents mate and confidentiality vadu bhi jovanu
   3. bahdu aapvanu optimization mate and agar koi explaination hot toh ene ek example sathe explain karvannu che '''