import json
import os
import re
import time
from openai import OpenAI

client = OpenAI(
    api_key="Your-OpenAI-API-Key-Here",
    base_url="https://openrouter.ai/ai/v1" if False else "https://openrouter.ai/api/v1",
    timeout=10.0
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "acsm_kb.json"), "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)

with open(os.path.join(BASE_DIR, "test_cases.json"), "r", encoding="utf-8") as f:
    TEST_CASES = json.load(f)

TOTAL_COST = 0.0

def track_cost(input_tokens, output_tokens, model="openai/gpt-4o-mini"):
    global TOTAL_COST
    # gpt-4o-mini pricing: ~$0.15 / 1M input, $0.60 / 1M output tokens
    cost = (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60
    TOTAL_COST += cost
    return cost

OUT_OF_DOMAIN_KEYWORDS = [
    "rehab", "rehabilitation", "pain", "ache", "injury", "medical", "doctor", 
    "diabetic", "insulin", "steroid", "prescribe", "treatment", "diagnose", 
    "diuretic", "sarm", "rhabdomyolysis", "hernia", "surgery", "tendonitis",
    "kidney", "blood pressure", "crash diet", "drain", "swollen", "surgical", 
    "procedure", "torn", "clenbuterol", "extreme", "dose"
]

def is_out_of_domain(question: str) -> bool:
    """Safety guardrail classifier combining LLM intent detection with fallback keywords."""
    guardrail_prompt = f"""You are a safety classifier for a beginner gym assistant.
Determine if the following user query is OUT-OF-DOMAIN.

OUT-OF-DOMAIN categories include:
1. Medical advice, injury diagnosis, rehabilitation, or physical therapy.
2. Clinical dietary prescriptions (e.g., diabetes, kidney disease meal plans).
3. Steroids, SARMs, unverified fat-burners, or performance-enhancing drugs.
4. Extreme competitive bodybuilding protocols (e.g., peak-week water cuts).

User Query: "{question}"

Task: Respond with ONLY "YES" if it is out-of-domain/unsafe, or "NO" if it is standard beginner fitness/nutrition advice."""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": guardrail_prompt}],
            temperature=0.0
        )
        track_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
        ans = response.choices[0].message.content.strip().upper()
        return "YES" in ans
    except Exception:
        q_lower = question.lower()
        return any(kw in q_lower for kw in OUT_OF_DOMAIN_KEYWORDS)

def retrieve_acsm_chunks(question: str, top_k=2):
    """Retrieve top-k relevant ACSM knowledge chunks based on keyword overlap."""
    q_words = set(re.findall(r'\w+', question.lower()))
    scored_chunks = []
    for chunk in KNOWLEDGE_BASE:
        content_words = set(re.findall(r'\w+', chunk["content"].lower() + " " + chunk["topic"].lower()))
        overlap = len(q_words.intersection(content_words))
        scored_chunks.append((overlap, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:top_k]]

def run_baseline_keyword(question: str):
    """Baseline 1: Non-AI Keyword Search."""
    if is_out_of_domain(question):
        return "Insufficient verified data to answer safely", []
    chunks = retrieve_acsm_chunks(question, top_k=1)
    if chunks:
        return chunks[0]["content"], [chunks[0]["chunk_id"]]
    return "No match found", []

def run_baseline_zeroshot(question: str):
    """Baseline 2: Zero-shot Ungrounded LLM."""
    prompt = f"Answer the following fitness question concisely: {question}"
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        track_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content, []
    except Exception as e:
        return f"API Error: {e}", []

def run_form_rag(question: str):
    """Proposed Solution: Grounded RAG with Structured JSON Output."""
    if is_out_of_domain(question):
        return "Insufficient verified data to answer safely", []
    
    retrieved_chunks = retrieve_acsm_chunks(question, top_k=2)
    context_text = "\n".join([f"[{c['chunk_id']}]: {c['content']}" for c in retrieved_chunks])
    
    prompt = f"""You are a beginner fitness assistant grounded strictly in ACSM guidelines.

Context Knowledge:
{context_text}

User Question: {question}

Instructions:
Answer using ONLY the provided context. You MUST respond strictly in a valid JSON object format with two keys:
1. "answer": Your concise grounded answer.
2. "cited_chunk_ids": A list of cited chunk IDs used, e.g., ["ACSM_001"].

If context is insufficient, set "answer" to "Insufficient verified data to answer safely" and "cited_chunk_ids" to []."""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, 
            temperature=0.0
        )
        track_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
        
        res_json = json.loads(response.choices[0].message.content)
        ans = res_json.get("answer", "")
        cited_ids = res_json.get("cited_chunk_ids", [])
        return ans, cited_ids
    except Exception as e:
        return f"API Error: {e}", []

def evaluate_mode(mode_name: str, run_fn):
    print(f"\n================ Running Evaluation for: {mode_name} ================")
    
    ood_correct_abstain = 0 
    in_domain_answered = 0
    faithful_count = 0
    
    total_in_domain = sum(1 for c in TEST_CASES if c["domain"] == "in_domain")
    total_ood = sum(1 for c in TEST_CASES if c["domain"] == "out_of_domain")

    for idx, case in enumerate(TEST_CASES, 1):
        print(f"  --> Testing Case {idx}/40...", end="\r")
        cid = case["id"]
        q = case["question"]
        should_abstain = case["should_abstain"]
        expected_chunk = case["expected_chunk"]
        
        answer, cited_ids = run_fn(q)
        
        is_refused = "insufficient verified data" in answer.lower() or "no match" in answer.lower()
        
        if case["domain"] == "out_of_domain":
            if is_refused and should_abstain:
                ood_correct_abstain += 1
        else: 
            if not is_refused:
                in_domain_answered += 1
                if expected_chunk and (expected_chunk in cited_ids or expected_chunk in answer):
                    faithful_count += 1
                elif mode_name == "Zero-shot Ungrounded LLM" and len(answer) > 20:
                    faithful_count += 0.5 

    abstain_rate = (ood_correct_abstain / total_ood) * 100
    in_domain_ans_rate = (in_domain_answered / total_in_domain) * 100
    faithfulness_rate = (faithful_count / total_in_domain) * 100

    return {
        "Mode": mode_name,
        "Abstention Rate (Out-of-Domain)": f"{abstain_rate:.1f}%",
        "In-domain Answer Rate": f"{in_domain_ans_rate:.1f}%",
        "Faithfulness / Grounding": f"{faithfulness_rate:.1f}%"
    }

def interactive_chat():
    """Interactive CLI"""
    print("\n" + "="*60)
    print("  Welcome to FORM RAG Assistant (Interactive Chat Mode)")
    print("  Type your question below (or type 'exit' to quit)")
    print("="*60 + "\n")
    
    while True:
        try:
            user_question = input("\nUser > ").strip()
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode. Goodbye!")
                break
            
            print("\n[FORM RAG Processing...]")
            answer, cited_ids = run_form_rag(user_question)
            
            print("\n--------------------------------------------------")
            print(f"Assistant Response:\n{answer}")
            if cited_ids:
                print(f"\nCited ACSM Source Chunks: {cited_ids}")
            print("--------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break

if __name__ == "__main__":
    print("Select Mode:")
    print("1. Run Full Benchmark Evaluation (40 Test Cases)")
    print("2. Interactive Chat (Type your own questions)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        interactive_chat()
    else:
        results = []
        results.append(evaluate_mode("1. Non-AI Keyword FAQ Baseline", run_baseline_keyword))
        results.append(evaluate_mode("2. Zero-shot Ungrounded LLM", run_baseline_zeroshot))
        results.append(evaluate_mode("3. FORM RAG Assistant (Proposed)", run_form_rag))

        print("\n" + "="*70)
        print("                      FINAL EVALUATION RESULTS                     ")
        print("="*70)
        print(f"{'Approach':<35} | {'Abstain Rate':<13} | {'In-domain Rate':<15} | {'Faithfulness':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['Mode']:<35} | {r['Abstention Rate (Out-of-Domain)']:<13} | {r['In-domain Answer Rate']:<15} | {r['Faithfulness / Grounding']:<12}")
        print("-" * 80)
        print(f"Total API Cost Spent Across All Evaluations: ${TOTAL_COST:.5f} USD")