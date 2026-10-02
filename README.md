# FORM: A Guardrail-Enforced RAG Assistant for Beginner Fitness Advice

> **Course:** PE6201 Emerging AI Technologies
> **System Name:** FORM (Fitness Guidance RAG Assistant)  
> **Author:** MA JIAN  

---

## 📌 Project Overview & User Persona

**FORM** is a domain-bounded Retrieval-Augmented Generation (RAG) system designed to provide grounded, evidence-based fitness and nutrition advice to gym beginners. Powered by an **LLM Safety Guardrail** and strictly grounded in **ACSM (American College of Sports Medicine) guidelines**, the system eliminates hazardous silent failures by intercepting medical, rehabilitation, or out-of-domain queries.

### Target User Persona
* **Persona:** "Tor", a 24-year-old novice gym-goer.
* **Context & Pain Point:** Tor stands in the weight room wanting to structure a safe 45-minute chest workout. Overwhelmed by contradictory online advice, he lacks exercise science expertise to distinguish safe practices from hazardous or injury-inducing protocols.
* **Intended Behavioral Change:** Shifts Tor away from scrolling unverified forums toward obtaining immediate, 3-step routine adjustments grounded directly in ACSM standards.

---

## 🔤 Input & Output Specification

### Input Specification
* **Format:** Natural language text query via CLI or API.
* **In-Domain Example:** `"How many sets should a beginner do for chest per week?"`
* **Out-of-Domain Example:** `"How can I rehab a torn rotator cuff at home?"`

### Output Specification
* **Format:** Strictly enforced JSON Object (`json_object`).
* **In-Domain Output (Structured Grounding):**
  ```json
  {
    "answer": "Beginners should perform 2-3 full-body sessions per week, targeting 1-3 sets per muscle group per session.",
    "cited_chunk_ids": ["ACSM_001"]
  }
* **Out-of-Domain Output (Safety Refusal):**
{
  "answer": "Insufficient verified data to answer safely",
  "cited_chunk_ids": []
}

---

## 🛠️ System Architecture

```text
               +-------------------------------------------------------+
               |                  User Natural Question                |
               +-------------------------------------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |          LLM Safety Guardrail Classifier              |
               +-------------------------------------------------------+
                                  /                 \
                     (Out-of-Domain/Medical)      (In-Domain Fitness)
                                /                     \
                               v                       v
               +-------------------------------+  +-------------------------------+
               | Standard Refusal Fallback:    |  | ACSM Vector / Keyword Search  |
               | "Insufficient verified data   |  | (15 Structured Chunks)        |
               |  to answer safely"            |  +-------------------------------+
               +-------------------------------+                  |
                                                                  v
                                                  +-------------------------------+
                                                  | Grounded Generator            |
                                                  | (gpt-4o-mini + JSON Schema)   |
                                                  +-------------------------------+
                                                                  |
                                                                  v
                                                  +-------------------------------+
                                                  | Verified Output + [Chunk_ID]  |
                                                  +-------------------------------+
```

---

## 📊 Empirical Evaluation & Benchmarks

The system was evaluated against **40 pre-frozen test cases** (25 in-domain fitness queries + 15 out-of-domain medical/injury queries) across 3 candidate approaches:

| Approach / Baseline | Abstention Rate (Out-of-Domain) | In-Domain Answer Rate | Faithfulness / Grounding | Total Evaluation Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Non-AI Keyword FAQ Baseline** | 100.0% | 96.0% | 76.0% | $0.00000 USD |
| **Zero-shot Ungrounded LLM** | 0.0% | 100.0% | 0.0% | $0.00120 USD |
| **FORM RAG Assistant (Proposed)** | **100.0%** | **84.0%** | **84.0%** | **$0.00586 USD** |

### Key Findings:
- **Zero-Shot Risk:** Ungrounded LLMs achieved **0.0% Abstention**, hallucinatorily answering dangerous medical/injury queries.
- **Safety Enforcement:** FORM RAG achieved **100.0% Abstention** on medical/rehab queries while maintaining an **84.0% In-Domain Utility Rate**.
- **Citation Grounding:** FORM RAG achieved **84.0% Faithfulness**, surpassing the 80% baseline requirement via structured JSON citation outputs.

---

## 🚀 Quick Start

### 1. Clone the repository & Install Dependencies
```bash
git clone [https://github.com/YOUR_USERNAME/FORM-RAG-Assistant.git](https://github.com/YOUR_USERNAME/FORM-RAG-Assistant.git)
cd FORM-RAG-Assistant
pip install -r requirements.txt
```

### 2. Set API Key & Run Evaluation
```bash
export OPENAI_API_KEY="your-openrouter-or-openai-key"
python main.py
```

---

## 📄 Repository Structure
- `main.py`: Core RAG execution, LLM Guardrail, and 40-case Evaluation Harness.
- `acsm_kb.json`: Cleaned ACSM guideline knowledge chunks.
- `test_cases.json`: Handcrafted 40 test cases paired with ground-truth assertions.
- `docs/`: Academic project report detailing business & technical trade-offs.
