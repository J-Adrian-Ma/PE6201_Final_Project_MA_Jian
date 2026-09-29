# FORM: A Guardrail-Enforced RAG Assistant for Beginner Fitness Advice

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Course Project:** PE6201 Emerging AI Technologies
> **Author:** MA JIAN  

## 📌 Project Overview
**FORM** is a domain-bounded Retrieval-Augmented Generation (RAG) system designed to provide grounded, evidence-based fitness and nutrition advice to gym beginners. Powered by an **LLM Safety Guardrail** and strictly grounded in **ACSM (American College of Sports Medicine) guidelines**, the system eliminates hazardous silent failures by intercepting medical, rehabilitation, or out-of-domain queries.

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
| **1. Non-AI Keyword FAQ Baseline** | 100.0% | 96.0% | 76.0% | $0.00000 USD |
| **2. Zero-shot Ungrounded LLM** | 0.0% | 100.0% | 0.0% | $0.00120 USD |
| **3. FORM RAG Assistant (Proposed)** | **100.0%** | **84.0%** | **84.0%** | **$0.00586 USD** |

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
