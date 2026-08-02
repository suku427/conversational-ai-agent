# Social-to-Lead Agentic AI Workflow

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python) ![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-orange?style=flat-square) ![Gemini](https://img.shields.io/badge/AI-Gemini_2.5_Pro-green?style=flat-square&logo=google) ![RAG](https://img.shields.io/badge/RAG-ChromaDB-purple?style=flat-square) ![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

A production-style **Conversational AI Sales Agent** built with LangGraph, Retrieval-Augmented Generation (RAG), and Google Gemini. The agent handles multi-turn conversations, answers pricing queries by retrieving from a local vector store, and intelligently captures sales leads only when all required details are present — no hallucinations, no premature actions.

---

## 🎯 What This Project Demonstrates

| Skill Area | Implementation |
|---|---|
| **Agentic AI** | Stateful cyclic graph with LangGraph — agent loops until task is complete |
| **RAG Pipeline** | ChromaDB + HuggingFace embeddings for grounded, hallucination-free answers |
| **Tool Calling** | LLM decides *when* to invoke tools vs. respond directly (ReAct pattern) |
| **State Management** | `AgentState` TypedDict persists full conversation history across turns |
| **LLM Integration** | Google Gemini 2.5 Pro via `langchain_google_genai` with temperature=0 for determinism |

---

## 🏗️ Architecture

```
User Message
     │
     ▼
┌─────────────┐
│  Agent Node  │  ◄──── SystemPrompt + Full Message History
│  (Gemini LLM)│
└──────┬──────┘
       │
       ├── Tool call needed? ──► ┌──────────────┐
       │                         │   Tool Node  │
       │                         │ ┌──────────┐ │
       │                         │ │RAG Lookup│ │ ◄── ChromaDB
       │                         │ └──────────┘ │
       │                         │ ┌──────────┐ │
       │                         │ │  Lead    │ │
       │                         │ │ Capture  │ │
       │                         │ └──────────┘ │
       │                         └──────┬───────┘
       │                                │
       │◄───── Tool result ─────────────┘
       │
       └── No tool needed? ──► Final Response to User
```

The graph is **cyclic by design**: after a tool executes, control returns to the Agent Node. This allows the agent to chain multiple tool calls (e.g., retrieve pricing → then capture a lead) or loop back to ask for missing information — something a linear chain (DAG) cannot do.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **LLM** | Google Gemini 2.5 Pro (`langchain_google_genai`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, local CPU) |
| **Vector Store** | ChromaDB (in-memory) |
| **RAG Framework** | LangChain |
| **Environment** | `python-dotenv` |

---

## 📂 Project Structure

```
social-lead-agent/
├── data/
│   └── knowledge_base.json    # Pricing plans and policies (RAG source of truth)
├── src/
│   ├── __init__.py
│   ├── agent.py               # LangGraph graph definition, tools, system prompt
│   └── rag.py                 # RAG pipeline: ChromaDB indexing + retriever
├── check_models.py            # Utility: lists available Gemini models for your API key
├── demo.py                    # Script for recording a demo session
├── test_agent.py              # Interactive CLI to chat with the agent
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A free [Google AI Studio](https://aistudio.google.com/) API key

### 1. Clone the Repository

```bash
git clone https://github.com/suku427/conversational-ai-agent.git
cd SocialScout
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB). This happens once automatically and is cached locally.

### 3. Configure Environment

Create a `.env` file in the project root:

```
GOOGLE_API_KEY="your_actual_api_key_here"
```

### 4. Run the Agent

```bash
python test_agent.py
```

You'll get an interactive CLI prompt. Type a message and press Enter to chat.

---

## 💬 Tested Conversation Scenarios

| User Input | Agent Behaviour |
|---|---|
| `"Hi"` | Responds with a polite greeting |
| `"What is the price of the Basic Plan?"` | Triggers `lookup_policy_pricing` → retrieves from ChromaDB → answers accurately |
| `"I want to sign up for the Pro plan"` | Detects high intent → asks for Name, Email, Platform sequentially |
| `"My name is Ravi, email is ravi@test.com, I use YouTube"` | All 3 fields present → calls `mock_lead_capture` → confirms lead captured |
| `"What is your refund policy?"` | Retrieves policy section from vector store → answers without hallucinating |

---

## 🎥 Demo



### Screenshots

**RAG Knowledge Retrieval** — Agent correctly identifies a pricing query, searches ChromaDB, and retrieves the Basic Plan details.

![RAG Proof](screenshots/rag_proof.png)

**Lead Capture** — Upon detecting high intent, the agent collects Name, Email, and Platform before calling the `mock_lead_capture` tool.

![Lead Capture Proof](screenshots/capture_proof.png)

---

## 🔑 Key Design Decisions

### Why LangGraph over LangChain LCEL?

Standard LCEL chains are **directed acyclic graphs (DAGs)** — they flow strictly from A → B → C. Real-world conversational agents need:

- **Cycles:** If a user provides partial information (only their name), the agent must loop back to ask for the remaining fields without restarting the entire session.
- **Dynamic branching:** The `tools_condition` edge decides at runtime whether to call a tool or respond directly — this cannot be hardcoded in a linear chain.

### Why Local Embeddings (HuggingFace) over OpenAI?

`sentence-transformers/all-MiniLM-L6-v2` runs entirely on CPU with **zero API calls and no rate limits**. For a knowledge base of this size (1 document), embedding quality is identical to paid alternatives while keeping the project free to run and reproduce.

### Temperature = 0

The agent is set to `temperature=0` for deterministic, repeatable outputs. Sales agents must behave consistently — a greeting should not randomly become a pricing pitch.

---

## 🌐 WhatsApp Deployment Strategy

To deploy this agent on WhatsApp Business:

1. **Wrap the agent in a FastAPI REST API** — expose a `POST /webhook` endpoint.
2. **Meta Webhook Integration** — Meta posts incoming WhatsApp messages as JSON to your endpoint.
3. **Per-user State Isolation** — use `sender_phone` as a `thread_id` key into Redis/Postgres to load each user's `AgentState` independently, enabling concurrent multi-user sessions.
4. **Send Responses** — POST the agent's output to `https://graph.facebook.com/v17.0/{phone_id}/messages` via the Meta Cloud API.
5. **Security** — Verify the `X-Hub-Signature-256` header on every incoming request to confirm it originates from Meta.

---

## ⚠️ API Rate Limits

This project uses **Gemini 2.5 Pro** on the free tier. If you encounter a `429 RESOURCE_EXHAUSTED` error during rapid testing, wait ~60 seconds for the quota window to reset. To check which models are available under your API key, run:

```bash
python check_models.py
```

---

## 👤 Author

**Bodapatla Sukumar**
[GitHub](https://github.com/suku427) · [LinkedIn](https://www.linkedin.com/in/sukumarbodapatla)
