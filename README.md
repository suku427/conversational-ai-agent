# AutoStream Conversational AI Agent

A practical conversational AI project for a SaaS business, designed to answer customer questions, retrieve product knowledge, and guide users through the sales/support flow using a stateful LangGraph agent.

This project is meant to be understandable by both non-technical stakeholders and engineers. In simple terms: it acts like an AI customer support assistant that can answer questions about pricing, plans, refund policy, and product features using company data and safe tool execution.

---

## 1. Why this project exists

Modern businesses often receive repetitive customer questions like:

- What does the Basic Plan cost?
- What is included in the Pro Plan?
- Do you offer refunds?
- Can I sign up for a plan?
- I need help with an issue

Instead of answering manually every time, this project builds an AI assistant that can:

- understand the user request,
- decide whether it needs company knowledge,
- get relevant context from a stored knowledge base,
- use tools when needed,
- give a grounded answer,
- and behave politely and consistently.

This is a real-world example of an agentic AI system: an LLM is not just answering arbitrary questions — it is making decisions, using tools, and staying within a business context.

---

## 2. What the project does in plain English

The AI agent handles product support and sales-related conversations for AutoStream.

It can:

- answer pricing and plan questions,
- summarize product features,
- answer refund/support policy questions,
- detect when a user wants to sign up,
- ask for name, email, and platform when needed,
- and respond safely when a request is out of scope, such as travel planning.

Example user conversations:

```text
User: What is the price of the Basic Plan?
Agent: The Basic Plan costs $29/month and includes 10 videos per month.

User: I want to sign up for the Pro plan.
Agent: I can help with that. Please share your name, email, and platform.

User: My name is Ravi and my email is ravi@example.com. I use YouTube.
Agent: Thanks, I have the details and can continue the workflow.

User: Can you plan a trip from Bengaluru to Kerala?
Agent: I can help with AutoStream pricing, plans, and product support — but I’m not set up to plan trips or itineraries.
```

---

## 3. Technical summary

This project combines:

- LangGraph for stateful agent orchestration
- LangChain for LLM integration
- Google Gemini as the model layer
- Retrieval-Augmented Generation (RAG) for grounded answers
- PostgreSQL + pgvector for semantic search
- SentenceTransformers for local embeddings when needed
- FastAPI for API exposure
- Docker for deployment readiness
- Python automation for testing and local usage

In simple engineering terms, the agent is a workflow engine that decides what to do next based on the user message.

---

## 4. High-level architecture

```text
User Request
    │
    ▼
LangGraph Agent
    │
    ├── Decide intent / route request
    │
    ├── If needed, retrieve product knowledge
    │   └── PostgreSQL pgvector + embeddings
    │
    ├── If needed, run safe tools
    │   ├── calculator
    │   ├── knowledge search
    │   └── current time
    │
    └── Finalize response
        │
        ├── Gemini LLM
        ├── model fallback chain
        └── safe fallback response
```

### Core idea

The system does not rely only on the LLM to “guess” the answer. It first checks whether the question requires factual data from the company knowledge base. If yes, it searches the retriever for relevant context. Then it combines that context with the user prompt and asks the LLM to produce a grounded response.

That reduces hallucination and increases reliability.

---

## 5. Why LangGraph is used

LangGraph is useful because real conversational agents are not one-step chains. They may need to:

- decide the next step,
- fetch context,
- use tools,
- loop back,
- and produce a final answer only after the right conditions are met.

This project uses a graph-based workflow instead of a flat prompt chain so that the agent can behave more like a real workflow engine.

Key benefits:

- better control over decision logic,
- clean separation of routing, retrieval, tool use, and finalization,
- easier debugging,
- scalable extension for future tools or business rules.

---

## 6. Why RAG matters here

RAG means Retrieval-Augmented Generation.

Instead of blindly trusting the LLM, the project retrieves relevant facts from a knowledge base before generating the final answer.

This is important because product questions need exact answers like:

- current pricing,
- support policy,
- business rules,
- product limitations,
- and plan comparisons.

The knowledge is stored in:

- `data/knowledge_base.json`
- PostgreSQL + pgvector for vector similarity search

This ensures answers are grounded in your business data rather than generic model memory.

---

## 7. Main project components

### 7.1 `src/agent.py`
This is the heart of the project.

It contains:

- the LangGraph state graph,
- routing logic,
- context decision logic,
- fallback model handling,
- final response generation,
- support/travel prompt handling,
- and the integration with Gemini.

This is where the agent decides:

- Should I answer directly?
- Do I need context?
- Do I need a tool?
- Should I finalize the answer?

### 7.2 `src/rag.py`
This file handles:

- local embeddings,
- PostgreSQL connection logic,
- knowledge-base ingestion,
- vector similarity retrieval,
- fallback to local JSON if DB retrieval fails.

It turns business knowledge into searchable vector data.

### 7.3 `src/tools.py`
This module defines the tools that the agent can safely use.

Built-in tools include:

- `calculator`: safe arithmetic evaluation
- `search_knowledge`: retrieves product information
- `current_time`: returns the current UTC timestamp

These tools provide real functionality while keeping operations constrained and safe.

### 7.4 `main.py`
This exposes the agent through a FastAPI web API.

Endpoints:

- `GET /health` → health check
- `POST /api/v1/chat` → sends a prompt and receives a response

This makes the project usable as a backend service or chat API.

### 7.5 `test_agent.py`
This is a simple CLI interface.

It allows a user to type a prompt in the terminal and get a response directly from the graph.

### 7.6 `ingest.py`
This is used to load the company knowledge into the database/vector system.

### 7.7 `data/knowledge_base.json`
This is the source of truth for product facts and support rules.

---

## 8. Project structure

```text
conversational-ai-agent/
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── rag.py
│   └── tools.py
├── data/
│   └── knowledge_base.json
├── tests/
│   ├── test_api_and_rag.py
│   └── test_agent_tools.py
├── .env
├── .env.example
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── check_models.py
├── demo.py
├── ingest.py
├── main.py
├── requirements.txt
├── test_agent.py
├── README.md
└── .gitignore
```

---

## 9. Technology stack

### LLM and AI

- Python 3.11+
- LangGraph
- LangChain
- Google Gemini
- LangChain Google GenAI integration

### Retrieval

- PostgreSQL
- pgvector
- SentenceTransformers
- JSON knowledge base fallback

### Backend

- FastAPI
- Pydantic

### Deployment and tooling

- Docker
- Docker Compose
- Python dotenv
- pytest

---

## 10. Environment configuration

Create a `.env` file based on `.env.example`.

Example:

```env
GOOGLE_API_KEY="your_google_api_key_here"
GEMINI_PRIMARY_MODEL="gemini-3.6-flash"
GEMINI_FALLBACK_MODELS="gemini-3.1-flash-lite,gemini-2.5-pro"
OPENAI_API_KEY="your_openai_api_key_here"
POSTGRES_DB="vectordb"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="admin"
PGHOST="localhost"
PGPORT="5432"
LANGSMITH_API_KEY="your_langsmith_key_here"
LANGCHAIN_TRACING_V2="false"
LANGCHAIN_PROJECT="autostream-agent"
```

### Important model configuration note

The project originally had an outdated Gemini model reference. The current version uses the valid model:

```text
gemini-3.6-flash
```

This is important because older names such as `gemini-2.5-flash` can return `404 NOT_FOUND` when they are no longer available to new users. The fallback logic is designed to retry only on valid retryable errors and move to the next configured model when needed.

---

## 11. Setup instructions

### Step 1: Clone the project

```bash
git clone https://github.com/your-username/conversational-ai-agent.git
cd conversational-ai-agent
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create environment file

```bash
copy .env.example .env
```

Then fill in your actual Google API key and database values.

### Step 5: Run the project

#### Option A: CLI mode

```bash
python test_agent.py
```

Example:

```text
You: hii
Agent: Hello! Welcome to AutoStream. How can I help you today?
```

#### Option B: API mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then call:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the price of the Basic Plan?"}'
```

#### Option C: Docker

```bash
docker compose up --build
```

---

## 12. How the agent makes decisions

The system executes a stateful decision flow.

### Routing flow

1. User sends a message.
2. The `decide_context_node` checks the message.
3. It decides whether the request is:
   - general greeting,
   - pricing question,
   - policy question,
   - lead capture flow,
   - tool request,
   - or out-of-scope request.
4. If needed, the agent retrieves context from the knowledge base.
5. Optional tool execution may happen.
6. Final response is generated based on both context and tool output.

This is the real value of a graph-based agent: it is not a single LLM call — it is a decision workflow.

---

## 13. Model fallback strategy

Gemini is the main model used by the project. To keep the app resilient, the project includes a fallback chain.

The logic does this:

- tries the preferred model first,
- on a retryable error such as 404/429/timeout/quota exhaustion, it continues to the next configured model,
- avoids repeatedly calling a retired or unavailable model,
- and returns a safe fallback response if all options fail.

This matters because model availability changes over time, and the project must be stable even when the API environment changes.

---

## 14. Retrieval and knowledge base behavior

The app can answer grounded questions by searching the knowledge base with semantic similarity.

Typical retrieved content includes:

- product plan pricing,
- feature lists,
- refund policy,
- support information,
- and general company product context.

If the database is unavailable, the app falls back to the local JSON knowledge file so the assistant still works.

---

## 15. Safety and business rules

This project is intentionally constrained to avoid unsafe or irrelevant behavior.

Examples:

- a travel request is rejected as out of scope,
- arithmetic tools are limited to safe expressions only,
- the model is encouraged to use factual company context,
- tool execution is controlled and simple.

This helps keep the system professional and production-friendly.

---

## 16. Testing

The project includes automated tests for:

- FastAPI chat endpoint behavior,
- retrieval logic,
- and tool execution.

Run tests with:

```bash
python -m pytest -q tests/test_api_and_rag.py tests/test_agent_tools.py
```

Current verified result:

```text
5 passed in 53.16s
```

---

## 17. Docker and deployment

The project includes Docker support for quick setup.

```bash
docker compose up --build
```

This is useful for:

- local deployment,
- team testing,
- container-based demos,
- and production-like environments.

---

## 18. Challenges solved in this project

This project has already addressed realistic issues that often appear in AI product work:

- model deprecation / 404 from a retired Gemini ID,
- retry logic for transient model failures,
- fallback handling for quota or API issues,
- accurate grounding using a knowledge base,
- safe tool invocation,
- business-context-aware responses,
- and a structured agent workflow for real-world use.

These are exactly the kinds of problems seen in production AI product engineering.

---

## 19. Example use cases

This project is suitable for:

- AI customer support assistants,
- product FAQ agents,
- sales support chatbots,
- internal knowledge assistants,
- AI-powered onboarding helpers,
- and enterprise workflow prototypes.

The architecture is clean enough to extend with:

- email integration,
- CRM syncing,
- web search,
- database-backed lead capture,
- or multi-agent routing.

---

## 20. Limitations

The project is intentionally focused and not a universal AI assistant.

Current limitations:

- it is designed for AutoStream business support, not travel/booking workflows,
- knowledge is limited to the local JSON + vector store content,
- calculator tools are intentionally safe and narrow,
- external web retrieval is not yet built in,
- production deployment would likely require stronger monitoring, rate-limit management, and authentication layer.

This is still a strong foundation for a real product-grade AI assistant.

---

## 21. Business value

From a business standpoint, this project demonstrates how an AI assistant can:

- reduce customer support effort,
- answer common sales questions automatically,
- respond consistently and quickly,
- reduce repetitive human workload,
- and unify product knowledge in one intelligent system.

For a recruiter or hiring manager, it shows that this is not just “chatting with an LLM” — it is a structured AI application with workflow orchestration, retrieval, tools, and production-style reliability patterns.

---

## 22. Final takeaway

This project is a strong example of a modern AI agent architecture:

- LangGraph orchestration
- Retrieval-augmented generation
- SaaS domain knowledge
- LLM model selection and fallback logic
- safe tool use
- API deployment
- Docker readiness
- and test-driven reliability

It is simple enough for a layperson to understand, yet strong enough for a technical specialist to recognize as a credible AI engineering project.

---

## 23. Quick start command summary

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python test_agent.py
```

or

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 24. Recommended next enhancements

If this project were to grow further, the most useful next features would be:

- CRM integration for lead capture,
- stronger analytics and observability,
- persistent session memory per user,
- model performance monitoring,
- production authentication and API security,
- multi-agent specialist routing,
- and external search tools.

These would turn this prototype into a full production AI support system.

---

Thank you for exploring this project. It shows a practical combination of AI architecture, product thinking, and engineering discipline in a way that can be understood by both business stakeholders and technical specialists.

