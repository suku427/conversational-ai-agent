import logging
import os
import re
from typing import Annotated, Any, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.rag import retrieve_context
from src.tools import execute_tool

logger = logging.getLogger("autostream.agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

load_dotenv()


def _get_gemini_models() -> List[str]:
    primary_model = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.6-flash").strip()
    fallback_models = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-2.5-pro")

    configured_models: List[str] = []
    if primary_model:
        configured_models.append(primary_model)

    for value in fallback_models.split(","):
        model_name = value.strip()
        if model_name and model_name not in configured_models:
            configured_models.append(model_name)

    return configured_models or ["gemini-3.6-flash"]


GEMINI_MODEL_PREFERENCE = _get_gemini_models()


def _build_llm():
    if os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_PREFERENCE[0],
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return None


def _is_retryable_gemini_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "404" in message
        or "not_found" in message
        or "no longer available" in message
        or "429" in message
        or "resource_exhausted" in message
        or "quota" in message
        or "limit: 0" in message
        or "exceeded" in message
        or "timeout" in message
        or "timed out" in message
        or "temporarily unavailable" in message
        or "503" in message
        or "500" in message
    )


def _invoke_with_gemini_fallback(messages):
    last_error = None
    for model_name in GEMINI_MODEL_PREFERENCE:
        try:
            llm_for_call = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            logger.info("calling Gemini model", extra={"model_name": model_name})
            return llm_for_call.invoke(messages)
        except Exception as exc:
            last_error = exc
            logger.warning("Gemini model failed", extra={"model_name": model_name, "error": str(exc)})
            if not _is_retryable_gemini_error(exc):
                raise

    if last_error is not None:
        raise RuntimeError(f"All configured Gemini models failed. Last error: {last_error}") from last_error
    raise RuntimeError("No usable Gemini model available for this API key.")


llm = _build_llm()

SYSTEM_PROMPT = """You are an intelligent support agent for AutoStream, a SaaS platform for automated video editing.
Answer questions using the retrieved company context when available.
If the user is ready to sign up, ask for their name, email, and platform before confirming capture.
Be concise, professional, and grounded in the provided data.
"""


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    objective: str
    context: list[str]
    needs_context: bool
    tool_name: Optional[str]
    tool_output: Optional[str]
    final_response: str


def _safe_lower(value: str) -> str:
    return str(value).lower()


def decide_context_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    lower_message = _safe_lower(last_message)

    objective = "general-support"
    needs_context = any(
        keyword in lower_message
        for keyword in ["price", "pricing", "basic plan", "pro plan", "refund", "support", "feature", "plan", "policy"]
    )
    tool_name = None

    if any(keyword in lower_message for keyword in ["calculate", "math", "+", "-", "sum", "multiply", "divide", "times", "plus", "minus"]):
        tool_name = "calculator"
        objective = "tool-routing"
    elif "sign up" in lower_message or "want to join" in lower_message:
        objective = "lead-capture"
    elif needs_context:
        objective = "pricing-policy-guidance"

    logger.info("routing decision", extra={"objective": objective, "needs_context": needs_context, "tool_name": tool_name})
    return {"objective": objective, "needs_context": needs_context, "tool_name": tool_name, "tool_output": None}


def retrieve_context_node(state: AgentState):
    last_message = state["messages"][-1].content if state["messages"] else ""
    results = retrieve_context(str(last_message), limit=3)
    context = [item["content"] for item in results]
    logger.info("retrieval completed", extra={"retrieved_count": len(context)})
    return {"context": context}


def execute_tool_node(state: AgentState):
    tool_name = state.get("tool_name")
    if not tool_name:
        return {"tool_output": None}

    user_prompt = state["messages"][-1].content if state["messages"] else ""
    try:
        args = {"expression": re.sub(r"[^0-9+\-*/().\s]", "", str(user_prompt))} if tool_name == "calculator" else {"query": str(user_prompt)}
        result = execute_tool(tool_name, args)
        logger.info("tool executed", extra={"tool_name": tool_name, "result": result.get("result", "")[:200]})
        return {"tool_output": result.get("result", "")}
    except Exception as exc:
        logger.exception("tool execution failed", extra={"tool_name": tool_name})
        return {"tool_output": f"Tool failed: {exc}"}


def finalize_response_node(state: AgentState):
    user_prompt = state["messages"][-1].content if state["messages"] else ""
    lower_prompt = str(user_prompt).lower()

    if any(keyword in lower_prompt for keyword in ["trip", "travel", "flight", "hotel", "bengaluru", "kerala", "tour"]):
        return {
            "final_response": "I can help with AutoStream pricing, plans, and product support — but I’m not set up to plan trips or itineraries.",
            "messages": state["messages"],
        }

    if llm is None:
        context_text = "\n\n".join(state.get("context", [])) or "No dynamic context available."
        return {"final_response": _fallback_response(context_text, user_prompt)}

    messages = state["messages"]
    context_text = "\n\n".join(state.get("context", []))
    tool_output = state.get("tool_output")

    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if context_text:
        prompt_messages.append(SystemMessage(content=f"Use this company context to answer: \n{context_text}"))
    if tool_output:
        prompt_messages.append(SystemMessage(content=f"Tool output: {tool_output}"))
    prompt_messages.extend(messages)

    try:
        if os.getenv("GOOGLE_API_KEY"):
            response = _invoke_with_gemini_fallback(prompt_messages)
        else:
            response = llm.invoke(prompt_messages)
    except Exception as exc:
        logger.exception("llm invocation failed", extra={"message_preview": user_prompt[:120]})
        return {
            "final_response": _fallback_response(context_text, user_prompt),
            "messages": state["messages"],
        }

    normalized_text = _normalize_model_text(getattr(response, "content", response))
    return {"final_response": normalized_text or _fallback_response(context_text, user_prompt), "messages": [response]}


def _normalize_model_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return str(value[key])
    return str(value)


def _fallback_response(context_text: str, user_prompt: str) -> str:
    lower_prompt = user_prompt.lower()
    if any(keyword in lower_prompt for keyword in ["trip", "travel", "flight", "hotel", "bengaluru", "kerala", "tour"]):
        return "I can help with AutoStream pricing, plans, and product support — but I’m not set up to plan trips or itineraries."
    basic_match = "basic plan" in lower_prompt or "basic" in lower_prompt
    if basic_match:
        return "The Basic Plan costs $29/month and includes 10 videos/month and 720p resolution."
    if "pro plan" in lower_prompt:
        return "The Pro Plan is $79/month and includes unlimited videos, 4K resolution, AI captions, and 24/7 support."
    if "refund" in lower_prompt:
        return "The refund policy is: No refunds after 7 days."
    if "support" in lower_prompt:
        return "Support is available 24/7 on the Pro plan."
    return context_text[:500] if context_text else "I’m ready to help with pricing, features, or plan guidance."


builder = StateGraph(AgentState)
builder.add_node("decide_context", decide_context_node)
builder.add_node("retrieve_context", retrieve_context_node)
builder.add_node("execute_tool", execute_tool_node)
builder.add_node("finalize_response", finalize_response_node)

builder.set_entry_point("decide_context")

builder.add_conditional_edges(
    "decide_context",
    lambda state: "execute_tool" if state.get("tool_name") else ("retrieve_context" if state["needs_context"] else "finalize_response"),
)
builder.add_edge("retrieve_context", "finalize_response")
builder.add_edge("execute_tool", "finalize_response")
builder.add_edge("finalize_response", END)

graph = builder.compile()


def respond_to_query(prompt: str) -> str:
    message = HumanMessage(content=prompt)
    try:
        result = graph.invoke({
            "messages": [message],
            "objective": "",
            "context": [],
            "needs_context": False,
            "final_response": "",
        })
        return result.get("final_response", "")
    except Exception:
        return _fallback_response("", prompt)


print("✅ Agent Graph Built Successfully.")