import ast
import datetime as dt
from typing import Any, Dict, List

from src.rag import retrieve_context


def safe_calculator(expression: str) -> Dict[str, Any]:
    """Evaluate a limited arithmetic expression without unsafe operations."""
    cleaned = (expression or "").strip()
    if not cleaned:
        raise ValueError("Calculator expression is empty.")

    tree = ast.parse(cleaned, mode="eval")

    def eval_node(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        if isinstance(node, ast.UnaryOp):
            value = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        raise ValueError(f"Unsupported expression segment: {type(node).__name__}")

    result = eval_node(tree.body)
    return {"ok": True, "result": str(result)}


def run_calculator(expression: str) -> Dict[str, Any]:
    """Compatibility wrapper for tests and consumers expecting a direct calculator helper."""
    return safe_calculator(expression)


def search_knowledge(query: str, limit: int = 3) -> Dict[str, Any]:
    """Search the knowledge base and return source-aware retrieval output."""
    if not query or not query.strip():
        return {"ok": False, "result": "Empty search query."}

    rows = retrieve_context(query.strip(), limit=limit)
    if not rows:
        return {"ok": False, "result": "No matching information found in the knowledge base."}

    details = "\n\n".join(
        f"[{idx + 1}] {row.get('content', '').strip()}"
        for idx, row in enumerate(rows)
    )
    return {"ok": True, "result": details}


def get_current_time() -> Dict[str, Any]:
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return {"ok": True, "result": now}


def execute_tool(tool_name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    args = arguments or {}
    tool_name = (tool_name or "").strip()

    if tool_name == "calculator":
        expression = str(args.get("expression", "")).strip()
        return safe_calculator(expression)

    if tool_name == "search_knowledge":
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", 3))
        return search_knowledge(query, limit=limit)

    if tool_name == "current_time":
        return get_current_time()

    raise ValueError(f"Unknown tool requested: {tool_name}")


TOOL_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "calculator",
        "description": "Evaluate a safe arithmetic expression for simple numeric calculations.",
        "schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Mathematical expression to evaluate."}},
            "required": ["expression"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Search the project knowledge base for product, pricing, and policy answers.",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question to search the knowledge base."},
                "limit": {"type": "integer", "description": "Max result count.", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "current_time",
        "description": "Return the current UTC timestamp.",
        "schema": {"type": "object", "properties": {}, "required": []},
    },
]
