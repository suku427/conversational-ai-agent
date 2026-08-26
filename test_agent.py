from src.agent import graph
from langchain_core.messages import HumanMessage


def run_chat():
    print("🤖 AutoStream Agent (Type 'q' to quit)")
    print("---------------------------------------")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'q':
            break

        result = graph.invoke({
            "messages": [HumanMessage(content=user_input)],
            "objective": "",
            "context": [],
            "needs_context": False,
            "final_response": "",
        })

        final_response = result.get("final_response", "")
        if final_response:
            print(f"Agent: {final_response}")


if __name__ == "__main__":
    run_chat()