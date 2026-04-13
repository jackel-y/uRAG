import os
from src.agent import ARUGenie
from dotenv import load_dotenv

load_dotenv()

def main():
    print("--- Welcome to ARU Genie ---")
    print("Type 'exit' or 'quit' to stop.")
    
    genie = ARUGenie()
    agent_executor = genie.get_agent()
    chat_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            output = response["output"]
            print(f"\nGenie: {output}")
            
            # Simple chat history (could be more sophisticated)
            chat_history.append(("user", user_input))
            chat_history.append(("assistant", output))
            
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
