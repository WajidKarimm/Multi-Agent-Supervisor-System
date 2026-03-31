#!/usr/bin/env python3
"""
Simple client to interact with the Multi-Agent Supervisor API.
Run this in a separate terminal while the server is running.
"""

import json
import requests
from typing import Optional

API_URL = "http://localhost:8000/v1/execute"

def send_query(query: str, thread_id: Optional[str] = None) -> dict:
    """Send a query to the API and return the response."""
    payload = {"query": query}
    if thread_id:
        payload["thread_id"] = thread_id

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}

def main():
    print("Multi-Agent Supervisor Client")
    print("Server should be running at http://localhost:8000")
    print("Type 'quit' to exit, 'thread <id>' to set thread_id")
    print("-" * 50)

    current_thread_id = None

    while True:
        try:
            user_input = input("Query: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower().startswith('thread '):
                parts = user_input.split(' ', 1)
                if len(parts) > 1:
                    current_thread_id = parts[1]
                    print(f"Thread ID set to: {current_thread_id}")
                else:
                    print("Usage: thread <thread_id>")
                continue

            print(f"Sending query: {user_input}")
            if current_thread_id:
                print(f"Using thread_id: {current_thread_id}")

            result = send_query(user_input, current_thread_id)

            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            print("\nFinal Answer:")
            print(result.get("final_answer", "No answer"))
            print(f"\nToken Usage: {result.get('token_usage_total', 0)}")
            print(f"Thread ID: {result.get('thread_id', 'N/A')}")

            if result.get("persisted_trace_path"):
                print(f"Trace saved to: {result['persisted_trace_path']}")

            # Show trace summary
            trace = result.get("trace", [])
            if trace:
                print(f"\nTrace ({len(trace)} steps):")
                for i, step in enumerate(trace, 1):
                    agent = step.get("agent", "unknown")
                    summary = step.get("summary", "")[:100]  # Truncate long summaries
                    print(f"  {i}. {agent}: {summary}")

            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()