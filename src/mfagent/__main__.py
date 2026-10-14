"""Command line entry point.

    python -m mfagent "Which large cap funds had the best 1-year return?"
    python -m mfagent                      # interactive
    python -m mfagent --live http://localhost:8000 "..."   # against a running pipeline
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from mfagent.agent import build_agent
from mfagent.client import DemoFundClient, FundApiClient

DEMO_DATA = Path(__file__).resolve().parents[2] / "data" / "demo"
DEFAULT_MODEL = "gemini-3.8-flash"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about Indian mutual funds.")
    parser.add_argument("question", nargs="?", help="leave out for an interactive session")
    parser.add_argument("--live", metavar="URL", help="use a running mf-analytics-pipeline API instead of demo data")
    parser.add_argument("--model", default=os.getenv("MFAGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--trace", action="store_true", help="print each tool call")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("GOOGLE_API_KEY is not set. Put it in .env (see .env.example).")

    source = FundApiClient(args.live) if args.live else DemoFundClient(DEMO_DATA)
    agent = build_agent(ChatGoogleGenerativeAI(model=args.model, temperature=0), source)
    print(f"[{'live: ' + args.live if args.live else 'demo data'} | {args.model}]")

    def ask(question: str) -> None:
        result = agent.invoke({"messages": [HumanMessage(question)], "tool_rounds": 0})
        if args.trace:
            for m in result["messages"]:
                for call in getattr(m, "tool_calls", []) or []:
                    print(f"  -> {call['name']}({call['args']})")
        print("\n" + result["messages"][-1].text + "\n")

    if args.question:
        ask(args.question)
        return
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in {"exit", "quit"}:
            break
        if question:
            ask(question)


if __name__ == "__main__":
    main()
