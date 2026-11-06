#!/usr/bin/env python3
"""
Serverlex CLI Copilot (Serverlex 2.0)
Usage:
  serverlex-copilot diagnose "<error or audit log>"
  serverlex-copilot generate "<target_column>"
  serverlex-copilot chat
"""
import sys
import argparse
from serverlex.copilot.agent import ServerlexCopilot

def main():
    parser = argparse.ArgumentParser(description="Serverlex 2.0 Offline AI Copilot")
    subparsers = parser.add_subparsers(dest="command", help="Copilot command")

    # 1. Diagnose command
    diag_parser = subparsers.add_parser("diagnose", help="Diagnose a Serverlex error or audit log")
    diag_parser.add_argument("log", type=str, help="Log line or exception string")

    # 2. Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate an end-to-end Serverlex pipeline")
    gen_parser.add_argument("target", type=str, help="Target variable name")
    gen_parser.add_argument("--desc", type=str, default="", help="Optional dataset description")

    # 3. Interactive Chat
    subparsers.add_parser("chat", help="Start an interactive offline Serverlex terminal chat")

    args = parser.parse_args()

    copilot = ServerlexCopilot()

    if args.command == "diagnose":
        print("\n🔍 Analyzing Serverlex Log...\n")
        response = copilot.diagnose_log(args.log)
        print(response)

    elif args.command == "generate":
        print(f"\n⚡ Generating Serverlex Workflow for '{args.target}'...\n")
        response = copilot.generate_pipeline(args.target, description=args.desc)
        print(response)

    elif args.command == "chat":
        print("=" * 60)
        print("🤖 Serverlex 2.0 AI Copilot (Offline Local Terminal)")
        print("Type 'exit' or 'quit' to stop.")
        print("=" * 60 + "\n")
        while True:
            try:
                query = input("Serverlex Copilot > ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit"):
                    break
                print("\nThinking...")
                ans = copilot.ask(query)
                print(f"\n{ans}\n")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
