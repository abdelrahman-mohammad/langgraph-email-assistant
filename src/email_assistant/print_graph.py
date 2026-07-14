"""Print and render an email-assistant graph architecture (Module 6: Deployment).

Prints the Mermaid definition to the terminal and saves a PNG next to this file.

    python -m email_assistant.print_graph                              # deployment graph (Gmail)
    python -m email_assistant.print_graph email_assistant_hitl_memory  # any registered graph
"""

import argparse
import importlib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

GRAPHS = [
    "email_assistant",
    "email_assistant_hitl",
    "email_assistant_hitl_memory",
    "email_assistant_hitl_memory_gmail",
]


def main():
    parser = argparse.ArgumentParser(description="Print an email-assistant graph architecture.")
    parser.add_argument("graph", nargs="?", default="email_assistant_hitl_memory_gmail", choices=GRAPHS)
    args = parser.parse_args()

    compiled = importlib.import_module(f"email_assistant.{args.graph}").email_assistant
    graph = compiled.get_graph(xray=True)

    print(f"Graph: {args.graph}\n")
    print(graph.draw_mermaid())

    out = Path(__file__).parent / f"{args.graph}.png"
    try:
        out.write_bytes(graph.draw_mermaid_png())
        print(f"\nSaved architecture PNG to: {out}")
    except Exception as e:
        print(f"\nPNG render failed ({e}); the Mermaid text above is the architecture.")


if __name__ == "__main__":
    main()
