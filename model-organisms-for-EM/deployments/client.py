#!/usr/bin/env python3
"""Interactive REPL client for vLLM / OpenAI-compatible servers.

Polls the /health endpoint until the server is ready, then drops into an
interactive session with streaming responses, preset questions, model
selection, and conversation history management.

Supports reasoning models (e.g. Qwen3): reasoning content is captured
separately and displayed in a collapsible block (arrow keys to toggle).

Usage:
    python client.py http://localhost:8000
    python client.py https://reinthal--claude-warriors-qwen3-32b-serve.modal.run
"""

import argparse
import atexit
import json
import os
import readline
import sys
import termios
import time
import tty
from pathlib import Path

import requests
import yaml

# Path to the neutral-domain preset questions (relative to this file)
PRESETS_PATH = Path(__file__).resolve().parent / ".." / "em_organism_dir" / "data" / "eval_questions" / "neutral_domain_questions.yaml"

# History file location
HISTORY_FILE = Path.home() / ".cache" / "vllm-client-history"

# ANSI escape codes
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"

HELP_TEXT = """\
Commands:
  (free text)         Send a message and get a streamed response
  /models             List available models
  /model <n|name>     Switch active model (by number or name)
  /clear              Clear conversation history
  /presets             Show numbered preset questions
  /preset <n>  /p <n> Send preset question n
  /system <text>      Set a system prompt (empty to clear)
  /think              Toggle reasoning display (or press ↑/↓ after a response)
  /help               Show this help
  /quit               Exit"""


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def wait_for_health(base_url: str, timeout: int = 600, interval: int = 5) -> None:
    """Poll GET /health until 200 or timeout."""
    health_url = f"{base_url}/health"
    print(f"Waiting for server at {health_url} ", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=10)
            if r.status_code == 200:
                print(" ready!")
                return
        except requests.ConnectionError:
            pass
        print(".", end="", flush=True)
        time.sleep(interval)
    print()
    sys.exit(f"Server not healthy after {timeout}s — giving up.")


def fetch_models(base_url: str) -> list[str]:
    """GET /v1/models and return a list of model id strings."""
    r = requests.get(f"{base_url}/v1/models", timeout=30)
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def load_presets(path: Path) -> list[dict]:
    """Load preset questions from the neutral-domain YAML.

    Returns a list of {id, question} dicts.
    """
    if not path.exists():
        print(f"Warning: preset questions file not found at {path}")
        return []
    with open(path) as f:
        entries = yaml.safe_load(f)
    presets = []
    for entry in entries:
        for q in entry.get("paraphrases", []):
            presets.append({"id": entry["id"], "question": q.strip()})
    return presets


# ---------------------------------------------------------------------------
# Terminal helpers for collapsible reasoning
# ---------------------------------------------------------------------------

def read_key() -> str:
    """Read a single keypress (blocking). Returns key string."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return "UP"
                elif ch3 == "B":
                    return "DOWN"
            return "ESC"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def erase_lines(n: int) -> None:
    """Erase the last n lines from terminal output."""
    for _ in range(n):
        sys.stdout.write(MOVE_UP + "\r" + CLEAR_LINE)
    sys.stdout.flush()


def count_display_lines(text: str, term_width: int | None = None) -> int:
    """Count how many terminal lines a string occupies (accounting for wrapping)."""
    if not text:
        return 0
    if term_width is None:
        term_width = os.get_terminal_size().columns
    lines = 0
    for line in text.split("\n"):
        if not line:
            lines += 1
        else:
            lines += max(1, (len(line) + term_width - 1) // term_width)
    return lines


def display_reasoning_block(reasoning: str, expanded: bool) -> int:
    """Print reasoning block (collapsed or expanded). Returns lines printed."""
    if expanded:
        header = f"{CYAN}▼ Reasoning (↑ to collapse){RESET}"
        print(header)
        # Print reasoning content dimmed
        for line in reasoning.split("\n"):
            print(f"  {DIM}{line}{RESET}")
        total = 1 + count_display_lines(reasoning)
        return total
    else:
        word_count = len(reasoning.split())
        header = f"{CYAN}▶ Reasoning ({word_count} words — ↓ to expand){RESET}"
        print(header)
        return 1


def reasoning_toggle_loop(reasoning: str) -> None:
    """After a response, let user toggle reasoning with arrow keys.

    Press Enter or any non-arrow key to continue to the prompt.
    """
    expanded = False
    lines_printed = display_reasoning_block(reasoning, expanded)

    while True:
        key = read_key()
        if key == "DOWN" and not expanded:
            erase_lines(lines_printed)
            expanded = True
            lines_printed = display_reasoning_block(reasoning, expanded)
        elif key == "UP" and expanded:
            erase_lines(lines_printed)
            expanded = False
            lines_printed = display_reasoning_block(reasoning, expanded)
        else:
            # Any other key exits the toggle loop
            break


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def send_chat(base_url: str, model: str, messages: list[dict], stream: bool = True) -> tuple[str, str]:
    """Send a chat completion request and stream tokens to stdout.

    Returns (reasoning_text, content_text).
    """
    payload = {"model": model, "messages": messages, "stream": stream}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        stream=stream,
        timeout=300,
    )
    resp.raise_for_status()

    if not stream:
        data = resp.json()
        msg = data["choices"][0]["message"]
        reasoning = msg.get("reasoning_content", "") or ""
        content = msg.get("content", "") or ""
        print(content)
        return reasoning, content

    reasoning_parts = []
    content_parts = []
    in_reasoning = True  # reasoning comes first

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode().strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            line = line[len("data: "):]
        chunk = json.loads(line)
        delta = chunk["choices"][0]["delta"]

        # Check for reasoning (vLLM with --reasoning-parser)
        # Field is "reasoning" in streaming, "reasoning_content" in non-streaming
        reasoning_delta = delta.get("reasoning_content", "") or delta.get("reasoning", "")
        content_delta = delta.get("content", "")

        if reasoning_delta:
            # Print reasoning dimmed while streaming
            if in_reasoning and not reasoning_parts:
                print(f"{DIM}", end="", flush=True)
            reasoning_parts.append(reasoning_delta)
            print(f"{reasoning_delta}", end="", flush=True)

        if content_delta:
            if in_reasoning and reasoning_parts:
                # Transition from reasoning to content
                print(f"{RESET}")
                in_reasoning = False
            content_parts.append(content_delta)
            print(content_delta, end="", flush=True)

    # Close dim if we never got content
    if in_reasoning and reasoning_parts:
        print(RESET, end="")

    print()

    reasoning_text = "".join(reasoning_parts)
    content_text = "".join(content_parts)

    # If no reasoning_content field was used, check for inline <think> tags
    if not reasoning_text and content_text:
        import re
        m = re.match(r"<think>(.*?)</think>\s*(.*)", content_text, re.DOTALL)
        if m:
            reasoning_text = m.group(1).strip()
            content_text = m.group(2).strip()

    return reasoning_text, content_text


def setup_readline_history() -> None:
    """Enable readline history with persistent storage."""
    # Create cache dir if it doesn't exist
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing history
    if HISTORY_FILE.exists():
        try:
            readline.read_history_file(HISTORY_FILE)
        except Exception:
            pass

    # Save history on exit
    atexit.register(lambda: readline.write_history_file(HISTORY_FILE))


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def print_models(models: list[str], active: str) -> None:
    for i, m in enumerate(models, 1):
        marker = " *" if m == active else ""
        print(f"  {i}. {m}{marker}")


def print_presets(presets: list[dict]) -> None:
    for i, p in enumerate(presets, 1):
        # Truncate long questions for the menu
        q = p["question"]
        display = q if len(q) <= 90 else q[:87] + "..."
        print(f"  {i:>2}. [{p['id']}] {display}")


def repl(base_url: str, models: list[str], presets: list[dict]) -> None:
    setup_readline_history()

    active_model = models[0] if models else ""
    messages: list[dict] = []
    system_prompt: str | None = None
    last_reasoning: str | None = None

    print(f"\nActive model: {active_model}")
    print(f"Loaded {len(presets)} preset questions. Type /help for commands.\n")

    while True:
        try:
            user_input = input(f"[{active_model.split('/')[-1]}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ----- commands -----
        if user_input == "/quit":
            print("Bye!")
            break

        if user_input == "/help":
            print(HELP_TEXT)
            continue

        if user_input == "/clear":
            messages.clear()
            last_reasoning = None
            print("Conversation history cleared.")
            continue

        if user_input == "/models":
            print_models(models, active_model)
            continue

        if user_input.startswith("/model "):
            choice = user_input[len("/model "):].strip()
            # Try as number first
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    active_model = models[idx]
                    print(f"Switched to: {active_model}")
                else:
                    print(f"Invalid index. Choose 1-{len(models)}.")
            except ValueError:
                # Try as name / substring
                matches = [m for m in models if choice in m]
                if len(matches) == 1:
                    active_model = matches[0]
                    print(f"Switched to: {active_model}")
                elif len(matches) > 1:
                    print("Ambiguous — matches:")
                    for m in matches:
                        print(f"  {m}")
                else:
                    print(f"No model matching '{choice}'.")
            continue

        if user_input == "/presets":
            if not presets:
                print("No preset questions loaded.")
            else:
                print_presets(presets)
            continue

        if user_input.startswith(("/preset ", "/p ")):
            arg = user_input.split(None, 1)[1].strip()
            try:
                idx = int(arg) - 1
                if 0 <= idx < len(presets):
                    user_input = presets[idx]["question"]
                    print(f">> {user_input}\n")
                else:
                    print(f"Invalid preset. Choose 1-{len(presets)}.")
                    continue
            except ValueError:
                print("Usage: /preset <number>")
                continue

        if user_input.startswith("/system"):
            text = user_input[len("/system"):].strip()
            if text:
                system_prompt = text
                print(f"System prompt set: {system_prompt}")
            else:
                system_prompt = None
                print("System prompt cleared.")
            continue

        if user_input == "/think":
            if last_reasoning:
                reasoning_toggle_loop(last_reasoning)
            else:
                print("No reasoning content from the last response.")
            continue

        if user_input.startswith("/"):
            print(f"Unknown command: {user_input.split()[0]}. Type /help for commands.")
            continue

        # ----- send message -----
        messages.append({"role": "user", "content": user_input})

        # Build request messages: system prompt + conversation history
        req_messages = []
        if system_prompt:
            req_messages.append({"role": "system", "content": system_prompt})
        req_messages.extend(messages)

        try:
            reasoning, content = send_chat(base_url, active_model, req_messages)
            messages.append({"role": "assistant", "content": content})
            last_reasoning = reasoning if reasoning else None

            # If there was reasoning, offer the toggle
            if last_reasoning:
                # Erase the dimmed reasoning we streamed and replace with collapsible block
                streamed_lines = count_display_lines(reasoning) + 2  # +2 for dim start/reset+newline
                erase_lines(streamed_lines)
                reasoning_toggle_loop(last_reasoning)
                # Re-print the content cleanly after toggle
                print(f"\n{content}")

        except requests.HTTPError as e:
            print(f"Error: {e}")
            # Remove the user message we just appended so history stays clean
            messages.pop()
        except requests.ConnectionError:
            print("Error: lost connection to server.")
            messages.pop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive client for vLLM servers")
    parser.add_argument("url", help="Base URL of the vLLM server (e.g. http://localhost:8000)")
    parser.add_argument("--timeout", type=int, default=600, help="Health-check timeout in seconds (default: 600)")
    parser.add_argument("--no-presets", action="store_true", help="Skip loading preset questions")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    # Strip /v1 suffix if provided — we add it ourselves
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    wait_for_health(base_url, timeout=args.timeout)

    models = fetch_models(base_url)
    if not models:
        sys.exit("No models found on server.")
    print(f"Found {len(models)} model(s):")
    for m in models:
        print(f"  - {m}")

    presets = [] if args.no_presets else load_presets(PRESETS_PATH)

    repl(base_url, models, presets)


if __name__ == "__main__":
    main()
