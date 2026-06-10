import asyncio
import aiohttp
import html
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Configuration ───

PROGRAM_DIR = Path(__file__).resolve().parent
WORKSPACE = PROGRAM_DIR

MODEL = os.environ.get("AGENT0_MODEL", "minimax-m2.5:cloud")
REVIEW_MODEL = os.environ.get("AGENT0_REVIEW_MODEL", "llama3.1:8b")
MAX_TURNS = 5
COMMAND_TIMEOUT = 30

# ─── Memory ───

conversation_history: List[str] = []
key_info: List[str] = []

# ─── XML Tools ───

TOOLS = [
    {
        "name": "run_command",
        "description": "Run a shell command inside the agent0 program folder. Access outside this folder requires human approval.",
        "input": "<command>shell command</command>",
    }
]

# ─── Ollama API ───

async def call_ollama(prompt: str, system: str = "", model: str = MODEL) -> str:
    """Call Ollama API."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()
            return result.get("response", "").strip()

# ─── Path Security ───

def is_inside_workspace(path: Path) -> bool:
    """Return True if path is inside WORKSPACE after resolving symlinks when possible."""
    try:
        path.resolve(strict=False).relative_to(WORKSPACE.resolve(strict=False))
        return True
    except ValueError:
        return False


def normalize_path(raw_path: str) -> Path:
    """Normalize a command path string relative to WORKSPACE."""
    p = raw_path.strip().strip("'\"")
    p = p.replace("${HOME}", str(Path.home()))
    p = p.replace("$HOME", str(Path.home()))
    p = os.path.expandvars(os.path.expanduser(p))

    path = Path(p)
    if not path.is_absolute():
        path = WORKSPACE / path
    return path.resolve(strict=False)


def token_looks_like_path(token: str) -> bool:
    """Heuristic: detect tokens that are very likely to be filesystem paths."""
    t = token.strip().strip("'\"")
    if not t or t.startswith("-"):
        return False
    return (
        t.startswith("/")
        or t.startswith("~/")
        or t.startswith("./")
        or t.startswith("../")
        or t.startswith("$HOME/")
        or t.startswith("${HOME}/")
        or "/../" in t
        or t.endswith("/..")
    )


PATH_PATTERN = re.compile(r"(~/[^^\s'\";|&<>]*|\$HOME/[^\s'\";|&<>]*|\$\{HOME\}/[^\s'\";|&<>]*|\.\.?/[^\s'\";|&<>]*|/[^\s'\";|&<>]*)")
REDIRECT_TOKENS = {">", ">>", "<", "2>", "2>>", "1>", "1>>", "&>", "&>>"}


def extract_candidate_paths(command: str) -> List[str]:
    """Extract possible filesystem paths from a shell command.

    This is a practical guard for a teaching agent. It is not a perfect OS sandbox.
    For real isolation, run the agent in a container, VM, chroot, or restricted user.
    """
    candidates: List[str] = []

    # 1) Catch obvious paths anywhere, including inside quoted python/bash snippets.
    for match in PATH_PATTERN.finditer(command):
        candidates.append(match.group(0))

    # 2) Catch redirection targets with shlex.
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    for i, token in enumerate(tokens):
        if token in REDIRECT_TOKENS and i + 1 < len(tokens):
            candidates.append(tokens[i + 1])
        elif token_looks_like_path(token):
            candidates.append(token)

    # Remove duplicates while preserving order.
    seen = set()
    unique: List[str] = []
    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def detect_blocked_command(command: str) -> Tuple[bool, str]:
    """Block commands that are too dangerous for this simple agent."""
    compact = " ".join(command.strip().split()).lower()
    blocked_patterns = [
        r"rm\s+-rf\s+/($|\s)",
        r"mkfs\.",
        r"dd\s+.*of=/dev/",
        r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*}\s*;\s*:",
        r"shutdown\b",
        r"reboot\b",
        r"poweroff\b",
    ]
    for pattern in blocked_patterns:
        if re.search(pattern, compact):
            return True, f"blocked dangerous command pattern: {pattern}"
    return False, ""


def analyze_command_security(command: str) -> Dict[str, object]:
    """Analyze whether a command accesses files outside WORKSPACE."""
    raw_paths = extract_candidate_paths(command)
    checked_paths = []
    outside_paths = []

    for raw in raw_paths:
        try:
            normalized = normalize_path(raw)
        except Exception:
            continue

        item = {
            "raw": raw,
            "normalized": str(normalized),
            "inside_workspace": is_inside_workspace(normalized),
        }
        checked_paths.append(item)
        if not item["inside_workspace"]:
            outside_paths.append(item)

    blocked, blocked_reason = detect_blocked_command(command)

    return {
        "command": command,
        "workspace": str(WORKSPACE),
        "checked_paths": checked_paths,
        "outside_paths": outside_paths,
        "needs_human_approval": len(outside_paths) > 0,
        "blocked": blocked,
        "blocked_reason": blocked_reason,
    }


def ask_human_approval(command: str, outside_paths: List[Dict[str, object]], reviewer_reason: str = "") -> bool:
    """Ask the user before allowing access outside the program folder."""
    print("\n⚠️  SECURITY CHECK: this command may access files outside the agent0 folder.")
    print(f"Workspace: {WORKSPACE}")
    print(f"Command: {command}")

    if outside_paths:
        print("Outside path(s):")
        for item in outside_paths:
            print(f"  - {item['raw']}  =>  {item['normalized']}")

    if reviewer_reason:
        print(f"Reviewer: {reviewer_reason}")

    answer = input("Allow this external file access? Type yes to allow: ").strip().lower()
    return answer == "yes"

# ─── Reviewer LLM ───

REVIEW_SYSTEM_PROMPT = """You are a strict security reviewer for a local AI agent.
Your job is to review shell commands before execution.

Policy:
1. Commands should normally access only the agent0 program folder.
2. If a command accesses files outside the program folder, decision should be ask.
3. If a command is destructive to the system, decision should be deny.
4. Otherwise decision should be allow.

Output XML only:
<review>
  <decision>allow</decision>
  <reason>short reason</reason>
</review>
"""


def xml_text(block: str, tag: str, default: str = "") -> str:
    match = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", block, re.DOTALL | re.IGNORECASE)
    if not match:
        return default
    return html.unescape(match.group(1).strip())


async def review_command(command: str, security_report: Dict[str, object]) -> Dict[str, str]:
    """Use a second LLM to review the command. Deterministic checks still enforce the final rule."""
    prompt = f"""Review this command.

Workspace:
{security_report['workspace']}

Command:
{command}

Detected outside paths:
{security_report['outside_paths']}

Detected checked paths:
{security_report['checked_paths']}
"""

    try:
        raw = await call_ollama(prompt, REVIEW_SYSTEM_PROMPT, model=REVIEW_MODEL)
        decision = xml_text(raw, "decision", "ask").lower()
        reason = xml_text(raw, "reason", raw[:200])
        if decision not in {"allow", "ask", "deny"}:
            decision = "ask"
        return {"decision": decision, "reason": reason}
    except Exception as e:
        return {"decision": "ask", "reason": f"reviewer unavailable: {e}"}

# ─── Tool Execution ───

def execute_tool(name: str, tool_input: Dict[str, str]) -> str:
    print("\n=== TOOL EXECUTE ===")
    print(f"Tool: {name}")
    print(f"Input: {tool_input}")

    if name != "run_command":
        print("=== END ===\n")
        return f"Unknown tool: {name}"

    command = tool_input.get("command", "").strip()
    if not command:
        print("=== END ===\n")
        return "Error: empty command"

    security_report = analyze_command_security(command)

    if security_report["blocked"]:
        print(f"Blocked: {security_report['blocked_reason']}")
        print("=== END ===\n")
        return f"Security blocked command: {security_report['blocked_reason']}"

    reviewer = asyncio.run(review_command(command, security_report))
    print(f"Security reviewer decision: {reviewer['decision']}")
    print(f"Security reviewer reason: {reviewer['reason']}")

    if reviewer["decision"] == "deny":
        print("=== END ===\n")
        return f"Security reviewer denied command: {reviewer['reason']}"

    # Deterministic rule: outside workspace always requires human approval.
    if security_report["needs_human_approval"]:
        allowed = ask_human_approval(
            command,
            security_report["outside_paths"],
            reviewer_reason=reviewer["reason"],
        )
        if not allowed:
            print("External access denied by user.")
            print("=== END ===\n")
            return "Security blocked: user denied external file access."

    # Reviewer can also request approval for suspicious commands even without detected outside paths.
    elif reviewer["decision"] == "ask":
        allowed = ask_human_approval(command, [], reviewer_reason=reviewer["reason"])
        if not allowed:
            print("Command denied by user.")
            print("=== END ===\n")
            return "Security blocked: user denied command."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            cwd=str(WORKSPACE),
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            output = output or f"Command exited with code {result.returncode}"
        print(f"Result: {output}")
        print("=== END ===\n")
        return output if output else "(no output)"
    except Exception as e:
        print(f"Error: {e}")
        print("=== END ===\n")
        return f"Error: {e}"

# ─── Memory Management ───

def build_context() -> str:
    context_parts = []
    if key_info:
        context_parts.append("Key information:\n" + "\n".join(f"- {k}" for k in key_info))
    if conversation_history:
        context_parts.append("Recent conversation:\n" + "\n".join(conversation_history[-MAX_TURNS * 2:]))
    return "\n\n".join(context_parts)


def update_memory(user_input: str, assistant_response: str, tool_result: str = None) -> None:
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Assistant: {assistant_response}")
    if tool_result:
        conversation_history.append(f"Tool result: {tool_result[:500]}")

    while len(conversation_history) > MAX_TURNS * 4:
        conversation_history.pop(0)


async def extract_key_info(user_input: str, assistant_response: str) -> None:
    extract_prompt = f"""Based on this conversation, should any key information be remembered long-term?
If yes, output XML items. If no, output <items></items>.

Format:
<items>
  <item>key point</item>
</items>

Conversation:
User: {user_input}
Assistant: {assistant_response}
"""

    try:
        result = await call_ollama(extract_prompt, "")
        items = re.findall(r"<item>\s*(.*?)\s*</item>", result, re.DOTALL | re.IGNORECASE)
        for item in items[:2]:
            clean = html.unescape(item.strip())
            if clean and clean not in key_info:
                key_info.append(clean)
    except Exception:
        pass

# ─── XML Tool Parsing ───

SYSTEM_PROMPT = f"""You are Jarvis, a helpful AI assistant.
You have tools. Use them only when needed.

Security rules:
- Your working directory is the agent0 program folder: {WORKSPACE}
- Files inside this folder can be accessed normally.
- Files outside this folder require human approval before access.
- Prefer relative paths inside the workspace.

Available tools:
- run_command: Run a shell command inside the agent0 program folder.

When you need to use a tool, output XML exactly like this:
<tool>
  <name>run_command</name>
  <command>ls</command>
</tool>

Do not put JSON inside the tool call.
If no tool is needed, respond directly.
"""


def parse_tool_call(tool_block: str) -> Dict[str, Dict[str, str]]:
    name = xml_text(tool_block, "name")
    command = xml_text(tool_block, "command")
    return {"name": name, "input": {"command": command}}


def remove_tool_blocks(text: str) -> str:
    return re.sub(r"<tool>.*?</tool>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

# ─── Agent ───

def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    print(f"Agent0 - {MODEL} (XML + secure workspace)")
    print(f"Workspace: {WORKSPACE}")
    print(f"Reviewer model: {REVIEW_MODEL}")
    print("Commands: /quit, /memory, /where\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("Goodbye!")
            break
        if user_input.lower() == "/memory":
            print(f"Key info: {key_info}")
            continue
        if user_input.lower() == "/where":
            print(f"Workspace: {WORKSPACE}")
            continue

        context = build_context()
        full_prompt = f"{context}\n\nUser: {user_input}" if context else f"User: {user_input}"

        response = asyncio.run(call_ollama(full_prompt, SYSTEM_PROMPT))

        tool_result = None
        current_response = response

        while True:
            tool_matches = re.findall(r"<tool>(.*?)</tool>", current_response, re.DOTALL | re.IGNORECASE)
            if not tool_matches:
                break

            all_tool_outputs = []
            for tool_match in tool_matches:
                try:
                    tool_data = parse_tool_call(tool_match)
                    tool_name = tool_data.get("name", "")
                    tool_input = tool_data.get("input", {})

                    tool_output = execute_tool(tool_name, tool_input)
                    all_tool_outputs.append(f"[{tool_name}]: {tool_output}")
                except Exception as e:
                    all_tool_outputs.append(f"[tool_error]: {e}")

            tool_result = (tool_result or "") + "\n" + "\n".join(all_tool_outputs)

            previous_visible = remove_tool_blocks(current_response)
            follow_up_prompt = f"""Previous context:
{context}

User:
{user_input}

Previous assistant visible response:
{previous_visible}

Tool outputs:
{chr(10).join(all_tool_outputs)}

If you need more tools, output XML tool calls. Otherwise, provide your final response to the user.
"""
            current_response = asyncio.run(call_ollama(follow_up_prompt, SYSTEM_PROMPT))

        response = remove_tool_blocks(current_response) or current_response

        print(f"\n🤖 {response}\n")

        update_memory(user_input, response, tool_result)
        if tool_result:
            asyncio.run(extract_key_info(user_input, response))


if __name__ == "__main__":
    main()
