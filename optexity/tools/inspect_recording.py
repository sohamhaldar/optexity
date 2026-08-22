"""Views over a recording directory.

    python -m optexity.tools.inspect_recording replay_cache/saucedemo nodes

Views: cache (what was recorded), nodes (what a replay runs), provenance (every
command traced back to the recording), meta.
"""

import json
import sys
from pathlib import Path

def load_actions(recording: Path) -> list[dict]:
    return [
        action
        for line in (recording / "cache.jsonl").read_text().splitlines()
        if line.strip()
        for action in json.loads(line)["actions"]
    ]


def nodes_of(recording: Path, name: str = "cached.json") -> list[tuple[str, dict]]:
    automation = json.loads((recording / name).read_text())
    out = []
    for node in automation["nodes"]:
        interaction = node["interaction_action"]
        kind = next(k for k in interaction if isinstance(interaction[k], dict))
        out.append((kind, interaction[kind]))
    return out


def cache(recording: Path) -> None:
    print(f"{'step':<6}{'action':<10}{'kept':<8}{'selector':<11}command")
    for line in (recording / "cache.jsonl").read_text().splitlines():
        row = json.loads(line)
        for action in row["actions"]:
            if action["action"] == "done":
                continue
            kept = "dropped" if action["read_only"] or action.get("error") else "kept"
            how = (
                "nth" if action.get("disambiguated") else "unique" if action.get("verified") else "-"
            )
            print(
                f"{row['step']:<6}{action['action']:<10}{kept:<8}{how:<11}"
                f"{(action.get('command') or '')[:56]}"
            )


def nodes(recording: Path) -> None:
    for i, (kind, body) in enumerate(nodes_of(recording)):
        extra = body.get("input_text") or ("press_enter" if body.get("press_enter") else "")
        print(f"{i:>3}. {kind:<14}{str(body.get('command', ''))[:54]:<56}{extra}")


def provenance(recording: Path) -> None:
    """Every command in the automation must appear in the recording."""
    recorded = {a["command"] for a in load_actions(recording) if a.get("command")}
    commands = [body.get("command") for _, body in nodes_of(recording) if body.get("command")]
    invented = [c for c in commands if c not in recorded]
    print(f"commands in automation : {len(commands)}")
    print(f"found in cache.jsonl   : {len(commands) - len(invented)}")
    print(f"invented               : {len(invented)}")
    for c in invented:
        print(f"  !! {c}")


def meta(recording: Path) -> None:
    print((recording / "meta.json").read_text().strip())


if __name__ == "__main__":
    recording, view = Path(sys.argv[1]), sys.argv[2]
    {"cache": cache, "nodes": nodes, "provenance": provenance, "meta": meta}[view](recording)
