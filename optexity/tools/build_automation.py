"""Turn a recorded agent run into a deterministic Automation.

    python -m optexity.tools.build_automation cache.jsonl out.json --url URL

Reads the JSONL from browser_use.step_cache, drops what a replay does not need,
and validates the result. No LLM: every `command` comes verbatim from the cache.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoBackAction,
    GoToUrlAction,
    InputTextAction,
    InteractionAction,
    KeyPressAction,
    SelectOptionAction,
    UploadFileAction,
)
from optexity.schema.actions.keyboard_keys import KEY_NAMES
from optexity.schema.automation import ActionNode, Automation, Parameters

logger = logging.getLogger(__name__)

# Anything outside this is reported, never dropped quietly: a silently missing
# step gives a broken automation that looks fine.
SUPPORTED = {
    "input",
    "click",
    "select_dropdown",
    "send_keys",
    "navigate",
    "go_to_url",
    "search",
    "go_back",
    "upload_file",
}

# ActionNode's 5s default dominates a short automation, but anything that may
# navigate or dismiss an overlay still needs to settle before the next node.
END_SLEEP_SECONDS = {"input": 0.3}
DEFAULT_END_SLEEP_SECONDS = 1.5


def key_names(raw: str) -> list[str] | None:
    """browser-use key names in optexity's spelling, or None if unrecognised.

    Playwright spells them "Enter"/"Control+a"; KEY_NAMES is lowercase, and note
    KeyPressType's own capitalised values do not pass KeyPressAction.
    """
    keys = [k if k in KEY_NAMES else k.lower() for k in raw.split("+") if k]
    return keys if keys and all(k in KEY_NAMES for k in keys) else None


def load_steps(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def identity(action: dict[str, Any]) -> str:
    """Key identifying one act on one element, for collapsing repeats.

    Not the element index: those are positions in a per-step DOM serialisation
    and shift as the page changes, so one field can carry different numbers
    across steps.
    """
    element = action.get("element") or {}
    target = action.get("command") or element.get("xpath")
    if target:
        return f"{action['action']}:{target}"
    return f"{action['action']}:{json.dumps(action.get('params'), sort_keys=True)}"


def prune(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Actions a replay has to reproduce, plus a count of what was dropped.

    Drops errored, read-only and `done` actions, and collapses repeats onto the
    last value set. Purely mechanical - nothing infers intent.
    """
    dropped = {"errored": 0, "read_only": 0, "done": 0, "repeat": 0, "unsupported": 0}
    kept: dict[Any, dict[str, Any]] = {}

    for step in steps:
        for action in step["actions"]:
            name = action.get("action")
            if action.get("error"):
                dropped["errored"] += 1
            elif action.get("read_only"):
                dropped["read_only"] += 1
            elif name == "done":
                dropped["done"] += 1
            elif name not in SUPPORTED:
                dropped["unsupported"] += 1
                logger.warning(f"unsupported action {name!r} at step {step['step']}; skipped")
            elif name == "send_keys" and key_names(action["params"].get("keys", "")) is None:
                dropped["unsupported"] += 1
                logger.warning(f"unknown key {action['params'].get('keys')!r} at step {step['step']}; skipped")
            else:
                key = identity(action)
                if key in kept:
                    dropped["repeat"] += 1
                # Reassigning keeps the key's original position, so the order the
                # agent acted in survives; deleting and reinserting would not.
                kept[key] = {**action, "step": step["step"], "url": step["url"]}

    actions = list(kept.values())

    # InputTextAction clicks the field itself, and a standalone click can be
    # intercepted once a settle gap lets the page shift under it.
    redundant = [
        i
        for i in range(len(actions) - 1)
        if actions[i]["action"] == "click"
        and actions[i + 1]["action"] == "input"
        and actions[i].get("command")
        and actions[i]["command"] == actions[i + 1].get("command")
    ]
    for i in reversed(redundant):
        dropped["redundant_click"] = dropped.get("redundant_click", 0) + 1
        del actions[i]

    # A key_press node has no element, so Enter lands wherever focus happens to
    # be; press_enter sends it to the field just filled.
    folded = [
        i
        for i in range(len(actions) - 1)
        if actions[i]["action"] == "input"
        and actions[i + 1]["action"] == "send_keys"
        and (actions[i + 1]["params"].get("keys") or "").lower() == "enter"
    ]
    for i in reversed(folded):
        actions[i]["press_enter"] = True
        dropped["folded_enter"] = dropped.get("folded_enter", 0) + 1
        del actions[i + 1]

    return actions, dropped


def describe(action: dict[str, Any]) -> str:
    """Element description for prompt_instructions, the LLM's only handle once a
    command stops matching."""
    element = action.get("element") or {}
    attributes = element.get("attributes") or {}
    label = (
        element.get("aria_name")
        or attributes.get("aria-label")
        or attributes.get("placeholder")
        or attributes.get("name")
        or element.get("text")
        or element.get("tag")
        or "element"
    )
    if action["action"] == "input":
        return f"Enter the value in the {label} field"
    if action["action"] == "click":
        return f"Click the {label} element"
    if action["action"] == "select_dropdown":
        return f"Select the option in the {label} dropdown"
    if action["action"] == "send_keys":
        return f"Press {action['params'].get('keys', 'the key')}"
    if action["action"] == "upload_file":
        return f"Upload the file to the {label} input"
    if action["action"] == "go_back":
        return "Go back to the previous page"
    return f"Interact with the {label} element"


def to_node(action: dict[str, Any]) -> ActionNode:
    name, params, command = action["action"], action.get("params") or {}, action.get("command")
    instructions = describe(action)

    if name == "input":
        interaction = InteractionAction(
            input_text=InputTextAction(
                command=command,
                prompt_instructions=instructions,
                input_text=params.get("text", ""),
                press_enter=action.get("press_enter", False),
            )
        )
    elif name == "click":
        interaction = InteractionAction(
            click_element=ClickElementAction(command=command, prompt_instructions=instructions)
        )
    elif name == "select_dropdown":
        interaction = InteractionAction(
            select_option=SelectOptionAction(
                command=command,
                prompt_instructions=instructions,
                select_values=[params["text"]] if params.get("text") else None,
            )
        )
    elif name == "send_keys":
        keys = key_names(params["keys"])
        assert keys is not None, f"unknown key {params['keys']!r} should have been pruned"
        interaction = InteractionAction(
            key_press=KeyPressAction(
                type=keys[0] if len(keys) == 1 else keys, prompt_instructions=instructions
            )
        )
    elif name == "upload_file":
        interaction = InteractionAction(
            upload_file=UploadFileAction(
                command=command, prompt_instructions=instructions, file_path=params.get("path")
            )
        )
    elif name == "go_back":
        interaction = InteractionAction(go_back=GoBackAction())
    else:  # navigate / go_to_url / search
        interaction = InteractionAction(go_to_url=GoToUrlAction(url=params.get("url") or action["url"]))

    return ActionNode(
        type="action_node",
        interaction_action=interaction,
        end_sleep_time=END_SLEEP_SECONDS.get(name, DEFAULT_END_SLEEP_SECONDS),
    )


def build(steps: list[dict[str, Any]], url: str | None, parameters: Parameters) -> tuple[Automation, dict[str, int]]:
    actions, dropped = prune(steps)
    if not actions:
        raise SystemExit("nothing replayable in the cache - did the agent do anything?")

    unverified = [a for a in actions if not a.get("verified")]
    for action in unverified:
        logger.warning(f"step {action['step']}: {action.get('command')!r} was not unique on the page")

    automation = Automation(
        url=url or steps[0]["url"],
        parameters=parameters,
        nodes=[to_node(a) for a in actions],
    )
    return automation, dropped


def main(argv: list[str] | None = None) -> int:
    # force=True: importing optexity configures the root logger, making a plain
    # basicConfig a no-op.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="JSONL written by browser_use.step_cache")
    parser.add_argument("out", type=Path, help="where to write the deterministic automation")
    parser.add_argument("--url", help="starting url (defaults to the first recorded step)")
    parser.add_argument(
        "--params",
        type=Path,
        help="automation to copy input_parameters from, so the rebuilt one keeps the same contract",
    )
    args = parser.parse_args(argv)

    steps = load_steps(args.cache)
    parameters = Parameters(input_parameters={}, generated_parameters={})
    if args.params:
        parameters = Automation.model_validate(json.loads(args.params.read_text())).parameters

    automation, dropped = build(steps, args.url, parameters)
    args.out.write_text(json.dumps(automation.model_dump(exclude_none=True, mode="json"), indent=4) + "\n")

    total = sum(len(s["actions"]) for s in steps)
    logger.info(f"{len(steps)} steps, {total} actions -> {len(automation.nodes)} nodes")
    logger.info("dropped: " + ", ".join(f"{v} {k}" for k, v in dropped.items() if v))
    logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
