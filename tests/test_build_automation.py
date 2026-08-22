"""Self-check for the pruning rules.

    pytest tests/          or      python tests/test_build_automation.py
"""

from optexity.tools.build_automation import build, identity, prune
from optexity.schema.automation import Parameters


def _action(name, command=None, xpath=None, text=None, error=None, read_only=False):
    return {
        "action": name,
        "params": {"text": text} if text is not None else {},
        "command": command,
        "verified": command is not None,
        "error": error,
        "read_only": read_only,
        "element": {"xpath": xpath, "attributes": {}, "aria_name": "", "text": "", "tag": "input"},
    }


def _step(n, *actions):
    return {"step": n, "url": "https://example.com", "actions": list(actions)}


def test_prune_drops_the_four_categories():
    fill = _action("input", command='locator("#a")', text="x")
    steps = [
        _step(1, fill),
        _step(2, _action("screenshot", read_only=True)),
        _step(3, _action("click", command='locator("#b")', error="boom")),
        _step(4, fill),  # the agent retrying because it could not verify itself
        _step(5, _action("done", text="summary")),
    ]
    kept, dropped = prune(steps)
    assert len(kept) == 1, kept
    assert dropped == {"errored": 1, "read_only": 1, "done": 1, "repeat": 1, "unsupported": 0}, dropped


def test_repeat_keeps_original_position():
    steps = [
        _step(1, _action("input", command='locator("#a")', text="first")),
        _step(2, _action("click", command='locator("#b")')),
        _step(3, _action("input", command='locator("#a")', text="second")),
    ]
    kept, _ = prune(steps)
    assert [a["action"] for a in kept] == ["input", "click"], "original order must survive"
    assert kept[0]["params"]["text"] == "second", "a repeat overwrites the value in place"


def test_identity_ignores_shifting_indices():
    a = _action("input", xpath="html/body/input")
    b = _action("input", xpath="html/body/input")
    a["index"], b["index"] = 11, 23  # same element, different serialisation
    assert identity(a) == identity(b)


def test_build_emits_valid_automation():
    steps = [
        _step(1, _action("input", command="locator(\"input[name='q']\")", text="hello")),
        _step(2, _action("screenshot", read_only=True)),
    ]
    automation, _ = build(steps, "https://example.com", Parameters(input_parameters={}, generated_parameters={}))
    assert len(automation.nodes) == 1
    node = automation.nodes[0].interaction_action.input_text
    assert node.command == "locator(\"input[name='q']\")"
    assert node.input_text == "hello"
    assert node.prompt_instructions, "the LLM fallback needs something to search on"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
