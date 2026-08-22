"""Run a cached automation repeatedly, fixing whatever still needs the LLM.

    python -m optexity.tools.learn replay_cache/saucedemo --endpoint NAME

Each pass runs the recording's cached.json, finds the nodes that fell back to
the LLM, and either corrects their locator, forces the click, or gives the page
longer to settle. Stops once nothing falls back, or once two passes fail to
improve on that count.

Needs an inference server already serving that cached.json via
OPTEXITY_LOCAL_AUTOMATION, with OPTEXITY_FALLBACK_LOG set to the recording's
fallbacks.jsonl.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from optexity.schema.automation import Automation
from optexity.tools.build_automation import END_SLEEP_SECONDS, automation_from, load_steps, prune

logger = logging.getLogger(__name__)

SETTLE_STEP_SECONDS = 1.5
MAX_SETTLE_SECONDS = 6.0

# Playwright refuses a click it cannot make safely. No wait or locator fixes
# these; the action has to stop asking for the checks.
UNACTIONABLE = ("outside of the viewport", "intercepts pointer events", "not enabled")


def run_once(server: str, endpoint: str, params: dict[str, Any]) -> float:
    """Trigger the automation, wait it out, and return how long it took."""
    started = time.monotonic()
    httpx.post(
        f"{server}/inference",
        json={"endpoint_name": endpoint, "input_parameters": params, "unique_parameter_names": []},
        timeout=60,
    ).raise_for_status()

    while time.monotonic() - started < 600:
        time.sleep(5)
        if not httpx.get(f"{server}/is_task_running", timeout=10).json():
            return time.monotonic() - started
    raise SystemExit("automation did not finish within 10 minutes")


def diagnose(fallbacks: Path) -> dict[int, dict[str, Any]]:
    """Node index -> what the fallback recorded, for the nodes that needed it."""
    if not fallbacks.exists():
        return {}
    rows = [json.loads(line) for line in fallbacks.read_text().splitlines() if line.strip()]
    return {row["step"]: row for row in rows}


def patch(actions: list[dict[str, Any]], failed: dict[int, dict[str, Any]]) -> list[str]:
    """Apply one correction per failed node. Returns what changed, for the log."""
    changes = []
    for index, row in failed.items():
        if index >= len(actions):
            continue
        action, better, error = actions[index], row.get("best_candidate"), row.get("error", "")
        if better:
            better = better.removeprefix("page.").rsplit(".", 1)[0]

        if better and better != action.get("command"):
            action["command"] = better
            changes.append(f"node {index}: command -> {better}")
        elif any(reason in error for reason in UNACTIONABLE) and not action.get("force"):
            action["force"] = True
            changes.append(f"node {index}: force click")
        elif index > 0:
            previous = actions[index - 1]
            current = previous.get("end_sleep_time", END_SLEEP_SECONDS.get(previous["action"], 1.5))
            if current < MAX_SETTLE_SECONDS:
                previous["end_sleep_time"] = min(current + SETTLE_STEP_SECONDS, MAX_SETTLE_SECONDS)
                changes.append(f"node {index - 1}: settle -> {previous['end_sleep_time']}s")
    return changes


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="directory built by build_automation")
    parser.add_argument("--endpoint", required=True, help="registered endpoint used to trigger a run")
    parser.add_argument("--server", default="http://localhost:9001")
    parser.add_argument("--max-passes", type=int, default=4)
    args = parser.parse_args(argv)

    automation_path = args.recording / "cached.json"
    fallbacks_path = args.recording / "fallbacks.jsonl"
    current = Automation.model_validate(json.loads(automation_path.read_text()))
    actions, _ = prune(load_steps(args.recording / "cache.jsonl"))
    fewest, stalled = None, 0

    logger.info(f"{'pass':<6}{'wall':<9}{'fallbacks':<12}changed")
    for attempt in range(1, args.max_passes + 1):
        fallbacks_path.unlink(missing_ok=True)
        seconds = run_once(args.server, args.endpoint, current.parameters.input_parameters)
        failed = diagnose(fallbacks_path)
        if not failed:
            logger.info(f"{attempt:<6}{seconds:<9.0f}{0:<12}converged")
            return 0

        stalled = stalled + 1 if fewest is not None and len(failed) >= fewest else 0
        fewest = min(fewest, len(failed)) if fewest is not None else len(failed)
        changes = patch(actions, failed)
        logger.info(f"{attempt:<6}{seconds:<9.0f}{len(failed):<12}{'; '.join(changes) or 'nothing to correct'}")

        if not changes or stalled >= 2:
            logger.info("no further improvement available; the rest needs the LLM")
            return 0

        automation = automation_from(actions, current.url, current.parameters)
        automation_path.write_text(
            json.dumps(automation.model_dump(exclude_none=True, mode="json"), indent=4)
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
