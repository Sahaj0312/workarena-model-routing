"""Accept one browser action. Never execute model code."""

import json
import math

STRING_ARGS = {
    "click": 1,
    "dblclick": 1,
    "hover": 1,
    "focus": 1,
    "clear": 1,
    "fill": 2,
    "press": 2,
    "select_option": 2,
    "send_msg_to_user": 1,
    "report_infeasible": 1,
}
NO_ARGS = {"go_back", "go_forward", "noop", "stop"}

ACTION_HELP = """Return one JSON object with exactly two keys: action and args.
action must be a JSON string. args must be a JSON array in the argument order below.
Use exactly the required arguments. Do not add empty strings as placeholders.
All arguments must be quoted JSON strings except for scroll and tab_focus.
This includes browser element IDs (bids), even when they contain only digits.
Use bids from the current accessibility tree.
Available actions and argument order:
click(bid), dblclick(bid), hover(bid), focus(bid), clear(bid)
fill(bid, text), press(bid, key_comb), select_option(bid, option)
scroll(delta_x, delta_y), tab_focus(index), go_back(), go_forward(), noop()
send_msg_to_user(text): submit an answer when the task asks for one.
report_infeasible(reason): report a task that cannot be done in this environment.
stop(): end the attempt when you have finished or cannot continue.
scroll takes two finite JSON numbers, not strings. Each must be from -10000 to
10000, inclusive. Values are pixels: horizontal delta first, then vertical delta.
tab_focus takes one JSON integer from 0 to 255, inclusive, not a string or decimal.
go_back, go_forward, noop, and stop take an empty args array: [].
press uses key strings such as "Enter", "Tab", or "Control+a".
Complete JSON examples:
{"action": "fill", "args": ["42", "hello"]}
{"action": "scroll", "args": [0, 600]}
{"action": "tab_focus", "args": [0]}
{"action": "stop", "args": []}
Do not output code, Markdown, or more than one action."""


def parse_action(content: str) -> str | None:
    """Return a literal-only BrowserGym call, or None for stop."""
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("The action must be a JSON object.") from exc
    if not isinstance(value, dict) or set(value) != {"action", "args"}:
        raise ValueError("The action must contain only action and args.")
    name, args = value["action"], value["args"]
    if not isinstance(name, str) or not isinstance(args, list):
        raise ValueError("action must be a string and args must be a list.")
    if name in STRING_ARGS:
        valid = len(args) == STRING_ARGS[name] and all(isinstance(x, str) for x in args)
    elif name in NO_ARGS:
        valid = not args
    elif name == "scroll":
        valid = len(args) == 2 and all(
            type(x) in (int, float) and math.isfinite(x) and abs(x) <= 10000 for x in args
        )
    elif name == "tab_focus":
        valid = len(args) == 1 and type(args[0]) is int and 0 <= args[0] <= 255
    else:
        raise ValueError("Unknown action.")
    if not valid:
        raise ValueError("Invalid action arguments.")
    if name == "stop":
        return None
    return f"{name}({', '.join(repr(x) for x in args)})"


def action_mapping():
    """Build BrowserGym's restricted action set."""
    from browsergym.core.action import functions
    from browsergym.core.action.highlevel import HighLevelActionSet

    names = sorted(set(STRING_ARGS) | (NO_ARGS - {"stop", "noop"}) | {"scroll", "tab_focus"})
    return HighLevelActionSet(
        subsets=["custom"],
        custom_actions=[getattr(functions, name) for name in names],
        multiaction=False,
        strict=True,
        retry_with_force=False,
    ).to_python_code
