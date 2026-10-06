import ast
import json

import pytest

from agent.actions import ACTION_HELP, parse_action


@pytest.mark.parametrize(
    "content",
    [
        "__import__('os').system('echo unsafe')",
        '{"action":"goto","args":["https://example.com"]}',
        '{"action":"click; print(1)","args":["42"]}',
        '[{"action":"click","args":["42"]}]',
        '{"action":"click","args":["42"],"code":"print(1)"}',
        '{"action":"scroll","args":[true,0]}',
        '{"action":"scroll","args":[NaN,0]}',
        '{"action":"scroll","args":["0","600"]}',
        '{"action":"tab_focus","args":[-1]}',
        '{"action":"tab_focus","args":[1.0]}',
        '{"action":"tab_focus","args":["1"]}',
        '{"action":"fill","args":["42"]}',
    ],
)
def test_rejects_code_and_invalid_actions(content):
    with pytest.raises(ValueError):
        parse_action(content)


def test_text_that_looks_like_code_remains_one_literal_argument():
    text = "'); __import__('os').system('echo unsafe') #\n"
    action = parse_action(json.dumps({"action": "fill", "args": ["42", text]}))
    module = ast.parse(action)

    assert len(module.body) == 1
    call = module.body[0].value
    assert isinstance(call, ast.Call)
    assert call.func.id == "fill"
    assert [ast.literal_eval(arg) for arg in call.args] == ["42", text]


def test_stop_has_no_browser_code():
    assert parse_action('{"action":"stop","args":[]}') is None


def test_prompt_examples_match_the_action_contract():
    examples = [line for line in ACTION_HELP.splitlines() if line.startswith('{"action"')]
    assert [parse_action(example) for example in examples] == [
        "fill('42', 'hello')",
        "scroll(0, 600)",
        "tab_focus(0)",
        None,
    ]
