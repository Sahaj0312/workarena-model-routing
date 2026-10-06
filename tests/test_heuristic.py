import pytest

from routers.heuristic import MAX_WORDS, RULE_ID, choose_provider


def test_frozen_rule_identity():
    assert RULE_ID == "goal-length-v1"
    assert MAX_WORDS == 100


@pytest.mark.parametrize("word_count,expected", [(1, "deepseek"), (100, "deepseek"), (101, "sol")])
def test_word_count_boundary(word_count, expected):
    assert choose_provider(" ".join(["synthetic"] * word_count)) == expected


def test_whitespace_does_not_add_words():
    goal = " \n\t" + " \t\n".join(["synthetic"] * 100) + "\n "
    assert choose_provider(goal) == "deepseek"
    assert choose_provider(goal + "extra") == "sol"


@pytest.mark.parametrize("goal", ["", " \t\n", None, 0, 1.5, False, [], {}, b"synthetic"])
def test_rejects_missing_or_non_string_goal(goal):
    with pytest.raises(ValueError, match="goal must be a nonempty string"):
        choose_provider(goal)
