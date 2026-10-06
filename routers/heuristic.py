"""A frozen word-count baseline, not a learned difficulty estimate."""

RULE_ID = "goal-length-v1"
MAX_WORDS = 100


def choose_provider(goal: str) -> str:
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a nonempty string")
    return "deepseek" if len(goal.split()) <= MAX_WORDS else "sol"
