"""Checks the dataset format. Used by build_dataset.py and view_dataset.py.

Kinds:
  single - one (user, assistant) pair. Mode comes from the user message.
  follow - multi-turn conversation. Every answer must be an explanation or a verdict.
  off    - off-topic question. Short answer, no format labels, exactly one question mark.
"""

MODE1 = ["Simple explanation:", "Example:", "Check yourself:"]
MODE2 = ["Verdict:", "What is right:", "What to improve:", "Think about it:"]
VERDICTS = ["Correct", "Partly correct", "Wrong"]
CHECK_PREFIX = "Check my understanding."


def normalize(ex):
    if isinstance(ex, dict):
        return ex["kind"], [tuple(t) for t in ex["turns"]]
    return "single", [tuple(ex)]


def infer_kind(turns):
    """Guess the kind when reading train.jsonl (used by view_dataset)."""
    if len(turns) > 1:
        return "follow"
    user, answer = turns[0]
    if user.startswith(CHECK_PREFIX) or answer.startswith(("Simple explanation:", "Verdict:")):
        return "single"
    return "off"


def _format(tag, answer, labels):
    problems = []
    lines = answer.split("\n")
    for label in labels:
        if label not in answer:
            problems.append(f"{tag}: '{label}' is missing")
        elif not any(line.startswith(label) for line in lines):
            problems.append(f"{tag}: '{label}' does not start a new line")
    return problems


def _common(tag, answer, max_words):
    problems = []
    if answer.count("?") != 1:
        problems.append(f"{tag}: {answer.count('?')} question marks (must be exactly 1)")
    words = len(answer.split())
    if words > max_words:
        problems.append(f"{tag}: answer too long ({words} words, max {max_words})")
    return problems


def validate_one(i, kind, turns):
    problems = []
    if kind == "follow" and len(turns) < 2:
        problems.append(f"#{i}: a follow-up conversation needs at least 2 turns")
    for t, (user, answer) in enumerate(turns, start=1):
        tag = f"#{i}" if len(turns) == 1 else f"#{i} (turn {t})"
        if kind == "off":
            if any(label in answer for label in ("Simple explanation:", "Verdict:")):
                problems.append(f"{tag}: off-topic answers must not contain format labels")
            problems += _common(tag, answer, 60)
            continue
        if kind == "single":
            mode2 = user.startswith(CHECK_PREFIX)
        elif answer.startswith("Simple explanation:"):
            mode2 = False
        elif answer.startswith("Verdict:"):
            mode2 = True
        else:
            problems.append(f"{tag}: answer starts with neither 'Simple explanation:' nor 'Verdict:'")
            continue
        problems += _format(tag, answer, MODE2 if mode2 else MODE1)
        if mode2:
            verdict = answer.split("\n")[0].replace("Verdict:", "").strip()
            if verdict not in VERDICTS:
                problems.append(f"{tag}: invalid verdict '{verdict}'")
        problems += _common(tag, answer, 160)
    return problems


def validate(examples):
    problems = []
    for i, ex in enumerate(examples, start=1):
        kind, turns = normalize(ex)
        problems += validate_one(i, kind, turns)
    return problems
