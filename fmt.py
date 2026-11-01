"""Builds answers in code, so line breaks can never get lost."""


def explain_reply(simple, example, question):
    return f"Simple explanation: {simple}\nExample: {example}\nCheck yourself: {question}"


def check_prompt(topic, claim):
    return f"Check my understanding.\nTopic: {topic}\nMy explanation: {claim}"


def check_reply(verdict, right, improve, think):
    return (f"Verdict: {verdict}\nWhat is right: {right}\n"
            f"What to improve: {improve}\nThink about it: {think}")


def E(user_question, simple, example, question):
    """Mode 1 example (single turn)."""
    return (user_question, explain_reply(simple, example, question))


def C(topic, claim, verdict, right, improve, think):
    """Mode 2 example (single turn)."""
    return (check_prompt(topic, claim), check_reply(verdict, right, improve, think))


def CR(user, verdict, right, improve, think):
    """Mode 2 reply to a plain student message (used inside follow-up conversations)."""
    return (user, check_reply(verdict, right, improve, think))


def OFF(user, reply):
    """Off-topic example."""
    return {"kind": "off", "turns": [(user, reply)]}


def FOLLOW(*turns):
    """Multi-turn conversation. Each turn is a (user, assistant) pair."""
    return {"kind": "follow", "turns": list(turns)}
