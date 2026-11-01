"""ML Learning Coach: Streamlit chat app powered ONLY by our locally loaded, fine-tuned model.

Base model: Qwen/Qwen2.5-1.5B-Instruct   |   Adapter: ./adapter (our own QLoRA run)
No external APIs are called. Start with:   streamlit run app.py
"""
import json
import os

import psutil
import streamlit as st

from engine import CoachEngine, ModelLoadError
from fmt import check_prompt

st.set_page_config(page_title="ML Learning Coach", page_icon="🎓", layout="wide")

HISTORY_FILE = os.environ.get("COACH_HISTORY_FILE", "chat_history.json")
MAX_HISTORY_MESSAGES = 12      # how many recent messages the model gets as memory
TOPICS = [
    "Overfitting", "Underfitting", "Train/test split", "Accuracy", "Gradient descent", "Loss function",
    "Linear regression", "Logistic regression", "Decision tree", "Random forest", "kNN", "k-means",
    "Precision and recall", "Confusion matrix", "Cross-validation", "Regularization", "Feature scaling",
    "Supervised vs unsupervised learning", "Bias and variance", "Hyperparameters", "Epoch and batch",
    "Features and labels", "Classification vs regression", "Neural network", "Data leakage",
]
LABELS = ["Simple explanation:", "Example:", "Check yourself:",
          "Verdict:", "What is right:", "What to improve:", "Think about it:"]
VERDICT_COLORS = {"Correct": "green", "Partly correct": "orange", "Wrong": "red"}


# ----------------------------------------------------------------------------- helpers
def format_answer(text):
    """Bold labels and a colored verdict, so the coach's structure is easy to read."""
    lines = []
    for line in text.split("\n"):
        for label in LABELS:
            if line.startswith(label):
                rest = line[len(label):].strip()
                if label == "Verdict:" and rest in VERDICT_COLORS:
                    rest = f":{VERDICT_COLORS[rest]}[**{rest}**]"
                line = f"**{label}** {rest}"
                break
        lines.append(line)
    return "\n\n".join(lines)


def load_history():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_history(messages):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def delete_history():
    try:
        os.remove(HISTORY_FILE)
    except OSError:
        pass


def history_for_model(messages):
    """Recent messages only, and the conversation must start with a user message."""
    recent = [{"role": m["role"], "content": m["content"]} for m in messages][-MAX_HISTORY_MESSAGES:]
    while recent and recent[0]["role"] != "user":
        recent.pop(0)
    return recent


def stats_caption(stats):
    if not stats:
        return ""
    return (f"⏱ first words after {stats['first_token_s']} s · {stats['tokens_per_s']} tokens/s · "
            f"{stats['cached_tokens']} prompt tokens reused from cache")


def render_message(m):
    with st.chat_message(m["role"]):
        st.markdown(m.get("display") or (format_answer(m["content"]) if m["role"] == "assistant" else m["content"]))
        if m["role"] == "assistant":
            if m.get("stats"):
                st.caption(stats_caption(m["stats"]))
            if m.get("base"):
                with st.expander("Base model answer (adapter disabled)"):
                    st.markdown(format_answer(m["base"]))


@st.cache_resource(show_spinner=False)
def load_engine():
    return CoachEngine()


def stream_into(box, engine, messages, max_tokens, adapter_enabled=True):
    """Streams the model's answer into `box`, token by token. Returns (text, stats)."""
    box.markdown("⏳ *Thinking... the first words can take a few seconds on a CPU.*")
    text = ""
    for piece in engine.stream(messages, max_new_tokens=max_tokens, adapter_enabled=adapter_enabled):
        text += piece
        box.markdown(format_answer(text) + " ▌")
    box.markdown(format_answer(text))
    return text.strip(), dict(engine.last_stats)


# ----------------------------------------------------------------------------- load the model
with st.spinner("Loading the model from disk (about 20 seconds)..."):
    try:
        engine = load_engine()
    except ModelLoadError as e:
        st.error(f"**The model could not be loaded.**\n\n{e}")
        st.button("Try again")          # clicking reruns the script, which retries the loading
        st.stop()
    except Exception as e:              # unexpected problem: still no crash, a readable message
        st.error(f"**Something unexpected went wrong while loading the model.**\n\n`{type(e).__name__}: {e}`")
        st.button("Try again")
        st.stop()

# ----------------------------------------------------------------------------- sidebar
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

with st.sidebar:
    st.title("🎓 ML Learning Coach")
    st.caption("A beginner-friendly ML tutor. QLoRA fine-tuned on Qwen2.5-1.5B-Instruct and running "
               "fully on this computer. No external APIs.")

    ram = psutil.virtual_memory()
    free_gb = ram.available / 1024**3
    st.caption(f"💾 Free memory: {free_gb:.1f} GB of {ram.total / 1024**3:.1f} GB")
    if free_gb < 1.0:
        st.warning("Free memory is very low, so answers will be slow or may fail. "
                   "Close other programs and browser tabs.")

    st.subheader("Check my understanding")
    with st.form("check_form", clear_on_submit=True):
        topic = st.selectbox("Topic", TOPICS)
        custom_topic = st.text_input("...or type your own topic")
        claim = st.text_area("Explain it in your own words", height=120,
                             placeholder="For example: Overfitting means the model has not learned enough.")
        submitted = st.form_submit_button("Check my explanation", use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    regen = c1.button("🔄 Regenerate", use_container_width=True, disabled=not st.session_state.messages)
    clear = c2.button("🗑 Clear chat", use_container_width=True, disabled=not st.session_state.messages)

    st.subheader("Settings")
    max_tokens = st.slider("Max answer length (tokens)", 64, 400, 256, 16,
                           help="Shorter answers finish faster on a CPU.")
    compare = st.toggle("Also show the base model's answer", value=False,
                        help="Runs the same question with the adapter disabled. Slow on a CPU.")
    persist = st.toggle("Save chat history to a file", value=True)

    with st.expander("Model info"):
        st.json(engine.info)
        if engine.last_stats:
            st.caption("Last answer")
            st.json(engine.last_stats)

# ----------------------------------------------------------------------------- state changes
if clear:
    st.session_state.messages = []
    st.session_state.pop("error", None)
    delete_history()
    st.rerun()

redo = False
if regen and st.session_state.messages:
    if st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages.pop()
    redo = bool(st.session_state.messages)

# ----------------------------------------------------------------------------- chat history
st.header("Chat")
if not st.session_state.messages:
    st.info("Ask me about a machine learning idea, for example: *What is overfitting?*, "
            "*Why do we split data into train and test sets?* or *What is gradient descent?*\n\n"
            "Or use **Check my understanding** in the sidebar: write what you think a concept means "
            "and I will tell you what is right and what to improve.")
for m in st.session_state.messages:
    render_message(m)
if st.session_state.get("error"):
    st.error(st.session_state["error"])        # kept across the page refresh after a failed answer

# ----------------------------------------------------------------------------- new input
prompt = st.chat_input("Ask a machine learning question...")
new_message = None
if submitted:
    topic_name = custom_topic.strip() or topic
    if not claim.strip():
        st.sidebar.warning("Please write your explanation first.")
    else:
        new_message = {
            "role": "user",
            "content": check_prompt(topic_name, claim.strip()),
            "display": f"**Check my understanding** · *{topic_name}*\n\n{claim.strip()}",
        }
elif prompt:
    new_message = {"role": "user", "content": prompt.strip()}

if new_message is not None:
    st.session_state.messages.append(new_message)
    render_message(new_message)

# ----------------------------------------------------------------------------- generate the answer
if new_message is not None or redo:
    st.session_state.pop("error", None)
    history = history_for_model(st.session_state.messages)
    answer, stats, base_answer, error = None, None, None, None
    with st.chat_message("assistant"):
        box = st.empty()
        try:
            answer, stats = stream_into(box, engine, history, max_tokens)
        except RuntimeError as e:
            error = f"{e}"
            box.error(error)
        except Exception as e:
            error = (f"**The answer could not be generated.**\n\n`{type(e).__name__}: {e}`\n\n"
                     "Try again, or lower the maximum answer length in the sidebar.")
            box.error(error)
        if answer is not None and compare:
            with st.expander("Base model answer (adapter disabled)", expanded=True):
                base_box = st.empty()
                try:
                    base_answer, _ = stream_into(base_box, engine, history, max_tokens, adapter_enabled=False)
                except Exception as e:
                    error = f"The base model answer failed: `{type(e).__name__}: {e}`"
                    base_box.error(error)
    if answer is not None:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "stats": stats, "base": base_answer})
    if error:
        st.session_state["error"] = error
    if persist:
        save_history(st.session_state.messages)
    st.rerun()
