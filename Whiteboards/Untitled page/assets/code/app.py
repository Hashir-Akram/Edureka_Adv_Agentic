"""
Groq Chatbot — Streamlit web UI (streaming).

Run:  streamlit run app.py
"""

import streamlit as st
from groq import Groq

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")

# config.py calls sys.exit() if the key is missing — show a friendly error instead
try:
    import config
except SystemExit as e:
    st.error(str(e))
    st.stop()

client = Groq(api_key=config.GROQ_API_KEY)

# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, config.TEMPERATURE, 0.1)
    max_tokens = st.slider("Max tokens", 128, 4096, config.MAX_TOKENS, 128)
    remember = st.toggle("Remember conversation", value=False,
                         help="Off = each question is sent alone (no memory).")
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Model: `{config.MODEL}`")

st.title("🤖 Groq Chatbot")

# ---------- Chat display (what's shown on screen) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def stream_reply(messages):
    """Yield text pieces from Groq as they arrive."""
    stream = client.chat.completions.create(
        model=config.MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""


# ---------- Input ----------
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # What we SEND to the model
    system = {"role": "system", "content": config.SYSTEM_PROMPT}
    if remember:
        payload = [system] + st.session_state.messages      # full history
    else:
        payload = [system, {"role": "user", "content": prompt}]  # this question only

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(payload))
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"⚠️ Error: {e}")