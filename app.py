import os
import streamlit as st
from huggingface_hub import InferenceClient

# --- Page config ---
st.set_page_config(page_title="MuruguChat", page_icon="💬")

# --- Hugging Face token handling (3 layers of fallback) ---
hf_token = None

# 1) Try Streamlit secrets
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    pass

# 2) Try environment variable
if not hf_token:
    hf_token = os.getenv("HF_TOKEN")

# 3) Sidebar input fallback
if not hf_token:
    with st.sidebar:
        st.warning("⚠️ No HF_TOKEN found. Please paste it below.")
        hf_token = st.text_input("HF token", type="password")

# --- Model setup ---
MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-20b")
client = InferenceClient(model=MODEL_ID, token=hf_token)

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, 512, 32)
    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")
    with st.expander("ℹ️ About MuruguChat"):
        st.write(
            """
            **MuruguChat** is a conversational AI chatbot built with:
            - [Streamlit](https://streamlit.io/) for the app interface  
            - [Hugging Face](https://huggingface.co/) Inference API for the model  

            Developed by **Murugu** ✨  
            """
        )

# --- App title ---
st.title("💬 MuruguChat")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Helper: stream response ---
def stream_chat(messages):
    """Stream tokens from Hugging Face model."""
    events = client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    for event in events:
        if not event.choices:  # skip empty events
            continue
        choice = event.choices[0]
        if choice.delta and choice.delta.content:
            yield choice.delta.content

# --- Chat input ---
user_text = st.chat_input("Type your message...")
if user_text:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Stream assistant reply
    with st.chat_message("assistant"):
        full_response = st.write_stream(
            stream_chat(st.session_state.messages)
        )

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Made with ❤️ by Murugu — MuruguChat powered by Streamlit & Hugging Face</div>",
    unsafe_allow_html=True,
)


