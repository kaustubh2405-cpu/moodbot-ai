import os
import json
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="MoodBot AI",
    page_icon="🤖",
    layout="wide",
)

MODES = {
    "😂 Funny Mode": (
        "You are a funny AI assistant. Be helpful, friendly, "
        "and humorous when appropriate."
    ),
    "🤝 Rude Mode": (
        "You are a rude AI assistant. Be direct, rude, "
        "and straightforward."
    ),
    "🎓 Sad Mode": (
        "You are a sad AI assistant. Be sad, empathetic, and understanding in your responses."
    ),
    "💼 Professional Mode": (
        "You are a professional AI assistant. Give clear, "
        "accurate, and concise answers."
    ),
}

HISTORY_FILE = "chat_history.json"


# --------------------------------------------------
# Local chat-history functions
# --------------------------------------------------

def load_saved_chats():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_saved_chats(chats):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(chats, file, indent=2, ensure_ascii=False)
    except OSError as error:
        st.error(f"Could not save chat history: {error}")


def make_chat_title(messages):
    for message in messages:
        if message.get("role") == "user":
            text = str(message.get("content", "")).strip()
            if text:
                return text[:40] + ("..." if len(text) > 40 else "")
    return "New Chat"


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "mode" not in st.session_state:
    st.session_state.mode = "😂 Funny Mode"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = load_saved_chats()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


# --------------------------------------------------
# Chat saving
# --------------------------------------------------

def save_current_chat():
    messages = st.session_state.messages

    if not messages:
        return

    chat_id = st.session_state.current_chat_id

    if chat_id is None:
        chat_id = str(int(time.time() * 1000))
        st.session_state.current_chat_id = chat_id

    now = time.strftime("%Y-%m-%d %H:%M:%S")

    existing = next(
        (
            chat
            for chat in st.session_state.saved_chats
            if chat.get("id") == chat_id
        ),
        None,
    )

    if existing is None:
        st.session_state.saved_chats.insert(
            0,
            {
                "id": chat_id,
                "title": make_chat_title(messages),
                "mode": st.session_state.mode,
                "created_at": now,
                "updated_at": now,
                "messages": list(messages),
            },
        )
    else:
        existing["title"] = make_chat_title(messages)
        existing["mode"] = st.session_state.mode
        existing["updated_at"] = now
        existing["messages"] = list(messages)

    save_saved_chats(st.session_state.saved_chats)


# --------------------------------------------------
# Groq model
# --------------------------------------------------

@st.cache_resource
def load_model(api_key):
    if not api_key:
        return None

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.9,
        groq_api_key=api_key,
    )


groq_api_key = os.getenv("GROQ_API_KEY")
model = load_model(groq_api_key)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    selected_mode = st.selectbox(
        "Choose personality",
        list(MODES.keys()),
        index=list(MODES.keys()).index(st.session_state.mode),
    )

    if selected_mode != st.session_state.mode:
        st.session_state.mode = selected_mode
        st.rerun()

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

    st.markdown("### 🕘 Messages History")

    if st.session_state.saved_chats:
        chat_options = ["➕ New Chat"] + [
            f'{chat.get("mode", "🤖")} | {chat.get("title", "Untitled")}'
            for chat in st.session_state.saved_chats
        ]

        selected_history = st.selectbox(
            "Previous conversations",
            chat_options,
            key="history_selector",
        )

        if selected_history != "➕ New Chat":
            selected_index = chat_options.index(selected_history) - 1
            selected_chat = st.session_state.saved_chats[selected_index]

            if st.session_state.current_chat_id != selected_chat.get("id"):
                st.session_state.current_chat_id = selected_chat.get("id")
                st.session_state.mode = selected_chat.get(
                    "mode", "😂 Funny Mode"
                )
                st.session_state.messages = list(
                    selected_chat.get("messages", [])
                )
                st.rerun()
    else:
        st.caption("No saved conversations yet.")

    st.divider()

    if st.button("🧹 Delete All History", use_container_width=True):
        st.session_state.saved_chats = []
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        save_saved_chats([])
        st.rerun()

    st.divider()
    st.caption("Model: openai/gpt-oss-120b")
    st.caption("Powered by LangChain + Groq")


# --------------------------------------------------
# Main interface
# --------------------------------------------------

# IMPORTANT:
# MODES values are strings, so do NOT use:
# current_prompt, _ = MODES[...]
current_prompt = MODES[st.session_state.mode]

st.title("🤖 MoodBot AI")
st.caption("A multi-personality AI chatbot built with LangChain + Groq")


if not st.session_state.messages:
    st.markdown(
        f"""
        <div style="
            padding: 28px;
            border-radius: 18px;
            text-align: center;
            background: rgba(255,255,255,.05);
            border: 1px solid rgba(255,255,255,.08);
            margin: 20px 0;
        ">
            <div style="font-size: 3rem;">
                {st.session_state.mode.split()[0]}
            </div>
            <h2>You're in {st.session_state.mode}</h2>
            <p>Send a message and let the personality take over.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Display existing messages
# --------------------------------------------------

for message in st.session_state.messages:
    role = message.get("role", "assistant")
    content = message.get("content", "")

    if role not in ("user", "assistant"):
        role = "assistant"

    with st.chat_message(role):
        st.markdown(content)


# --------------------------------------------------
# Chat input
# --------------------------------------------------

prompt = st.chat_input("Type your message here...")


if prompt:
    if not groq_api_key:
        st.error(
            "GROQ_API_KEY is missing. Add GROQ_API_KEY=your_key "
            "to the .env file and restart Streamlit."
        )
        st.stop()

    if model is None:
        st.error("The Groq model could not be initialized.")
        st.stop()

    # Add and display user message
    user_message = {
        "role": "user",
        "content": prompt,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build LangChain message history
    history = [SystemMessage(content=current_prompt)]

    for message in st.session_state.messages:
        if message["role"] == "user":
            history.append(
                HumanMessage(content=message["content"])
            )
        elif message["role"] == "assistant":
            history.append(
                AIMessage(content=message["content"])
            )

    # Ask model
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                response = model.invoke(history)
                answer = str(response.content).strip()

                if not answer:
                    answer = "I received an empty response from the model."

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                save_current_chat()

            except Exception as error:
                st.error(f"Groq/API error: {error}")

                # Keep the user message saved even if the API fails.
                save_current_chat()


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    "Built with 🧠 LangChain • ⚡ Groq • 🎨 Streamlit • 💾 Local Chat History"
)