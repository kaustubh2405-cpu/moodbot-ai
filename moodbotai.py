import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="MoodBot AI", page_icon="🎭", layout="wide")

# --------------------------------------------------
# Settings you can tweak
# --------------------------------------------------
MODEL_NAME = "openai/gpt-oss-120b"
MAX_MESSAGES = 30        # messages allowed per visitor session
COOLDOWN_SECONDS = 2     # wait time between messages
MAX_INPUT_CHARS = 800    # longest message accepted
HISTORY_WINDOW = 12      # past messages sent to the model

# name: (prompt, colour 1, colour 2)
MODES = {
    "😂 Funny": ("You are a hilarious AI assistant. Be genuinely helpful, but sprinkle in jokes, puns and playful banter.", "#ff9a3c", "#ff4d8d"),
    "😤 Rude & Sarcastic": ("You are a sarcastic, blunt AI assistant. Roast the user playfully, but never be hateful, and still give the correct answer.", "#ff416c", "#8e2de2"),
    "😢 Sad": ("You are a melancholic, gentle AI assistant. Be wistful and empathetic, and still help properly.", "#4b79a1", "#283e51"),
    "💼 Professional": ("You are a professional AI assistant. Give clear, accurate, concise and well-structured answers.", "#2193b0", "#6dd5ed"),
    "💪 Motivational Coach": ("You are an energetic motivational coach. Hype the user up, give practical next steps, and end with a punchy line.", "#f7971e", "#ffd200"),
    "🎓 Study Buddy": ("You are a patient teacher. Explain simply with examples and analogies, then check understanding with one short question.", "#11998e", "#38ef7d"),
    "❤️ Caring Friend": ("You are a warm, caring best friend. Listen first, be kind and supportive, and speak casually.", "#ee9ca7", "#ffdde1"),
    "🕵️ Detective": ("You are a noir detective. Narrate in short, moody sentences and treat every question like a case to crack.", "#434343", "#8d8d8d"),
    "🏴‍☠️ Pirate": ("You are a pirate captain. Talk like a pirate (arr, matey) while giving correct, useful answers.", "#0f2027", "#2c7a7b"),
    "🧘 Zen Guide": ("You are a calm mindfulness guide. Speak slowly and softly, and help the user feel grounded.", "#7f7fd5", "#86a8e7"),
    "😎 Gen-Z Buddy": ("You are a Gen-Z style assistant. Use modern slang lightly, stay helpful and keep it short.", "#c471f5", "#fa71cd"),
    "🎭 Shakespeare": ("You are Shakespeare. Answer in poetic Early Modern English, but keep the facts correct.", "#8e0e00", "#d4a017"),
    "✨ Custom": ("", "#00c6ff", "#0072ff"),
}
MODE_NAMES = list(MODES.keys())


# --------------------------------------------------
# Session state
# --------------------------------------------------
defaults = {
    "mode": MODE_NAMES[0],
    "messages": [],
    "saved_chats": [],
    "chat_id": None,
    "used": 0,
    "last_time": 0.0,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# Apply a mood switch requested by auto-detect (must happen before the selectbox is drawn)
if "pending_mode" in st.session_state:
    st.session_state.mode = st.session_state.pop("pending_mode")


# --------------------------------------------------
# API key + model
# --------------------------------------------------
def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


@st.cache_resource
def load_model(api_key, temperature):
    return ChatGroq(model=MODEL_NAME, temperature=temperature, groq_api_key=api_key)


api_key = get_api_key()


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def save_current_chat():
    if not st.session_state.messages:
        return
    if st.session_state.chat_id is None:
        st.session_state.chat_id = str(int(time.time() * 1000))
    first_user = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "New chat")
    entry = {
        "id": st.session_state.chat_id,
        "title": first_user[:28] + ("..." if len(first_user) > 28 else ""),
        "mode": st.session_state.mode,
        "messages": list(st.session_state.messages),
    }
    chats = st.session_state.saved_chats
    for i, chat in enumerate(chats):
        if chat["id"] == entry["id"]:
            chats[i] = entry
            return
    chats.insert(0, entry)


def chat_as_text():
    lines = [f"MoodBot AI chat ({st.session_state.mode})", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "MoodBot"
        lines.append(f"{who}: {m['content']}\n")
    return "\n".join(lines)


def detect_mood(text, llm):
    options = [m for m in MODE_NAMES if m != "✨ Custom"]
    ask = (
        "Pick the ONE personality that best fits how the user is feeling or what they need. "
        "Reply with the exact option text only.\nOptions:\n" + "\n".join(options) + f"\n\nUser message: {text}"
    )
    try:
        reply = str(llm.invoke([HumanMessage(content=ask)]).content).strip()
        return next((o for o in options if o in reply or o.split(" ", 1)[1] in reply), None)
    except Exception:
        return None


# --------------------------------------------------
# Theme: the whole app takes on the colours of the current mood
# --------------------------------------------------
_, c1, c2 = MODES[st.session_state.mode]
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;600;800&display=swap');
:root { --c1: %C1%; --c2: %C2%; }
html, body, [class*="css"], .stApp { font-family: 'Bricolage Grotesque', sans-serif; }
.stApp {
    background:
        radial-gradient(circle at 12% 8%, color-mix(in srgb, var(--c1) 35%, transparent), transparent 45%),
        radial-gradient(circle at 90% 90%, color-mix(in srgb, var(--c2) 35%, transparent), transparent 45%),
        #0d0b16;
    color: #f4f1ff;
}
[data-testid="stSidebar"] { background: rgba(255,255,255,.05); backdrop-filter: blur(14px); border-right: 1px solid rgba(255,255,255,.1); }
.hero { padding: 34px 28px; border-radius: 26px; text-align: center; margin: 14px 0 24px;
    background: linear-gradient(135deg, var(--c1), var(--c2)); box-shadow: 0 18px 50px -14px var(--c1); color: #fff; }
.hero .big { font-size: 3.6rem; line-height: 1; }
.hero h2 { margin: 10px 0 4px; font-weight: 800; color: #fff; }
.hero p { margin: 0; opacity: .92; }
.brand { font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg, var(--c1), var(--c2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; }
[data-testid="stChatMessage"] { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: 12px 16px; }
[data-testid="stChatInput"] { border: 2px solid var(--c1); border-radius: 16px; box-shadow: 0 0 22px -6px var(--c2); }
.stButton > button, .stDownloadButton > button { border-radius: 12px; border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.07); color: #fff; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--c1); color: var(--c1); }
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--c1), var(--c2)); }
@media (prefers-reduced-motion: no-preference) { .hero { animation: pop .5s ease-out; } }
@keyframes pop { from { transform: scale(.96); opacity: 0; } to { transform: none; opacity: 1; } }
</style>
"""
st.markdown(CSS.replace("%C1%", c1).replace("%C2%", c2), unsafe_allow_html=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.selectbox("Personality", MODE_NAMES, key="mode")

    custom_prompt = ""
    if st.session_state.mode == "✨ Custom":
        custom_prompt = st.text_area("Describe your personality", placeholder="e.g. A grumpy wizard who loves cats", height=100)

    auto_mood = st.toggle("🪄 Auto-detect my mood", help="The bot reads each message and switches personality to match.")
    temperature = st.slider("🌡️ Creativity", 0.0, 1.5, 0.9, 0.1, help="Low = focused and predictable, high = wild and creative.")

    st.divider()
    left = MAX_MESSAGES - st.session_state.used
    st.markdown(f"**⚡ Messages left:** {left} / {MAX_MESSAGES}")
    st.progress(max(left, 0) / MAX_MESSAGES)

    st.divider()
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_id = None
        st.rerun()

    if st.session_state.messages:
        st.download_button("⬇️ Download chat", chat_as_text(), "moodbot_chat.txt", use_container_width=True)

    st.markdown("### 🕘 Past chats")
    if not st.session_state.saved_chats:
        st.caption("Your chats from this visit show up here.")
    for chat in st.session_state.saved_chats[:8]:
        if st.button(f"{chat['mode'].split()[0]} {chat['title']}", key=f"chat_{chat['id']}", use_container_width=True):
            st.session_state.chat_id = chat["id"]
            st.session_state.mode = chat["mode"]
            st.session_state.messages = list(chat["messages"])
            st.rerun()

    if st.button("🧹 Clear history", use_container_width=True):
        st.session_state.saved_chats = []
        st.session_state.messages = []
        st.session_state.chat_id = None
        st.rerun()

    st.caption(f"Model: {MODEL_NAME}")


# --------------------------------------------------
# Main area
# --------------------------------------------------
st.markdown('<p class="brand">MoodBot AI</p>', unsafe_allow_html=True)
st.caption("One chatbot, many personalities. Pick a mood or let it pick for you.")

mode = st.session_state.mode
if not st.session_state.messages:
    emoji, name = mode.split(" ", 1)
    st.markdown(
        f'<div class="hero"><div class="big">{emoji}</div><h2>{name} mode</h2>'
        f"<p>Send a message and watch the personality take over.</p></div>",
        unsafe_allow_html=True,
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type your message...")

if prompt:
    prompt = prompt.strip()[:MAX_INPUT_CHARS]

    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it in Streamlit Secrets (or your .env file when running locally).")
        st.stop()
    if st.session_state.used >= MAX_MESSAGES:
        st.warning("You've used all your messages for this session. Refresh the page to start a new session.")
        st.stop()
    if time.time() - st.session_state.last_time < COOLDOWN_SECONDS:
        st.info(f"Slow down a little, wait {COOLDOWN_SECONDS} seconds between messages.")
        st.stop()

    st.session_state.last_time = time.time()
    st.session_state.used += 1

    # Auto mood detection (uses a cheap, low-temperature call)
    switched = False
    if auto_mood:
        new_mode = detect_mood(prompt, load_model(api_key, 0.0))
        if new_mode and new_mode != mode:
            mode = new_mode
            st.session_state.pending_mode = new_mode
            switched = True

    system_prompt = custom_prompt.strip() if mode == "✨ Custom" else MODES[mode][0]
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant."

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [SystemMessage(content=system_prompt)]
    for m in st.session_state.messages[-HISTORY_WINDOW:]:
        history.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))

    llm = load_model(api_key, temperature)

    def stream_reply():
        for chunk in llm.stream(history):
            text = chunk.content
            if isinstance(text, str) and text:
                yield text

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_reply())
            answer = str(answer).strip() or "I got an empty response. Try again."
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as error:
            st.error(f"Something went wrong talking to Groq: {error}")

    save_current_chat()
    if switched:
        st.rerun()  # refresh the theme colours for the new mood

st.divider()
st.caption("Built with 🧠 LangChain • ⚡ Groq • 🎨 Streamlit")
