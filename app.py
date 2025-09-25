# --- imports ---
import os
import tempfile
from dotenv import load_dotenv

import streamlit as st
import google.generativeai as gen_ai
from streamlit_mic_recorder import mic_recorder
from elevenlabs.client import ElevenLabs

# --- CONFIG ---
st.set_page_config(
    page_title="AgriBuddy",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv(dotenv_path=".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# --- GLOBAL CSS (AgriBuddy Style) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    body, .stApp { font-family: 'Inter', sans-serif !important; background-color: #FCFCEF !important; }
    h1, h2, h3, p, div { color: #2D2D2D !important; }
    [data-testid="stSidebar"] { background-color: #ECF3EC !important; border-right: 1px solid #D6E4D6; }
    [data-testid="stSidebar"] * { color: #2D2D2D !important; }
    .stChatMessage { border: 1px solid #E0E0E0; border-radius: 0.75rem !important; padding: 0.75rem; background-color: #f5f5f5 !important; margin-bottom: 0.5rem; color: #000 !important; }
    [data-testid="stChatInput"] textarea { background: #f5f5f5 !important; border: 1px solid #D6D6D6 !important; border-radius: 0.5rem !important; color: #000000 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #555555 !important; }
    button, [data-testid="stSelectbox"] > div { border-radius: 0.5rem !important; border: 1px solid #D6D6D6 !important; }
</style>
""", unsafe_allow_html=True)

# --- API KEYS CHECK ---
if not GOOGLE_API_KEY or not ELEVEN_API_KEY:
    st.error("API keys not found. Add GOOGLE_API_KEY and ELEVEN_API_KEY in `.env`")
    st.stop()

# --- CACHE Gemini + ElevenLabs clients ---
@st.cache_resource
def get_gemini_client():
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    return gen_ai

@st.cache_resource
def get_elevenlabs_client():
    return ElevenLabs(api_key=ELEVEN_API_KEY)

gen_ai_client = get_gemini_client()
try:
    eleven_client = get_elevenlabs_client()
except Exception as e:
    eleven_client = None
    print(f"[WARN] ElevenLabs not available, fallback to text only. ({e})")

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "language" not in st.session_state:
    st.session_state.language = "Malayalam"
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = True
if "chat" not in st.session_state:
    model = gen_ai_client.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction="You are a helpful assistant. Respond ONLY in Malayalam."
    )
    st.session_state.chat = model.start_chat()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Agriculture_icon.png", width=100)
    st.markdown("## AgriBuddy 🌱")
    st.caption("Your AI-powered Farming Buddy")
    st.markdown("---")

    with st.expander("⚙️ Settings", expanded=True):
        st.session_state.tts_enabled = st.toggle("Enable TTS", value=st.session_state.tts_enabled)
        st.session_state.language = st.selectbox("Language", ["Malayalam", "English", "Hindi", "Spanish"], index=0)
        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.rerun()

    with st.expander("💡 Tips", expanded=False):
        st.markdown("""
        - Use mic 🎤 to ask in your chosen language.  
        - Enable TTS for spoken replies.  
        - Replies stream word-by-word!  
        """)

    with st.expander("📌 Credits", expanded=False):
        st.markdown("Gemini (Google) · ElevenLabs (STT+TTS) · Streamlit")

# --- HEADER ---
st.markdown("""
<div style="text-align:center; margin-bottom:1rem;">
    <h1 style="color:#5BA96A; margin:0;">🌱 AgriBuddy</h1>
    <p style="color:#666; font-size:0.95rem;">Your AI-powered Agriculture Chat Assistant</p>
</div>
""", unsafe_allow_html=True)

# --- Navigation Tabs ---
tab_choice = st.radio("Navigation", ["New Chat", "Chat History"], horizontal=True, label_visibility="collapsed")

# --- MAIN CONTENT ---
if tab_choice == "New Chat":
    st.subheader("💬 Start a Conversation")
    st.caption("Type or Speak your query to talk with AgriBuddy.")
    chat_container = st.container()

    # Display history
    for msg in st.session_state.history:
        with chat_container.chat_message(msg["role"]):
            st.write(msg["text"])

    # --- Mic Input ---
    mic_audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹️ Stop", key="mic")
    user_prompt = st.chat_input("Type your message...")

    # Language → code + voice mapping
    lang_voice_map = {
        "Malayalam": {"code": "ml", "voice": "Aria"},
        "Hindi": {"code": "hi", "voice": "Aria"},
        "English": {"code": "en", "voice": "Bella"},
        "Spanish": {"code": "es", "voice": "Sofia"}
    }
    selected_lang = st.session_state.language
    lang_config = lang_voice_map.get(selected_lang, {"code": "en", "voice": "Bella"})
    lang_code, voice_name = lang_config["code"], lang_config["voice"]

    # If user speaks instead of typing
    if mic_audio and "bytes" in mic_audio and eleven_client:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(mic_audio["bytes"])
            audio_path = tmp_file.name
        try:
            transcript = eleven_client.speech_to_text.convert(
                file=audio_path, model="eleven_multilingual_v2", language_code=lang_code
            )
            user_prompt = transcript.text
        except Exception as e:
            print(f"[ERROR] STT failed, fallback to text input. ({e})")

    if user_prompt:
        st.session_state.history.append({"role": "user", "text": user_prompt})
        with chat_container.chat_message("user"):
            st.write(user_prompt)

        # Gemini reply
        with chat_container.chat_message("assistant"):
            placeholder = st.empty()
            bot_text = ""
            try:
                model = gen_ai_client.GenerativeModel(
                    "gemini-1.5-flash",
                    system_instruction=f"You are a helpful assistant. Respond ONLY in {selected_lang}."
                )
                st.session_state.chat = model.start_chat()
                for chunk in st.session_state.chat.send_message(user_prompt, stream=True):
                    if chunk.text:
                        bot_text += chunk.text
                        placeholder.markdown(bot_text + "▌")
                placeholder.markdown(bot_text)
            except Exception as e:
                bot_text = f"⚠️ Gemini error: {e}"
                placeholder.write(bot_text)

        st.session_state.history.append({"role": "assistant", "text": bot_text})

        # --- TTS reply ---
        if st.session_state.tts_enabled and bot_text and eleven_client:
            try:
                audio = eleven_client.text_to_speech.convert(
                    voice=voice_name, text=bot_text, model="eleven_multilingual_v2", language_code=lang_code
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(audio)
                    audio_path = tmp_file.name
                st.audio(audio_path, format="audio/mp3")
            except Exception as e:
                print(f"[ERROR] TTS failed, showing text only. ({e})")

elif tab_choice == "Chat History":
    st.subheader("📜 Previous Conversations")
    if not st.session_state.history:
        st.info("No chat history yet.")
    else:
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.markdown(f"**🧑 You:** {msg['text']}")
            else:
                st.markdown(f"**🤖 AgriBuddy:** {msg['text']}")
