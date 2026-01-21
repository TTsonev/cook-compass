import logging

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src.inference import InferenceEngine

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Cook Compass", page_icon="🧭")


@st.cache_resource
def get_inference_engine():
    return InferenceEngine.from_config()


inference_engine = get_inference_engine()

st.title("🧭 Cook Compass")

with st.sidebar:
    st.header("Settings")
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# 1. Automatic Greeting
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! I'm Cook Compass. 🧭\n\n"
                    "I can help you find recipes based on your ingredients. What would you like to cook today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 1:
    st.markdown("Try one of these to get started:")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🌱 Vegan Recipe"):
            st.session_state.messages.append({"role": "user", "content": "Suggest a vegan recipe."})
            st.rerun()

    with col2:
        if st.button("⏱️ 15 mins or less"):
            st.session_state.messages.append({"role": "user", "content": "What can I make in 15 minutes or less?"})
            st.rerun()

    with col3:
        if st.button("🥗 Healthy"):
            st.session_state.messages.append({"role": "user", "content": "Give me a healthy recipe."})
            st.rerun()

if prompt := st.chat_input("What's cooking?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        stream = inference_engine.stream_response(
            [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
