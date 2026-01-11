import logging

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src.inference import InferenceEngine

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Cook Compass")


@st.cache_resource
def get_inference_engine():
    return InferenceEngine.from_config()


inference_engine = get_inference_engine()

st.title("Cook Compass")
st.markdown("Beschreibung")

with st.sidebar:
    st.header("Settings")
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's cooking, good-looking?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = inference_engine.stream_response(
            [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
