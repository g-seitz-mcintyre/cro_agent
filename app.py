# app.py
import uuid

import streamlit as st

from cro_agent import graph

# 🔗 public-domain PNG served by Wikimedia
LOGO_URL = "Cheil-Logo.svg"

st.set_page_config(page_title="Conversion Companion")

# ---------------  BRANDING  -------------- #
st.image(LOGO_URL, width=160)  # 👈 centred banner
st.title("Conversion Companion")
# ---------------------------------------- #

# one-time per browser tab
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.thread_id = str(uuid.uuid4())  # critical for memories

for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Which URL would you like to analyse?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):

            cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            resp = graph.invoke({"url": prompt}, config=cfg)
            answer = resp["cro_summary"]
            st.image(
                resp["screenshot_url"],
                caption="Page snapshot",
                use_container_width=True,  # fills the bubble nicely
            )
            st.markdown(answer)
            st.session_state.history.append({"role": "assistant", "content": answer})
