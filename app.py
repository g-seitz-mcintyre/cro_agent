import csv
import os
import uuid

import streamlit as st

from cro_agent import graph

LOGO_URL = "Cheil-Logo.svg"

# ────────────────────────────────────────────────────────────────────────────────
# Page & global config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Conversion Companion")

# ────────────────────────────────────────────────────────────────────────────────
# 1) Authentication gate (Google OIDC via Streamlit built‑in)
# ────────────────────────────────────────────────────────────────────────────────
if not getattr(st.user, "is_logged_in", False):
    st.image(LOGO_URL, width=160)
    st.title("Conversion Companion")
    st.markdown("Please sign in with your Google account to continue.")

    if st.button("Log in with Google"):
        st.login()

    st.stop()  # halt the script until the user returns from OAuth callback


# ────────────────────────────────────────────────────────────────────────────────
# 2) Sidebar — user info & logout
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(LOGO_URL, width=110)
    st.write(f"👤 **{st.user.name or st.user.email}**")
    if st.button("Log out"):
        st.logout()

# ────────────────────────────────────────────────────────────────────────────────
# 3) Main application logic (unchanged apart from auth guard)
# ────────────────────────────────────────────────────────────────────────────────

# after you detect a fresh login...
email = st.user.email
log_path = "signins.csv"


with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([email])

# Repeat branding in the main pane if desired
st.image(LOGO_URL, width=160)
st.title("Conversion Companion")

# one‑time per browser tab
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.thread_id = str(uuid.uuid4())  # critical for memories

# replay chat
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# chat input & LangGraph invocation
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
                use_container_width=True,
            )
            st.markdown(answer)
            st.session_state.history.append({"role": "assistant", "content": answer})
