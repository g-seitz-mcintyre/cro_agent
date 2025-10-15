import uuid
from datetime import datetime, timezone

import streamlit as st
from supabase import Client, create_client

from cro_agent import graph

# ────────────────────────────────────────────────────────────────────────────────
# Page & global config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Conversion Companion")


# ────────────────────────────────────────────────────────────────────────────────
# Initialize Supabase client
# ────────────────────────────────────────────────────────────────────────────────
def get_supabase_client() -> Client:
    """Initialize and return Supabase client using credentials from secrets."""
    supabase_url = st.secrets.get("SUPABASE_URL")
    supabase_key = st.secrets.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        st.warning("Supabase credentials not found. User logging disabled.")
        return None

    return create_client(supabase_url, supabase_key)


def log_user_login(email: str, name: str = None):
    """Log user login to Supabase database."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return

        data = {
            "email": email,
            "name": name,
            "last_login": datetime.now(timezone.utc).isoformat(),
        }

        # Upsert: insert or update if email already exists
        supabase.table("user_logins").upsert(data, on_conflict="email").execute()

    except Exception as e:
        # Silently fail - don't break the app if logging fails
        print(f"Failed to log user: {e}")


def log_user_input(email: str, url: str):
    """Log user input (URL analysis request) to Supabase database."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return

        data = {
            "user_email": email,
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        supabase.table("user_inputs").insert(data).execute()

    except Exception as e:
        # Silently fail - don't break the app if logging fails
        print(f"Failed to log user input: {e}")


# ────────────────────────────────────────────────────────────────────────────────
# 1) Authentication gate (Google OIDC via Streamlit built‑in)
# ────────────────────────────────────────────────────────────────────────────────
if not getattr(st.user, "is_logged_in", False):
    st.title("Conversion Companion")
    st.markdown("Please sign in with your Google account to continue.")

    if st.button("Log in with Google"):
        st.login()

    st.stop()  # halt the script until the user returns from OAuth callback

# ────────────────────────────────────────────────────────────────────────────────
# Log user login to Supabase (only once per session)
# ────────────────────────────────────────────────────────────────────────────────
if "user_logged" not in st.session_state:
    log_user_login(st.user.email, st.user.name)
    st.session_state.user_logged = True

# ────────────────────────────────────────────────────────────────────────────────
# 2) Sidebar — user info & logout
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.write(f"👤 **{st.user.name or st.user.email}**")
    if st.button("Log out"):
        st.logout()

# ────────────────────────────────────────────────────────────────────────────────
# 3) Main application logic (unchanged apart from auth guard)
# ────────────────────────────────────────────────────────────────────────────────


# Repeat branding in the main pane if desired
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

    # Log user input to database
    log_user_input(st.user.email, prompt)

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
