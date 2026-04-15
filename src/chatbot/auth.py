"""
Lightweight session auth for Cricket Analytics.
Users and hashed passwords live in .streamlit/secrets.toml under [users.*].
No external auth library required — just bcrypt + Streamlit session state.

secrets.toml format:
  [users.alice]
  password_hash = "$2b$12$..."
  role          = "user"          # "user" | "admin"
  display_name  = "Alice"
"""

import streamlit as st

try:
    import bcrypt
    _BCRYPT_OK = True
except ImportError:
    _BCRYPT_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Secrets helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_users() -> dict:
    """Return {username: {password_hash, role, display_name}} from secrets."""
    try:
        raw = st.secrets.get("users", {})
        return {k: dict(v) for k, v in raw.items()}
    except Exception:
        return {}


def _verify_password(plain: str, hashed: str) -> bool:
    if not _BCRYPT_OK:
        return plain == hashed  # fallback (dev only)
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

_KEY_USER    = "_auth_user"
_KEY_ROLE    = "_auth_role"
_KEY_DISPLAY = "_auth_display"


def current_user() -> str | None:
    return st.session_state.get(_KEY_USER)


def current_role() -> str:
    return st.session_state.get(_KEY_ROLE, "user")


def display_name() -> str:
    return st.session_state.get(_KEY_DISPLAY, current_user() or "")


def is_admin() -> bool:
    return current_role() == "admin"


def is_logged_in() -> bool:
    return bool(current_user())


def logout() -> None:
    for k in (_KEY_USER, _KEY_ROLE, _KEY_DISPLAY):
        st.session_state.pop(k, None)
    # Clear per-user chat state too
    st.session_state.pop("gpt_chat", None)
    st.session_state.pop("gpt_conv_id", None)


# ─────────────────────────────────────────────────────────────────────────────
# Login widget
# ─────────────────────────────────────────────────────────────────────────────

def login_form() -> bool:
    """
    Render a centered login form. Returns True if the user just logged in
    successfully (caller should st.rerun()).  Call this in the main app
    when is_logged_in() is False.
    """
    st.markdown("""
    <style>
    .login-wrap {
        max-width: 380px;
        margin: 6rem auto 0;
        padding: 2.5rem 2rem;
        background: #fff;
        border: 2px solid #E2E8F4;
        border-radius: 8px;
        box-shadow: 0 4px 24px rgba(15,23,42,.08);
    }
    .login-wrap h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    .login-wrap p {
        font-size: .82rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    </style>
    <div class="login-wrap">
      <h2>🏏 Cricket Analytics</h2>
      <p>Sign in to continue</p>
    </div>
    """, unsafe_allow_html=True)

    users = _get_users()
    if not users:
        st.error(
            "No users configured. Add `[users.*]` entries to `.streamlit/secrets.toml`."
        )
        return False

    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username").strip().lower()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        user_cfg = users.get(username)
        if user_cfg and _verify_password(password, user_cfg.get("password_hash", "")):
            st.session_state[_KEY_USER]    = username
            st.session_state[_KEY_ROLE]    = user_cfg.get("role", "user")
            st.session_state[_KEY_DISPLAY] = user_cfg.get("display_name", username)
            return True
        elif submitted:
            st.error("Invalid username or password.")

    return False


def require_login() -> bool:
    """
    Call at the top of app pages. If not logged in, show login form and
    return False (caller should st.stop()). Returns True if authenticated.
    """
    if is_logged_in():
        return True
    logged_in = login_form()
    if logged_in:
        st.rerun()
    return False
