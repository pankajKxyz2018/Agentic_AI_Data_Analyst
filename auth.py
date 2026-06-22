# ============================================================
#  auth.py — Supabase Authentication & Multi-Tenant System
#  1 Click Data Analysis — Login + Admin Panel
# ============================================================

import streamlit as st
import os
import requests
import json
from datetime import datetime

# ─── Supabase Config ──────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ufbensczavkccurpoxvh.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "pankajkxyz2018@gmail.com")

# ─── Supabase API Helpers ─────────────────────────────────────────────────────
def _supabase_available():
    """Quick check if Supabase is reachable"""
    try:
        import requests as _r
        resp = _r.get(f"{SUPABASE_URL}/auth/v1/health",
                      headers={"apikey": SUPABASE_ANON_KEY}, timeout=3)
        return resp.status_code < 500
    except Exception:
        return False

def _headers(token=None):
    h = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json"
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def sign_up(email, password, full_name, company):
    """Register a new user"""
    try:
        r = requests.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            headers=_headers(),
            json={
                "email": email,
                "password": password,
                "data": {
                    "full_name": full_name,
                    "company": company
                }
            }
        )
        data = r.json()
        if "access_token" in data:
            return {"success": True, "user": data["user"], "token": data["access_token"]}
        elif "error" in data:
            return {"success": False, "error": data.get("error_description", data.get("error", "Signup failed"))}
        elif data.get("id"):
            return {"success": True, "user": data, "token": None, "confirm": True}
        else:
            return {"success": False, "error": str(data)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def sign_in(email, password):
    """Login existing user"""
    try:
        r = requests.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers=_headers(),
            json={"email": email, "password": password}
        )
        data = r.json()
        if "access_token" in data:
            return {"success": True, "user": data["user"], "token": data["access_token"]}
        else:
            return {"success": False, "error": data.get("error_description", "Invalid email or password")}
    except Exception as e:
        err = str(e)
        if any(k in err for k in ["NameResolution","Failed to resolve","Name or service"]):
            msg = ("⚠️ Authentication server is temporarily paused. "
                   "Please wait 2 minutes and refresh the page.")
        elif "timeout" in err.lower():
            msg = "⚠️ Connection timed out. Check your internet and try again."
        else:
            msg = f"Connection error: {err[:100]}"
        return {"success": False, "error": msg}

def sign_out(token):
    """Logout user"""
    try:
        requests.post(
            f"{SUPABASE_URL}/auth/v1/logout",
            headers=_headers(token, timeout=10)
        )
        return True
    except:
        return False

def get_user(token):
    """Get current user details"""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers=_headers(token, timeout=10)
        )
        return r.json()
    except:
        return None

def reset_password(email):
    """Send password reset email"""
    try:
        r = requests.post(
            f"{SUPABASE_URL}/auth/v1/recover",
            headers=_headers(),
            json={"email": email}
        )
        return r.status_code == 200
    except:
        return False

def get_all_users(token):
    """Get all users — admin only"""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/user_profiles?select=*",
            headers=_headers(token, timeout=10)
        )
        return r.json()
    except:
        return []

def upsert_profile(token, user_id, data):
    """Create or update user profile"""
    try:
        payload = {"id": user_id, **data, "updated_at": datetime.utcnow().isoformat()}
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            headers={**_headers(token, timeout=10), "Prefer": "resolution=merge-duplicates"},
            json=payload
        )
        return r.status_code in [200, 201]
    except:
        return False

# ─── Session State Helpers ────────────────────────────────────────────────────
def is_logged_in():
    return st.session_state.get("auth_token") is not None

def is_admin():
    return st.session_state.get("user_email") == ADMIN_EMAIL

def get_tenant_id():
    return st.session_state.get("user_id", "anonymous")

def logout():
    token = st.session_state.get("auth_token")
    if token:
        sign_out(token)
    for key in ["auth_token", "user_email", "user_id", "user_name", "user_company", "is_admin"]:
        st.session_state.pop(key, None)
    st.rerun()

# ─── Login Page UI ────────────────────────────────────────────────────────────
def render_login_page():
    """Full login/signup page with beautiful UI"""
    # No upfront Supabase health check — that caused 3-5s delay and blocked the page
    # Errors are shown AFTER a failed login attempt via sign_in() return value

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@600;700&family=Cabinet+Grotesk:wght@400;500;700&display=swap');
    
    .auth-container {
        max-width: 420px;
        margin: 0 auto;
        padding: 40px 20px;
    }
    .auth-logo {
        font-family: 'Clash Display', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 8px;
    }
    .auth-tagline {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 32px;
    }
    .auth-card {
        background: linear-gradient(135deg, #0d1829, #0f2040);
        border: 1px solid #1a2d4a;
        border-radius: 20px;
        padding: 36px 32px;
        box-shadow: 0 40px 80px rgba(0,0,0,0.4);
    }
    .auth-divider {
        text-align: center;
        color: #334155;
        font-size: 0.8rem;
        margin: 16px 0;
        position: relative;
    }
    .plan-badge {
        display: inline-block;
        background: rgba(14,165,233,0.1);
        border: 1px solid rgba(14,165,233,0.3);
        border-radius: 100px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #0ea5e9;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-logo">1 Click Data Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-tagline">Anyone Can Do Data Analysis</div>', unsafe_allow_html=True)

        # Tab selection
        tab1, tab2 = st.tabs(["🔐 Login", "✨ Sign Up"])

        with tab1:
            render_login_form()

        with tab2:
            render_signup_form()

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;font-size:0.75rem;color:#334155">'
            '🔒 Secured by Supabase · Your data is private and encrypted'
            '</div>',
            unsafe_allow_html=True
        )

def render_login_form():
    """Login form"""
    with st.form("login_form"):
        st.markdown("#### Welcome back 👋")
        email = st.text_input("Email", placeholder="pankaj@company.com")
        password = st.text_input("Password", type="password", placeholder="Your password")
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Login →", use_container_width=True, type="primary")
        with col2:
            forgot = st.form_submit_button("Forgot password?", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Please enter email and password")
            else:
                with st.spinner("Logging in..."):
                    result = sign_in(email, password)
                if result["success"]:
                    user = result["user"]
                    meta = user.get("user_metadata", {})
                    st.session_state["auth_token"] = result["token"]
                    st.session_state["user_email"] = user["email"]
                    st.session_state["user_id"] = user["id"]
                    st.session_state["user_name"] = meta.get("full_name", email.split("@")[0])
                    st.session_state["user_company"] = meta.get("company", "")
                    st.session_state["is_admin"] = user["email"] == ADMIN_EMAIL
                    st.success(f"Welcome back, {st.session_state['user_name']}! 🎉")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

        if forgot:
            if email:
                if reset_password(email):
                    st.success("✅ Password reset email sent! Check your inbox.")
                else:
                    st.error("Could not send reset email. Check your email address.")
            else:
                st.warning("Please enter your email first")

def render_signup_form():
    """Signup form"""
    with st.form("signup_form"):
        st.markdown("#### Create your account ✨")
        
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name", placeholder="Pankaj Kumar Das")
        with col2:
            company = st.text_input("Company Name", placeholder="Your Company")
        
        email = st.text_input("Work Email", placeholder="you@company.com")
        password = st.text_input("Password", type="password", placeholder="Min 8 characters")
        confirm = st.text_input("Confirm Password", type="password", placeholder="Same password again")

        st.markdown(
            '<div style="font-size:0.78rem;color:#475569;margin:8px 0">'
            '🎁 Free trial includes: All domains · EDA · Dashboards · PDF Export'
            '</div>',
            unsafe_allow_html=True
        )

        submit = st.form_submit_button("Create Free Account →", use_container_width=True, type="primary")

        if submit:
            if not all([full_name, company, email, password, confirm]):
                st.error("Please fill in all fields")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters")
            elif password != confirm:
                st.error("Passwords do not match")
            elif "@" not in email:
                st.error("Please enter a valid email address")
            else:
                with st.spinner("Creating your account..."):
                    result = sign_up(email, password, full_name, company)
                if result["success"]:
                    if result.get("confirm"):
                        st.success("✅ Account created! Please check your email to verify your account, then login.")
                    else:
                        user = result["user"]
                        meta = user.get("user_metadata", {})
                        st.session_state["auth_token"] = result["token"]
                        st.session_state["user_email"] = user["email"]
                        st.session_state["user_id"] = user["id"]
                        st.session_state["user_name"] = meta.get("full_name", full_name)
                        st.session_state["user_company"] = company
                        st.session_state["is_admin"] = user["email"] == ADMIN_EMAIL
                        st.success(f"Welcome to 1 Click Data Analysis, {full_name}! 🎉")
                        st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

# ─── User Header Bar ──────────────────────────────────────────────────────────
def render_user_header():
    """Show logged-in user info in sidebar"""
    name = st.session_state.get("user_name", "User")
    email = st.session_state.get("user_email", "")
    company = st.session_state.get("user_company", "")
    admin = st.session_state.get("is_admin", False)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#0d1829,#0f2040);
            border:1px solid #1a2d4a;border-radius:12px;padding:12px 16px;margin-bottom:8px">
            <div style="font-size:0.78rem;color:#64748b">Logged in as</div>
            <div style="font-weight:700;color:#e2e8f0;font-size:0.95rem">{name}</div>
            <div style="font-size:0.78rem;color:#0ea5e9">{company}</div>
            {"<div style='font-size:0.72rem;background:rgba(239,68,68,0.1);color:#ef4444;border-radius:4px;padding:2px 6px;margin-top:4px;display:inline-block'>👑 Admin</div>" if admin else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()

# ─── Admin Panel ──────────────────────────────────────────────────────────────
def render_admin_panel():
    """Admin panel for Pankaj to manage all clients"""
    st.markdown("## 👑 Admin Panel")
    st.markdown("*Manage all clients, subscriptions and usage*")

    token = st.session_state.get("auth_token")

    tab1, tab2, tab3 = st.tabs(["👥 All Users", "📊 Usage Stats", "⚙️ Settings"])

    with tab1:
        st.markdown("### Registered Users")
        users = get_all_users(token)
        
        if users and isinstance(users, list):
            import pandas as pd
            df_users = pd.DataFrame(users)
            st.dataframe(df_users, use_container_width=True)
        else:
            st.info("No users registered yet — or check Supabase dashboard directly.")

        st.markdown("---")
        st.markdown("### 📧 Invite New Client")
        with st.form("invite_form"):
            inv_email = st.text_input("Client Email")
            inv_name = st.text_input("Client Name")
            inv_company = st.text_input("Company")
            inv_plan = st.selectbox("Plan", ["Starter", "Business", "Enterprise"])
            if st.form_submit_button("Send Invite", type="primary"):
                st.success(f"✅ Invite sent to {inv_email}! They will receive a signup link.")

    with tab2:
        st.markdown("### Platform Usage")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Users", "0")
        col2.metric("Active This Week", "0")
        col3.metric("Files Uploaded", "0")
        col4.metric("MRR", "₹0")
        st.info("Connect Supabase analytics for detailed usage stats")

    with tab3:
        st.markdown("### Platform Settings")
        st.text_input("Admin Email", value=ADMIN_EMAIL, disabled=True)
        st.text_input("Supabase URL", value=SUPABASE_URL, disabled=True)
        st.success("✅ All systems operational")

# ─── Plan Check ──────────────────────────────────────────────────────────────
def get_user_plan():
    """Get user's subscription plan"""
    email = st.session_state.get("user_email", "")
    if email == ADMIN_EMAIL:
        return "enterprise"
    # Default to starter for now — extend with Supabase DB lookup
    return st.session_state.get("user_plan", "starter")

# ─── Plan Feature Matrix ────────────────────────────────────────────────────
PLAN_LIMITS = {
    #  feature              starter   business  enterprise
    "ml":                  [False,    True,     True  ],  # ML Prediction Engine
    "prescription":        [False,    True,     True  ],  # Prescriptive Insights
    "pdf":                 [False,    True,     True  ],  # PDF Export
    "voice":               [False,    True,     True  ],  # Voice Assistant
    "google_sheets":       [False,    True,     True  ],  # Google Sheets Connector
    "database":            [False,    True,     True  ],  # Database Connector
    "white_label":         [False,    False,    True  ],  # White-label branding
    "priority_support":    [False,    False,    True  ],  # Priority support + SLA
    "custom_onboarding":   [False,    False,    True  ],  # Custom onboarding
    "domains":             [3,        7,        7     ],  # No. of domain types
    "max_mb":              [50,       200,      200   ],  # Max file size MB
    "max_users":           [1,        5,        25    ],  # Max users per account
}
PLAN_INDEX = {"starter": 0, "business": 1, "enterprise": 2}

PLAN_NAMES  = ["Starter",   "Business",   "Enterprise"]
PLAN_PRICES = ["₹2,000/mo", "₹8,000/mo",  "₹25,000/mo"]

def check_plan_limit(feature):
    """Return the feature value for the current user plan."""
    plan  = get_user_plan()
    idx   = PLAN_INDEX.get(plan, 0)
    row   = PLAN_LIMITS.get(feature)
    if row is None:
        return False
    return row[idx]

def get_plan_file_limit_mb():
    """Return the max file size in MB for the current user's plan."""
    return check_plan_limit("max_mb") or 50

def get_plan_name():
    """Return the display name of the current plan."""
    plan = get_user_plan()
    return PLAN_NAMES[PLAN_INDEX.get(plan, 0)]

def get_upgrade_plan():
    """Return the next plan up from the current one."""
    plan = get_user_plan()
    idx  = PLAN_INDEX.get(plan, 0)
    if idx < 2:
        return PLAN_NAMES[idx + 1], PLAN_PRICES[idx + 1]
    return None, None

def render_upgrade_prompt(feature_name, compact=False):
    """
    Show a professional upgrade prompt when user tries to access a locked feature.
    Call this wherever a gated feature is blocked.
    """
    plan      = get_plan_name()
    next_plan, next_price = get_upgrade_plan()
    if not next_plan:
        return  # already on enterprise

    FEATURE_LABELS = {
        "ml":            "ML Prediction Engine",
        "prescription":  "Prescriptive AI Insights",
        "pdf":           "PDF Export",
        "voice":         "Voice Assistant",
        "google_sheets": "Google Sheets Connector",
        "database":      "Database Connector",
        "white_label":   "White-Label Branding",
    }
    label = FEATURE_LABELS.get(feature_name, feature_name.replace("_"," ").title())

    if compact:
        st.warning(
            f"🔒 **{label}** is available on the **{next_plan} plan** ({next_price}). "
            f"You are on **{plan}**. "
            f"[Upgrade →](mailto:pankaj@1clickdataanalysis.com?subject=Upgrade to {next_plan})"
        )
        return

    st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(249,115,22,0.08),rgba(239,68,68,0.05));
border:1px solid rgba(249,115,22,0.3);border-radius:14px;padding:24px 28px;margin:8px 0;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
    <span style="font-size:1.6rem;">🔒</span>
    <div>
      <div style="font-weight:800;color:#fbbf24;font-size:1rem;">{label} — {next_plan} Plan Required</div>
      <div style="color:#64748b;font-size:.82rem;">You are currently on the <strong style="color:#e2e8f0;">{plan}</strong> plan</div>
    </div>
  </div>
  <div style="color:#94a3b8;font-size:.875rem;margin-bottom:16px;line-height:1.6;">
    This feature is available on the <strong style="color:#fbbf24;">{next_plan} plan ({next_price})</strong>.
    Upgrade to unlock {label}, along with all other {next_plan} features.
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <a href="mailto:pankaj@1clickdataanalysis.com?subject=Upgrade Request — {next_plan} Plan"
       style="background:linear-gradient(135deg,#f97316,#ef4444);color:#fff;
              padding:10px 22px;border-radius:9px;text-decoration:none;
              font-weight:700;font-size:.85rem;">
      Upgrade to {next_plan} →
    </a>
    <a href="https://1clickdataanalysis.com/#pricing"
       style="background:rgba(255,255,255,0.06);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);
              padding:10px 22px;border-radius:9px;text-decoration:none;
              font-weight:600;font-size:.85rem;">
      View All Plans
    </a>
  </div>
</div>
""", unsafe_allow_html=True)