# ============================================================
#  master_pipeline.py  —  Universal Agentic AI Data Analyst
#  Domains: Sales · Marketing · HR · Ecommerce · Retail · Generic
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dask.dataframe as dd
import sqlite3
import chardet

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main { background-color: #080c14; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.domain-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.badge-sales      { background:#00d4ff22; color:#00d4ff; border:1px solid #00d4ff44; }
.badge-marketing  { background:#ff6b6b22; color:#ff6b6b; border:1px solid #ff6b6b44; }
.badge-hr         { background:#a78bfa22; color:#a78bfa; border:1px solid #a78bfa44; }
.badge-ecommerce  { background:#34d39922; color:#34d399; border:1px solid #34d39944; }
.badge-retail     { background:#fbbf2422; color:#fbbf24; border:1px solid #fbbf2444; }
.badge-generic    { background:#94a3b822; color:#94a3b8; border:1px solid #94a3b844; }
.badge-fraud      { background:#ff6b6b22; color:#ff6b6b; border:1px solid #ff6b6b44; }

.section-header {
    background: linear-gradient(90deg, #ffffff0a, transparent);
    border-left: 3px solid #00d4ff;
    padding: 0.55rem 1rem;
    margin: 2rem 0 1rem 0;
    border-radius: 0 8px 8px 0;
    color: #e2e8f0;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.section-header.marketing { border-color: #ff6b6b; }
.section-header.hr        { border-color: #a78bfa; }
.section-header.ecommerce { border-color: #34d399; }
.section-header.retail    { border-color: #fbbf24; }

div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#111827,#1a2235);
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: border-color .2s;
}
div[data-testid="metric-container"]:hover { border-color: #00d4ff55; }
div[data-testid="metric-container"] label { color: #64748b !important; font-size:0.78rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important; font-family: 'Syne', sans-serif; font-size:1.6rem;
}

.insight-box {
    background: linear-gradient(135deg,#111827,#151f2e);
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
}
.insight-box strong { color: #00d4ff; }

.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; gap:4px; padding:4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b; font-family:'DM Sans'; }
.stTabs [aria-selected="true"] { background:#1e2d45 !important; color:#e2e8f0 !important; }

.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

DARK = "plotly_dark"
C_BLUE   = "#00d4ff"
C_GREEN  = "#34d399"
C_RED    = "#ff6b6b"
C_PURPLE = "#a78bfa"
C_YELLOW = "#fbbf24"
C_ORANGE = "#fb923c"

DOMAIN_COLOR = {
    "Sales":     C_BLUE,
    "Marketing": C_RED,
    "HR":        C_PURPLE,
    "Ecommerce": C_GREEN,
    "Retail":    C_YELLOW,
    "Generic":   "#94a3b8",
    "Fraud":     "#ff6b6b",
}

# ─── LLM ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")

def query_llm(prompt):
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        r = client.chat.completions.create(model="gpt-4",
            messages=[{"role":"user","content":prompt}])
        return r.choices[0].message.content
    elif LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        r = client.messages.create(model="claude-3-opus-20240229", max_tokens=600,
            messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    elif LLM_PROVIDER == "cohere":
        import cohere
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
        return co.chat(model="command-r", message=prompt).text
    else:
        from transformers import pipeline
        gen = pipeline("text-generation", model="sshleifer/tiny-gpt2")
        return gen(prompt, max_new_tokens=200)[0]["generated_text"]

# ─── Data Loader ─────────────────────────────────────────────────────────────
def load_data(uploaded_file):
    import io, tempfile, os
    fname = uploaded_file.name.lower()
    try:
        # Always read raw bytes first — avoids ALL UploadedFile compatibility issues
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        size_mb = len(raw_bytes) / 1024**2

        if fname.endswith((".csv", ".txt")):
            enc = chardet.detect(raw_bytes).get("encoding") or "utf-8"
            if enc.lower() in ["ascii", "utf-8", "utf8"]:
                enc = "utf-8"
            buf = io.BytesIO(raw_bytes)

            if size_mb > 100:
                # Dask needs a real file path — save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name
                try:
                    df = dd.read_csv(tmp_path, encoding=enc,
                                     assume_missing=True,
                                     dtype="object").compute()
                    # Convert numeric columns back after load
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors="ignore")
                        except Exception:
                            pass
                finally:
                    os.unlink(tmp_path)
                return df
            else:
                return pd.read_csv(buf, encoding=enc, low_memory=False)

        elif fname.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(raw_bytes))

        elif fname.endswith(".xml"):
            return pd.read_xml(io.BytesIO(raw_bytes))

        elif fname.endswith((".html", ".htm")):
            return pd.read_html(io.BytesIO(raw_bytes))[0]

        elif fname.endswith(".pdf"):
            import tabula
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                dfs = tabula.read_pdf(tmp_path, pages="all", multiple_tables=True)
            finally:
                os.unlink(tmp_path)
            return dfs[0] if dfs else None

        elif fname.endswith((".db", ".sqlite")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                conn = sqlite3.connect(tmp_path)
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table'", conn
                )
                if tables.empty:
                    conn.close()
                    return None
                first_table = tables["name"].iloc[0]
                df = pd.read_sql_query(f"SELECT * FROM '{first_table}'", conn)
                conn.close()
            finally:
                os.unlink(tmp_path)
            return df

        else:
            st.error(f"Unsupported file type: {fname}")
            return None

    except Exception as e:
        st.error(f"Load error: {e}")
        return None

# ─── Clean ────────────────────────────────────────────────────────────────────
def clean_data(df):
    df = df.drop_duplicates().ffill().bfill()
    for col in df.columns:
        if any(k in col.lower() for k in ["date","time","period","month","year"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df

# ─── Column Detector ─────────────────────────────────────────────────────────
def detect_columns(df):
    """4-pass column detector + aggressive numeric fallback."""
    import re
    def norm(s):
        return re.sub(r'[\s_\-]+', ' ', str(s).lower().strip())

    col_norm  = {norm(c): c for c in df.columns}
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    found     = {}

    # ── Synonym maps (priority-ordered, first match wins) ──────────────────
    maps = {
        "date":        ["order date","order_date","transaction date","sale date",
                        "invoice date","created date","purchase date","shipdate",
                        "ship date","date","period","fiscal date","trans date",
                        "order month","month"],
        "sales":       ["sales","revenue","total revenue","net revenue","gross revenue",
                        "total sales","net sales","gross sales","sale amount","total price",
                        "totalprice","order value","order total","invoice amount",
                        "transaction amount","total amount","amount","net amount",
                        "selling price","sales amount","turnover","income"],
        "profit":      ["profit","net profit","gross profit","operating profit",
                        "net income","earnings","ebit","ebitda","margin amount"],
        "product":     ["product name","productname","product title","product",
                        "item name","item","sku","product code","product id",
                        "goods","merchandise","article","product description"],
        "sub_category":["sub category","sub-category","subcategory",
                        "product sub category","sub cat"],
        "category":    ["category","product category","product group","product type",
                        "product class","brand","class","line"],
        "region":      ["region","territory","area","zone","market","geography"],
        "city":        ["city","town","municipality"],
        "state":       ["state","province","county"],
        "country":     ["country","nation","country code","country name"],
        "quantity":    ["quantity","qty","units sold","quantity sold",
                        "quantity ordered","units","order qty","items sold",
                        "volume","no of units","number of units","count"],
        "customer":    ["customer name","customer id","customerid","customer",
                        "client name","client id","client","buyer","consumer",
                        "account","contact","user"],
        "salary":      ["salary","annual salary","monthly salary","base salary",
                        "wage","compensation","pay","ctc","total compensation"],
        "spend":       ["ad spend","marketing spend","campaign cost","ad cost",
                        "marketing budget","media spend","advertising spend","spend"],
        "channel":     ["marketing channel","ad channel","traffic source",
                        "utm source","campaign type","acquisition channel",
                        "ad platform","media channel"],
        "distribution_channel": ["channel","distribution channel","sales channel",
                        "order channel","purchase channel","buying channel"],
        "discount":    ["discount","discount amount","discount %","discount percent",
                        "promo","rebate","coupon","offer","markdown"],
        "returns":     ["return status","returned","return","returns","refund",
                        "refunded","is returned"],
        "delivery":    ["delivery time","shipping days","days to ship","lead time",
                        "fulfillment time","days to deliver","shipping time",
                        "transit days","delivery days"],
        "payment":     ["payment method","payment type","payment mode","pay method",
                        "mode of payment","transaction type"],
        "store":       ["store name","store id","store","branch","outlet","shop",
                        "location","point of sale","pos","store location"],
        "department":  ["department","dept","division","team","function",
                        "business unit","org unit","cost center"],
        "gender":      ["gender","sex","employee gender"],
        "age":         ["age","age group","age band","employee age"],
        "tenure":      ["tenure","years of service","experience",
                        "years at company","seniority","service years"],
        "attrition":   ["attrition","left company","churn","resigned","turnover",
                        "exit","is churned","employee status","employment status"],
        "satisfaction":["satisfaction score","satisfaction","nps","csat",
                        "review score","feedback score","star rating","rating"],
        "impressions": ["impressions","impression","views","reach","page views"],
        "clicks":      ["clicks","click","click through","link clicks"],
        "conversions": ["conversions","conversion","leads","sign ups","signups",
                        "purchases","transactions","goals"],
        "roi":         ["roi","roas","return on investment","return on ad spend"],
        "price":       ["unit price","selling price","list price","mrp","price",
                        "rate","cost price","base price","retail price"],
        "cost":        ["unit cost","cost of goods","cogs","purchase price",
                        "product cost","manufacturing cost","cost"],
        "order_id":    ["order id","order no","orderid","transaction id",
                        "invoice id","receipt id","order number","invoice number"],
        "ship_mode":   ["ship mode","shipping mode","delivery mode",
                        "shipment type","ship method","shipping method"],
        "segment":     ["customer segment","market segment","customer type",
                        "business segment","user segment"],
        "profit_margin":["profit margin","margin %","gross margin","net margin"],
    }

    # Pass 1: exact normalised match
    for key, synonyms in maps.items():
        if key in found: continue
        for s in synonyms:
            if norm(s) in col_norm:
                found[key] = col_norm[norm(s)]
                break

    # Pass 2: contains match (col contains keyword OR keyword contains col)
    for key, synonyms in maps.items():
        if key in found: continue
        for col_n, col_orig in col_norm.items():
            for s in synonyms:
                ns = norm(s)
                # only match if both sides are at least 4 chars to avoid false positives
                if len(ns) >= 4 and len(col_n) >= 4:
                    if ns in col_n or col_n in ns:
                        found[key] = col_orig
                        break
            if key in found: break

    # Pass 3: aggressive numeric fallback — if sales/profit/qty still missing,
    # grab the first numeric column matching any keyword fragment
    num_fallback = {
        "sales":   ["sale","revenue","amount","total","price","value","turnover","income","gross"],
        "profit":  ["profit","earn","margin","net","income"],
        "quantity":["qty","quant","unit","count","volume","sold"],
        "discount":["disc","promo","rebate","off"],
        "salary":  ["salary","wage","pay","comp","ctc"],
        "spend":   ["spend","budget","cost","adcost"],
        "impressions":["impression","view","reach"],
        "clicks":  ["click"],
        "conversions":["conv","lead","goal"],
    }
    for key, kws in num_fallback.items():
        if key in found: continue
        for col in num_cols:
            cn = norm(col)
            if any(kw in cn for kw in kws):
                found[key] = col
                break

    # Pass 4: last-resort — if sales still not found, use largest-sum numeric column
    if "sales" not in found and num_cols:
        best_col = max(num_cols, key=lambda c: df[c].sum())
        found["sales"] = best_col

    # Pass 5: date fallback
    if "date" not in found:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                found["date"] = col; break
        if "date" not in found:
            for col in df.columns:
                cn = norm(col)
                if any(k in cn for k in ["date","time","month","year","period"]):
                    found["date"] = col; break

    return found


# ─── Fraud Dataset Detection ─────────────────────────────────────────────────
def is_fraud_dataset(df):
    cols_lower = [c.lower().strip() for c in df.columns]
    fraud_label_cols = ["class","label","fraud","is_fraud","isfraud","target","is fraud"]
    has_class = any(c in cols_lower for c in fraud_label_cols)
    if not has_class:
        return False
    v_cols = sum(1 for c in cols_lower if len(c) > 1 and c[0] == "v" and c[1:].isdigit())
    has_amount = "amount" in cols_lower
    has_time   = "time" in cols_lower
    for col in df.columns:
        if col.lower().strip() in fraud_label_cols:
            unique_vals = set(df[col].dropna().unique())
            is_binary = unique_vals.issubset({0,1,"0","1",True,False,"yes","no","fraud","legitimate"})
            if is_binary and (v_cols >= 3 or (has_amount and has_time)):
                return True
    return False


# ─── Domain Detection (SCORING-BASED) ────────────────────────────────────────
def detect_domain(df, found):
    """Score each domain; highest score wins."""
    if is_fraud_dataset(df):
        return "Fraud"

    keys = set(found.keys())
    h    = " ".join(df.columns).lower()
    scores = {"Sales": 0, "Marketing": 0, "HR": 0, "Ecommerce": 0, "Retail": 0}

    if "sales"   in keys: scores["Sales"] += 3
    if "profit"  in keys: scores["Sales"] += 3
    if "product" in keys: scores["Sales"] += 2
    if "quantity"in keys: scores["Sales"] += 2
    if "discount"in keys: scores["Sales"] += 1
    if "customer"in keys: scores["Sales"] += 1
    if any(k in h for k in ["order","invoice","sales","revenue","profit"]): scores["Sales"] += 2

    if "impressions" in keys: scores["Marketing"] += 4
    if "clicks"      in keys: scores["Marketing"] += 4
    if "conversions" in keys: scores["Marketing"] += 3
    if "channel"     in keys: scores["Marketing"] += 3
    if "spend"       in keys: scores["Marketing"] += 3
    if "roi"         in keys: scores["Marketing"] += 3
    if any(k in h for k in ["campaign","ctr","cpm","roas","ad spend","utm"]): scores["Marketing"] += 3
    if "distribution_channel" in keys and "impressions" not in keys and "clicks" not in keys:
        scores["Marketing"] -= 2

    if "salary"    in keys: scores["HR"] += 4
    if "attrition" in keys: scores["HR"] += 4
    if "department"in keys: scores["HR"] += 3
    if "tenure"    in keys: scores["HR"] += 3
    if "gender"    in keys: scores["HR"] += 2
    if "age"       in keys: scores["HR"] += 2
    if any(k in h for k in ["employee","headcount","appraisal","payroll","hire"]): scores["HR"] += 3

    if "delivery"in keys: scores["Ecommerce"] += 4
    if "returns" in keys: scores["Ecommerce"] += 4
    if "payment" in keys: scores["Ecommerce"] += 3
    if "customer"in keys: scores["Ecommerce"] += 1
    if any(k in h for k in ["cart","checkout","ecommerce","e-commerce","wishlist","tracking"]): scores["Ecommerce"] += 4

    if "store" in keys: scores["Retail"] += 5
    if any(k in h for k in ["store","retail","pos","point of sale","branch","outlet","franchise"]): scores["Retail"] += 4

    winner = max(scores, key=scores.get)
    if scores[winner] < 2:
        return "Generic"
    return winner

# ─── Domain Detection ────────────────────────────────────────────────────────
def detect_domain(df, found):
    h = " ".join(df.columns).lower()
    keys = set(found.keys())

    if "attrition" in keys or "salary" in keys or "department" in keys or \
       any(k in h for k in ["employee","headcount","tenure","appraisal"]):
        return "HR"

    if "impressions" in keys or "clicks" in keys or "channel" in keys or \
       any(k in h for k in ["campaign","ctr","cpm","roas","ad spend"]):
        return "Marketing"

    if "delivery" in keys or "returns" in keys or "payment" in keys or \
       any(k in h for k in ["customer segment","cart","checkout","order value","ecommerce","e-commerce"]):
        return "Ecommerce"

    if "store" in keys or any(k in h for k in ["store","retail","pos","point of sale","transaction","branch","outlet"]):
        return "Retail"

    if "sales" in keys or "profit" in keys or "product" in keys or \
       any(k in h for k in ["sales","revenue","profit","order"]):
        return "Sales"

    return "Generic"

# ─── Helpers ─────────────────────────────────────────────────────────────────
def section(title, domain="sales"):
    color_map = {"sales":C_BLUE,"marketing":C_RED,"hr":C_PURPLE,"ecommerce":C_GREEN,"retail":C_YELLOW,"generic":"#94a3b8"}
    color = color_map.get(domain.lower(), C_BLUE)
    st.markdown(
        f'<div class="section-header" style="border-color:{color}">{title}</div>',
        unsafe_allow_html=True
    )

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def chart_defaults(height=400):
    return dict(template=DARK, height=height,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#94a3b8"),
                margin=dict(t=50, b=40, l=10, r=10))

def top_n(df, group_col, value_col, n=10, agg="sum"):
    g = df.groupby(group_col)[value_col]
    result = g.sum() if agg == "sum" else g.mean()
    return result.sort_values(ascending=False).head(n).reset_index()

def prep_time(df, date_col, value_col):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["_M"] = d[date_col].dt.to_period("M").astype(str)
    d["_Q"] = d[date_col].dt.to_period("Q").astype(str)
    d["_Y"] = d[date_col].dt.year
    d["_DOW"] = d[date_col].dt.day_name()
    return d

# ─── KPI Cards ───────────────────────────────────────────────────────────────
def render_kpis(df, found, domain):
    accent = DOMAIN_COLOR.get(domain, C_BLUE)
    section(f"📌 Key Metrics — {domain}", domain.lower())
    kpis = []

    def add(label, val): kpis.append((label, val))

    if "sales" in found:
        s = df[found["sales"]].dropna()
        add("💰 Total Revenue", f"{s.sum():,.0f}")
        add("📈 Avg per Record", f"{s.mean():,.2f}")
        add("🔝 Peak Value", f"{s.max():,.0f}")

    if "profit" in found:
        p = df[found["profit"]].dropna()
        add("🏆 Total Profit", f"{p.sum():,.0f}")
        if "sales" in found:
            m = p.sum() / df[found["sales"]].sum() * 100
            add("📊 Profit Margin", f"{m:.1f}%")

    if "quantity" in found:
        add("📦 Units Sold", f"{df[found['quantity']].sum():,.0f}")

    if "customer" in found:
        add("👥 Unique Customers", f"{df[found['customer']].nunique():,}")

    if "product" in found:
        add("🛒 Products", f"{df[found['product']].nunique():,}")

    if "salary" in found:
        add("💼 Avg Salary", f"{df[found['salary']].mean():,.0f}")
        add("💼 Max Salary", f"{df[found['salary']].max():,.0f}")

    if "impressions" in found:
        add("👁 Total Impressions", f"{df[found['impressions']].sum():,.0f}")

    if "clicks" in found:
        add("🖱 Total Clicks", f"{df[found['clicks']].sum():,.0f}")

    if "conversions" in found:
        add("✅ Conversions", f"{df[found['conversions']].sum():,.0f}")

    if "returns" in found:
        ret_col = found["returns"]
        if df[ret_col].dtype == object:
            pct = (df[ret_col].str.lower().isin(["yes","true","1","returned"]).sum() / len(df)) * 100
            add("↩ Return Rate", f"{pct:.1f}%")

    if "store" in found:
        add("🏪 Stores", f"{df[found['store']].nunique():,}")

    if "discount" in found:
        add("🏷 Avg Discount", f"{df[found['discount']].mean():,.2f}")

    add("🗂 Records", f"{len(df):,}")

    per_row = 5
    for row_start in range(0, min(len(kpis), 10), per_row):
        row_kpis = kpis[row_start:row_start + per_row]
        cols = st.columns(len(row_kpis))
        for i, (label, value) in enumerate(row_kpis):
            with cols[i]:
                st.metric(label=label, value=value)

# ─── Shared Time Analysis ────────────────────────────────────────────────────
def render_time_analysis(df, found, domain, value_label="Sales"):
    if "date" not in found:
        return
    date_col = found["date"]
    value_col = found.get("sales") or found.get("revenue") or found.get("order_value")
    if not value_col:
        return

    accent = DOMAIN_COLOR.get(domain, C_BLUE)
    section(f"📅 Time Series — {value_label} Trends", domain.lower())

    d = prep_time(df, date_col, value_col)
    if d.empty:
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📆 Monthly", "📊 Quarterly", "📅 Yearly", "📅 Day of Week"])

    with tab1:
        monthly = d.groupby("_M")[value_col].sum().reset_index()
        monthly.columns = ["Month", value_label]
        fig = px.bar(monthly, x="Month", y=value_label, title=f"Monthly {value_label}",
                     color=value_label, color_continuous_scale="Blues", text_auto=".2s")
        fig.add_scatter(x=monthly["Month"], y=monthly[value_label],
                        mode="lines+markers", line=dict(color=accent, width=2.5), name="Trend")
        fig.update_layout(**chart_defaults(430), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if len(monthly) > 1:
            monthly["MoM %"] = monthly[value_label].pct_change() * 100
            fig2 = px.bar(monthly.dropna(), x="Month", y="MoM %",
                          title="Month-over-Month Growth %",
                          color="MoM %", color_continuous_scale=["#ff4444", accent],
                          text_auto=".1f")
            fig2.update_layout(**chart_defaults(340))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        qtr = d.groupby("_Q")[value_col].sum().reset_index()
        qtr.columns = ["Quarter", value_label]
        fig = px.bar(qtr, x="Quarter", y=value_label, title=f"Quarterly {value_label}",
                     color=value_label, color_continuous_scale="Teal", text_auto=".2s")
        fig.update_layout(**chart_defaults(420))
        st.plotly_chart(fig, use_container_width=True)

        profit_col = found.get("profit")
        if profit_col:
            qp = d.groupby("_Q")[profit_col].sum().reset_index()
            qp.columns = ["Quarter", "Profit"]
            merged = qtr.merge(qp, on="Quarter")
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_bar(x=merged["Quarter"], y=merged[value_label], name=value_label, marker_color=accent)
            fig2.add_scatter(x=merged["Quarter"], y=merged["Profit"], mode="lines+markers",
                             name="Profit", line=dict(color=C_RED, width=3), secondary_y=True)
            fig2.update_layout(title=f"Quarterly {value_label} vs Profit", **chart_defaults(380))
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        yearly = d.groupby("_Y")[value_col].sum().reset_index()
        yearly.columns = ["Year", value_label]
        fig = px.bar(yearly, x="Year", y=value_label, title=f"Yearly {value_label}",
                     color=value_label, color_continuous_scale="Viridis", text_auto=".2s")
        fig.update_layout(**chart_defaults(400))
        st.plotly_chart(fig, use_container_width=True)
        if len(yearly) > 1:
            yearly["YoY %"] = yearly[value_label].pct_change() * 100
            for _, row in yearly.dropna().iterrows():
                arrow = "🔺" if row["YoY %"] > 0 else "🔻"
                st.write(f"{arrow} **{int(row['Year'])}**: {row['YoY %']:.1f}% YoY growth")

    with tab4:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = d.groupby("_DOW")[value_col].sum().reindex(day_order).reset_index()
        dow.columns = ["Day", value_label]
        fig = px.bar(dow, x="Day", y=value_label, title=f"{value_label} by Day of Week",
                     color=value_label, color_continuous_scale="Sunset", text_auto=".2s")
        fig.update_layout(**chart_defaults(400))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            best_m = (d.groupby(["_M","_DOW"])[value_col].sum().reset_index()
                      .sort_values(value_col, ascending=False)
                      .groupby("_M").first().reset_index())
            best_m.columns = ["Month","Best Day",value_label]
            st.markdown("**🏆 Best Sales Day — by Month**")
            st.dataframe(best_m.style.format({value_label:"{:,.0f}"}), use_container_width=True)
        with c2:
            best_q = (d.groupby(["_Q","_DOW"])[value_col].sum().reset_index()
                      .sort_values(value_col, ascending=False)
                      .groupby("_Q").first().reset_index())
            best_q.columns = ["Quarter","Best Day",value_label]
            st.markdown("**🏆 Best Sales Day — by Quarter**")
            st.dataframe(best_q.style.format({value_label:"{:,.0f}"}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN: SALES
# ══════════════════════════════════════════════════════════════════════════════
def render_sales(df, found):
    d = "Sales"
    sales_col   = found.get("sales")
    profit_col  = found.get("profit")
    product_col = found.get("product")
    cat_col     = found.get("category")
    region_col  = found.get("region")
    qty_col     = found.get("quantity")
    cust_col    = found.get("customer")
    price_col   = found.get("price")

    render_time_analysis(df, found, d, "Sales")

    # ── Top N ──
    section("🏆 Product & Category Performance", d)
    c1, c2 = st.columns(2)
    if product_col and sales_col:
        with c1:
            t = top_n(df, product_col, sales_col, 10)
            t.columns = ["Product","Sales"]
            fig = px.bar(t, x="Sales", y="Product", orientation="h",
                         title="Top 10 Products by Sales", color="Sales",
                         color_continuous_scale="Blues", text_auto=".2s")
            fig.update_layout(**chart_defaults(450), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if profit_col:
                tp = top_n(df, product_col, profit_col, 5)
                tp.columns = ["Product","Profit"]
                fig = px.bar(tp, x="Profit", y="Product", orientation="h",
                             title="Top 5 Products by Profit", color="Profit",
                             color_continuous_scale="Greens", text_auto=".2s")
                fig.update_layout(**chart_defaults(380), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            elif qty_col:
                tq = top_n(df, product_col, qty_col, 5)
                tq.columns = ["Product","Units"]
                fig = px.bar(tq, x="Units", y="Product", orientation="h",
                             title="Top 5 Products by Units Sold", color="Units",
                             color_continuous_scale="Purples", text_auto=".2s")
                fig.update_layout(**chart_defaults(380), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    if cat_col and sales_col:
        with c3:
            cs = top_n(df, cat_col, sales_col, 5)
            cs.columns = ["Category","Sales"]
            fig = px.pie(cs, values="Sales", names="Category",
                         title="Top 5 Categories — Sales Share",
                         color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.42)
            fig.update_layout(**chart_defaults(400))
            st.plotly_chart(fig, use_container_width=True)
    if region_col and sales_col:
        with c4:
            rs = top_n(df, region_col, sales_col, 5)
            rs.columns = ["Region","Sales"]
            fig = px.funnel(rs, x="Sales", y="Region", title="Top 5 Regions by Sales",
                            color_discrete_sequence=[C_BLUE])
            fig.update_layout(**chart_defaults(400))
            st.plotly_chart(fig, use_container_width=True)

    if cust_col and sales_col:
        section("👥 Customer Analysis", d)
        tc = top_n(df, cust_col, sales_col, 5)
        tc.columns = ["Customer","Sales"]
        fig = px.bar(tc, x="Customer", y="Sales", title="Top 5 Customers by Revenue",
                     color="Sales", color_continuous_scale="Oranges", text_auto=".2s")
        fig.update_layout(**chart_defaults(380))
        st.plotly_chart(fig, use_container_width=True)

    # ── Profitability ──
    if profit_col and sales_col:
        section("💹 Profitability Analysis", d)
        df2 = df.copy()
        df2["_Margin%"] = (df2[profit_col] / df2[sales_col].replace(0, np.nan)) * 100
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df2.sample(min(800,len(df2))), x=sales_col, y=profit_col,
                             color="_Margin%", color_continuous_scale="RdYlGn",
                             title="Sales vs Profit", opacity=0.7,
                             hover_data=[product_col] if product_col else None)
            fig.update_layout(**chart_defaults(420))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if cat_col:
                cm = df2.groupby(cat_col)["_Margin%"].mean().sort_values(ascending=False).reset_index()
                cm.columns = ["Category","Avg Margin %"]
                fig = px.bar(cm, x="Category", y="Avg Margin %",
                             title="Profit Margin by Category",
                             color="Avg Margin %", color_continuous_scale="RdYlGn", text_auto=".1f")
                fig.update_layout(**chart_defaults(420))
                st.plotly_chart(fig, use_container_width=True)

    # ── Bottom 5 ──
    if product_col and sales_col:
        section("⚠️ Underperformers", d)
        bot = df.groupby(product_col)[sales_col].sum().sort_values().head(5).reset_index()
        bot.columns = ["Product","Sales"]
        fig = px.bar(bot, x="Sales", y="Product", orientation="h",
                     title="Bottom 5 Products — Lowest Sales",
                     color="Sales", color_continuous_scale="Reds", text_auto=".2s")
        fig.update_layout(**chart_defaults(340), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN: MARKETING
# ══════════════════════════════════════════════════════════════════════════════
def render_marketing(df, found):
    d = "Marketing"
    spend_col   = found.get("spend")
    revenue_col = found.get("sales")
    channel_col = found.get("channel")
    imp_col     = found.get("impressions")
    click_col   = found.get("clicks")
    conv_col    = found.get("conversions")
    roi_col     = found.get("roi")
    date_col    = found.get("date")
    cat_col     = found.get("category")

    render_time_analysis(df, found, d, "Revenue")

    # ── ROI & Spend ──
    section("💸 Spend vs Revenue & ROI", d)
    df2 = df.copy()

    if spend_col and revenue_col:
        df2["_ROI%"] = ((df2[revenue_col] - df2[spend_col]) / df2[spend_col].replace(0,np.nan)) * 100
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df2, x=spend_col, y=revenue_col, color="_ROI%",
                             color_continuous_scale="RdYlGn", title="Spend vs Revenue",
                             hover_data=[channel_col] if channel_col else None, opacity=0.75)
            fig.update_layout(**chart_defaults(420))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if channel_col:
                ch = df2.groupby(channel_col)["_ROI%"].mean().sort_values(ascending=False).reset_index()
                ch.columns = ["Channel","Avg ROI %"]
                fig = px.bar(ch, x="Channel", y="Avg ROI %", title="ROI by Channel",
                             color="Avg ROI %", color_continuous_scale="RdYlGn", text_auto=".1f")
                fig.update_layout(**chart_defaults(420))
                st.plotly_chart(fig, use_container_width=True)

        if channel_col:
            cs = df2.groupby(channel_col)[[spend_col, revenue_col]].sum().reset_index()
            fig = go.Figure()
            fig.add_bar(x=cs[channel_col], y=cs[spend_col], name="Spend", marker_color=C_RED)
            fig.add_bar(x=cs[channel_col], y=cs[revenue_col], name="Revenue", marker_color=C_GREEN)
            fig.update_layout(title="Spend vs Revenue by Channel", barmode="group",
                              **chart_defaults(400))
            st.plotly_chart(fig, use_container_width=True)

    # ── Funnel: Impressions → Clicks → Conversions ──
    if imp_col or click_col or conv_col:
        section("🔻 Campaign Funnel", d)
        funnel_vals, funnel_labels = [], []
        for col, label in [(imp_col,"Impressions"),(click_col,"Clicks"),(conv_col,"Conversions")]:
            if col:
                funnel_vals.append(df[col].sum())
                funnel_labels.append(label)

        if len(funnel_vals) >= 2:
            fig = go.Figure(go.Funnel(
                y=funnel_labels, x=funnel_vals,
                textinfo="value+percent initial",
                marker=dict(color=[C_BLUE, C_YELLOW, C_GREEN][:len(funnel_vals)])
            ))
            fig.update_layout(title="Campaign Conversion Funnel", **chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)

        # CTR & CVR
        c1, c2 = st.columns(2)
        if imp_col and click_col:
            with c1:
                df2["_CTR%"] = df2[click_col] / df2[imp_col].replace(0,np.nan) * 100
                if channel_col:
                    ctr = df2.groupby(channel_col)["_CTR%"].mean().sort_values(ascending=False).reset_index()
                    ctr.columns = ["Channel","CTR %"]
                    fig = px.bar(ctr, x="Channel", y="CTR %", title="CTR by Channel",
                                 color="CTR %", color_continuous_scale="Teal", text_auto=".2f")
                    fig.update_layout(**chart_defaults(380))
                    st.plotly_chart(fig, use_container_width=True)
        if click_col and conv_col:
            with c2:
                df2["_CVR%"] = df2[conv_col] / df2[click_col].replace(0,np.nan) * 100
                if channel_col:
                    cvr = df2.groupby(channel_col)["_CVR%"].mean().sort_values(ascending=False).reset_index()
                    cvr.columns = ["Channel","CVR %"]
                    fig = px.bar(cvr, x="Channel", y="CVR %", title="Conversion Rate by Channel",
                                 color="CVR %", color_continuous_scale="Purples", text_auto=".2f")
                    fig.update_layout(**chart_defaults(380))
                    st.plotly_chart(fig, use_container_width=True)

    # ── Top Channels ──
    if channel_col:
        section("📣 Channel Performance", d)
        metrics = [c for c in [revenue_col, spend_col, conv_col, imp_col] if c]
        if metrics:
            agg = df.groupby(channel_col)[metrics].sum().reset_index()
            st.dataframe(agg.style.format({c:"{:,.0f}" for c in metrics}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN: HR
# ══════════════════════════════════════════════════════════════════════════════
def render_hr(df, found):
    d = "HR"
    salary_col  = found.get("salary")
    dept_col    = found.get("department")
    attr_col    = found.get("attrition")
    gender_col  = found.get("gender")
    age_col     = found.get("age")
    tenure_col  = found.get("tenure")
    sat_col     = found.get("satisfaction")
    role_col    = found.get("product")  # sometimes "role" maps here

    # ── Salary ──
    if salary_col:
        section("💼 Salary Analysis", d)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=salary_col, nbins=35, title="Salary Distribution",
                               color_discrete_sequence=[C_PURPLE], marginal="box")
            fig.update_layout(**chart_defaults(400))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if dept_col:
                ds = df.groupby(dept_col)[salary_col].mean().sort_values(ascending=False).reset_index()
                ds.columns = ["Department","Avg Salary"]
                fig = px.bar(ds, x="Avg Salary", y="Department", orientation="h",
                             title="Avg Salary by Department", color="Avg Salary",
                             color_continuous_scale="Purples", text_auto=".0f")
                fig.update_layout(**chart_defaults(420), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        if dept_col:
            box = df.groupby(dept_col)[salary_col].describe().reset_index()
            st.markdown("**Salary Stats by Department**")
            st.dataframe(box.style.format({c:"{:,.0f}" for c in box.columns if box[c].dtype in [float,int] and c!=dept_col}),
                         use_container_width=True)

    # ── Attrition ──
    if attr_col:
        section("📉 Attrition Analysis", d)
        c1, c2 = st.columns(2)
        with c1:
            ac = df[attr_col].value_counts().reset_index()
            ac.columns = ["Status","Count"]
            fig = px.pie(ac, values="Count", names="Status", title="Overall Attrition",
                         hole=0.45, color_discrete_sequence=[C_RED, C_GREEN])
            fig.update_layout(**chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if dept_col:
                da = df.groupby([dept_col, attr_col]).size().reset_index(name="Count")
                fig = px.bar(da, x=dept_col, y="Count", color=attr_col, barmode="group",
                             title="Attrition by Department",
                             color_discrete_sequence=[C_RED, C_GREEN])
                fig.update_layout(**chart_defaults(380))
                st.plotly_chart(fig, use_container_width=True)

        if salary_col:
            fig = px.box(df, x=attr_col, y=salary_col, color=attr_col,
                         title="Salary Distribution — Attrition vs Retained",
                         color_discrete_sequence=[C_RED, C_GREEN])
            fig.update_layout(**chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)

    # ── Workforce Demographics ──
    section("👥 Workforce Demographics", d)
    c1, c2 = st.columns(2)
    if gender_col:
        with c1:
            gc = df[gender_col].value_counts().reset_index()
            gc.columns = ["Gender","Count"]
            fig = px.pie(gc, values="Count", names="Gender", title="Gender Distribution",
                         hole=0.4, color_discrete_sequence=[C_BLUE, C_PURPLE, C_GREEN])
            fig.update_layout(**chart_defaults(360))
            st.plotly_chart(fig, use_container_width=True)
    if age_col:
        with c2:
            fig = px.histogram(df, x=age_col, nbins=20, title="Age Distribution",
                               color_discrete_sequence=[C_PURPLE])
            fig.update_layout(**chart_defaults(360))
            st.plotly_chart(fig, use_container_width=True)

    if tenure_col:
        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(df, x=tenure_col, nbins=20, title="Tenure Distribution",
                               color_discrete_sequence=[C_BLUE])
            fig.update_layout(**chart_defaults(360))
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            if salary_col:
                fig = px.scatter(df.sample(min(500,len(df))), x=tenure_col, y=salary_col,
                                 color=dept_col if dept_col else None,
                                 title="Tenure vs Salary", opacity=0.7)
                fig.update_layout(**chart_defaults(360))
                st.plotly_chart(fig, use_container_width=True)

    # ── Satisfaction ──
    if sat_col:
        section("⭐ Satisfaction Analysis", d)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=sat_col, nbins=10, title="Satisfaction Score Distribution",
                               color_discrete_sequence=[C_YELLOW])
            fig.update_layout(**chart_defaults(360))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if dept_col:
                ds2 = df.groupby(dept_col)[sat_col].mean().sort_values(ascending=False).reset_index()
                ds2.columns = ["Department","Avg Satisfaction"]
                fig = px.bar(ds2, x="Department", y="Avg Satisfaction",
                             title="Avg Satisfaction by Department",
                             color="Avg Satisfaction", color_continuous_scale="RdYlGn",
                             text_auto=".2f")
                fig.update_layout(**chart_defaults(360))
                st.plotly_chart(fig, use_container_width=True)

    # ── Top Earners ──
    if salary_col and dept_col:
        section("🏆 Top 5 Earners per Department", d)
        top_earners = (df.groupby(dept_col)[salary_col]
                       .nlargest(5).reset_index(level=0).reset_index())
        top_earners = top_earners[[dept_col, salary_col]].rename(columns={salary_col:"Salary"})
        st.dataframe(top_earners.style.format({"Salary":"{:,.0f}"}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN: ECOMMERCE
# ══════════════════════════════════════════════════════════════════════════════
def render_ecommerce(df, found):
    d = "Ecommerce"
    sales_col    = found.get("sales")
    product_col  = found.get("product")
    cat_col      = found.get("category")
    cust_col     = found.get("customer")
    delivery_col = found.get("delivery")
    returns_col  = found.get("returns")
    payment_col  = found.get("payment")
    discount_col = found.get("discount")
    qty_col      = found.get("quantity")
    region_col   = found.get("region")
    profit_col   = found.get("profit")

    render_time_analysis(df, found, d, "Revenue")

    # ── Product & Category ──
    section("🛒 Product & Category Performance", d)
    c1, c2 = st.columns(2)
    if product_col and sales_col:
        with c1:
            t = top_n(df, product_col, sales_col, 10)
            t.columns = ["Product","Revenue"]
            fig = px.bar(t, x="Revenue", y="Product", orientation="h",
                         title="Top 10 Products by Revenue", color="Revenue",
                         color_continuous_scale="Greens", text_auto=".2s")
            fig.update_layout(**chart_defaults(460), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    if cat_col and sales_col:
        with c2:
            cs = top_n(df, cat_col, sales_col, 6)
            cs.columns = ["Category","Revenue"]
            fig = px.pie(cs, values="Revenue", names="Category",
                         title="Revenue Share by Category",
                         color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.4)
            fig.update_layout(**chart_defaults(420))
            st.plotly_chart(fig, use_container_width=True)

    # ── Customer Behaviour ──
    if cust_col and sales_col:
        section("👤 Customer Behaviour", d)
        c1, c2 = st.columns(2)
        with c1:
            tc = top_n(df, cust_col, sales_col, 5)
            tc.columns = ["Customer","Revenue"]
            fig = px.bar(tc, x="Revenue", y="Customer", orientation="h",
                         title="Top 5 Customers by Revenue", color="Revenue",
                         color_continuous_scale="Greens", text_auto=".2s")
            fig.update_layout(**chart_defaults(380), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            orders_per_cust = df.groupby(cust_col).size().reset_index(name="Orders")
            fig = px.histogram(orders_per_cust, x="Orders", nbins=30,
                               title="Orders per Customer Distribution",
                               color_discrete_sequence=[C_GREEN])
            fig.update_layout(**chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)

        # RFM-like table
        if "date" in found:
            df3 = df.copy()
            df3[found["date"]] = pd.to_datetime(df3[found["date"]], errors="coerce")
            ref = df3[found["date"]].max()
            rfm = df3.groupby(cust_col).agg(
                Recency=(found["date"], lambda x: (ref - x.max()).days),
                Frequency=(found["date"], "count"),
                Monetary=(sales_col, "sum")
            ).sort_values("Monetary", ascending=False).head(10).reset_index()
            st.markdown("**🏅 Top 10 Customers — RFM Summary**")
            st.dataframe(rfm.style.format({"Recency":"{:.0f}d","Frequency":"{:.0f}","Monetary":"{:,.0f}"}),
                         use_container_width=True)

    # ── Returns ──
    if returns_col:
        section("↩ Returns Analysis", d)
        c1, c2 = st.columns(2)
        with c1:
            rc = df[returns_col].value_counts().reset_index()
            rc.columns = ["Status","Count"]
            fig = px.pie(rc, values="Count", names="Status", title="Return Rate",
                         hole=0.45, color_discrete_sequence=[C_RED, C_GREEN])
            fig.update_layout(**chart_defaults(360))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if cat_col:
                ret_df = df.copy()
                ret_df["_ret"] = ret_df[returns_col].astype(str).str.lower().isin(["yes","true","1","returned"]).astype(int)
                ret_by_cat = ret_df.groupby(cat_col)["_ret"].mean().mul(100).sort_values(ascending=False).reset_index()
                ret_by_cat.columns = ["Category","Return Rate %"]
                fig = px.bar(ret_by_cat, x="Category", y="Return Rate %",
                             title="Return Rate by Category", color="Return Rate %",
                             color_continuous_scale="Reds", text_auto=".1f")
                fig.update_layout(**chart_defaults(360))
                st.plotly_chart(fig, use_container_width=True)

    # ── Payment & Delivery ──
    c1, c2 = st.columns(2)
    if payment_col:
        section("💳 Payment Methods", d)
        with c1:
            pm = df[payment_col].value_counts().reset_index()
            pm.columns = ["Method","Count"]
            fig = px.bar(pm, x="Method", y="Count", title="Payment Method Distribution",
                         color="Count", color_continuous_scale="Blues", text_auto=True)
            fig.update_layout(**chart_defaults(360))
            st.plotly_chart(fig, use_container_width=True)
    if delivery_col:
        with c2:
            fig = px.histogram(df, x=delivery_col, nbins=20, title="Delivery Time Distribution (days)",
                               color_discrete_sequence=[C_GREEN])
            fig.update_layout(**chart_defaults(360))
            st.plotly_chart(fig, use_container_width=True)
            avg_del = df[delivery_col].mean()
            insight(f"📦 <strong>Average delivery time:</strong> {avg_del:.1f} days | "
                    f"Fastest: {df[delivery_col].min():.0f}d | Slowest: {df[delivery_col].max():.0f}d")

    # ── Discount Impact ──
    if discount_col and sales_col:
        section("🏷 Discount Impact", d)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df.sample(min(600,len(df))), x=discount_col, y=sales_col,
                             title="Discount vs Revenue", color=sales_col,
                             color_continuous_scale="Greens", opacity=0.7)
            fig.update_layout(**chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x=discount_col, nbins=20, title="Discount Distribution",
                               color_discrete_sequence=[C_YELLOW])
            fig.update_layout(**chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN: RETAIL
# ══════════════════════════════════════════════════════════════════════════════
def render_retail(df, found):
    d = "Retail"
    sales_col   = found.get("sales")
    store_col   = found.get("store")
    product_col = found.get("product")
    cat_col     = found.get("category")
    qty_col     = found.get("quantity")
    profit_col  = found.get("profit")
    discount_col= found.get("discount")
    region_col  = found.get("region")
    price_col   = found.get("price")
    cost_col    = found.get("cost")
    cust_col    = found.get("customer")
    payment_col = found.get("payment")

    render_time_analysis(df, found, d, "Sales")

    # ── Store Performance ──
    if store_col and sales_col:
        section("🏪 Store Performance", d)
        c1, c2 = st.columns(2)
        with c1:
            ts = top_n(df, store_col, sales_col, 10)
            ts.columns = ["Store","Sales"]
            fig = px.bar(ts, x="Sales", y="Store", orientation="h",
                         title="Top 10 Stores by Sales", color="Sales",
                         color_continuous_scale="YlOrBr", text_auto=".2s")
            fig.update_layout(**chart_defaults(460), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            bs = df.groupby(store_col)[sales_col].sum().sort_values().head(5).reset_index()
            bs.columns = ["Store","Sales"]
            fig = px.bar(bs, x="Sales", y="Store", orientation="h",
                         title="Bottom 5 Stores by Sales", color="Sales",
                         color_continuous_scale="Reds", text_auto=".2s")
            fig.update_layout(**chart_defaults(380), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        if profit_col:
            sp = df.groupby(store_col)[[sales_col, profit_col]].sum().reset_index()
            sp["Margin%"] = sp[profit_col] / sp[sales_col] * 100
            fig = px.scatter(sp, x=sales_col, y=profit_col, size="Margin%",
                             hover_name=store_col, title="Store: Sales vs Profit",
                             color="Margin%", color_continuous_scale="RdYlGn")
            fig.update_layout(**chart_defaults(420))
            st.plotly_chart(fig, use_container_width=True)

    # ── Product & Category ──
    section("🛍 Product & Category Mix", d)
    c1, c2 = st.columns(2)
    if product_col and sales_col:
        with c1:
            tp = top_n(df, product_col, sales_col, 10)
            tp.columns = ["Product","Sales"]
            fig = px.bar(tp, x="Sales", y="Product", orientation="h",
                         title="Top 10 Products by Sales", color="Sales",
                         color_continuous_scale="YlOrBr", text_auto=".2s")
            fig.update_layout(**chart_defaults(460), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    if cat_col and sales_col:
        with c2:
            cs = top_n(df, cat_col, sales_col, 6)
            cs.columns = ["Category","Sales"]
            fig = px.pie(cs, values="Sales", names="Category",
                         title="Sales Share by Category",
                         color_discrete_sequence=px.colors.sequential.YlOrBr_r, hole=0.4)
            fig.update_layout(**chart_defaults(420))
            st.plotly_chart(fig, use_container_width=True)

    # ── Discount & Pricing ──
    if discount_col and sales_col:
        section("🏷 Discount Analysis", d)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df.sample(min(600,len(df))), x=discount_col, y=sales_col,
                             title="Discount vs Sales", color=sales_col,
                             color_continuous_scale="YlOrBr", opacity=0.7)
            fig.update_layout(**chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if cat_col:
                avg_disc = df.groupby(cat_col)[discount_col].mean().sort_values(ascending=False).reset_index()
                avg_disc.columns = ["Category","Avg Discount"]
                fig = px.bar(avg_disc, x="Category", y="Avg Discount",
                             title="Avg Discount by Category",
                             color="Avg Discount", color_continuous_scale="Reds", text_auto=".2f")
                fig.update_layout(**chart_defaults(380))
                st.plotly_chart(fig, use_container_width=True)

    # ── Region ──
    if region_col and sales_col:
        section("🌍 Regional Performance", d)
        rg = top_n(df, region_col, sales_col, 8)
        rg.columns = ["Region","Sales"]
        fig = px.funnel(rg, x="Sales", y="Region", title="Top Regions by Sales",
                        color_discrete_sequence=[C_YELLOW])
        fig.update_layout(**chart_defaults(420))
        st.plotly_chart(fig, use_container_width=True)

    # ── Payment ──
    if payment_col:
        section("💳 Payment Methods", d)
        pm = df[payment_col].value_counts().reset_index()
        pm.columns = ["Method","Count"]
        fig = px.bar(pm, x="Method", y="Count", title="Transactions by Payment Method",
                     color="Count", color_continuous_scale="YlOrBr", text_auto=True)
        fig.update_layout(**chart_defaults(360))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN: GENERIC
# ══════════════════════════════════════════════════════════════════════════════
def render_generic(df, found):
    d = "Generic"
    section("🔍 Auto Distribution Analysis", d)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(include=["object","category"]).columns.tolist()

    c1, c2 = st.columns(2)
    for i, col in enumerate(numeric_cols[:8]):
        with (c1 if i % 2 == 0 else c2):
            fig = px.histogram(df, x=col, nbins=40, title=f"{col} — Distribution",
                               color_discrete_sequence=[C_BLUE], marginal="box")
            fig.update_layout(**chart_defaults(340))
            st.plotly_chart(fig, use_container_width=True)

    for cat in cat_cols[:5]:
        if 2 <= df[cat].nunique() <= 30:
            counts = df[cat].value_counts().head(10).reset_index()
            counts.columns = [cat,"Count"]
            fig = px.bar(counts, x=cat, y="Count", title=f"Top Values — {cat}",
                         color="Count", color_continuous_scale="Blues", text_auto=True)
            fig.update_layout(**chart_defaults(340))
            st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 3:
        section("🔗 Correlation Heatmap", d)
        corr = df[numeric_cols[:12]].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Feature Correlation", text_auto=".2f")
        fig.update_layout(**chart_defaults(520))
        st.plotly_chart(fig, use_container_width=True)

    render_time_analysis(df, found, d, "Value")
    render_kpis(df, found, d)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN: FRAUD DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def render_fraud(df, found):
    d = "Fraud"
    C_FRAUD = "#ff6b6b"

    # Identify key columns
    cols_lower = {c.lower().strip(): c for c in df.columns}
    fraud_label_cols = ["class","label","fraud","is_fraud","isfraud","target","is fraud"]
    class_col = next((cols_lower[c] for c in fraud_label_cols if c in cols_lower), None)
    amount_col = cols_lower.get("amount")
    time_col   = cols_lower.get("time")
    v_cols     = [c for c in df.columns if c.lower().startswith("v") and c[1:].isdigit()]

    if class_col is None:
        st.warning("Could not find fraud label column (Class/Label/Fraud).")
        return

    df2 = df.copy()
    # Normalise class labels to 0/1
    df2["_label"] = df2[class_col].astype(str).str.lower().map(
        lambda x: 1 if x in ["1","true","yes","fraud"] else 0
    )
    total      = len(df2)
    fraud_ct   = df2["_label"].sum()
    legit_ct   = total - fraud_ct
    fraud_pct  = fraud_ct / total * 100

    # ── KPIs ──────────────────────────────────────────────────────────────
    section("🚨 Fraud Detection Overview", d)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📋 Total Transactions", f"{total:,}")
    c2.metric("✅ Legitimate",          f"{legit_ct:,}")
    c3.metric("🚨 Fraudulent",          f"{fraud_ct:,}")
    c4.metric("⚠️ Fraud Rate",          f"{fraud_pct:.3f}%")

    # ── Fraud vs Legitimate ───────────────────────────────────────────────
    section("📊 Fraud vs Legitimate Breakdown", d)
    c1, c2 = st.columns(2)
    with c1:
        pie_df = pd.DataFrame({"Status":["Legitimate","Fraud"],"Count":[legit_ct,fraud_ct]})
        fig = px.pie(pie_df, values="Count", names="Status",
                     title="Transaction Class Distribution",
                     color_discrete_sequence=[C_GREEN, C_RED], hole=0.45)
        fig.update_layout(**chart_defaults(380))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bar_df = pd.DataFrame({"Status":["Legitimate","Fraud"],"Count":[legit_ct,fraud_ct]})
        fig = px.bar(bar_df, x="Status", y="Count",
                     title="Transaction Count — Fraud vs Legitimate",
                     color="Status", color_discrete_map={"Legitimate":C_GREEN,"Fraud":C_RED},
                     text_auto=True)
        fig.update_layout(**chart_defaults(380))
        st.plotly_chart(fig, use_container_width=True)

    # ── Amount Analysis ───────────────────────────────────────────────────
    if amount_col:
        section("💳 Transaction Amount Analysis", d)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df2, x="_label", y=amount_col,
                         title="Amount Distribution: Legitimate (0) vs Fraud (1)",
                         color="_label",
                         color_discrete_map={0:C_GREEN, 1:C_RED},
                         labels={"_label":"Class","Amount":"Amount ($)"})
            fig.update_layout(**chart_defaults(400))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df2, x=amount_col, color="_label", nbins=50,
                               title="Amount Histogram by Class",
                               color_discrete_map={0:C_GREEN, 1:C_RED},
                               barmode="overlay", opacity=0.7,
                               labels={"_label":"0=Legit 1=Fraud"})
            fig.update_layout(**chart_defaults(400))
            st.plotly_chart(fig, use_container_width=True)

        # Stats comparison
        legit_amt  = df2[df2["_label"]==0][amount_col]
        fraud_amt  = df2[df2["_label"]==1][amount_col]
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**💚 Legitimate Transaction Amounts**")
            st.dataframe(legit_amt.describe().round(2).reset_index(), use_container_width=True, hide_index=True)
        with c4:
            st.markdown("**🔴 Fraudulent Transaction Amounts**")
            if len(fraud_amt) > 0:
                st.dataframe(fraud_amt.describe().round(2).reset_index(), use_container_width=True, hide_index=True)
            else:
                st.info("No fraud cases in this sample.")

    # ── Time Analysis ─────────────────────────────────────────────────────
    if time_col:
        section("⏱ Transaction Time Analysis", d)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df2, x=time_col, color="_label", nbins=48,
                               title="Transactions over Time (seconds)",
                               color_discrete_map={0:C_GREEN, 1:C_RED},
                               barmode="overlay", opacity=0.7,
                               labels={"_label":"0=Legit 1=Fraud"})
            fig.update_layout(**chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Hourly buckets (assuming Time is seconds from midnight)
            df2["_hour"] = (df2[time_col] / 3600).astype(int) % 24
            hourly = df2.groupby(["_hour","_label"]).size().reset_index(name="Count")
            fig = px.bar(hourly, x="_hour", y="Count", color="_label",
                         title="Transactions by Hour of Day",
                         color_discrete_map={0:C_GREEN, 1:C_RED},
                         barmode="group",
                         labels={"_hour":"Hour","_label":"0=Legit 1=Fraud"})
            fig.update_layout(**chart_defaults(380))
            st.plotly_chart(fig, use_container_width=True)

    # ── Feature Analysis (V1-V28) ─────────────────────────────────────────
    if v_cols:
        section("🔬 Feature Distribution (Top 6 PCA Features)", d)
        # Show only top 6 most discriminating features
        top_v = v_cols[:6]
        cols_grid = st.columns(2)
        for i, vcol in enumerate(top_v):
            with cols_grid[i % 2]:
                fig = px.box(df2, x="_label", y=vcol,
                             title=f"{vcol} — Fraud vs Legitimate",
                             color="_label",
                             color_discrete_map={0:C_GREEN, 1:C_RED},
                             labels={"_label":"0=Legit 1=Fraud"})
                fig.update_layout(**chart_defaults(320))
                st.plotly_chart(fig, use_container_width=True)

    # ── Correlation with fraud ────────────────────────────────────────────
    section("🔗 Feature Correlation with Fraud", d)
    num_cols_list = df2.select_dtypes(include="number").columns.tolist()
    if "_label" in num_cols_list:
        corr_with_fraud = (df2[num_cols_list].corr()["_label"]
                           .drop("_label").sort_values(key=abs, ascending=False).head(15))
        corr_df = corr_with_fraud.reset_index()
        corr_df.columns = ["Feature","Correlation with Fraud"]
        fig = px.bar(corr_df, x="Correlation with Fraud", y="Feature",
                     orientation="h",
                     title="Top 15 Features Most Correlated with Fraud",
                     color="Correlation with Fraud",
                     color_continuous_scale="RdBu_r",
                     text_auto=".3f")
        fig.update_layout(**chart_defaults(500), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Insight box ───────────────────────────────────────────────────────
    insight(f"🚨 <strong>Fraud Rate: {fraud_pct:.3f}%</strong> — "
            f"Out of <strong>{total:,}</strong> transactions, "
            f"<strong>{fraud_ct:,}</strong> are fraudulent. "
            + (f"Average fraudulent amount: <strong>${fraud_amt.mean():,.2f}</strong> vs "
               f"legitimate: <strong>${legit_amt.mean():,.2f}</strong>"
               if amount_col and len(df2[df2['_label']==1]) > 0 else ""))


# ══════════════════════════════════════════════════════════════════════════════
#  BOARDROOM SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def render_summary(df, found, domain):
    section(f"📋 Executive Summary — {domain}", domain.lower())
    lines = []
    sc = found.get("sales"); pc = found.get("profit"); prod = found.get("product")
    dc = found.get("date");  qty = found.get("quantity"); cust = found.get("customer")
    sat = found.get("satisfaction"); attr = found.get("attrition")
    imp = found.get("impressions"); clk = found.get("clicks"); conv = found.get("conversions")
    store = found.get("store"); salary = found.get("salary")

    if sc:
        lines += [f"- **Total Revenue:** {df[sc].sum():,.2f}",
                  f"- **Avg per Record:** {df[sc].mean():,.2f}",
                  f"- **Peak single record:** {df[sc].max():,.2f}"]
    if pc and sc:
        m = df[pc].sum() / df[sc].sum() * 100
        lines += [f"- **Total Profit:** {df[pc].sum():,.2f}",
                  f"- **Profit Margin:** {m:.1f}%"]
    if prod and sc:
        best = df.groupby(prod)[sc].sum().idxmax()
        lines.append(f"- **Best Product:** {best}")
    if cust:
        lines.append(f"- **Unique Customers:** {df[cust].nunique():,}")
    if qty:
        lines.append(f"- **Total Units Sold:** {df[qty].sum():,.0f}")
    if salary:
        lines += [f"- **Avg Salary:** {df[salary].mean():,.0f}",
                  f"- **Salary Range:** {df[salary].min():,.0f} – {df[salary].max():,.0f}"]
    if attr:
        rate = df[attr].astype(str).str.lower().isin(["yes","true","1"]).mean() * 100
        lines.append(f"- **Attrition Rate:** {rate:.1f}%")
    if sat:
        lines.append(f"- **Avg Satisfaction:** {df[sat].mean():.2f}")
    if imp:
        lines.append(f"- **Total Impressions:** {df[imp].sum():,.0f}")
    if clk and imp:
        ctr = df[clk].sum() / df[imp].sum() * 100
        lines.append(f"- **Overall CTR:** {ctr:.2f}%")
    if conv and clk:
        cvr = df[conv].sum() / df[clk].sum() * 100
        lines.append(f"- **Overall CVR:** {cvr:.2f}%")
    if store:
        lines.append(f"- **Stores Tracked:** {df[store].nunique():,}")
    if dc:
        ddc = pd.to_datetime(df[dc], errors="coerce").dropna()
        if len(ddc):
            lines.append(f"- **Date Range:** {ddc.min().date()} → {ddc.max().date()}")

    lines.append(f"- **Total Records:** {len(df):,} | **Domain:** {domain}")

    for l in lines:
        st.markdown(l)

# ══════════════════════════════════════════════════════════════════════════════
#  LLM Q&A
# ══════════════════════════════════════════════════════════════════════════════
def render_llm_qa(df, domain, found):
    section("🤖 AI-Powered Boardroom Q&A", domain.lower())
    tab_auto, tab_llm = st.tabs(["📊 Auto Insights (Always Works)", "🤖 LLM Q&A (API Key Required)"])

    with tab_auto:
        st.markdown("**Auto-computed boardroom insights from your actual data:**")
        qa = []
        sc   = found.get("sales");    pc   = found.get("profit")
        prod = found.get("product");  cat  = found.get("category")
        dc   = found.get("date");     qty  = found.get("quantity")
        cust = found.get("customer"); reg  = found.get("region")
        sal  = found.get("salary");   attr = found.get("attrition")
        imp  = found.get("impressions"); clk = found.get("clicks")
        conv = found.get("conversions"); sp = found.get("spend")
        store= found.get("store");    disc = found.get("discount")
        dept = found.get("department")

        if sc:
            total = df[sc].sum(); avg = df[sc].mean(); mx = df[sc].max()
            qa.append(("💰 What is our total revenue?",
                f"Total revenue is **{total:,.2f}**, average **{avg:,.2f}** per record, peak **{mx:,.2f}**."))

        if sc and dc:
            dfc = df.copy()
            dfc[dc] = pd.to_datetime(dfc[dc], errors="coerce")
            dfc = dfc.dropna(subset=[dc])
            if not dfc.empty:
                dfc["_M"] = dfc[dc].dt.to_period("M").astype(str)
                monthly = dfc.groupby("_M")[sc].sum()
                bm = monthly.idxmax(); bmv = monthly.max()
                wm = monthly.idxmin(); wmv = monthly.min()
                qa.append(("📅 Which month had the highest and lowest sales?",
                    f"Best: **{bm}** ({bmv:,.2f}) | Worst: **{wm}** ({wmv:,.2f})."))

        if sc and prod:
            tp = df.groupby(prod)[sc].sum()
            top3 = ", ".join([f"{p} ({v:,.0f})" for p,v in tp.nlargest(3).items()])
            bot3 = ", ".join([f"{p} ({v:,.0f})" for p,v in tp.nsmallest(3).items()])
            qa.append(("🛒 Top 3 and bottom 3 products by revenue?",
                f"**Top 3:** {top3}. **Bottom 3:** {bot3}."))

        if sc and pc:
            margin = df[pc].sum() / df[sc].sum() * 100
            qa.append(("📊 What is our overall profit margin?",
                f"Overall margin: **{margin:.1f}%** | Total profit: **{df[pc].sum():,.2f}**."))

        if sc and cat:
            tc = df.groupby(cat)[sc].sum().sort_values(ascending=False)
            top_cats = ", ".join([f"{k} ({v:,.0f})" for k,v in tc.head(3).items()])
            qa.append(("🗂 Which categories drive the most revenue?",
                f"Top categories: {top_cats}."))

        if sc and reg:
            tr = df.groupby(reg)[sc].sum().sort_values(ascending=False)
            qa.append(("🌍 Which region generates the highest revenue?",
                f"Top: **{tr.index[0]}** ({tr.iloc[0]:,.2f}) | Bottom: **{tr.index[-1]}** ({tr.iloc[-1]:,.2f})."))

        if sc and cust:
            tc2 = df.groupby(cust)[sc].sum().sort_values(ascending=False).head(5)
            qa.append(("👥 Who are our top 5 customers?",
                "**Top 5:** " + ", ".join([f"{c} ({v:,.0f})" for c,v in tc2.items()]) + "."))

        if qty:
            qa.append(("📦 How many total units sold?",
                f"Total: **{df[qty].sum():,.0f}** units | Avg per order: **{df[qty].mean():,.1f}**."))

        if disc and sc:
            corr = df[[disc,sc]].dropna().corr().iloc[0,1]
            direction = "Positive — higher discounts tend to increase order value." if corr > 0.1 else "Negative/weak — discounts do not strongly boost sales." if corr < -0.1 else "Neutral."
            qa.append(("🏷 Do discounts increase sales?",
                f"Discount–Sales correlation: **{corr:.2f}**. {direction}"))

        if sal:
            qa.append(("💼 What is our salary distribution?",
                f"Avg: **{df[sal].mean():,.0f}** | Median: **{df[sal].median():,.0f}** | Range: **{df[sal].min():,.0f}–{df[sal].max():,.0f}**."))

        if attr:
            rate = df[attr].astype(str).str.lower().isin(["yes","true","1"]).mean() * 100
            health = "⚠️ Above 15% benchmark." if rate > 15 else "✅ Within healthy range."
            qa.append(("📉 What is the attrition rate?",
                f"**{rate:.1f}%** of employees have left. {health}"))

        if imp and clk:
            ctr_val = df[clk].sum() / df[imp].sum() * 100
            health2 = "✅ Strong (>2%)." if ctr_val > 2 else "⚠️ Below 2% — consider creative refresh."
            qa.append(("🖱 What is our CTR?",
                f"CTR: **{ctr_val:.2f}%** ({df[clk].sum():,.0f} clicks / {df[imp].sum():,.0f} impressions). {health2}"))

        if clk and conv:
            cvr_val = df[conv].sum() / df[clk].sum() * 100
            qa.append(("✅ What is our conversion rate?",
                f"CVR: **{cvr_val:.2f}%** ({df[conv].sum():,.0f} conversions from {df[clk].sum():,.0f} clicks)."))

        if sp and sc:
            roi_val = (df[sc].sum() - df[sp].sum()) / df[sp].sum() * 100
            health3 = "✅ Positive ROI." if roi_val > 0 else "⚠️ Negative ROI — spend exceeds revenue."
            qa.append(("💸 What is our marketing ROI?",
                f"Spend: **{df[sp].sum():,.2f}** | Revenue: **{df[sc].sum():,.2f}** | ROI: **{roi_val:.1f}%**. {health3}"))

        if not qa:
            for col in df.select_dtypes(include="number").columns[:5]:
                s = df[col].dropna()
                qa.append((f"📈 {col} summary?",
                    f"Total: **{s.sum():,.2f}** | Mean: **{s.mean():,.2f}** | Range: **{s.min():,.2f}–{s.max():,.2f}**"))

        for i, (q, a) in enumerate(qa, 1):
            with st.expander(f"Q{i}: {q}", expanded=(i == 1)):
                st.markdown(f"**A{i}:** {a}")

    with tab_llm:
        if LLM_PROVIDER == "huggingface":
            st.info("💡 **tiny-gpt2 cannot answer questions** — it only generates random text.\n\n"
                    "Set `LLM_PROVIDER` in Streamlit Secrets to `openai`, `anthropic`, or `cohere` with the API key to enable real AI answers.")
        else:
            if st.button("🧠 Generate AI Boardroom Analysis"):
                with st.spinner("Querying AI..."):
                    num_cols = df.select_dtypes(include="number").columns.tolist()
                    ctx = df[num_cols].describe().round(2).to_string() if num_cols else "No numeric data."
                    prompt = (
                        f"You are a senior {domain} analyst. Dataset: {len(df):,} records.\n"
                        f"Columns: {', '.join(df.columns)}\nStats:\n{ctx}\n\n"
                        f"Write 5 data-backed {domain} boardroom Q&As citing actual numbers.\n"
                        f"Format: Q1: [question]\nA1: [answer]\n\nQ2: ..."
                    )
                    try:
                        st.markdown(query_llm(prompt))
                    except Exception as e:
                        st.error(f"LLM error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Agentic AI\n### Data Analyst")
        st.markdown("---")
        st.markdown("**🗂 Supported Formats**\nCSV · Excel · XML · HTML · PDF · SQLite")
        st.markdown("---")
        st.markdown("**🎯 Domains**\n💼 Sales\n📣 Marketing\n👥 HR\n🛒 Ecommerce\n🏪 Retail\n🔍 Generic")
        st.markdown("---")
        st.markdown("**🔍 Auto-Detects**\n30+ column types including Sales, Profit, Date, Product, Category, Region, Customer, Salary, Spend, Channel, Returns, Delivery, Payment, Store, Gender, Tenure, Satisfaction, CTR, CVR...")
        st.markdown("---")
        st.markdown(f"**LLM:** `{LLM_PROVIDER}`")

    # Header
    st.markdown("""
    <h1 style='font-size:2.2rem; margin-bottom:0.2rem'>
        📊 Universal Agentic AI Data Analyst
    </h1>
    <p style='color:#64748b; font-size:1rem; margin-top:0'>
        Upload any business dataset — Sales, Marketing, HR, Ecommerce, Retail or Generic.<br>
        Get instant domain-aware KPIs, smart charts, time trends, top/bottom analysis and AI insights.
    </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Drop your data file here",
        type=["csv","txt","xlsx","xls","xml","html","htm","pdf","db","sqlite"]
    )

    if not uploaded_file:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div class='insight-box'>
            <strong>💼 Sales</strong><br>Monthly/quarterly/yearly trends, top products, regions, customers, profit margin analysis
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class='insight-box'>
            <strong>📣 Marketing</strong><br>Spend vs revenue, ROI by channel, funnel (impressions→clicks→conversions), CTR, CVR
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class='insight-box'>
            <strong>👥 HR</strong><br>Salary distribution, attrition, department breakdown, demographics, tenure vs salary, satisfaction
            </div>""", unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown("""<div class='insight-box'>
            <strong>🛒 Ecommerce</strong><br>Product/category revenue, RFM customer analysis, returns, delivery time, payment methods, discounts
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown("""<div class='insight-box'>
            <strong>🏪 Retail</strong><br>Store performance, top/bottom stores, category mix, regional funnel, discount impact, payment breakdown
            </div>""", unsafe_allow_html=True)
        with c6:
            st.markdown("""<div class='insight-box'>
            <strong>🔍 Generic</strong><br>Auto distribution histograms, value counts, correlation heatmap for any dataset
            </div>""", unsafe_allow_html=True)
        return

    with st.spinner("⚙️ Loading and cleaning data..."):
        df = load_data(uploaded_file)

    if df is None or df.empty:
        st.error("❌ Failed to load. Please check file format.")
        return

    df     = clean_data(df)
    found  = detect_columns(df)
    domain = detect_domain(df, found)

    # ── Detection banner ──────────────────────────────────────────────────
    badge_class = f"badge-{domain.lower()}"
    st.markdown(
        f'<span class="domain-badge {badge_class}">🎯 {domain}</span> &nbsp; '
        f'<span style="color:#64748b;font-size:0.9rem">'
        f'{len(df):,} rows × {len(df.columns)} cols &nbsp;|&nbsp; '
        f'Detected: <code>{"</code> <code>".join(found.keys())}</code>'
        f'</span>', unsafe_allow_html=True
    )

    # ── Manual Override Panel ─────────────────────────────────────────────
    with st.expander("⚙️ Column & Domain Override (click if charts are missing)", expanded=False):
        st.markdown("**If the domain or columns were mis-detected, override them here:**")
        all_cols   = ["— (not mapped) —"] + list(df.columns)
        num_cols_l = ["— (not mapped) —"] + df.select_dtypes(include="number").columns.tolist()

        oc1, oc2 = st.columns(2)
        with oc1:
            domain = st.selectbox("🎯 Domain", ["Sales","Marketing","HR","Ecommerce","Retail","Generic"],
                                  index=["Sales","Marketing","HR","Ecommerce","Retail","Generic"].index(domain))
        with oc2:
            st.markdown(" ")

        st.markdown("**Map key columns** *(auto-detected values shown as defaults)*:")
        key_fields = {
            "sales":    ("💰 Revenue / Sales column",  num_cols_l),
            "profit":   ("🏆 Profit column",           num_cols_l),
            "quantity": ("📦 Quantity / Units column", num_cols_l),
            "discount": ("🏷 Discount column",         num_cols_l),
            "date":     ("📅 Date column",             all_cols),
            "product":  ("🛒 Product column",          all_cols),
            "category": ("🗂 Category column",         all_cols),
            "region":   ("🌍 Region column",           all_cols),
            "customer": ("👥 Customer column",         all_cols),
        }
        r1c = st.columns(3)
        for i, (key, (label, options)) in enumerate(key_fields.items()):
            current = found.get(key, "— (not mapped) —")
            default_idx = options.index(current) if current in options else 0
            chosen = r1c[i % 3].selectbox(label, options, index=default_idx, key=f"ovr_{key}")
            if chosen != "— (not mapped) —":
                found[key] = chosen
            elif key in found and chosen == "— (not mapped) —":
                del found[key]

        st.markdown("---")
        st.markdown("**All detected column mappings:**")
        det_df = pd.DataFrame({"Key": list(found.keys()), "Mapped to": list(found.values())})
        st.dataframe(det_df, use_container_width=True, hide_index=True)

        st.markdown("**All dataset columns & types:**")
        type_df = pd.DataFrame({
            "Column": df.columns,
            "Type":   [str(df[c].dtype) for c in df.columns],
            "Sample": [str(df[c].dropna().iloc[0]) if len(df[c].dropna()) > 0 else "" for c in df.columns],
            "Non-null": [df[c].notna().sum() for c in df.columns],
        })
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    # ── Data preview ──────────────────────────────────────────────────────
    with st.expander("🔍 Preview Raw Data"):
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Showing first 50 of {len(df):,} rows · {len(df.columns)} columns")

    # ── KPIs ──────────────────────────────────────────────────────────────
    render_kpis(df, found, domain)

    # ── Domain routing ────────────────────────────────────────────────────
    if domain == "Sales":
        render_sales(df, found)
    elif domain == "Marketing":
        render_marketing(df, found)
    elif domain == "HR":
        render_hr(df, found)
    elif domain == "Ecommerce":
        render_ecommerce(df, found)
    elif domain == "Retail":
        render_retail(df, found)
    elif domain == "Fraud":
        render_fraud(df, found)
    else:
        render_generic(df, found)

    render_summary(df, found, domain)
    render_llm_qa(df, domain, found)


if __name__ == "__main__":
    main()
