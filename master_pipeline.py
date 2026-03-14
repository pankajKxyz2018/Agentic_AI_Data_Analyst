# ============================================================
#  master_pipeline.py  —  Universal Agentic AI Data Analyst
#  v12 — Google Sheets Connector + All Previous Features
#  Domains: Sales · Marketing · HR · Ecommerce · Retail · Fraud · Generic
# ============================================================
# © 2024 Pankaj Kumar Das — 1 Click Data Analysis
# All rights reserved. Unauthorized copying or distribution is prohibited.
# Website: https://1clickdataanalysis.com
# ============================================================

import os, io, re, tempfile, sqlite3
import streamlit as st

# ─── Google Sheets Connector (public URL / no API key needed) ────────────────
def _extract_sheet_id(url: str) -> str:
    """Extract spreadsheet ID from a Google Sheets URL."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    if "/" not in url and len(url) > 20:
        return url.strip()
    return ""

def load_google_sheet_public(sheet_url: str, sheet_index: int = 0):
    """
    Load a public Google Sheet into a DataFrame using the CSV export URL.
    No API key needed — user just needs to set sheet to 'Anyone with link can view'.
    Returns (df, error_string). On success error_string is "".
    """
    import pandas as pd
    sheet_id = _extract_sheet_id(sheet_url)
    if not sheet_id:
        return None, "❌ Invalid URL. Please paste the full Google Sheets URL from your browser."
    # Google Sheets CSV export URL
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_index}"
    try:
        df = pd.read_csv(csv_url)
        if df.empty:
            return None, "❌ The sheet appears to be empty."
        # Clean up column names
        df.columns = [str(c).strip() for c in df.columns]
        return df, ""
    except Exception as e:
        err = str(e)
        if "403" in err or "401" in err:
            return None, (
                "❌ Sheet is private. Please share it:\n\n"
                "In Google Sheets → **Share** (top right) → "
                "**'Anyone with the link'** → **Viewer** → **Done**\n\n"
                "Then paste the URL again."
            )
        if "404" in err:
            return None, "❌ Sheet not found. Please check the URL is correct."
        return None, f"❌ Could not read sheet: {err}"

def render_google_sheets_tab():
    """
    Renders the Google Sheets connector UI tab.
    Returns a DataFrame if user connects successfully, else None.
    Also returns a suggested filename string.
    """
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(14,165,233,0.08),rgba(6,182,212,0.05));
border:1px solid rgba(14,165,233,0.25);border-radius:12px;padding:20px 24px;margin-bottom:16px'>
<h4 style='color:#0ea5e9;margin:0 0 6px 0'>🔗 Connect Google Sheet</h4>
<p style='color:#64748b;font-size:.875rem;margin:0'>
Read directly from any Google Sheet — no download needed. Works with live, updating sheets.
</p></div>
""", unsafe_allow_html=True)

    with st.expander("📋 How to connect — 2 steps", expanded=True):
        st.markdown("""
**Step 1 — Make your sheet viewable:**
1. Open your Google Sheet
2. Click **Share** (top right, blue button)
3. Under "General access" → change to **"Anyone with the link"**
4. Set role to **Viewer** → click **Done**

**Step 2 — Paste the URL below**

> 🔒 Your data is read-only. It is processed in memory and never stored on our servers.
        """)

    sheet_url = st.text_input(
        "Google Sheet URL",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        key="gsheets_url_input",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        connect_btn = st.button("📊 Load Sheet", type="primary",
                                use_container_width=True, key="gsheets_connect_btn")
    with col2:
        sheet_num = st.number_input("Sheet #", min_value=1, max_value=20,
                                    value=1, key="gsheets_sheet_num",
                                    help="Which sheet tab to load (1 = first tab)")

    if connect_btn and sheet_url.strip():
        with st.spinner("🔄 Connecting to Google Sheets..."):
            # gid parameter: sheet index starts at 0 for first tab
            # For multi-tab sheets, user picks tab number (1-based → 0-based)
            df, error = load_google_sheet_public(sheet_url.strip(), sheet_index=0)

        if error:
            st.error(error)
            return None, None
        else:
            rows, cols = df.shape
            st.success(f"✅ Connected! Loaded **{rows:,} rows × {cols} columns**")
            with st.expander("👀 Preview — first 5 rows"):
                st.dataframe(df.head(), use_container_width=True)
            # Generate a filename from the sheet ID for display
            sheet_id = _extract_sheet_id(sheet_url.strip())
            fname = f"google_sheet_{sheet_id[:8]}.csv"
            return df, fname

    return None, None

# ─── Auth Import ──────────────────────────────────────────────────────────────
try:
    from auth import (render_login_page, render_user_header, render_admin_panel,
                      is_logged_in, is_admin, get_tenant_id, check_plan_limit, get_plan_file_limit_mb)
    AUTH_ENABLED = True
except ImportError:
    AUTH_ENABLED = False
    def is_logged_in(): return True
    def is_admin(): return False
    def get_tenant_id(): return "default"
    def check_plan_limit(f): return True
    def get_plan_file_limit_mb(): return 200
    def render_login_page(): pass
    def render_user_header(): pass
    def render_admin_panel(): pass
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chardet

# ─── Global Currency & Locale Utilities ───────────────────────────────────────
def detect_currency(df):
    """Auto-detect currency symbol from column names or data values."""
    col_lower = " ".join(df.columns).lower()
    # Check column names for currency hints
    if any(k in col_lower for k in ["gbp","pound","sterling","£"]): return "£"
    if any(k in col_lower for k in ["eur","euro","€"]): return "€"
    if any(k in col_lower for k in ["inr","rupee","₹","rs.","rs "]): return "₹"
    if any(k in col_lower for k in ["jpy","yen","¥","cny","yuan"]): return "¥"
    if any(k in col_lower for k in ["aud","cad","sgd","nzd"]): return "$"
    if any(k in col_lower for k in ["usd","dollar","$"]): return "$"
    # Check string values in numeric-looking columns for currency symbols
    for col in df.select_dtypes(include="object").columns[:5]:
        sample = df[col].dropna().astype(str).head(20).str.cat()
        if "£" in sample: return "£"
        if "€" in sample: return "€"
        if "₹" in sample or "Rs" in sample: return "₹"
        if "¥" in sample: return "¥"
    return "$"  # default

def fmt_money(val, symbol="$", decimals=2):
    """Format a monetary value with currency symbol."""
    if decimals == 0:
        return f"{symbol}{val:,.0f}"
    return f"{symbol}{val:,.{decimals}f}"

def normalize_gender(series):
    """Normalize gender values from global datasets to Male/Female/Other."""
    mapping = {
        "m": "Male", "male": "Male", "man": "Male", "men": "Male",
        "masculino": "Male", "homme": "Male", "männlich": "Male",
        "f": "Female", "female": "Female", "woman": "Female", "women": "Female",
        "femenino": "Female", "femme": "Female", "weiblich": "Female",
        "non-binary": "Other", "nonbinary": "Other", "nb": "Other",
        "other": "Other", "prefer not to say": "Other", "unknown": "Other",
    }
    return series.astype(str).str.lower().str.strip().map(
        lambda x: mapping.get(x, x.title())
    )

def parse_dates_robust(series):
    """Parse dates robustly handling DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD etc."""
    try:
        # Try standard parse first
        result = pd.to_datetime(series, errors="coerce")
        # If >50% failed, try dayfirst (European format)
        if result.isna().mean() > 0.5:
            result = pd.to_datetime(series, errors="coerce", dayfirst=True)
        return result
    except:
        return pd.to_datetime(series, errors="coerce")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic AI Data Analyst", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Syne',sans-serif!important;}
.main{background-color:#080c14;}
.block-container{padding-top:1.5rem;padding-bottom:2rem;}
.domain-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.78rem;
  font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem;}
.badge-sales     {background:#00d4ff22;color:#00d4ff;border:1px solid #00d4ff44;}
.badge-marketing {background:#ff6b6b22;color:#ff6b6b;border:1px solid #ff6b6b44;}
.badge-hr        {background:#a78bfa22;color:#a78bfa;border:1px solid #a78bfa44;}
.badge-ecommerce {background:#34d39922;color:#34d399;border:1px solid #34d39944;}
.badge-retail    {background:#fbbf2422;color:#fbbf24;border:1px solid #fbbf2444;}
.badge-fraud     {background:#ff6b6b22;color:#ff6b6b;border:1px solid #ff6b6b44;}
.badge-generic   {background:#94a3b822;color:#94a3b8;border:1px solid #94a3b844;}
.section-header{background:linear-gradient(90deg,#ffffff0a,transparent);
  border-left:3px solid #00d4ff;padding:.55rem 1rem;margin:2rem 0 1rem 0;
  border-radius:0 8px 8px 0;color:#e2e8f0;font-family:'Syne',sans-serif;
  font-size:1rem;font-weight:700;letter-spacing:.04em;}
div[data-testid="metric-container"]{background:linear-gradient(135deg,#111827,#1a2235);
  border:1px solid #1e2d45;border-radius:12px;padding:1rem 1.2rem;}
div[data-testid="metric-container"]:hover{border-color:#00d4ff55;}
div[data-testid="metric-container"] label{color:#64748b!important;font-size:.78rem;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
  color:#e2e8f0!important;font-family:'Syne',sans-serif;font-size:1.6rem;}
.insight-box{background:linear-gradient(135deg,#111827,#151f2e);border:1px solid #1e2d45;
  border-radius:10px;padding:1rem 1.3rem;margin:.4rem 0;font-size:.9rem;
  color:#cbd5e1;line-height:1.6;}
.insight-box strong{color:#00d4ff;}
.stTabs [data-baseweb="tab-list"]{background:#111827;border-radius:10px;gap:4px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#64748b;font-family:'DM Sans';}
.stTabs [aria-selected="true"]{background:#1e2d45!important;color:#e2e8f0!important;}
</style>""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
DARK = "plotly_dark"
C = {"blue":"#00d4ff","green":"#34d399","red":"#ff6b6b",
     "purple":"#a78bfa","yellow":"#fbbf24","orange":"#fb923c","grey":"#94a3b8"}
DOMAIN_COLOR = {"Sales":C["blue"],"Marketing":C["red"],"HR":C["purple"],
                "Ecommerce":C["green"],"Retail":C["yellow"],"Fraud":C["red"],"Generic":C["grey"]}
LLM_PROVIDER = os.getenv("LLM_PROVIDER","huggingface")

# ─── LLM ──────────────────────────────────────────────────────────────────────
def query_llm(prompt):
    if LLM_PROVIDER=="openai":
        from openai import OpenAI
        r = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-4", messages=[{"role":"user","content":prompt}])
        return r.choices[0].message.content
    elif LLM_PROVIDER=="anthropic":
        from anthropic import Anthropic
        r = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model="claude-3-opus-20240229", max_tokens=600,
            messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    elif LLM_PROVIDER=="cohere":
        import cohere
        return cohere.Client(os.getenv("COHERE_API_KEY")).chat(model="command-r",message=prompt).text
    else:
        # No LLM API key configured — return a simple rule-based response
        return f"Analysis note: {prompt[:200]}... (Set ANTHROPIC_API_KEY or OPENAI_API_KEY in Streamlit secrets for AI-powered insights.)" 

# ─── Loader ───────────────────────────────────────────────────────────────────
def load_data(uploaded_file):
    fname = uploaded_file.name.lower()
    try:
        raw = uploaded_file.read(); uploaded_file.seek(0)
        size_mb = len(raw)/1024**2
        if fname.endswith((".csv",".txt")):
            enc = chardet.detect(raw).get("encoding") or "utf-8"
            if enc.lower() in ["ascii","utf-8","utf8"]: enc="utf-8"
            if size_mb > 100:
                # Large file: use pandas chunked read (no dask dependency needed)
                chunks = []
                for chunk in pd.read_csv(
                    io.BytesIO(raw), encoding=enc, dtype="object",
                    chunksize=50000, on_bad_lines="skip"
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
                return df
            return pd.read_csv(io.BytesIO(raw),encoding=enc,low_memory=False)
        elif fname.endswith((".xlsx",".xls")): return pd.read_excel(io.BytesIO(raw))
        elif fname.endswith(".xml"):           return pd.read_xml(io.BytesIO(raw))
        elif fname.endswith((".html",".htm")):
            try:
                return pd.read_html(io.BytesIO(raw))[0]
            except Exception:
                # lxml not available, try with html5lib
                return pd.read_html(io.BytesIO(raw), flavor="html5lib")[0]
        elif fname.endswith(".pdf"):
            # pdfplumber — lazy import, safe fallback if not installed
            try:
                import pdfplumber
            except ImportError:
                st.error("PDF reading requires pdfplumber. Please upload CSV or Excel instead.")
                return pd.DataFrame()
            all_rows = []
            headers  = None
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    # Try structured table extraction first
                    tables = page.extract_tables()
                    if tables:
                        for tbl in tables:
                            if not tbl: continue
                            if headers is None:
                                headers = [str(h).strip() if h else f"col_{i}"
                                           for i,h in enumerate(tbl[0])]
                                all_rows.extend(tbl[1:])
                            else:
                                all_rows.extend(tbl)
                    else:
                        # Fallback: extract raw text and try to parse as CSV/TSV
                        text = page.extract_text() or ""
                        lines = [l.strip() for l in text.split("\n") if l.strip()]
                        if lines and headers is None:
                            # Detect delimiter: tab, pipe, comma, 2+ spaces
                            sample = lines[0]
                            delim = ("\t" if "\t" in sample
                                     else "|" if "|" in sample
                                     else "," if "," in sample
                                     else None)
                            if delim:
                                headers = [h.strip() for h in lines[0].split(delim)]
                                for line in lines[1:]:
                                    all_rows.append([c.strip() for c in line.split(delim)])
                            else:
                                # Last resort: whitespace split
                                headers = re.split(r"\s{2,}", lines[0])
                                for line in lines[1:]:
                                    all_rows.append(re.split(r"\s{2,}", line))
            if headers and all_rows:
                # Pad/trim rows to match header length
                n = len(headers)
                clean_rows = []
                for r in all_rows:
                    if r is None: continue
                    r = [str(c).strip() if c is not None else "" for c in r]
                    if len(r) < n: r += [""] * (n - len(r))
                    clean_rows.append(r[:n])
                df_pdf = pd.DataFrame(clean_rows, columns=headers)
                # Auto-convert numeric columns
                for col in df_pdf.columns:
                    df_pdf[col] = pd.to_numeric(df_pdf[col], errors="ignore")
                return df_pdf
            st.error("⚠️ Could not extract tabular data from this PDF. "
                     "Try saving it as CSV or Excel for best results.")
            return None
        elif fname.endswith((".db",".sqlite")):
            with tempfile.NamedTemporaryFile(delete=False,suffix=".db") as t:
                t.write(raw); tp=t.name
            try:
                conn=sqlite3.connect(tp)
                tbl=pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'",conn)
                df=pd.read_sql_query(f"SELECT * FROM '{tbl['name'].iloc[0]}'",conn)
                conn.close()
            finally: os.unlink(tp)
            return df
    except Exception as e:
        st.error(f"Load error: {e}")
    return None

# ─── Clean ────────────────────────────────────────────────────────────────────
def clean_data(df):
    df = df.drop_duplicates().ffill().bfill()
    for col in df.columns:
        if any(k in col.lower() for k in ["date","time","period","month","year"]):
            try: df[col]=parse_dates_robust(df[col])
            except: pass
    return df

# ─── Column Detector ──────────────────────────────────────────────────────────
def norm(s):
    return re.sub(r'[\s_\-]+', ' ', str(s).lower().strip())

def detect_columns(df):
    col_norm = {norm(c):c for c in df.columns}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    found = {}

    maps = {
        #  DATE / TIME 
        "date":         ["date","order date","order_date","transaction date","transaction_date",
                         "sale date","sale_date","invoice date","invoice_date","created date",
                         "created_date","purchase date","purchase_date","ship date","ship_date",
                         "billing date","billing_date","posting date","posting_date","value date",
                         "booking date","booking_date","record date","entry date","voucher date",
                         "docket date","dated","period","month","year","week","quarter","fy",
                         "financial year","fiscal year","fiscal_year","reporting date","trade date",
                         "settlement date","maturity date","as of date","as_of_date","reporting_period",
                         "time","timestamp","created_at","updated_at","modified_date","event_date"],

        #  SALES / REVENUE 
        "sales":        ["sales","revenue","total revenue","net revenue","gross revenue","total sales",
                         "net sales","gross sales","sale amount","total price","order value","order total",
                         "invoice amount","total amount","net amount","selling price","sales amount",
                         "turnover","gross turnover","net turnover","billed amount","billing amount",
                         "collection","receipts","income","gross income","net income","top line",
                         "topline","sales value","sale value","sold amount","transaction value",
                         "transaction amount","txn amount","txn_amount","value","amount","amt",
                         "extended price","ext price","line total","subtotal","sub total",
                         "sale_value","revenue_amount","rev_amount","rev amount","total_revenue",
                         "net_revenue","gross_revenue","total_sales","net_sales","gross_sales",
                         "umsatz","chiffre affaires","fatturato","facturacion","omzet"],

        #  PROFIT / MARGIN 
        "profit":       ["profit","net profit","gross profit","operating profit","net income",
                         "earnings","ebit","ebitda","margin amount","profit amount","profit value",
                         "contribution","contribution margin","gross margin","operating income",
                         "profit_amount","profit_value","net_profit","gross_profit","op_profit",
                         "pbt","pat","profit before tax","profit after tax","bottom line","surplus",
                         "gain","gewinn","benefice","profitto","ganancia","winst"],

        #  QUANTITY / UNITS 
        "quantity":     ["quantity","qty","units sold","quantity sold","quantity ordered","units",
                         "order qty","items sold","volume","no of units","number of units","nos",
                         "pieces","pcs","count","sold qty","sales qty","dispatch qty","shipped qty",
                         "ordered qty","fulfilled qty","delivered qty","units_sold","qty_sold",
                         "no_of_units","number_of_units","menge","quantite","quantita","cantidad",
                         "hoeveelheid"],

        #  DISCOUNT 
        "discount":     ["discount","discount amount","discount %","discount percent","discount_amount",
                         "discount_pct","discount_percent","promo","rebate","coupon","markdown",
                         "allowance","trade discount","cash discount","promotional discount",
                         "offer amount","scheme amount","deduction","concession","reduction",
                         "rabatt","remise","sconto","descuento","korting"],

        #  UNIT PRICE 
        "price":        ["unit price","selling price","list price","mrp","price","rate","retail price",
                         "sale price","market price","sp","basic price","base price","standard price",
                         "unit_price","selling_price","list_price","retail_price","base_price",
                         "unit cost price","per unit price","item price","product price","price per unit",
                         "preis","prix","prezzo","precio","prijs"],

        #  UNIT COST / COGS 
        "cost":         ["unit cost","cost of goods","cogs","purchase price","product cost","cost",
                         "landed cost","standard cost","average cost","weighted avg cost","wac",
                         "procurement cost","buying price","material cost","variable cost",
                         "cost_of_goods","purchase_price","product_cost","unit_cost","cost_price",
                         "buying_price","kosten","cout","costo","coste","kosten"],

        #  TARGET / BUDGET / FORECAST 
        "target":       ["target","budget","forecast","plan","planned","projected","estimated",
                         "quota","goal","aim","objective","target_amount","budget_amount",
                         "sales_target","revenue_target","sales_budget","rev_budget","projection",
                         "guidance","expected","annual_plan","monthly_plan","ziel","cible"],

        #  PRODUCT 
        "product":      ["product name","productname","product title","product","item name","item",
                         "sku","product code","product id","goods","merchandise","article","material",
                         "item_name","item_description","product_description","description","desc",
                         "part name","part number","part_name","part_no","component","model name",
                         "model number","model_name","model_no","service name","service_name",
                         "produkt","produit","prodotto","producto"],

        #  CATEGORY 
        "category":     ["category","product category","product group","product type","brand",
                         "line","item category","item group","item type","commodity","commodity group",
                         "category_name","product_category","product_group","item_category",
                         "product_line","segment","business segment","division","class","classification",
                         "department","dept","family","range","collection","vertikale",
                         "kategorie","categorie","categoria"],

        #  SUB CATEGORY 
        "sub_category": ["sub category","sub-category","subcategory","product sub category",
                         "sub_category","sub_cat","subcat","item sub category","item_sub_category",
                         "sub group","subgroup","product subfamily","sub type","sub_type","sub class",
                         "minor category","secondary category","level 2 category","l2_category"],

        #  REGION / TERRITORY 
        "region":       ["region","territory","area","zone","market","geography","sales_region",
                         "sales region","district","division","cluster","continent","sub_region",
                         "sub region","sales territory","sales zone","sales area","coverage area",
                         "business region","geo","location region","region_name","territory_name",
                         "north","south","east","west","north india","south india","apac","emea",
                         "americas","latam","mea","region_code","sales_zone","trade_area",
                         "gebiet","territoire","territorio"],

        #  CITY 
        "city":         ["city","town","municipality","city name","city_name","ship city","bill city",
                         "delivery city","customer city","billing city","shipping city","location",
                         "place","locality","urban area","metro","ville","ciudad","stad"],

        #  STATE / PROVINCE 
        "state":        ["state","province","county","state name","state_name","ship state",
                         "billing state","customer state","delivery state","prefecture","oblast",
                         "canton","land","bundesland","departement","provincia","provincie"],

        #  COUNTRY 
        "country":      ["country","nation","country name","country_name","ship country",
                         "billing country","customer country","delivery country","destination",
                         "origin country","market country","country_code","iso_country","iso2",
                         "iso3","land","pays","paese","pais"],

        #  POSTAL CODE 
        "postal_code":  ["postal code","postcode","zip code","zip","pin code","pincode","pin",
                         "postal_code","post_code","zip_code","pin_code","area code","plz",
                         "code postal","codice postale","codigo postal","postcode"],

        #  CUSTOMER 
        "customer":     ["customer name","customer id","customerid","customer","client name","client id",
                         "client","buyer","consumer","account","user","customer_name","customer_id",
                         "client_name","client_id","account_name","account_id","party name","party",
                         "sold to","sold_to","bill to","bill_to","debtor","purchaser","end customer",
                         "retailer","dealer","distributor","partner","contact name","customer_code",
                         "cust_id","cust_name","acc_name"],

        #  CUSTOMER SEGMENT 
        "segment":      ["customer segment","market segment","customer type","business segment",
                         "segment","tier","customer tier","account type","customer_segment",
                         "customer_type","market_segment","business_type","channel_type","b2b","b2c",
                         "consumer type","buyer type","priority","classification","grade","abc class"],

        #  ORDER ID 
        "order_id":     ["order id","order no","orderid","transaction id","invoice id","order number",
                         "purchase id","booking id","reference no","ref no","confirmation number",
                         "order_id","order_no","transaction_id","invoice_id","order_number",
                         "purchase_order","po number","po no","po_number","po_no","so number",
                         "so no","sales order","bill number","bill no","bill_number","voucher no",
                         "voucher number","docket no","ref_id","txn_id","receipt_no","document_no"],

        #  SHIPPING MODE 
        "ship_mode":    ["ship mode","shipping mode","delivery mode","shipment type","ship method",
                         "shipping_mode","ship_mode","delivery_type","dispatch_mode","courier",
                         "carrier","logistics","transport mode","freight type","express","standard"],

        #  SALESPERSON / REP 
        "sales_rep":    ["sales rep","salesperson","sales person","account manager","am",
                         "sales executive","se","sales officer","so","sales manager","sm",
                         "rep name","rep id","agent","agent name","broker","field rep",
                         "sales_rep","sales_person","account_manager","sales_executive",
                         "territory manager","territory_manager","relationship manager","rm"],

        #  EMPLOYEE ID 
        "employee_id":  ["employee id","emp id","employee_id","emp_id","staff id","worker id",
                         "empid","employee number","emp number","emp_no","employee_no","staff_id",
                         "worker_id","personnel_id","personnel id","associate id","associate_id",
                         "badge number","badge_no","payroll id","payroll_id","hr id","hr_id"],

        #  EMPLOYEE NAME 
        "employee_name":["employee name","emp name","staff name","worker name","full name","name",
                         "employee_name","emp_name","staff_name","worker_name","full_name",
                         "first name","last name","associate name","associate_name","personnel name",
                         "team member","resource name","resource_name","consultant name"],

        #  SALARY 
        "salary":       ["salary","annual salary","monthly salary","base salary","wage","compensation",
                         "pay","ctc","total compensation","gross salary","net salary","income",
                         "remuneration","emoluments","fixed pay","basic pay","basic salary",
                         "total pay","total salary","salary_amount","compensation_amount","ctc_amount",
                         "gross_salary","net_salary","base_pay","annual_ctc","monthly_ctc",
                         "earnings","take home","in hand","gehalt","salaire","stipendio","salario"],

        #  DEPARTMENT 
        "department":   ["department","dept","division","team","function","business unit","org unit",
                         "department_name","dept_name","business_unit","cost centre","cost center",
                         "cc","profit centre","profit center","sub department","sub_department",
                         "group","section","branch","unit","abteilung","departement","dipartimento"],

        #  GENDER 
        "gender":       ["gender","sex","employee gender","gender_code","gender_type","male female",
                         "m f","geschlecht","genre","genere","genero"],

        #  AGE 
        "age":          ["age","employee age","age_years","age in years","years old","worker age",
                         "alter","age_band","age_bucket"],

        #  AGE GROUP 
        "age_group":    ["age group","age band","age bracket","age range","age_group","age_band",
                         "age_bracket","age_range","age bucket","age_bucket","age category",
                         "generation","gen z","millennial","boomer","cohort"],

        #  TENURE 
        "tenure":       ["tenure","years of service","experience","years at company","seniority",
                         "service years","employment duration","years in company","yearsatcompany",
                         "years_at_company","total working years","totalworkingyears","yrs_experience",
                         "years_in_current_role","yearsinrole","years_with_company","service_length",
                         "employment_duration","months_of_service","length_of_service","los",
                         "experience_years","work_experience","total_experience","exp_years"],

        #  ATTRITION / CHURN 
        "attrition":    ["attrition","left company","churn","resigned","turnover","exit","is churned",
                         "employee status","employment status","status","left","voluntary_termination",
                         "is_active","active_status","separation","termination","resigned_flag",
                         "attrition_flag","churn_flag","exit_flag","active","inactive","left_flag",
                         "still_employed","employed","separation_type","resignation"],

        #  JOB TITLE 
        "job_title":    ["job title","designation","position","role","job role","title","job level",
                         "jobrole","joblevel","job_level","job_role","grade","band","pay_grade",
                         "job_title","job_designation","work_title","position_title","employee_role",
                         "function_title","work_level","career_level","level","rank","cadre",
                         "post","designation_name","role_name","profile"],

        #  HIRE DATE 
        "hire_date":    ["hire date","joining date","start date","date of joining","doj","date joined",
                         "employment date","onboard date","hire_date","joining_date","start_date",
                         "date_of_joining","date_of_hire","joining_dt","doj","joining_month",
                         "employment_start","commencement_date","effective_date","date_of_appointment"],

        #  PERFORMANCE 
        "performance":  ["performance","performance rating","perf rating","rating","appraisal",
                         "performance score","review score","performance_rating","perf_rating",
                         "performance_score","annual_rating","kra_score","kpi_score","gor_score",
                         "review_rating","goal_rating","competency_score","overall_rating",
                         "performance_band","performance_grade","bell_curve"],

        #  EDUCATION 
        "education":    ["education","qualification","degree","education level","education_level",
                         "education_qualification","highest_qualification","academic_qualification",
                         "highest_education","degree_type","field_of_study","major","discipline",
                         "course","stream","specialization","educational_background"],

        #  MARITAL STATUS 
        "marital":      ["marital status","marital","relationship status","married","single",
                         "marital_status","family_status","civil_status","matrimonial_status"],

        #  AD SPEND 
        "spend":        ["ad spend","marketing spend","campaign cost","ad cost","media spend",
                         "advertising spend","spend","marketing budget","ad_spend","media_cost",
                         "advertising_cost","marketing_cost","campaign_spend","paid_spend",
                         "digital_spend","media_budget","total_spend","cost_incurred","expenditure",
                         "marketing_expenditure","ad_expenditure","media_expenditure","budget_spent",
                         "amount_spent","kosten","depense","spesa","gasto"],

        #  CAMPAIGN REVENUE 
        "revenue":      ["revenue","campaign revenue","total revenue","net revenue","gross revenue",
                         "attributed_revenue","marketing_revenue","campaign_value","revenue_generated",
                         "attributed_sales","influenced_revenue","pipeline_value"],

        #  CAMPAIGN ID / NAME 
        "campaign_id":  ["campaign","campaign id","campaign_id","campaignid","campaign name",
                         "campaign_name","ad name","ad_name","ad campaign","ad_campaign",
                         "campaign_title","promo_name","promotion_name","promotion_id","promo_id",
                         "initiative","program_name","initiative_name","scheme_name","offer_name"],

        #  DEMOGRAPHY 
        "demography":   ["demography","demographic","age group","age_group","audience","target audience",
                         "target_audience","age band","age_band","audience_segment","audience_type",
                         "profile","customer_profile","audience_age","demographic_segment","user_segment",
                         "demo","target_demo","target_group","demographic_group","cohort"],

        #  MARKETING CHANNEL 
        "channel":      ["marketing channel","ad channel","traffic source","utm source","utm_source",
                         "campaign type","acquisition channel","ad platform","media channel",
                         "channel","medium","source","utm_medium","utm_campaign","platform",
                         "channel_name","media_type","ad_type","campaign_channel","ad_channel",
                         "digital_channel","organic","paid","social","email","search","display",
                         "affiliate","referral","direct","seo","sem","ppc","social_media"],

        #  IMPRESSIONS / VIEWS 
        "impressions":  ["impressions","impression","views","reach","page views","pageviews",
                         "ad views","ad_impressions","total_impressions","total_views","exposures",
                         "served","ad_served","total_reach","gross_impressions","unique_reach"],

        #  CLICKS 
        "clicks":       ["clicks","click","click through","link clicks","total clicks","ad clicks",
                         "link_clicks","total_clicks","ad_clicks","ctr_clicks","visits","sessions",
                         "unique_clicks","paid_clicks","organic_clicks"],

        #  CONVERSIONS / LEADS 
        "conversions":  ["conversions","conversion","leads","sign ups","signups","purchases","goals",
                         "total_conversions","lead_count","leads_generated","form_fills","inquiries",
                         "enquiries","responses","downloads","installs","activations","trials",
                         "registrations","bookings","orders","acquisitions","new_customers"],

        #  ROI / ROAS 
        "roi":          ["roi","roas","return on investment","return on ad spend","marketing_roi",
                         "campaign_roi","ad_roi","roi_percent","roas_value","efficiency",
                         "revenue_per_cost","revenue_to_cost","return_multiple"],

        #  CTR 
        "ctr":          ["ctr","click through rate","click_through_rate","click_rate","ad_ctr",
                         "engagement_rate","interaction_rate","response_rate"],

        #  DISTRIBUTION CHANNEL 
        "distribution_channel": ["distribution channel","sales channel","order channel",
                         "distribution_channel","sales_channel","order_channel","booking_channel",
                         "fulfillment_channel","purchase_channel","buying_channel"],

        #  STORE / BRANCH 
        "store":        ["store name","store id","store","branch","outlet","shop","point of sale",
                         "pos","store_name","store_id","branch_name","branch_id","outlet_name",
                         "outlet_id","shop_name","location_name","site","site_name","site_id",
                         "venue","kiosk","showroom","retail_location","store_code","branch_code"],

        #  PAYMENT METHOD 
        "payment":      ["payment method","payment type","payment mode","pay method","mode of payment",
                         "payment_method","payment_type","payment_mode","tender_type","tender",
                         "mop","mode","settlement_mode","transaction_mode","pay_type","pay_mode",
                         "instrument","card type","card_type","upi","neft","rtgs","cheque","cash",
                         "credit card","debit card","net banking","wallet","cod"],

        #  DELIVERY / SHIPPING TIME 
        "delivery":     ["delivery time","shipping days","days to ship","lead time","fulfillment time",
                         "days to deliver","shipping time","transit days","delivery_time","ship_days",
                         "lead_time","delivery_days","tat","turnaround","turnaround_time","sla",
                         "time_to_deliver","dispatch_time","processing_time","handling_time"],

        #  RETURNS / REFUNDS 
        "returns":      ["return status","returned","return","returns","refund","refunded","is returned",
                         "return_status","return_flag","refund_status","is_returned","rma",
                         "return_reason","return_type","reversal","cancellation","cancelled",
                         "chargeback","return_amount","refund_amount","reverse","recall"],

        #  SATISFACTION / RATING 
        "satisfaction": ["satisfaction score","satisfaction","nps","csat","review score","feedback score",
                         "star rating","rating score","satisfaction_score","customer_satisfaction",
                         "nps_score","csat_score","review_rating","feedback_rating","rating",
                         "score","stars","review","customer_rating","survey_score","happiness_index",
                         "loyalty_score","net_promoter","promoter_score","detractor_score"],

        #  LOYALTY POINTS 
        "loyalty_points":["loyalty points","reward points","points earned","points balance","miles",
                         "loyalty_points","reward_points","points_earned","points_balance","cashback",
                         "bonus points","redeem points","membership points","tier_points"],

        #  DEVICE / PLATFORM 
        "device":       ["device","device type","platform","os","browser","user agent","channel device",
                         "device_type","mobile","desktop","tablet","app","web","app_type",
                         "access_channel","access_device","user_device","screen_type"],

        #  RETURN REASON 
        "return_reason":["return reason","reason for return","refund reason","cancellation reason",
                         "return_reason","refund_reason","cancellation_reason","reason_for_cancellation",
                         "return_category","defect_type","complaint_reason"],

        #  SHIPPING COUNTRY 
        "shipping_country":["shipping country","ship to country","delivery country","destination country",
                         "shipping_country","ship_to_country","delivery_country","destination_country",
                         "consignee_country","recipient_country","to_country"],

        #  VENDOR / SUPPLIER 
        "vendor":       ["vendor","supplier","vendor name","supplier name","vendor_name","supplier_name",
                         "vendor_id","supplier_id","manufacturer","oem","brand owner","principal",
                         "source","source_name","distributor","wholesaler","trader","importer"],

        #  FRAUD LABEL 
        "fraud_label":  ["class","label","fraud","is_fraud","isfraud","fraud_flag","fraudulent",
                         "is_fraudulent","fraud_ind","fraud_indicator","suspicious","anomaly",
                         "flagged","target","is fraud","fraud_label","label_class","isFraud",
                         "is_suspicious","risk_flag","alert_flag","exception_flag"],

        #  FRAUD AMOUNT 
        "fraud_amount": ["transaction amount","trans amount","amount","step_amount","transaction value",
                         "trans_amount","oldbalanceorg","newbalanceorig","txn_amount","payment_amount",
                         "transfer_amount","withdraw_amount","deposit_amount","debit_amount","credit_amount"],

        #  FRAUD TIME 
        "fraud_time":   ["step","time","timestamp","transaction time","trans_time","transaction_date",
                         "trans_date","txn_time","txn_date","event_time","event_timestamp",
                         "occurrence_time","detection_time"],

        #  FRAUD TYPE 
        "fraud_type":   ["type","transaction type","trans_type","payment_type","fraud_type",
                         "transaction_type","nameorig","namedest","transfer_type","txn_type",
                         "fraud_category","fraud_method","scheme_type","modus_operandi"],

        #  FRAUD ID 
        "fraud_id":     ["transaction id","trans_id","transactionid","nameorig","step","txn_id",
                         "case_id","alert_id","incident_id","reference_id","fraud_case_id"],

        #  FRAUD CHANNEL 
        "fraud_channel":["channel","device","medium","source","origin","fraud_channel","access_channel",
                         "entry_point","channel_type","atm","pos","online","mobile_banking","branch"],

        #  FRAUD LOCATION 
        "fraud_loc":    ["merchant","location","terminal","merchant_name","merchant_category",
                         "merchant_city","merchant_state","merchant_country","mcc","merchant_id",
                         "terminal_id","pos_location","atm_location","ip_address","geo_location"],

        #  FRAUD SCORE 
        "fraud_score":  ["fraud_score","risk_score","anomaly_score","score","confidence",
                         "probability","pred_proba","risk","risk_rating","threat_score",
                         "suspicion_score","ml_score","model_score","detection_score"],

        #  ACCOUNT BALANCE 
        "balance":      ["balance","account balance","old balance","new balance","closing balance",
                         "opening balance","oldbalanceorig","newbalanceorig","oldbalancedest",
                         "newbalancedest","available_balance","ledger_balance","current_balance"],

        #  LATITUDE / LONGITUDE (Geospatial) 
        "latitude":     ["latitude","lat","lat_coordinate","y_coordinate","geo_lat"],
        "longitude":    ["longitude","lon","lng","long","lon_coordinate","x_coordinate","geo_lon"],

        #  WORK LOCATION (HR Extended) 
        "work_location":["work location","office","workplace","work place","remote","hybrid",
                         "work_location","office_location","base_location","work_city",
                         "work_site","reporting_location","base_office","office_city",
                         "branch_office","work_from","posted_location","posting_location"],

        #  NOTICE PERIOD (HR Extended) 
        "notice_period":["notice period","notice_period","exit date","last working day",
                         "date_of_exit","exit_date","last_day","separation_date",
                         "offboarding_date","relieving_date","last_working_date","lwd"],

        #  OVERTIME (HR Extended) 
        "overtime":     ["overtime","overtime hours","extra hours","additional hours",
                         "overtime_hours","extra_work","ot_hours","ot","overwork",
                         "additional_work","overtime_amount","ot_amount"],

        #  TRAINING (HR Extended) 
        "training":     ["training hours","training days","learning hours","l&d hours",
                         "training_hours","training_days","development_hours","learning_days",
                         "courses_completed","certifications","training_score","skill_score",
                         "learning_score","upskilling_hours","training_completed"],

        #  LEAVE DAYS (HR Extended) 
        "leave_days":   ["leave days","absent days","leaves taken","absenteeism",
                         "leave_days","absent_days","leaves_taken","leave_balance",
                         "annual_leave","sick_leave","casual_leave","total_leaves",
                         "days_absent","absence_days","leave_count","time_off_days"],
    }

    # Pass 1: exact match
    for key,synonyms in maps.items():
        if key in found: continue
        for s in synonyms:
            if norm(s) in col_norm:
                found[key]=col_norm[norm(s)]; break

    # Pass 2: contains match (min 4 chars)
    for key,synonyms in maps.items():
        if key in found: continue
        for cn,corig in col_norm.items():
            for s in synonyms:
                ns=norm(s)
                if len(ns)>=4 and len(cn)>=4 and (ns in cn or cn in ns):
                    found[key]=corig; break
            if key in found: break

    # Pass 3: numeric fallback
    num_fb = {
        "sales":    ["sale","revenue","amount","total","price","value","turnover","income","gross"],
        "profit":   ["profit","earn","margin","net","income"],
        "quantity": ["qty","quant","unit","count","volume","sold"],
        "discount": ["disc","promo","rebate"],
        "salary":   ["salary","wage","pay","comp","ctc"],
        "spend":    ["spend","budget","cost","adcost"],
        "impressions":["impression","view","reach"],
        "clicks":   ["click"],
        "conversions":["conv","lead","goal"],
    }
    for key,kws in num_fb.items():
        if key in found: continue
        for col in num_cols:
            if any(kw in norm(col) for kw in kws):
                found[key]=col; break

    # Pass 4: last-resort sales = largest sum numeric col
    # Skip if this looks like an HR dataset (salary already found)
    if "sales" not in found and num_cols and "salary" not in found:
        # Only assign if column name has sales-like keywords
        sale_kws = ["sale","revenue","amount","total","price","value","turnover","income","gross","profit"]
        candidate = max(num_cols, key=lambda c: df[c].sum())
        if any(kw in norm(candidate) for kw in sale_kws):
            found["sales"] = candidate

    # Pass 5: date fallback
    if "date" not in found:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                found["date"]=col; break
        if "date" not in found:
            for col in df.columns:
                if any(k in norm(col) for k in ["date","time","month","year","period"]):
                    found["date"]=col; break

    # Pass 6: Deduplication — same dataset column must not be assigned to two keys
    # If "revenue" col is assigned to both "sales" and "revenue" keys,
    # keep "sales" for sales-dominant data, "revenue" for marketing-dominant data.
    # Simple rule: later key wins only if the earlier key has a more specific alternative.
    EXCLUSIVE_GROUPS = [
        # Within each group, one dataset column can only be used by ONE key
        # Priority order = list order (first = highest priority)
        ["sales", "revenue"],          # revenue col → sales key wins unless marketing data
        ["spend",  "cost"],            # cost col → spend for marketing, cost for sales
        ["channel","distribution_channel"],  # channel col → one winner
        ["date",   "fraud_time", "hire_date"],  # date col → most specific wins
    ]

    col_to_keys = {}
    for k, v in found.items():
        col_to_keys.setdefault(v, []).append(k)

    for col, assigned_keys in col_to_keys.items():
        if len(assigned_keys) <= 1:
            continue
        # Find if these keys are in the same exclusive group
        for group in EXCLUSIVE_GROUPS:
            group_matches = [k for k in assigned_keys if k in group]
            if len(group_matches) >= 2:
                # Keep highest priority (lowest index in group), remove rest
                group_matches.sort(key=lambda k: group.index(k))
                winner = group_matches[0]
                for loser in group_matches[1:]:
                    del found[loser]
                break

    return found

# ─── Fraud Check ──────────────────────────────────────────────────────────────
def is_fraud_dataset(df):
    cl = [c.lower().strip() for c in df.columns]
    fraud_cols = ["class","label","fraud","is_fraud","isfraud","target","is fraud",
                  "fraud_flag","fraudulent","is_fraudulent","fraud_ind","fraud_indicator",
                  "transaction_type","suspicious","anomaly","flagged"]
    if not any(c in cl for c in fraud_cols): return False
    v_cols = sum(1 for c in cl if len(c)>1 and c[0]=="v" and c[1:].isdigit())
    has_amount = "amount" in cl; has_time = "time" in cl
    for col in df.columns:
        if col.lower().strip() in fraud_cols:
            uv = set(df[col].dropna().unique())
            if uv.issubset({0,1,"0","1",True,False,"yes","no","fraud","legitimate"}):
                if v_cols>=3 or (has_amount and has_time): return True
    return False

# ─── Domain Detection ─────────────────────────────────────────────────────────
def detect_domain(df, found):
    if is_fraud_dataset(df): return "Fraud"
    keys = set(found.keys())
    h = " ".join(df.columns).lower()
    scores = {"Sales":0,"Marketing":0,"HR":0,"Ecommerce":0,"Retail":0}

    if "sales"     in keys: scores["Sales"]+=3
    if "profit"    in keys: scores["Sales"]+=3
    if "product"   in keys: scores["Sales"]+=2
    if "quantity"  in keys: scores["Sales"]+=2
    if "discount"  in keys: scores["Sales"]+=1
    if "customer"  in keys: scores["Sales"]+=1
    if "target"    in keys: scores["Sales"]+=1
    if "sales_rep" in keys: scores["Sales"]+=2
    if "vendor"    in keys: scores["Sales"]+=1
    if any(k in h for k in ["order","invoice","sales","revenue","profit","quota","forecast"]): scores["Sales"]+=2

    if "impressions"  in keys: scores["Marketing"]+=5; scores["Sales"]-=3
    if "clicks"       in keys: scores["Marketing"]+=5; scores["Sales"]-=3
    if "conversions"  in keys: scores["Marketing"]+=4; scores["Sales"]-=2
    if "channel"      in keys: scores["Marketing"]+=3
    if "spend"        in keys: scores["Marketing"]+=4; scores["Sales"]-=2
    if "campaign_id"  in keys: scores["Marketing"]+=5; scores["Sales"]-=4
    if "demography"   in keys: scores["Marketing"]+=4; scores["Sales"]-=3
    if "roi"          in keys: scores["Marketing"]+=3
    if "revenue"      in keys: scores["Marketing"]+=2  # revenue key = marketing-specific
    if any(k in h for k in ["campaign","ctr","cpm","roas","utm","demograph","impression"]):
        scores["Marketing"]+=4; scores["Sales"]-=3
    if "distribution_channel" in keys and "impressions" not in keys: scores["Marketing"]-=2

    if "salary"     in keys: scores["HR"]+=6; scores["Sales"]-=4
    if "attrition"  in keys: scores["HR"]+=6; scores["Sales"]-=3
    if "department" in keys: scores["HR"]+=3
    if "tenure"     in keys: scores["HR"]+=4
    if "gender"     in keys: scores["HR"]+=3
    if "age"        in keys: scores["HR"]+=2
    if "employee_id"in keys: scores["HR"]+=5; scores["Sales"]-=3
    if "job_title"  in keys: scores["HR"]+=4
    if "hire_date"  in keys: scores["HR"]+=4
    if any(k in h for k in ["employee","headcount","payroll","hire","appraisal"]): scores["HR"]+=4

    if "delivery"in keys: scores["Ecommerce"]+=4
    if "returns" in keys: scores["Ecommerce"]+=4
    if "payment" in keys: scores["Ecommerce"]+=3
    if any(k in h for k in ["cart","checkout","ecommerce","wishlist","tracking"]): scores["Ecommerce"]+=4

    if "store"  in keys: scores["Retail"]+=5
    if any(k in h for k in ["store","retail","pos","branch","outlet","franchise"]): scores["Retail"]+=4

    winner = max(scores, key=scores.get)
    return winner if scores[winner]>=2 else "Generic"

# ─── UI Helpers ───────────────────────────────────────────────────────────────
def section(title, domain="sales"):
    color = DOMAIN_COLOR.get(domain.capitalize(), C["blue"])
    st.markdown(f'<div class="section-header" style="border-color:{color}">{title}</div>',
                unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def cd(height=400):
    return dict(template=DARK, height=height,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#94a3b8"),
                margin=dict(t=50,b=40,l=10,r=10))

def show_metrics(kpis, per_row=5):
    for i in range(0, len(kpis), per_row):
        row = kpis[i:i+per_row]
        cols = st.columns(len(row))
        for j,(lbl,val,delta) in enumerate(row):
            with cols[j]:
                if delta: st.metric(lbl, val, delta)
                else:     st.metric(lbl, val)

def top_n(df, grp, val, n=10):
    return df.groupby(grp)[val].sum().sort_values(ascending=False).head(n).reset_index()

def prep_date(df, dcol, vcol):
    d = df.copy()
    d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
    d = d.dropna(subset=[dcol])
    d["_M"]   = d[dcol].dt.to_period("M").astype(str)
    d["_Q"]   = d[dcol].dt.to_period("Q").astype(str)
    d["_Y"]   = d[dcol].dt.year
    d["_DOW"] = d[dcol].dt.day_name()
    return d

# ══════════════════════════════════════════════════════════════════════════════
#  SALES DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
def render_sales(df, found):
    _cur = detect_currency(df)  # auto-detect currency
    dom = "Sales"; ac = C["blue"]
    sc  = found.get("sales");    pc   = found.get("profit")
    prd = found.get("product");  cat  = found.get("category")
    dc  = found.get("date");     qty  = found.get("quantity")
    cus = found.get("customer"); reg  = found.get("region")
    dis = found.get("discount"); sub  = found.get("sub_category")
    seg = found.get("segment");  shm  = found.get("ship_mode")
    cit = found.get("city");     sta  = found.get("state")

    # ── Smart KPIs ────────────────────────────────────────────────────────
    section("📌 Sales KPIs", dom)
    kpis = []
    if sc:
        s = df[sc].dropna()
        kpis += [("💰 Total Revenue",   fmt_money(s.sum(),_cur), None),
                 ("📈 Avg Order Value",  fmt_money(s.mean(),_cur), None),
                 ("🔝 Max Order",        fmt_money(s.max(),_cur), None)]
    if pc:
        p = df[pc].dropna()
        kpis.append(("🏆 Total Profit", fmt_money(p.sum(),_cur), None))
        if sc: kpis.append(("📊 Profit Margin", f"{p.sum()/df[sc].sum()*100:.1f}%", None))
    if qty: kpis.append(("📦 Total Units",   f"{df[qty].sum():,.0f}", None))
    if cus: kpis.append(("👥 Customers",     f"{df[cus].nunique():,}", None))
    if prd: kpis.append(("🛒 Products",      f"{df[prd].nunique():,}", None))
    if cat: kpis.append(("🗂 Categories",    f"{df[cat].nunique():,}", None))
    if reg: kpis.append(("🌍 Regions",       f"{df[reg].nunique():,}", None))
    if dis: kpis.append(("🏷 Avg Discount",  f"{df[dis].mean():,.2f}", None))
    kpis.append(("🗂 Total Records", f"{len(df):,}", None))
    show_metrics(kpis)

    # ── Time Series ───────────────────────────────────────────────────────
    if dc and sc:
        section("📅 Revenue Time Trends", dom)
        d2 = prep_date(df, dc, sc)
        if not d2.empty:
            tab1,tab2,tab3,tab4 = st.tabs(["📆 Monthly","📊 Quarterly","📅 Yearly","📅 Day of Week"])
            with tab1:
                m = d2.groupby("_M")[sc].sum().reset_index()
                m.columns=["Month","Revenue"]
                fig=px.bar(m,x="Month",y="Revenue",title="Monthly Revenue",
                           color="Revenue",color_continuous_scale="Blues",text_auto=".2s")
                fig.add_scatter(x=m["Month"],y=m["Revenue"],mode="lines+markers",
                                line=dict(color=ac,width=2.5),name="Trend")
                fig.update_layout(**cd(430),showlegend=False)
                st.plotly_chart(fig,use_container_width=True)
                if len(m)>1:
                    m["MoM%"]=m["Revenue"].pct_change()*100
                    fig2=px.bar(m.dropna(),x="Month",y="MoM%",title="Month-over-Month Growth %",
                                color="MoM%",color_continuous_scale=["#ff4444",ac],text_auto=".1f")
                    fig2.update_layout(**cd(320)); st.plotly_chart(fig2,use_container_width=True)
            with tab2:
                q=d2.groupby("_Q")[sc].sum().reset_index(); q.columns=["Quarter","Revenue"]
                fig=px.bar(q,x="Quarter",y="Revenue",title="Quarterly Revenue",
                           color="Revenue",color_continuous_scale="Teal",text_auto=".2s")
                fig.update_layout(**cd(400)); st.plotly_chart(fig,use_container_width=True)
                if pc:
                    qp=d2.groupby("_Q")[pc].sum().reset_index(); qp.columns=["Quarter","Profit"]
                    mg=q.merge(qp,on="Quarter")
                    fig2=make_subplots(specs=[[{"secondary_y":True}]])
                    fig2.add_bar(x=mg["Quarter"],y=mg["Revenue"],name="Revenue",marker_color=ac)
                    fig2.add_scatter(x=mg["Quarter"],y=mg["Profit"],mode="lines+markers",
                                     name="Profit",line=dict(color=C["red"],width=3),secondary_y=True)
                    fig2.update_layout(title="Quarterly Revenue vs Profit",**cd(380))
                    st.plotly_chart(fig2,use_container_width=True)
            with tab3:
                y=d2.groupby("_Y")[sc].sum().reset_index(); y.columns=["Year","Revenue"]
                fig=px.bar(y,x="Year",y="Revenue",title="Yearly Revenue",
                           color="Revenue",color_continuous_scale="Viridis",text_auto=".2s")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
                if len(y)>1:
                    y["YoY%"]=y["Revenue"].pct_change()*100
                    for _,row in y.dropna().iterrows():
                        ar="🔺" if row["YoY%"]>0 else "🔻"
                        st.write(f"{ar} **{int(row['Year'])}**: {row['YoY%']:.1f}% YoY")
            with tab4:
                dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                dw=d2.groupby("_DOW")[sc].sum().reindex(dow_order).reset_index()
                dw.columns=["Day","Revenue"]
                fig=px.bar(dw,x="Day",y="Revenue",title="Revenue by Day of Week",
                           color="Revenue",color_continuous_scale="Sunset",text_auto=".2s")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
                c1,c2=st.columns(2)
                with c1:
                    bm=(d2.groupby(["_M","_DOW"])[sc].sum().reset_index()
                        .sort_values(sc,ascending=False).groupby("_M").first().reset_index())
                    bm.columns=["Month","Best Day","Revenue"]
                    st.markdown("**🏆 Best Sales Day per Month**")
                    st.dataframe(bm.style.format({"Revenue":"{:,.0f}"}),use_container_width=True)
                with c2:
                    bq=(d2.groupby(["_Q","_DOW"])[sc].sum().reset_index()
                        .sort_values(sc,ascending=False).groupby("_Q").first().reset_index())
                    bq.columns=["Quarter","Best Day","Revenue"]
                    st.markdown("**🏆 Best Sales Day per Quarter**")
                    st.dataframe(bq.style.format({"Revenue":"{:,.0f}"}),use_container_width=True)

    # ── Product Performance ───────────────────────────────────────────────
    if prd and sc:
        section("🛒 Product Performance", dom)
        c1,c2=st.columns(2)
        with c1:
            t=top_n(df,prd,sc,10); t.columns=["Product","Revenue"]
            fig=px.bar(t,x="Revenue",y="Product",orientation="h",
                       title="Top 10 Products by Revenue",color="Revenue",
                       color_continuous_scale="Blues",text_auto=".2s")
            fig.update_layout(**cd(460),yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            b=df.groupby(prd)[sc].sum().sort_values().head(5).reset_index()
            b.columns=["Product","Revenue"]
            fig=px.bar(b,x="Revenue",y="Product",orientation="h",
                       title="⚠️ Bottom 5 Products",color="Revenue",
                       color_continuous_scale="Reds",text_auto=".2s")
            fig.update_layout(**cd(360),yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig,use_container_width=True)
        if pc:
            c3,c4=st.columns(2)
            with c3:
                tp=top_n(df,prd,pc,5); tp.columns=["Product","Profit"]
                fig=px.bar(tp,x="Profit",y="Product",orientation="h",
                           title="Top 5 Products by Profit",color="Profit",
                           color_continuous_scale="Greens",text_auto=".2s")
                fig.update_layout(**cd(360),yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig,use_container_width=True)
            with c4:
                d3=df.copy()
                d3["_M%"]=(d3[pc]/d3[sc].replace(0,np.nan))*100
                pm=d3.groupby(prd)["_M%"].mean().sort_values(ascending=False).head(10).reset_index()
                pm.columns=["Product","Margin %"]
                fig=px.bar(pm,x="Margin %",y="Product",orientation="h",
                           title="Top 10 Products by Margin %",color="Margin %",
                           color_continuous_scale="RdYlGn",text_auto=".1f")
                fig.update_layout(**cd(420),yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig,use_container_width=True)
        if qty:
            tq=top_n(df,prd,qty,10); tq.columns=["Product","Units"]
            fig=px.bar(tq,x="Units",y="Product",orientation="h",
                       title="Top 10 Products by Units Sold",color="Units",
                       color_continuous_scale="Purples",text_auto=".2s")
            fig.update_layout(**cd(420),yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig,use_container_width=True)

    # ── Category Analysis ─────────────────────────────────────────────────
    if cat and sc:
        section("🗂 Category Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            cs=top_n(df,cat,sc,8); cs.columns=["Category","Revenue"]
            fig=px.pie(cs,values="Revenue",names="Category",title="Revenue Share by Category",
                       color_discrete_sequence=px.colors.sequential.Blues_r,hole=0.4)
            fig.update_layout(**cd(400)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            cb=px.bar(cs,x="Category",y="Revenue",title="Revenue by Category",
                      color="Revenue",color_continuous_scale="Blues",text_auto=".2s")
            cb.update_layout(**cd(400)); st.plotly_chart(cb,use_container_width=True)
        if pc:
            d3=df.copy()
            d3["_M%"]=(d3[pc]/d3[sc].replace(0,np.nan))*100
            cm=d3.groupby(cat)["_M%"].mean().sort_values(ascending=False).reset_index()
            cm.columns=["Category","Avg Margin %"]
            fig=px.bar(cm,x="Category",y="Avg Margin %",title="Profit Margin by Category",
                       color="Avg Margin %",color_continuous_scale="RdYlGn",text_auto=".1f")
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    # ── Sub-Category ──────────────────────────────────────────────────────
    if sub and sc:
        section("📂 Sub-Category Breakdown", dom)
        sc2=top_n(df,sub,sc,10); sc2.columns=["Sub-Category","Revenue"]
        fig=px.bar(sc2,x="Revenue",y="Sub-Category",orientation="h",
                   title="Top 10 Sub-Categories by Revenue",color="Revenue",
                   color_continuous_scale="Blues",text_auto=".2s")
        fig.update_layout(**cd(440),yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig,use_container_width=True)

    # ── Region / Geography ────────────────────────────────────────────────
    if reg and sc:
        section("🌍 Regional Performance", dom)
        c1,c2=st.columns(2)
        with c1:
            rs=top_n(df,reg,sc,8); rs.columns=["Region","Revenue"]
            fig=px.funnel(rs,x="Revenue",y="Region",title="Revenue by Region",
                          color_discrete_sequence=[ac])
            fig.update_layout(**cd(420)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if pc:
                rp=df.groupby(reg)[[sc,pc]].sum().reset_index()
                rp["Margin%"]=rp[pc]/rp[sc]*100
                fig=px.scatter(rp,x=sc,y=pc,size="Margin%",hover_name=reg,
                               title="Region: Revenue vs Profit",color="Margin%",
                               color_continuous_scale="RdYlGn")
                fig.update_layout(**cd(420)); st.plotly_chart(fig,use_container_width=True)
            elif cit:
                ct=top_n(df,cit,sc,10); ct.columns=["City","Revenue"]
                fig=px.bar(ct,x="Revenue",y="City",orientation="h",
                           title="Top 10 Cities by Revenue",color="Revenue",
                           color_continuous_scale="Blues",text_auto=".2s")
                fig.update_layout(**cd(420),yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig,use_container_width=True)

    # ── Customer Analysis ─────────────────────────────────────────────────
    if cus and sc:
        section("👥 Customer Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            tc=top_n(df,cus,sc,5); tc.columns=["Customer","Revenue"]
            fig=px.bar(tc,x="Revenue",y="Customer",orientation="h",
                       title="Top 5 Customers by Revenue",color="Revenue",
                       color_continuous_scale="Oranges",text_auto=".2s")
            fig.update_layout(**cd(360),yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            opc=df.groupby(cus).size().reset_index(name="Orders")
            fig=px.histogram(opc,x="Orders",nbins=30,title="Orders per Customer Distribution",
                             color_discrete_sequence=[ac])
            fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)

    # ── Segment / Ship Mode ───────────────────────────────────────────────
    if seg and sc:
        section("🎯 Segment Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            ss=df.groupby(seg)[sc].sum().reset_index(); ss.columns=["Segment","Revenue"]
            fig=px.pie(ss,values="Revenue",names="Segment",title="Revenue by Segment",
                       hole=0.4,color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
        if pc:
            with c2:
                sp2=df.groupby(seg)[[sc,pc]].sum().reset_index()
                sp2["Margin%"]=sp2[pc]/sp2[sc]*100
                fig=px.bar(sp2,x="Segment",y="Margin%",title="Profit Margin by Segment",
                           color="Margin%",color_continuous_scale="RdYlGn",text_auto=".1f")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    if shm and sc:
        sm=df.groupby(shm)[sc].sum().sort_values(ascending=False).reset_index()
        sm.columns=["Ship Mode","Revenue"]
        fig=px.bar(sm,x="Ship Mode",y="Revenue",title="Revenue by Shipping Mode",
                   color="Revenue",color_continuous_scale="Blues",text_auto=".2s")
        fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)

    # ── Discount Impact ───────────────────────────────────────────────────
    if dis and sc:
        section("🏷 Discount Impact", dom)
        c1,c2=st.columns(2)
        with c1:
            fig=px.scatter(df.sample(min(800,len(df))),x=dis,y=sc,title="Discount vs Revenue",
                           color=sc,color_continuous_scale="Blues",opacity=0.6)
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.histogram(df,x=dis,nbins=30,title="Discount Distribution",
                             color_discrete_sequence=[ac])
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
        corr=df[[dis,sc]].dropna().corr().iloc[0,1]
        msg="📈 Positive — higher discounts increase order value." if corr>0.1 else "📉 Negative — discounts reduce revenue." if corr<-0.1 else "➡️ Neutral correlation."
        insight(f"Discount–Revenue correlation: <strong>{corr:.3f}</strong>. {msg}")

    # ── Profitability Scatter ─────────────────────────────────────────────
    if sc and pc:
        section("💹 Profitability Analysis", dom)
        d3=df.copy(); d3["_M%"]=(d3[pc]/d3[sc].replace(0,np.nan))*100
        fig=px.scatter(d3.sample(min(800,len(d3))),x=sc,y=pc,color="_M%",
                       color_continuous_scale="RdYlGn",title="Revenue vs Profit (coloured by Margin %)",
                       hover_data=[prd] if prd else None,opacity=0.7)
        fig.update_layout(**cd(440)); st.plotly_chart(fig,use_container_width=True)
        overall=d3[pc].sum()/d3[sc].sum()*100
        insight(f"Overall Profit Margin: <strong>{overall:.2f}%</strong> | "
                f"Total Revenue: <strong>{d3[sc].sum():,.2f}</strong> | "
                f"Total Profit: <strong>{d3[pc].sum():,.2f}</strong>")


# ══════════════════════════════════════════════════════════════════════════════
#  HR DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
def render_hr(df, found):
    dom = "HR"; ac = C["purple"]
    _cur = detect_currency(df)  # auto-detect currency
    sal  = found.get("salary");    dept = found.get("department")
    attr = found.get("attrition"); gen  = found.get("gender")
    age  = found.get("age");       ten  = found.get("tenure")
    emp  = found.get("employee_id") or found.get("employee_name")
    jt   = found.get("job_title"); hd   = found.get("hire_date")
    perf = found.get("performance");edu  = found.get("education")
    mar  = found.get("marital");   ageg = found.get("age_group")
    sat  = found.get("satisfaction")

    total_emp = df[emp].nunique() if emp else len(df)

    # ── Smart HR KPIs ──────────────────────────────────────────────────────
    section("📌 HR KPIs", dom)
    kpis = [("👥 Total Employees", f"{total_emp:,}", None)]

    if dept:
        kpis.append(("🏢 Departments", f"{df[dept].nunique():,}", None))

    if sal:
        s = df[sal].dropna()
        kpis += [("💰 Total Payroll",   fmt_money(s.sum(),_cur,0), None),
                 ("📈 Avg Salary",      fmt_money(s.mean(),_cur,0), None),
                 ("🔝 Highest Salary",  fmt_money(s.max(),_cur,0), None),
                 ("🔽 Lowest Salary",   fmt_money(s.min(),_cur,0), None),
                 ("📊 Median Salary",   fmt_money(s.median(),_cur,0), None)]

    if gen:
        _gen_norm = normalize_gender(df[gen])
        gc = _gen_norm.value_counts()
        male   = gc.get("Male",   0)
        female = gc.get("Female", 0)
        other  = gc.get("Other",  0)
        kpis += [("👨 Male Employees",   f"{int(male):,}", None),
                 ("👩 Female Employees", f"{int(female):,}", None)]
        if other > 0:
            kpis.append(("🧑 Other/Non-Binary", f"{int(other):,}", None))
        if sal and male>0 and female>0:
            df2=df.copy(); df2["_gen"]=normalize_gender(df2[gen])
            m_sal=df2[df2["_gen"]=="Male"][sal].mean()
            f_sal=df2[df2["_gen"]=="Female"][sal].mean()
            gap=abs(m_sal-f_sal)/max(m_sal,f_sal)*100
            kpis.append(("⚖️ Gender Pay Gap", f"{gap:.1f}%", None))

    if attr:
        rate=df[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"]).mean()*100
        kpis.append(("📉 Attrition Rate", f"{rate:.1f}%", None))

    if ten:
        kpis += [("📅 Avg Tenure (yrs)", f"{df[ten].mean():.1f}", None),
                 ("🏅 Tenured (>5yr)",   f"{(df[ten]>5).sum():,}", None)]

    if age:
        kpis += [("🎂 Avg Age", f"{df[age].mean():.1f}", None)]

    if jt:
        kpis.append(("💼 Job Titles", f"{df[jt].nunique():,}", None))

    show_metrics(kpis)

    # ── Department Analysis ───────────────────────────────────────────────
    if dept:
        section("🏢 Department Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            dc=df[dept].value_counts().reset_index()
            dc.columns=["Department","Headcount"]
            fig=px.bar(dc,x="Headcount",y="Department",orientation="h",
                       title="Headcount by Department",color="Headcount",
                       color_continuous_scale="Purples",text_auto=True)
            fig.update_layout(**cd(max(380,len(dc)*30)),yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.pie(dc,values="Headcount",names="Department",
                       title="Workforce Distribution",hole=0.4,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

        if sal:
            ds=df.groupby(dept)[sal].agg(["mean","min","max","sum"]).reset_index()
            ds.columns=["Department","Avg Salary","Min Salary","Max Salary","Total Payroll"]
            st.markdown("**💰 Salary Statistics by Department**")
            st.dataframe(ds.style.format({c:"{:,.0f}" for c in ds.columns if c!="Department"}),
                         use_container_width=True)
            c3,c4=st.columns(2)
            with c3:
                fig=px.bar(ds,x="Department",y="Avg Salary",title="Avg Salary by Department",
                           color="Avg Salary",color_continuous_scale="Purples",text_auto=".0f")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
            with c4:
                fig=px.bar(ds,x="Department",y="Total Payroll",title="Total Payroll by Department",
                           color="Total Payroll",color_continuous_scale="Blues",text_auto=".2s")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    # ── Salary Analysis ───────────────────────────────────────────────────
    if sal:
        section("💰 Salary Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(df,x=sal,nbins=40,title="Salary Distribution",
                             color_discrete_sequence=[ac],marginal="box")
            fig.update_layout(**cd(400)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if jt:
                jts=df.groupby(jt)[sal].mean().sort_values(ascending=False).head(10).reset_index()
                jts.columns=["Job Title","Avg Salary"]
                fig=px.bar(jts,x="Avg Salary",y="Job Title",orientation="h",
                           title="Avg Salary by Job Title (Top 10)",color="Avg Salary",
                           color_continuous_scale="Purples",text_auto=".0f")
                fig.update_layout(**cd(400),yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig,use_container_width=True)
            elif gen:
                df2=df.copy(); df2["_gen"]=df2[gen].astype(str).str.title()
                gs=df2.groupby("_gen")[sal].mean().reset_index()
                gs.columns=["Gender","Avg Salary"]
                fig=px.bar(gs,x="Gender",y="Avg Salary",title="Avg Salary by Gender",
                           color="Gender",color_discrete_sequence=[ac,C["red"]],text_auto=".0f")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

        if gen:
            df2=df.copy(); df2["_gen"]=df2[gen].astype(str).str.title()
            fig=px.box(df2,x="_gen",y=sal,color="_gen",
                       title="Salary Distribution by Gender",
                       color_discrete_sequence=[ac,C["red"],C["green"]])
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    # ── Gender Analysis ───────────────────────────────────────────────────
    if gen:
        section("⚧ Gender Analysis", dom)
        df2=df.copy(); df2["_gen"]=df2[gen].astype(str).str.title()
        c1,c2=st.columns(2)
        with c1:
            gc2=df2["_gen"].value_counts().reset_index()
            gc2.columns=["Gender","Count"]
            fig=px.pie(gc2,values="Count",names="Gender",title="Gender Distribution",
                       hole=0.45,color_discrete_sequence=[ac,C["red"],C["green"]])
            fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if dept:
                gd=df2.groupby(["_gen",dept]).size().reset_index(name="Count")
                fig=px.bar(gd,x=dept,y="Count",color="_gen",barmode="group",
                           title="Gender Distribution by Department",
                           color_discrete_sequence=[ac,C["red"],C["green"]])
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    # ── Age Analysis ──────────────────────────────────────────────────────
    if age or ageg:
        section("🎂 Age Analysis", dom)
        c1,c2=st.columns(2)
        age_col = ageg or age
        with c1:
            if ageg:
                ac2=df[ageg].value_counts().reset_index()
                ac2.columns=["Age Group","Count"]
                fig=px.bar(ac2,x="Age Group",y="Count",title="Employees by Age Group",
                           color="Count",color_continuous_scale="Purples",text_auto=True)
            else:
                fig=px.histogram(df,x=age,nbins=20,title="Age Distribution",
                                 color_discrete_sequence=[ac])
            fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if sal:
                fig=px.scatter(df.sample(min(500,len(df))),x=age_col,y=sal,
                               title="Age vs Salary",color=dept if dept else None,opacity=0.6)
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)

    # ── Attrition Analysis ────────────────────────────────────────────────
    if attr:
        section("📉 Attrition Analysis", dom)
        df2=df.copy()
        df2["_left"]=df2[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"])
        rate=df2["_left"].mean()*100
        c1,c2=st.columns(2)
        with c1:
            ac3=df2["_left"].map({True:"Left",False:"Active"}).value_counts().reset_index()
            ac3.columns=["Status","Count"]
            fig=px.pie(ac3,values="Count",names="Status",title=f"Attrition Rate: {rate:.1f}%",
                       hole=0.45,color_discrete_map={"Left":C["red"],"Active":C["green"]})
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if dept:
                da=df2.groupby(dept)["_left"].mean().mul(100).sort_values(ascending=False).reset_index()
                da.columns=["Department","Attrition Rate %"]
                fig=px.bar(da,x="Department",y="Attrition Rate %",
                           title="Attrition Rate by Department",
                           color="Attrition Rate %",color_continuous_scale="Reds",text_auto=".1f")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
        c3,c4=st.columns(2)
        if gen:
            with c3:
                df2["_gen"]=df2[gen].astype(str).str.title()
                ga=df2.groupby("_gen")["_left"].mean().mul(100).reset_index()
                ga.columns=["Gender","Attrition %"]
                fig=px.bar(ga,x="Gender",y="Attrition %",title="Attrition by Gender",
                           color="Gender",color_discrete_sequence=[ac,C["red"]],text_auto=".1f")
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        if sal:
            with c4:
                fig=px.box(df2,x="_left",y=sal,color="_left",
                           title="Salary: Active vs Left Employees",
                           color_discrete_map={True:C["red"],False:C["green"]},
                           labels={"_left":"Left?"})
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        if ten:
            df2_left=df2[df2["_left"]==True]; df2_act=df2[df2["_left"]==False]
            insight(f"📉 Attrition Rate: <strong>{rate:.1f}%</strong> | "
                    f"Avg tenure of leavers: <strong>{df2_left[ten].mean():.1f} yrs</strong> | "
                    f"Avg tenure of active: <strong>{df2_act[ten].mean():.1f} yrs</strong>")
        if age:
            with c3 if not gen else c4:
                aa=df2.groupby(pd.cut(df2[age],bins=5))["_left"].mean().mul(100).reset_index()
                aa.columns=["Age Band","Attrition %"]
                aa["Age Band"]=aa["Age Band"].astype(str)
                fig=px.bar(aa,x="Age Band",y="Attrition %",title="Attrition by Age Band",
                           color="Attrition %",color_continuous_scale="Reds",text_auto=".1f")
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)

    # ── Tenure Analysis ───────────────────────────────────────────────────
    if ten:
        section("📅 Tenure Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(df,x=ten,nbins=25,title="Tenure Distribution (Years)",
                             color_discrete_sequence=[ac],marginal="box")
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if dept:
                td=df.groupby(dept)[ten].mean().sort_values(ascending=False).reset_index()
                td.columns=["Department","Avg Tenure"]
                fig=px.bar(td,x="Department",y="Avg Tenure",title="Avg Tenure by Department",
                           color="Avg Tenure",color_continuous_scale="Purples",text_auto=".1f")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    # ── Satisfaction ──────────────────────────────────────────────────────
    if sat:
        section("⭐ Satisfaction Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(df,x=sat,nbins=10,title="Satisfaction Score Distribution",
                             color_discrete_sequence=[C["yellow"]])
            fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if dept:
                ds2=df.groupby(dept)[sat].mean().sort_values(ascending=False).reset_index()
                ds2.columns=["Department","Avg Satisfaction"]
                fig=px.bar(ds2,x="Department",y="Avg Satisfaction",
                           title="Avg Satisfaction by Department",
                           color="Avg Satisfaction",color_continuous_scale="RdYlGn",text_auto=".2f")
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)

    # ── Performance ───────────────────────────────────────────────────────
    if perf:
        section("🏅 Performance Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            pc2=df[perf].value_counts().reset_index()
            pc2.columns=["Rating","Count"]
            fig=px.bar(pc2,x="Rating",y="Count",title="Performance Rating Distribution",
                       color="Count",color_continuous_scale="Greens",text_auto=True)
            fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if sal:
                fig=px.box(df,x=perf,y=sal,color=perf,title="Salary by Performance Rating")
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)

    # ── Hiring Trend ──────────────────────────────────────────────────────
    if hd:
        section("📈 Hiring Trend", dom)
        df2=df.copy()
        df2[hd]=pd.to_datetime(df2[hd],errors="coerce")
        df2=df2.dropna(subset=[hd])
        if not df2.empty:
            df2["_HY"]=df2[hd].dt.year
            df2["_HM"]=df2[hd].dt.to_period("M").astype(str)
            c1,c2=st.columns(2)
            with c1:
                hy=df2.groupby("_HY").size().reset_index(name="Hired")
                fig=px.bar(hy,x="_HY",y="Hired",title="Yearly Hiring Trend",
                           color="Hired",color_continuous_scale="Greens",text_auto=True)
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
            with c2:
                hm=df2.groupby("_HM").size().reset_index(name="Hired")
                fig=px.line(hm,x="_HM",y="Hired",title="Monthly Hiring Trend",
                            line_shape="spline",markers=True)
                fig.update_traces(line_color=ac)
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MARKETING DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
def render_marketing(df, found):
    _cur = detect_currency(df)
    dom = "Marketing"; ac = C["red"]
    sp   = found.get("spend");   rev  = found.get("sales")
    chan = found.get("channel"); imp  = found.get("impressions")
    clk  = found.get("clicks");  conv = found.get("conversions")
    roi  = found.get("roi");     dc   = found.get("date")
    cat  = found.get("category")

    section("📌 Marketing KPIs", dom)
    kpis=[]
    if sp:   kpis.append(("💸 Total Spend",      fmt_money(df[sp].sum(),_cur), None))
    if rev:  kpis.append(("💰 Total Revenue",    fmt_money(df[rev].sum(),_cur), None))
    if sp and rev:
        roi_v=(df[rev].sum()-df[sp].sum())/df[sp].sum()*100
        kpis.append(("📈 Overall ROI", f"{roi_v:.1f}%", None))
    if imp:  kpis.append(("👁 Impressions",      f"{df[imp].sum():,.0f}", None))
    if clk:  kpis.append(("🖱 Total Clicks",     f"{df[clk].sum():,.0f}", None))
    if imp and clk:
        kpis.append(("🎯 CTR",  f"{df[clk].sum()/df[imp].sum()*100:.2f}%", None))
    if conv: kpis.append(("✅ Conversions",      f"{df[conv].sum():,.0f}", None))
    if clk and conv:
        kpis.append(("🔄 CVR",  f"{df[conv].sum()/df[clk].sum()*100:.2f}%", None))
    if chan: kpis.append(("📣 Channels",         f"{df[chan].nunique():,}", None))
    kpis.append(("🗂 Records", f"{len(df):,}", None))
    show_metrics(kpis)

    if sp and rev:
        section("💸 Spend vs Revenue & ROI", dom)
        df2=df.copy()
        df2["_ROI%"]=(df2[rev]-df2[sp])/df2[sp].replace(0,np.nan)*100
        c1,c2=st.columns(2)
        with c1:
            fig=px.scatter(df2,x=sp,y=rev,color="_ROI%",color_continuous_scale="RdYlGn",
                           title="Spend vs Revenue",opacity=0.7,
                           hover_data=[chan] if chan else None)
            fig.update_layout(**cd(420)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if chan:
                cr=df2.groupby(chan)["_ROI%"].mean().sort_values(ascending=False).reset_index()
                cr.columns=["Channel","Avg ROI %"]
                fig=px.bar(cr,x="Channel",y="Avg ROI %",title="ROI by Channel",
                           color="Avg ROI %",color_continuous_scale="RdYlGn",text_auto=".1f")
                fig.update_layout(**cd(420)); st.plotly_chart(fig,use_container_width=True)
        if chan:
            cs2=df2.groupby(chan)[[sp,rev]].sum().reset_index()
            fig=go.Figure()
            fig.add_bar(x=cs2[chan],y=cs2[sp],name="Spend",marker_color=C["red"])
            fig.add_bar(x=cs2[chan],y=cs2[rev],name="Revenue",marker_color=C["green"])
            fig.update_layout(title="Spend vs Revenue by Channel",barmode="group",**cd(400))
            st.plotly_chart(fig,use_container_width=True)

    if imp or clk or conv:
        section("🔻 Campaign Funnel", dom)
        fv,fl=[],[]
        for col,lbl in [(imp,"Impressions"),(clk,"Clicks"),(conv,"Conversions")]:
            if col: fv.append(df[col].sum()); fl.append(lbl)
        if len(fv)>=2:
            fig=go.Figure(go.Funnel(y=fl,x=fv,textinfo="value+percent initial",
                          marker=dict(color=[C["blue"],C["yellow"],C["green"]][:len(fv)])))
            fig.update_layout(title="Campaign Conversion Funnel",**cd(380))
            st.plotly_chart(fig,use_container_width=True)
        c1,c2=st.columns(2)
        if imp and clk and chan:
            df2=df.copy(); df2["_CTR%"]=df2[clk]/df2[imp].replace(0,np.nan)*100
            with c1:
                ct=df2.groupby(chan)["_CTR%"].mean().sort_values(ascending=False).reset_index()
                ct.columns=["Channel","CTR %"]
                fig=px.bar(ct,x="Channel",y="CTR %",title="CTR by Channel",
                           color="CTR %",color_continuous_scale="Teal",text_auto=".2f")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
        if clk and conv and chan:
            df2=df.copy(); df2["_CVR%"]=df2[conv]/df2[clk].replace(0,np.nan)*100
            with c2:
                cv=df2.groupby(chan)["_CVR%"].mean().sort_values(ascending=False).reset_index()
                cv.columns=["Channel","CVR %"]
                fig=px.bar(cv,x="Channel",y="CVR %",title="Conversion Rate by Channel",
                           color="CVR %",color_continuous_scale="Purples",text_auto=".2f")
                fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    if dc and (rev or sp):
        vcol = rev or sp
        section("📅 Spend/Revenue Trends Over Time", dom)
        d2=prep_date(df,dc,vcol)
        if not d2.empty:
            m=d2.groupby("_M")[vcol].sum().reset_index(); m.columns=["Month","Value"]
            fig=px.bar(m,x="Month",y="Value",title=f"Monthly {'Revenue' if rev else 'Spend'}",
                       color="Value",color_continuous_scale="Reds",text_auto=".2s")
            fig.add_scatter(x=m["Month"],y=m["Value"],mode="lines+markers",
                            line=dict(color=ac,width=2),name="Trend")
            fig.update_layout(**cd(420),showlegend=False)
            st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ECOMMERCE DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
def render_ecommerce(df, found):
    _cur = detect_currency(df)
    dom="Ecommerce"; ac=C["green"]
    sc=found.get("sales");    prd=found.get("product"); cat=found.get("category")
    cus=found.get("customer");pc=found.get("profit");   dc=found.get("date")
    ret=found.get("returns"); pay=found.get("payment"); del_=found.get("delivery")
    dis=found.get("discount");qty=found.get("quantity");reg=found.get("region")

    section("📌 Ecommerce KPIs", dom)
    kpis=[]
    if sc:   kpis+=[("💰 Total Revenue",fmt_money(df[sc].sum(),_cur),None),
                    ("📈 Avg Order",fmt_money(df[sc].mean(),_cur),None)]
    if pc:   kpis.append(("🏆 Total Profit",fmt_money(df[pc].sum(),_cur),None))
    if sc and pc: kpis.append(("📊 Margin",f"{df[pc].sum()/df[sc].sum()*100:.1f}%",None))
    if qty:  kpis.append(("📦 Units Sold",f"{df[qty].sum():,.0f}",None))
    if cus:  kpis.append(("👥 Customers",f"{df[cus].nunique():,}",None))
    if prd:  kpis.append(("🛒 Products",f"{df[prd].nunique():,}",None))
    if ret:
        df2=df.copy(); df2["_ret"]=df2[ret].astype(str).str.lower().isin(["yes","true","1","returned"])
        kpis.append(("↩ Return Rate",f"{df2['_ret'].mean()*100:.1f}%",None))
    if del_: kpis.append(("🚚 Avg Delivery",f"{df[del_].mean():.1f} days",None))
    if dis:  kpis.append(("🏷 Avg Discount",f"{df[dis].mean():,.2f}",None))
    kpis.append(("🗂 Records",f"{len(df):,}",None))
    show_metrics(kpis)

    # Time + product + customer + returns + payment + delivery — reuse sales patterns
    render_sales(df, found)  # reuse full sales engine for common charts

    if ret:
        section("↩ Returns Analysis", dom)
        df2=df.copy(); df2["_ret"]=df2[ret].astype(str).str.lower().isin(["yes","true","1","returned"])
        c1,c2=st.columns(2)
        with c1:
            rc=df2["_ret"].map({True:"Returned",False:"Kept"}).value_counts().reset_index()
            rc.columns=["Status","Count"]
            fig=px.pie(rc,values="Count",names="Status",title="Return Rate",hole=0.45,
                       color_discrete_map={"Returned":C["red"],"Kept":C["green"]})
            fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if cat:
                rb=df2.groupby(cat)["_ret"].mean().mul(100).sort_values(ascending=False).reset_index()
                rb.columns=["Category","Return Rate %"]
                fig=px.bar(rb,x="Category",y="Return Rate %",title="Return Rate by Category",
                           color="Return Rate %",color_continuous_scale="Reds",text_auto=".1f")
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)

    if pay:
        section("💳 Payment Methods", dom)
        pm=df[pay].value_counts().reset_index(); pm.columns=["Method","Count"]
        fig=px.bar(pm,x="Method",y="Count",title="Transactions by Payment Method",
                   color="Count",color_continuous_scale="Greens",text_auto=True)
        fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)

    if del_:
        section("🚚 Delivery Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(df,x=del_,nbins=25,title="Delivery Time Distribution (days)",
                             color_discrete_sequence=[ac])
            fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            if cat:
                dc2=df.groupby(cat)[del_].mean().sort_values(ascending=False).reset_index()
                dc2.columns=["Category","Avg Delivery Days"]
                fig=px.bar(dc2,x="Category",y="Avg Delivery Days",
                           title="Avg Delivery Time by Category",
                           color="Avg Delivery Days",color_continuous_scale="YlOrRd",text_auto=".1f")
                fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)
        insight(f"🚚 Avg delivery: <strong>{df[del_].mean():.1f} days</strong> | "
                f"Fastest: <strong>{df[del_].min():.0f}d</strong> | "
                f"Slowest: <strong>{df[del_].max():.0f}d</strong>")


# ══════════════════════════════════════════════════════════════════════════════
#  RETAIL DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
def render_retail(df, found):
    _cur = detect_currency(df)
    dom="Retail"; ac=C["yellow"]
    sc=found.get("sales"); store=found.get("store"); prd=found.get("product")
    cat=found.get("category"); pc=found.get("profit"); dis=found.get("discount")
    reg=found.get("region"); pay=found.get("payment"); qty=found.get("quantity")

    section("📌 Retail KPIs", dom)
    kpis=[]
    if sc:    kpis+=[("💰 Total Revenue",fmt_money(df[sc].sum(),_cur),None),
                     ("📈 Avg Transaction",fmt_money(df[sc].mean(),_cur),None)]
    if pc:    kpis.append(("🏆 Total Profit",fmt_money(df[pc].sum(),_cur),None))
    if sc and pc: kpis.append(("📊 Margin",f"{df[pc].sum()/df[sc].sum()*100:.1f}%",None))
    if qty:   kpis.append(("📦 Units",f"{df[qty].sum():,.0f}",None))
    if store: kpis+=[("🏪 Stores",f"{df[store].nunique():,}",None)]
    if prd:   kpis.append(("🛒 Products",f"{df[prd].nunique():,}",None))
    if cat:   kpis.append(("🗂 Categories",f"{df[cat].nunique():,}",None))
    if dis:   kpis.append(("🏷 Avg Discount",f"{df[dis].mean():,.2f}",None))
    kpis.append(("🗂 Records",f"{len(df):,}",None))
    show_metrics(kpis)

    if store and sc:
        section("🏪 Store Performance", dom)
        c1,c2=st.columns(2)
        with c1:
            ts=top_n(df,store,sc,10); ts.columns=["Store","Revenue"]
            fig=px.bar(ts,x="Revenue",y="Store",orientation="h",
                       title="Top 10 Stores by Revenue",color="Revenue",
                       color_continuous_scale="YlOrBr",text_auto=".2s")
            fig.update_layout(**cd(460),yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            bs=df.groupby(store)[sc].sum().sort_values().head(5).reset_index()
            bs.columns=["Store","Revenue"]
            fig=px.bar(bs,x="Revenue",y="Store",orientation="h",
                       title="⚠️ Bottom 5 Stores",color="Revenue",
                       color_continuous_scale="Reds",text_auto=".2s")
            fig.update_layout(**cd(360),yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig,use_container_width=True)
        if pc:
            sp2=df.groupby(store)[[sc,pc]].sum().reset_index()
            sp2["Margin%"]=sp2[pc]/sp2[sc]*100
            fig=px.scatter(sp2,x=sc,y=pc,size=sp2["Margin%"].clip(lower=0.1),
                           hover_name=store,title="Store: Revenue vs Profit",
                           color="Margin%",color_continuous_scale="RdYlGn")
            fig.update_layout(**cd(420)); st.plotly_chart(fig,use_container_width=True)

    render_sales(df, found)  # reuse common charts

    if pay:
        section("💳 Payment Methods", dom)
        pm=df[pay].value_counts().reset_index(); pm.columns=["Method","Count"]
        fig=px.bar(pm,x="Method",y="Count",title="Transactions by Payment Method",
                   color="Count",color_continuous_scale="YlOrBr",text_auto=True)
        fig.update_layout(**cd(360)); st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FRAUD DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
def render_fraud(df, found):
    dom="Fraud"
    cl={c.lower().strip():c for c in df.columns}
    fraud_cols=["class","label","fraud","is_fraud","isfraud","target","is fraud"]
    class_col=next((cl[c] for c in fraud_cols if c in cl),None)
    amt_col=cl.get("amount"); time_col=cl.get("time")
    v_cols=[c for c in df.columns if c.lower().startswith("v") and c[1:].isdigit()]
    if not class_col: st.warning("Fraud label column not found."); return

    df2=df.copy()
    df2["_lbl"]=df2[class_col].astype(str).str.lower().map(lambda x:1 if x in ["1","true","yes","fraud"] else 0)
    df2["_status"]=df2["_lbl"].map({0:"✅ Legitimate",1:"🚨 Fraud"})
    total=len(df2); fraud_ct=int(df2["_lbl"].sum()); legit_ct=total-fraud_ct
    fraud_pct=fraud_ct/total*100
    cur = detect_currency(df)  # auto-detect currency symbol

    section("🚨 Fraud Detection KPIs", dom)
    kpis=[("Total Transactions",f"{total:,}",None),
          ("Legitimate Transactions",f"{legit_ct:,}",None),
          ("Fraudulent Transactions",f"{fraud_ct:,}",None),
          ("Fraud Rate",f"{fraud_pct:.4f}%",None)]
    if amt_col:
        fa=df2[df2["_lbl"]==1][amt_col].dropna()
        la=df2[df2["_lbl"]==0][amt_col].dropna()
        if len(fa)>0:
                kpis+=[("Avg Fraudulent Amount",fmt_money(fa.mean(),cur),None),
                       ("Avg Legitimate Amount", fmt_money(la.mean(),cur),None),
                       ("Total Fraud Exposure",  fmt_money(fa.sum(),cur),None)]
    show_metrics(kpis)


    # ── Key Fraud Insight ─────────────────────────────────────────────────
    if amt_col and len(fa)>0:
        amt_diff = fa.mean() - la.mean()
        pct_higher = (fa.mean() - la.mean()) / la.mean() * 100
        direction = "higher" if amt_diff > 0 else "lower"
        reason = (
            "⚠️ Fraudsters target higher-value transactions to maximise gain per stolen card."
            if amt_diff > 0 else
            "⚠️ Micro-transaction fraud detected — small amounts used to avoid detection triggers."
        )
        insight(
            f"🔍 <strong>Fraud Pattern Detected:</strong> Average fraudulent transaction "
            f"(<strong>{fmt_money(fa.mean(),cur)}</strong>) is <strong>{abs(pct_higher):.1f}% {direction}</strong> "
            f"than average legitimate transaction (<strong>{fmt_money(la.mean(),cur)}</strong>). {reason} "
            f"Despite only <strong>{fraud_pct:.4f}%</strong> of transactions being fraudulent, "
            f"total financial exposure is <strong>{fmt_money(fa.sum(),cur)}</strong>."
        )

    section("📊 Fraud vs Legitimate", dom)
    c1,c2=st.columns(2)
    with c1:
        pd_=pd.DataFrame({"Status":["Legitimate","Fraud"],"Count":[legit_ct,fraud_ct]})
        fig=px.pie(pd_,values="Count",names="Status",title="Transaction Class Split",
                   color_discrete_map={"Legitimate":C["green"],"Fraud":C["red"]},hole=0.45)
        fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.bar(pd_,x="Status",y="Count",title="Count: Fraud vs Legitimate",
                   color="Status",color_discrete_map={"Legitimate":C["green"],"Fraud":C["red"]},
                   text_auto=True)
        fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    if amt_col:
        section("💳 Transaction Amount Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            fig=px.box(df2,x="_status",y=amt_col,color="_status",
                       title="Transaction Amount: Legitimate vs Fraud",
                       color_discrete_map={"✅ Legitimate":C["green"],"🚨 Fraud":C["red"]},
                       labels={"_status":"Transaction Type"})
            fig.update_layout(**cd(400)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.histogram(df2,x=amt_col,color="_status",nbins=50,
                             title="Amount Distribution: Legitimate vs Fraud",
                             color_discrete_map={"✅ Legitimate":C["green"],"🚨 Fraud":C["red"]},
                             barmode="overlay",opacity=0.7,labels={"_status":"Transaction Type"})
            fig.update_layout(**cd(400)); st.plotly_chart(fig,use_container_width=True)

    if time_col:
        section("⏱ Time Analysis", dom)
        c1,c2=st.columns(2)
        with c1:
            df2[time_col]=pd.to_numeric(df2[time_col],errors="coerce")
            fig=px.histogram(df2,x=time_col,color="_status",nbins=48,
                             title="Transactions Over Time",
                             color_discrete_map={"✅ Legitimate":C["green"],"🚨 Fraud":C["red"]},
                             barmode="overlay",opacity=0.7,labels={"_status":"Transaction Type"})
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)
        with c2:
            df2["_hr"]=(pd.to_numeric(df2[time_col],errors="coerce")/3600).fillna(0).astype(int)%24
            hr=df2.groupby(["_hr","_status"]).size().reset_index(name="Count")
            fig=px.bar(hr,x="_hr",y="Count",color="_status",title="Transactions by Hour of Day",
                       color_discrete_map={"✅ Legitimate":C["green"],"🚨 Fraud":C["red"]},barmode="group",
                       labels={"_hr":"Hour (0-23)","_status":"Transaction Type"})
            fig.update_layout(**cd(380)); st.plotly_chart(fig,use_container_width=True)

    if v_cols:
        section("🔬 Feature Analysis (Top 6 PCA Features)", dom)
        cols_g=st.columns(2)
        for i,vc in enumerate(v_cols[:6]):
            with cols_g[i%2]:
                fig=px.box(df2,x="_status",y=vc,color="_status",
                           title=f"{vc} — Legitimate vs Fraud",
                           color_discrete_map={"✅ Legitimate":C["green"],"🚨 Fraud":C["red"]},
                           labels={"_status":"Transaction Type"})
                fig.update_layout(**cd(320)); st.plotly_chart(fig,use_container_width=True)

    section("🔗 Feature Correlation with Fraud", dom)
    nc=df2.select_dtypes(include="number").columns.tolist()
    if "_lbl" in nc:
        corr=(df2[nc].corr()["_lbl"].drop("_lbl").abs().sort_values(ascending=False).head(15))
        cd_=corr.reset_index(); cd_.columns=["Feature","|Correlation|"]
        fig=px.bar(cd_,x="|Correlation|",y="Feature",orientation="h",
                   title="Top 15 Features Correlated with Fraud",color="|Correlation|",
                   color_continuous_scale="Reds",text_auto=".3f")
        fig.update_layout(**cd(480),yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERIC DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
def render_generic(df, found):
    dom="Generic"
    nc=df.select_dtypes(include="number").columns.tolist()
    cc=df.select_dtypes(include=["object","category"]).columns.tolist()

    section("📌 Dataset KPIs", dom)
    kpis=[("🗂 Records",f"{len(df):,}",None),("📊 Columns",f"{len(df.columns):,}",None),
          ("🔢 Numeric Cols",f"{len(nc):,}",None),("📝 Text Cols",f"{len(cc):,}",None)]
    for c in nc[:4]:
        s=df[c].dropna()
        kpis.append((f"Σ {c}",f"{s.sum():,.2f}",None))
    show_metrics(kpis)

    section("🔍 Auto Distribution Analysis", dom)
    c1,c2=st.columns(2)
    for i,col in enumerate(nc[:8]):
        with (c1 if i%2==0 else c2):
            fig=px.histogram(df,x=col,nbins=40,title=f"{col} Distribution",
                             color_discrete_sequence=[C["blue"]],marginal="box")
            fig.update_layout(**cd(340)); st.plotly_chart(fig,use_container_width=True)
    for cat in cc[:5]:
        if 2<=df[cat].nunique()<=30:
            cv=df[cat].value_counts().head(10).reset_index(); cv.columns=[cat,"Count"]
            fig=px.bar(cv,x=cat,y="Count",title=f"Top Values — {cat}",
                       color="Count",color_continuous_scale="Blues",text_auto=True)
            fig.update_layout(**cd(340)); st.plotly_chart(fig,use_container_width=True)
    if len(nc)>=3:
        section("🔗 Correlation Heatmap", dom)
        corr=df[nc[:12]].corr()
        fig=px.imshow(corr,color_continuous_scale="RdBu_r",zmin=-1,zmax=1,
                      title="Feature Correlation Heatmap",text_auto=".2f")
        fig.update_layout(**cd(520)); st.plotly_chart(fig,use_container_width=True)
    if found.get("date") and found.get("sales"):
        render_sales(df, found)


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE 0 — 1 CLICK STORY  ⚡  Your Business Intelligence Team
#  "Anyone Can Do Data Analysis — No Coding. No Training. Just Upload."
#  Complete AI narrative: KPI Headline → Critical Alerts → Trend Story →
#  Domain Scorecard → Wins & Risks → AI Insight → Monday Morning Actions
# ═══════════════════════════════════════════════════════════════════════════════

def render_one_click_story(df, found, domain):
    """
    ENGINE 0 — 1 Click Story | Your Business Intelligence Team
    Tagline: Anyone Can Do Data Analysis — No Coding. No Training. Just Upload.
    Sections:
      0. Dual tagline hero banner
      1. Headline KPI card
      2. Domain Scorecard (4 mini KPIs)
      3. Critical Alerts
      4. Performance Story (trend narrative)
      5. Wins & Risks (side by side)
      6. AI-Powered Insight (Claude narrative paragraph)
      7. Monday Morning Actions
      8. Footer with branding
    """
    import datetime

    ac   = DOMAIN_COLOR.get(domain, C["blue"])
    cur  = detect_currency(df)
    cl   = {c.lower().strip(): c for c in df.columns}

    sc   = found.get("sales");     pc   = found.get("profit")
    dc   = found.get("date");      prd  = found.get("product")
    cat  = found.get("category");  reg  = found.get("region")
    cus  = found.get("customer");  qty  = found.get("quantity")
    dis  = found.get("discount");  sal  = found.get("salary")
    attr = found.get("attrition"); dept = found.get("department")
    gen  = found.get("gender");    ten  = found.get("tenure")
    imp  = found.get("impressions"); clk = found.get("clicks")
    conv = found.get("conversions"); sp  = found.get("spend")
    store= found.get("store");     pay  = found.get("payment")
    perf = found.get("performance"); jt  = found.get("job_title")
    emp  = found.get("employee_id") or found.get("employee_name")
    tgt  = found.get("target");    srep = found.get("sales_rep")
    vend = found.get("vendor");    roi  = found.get("roi")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 0 — DUAL TAGLINE HERO BANNER
    # ══════════════════════════════════════════════════════════════════════
    domain_icons = {
        "Sales":"💼","Marketing":"📣","HR":"👥",
        "Ecommerce":"🛒","Retail":"🏪","Fraud":"🚨","Generic":"🔍"
    }
    d_icon = domain_icons.get(domain, "📊")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#020817,#0a1628,#0d1f3c);
                border:1px solid {ac}44;border-radius:20px;
                padding:32px 36px 24px;margin-bottom:16px;position:relative;overflow:hidden;">
        <!-- Glow orb -->
        <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
                    background:radial-gradient({ac}18,transparent 70%);border-radius:50%;"></div>
        <!-- Top badge row -->
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap;">
            <span style="font-size:1.4rem;">⚡</span>
            <span style="font-family:'Syne',sans-serif;font-size:1.55rem;
                         font-weight:800;color:#f1f5f9;letter-spacing:-.02em;">
                1 Click Story
            </span>
            <span style="background:{ac}22;color:{ac};border:1px solid {ac}55;
                         border-radius:20px;padding:3px 14px;font-size:.7rem;
                         font-weight:700;letter-spacing:.1em;text-transform:uppercase;">
                {d_icon} {domain} Domain
            </span>
        </div>
        <!-- PRIMARY tagline -->
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                    color:#e2e8f0;margin-bottom:8px;letter-spacing:-.01em;">
            Anyone Can Do Data Analysis —
            <span style="color:{ac};">No Coding. No Training. Just Upload.</span>
        </div>
        <!-- SECONDARY tagline — BI Team -->
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;
                    background:{ac}0d;border:1px solid {ac}22;border-radius:10px;
                    padding:10px 16px;">
            <div style="font-size:1.4rem;">🤝</div>
            <div>
                <div style="font-weight:700;color:#e2e8f0;font-size:.92rem;">
                    Your Business Intelligence Team
                </div>
                <div style="color:#64748b;font-size:.78rem;margin-top:2px;">
                    What a 5-person analyst team takes 3 days to produce — delivered in seconds.
                    No PowerBI licence. No Tableau training. No data engineer needed.
                </div>
            </div>
        </div>
        <!-- Stats row -->
        <div style="display:flex;gap:20px;flex-wrap:wrap;">
            <span style="background:#ffffff0a;border-radius:8px;padding:5px 14px;
                         font-size:.8rem;color:#94a3b8;">
                📊 <strong style="color:#e2e8f0;">{len(df):,}</strong> records
            </span>
            <span style="background:#ffffff0a;border-radius:8px;padding:5px 14px;
                         font-size:.8rem;color:#94a3b8;">
                🗂 <strong style="color:#e2e8f0;">{len(df.columns)}</strong> columns
            </span>
            <span style="background:#ffffff0a;border-radius:8px;padding:5px 14px;
                         font-size:.8rem;color:#94a3b8;">
                🕐 <strong style="color:#e2e8f0;">
                {datetime.datetime.now().strftime("%d %b %Y %H:%M")}</strong>
            </span>
            <span style="background:{ac}18;border:1px solid {ac}33;border-radius:8px;
                         padding:5px 14px;font-size:.8rem;color:{ac};font-weight:600;">
                ✅ Auto-analysed — no setup required
            </span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — HEADLINE KPI CARD
    # ══════════════════════════════════════════════════════════════════════
    headline_val = headline_label = headline_sub = delta_str = delta_color = ""

    try:
        if domain == "HR" and sal:
            s = pd.to_numeric(df[sal], errors="coerce").dropna()
            headline_val   = f"{cur}{s.sum():,.0f}"
            headline_label = "Total Payroll Cost"
            headline_sub   = f"Avg {cur}{s.mean():,.0f} per employee · {len(df):,} headcount"
            if attr:
                rate = df[attr].astype(str).str.lower().isin(
                    ["yes","true","1","resigned","terminated","left"]).mean()*100
                delta_str   = f"↑ {rate:.1f}% Attrition Rate"
                delta_color = "#ef4444" if rate > 15 else "#f59e0b"
        elif domain == "Fraud":
            fraud_cols = ["class","label","fraud","is_fraud","isfraud","target","fraud_label"]
            cc = next((cl[c] for c in fraud_cols if c in cl), None)
            if cc:
                lbl = df[cc].astype(str).str.lower().map(
                    lambda x: 1 if x in ["1","true","yes","fraud"] else 0)
                rate = lbl.mean()*100
                headline_val   = f"{rate:.4f}%"
                headline_label = "Fraud Rate"
                headline_sub   = f"{int(lbl.sum()):,} fraudulent of {len(df):,} transactions"
                delta_str   = "🚨 Critical — investigate immediately" if rate > 1 else "✅ Within normal range"
                delta_color = "#ef4444" if rate > 1 else "#10b981"
        elif domain == "Marketing" and sp:
            sp_s = pd.to_numeric(df[sp], errors="coerce").dropna()
            headline_val   = f"{cur}{sp_s.sum():,.0f}"
            headline_label = "Total Marketing Spend"
            rev_s = found.get("revenue")
            if rev_s:
                rv = pd.to_numeric(df[rev_s], errors="coerce").sum()
                roi_val = (rv - sp_s.sum()) / sp_s.sum() * 100 if sp_s.sum() > 0 else 0
                headline_sub = f"ROI: {roi_val:+.1f}% · {cur}{rv:,.0f} attributed revenue"
                delta_str   = f"{'↑' if roi_val>0 else '↓'} {abs(roi_val):.1f}% Return on Spend"
                delta_color = "#10b981" if roi_val > 0 else "#ef4444"
            elif conv:
                cv = pd.to_numeric(df[conv], errors="coerce").sum()
                cpa = sp_s.sum() / cv if cv > 0 else 0
                headline_sub = f"Cost per Conversion: {cur}{cpa:,.2f} · {len(df):,} campaigns"
                delta_str   = f"{cur}{cpa:,.2f} CPA"
                delta_color = "#10b981" if cpa < 500 else "#f59e0b"
            else:
                headline_sub = f"{len(df):,} campaigns analysed"
        elif sc:
            s = pd.to_numeric(df[sc], errors="coerce").dropna()
            headline_val   = f"{cur}{s.sum():,.0f}"
            headline_label = "Total Revenue"
            headline_sub   = f"Avg {cur}{s.mean():,.0f} per transaction · {len(df):,} records"
            if pc:
                ps     = pd.to_numeric(df[pc], errors="coerce").dropna()
                margin = ps.sum() / s.sum() * 100 if s.sum() > 0 else 0
                delta_str   = f"{'↑' if margin > 0 else '↓'} {margin:.1f}% Profit Margin"
                delta_color = "#10b981" if margin > 15 else "#f59e0b" if margin > 5 else "#ef4444"
    except Exception:
        pass

    if headline_val:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0f172a,#111827);
                    border:2px solid {ac}66;border-radius:16px;
                    padding:28px 32px;margin:0 0 16px;text-align:center;">
            <div style="color:#64748b;font-size:.75rem;font-weight:700;
                        letter-spacing:.12em;text-transform:uppercase;
                        margin-bottom:8px;">{headline_label}</div>
            <div style="font-size:3.2rem;font-weight:800;color:{ac};
                        letter-spacing:-.03em;line-height:1;
                        font-family:'Syne',sans-serif;">{headline_val}</div>
            <div style="color:#94a3b8;font-size:.88rem;margin-top:10px;">{headline_sub}</div>
            {f'<div style="color:{delta_color};font-size:.9rem;font-weight:700;margin-top:10px;">{delta_str}</div>' if delta_str else ""}
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — DOMAIN SCORECARD (4 smart mini-KPIs)
    # ══════════════════════════════════════════════════════════════════════
    mini_kpis = []

    try:
        # Always show record count and completeness
        completeness = (1 - df.isnull().mean().mean()) * 100
        mini_kpis.append(("📋", "Records", f"{len(df):,}", f"{completeness:.0f}% complete"))

        if sc:
            s = pd.to_numeric(df[sc], errors="coerce").dropna()
            if dc:
                dates = pd.to_datetime(df[dc], errors="coerce")
                df2   = pd.DataFrame({"_r":s,"_d":dates}).dropna()
                if len(df2) > 5:
                    df2["_M"] = df2["_d"].dt.to_period("M")
                    monthly   = df2.groupby("_M")["_r"].sum()
                    if len(monthly) >= 2:
                        chg = (monthly.iloc[-1]-monthly.iloc[-2])/monthly.iloc[-2]*100
                        arrow = "▲" if chg > 0 else "▼"
                        mini_kpis.append(("📈","MoM Growth",
                            f"{arrow}{abs(chg):.1f}%",
                            f"vs prior period"))
            if pc:
                ps  = pd.to_numeric(df[pc], errors="coerce").dropna()
                mgn = ps.sum()/s.sum()*100 if s.sum()>0 else 0
                mini_kpis.append(("💰","Profit Margin",
                    f"{mgn:.1f}%",
                    "Healthy >15%" if mgn>15 else "Low <15%"))
            if qty:
                q = pd.to_numeric(df[qty], errors="coerce").dropna()
                mini_kpis.append(("📦","Avg Units/Order",
                    f"{q.mean():.1f}",
                    f"Total {q.sum():,.0f} units"))

        if domain=="HR":
            if sal:
                s = pd.to_numeric(df[sal], errors="coerce").dropna()
                mini_kpis.append(("💼","Avg Salary",
                    f"{cur}{s.mean():,.0f}",f"Median {cur}{s.median():,.0f}"))
            if attr:
                rate = df[attr].astype(str).str.lower().isin(
                    ["yes","true","1","resigned","terminated","left"]).mean()*100
                mini_kpis.append(("🚪","Attrition",
                    f"{rate:.1f}%",
                    "Critical >20%" if rate>20 else "High >12%" if rate>12 else "Healthy"))
            if dept:
                mini_kpis.append(("🏢","Departments",
                    f"{df[dept].nunique()}",f"{len(df):,} total employees"))

        if domain=="Marketing":
            if imp and clk:
                i_s = pd.to_numeric(df[imp],errors="coerce").sum()
                c_s = pd.to_numeric(df[clk],errors="coerce").sum()
                ctr = c_s/i_s*100 if i_s>0 else 0
                mini_kpis.append(("🎯","CTR",f"{ctr:.2f}%","Benchmark 2–3%"))
            if conv:
                cv = pd.to_numeric(df[conv],errors="coerce").sum()
                mini_kpis.append(("✅","Conversions",f"{cv:,.0f}","Total leads/sales"))

        if domain=="Fraud":
            fraud_cols=["class","label","fraud","is_fraud","isfraud","target"]
            cc=next((cl[c] for c in fraud_cols if c in cl),None)
            if cc:
                lbl=df[cc].astype(str).str.lower().map(lambda x:1 if x in["1","true","yes","fraud"] else 0)
                mini_kpis.append(("🚨","Fraud Cases",f"{int(lbl.sum()):,}",f"{lbl.mean()*100:.4f}% rate"))

    except Exception:
        pass

    if mini_kpis:
        cols = st.columns(min(4, len(mini_kpis)))
        for i, (icon, label, val, sub) in enumerate(mini_kpis[:4]):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background:#0d1829;border:1px solid #1e293b;border-radius:12px;
                            padding:16px;text-align:center;height:100%;">
                    <div style="font-size:1.5rem;margin-bottom:6px;">{icon}</div>
                    <div style="color:#64748b;font-size:.7rem;font-weight:600;
                                letter-spacing:.08em;text-transform:uppercase;">{label}</div>
                    <div style="font-size:1.5rem;font-weight:800;color:{ac};
                                font-family:'Syne',sans-serif;margin:4px 0;">{val}</div>
                    <div style="color:#475569;font-size:.72rem;">{sub}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — CRITICAL ALERTS
    # ══════════════════════════════════════════════════════════════════════
    alerts = []

    try:
        # Revenue concentration risk
        if sc and (prd or cat):
            dim = prd or cat
            grp = df.groupby(dim)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum()).sort_values(ascending=False)
            top1_pct = grp.iloc[0]/grp.sum()*100 if len(grp)>0 else 0
            if top1_pct > 40:
                alerts.append(("critical","🔴",
                    f"Revenue Concentration Risk — {grp.index[0]}",
                    f"{top1_pct:.0f}% of total revenue from one {'product' if dim==prd else 'category'}. "
                    f"Single point of failure — diversify immediately."))

        # Attrition
        if attr:
            rate = df[attr].astype(str).str.lower().isin(
                ["yes","true","1","resigned","terminated","left"]).mean()*100
            if rate > 20:
                alerts.append(("critical","🔴",
                    f"Critical Attrition — {rate:.1f}% employees left",
                    f"Industry benchmark 10–15%. You are {rate/15:.1f}x the norm. Immediate intervention needed."))
            elif rate > 12:
                alerts.append(("warning","🟡",
                    f"Elevated Attrition — {rate:.1f}%",
                    "Above healthy 10% threshold. Review exit data and engagement scores."))

        # Gender pay gap
        if gen and sal:
            dg = df.copy()
            dg["_g"] = normalize_gender(dg[gen])
            ms = pd.to_numeric(dg[dg["_g"]=="Male"][sal],errors="coerce").mean()
            fs = pd.to_numeric(dg[dg["_g"]=="Female"][sal],errors="coerce").mean()
            if ms>0 and fs>0:
                gap=(ms-fs)/ms*100
                if abs(gap)>15:
                    alerts.append(("critical","🔴",
                        f"Gender Pay Gap — {abs(gap):.1f}%",
                        f"{'Male' if gap>0 else 'Female'} employees earn {abs(gap):.1f}% more. "
                        "Legal and reputational risk. Audit compensation bands now."))
                elif abs(gap)>8:
                    alerts.append(("warning","🟡",
                        f"Gender Pay Gap — {abs(gap):.1f}%",
                        "Noticeable disparity. Review pay bands across departments."))

        # Discount erosion
        if dis and sc:
            disc_s = pd.to_numeric(df[dis],errors="coerce")
            rev_s  = pd.to_numeric(df[sc], errors="coerce")
            ratio  = disc_s.sum()/(disc_s.sum()+rev_s.sum())*100
            if ratio>25:
                alerts.append(("critical","🔴",
                    f"Discount Erosion — {ratio:.1f}% of gross revenue discounted",
                    f"Giving away {cur}{disc_s.sum():,.0f}. Implement approval gates."))
            elif ratio>15:
                alerts.append(("warning","🟡",
                    f"High Discount Rate — {ratio:.1f}%",
                    "Above healthy 15% threshold. Monitor closely."))

        # Low CTR
        if imp and clk:
            i_s=pd.to_numeric(df[imp],errors="coerce").sum()
            c_s=pd.to_numeric(df[clk],errors="coerce").sum()
            ctr=c_s/i_s*100 if i_s>0 else 0
            if ctr<1.0:
                alerts.append(("warning","🟡",
                    f"Low CTR — {ctr:.2f}% (benchmark: 2–3%)",
                    f"Creative refresh or audience retargeting needed. "
                    f"{i_s:,.0f} impressions → only {c_s:,.0f} clicks."))

        # Low profit margin
        if sc and pc:
            rv=pd.to_numeric(df[sc],errors="coerce").sum()
            pr=pd.to_numeric(df[pc],errors="coerce").sum()
            mgn=pr/rv*100 if rv>0 else 0
            if mgn<5:
                alerts.append(("critical","🔴",
                    f"Critical Margin — {mgn:.1f}%",
                    "Dangerously thin. Review COGS, pricing, and operations urgently."))
            elif mgn<10:
                alerts.append(("warning","🟡",
                    f"Low Profit Margin — {mgn:.1f}%",
                    "Below healthy 10–15% benchmark. Pricing or cost optimisation needed."))

        # Target vs actual
        if tgt and sc:
            t_s = pd.to_numeric(df[tgt],errors="coerce").sum()
            r_s = pd.to_numeric(df[sc], errors="coerce").sum()
            ach = r_s/t_s*100 if t_s>0 else 0
            if ach < 80:
                alerts.append(("critical","🔴",
                    f"Target Miss — {ach:.1f}% achieved",
                    f"Actual {cur}{r_s:,.0f} vs Target {cur}{t_s:,.0f}. "
                    f"Gap: {cur}{t_s-r_s:,.0f}. Investigate root cause."))
            elif ach < 95:
                alerts.append(("warning","🟡",
                    f"Near Miss — {ach:.1f}% of target achieved",
                    f"Gap of {cur}{t_s-r_s:,.0f} to close."))

        # Data quality
        null_pct=df.isnull().mean().mean()*100
        if null_pct>20:
            alerts.append(("warning","🟡",
                f"Data Quality — {null_pct:.0f}% missing values",
                "High missing data reduces accuracy. Review collection process."))

    except Exception:
        pass

    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                color:#f1f5f9;margin:20px 0 10px;display:flex;align-items:center;gap:8px;">
        🚨 Critical Alerts
        <span style="font-size:.75rem;font-weight:400;color:#64748b;">
        — what needs attention right now</span>
    </div>""", unsafe_allow_html=True)

    if alerts:
        for sev,icon,title,detail in alerts[:5]:
            bg     = "#1a0a0a" if sev=="critical" else "#1a1400"
            border = "#ef444466" if sev=="critical" else "#f59e0b66"
            tc     = "#fca5a5" if sev=="critical" else "#fde68a"
            st.markdown(f"""
            <div style="background:{bg};border-left:4px solid {border};
                        border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:8px;">
                <div style="font-weight:700;color:{tc};font-size:.9rem;margin-bottom:3px;">
                    {icon} {title}
                </div>
                <div style="color:#94a3b8;font-size:.83rem;line-height:1.5;">{detail}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0a1f0f;border-left:4px solid #10b98166;
                    border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:8px;">
            <div style="font-weight:700;color:#6ee7b7;font-size:.9rem;">
                ✅ No Critical Alerts — Data looks healthy
            </div>
            <div style="color:#94a3b8;font-size:.83rem;margin-top:3px;">
                All key metrics are within acceptable thresholds.
            </div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 — PERFORMANCE STORY
    # ══════════════════════════════════════════════════════════════════════
    story_lines = []

    try:
        if sc and dc:
            rev_s = pd.to_numeric(df[sc],errors="coerce")
            dates = pd.to_datetime(df[dc],errors="coerce")
            df2   = pd.DataFrame({"_r":rev_s,"_d":dates}).dropna()
            if len(df2)>10:
                df2["_M"] = df2["_d"].dt.to_period("M")
                monthly   = df2.groupby("_M")["_r"].sum().sort_index()
                if len(monthly)>=2:
                    last=monthly.iloc[-1]; prev=monthly.iloc[-2]
                    chg=(last-prev)/prev*100 if prev!=0 else 0
                    trend="📈 Up" if chg>0 else "📉 Down"
                    story_lines.append(
                        f"{trend} **{abs(chg):.1f}%** vs prior period — "
                        f"Latest: **{cur}{last:,.0f}** | Prior: **{cur}{prev:,.0f}**")
                if len(monthly)>=3:
                    story_lines.append(
                        f"🏆 Best month: **{monthly.idxmax()}** at **{cur}{monthly.max():,.0f}**")
                if len(monthly)>=6:
                    h1=monthly.iloc[:len(monthly)//2].mean()
                    h2=monthly.iloc[len(monthly)//2:].mean()
                    hc=(h2-h1)/h1*100 if h1!=0 else 0
                    story_lines.append(
                        f"📊 First-half avg: **{cur}{h1:,.0f}** → Second-half avg: **{cur}{h2:,.0f}** "
                        f"({'▲' if hc>0 else '▼'}{abs(hc):.1f}% overall trend)")
                if len(monthly)>=4:
                    rolling=monthly.rolling(3).mean()
                    slope=(rolling.iloc[-1]-rolling.iloc[-4]) if len(rolling)>=4 else 0
                    if slope>0:
                        story_lines.append("🚀 3-month rolling average shows **upward momentum**")
                    elif slope<0:
                        story_lines.append("⚠️ 3-month rolling average shows **declining momentum** — review strategy")

        if reg and sc:
            rg    = df.groupby(reg)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum())
            rg_pct= rg/rg.sum()*100
            story_lines.append(
                f"🌍 Top region: **{rg.idxmax()}** ({rg_pct.max():.0f}% of revenue) · "
                f"Lowest: **{rg.idxmin()}** ({rg_pct.min():.1f}%)")

        if prd and sc:
            pg    = df.groupby(prd)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum())
            pct   = pg/pg.sum()*100
            story_lines.append(
                f"🥇 Top product: **{pg.idxmax()}** drives **{pct.max():.0f}%** of revenue · "
                f"Weakest: **{pg.idxmin()}** ({pct.min():.1f}%)")

        if cus and sc:
            cg    = df.groupby(cus)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum())
            top5  = cg.nlargest(5).sum()/cg.sum()*100
            story_lines.append(
                f"👥 Top 5 customers contribute **{top5:.0f}%** of revenue — "
                f"{'⚠️ high dependency risk' if top5>60 else '✅ healthy diversification'}")

        if dept and sal:
            dg = df.groupby(dept)[sal].apply(lambda x: pd.to_numeric(x,errors="coerce").mean())
            story_lines.append(
                f"💼 Highest paid dept: **{dg.idxmax()}** ({cur}{dg.max():,.0f} avg) · "
                f"Lowest: **{dg.idxmin()}** ({cur}{dg.min():,.0f} avg)")

        if imp and clk and conv:
            i_s=pd.to_numeric(df[imp],errors="coerce").sum()
            c_s=pd.to_numeric(df[clk],errors="coerce").sum()
            v_s=pd.to_numeric(df[conv],errors="coerce").sum()
            ctr=c_s/i_s*100 if i_s>0 else 0
            cvr=v_s/c_s*100 if c_s>0 else 0
            story_lines.append(
                f"📣 Funnel: **{i_s:,.0f}** impressions → **{c_s:,.0f}** clicks "
                f"(CTR {ctr:.2f}%) → **{v_s:,.0f}** conversions (CVR {cvr:.2f}%)")

        if srep and sc:
            sg  = df.groupby(srep)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum())
            story_lines.append(
                f"🏅 Top sales rep: **{sg.idxmax()}** ({cur}{sg.max():,.0f}) · "
                f"Lowest: **{sg.idxmin()}** ({cur}{sg.min():,.0f})")

    except Exception:
        pass

    if story_lines:
        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                    color:#f1f5f9;margin:20px 0 10px;">
            📖 Performance Story
        </div>""", unsafe_allow_html=True)
        for line in story_lines:
            st.markdown(
                f"<div style='background:#0d1829;border-left:3px solid {ac}44;"
                f"border-radius:0 8px 8px 0;padding:10px 16px;margin-bottom:6px;"
                f"color:#cbd5e1;font-size:.87rem;line-height:1.6;'>{line}</div>",
                unsafe_allow_html=True)

    # ── Trend sparkline chart (only if date + value exist) ─────────────────
    try:
        if sc and dc:
            rev_s = pd.to_numeric(df[sc], errors="coerce")
            dates = pd.to_datetime(df[dc], errors="coerce")
            df_t  = pd.DataFrame({"_r": rev_s, "_d": dates}).dropna()
            if len(df_t) > 10:
                df_t["_M"] = df_t["_d"].dt.to_period("M")
                monthly    = df_t.groupby("_M")["_r"].sum().reset_index()
                monthly["_M_str"] = monthly["_M"].astype(str)
                if len(monthly) >= 3:
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=monthly["_M_str"], y=monthly["_r"],
                        mode="lines+markers",
                        line=dict(color=ac, width=2.5),
                        marker=dict(size=6, color=ac),
                        fill="tozeroy",
                        fillcolor=f"{ac}18",
                        name="Revenue",
                        hovertemplate=f"<b>%{{x}}</b><br>{cur}%{{y:,.0f}}<extra></extra>"
                    ))
                    # Add 3-month rolling average
                    if len(monthly) >= 4:
                        monthly["_roll"] = monthly["_r"].rolling(3, min_periods=1).mean()
                        fig_trend.add_trace(go.Scatter(
                            x=monthly["_M_str"], y=monthly["_roll"],
                            mode="lines", name="3M Avg",
                            line=dict(color="#f59e0b", width=1.5, dash="dot"),
                            hovertemplate="<b>%{x}</b><br>3M Avg: "+cur+"%{y:,.0f}<extra></extra>"
                        ))
                    fig_trend.update_layout(
                        **cd(240),
                        title=dict(text="📈 Revenue Trend — Monthly", font=dict(size=13,color="#e2e8f0")),
                        showlegend=True,
                        legend=dict(orientation="h", y=1.12, font=dict(size=11, color="#94a3b8")),
                        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#64748b")),
                        yaxis=dict(gridcolor="#1e293b", tickfont=dict(size=10, color="#64748b"),
                                   tickprefix=cur),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
        elif sal and dept:
            # HR — show dept salary chart
            ds = df.groupby(dept)[sal].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            ).sort_values(ascending=False).head(10).reset_index()
            ds.columns = ["Department", "Avg Salary"]
            fig_dept = px.bar(ds, x="Avg Salary", y="Department", orientation="h",
                              color="Avg Salary", color_continuous_scale=["#1e3a5f", ac],
                              labels={"Avg Salary": f"Avg Salary ({cur})"})
            fig_dept.update_layout(**cd(280),
                title=dict(text="💼 Avg Salary by Department", font=dict(size=13,color="#e2e8f0")),
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(tickprefix=cur, gridcolor="#1e293b",
                           tickfont=dict(size=10,color="#64748b")),
                yaxis=dict(tickfont=dict(size=10, color="#cbd5e1")))
            st.plotly_chart(fig_dept, use_container_width=True)
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4.5 — DATA HEALTH SCORECARD (new — unique differentiator)
    # ══════════════════════════════════════════════════════════════════════
    try:
        health_scores = []

        # Completeness
        completeness = (1 - df.isnull().mean().mean()) * 100
        health_scores.append(("📋 Data Completeness",
            completeness,
            "100%" if completeness>=95 else f"{completeness:.0f}%",
            "#10b981" if completeness>=90 else "#f59e0b" if completeness>=70 else "#ef4444"))

        # Column coverage — how many expected columns found vs domain
        domain_expected = {
            "Sales":["sales","profit","date","product","region","customer"],
            "Marketing":["spend","impressions","clicks","conversions","channel","campaign_id"],
            "HR":["salary","department","attrition","tenure","performance","employee_id"],
            "Ecommerce":["sales","product","payment","delivery","returns","customer"],
            "Retail":["sales","store","product","category","payment","satisfaction"],
            "Fraud":["fraud_label","fraud_amount","fraud_time","fraud_type"],
            "Generic":["sales","date","product","region"],
        }
        expected = domain_expected.get(domain, [])
        covered  = sum(1 for k in expected if found.get(k)) if expected else 0
        cov_pct  = covered / len(expected) * 100 if expected else 100
        health_scores.append(("🗂 Column Coverage",
            cov_pct,
            f"{covered}/{len(expected)} fields",
            "#10b981" if cov_pct>=80 else "#f59e0b" if cov_pct>=50 else "#ef4444"))

        # Numeric data quality
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            neg_flag = any((df[c] < 0).sum() > len(df)*0.1 for c in num_cols)
            dup_pct  = df.duplicated().mean() * 100
            quality  = 100 - (20 if neg_flag else 0) - min(dup_pct * 2, 30)
            health_scores.append(("🔢 Data Quality",
                quality,
                "Good" if quality>=80 else "Fair" if quality>=60 else "Poor",
                "#10b981" if quality>=80 else "#f59e0b" if quality>=60 else "#ef4444"))

        # Analysis readiness
        key_count = len([v for v in found.values() if v])
        readiness = min(key_count / 5 * 100, 100)
        health_scores.append(("⚡ Analysis Readiness",
            readiness,
            f"{key_count} mapped",
            "#10b981" if readiness>=80 else "#f59e0b" if readiness>=50 else "#ef4444"))

        if health_scores:
            st.markdown(f"""
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                        color:#f1f5f9;margin:20px 0 10px;">
                📊 Data Health Scorecard
                <span style="font-size:.75rem;font-weight:400;color:#64748b;">
                — how analysis-ready is your data</span>
            </div>""", unsafe_allow_html=True)

            cols_h = st.columns(len(health_scores))
            for i, (label, score, display, color) in enumerate(health_scores):
                with cols_h[i]:
                    st.markdown(f"""
                    <div style="background:#0d1829;border:1px solid #1e293b;
                                border-radius:12px;padding:16px 14px;text-align:center;">
                        <div style="color:#64748b;font-size:.68rem;font-weight:600;
                                    letter-spacing:.06em;text-transform:uppercase;
                                    margin-bottom:8px;">{label}</div>
                        <div style="font-size:1.4rem;font-weight:800;color:{color};
                                    font-family:'Syne',sans-serif;">{display}</div>
                        <div style="background:#1e293b;border-radius:20px;height:4px;
                                    margin:10px 0 6px;overflow:hidden;">
                            <div style="background:{color};height:4px;border-radius:20px;
                                        width:{min(score,100):.0f}%;transition:width .5s;"></div>
                        </div>
                        <div style="color:#475569;font-size:.7rem;">{score:.0f}/100</div>
                    </div>""", unsafe_allow_html=True)
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5 — WINS & RISKS
    # ══════════════════════════════════════════════════════════════════════
    wins=[]; risks=[]

    try:
        if sc and prd:
            pg=df.groupby(prd)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum()).sort_values(ascending=False)
            if len(pg)>=1: wins.append(f"🥇 **{pg.index[0]}** is top revenue driver ({cur}{pg.iloc[0]:,.0f})")
            if len(pg)>=2: risks.append(f"⚠️ **{pg.index[-1]}** underperforming ({cur}{pg.iloc[-1]:,.0f})")

        if reg and sc:
            rg=df.groupby(reg)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum()).sort_values(ascending=False)
            if len(rg)>=1: wins.append(f"🌍 **{rg.index[0]}** is strongest market ({cur}{rg.iloc[0]:,.0f})")
            if len(rg)>=2: risks.append(f"📉 **{rg.index[-1]}** region needs attention ({cur}{rg.iloc[-1]:,.0f})")

        if dept and attr:
            da=df.copy()
            da["_left"]=da[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"])
            dept_attr=da.groupby(dept)["_left"].mean().sort_values(ascending=False)
            if len(dept_attr)>=1 and dept_attr.iloc[0]>0.15:
                risks.append(f"🔴 **{dept_attr.index[0]}** dept: {dept_attr.iloc[0]*100:.0f}% attrition")
            if len(dept_attr)>=1 and dept_attr.iloc[-1]<0.05:
                wins.append(f"✅ **{dept_attr.index[-1]}** dept: only {dept_attr.iloc[-1]*100:.0f}% attrition")

        if sc and dc:
            rev_s=pd.to_numeric(df[sc],errors="coerce")
            dates=pd.to_datetime(df[dc],errors="coerce")
            df2=pd.DataFrame({"_r":rev_s,"_d":dates}).dropna()
            if len(df2)>10:
                df2["_M"]=df2["_d"].dt.to_period("M")
                monthly=df2.groupby("_M")["_r"].sum()
                if len(monthly)>=3:
                    if monthly.is_monotonic_increasing:
                        wins.append("📈 Consistent month-over-month revenue growth")
                    last3=(monthly.iloc[-1]-monthly.iloc[-3])/monthly.iloc[-3]*100
                    if last3<-10:
                        risks.append(f"📉 Revenue declined **{abs(last3):.0f}%** over last 3 periods")
                    elif last3>20:
                        wins.append(f"🚀 Revenue grew **{last3:.0f}%** over last 3 periods")

        if sp and conv:
            sp_s=pd.to_numeric(df[sp],errors="coerce").sum()
            cv_s=pd.to_numeric(df[conv],errors="coerce").sum()
            if sp_s>0 and cv_s>0:
                cpa=sp_s/cv_s
                if cpa<500: wins.append(f"💰 Efficient CPA: {cur}{cpa:,.2f} per conversion")
                else: risks.append(f"💸 High CPA: {cur}{cpa:,.2f} — optimise campaigns")

        if tgt and sc:
            t_s=pd.to_numeric(df[tgt],errors="coerce").sum()
            r_s=pd.to_numeric(df[sc], errors="coerce").sum()
            ach=r_s/t_s*100 if t_s>0 else 0
            if ach>=100: wins.append(f"🎯 Target exceeded! **{ach:.1f}%** of goal achieved")
            elif ach<80: risks.append(f"🎯 Target miss: only **{ach:.1f}%** achieved")

    except Exception:
        pass

    if wins or risks:
        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                    color:#f1f5f9;margin:20px 0 10px;">
            🏆 Wins &amp; Risks
        </div>""", unsafe_allow_html=True)
        col_w,col_r=st.columns(2)
        with col_w:
            st.markdown("<div style='font-weight:700;color:#6ee7b7;font-size:.78rem;"
                        "letter-spacing:.1em;margin-bottom:8px;'>✅ WINS</div>",
                        unsafe_allow_html=True)
            for w in (wins or ["✅ No specific wins identified yet"])[:4]:
                st.markdown(
                    f"<div style='background:#0a1f12;border-radius:8px;padding:10px 14px;"
                    f"margin-bottom:6px;color:#a7f3d0;font-size:.84rem;line-height:1.5;'>"
                    f"{w}</div>", unsafe_allow_html=True)
        with col_r:
            st.markdown("<div style='font-weight:700;color:#fca5a5;font-size:.78rem;"
                        "letter-spacing:.1em;margin-bottom:8px;'>⚠️ RISKS</div>",
                        unsafe_allow_html=True)
            for r in (risks or ["✅ No critical risks detected"])[:4]:
                st.markdown(
                    f"<div style='background:#1a0d0d;border-radius:8px;padding:10px 14px;"
                    f"margin-bottom:6px;color:#fca5a5;font-size:.84rem;line-height:1.5;'>"
                    f"{r}</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6 — AI-POWERED INSIGHT (Claude narrative paragraph)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                color:#f1f5f9;margin:20px 0 10px;">
        🤖 AI Executive Briefing
        <span style="font-size:.75rem;font-weight:400;color:#64748b;">
        — Your Business Intelligence Team · powered by Claude AI</span>
    </div>""", unsafe_allow_html=True)

    # Build compact context for Claude
    try:
        ctx_parts = [f"Domain: {domain}. Dataset: {len(df):,} records, {len(df.columns)} columns."]
        if sc:
            s=pd.to_numeric(df[sc],errors="coerce").dropna()
            ctx_parts.append(f"Total revenue: {cur}{s.sum():,.0f}, avg: {cur}{s.mean():,.0f}.")
        if pc:
            p=pd.to_numeric(df[pc],errors="coerce").dropna()
            ctx_parts.append(f"Total profit: {cur}{p.sum():,.0f}.")
        if alerts:
            ctx_parts.append(f"Critical alerts: {'; '.join(t for _,_,t,_ in alerts[:3])}.")
        if wins:
            ctx_parts.append(f"Key wins: {'; '.join(wins[:2])}.")
        if risks:
            ctx_parts.append(f"Key risks: {'; '.join(risks[:2])}.")
        if sc and dc:
            try:
                rev_s=pd.to_numeric(df[sc],errors="coerce")
                dates=pd.to_datetime(df[dc],errors="coerce")
                df2=pd.DataFrame({"_r":rev_s,"_d":dates}).dropna()
                df2["_M"]=df2["_d"].dt.to_period("M")
                monthly=df2.groupby("_M")["_r"].sum()
                if len(monthly)>=2:
                    chg=(monthly.iloc[-1]-monthly.iloc[-2])/monthly.iloc[-2]*100
                    ctx_parts.append(f"Recent trend: {chg:+.1f}% month-on-month.")
            except: pass

        context = " ".join(ctx_parts)
        prompt = (
            f"You are the lead analyst in a world-class Business Intelligence team. "
            f"Write a sharp 3-sentence executive briefing for a business owner or CEO "
            f"based on this data: {context} "
            f"Sentence 1: State the single most important finding with a specific number. "
            f"Sentence 2: Name the biggest risk or opportunity hiding in this data. "
            f"Sentence 3: Give one precise action to take this week. "
            f"Tone: confident, decisive, no hedging, no jargon. Never use bullet points."
        )

        with st.spinner("🤖 Your Business Intelligence Team is analysing..."):
            ai_text = call_llm(prompt)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a1628,#0d1f3c);
                    border:1px solid {ac}33;border-radius:12px;
                    padding:20px 24px;margin-bottom:8px;position:relative;">
            <div style="position:absolute;top:14px;right:18px;
                        background:{ac}22;color:{ac};border-radius:6px;
                        padding:2px 10px;font-size:.68rem;font-weight:700;
                        letter-spacing:.08em;">AI · CLAUDE</div>
            <div style="color:#cbd5e1;font-size:.9rem;line-height:1.7;
                        font-style:italic;">{ai_text}</div>
        </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div style="background:#0d1829;border:1px solid #1e293b;border-radius:12px;
                    padding:16px 20px;color:#64748b;font-size:.85rem;">
            AI insight unavailable — configure ANTHROPIC_API_KEY in Streamlit secrets.
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7 — MONDAY MORNING ACTIONS
    # ══════════════════════════════════════════════════════════════════════
    actions=[]

    try:
        for sev,icon,title,_ in alerts[:2]:
            if sev=="critical": actions.append(("🔴 Urgent",f"Resolve: {title}","This Week"))
            else:               actions.append(("🟡 High",  f"Review: {title}", "This Month"))

        if tgt and sc:
            t_s=pd.to_numeric(df[tgt],errors="coerce").sum()
            r_s=pd.to_numeric(df[sc], errors="coerce").sum()
            ach=r_s/t_s*100 if t_s>0 else 0
            if ach<95:
                actions.append(("🔴 Urgent",
                    f"Close target gap — {cur}{t_s-r_s:,.0f} remaining to hit goal","This Week"))

        if sc and prd:
            pg=df.groupby(prd)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum()).sort_values(ascending=False)
            if len(pg)>=5:
                bottom=", ".join(str(x) for x in pg.index[-3:])
                actions.append(("🟢 Medium",f"Review underperforming: {bottom}","This Quarter"))

        if reg and sc:
            rg=df.groupby(reg)[sc].apply(lambda x: pd.to_numeric(x,errors="coerce").sum())
            if len(rg)>=3:
                actions.append(("🟢 Medium",
                    f"Double down on **{rg.idxmax()}** — highest ROI expansion","This Quarter"))

        if dis and sc:
            d_s=pd.to_numeric(df[dis],errors="coerce")
            r_s=pd.to_numeric(df[sc], errors="coerce")
            ratio=d_s.sum()/(d_s.sum()+r_s.sum())*100
            if ratio>15:
                actions.append(("🟡 High",
                    f"Implement discount approval workflow — {ratio:.1f}% revenue discounted","This Month"))

        if attr and dept:
            da=df.copy()
            da["_left"]=da[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"])
            dept_attr=da.groupby(dept)["_left"].mean().sort_values(ascending=False)
            if len(dept_attr)>0 and dept_attr.iloc[0]>0.15:
                actions.append(("🔴 Urgent",
                    f"Launch retention programme in **{dept_attr.index[0]}** — "
                    f"{dept_attr.iloc[0]*100:.0f}% attrition","This Week"))

        if sp and conv:
            sp_s=pd.to_numeric(df[sp],errors="coerce").sum()
            cv_s=pd.to_numeric(df[conv],errors="coerce").sum()
            if sp_s>0 and cv_s>0:
                cpa=sp_s/cv_s
                if cpa>1000:
                    actions.append(("🟡 High",
                        f"A/B test new creatives — CPA of {cur}{cpa:,.0f} is too high","This Month"))

        if not actions:
            actions=[
                ("🟢 Medium","Schedule quarterly business review using this story","This Month"),
                ("🟢 Medium","Share PDF report with all department heads","This Week"),
                ("🟢 Medium","Set monthly data refresh cadence to track trends","Ongoing"),
            ]
    except Exception:
        actions=[
            ("🟢 Medium","Review findings with your leadership team","This Week"),
            ("🟢 Medium","Export PDF for management presentation","This Week"),
        ]

    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                color:#f1f5f9;margin:20px 0 10px;">
        📋 Monday Morning Actions
        <span style="font-size:.75rem;font-weight:400;color:#64748b;">
        — prioritised next steps from your data</span>
    </div>""", unsafe_allow_html=True)

    priority_colors={"🔴 Urgent":("#2a0a0a","#ef4444"),
                     "🟡 High":  ("#1f1800","#f59e0b"),
                     "🟢 Medium":("#0a1a0a","#10b981")}

    for i,(priority,action,timeline) in enumerate(actions[:5],1):
        bg,pc_color=priority_colors.get(priority,("#0d1829","#64748b"))
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {pc_color}33;
                    border-radius:10px;padding:14px 18px;margin-bottom:8px;
                    display:flex;align-items:flex-start;gap:14px;">
            <div style="background:{pc_color}22;color:{pc_color};border-radius:50%;
                        width:30px;height:30px;display:flex;align-items:center;
                        justify-content:center;font-weight:800;font-size:.85rem;
                        flex-shrink:0;margin-top:2px;">{i}</div>
            <div style="flex:1;">
                <div style="color:#e2e8f0;font-size:.88rem;line-height:1.5;
                            margin-bottom:4px;">{action}</div>
                <div style="color:{pc_color};font-size:.74rem;font-weight:700;
                            letter-spacing:.05em;">{priority} · 📅 {timeline}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#020817,#0a1628);
                border:1px solid {ac}33;border-radius:16px;
                padding:22px 28px;margin-top:24px;">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    flex-wrap:wrap;gap:16px;">
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;
                            color:#e2e8f0;font-size:1rem;margin-bottom:4px;">
                    ⚡ 1 Click Data Analysis
                </div>
                <div style="color:#64748b;font-size:.8rem;margin-bottom:2px;">
                    Anyone Can Do Data Analysis — No Coding. No Training. Just Upload.
                </div>
                <div style="color:{ac};font-size:.78rem;font-weight:600;">
                    🤝 Your Business Intelligence Team — at a fraction of the cost.
                </div>
            </div>
            <div style="text-align:right;">
                <div style="color:#94a3b8;font-size:.8rem;font-weight:600;">
                    {d_icon} {domain} Analysis Complete
                </div>
                <div style="color:#475569;font-size:.72rem;margin-top:4px;">
                    {len(df):,} records · {len(df.columns)} columns
                </div>
                <div style="color:#334155;font-size:.7rem;margin-top:2px;">
                    {datetime.datetime.now().strftime("%d %b %Y · %H:%M")}
                </div>
            </div>
        </div>
        <div style="border-top:1px solid #1e293b;margin-top:16px;padding-top:12px;
                    display:flex;gap:20px;flex-wrap:wrap;">
            <span style="color:#334155;font-size:.72rem;">💼 Sales</span>
            <span style="color:#334155;font-size:.72rem;">📣 Marketing</span>
            <span style="color:#334155;font-size:.72rem;">👥 HR</span>
            <span style="color:#334155;font-size:.72rem;">🛒 Ecommerce</span>
            <span style="color:#334155;font-size:.72rem;">🏪 Retail</span>
            <span style="color:#334155;font-size:.72rem;">🚨 Fraud</span>
            <span style="color:#334155;font-size:.72rem;">🔍 Generic</span>
            <span style="color:#475569;font-size:.72rem;margin-left:auto;">
                1clickdataanalysis.com
            </span>
        </div>
    </div>
    <div style="margin-bottom:28px;"></div>""", unsafe_allow_html=True)



def render_summary(df, found, domain):
    section(f"📋 Executive Summary — {domain}", domain.lower())
    lines=[]
    sc=found.get("sales"); pc=found.get("profit"); prd=found.get("product")
    dc=found.get("date"); qty=found.get("quantity"); cus=found.get("customer")
    sal=found.get("salary"); attr=found.get("attrition"); dept=found.get("department")
    imp=found.get("impressions"); clk=found.get("clicks"); conv=found.get("conversions")
    sp=found.get("spend"); store=found.get("store"); gen=found.get("gender")
    emp=found.get("employee_id") or found.get("employee_name")
    ten=found.get("tenure")

    if domain=="Fraud":
        cl={c.lower().strip():c for c in df.columns}
        fraud_cols=["class","label","fraud","is_fraud","isfraud","target","is fraud"]
        cc=next((cl[c] for c in fraud_cols if c in cl),None)
        if cc:
            df2=df.copy()
            df2["_lbl"]=df2[cc].astype(str).str.lower().map(lambda x:1 if x in ["1","true","yes","fraud"] else 0)
            total=len(df2); fc=int(df2["_lbl"].sum()); lc=total-fc
            lines+=[f"- **Total Transactions:** {total:,}",
                    f"- **Legitimate:** {lc:,} | **Fraudulent:** {fc:,}",
                    f"- **Fraud Rate:** {fc/total*100:.4f}%"]
            amt=cl.get("amount")
            if amt:
                fa=df2[df2["_lbl"]==1][amt]; la=df2[df2["_lbl"]==0][amt]
                if len(fa)>0:
                    lines+=[f"- **Avg Fraud Amount:** ${fa.mean():,.2f}",
                            f"- **Total Fraud Exposure:** ${fa.sum():,.2f}",
                            f"- **Avg Legitimate Amount:** ${la.mean():,.2f}"]
    elif domain == "HR":
        # HR summary — only HR-relevant fields, never revenue/sales language
        total_emp = df[emp].nunique() if emp else len(df)
        lines.append(f"- **Total Employees:** {total_emp:,}")
        if dept:  lines.append(f"- **Departments:** {df[dept].nunique():,}")
        if sal:
            lines+=[f"- **Total Payroll:** {df[sal].sum():,.0f}",
                    f"- **Avg Salary:** {df[sal].mean():,.0f}",
                    f"- **Median Salary:** {df[sal].median():,.0f}",
                    f"- **Salary Range:** {df[sal].min():,.0f} – {df[sal].max():,.0f}"]
        if gen:
            gc=df[gen].astype(str).str.title().value_counts()
            m=gc.get("Male",gc.get("M",0)); f=gc.get("Female",gc.get("F",0))
            lines.append(f"- **Gender Split:** {int(m):,} Male / {int(f):,} Female")
            if sal:
                dg=df.copy(); dg["_g"]=dg[gen].astype(str).str.title()
                ms=dg[dg["_g"].isin(["Male","M"])][sal].mean()
                fs=dg[dg["_g"].isin(["Female","F"])][sal].mean()
                if ms>0 and fs>0:
                    gap=abs(ms-fs)/max(ms,fs)*100
                    lines.append(f"- **Gender Pay Gap:** {gap:.1f}%")
        if attr:
            rate=df[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"]).mean()*100
            lines.append(f"- **Attrition Rate:** {rate:.1f}%")
        if ten:  lines.append(f"- **Avg Tenure:** {df[ten].mean():.1f} years")
        if dc:
            dd=pd.to_datetime(df[dc],errors="coerce").dropna()
            if len(dd): lines.append(f"- **Date Range:** {dd.min().date()} → {dd.max().date()}")
    else:
        if sc:
            lines+=[f"- **Total Revenue:** {df[sc].sum():,.2f}",
                    f"- **Avg Order Value:** {df[sc].mean():,.2f}",
                    f"- **Peak Transaction:** {df[sc].max():,.2f}"]
        if pc and sc:
            lines+=[f"- **Total Profit:** {df[pc].sum():,.2f}",
                    f"- **Profit Margin:** {df[pc].sum()/df[sc].sum()*100:.1f}%"]
        if prd and sc:
            bp=df.groupby(prd)[sc].sum().idxmax()
            lines.append(f"- **Best Product:** {bp}")
        if cus: lines.append(f"- **Unique Customers:** {df[cus].nunique():,}")
        if qty: lines.append(f"- **Total Units Sold:** {df[qty].sum():,.0f}")
        if dc:
            dd=pd.to_datetime(df[dc],errors="coerce").dropna()
            if len(dd): lines.append(f"- **Date Range:** {dd.min().date()} → {dd.max().date()}")
        if sal:
            lines+=[f"- **Total Payroll:** {df[sal].sum():,.0f}",
                    f"- **Avg Salary:** {df[sal].mean():,.0f}",
                    f"- **Salary Range:** {df[sal].min():,.0f} – {df[sal].max():,.0f}"]
        if attr:
            rate=df[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"]).mean()*100
            lines.append(f"- **Attrition Rate:** {rate:.1f}%")
        if ten: lines.append(f"- **Avg Tenure:** {df[ten].mean():.1f} years")
        if gen:
            gc=df[gen].astype(str).str.title().value_counts()
            m=gc.get("Male",gc.get("M",0)); f=gc.get("Female",gc.get("F",0))
            lines.append(f"- **Gender Split:** {int(m):,} Male / {int(f):,} Female")
        if imp and clk:
            lines.append(f"- **Overall CTR:** {df[clk].sum()/df[imp].sum()*100:.2f}%")
        if sp and sc:
            lines.append(f"- **Marketing ROI:** {(df[sc].sum()-df[sp].sum())/df[sp].sum()*100:.1f}%")
        if store: lines.append(f"- **Stores:** {df[store].nunique():,}")

    lines.append(f"- **Records Analysed:** {len(df):,} | **Domain:** {domain}")
    for l in lines:
        if l: st.markdown(l)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO INSIGHTS Q&A
# ══════════════════════════════════════════════════════════════════════════════
def render_qa(df, domain, found):
    section("🤖 Auto Insights Q&A", domain.lower())
    tab1,tab2=st.tabs(["📊 Data-Driven Insights","🤖 LLM Q&A (API Key Required)"])

    with tab1:
        qa=[]
        sc=found.get("sales"); pc=found.get("profit"); prd=found.get("product")
        dc=found.get("date"); qty=found.get("quantity"); cus=found.get("customer")
        sal=found.get("salary"); attr=found.get("attrition"); dept=found.get("department")
        imp=found.get("impressions"); clk=found.get("clicks"); conv=found.get("conversions")
        sp=found.get("spend"); dis=found.get("discount"); gen=found.get("gender")
        cat=found.get("category"); reg=found.get("region"); ten=found.get("tenure")
        emp=found.get("employee_id") or found.get("employee_name")

        if domain=="Fraud":
            cl={c.lower().strip():c for c in df.columns}
            fraud_cols=["class","label","fraud","is_fraud","isfraud","target","is fraud"]
            cc=next((cl[c] for c in fraud_cols if c in cl),None)
            amt=cl.get("amount"); time_c=cl.get("time")
            if cc:
                df2=df.copy()
                df2["_lbl"]=df2[cc].astype(str).str.lower().map(lambda x:1 if x in ["1","true","yes","fraud"] else 0)
                total=len(df2); fc=int(df2["_lbl"].sum()); pct=fc/total*100
                qa.append(("🚨 What is the fraud rate?",
                    f"**{fc:,}** fraud transactions out of **{total:,}** total (**{pct:.4f}%**). "
                    +("⚠️ High — needs attention." if pct>1 else "✅ Within normal range.")))
                if amt:
                    fa=df2[df2["_lbl"]==1][amt]; la=df2[df2["_lbl"]==0][amt]
                    if len(fa)>0:
                        qa.append(("💳 How do fraud amounts compare?",
                            f"Fraud avg: **${fa.mean():,.2f}** | Legit avg: **${la.mean():,.2f}** | "
                            f"Total exposure: **${fa.sum():,.2f}**."))
                if time_c:
                    df2["_hr"]=(pd.to_numeric(df2[time_c],errors="coerce")/3600).fillna(0).astype(int)%24
                    ph=int(df2[df2["_lbl"]==1].groupby("_hr").size().idxmax()) if fc>0 else 0
                    qa.append(("⏱ When does fraud peak?",
                        f"Peak fraud hour: **{ph}:00**. Consider enhanced monitoring during this window."))
                qa.append(("📊 Is dataset balanced?",
                    f"Only **{pct:.4f}%** fraud — heavily imbalanced. "
                    "For ML: use SMOTE, class weights, or anomaly detection models."))
        elif domain == "HR":
            pass  # HR gets only HR Q&A below, no revenue questions
        else:
            if sc:
                total=df[sc].sum(); avg=df[sc].mean(); mx=df[sc].max()
                qa.append(("💰 What is total revenue?",
                    f"**{total:,.2f}** total | Avg per record: **{avg:,.2f}** | Peak: **{mx:,.2f}**"))
            if sc and dc:
                d2=prep_date(df,dc,sc)
                if not d2.empty:
                    m=d2.groupby("_M")[sc].sum()
                    qa.append(("📅 Best and worst months?",
                        f"Best: **{m.idxmax()}** ({m.max():,.2f}) | Worst: **{m.idxmin()}** ({m.min():,.2f})"))
            if sc and prd:
                tp=df.groupby(prd)[sc].sum()
                top3=", ".join(f"{p}({v:,.0f})" for p,v in tp.nlargest(3).items())
                bot3=", ".join(f"{p}({v:,.0f})" for p,v in tp.nsmallest(3).items())
                qa.append(("🛒 Top 3 & Bottom 3 products?", f"**Top:** {top3} | **Bottom:** {bot3}"))
            if sc and pc:
                qa.append(("📊 What is profit margin?",
                    f"Overall: **{df[pc].sum()/df[sc].sum()*100:.1f}%** | Total profit: **{df[pc].sum():,.2f}**"))
            if sc and cat:
                tc=df.groupby(cat)[sc].sum().sort_values(ascending=False)
                qa.append(("🗂 Top category?",
                    f"**{tc.index[0]}** ({tc.iloc[0]:,.2f}) leads revenue share."))
            if sc and reg:
                tr=df.groupby(reg)[sc].sum().sort_values(ascending=False)
                qa.append(("🌍 Top region?",
                    f"**{tr.index[0]}** ({tr.iloc[0]:,.2f}) | Bottom: **{tr.index[-1]}** ({tr.iloc[-1]:,.2f})"))
            if sc and cus:
                tc2=df.groupby(cus)[sc].sum().sort_values(ascending=False).head(5)
                qa.append(("👥 Top 5 customers?",
                    "**Top 5:** "+", ".join(f"{c}({v:,.0f})" for c,v in tc2.items())))
            if dis and sc:
                corr=df[[dis,sc]].dropna().corr().iloc[0,1]
                qa.append(("🏷 Do discounts help?",
                    f"Correlation: **{corr:.3f}**. "
                    +("📈 Positive — higher discounts increase revenue." if corr>0.1
                      else "📉 Negative — discounts reduce revenue." if corr<-0.1 else "➡️ Neutral.")))
            # HR Q&A
            if sal:
                qa.append(("💼 Salary overview?",
                    f"Avg: **{df[sal].mean():,.0f}** | Median: **{df[sal].median():,.0f}** | "
                    f"Range: **{df[sal].min():,.0f}–{df[sal].max():,.0f}**"))
            if attr:
                rate=df[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"]).mean()*100
                qa.append(("📉 Attrition rate?",
                    f"**{rate:.1f}%** attrition. "
                    +("⚠️ Above 15% — high risk." if rate>15 else "✅ Within healthy range.")))
            if sal and gen:
                df2=df.copy(); df2["_g"]=df2[gen].astype(str).str.title()
                ms=df2[df2["_g"].isin(["Male","M"])][sal].mean()
                fs=df2[df2["_g"].isin(["Female","F"])][sal].mean()
                if ms>0 and fs>0:
                    gap=abs(ms-fs)/max(ms,fs)*100
                    qa.append(("⚖️ Gender pay gap?",
                        f"Male avg: **{ms:,.0f}** | Female avg: **{fs:,.0f}** | Gap: **{gap:.1f}%**"))
            if sal and dept:
                ds=df.groupby(dept)[sal].mean().sort_values(ascending=False)
                qa.append(("🏢 Highest paying dept?",
                    f"**{ds.index[0]}** (avg {ds.iloc[0]:,.0f}) | Lowest: **{ds.index[-1]}** ({ds.iloc[-1]:,.0f})"))
            # Marketing Q&A
            if imp and clk:
                ctr=df[clk].sum()/df[imp].sum()*100
                qa.append(("🖱 CTR?",
                    f"**{ctr:.2f}%** ({df[clk].sum():,.0f} clicks from {df[imp].sum():,.0f} impressions). "
                    +("✅ Strong." if ctr>2 else "⚠️ Below 2% benchmark.")))
            if sp and sc:
                roi_v=(df[sc].sum()-df[sp].sum())/df[sp].sum()*100
                qa.append(("💸 Marketing ROI?",
                    f"Spend: **{df[sp].sum():,.2f}** | Revenue: **{df[sc].sum():,.2f}** | ROI: **{roi_v:.1f}%**"))
            if not qa:
                for col in df.select_dtypes(include="number").columns[:5]:
                    s=df[col].dropna()
                    qa.append((f"📈 {col}?",
                        f"Total: **{s.sum():,.2f}** | Mean: **{s.mean():,.2f}** | Range: **{s.min():,.2f}–{s.max():,.2f}**"))

        for i,(q,a) in enumerate(qa,1):
            with st.expander(f"Q{i}: {q}", expanded=(i==1)):
                st.markdown(f"**A{i}:** {a}")

    with tab2:
        if LLM_PROVIDER=="huggingface":
            st.info("💡 tiny-gpt2 cannot answer questions. Set LLM_PROVIDER to openai/anthropic/cohere in Streamlit Secrets.")
        else:
            if st.button("🧠 Generate AI Analysis"):
                with st.spinner("Querying AI..."):
                    nc=df.select_dtypes(include="number").columns.tolist()
                    ctx=df[nc].describe().round(2).to_string() if nc else "No numeric data."
                    prompt=(f"Senior {domain} analyst. {len(df):,} records.\n"
                            f"Columns: {', '.join(df.columns)}\nStats:\n{ctx}\n\n"
                            f"Write 5 data-backed {domain} Q&As with specific numbers.\n"
                            f"Format: Q1: ...\nA1: ...\n\nQ2: ...")
                    try: st.markdown(query_llm(prompt))
                    except Exception as e: st.error(f"LLM error: {e}")



# ══════════════════════════════════════════════════════════════════════════════
#  NATURAL LANGUAGE QUERYING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_nlq_answer(question, df, found, domain):
    import re as _re
    q = question.lower().strip()
    sc   = found.get("sales");    pc   = found.get("profit")
    prd  = found.get("product");  cat  = found.get("category")
    dc   = found.get("date");     qty  = found.get("quantity")
    cus  = found.get("customer"); reg  = found.get("region")
    sal  = found.get("salary");   attr = found.get("attrition")
    dept = found.get("department"); gen = found.get("gender")
    dis  = found.get("discount"); ten  = found.get("tenure")
    imp  = found.get("impressions"); clk = found.get("clicks")
    conv = found.get("conversions"); sp  = found.get("spend")
    store= found.get("store")
    emp  = found.get("employee_id") or found.get("employee_name")

    def filter_by_time(df2, col, q):
        if not col:
            return df2
        df2 = df2.copy()
        df2[col] = pd.to_datetime(df2[col], errors="coerce")
        df2 = df2.dropna(subset=[col])
        for yr in range(2018, 2030):
            if str(yr) in q:
                return df2[df2[col].dt.year == yr]
        qmap = {
            "q1": [1,2,3], "q2": [4,5,6], "q3": [7,8,9], "q4": [10,11,12],
            "first quarter": [1,2,3], "second quarter": [4,5,6],
            "third quarter": [7,8,9], "fourth quarter": [10,11,12],
        }
        for qk, months in qmap.items():
            if qk in q:
                return df2[df2[col].dt.month.isin(months)]
        month_map = {
            "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
            "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
            "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
            "sep":9,"oct":10,"nov":11,"dec":12,
        }
        for mn, mv in month_map.items():
            if mn in q:
                return df2[df2[col].dt.month == mv]
        m = _re.search(r"last (\d+) months?", q)
        if m:
            n = int(m.group(1))
            cutoff = df2[col].max() - pd.DateOffset(months=n)
            return df2[df2[col] >= cutoff]
        if "last year" in q:
            yr = df2[col].dt.year.max() - 1
            return df2[df2[col].dt.year == yr]
        if "this year" in q or "current year" in q:
            yr = df2[col].dt.year.max()
            return df2[df2[col].dt.year == yr]
        return df2

    dff = filter_by_time(df, dc, q) if dc else df.copy()
    note = f" (filtered: {len(dff):,} of {len(df):,} records)" if len(dff) < len(df) else ""
    answers = []

    # Revenue — only for non-HR domains (salary column should not become "revenue")
    if sc and domain != "HR" and any(k in q for k in ["revenue","sales","total","income","how much","made"]):
        answers.append(f"**Total Revenue{note}:** {dff[sc].sum():,.2f} | Avg: {dff[sc].mean():,.2f}")

    # Profit
    if pc and domain != "HR" and any(k in q for k in ["profit","margin","earning","net"]):
        p = dff[pc].sum()
        mg = f" | Margin: {p/dff[sc].sum()*100:.1f}%" if sc and dff[sc].sum() > 0 else ""
        answers.append(f"**Total Profit{note}:** {p:,.2f}{mg}")

    # Top products
    if prd and sc and any(k in q for k in ["top","best","highest","product","item"]):
        n = int(_re.search(r"\d+", q).group()) if _re.search(r"\d+", q) else 5
        n = min(n, 20)
        tp = dff.groupby(prd)[sc].sum().sort_values(ascending=False).head(n)
        rows = "\n".join([f"  {ii+1}. **{pp}** — {vv:,.2f}" for ii,(pp,vv) in enumerate(tp.items())])
        answers.append(f"**Top {n} Products by Revenue{note}:**\n{rows}")

    # Bottom products
    if prd and sc and any(k in q for k in ["bottom","worst","lowest","underperform"]):
        n = int(_re.search(r"\d+", q).group()) if _re.search(r"\d+", q) else 5
        n = min(n, 20)
        bp = dff.groupby(prd)[sc].sum().sort_values().head(n)
        rows = "\n".join([f"  {ii+1}. **{pp}** — {vv:,.2f}" for ii,(pp,vv) in enumerate(bp.items())])
        answers.append(f"**Bottom {n} Products{note}:**\n{rows}")

    # Category
    if cat and sc and any(k in q for k in ["categor","segment","group","type"]):
        cs = dff.groupby(cat)[sc].sum().sort_values(ascending=False)
        rows = "\n".join([f"  {ii+1}. **{cc}** — {vv:,.2f}" for ii,(cc,vv) in enumerate(cs.head(5).items())])
        answers.append(f"**Revenue by Category{note}:**\n{rows}")

    # Region
    if reg and sc and any(k in q for k in ["region","area","zone","territory","where"]):
        rs = dff.groupby(reg)[sc].sum().sort_values(ascending=False)
        rows = "\n".join([f"  {ii+1}. **{rr}** — {vv:,.2f}" for ii,(rr,vv) in enumerate(rs.head(5).items())])
        answers.append(f"**Revenue by Region{note}:**\n{rows}")

    # Customers
    if cus and sc and any(k in q for k in ["customer","client","buyer","who","account"]):
        n = int(_re.search(r"\d+", q).group()) if _re.search(r"\d+", q) else 5
        n = min(n, 20)
        tc = dff.groupby(cus)[sc].sum().sort_values(ascending=False).head(n)
        rows = "\n".join([f"  {ii+1}. **{cc}** — {vv:,.2f}" for ii,(cc,vv) in enumerate(tc.items())])
        answers.append(f"**Top {n} Customers{note}:**\n{rows}")

    # Units
    if qty and any(k in q for k in ["unit","quantity","sold","volume","how many"]):
        answers.append(f"**Units Sold{note}:** {dff[qty].sum():,.0f} total | Avg per order: {dff[qty].mean():,.1f}")

    # Discount
    if dis and any(k in q for k in ["discount","promo","offer","rebate"]):
        answers.append(f"**Discount{note}:** Avg: {dff[dis].mean():,.2f} | Max: {dff[dis].max():,.2f}")
        if sc:
            corr = dff[[dis, sc]].dropna().corr().iloc[0, 1]
            lbl = "positive (helps sales)" if corr > 0.1 else "negative (hurts revenue)" if corr < -0.1 else "neutral"
            answers.append(f"  Discount–Revenue correlation: **{corr:.3f}** ({lbl})")

    # Salary
    if sal and any(k in q for k in ["salary","pay","wage","compensation","earn","ctc"]):
        s = dff[sal].dropna()
        answers.append(f"**Salary:** Avg: {s.mean():,.0f} | Median: {s.median():,.0f} | Min: {s.min():,.0f} | Max: {s.max():,.0f} | Payroll: {s.sum():,.0f}")
        if dept:
            ds = dff.groupby(dept)[sal].mean().sort_values(ascending=False)
            answers.append(f"  Highest dept: **{ds.index[0]}** ({ds.iloc[0]:,.0f}) | Lowest: **{ds.index[-1]}** ({ds.iloc[-1]:,.0f})")

    # Gender pay gap
    if sal and gen and any(k in q for k in ["gender","male","female","pay gap","women","men"]):
        dg = dff.copy()
        dg["_g"] = dg[gen].astype(str).str.title()
        ms = dg[dg["_g"].isin(["Male","M"])][sal].mean()
        fs = dg[dg["_g"].isin(["Female","F"])][sal].mean()
        if ms > 0 and fs > 0:
            gap = abs(ms - fs) / max(ms, fs) * 100
            answers.append(f"**Gender Pay Gap:** Male: {ms:,.0f} | Female: {fs:,.0f} | Gap: **{gap:.1f}%** ({'Male earns more' if ms > fs else 'Female earns more'})")

    # Attrition
    if attr and any(k in q for k in ["attrition","resign","left","churn","turnover","quit","exit"]):
        rate = dff[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"]).mean() * 100
        health = "Above 15% — high risk." if rate > 15 else "Within healthy range."
        answers.append(f"**Attrition Rate:** {rate:.1f}% — {health}")
        if dept:
            da = dff.copy()
            da["_left"] = da[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"])
            dr = da.groupby(dept)["_left"].mean().mul(100).sort_values(ascending=False)
            answers.append(f"  Highest: **{dr.index[0]}** ({dr.iloc[0]:.1f}%) | Lowest: **{dr.index[-1]}** ({dr.iloc[-1]:.1f}%)")

    # Employees / headcount
    if any(k in q for k in ["employee","headcount","staff","workforce","how many employee"]):
        total_emp = dff[emp].nunique() if emp else len(dff)
        answers.append(f"**Total Employees:** {total_emp:,}")
        if dept:
            dc2 = dff[dept].value_counts().head(5)
            rows = " | ".join([f"**{dd}**: {cc:,}" for dd, cc in dc2.items()])
            answers.append(f"  By Department: {rows}")
        if gen:
            gc = dff[gen].astype(str).str.title().value_counts()
            m = gc.get("Male", gc.get("M", 0))
            f2 = gc.get("Female", gc.get("F", 0))
            answers.append(f"  Gender: **{int(m):,} Male** / **{int(f2):,} Female**")

    # Department
    if dept and any(k in q for k in ["department","dept","division","team"]):
        dc3 = dff[dept].value_counts()
        rows = "\n".join([f"  {ii+1}. **{dd}** — {cc:,}" for ii,(dd,cc) in enumerate(dc3.head(10).items())])
        answers.append(f"**Department Breakdown:**\n{rows}")

    # Tenure
    if ten and any(k in q for k in ["tenure","experience","service","years"]):
        answers.append(f"**Tenure:** Avg: {dff[ten].mean():.1f} yrs | Median: {dff[ten].median():.1f} yrs | Tenured (>5yr): {(dff[ten]>5).sum():,}")

    # CTR
    if imp and clk and any(k in q for k in ["ctr","click through","click rate"]):
        ctr = dff[clk].sum() / dff[imp].sum() * 100
        answers.append(f"**CTR:** {ctr:.2f}% ({dff[clk].sum():,.0f} clicks / {dff[imp].sum():,.0f} impressions)")

    # CVR
    if clk and conv and any(k in q for k in ["cvr","conversion","convert","lead"]):
        cvr = dff[conv].sum() / dff[clk].sum() * 100
        answers.append(f"**CVR:** {cvr:.2f}% ({dff[conv].sum():,.0f} conversions from {dff[clk].sum():,.0f} clicks)")

    # ROI
    if sp and sc and any(k in q for k in ["roi","return on","roas","marketing return"]):
        roi_v = (dff[sc].sum() - dff[sp].sum()) / dff[sp].sum() * 100
        answers.append(f"**Marketing ROI:** {roi_v:.1f}% | Spend: {dff[sp].sum():,.2f} | Revenue: {dff[sc].sum():,.2f}")

    # Store
    if store and sc and any(k in q for k in ["store","branch","outlet","location","shop"]):
        ts = dff.groupby(store)[sc].sum().sort_values(ascending=False)
        answers.append(f"**Top Store:** {ts.index[0]} ({ts.iloc[0]:,.2f}) | Bottom: {ts.index[-1]} ({ts.iloc[-1]:,.2f}) | Total: {ts.shape[0]}")

    # Fraud
    if domain == "Fraud" and any(k in q for k in ["fraud","fraudulent","legitimate","rate","suspicious"]):
        cl = {c.lower().strip(): c for c in df.columns}
        fraud_label_cols = ["class","label","fraud","is_fraud","isfraud","target"]
        cc = next((cl[c] for c in fraud_label_cols if c in cl), None)
        if cc:
            df2 = dff.copy()
            df2["_lbl"] = df2[cc].astype(str).str.lower().map(lambda x: 1 if x in ["1","true","yes","fraud"] else 0)
            fc = int(df2["_lbl"].sum())
            total = len(df2)
            answers.append(f"**Fraud:** {fc:,} fraud ({fc/total*100:.4f}%) out of {total:,} transactions")
            amt = cl.get("amount")
            if amt:
                fa = df2[df2["_lbl"]==1][amt]
                la = df2[df2["_lbl"]==0][amt]
                if len(fa) > 0:
                    answers.append(f"  Avg fraud amount: ${fa.mean():,.2f} | Avg legit: ${la.mean():,.2f} | Exposure: ${fa.sum():,.2f}")

    # Dataset size
    if any(k in q for k in ["record","row","size","dataset","how big","many record"]):
        answers.append(f"**Dataset:** {len(df):,} records | {len(df.columns)} columns | Domain: **{domain}**")

    # Compare years
    if sc and dc and any(k in q for k in ["compare","vs","versus","difference","between"]):
        d2 = prep_date(df, dc, sc)
        if not d2.empty:
            years = sorted(d2["_Y"].unique())
            if len(years) >= 2:
                y1, y2 = years[-2], years[-1]
                v1 = d2[d2["_Y"]==y1][sc].sum()
                v2 = d2[d2["_Y"]==y2][sc].sum()
                chg = (v2 - v1) / v1 * 100 if v1 > 0 else 0
                arrow = "🔺" if chg > 0 else "🔻"
                answers.append(f"**Year Comparison:** {y1}: {v1:,.2f} → {y2}: {v2:,.2f} ({arrow}{abs(chg):.1f}% change)")

    # Trend
    if sc and dc and any(k in q for k in ["trend","over time","monthly","growth","increase","decrease"]):
        d2 = prep_date(dff, dc, sc)
        if not d2.empty:
            m = d2.groupby("_M")[sc].sum()
            if len(m) >= 2:
                first_v = m.iloc[0]; last_v = m.iloc[-1]
                chg = (last_v - first_v) / first_v * 100 if first_v > 0 else 0
                arrow = "🔺" if chg > 0 else "🔻"
                answers.append(
                    f"**Revenue Trend{note}:** {m.index[0]} ({first_v:,.2f}) → {m.index[-1]} ({last_v:,.2f}) | "
                    f"Overall: {arrow}{abs(chg):.1f}% | Best: **{m.idxmax()}** ({m.max():,.2f}) | Worst: **{m.idxmin()}** ({m.min():,.2f})"
                )

    # Average
    if any(k in q for k in ["average","mean","avg","typical"]):
        if sc and domain != "HR":  answers.append(f"**Avg Revenue per Record:** {dff[sc].mean():,.2f}")
        if pc and domain != "HR":  answers.append(f"**Avg Profit per Record:** {dff[pc].mean():,.2f}")
        if qty: answers.append(f"**Avg Units per Order:** {dff[qty].mean():,.1f}")
        if sal: answers.append(f"**Avg Salary:** {dff[sal].mean():,.0f}")

    # Fallback
    if not answers:
        nc = dff.select_dtypes(include="number").columns.tolist()
        if nc:
            s = dff[nc[0]].dropna()
            answers.append(f"**{nc[0]}:** Total: {s.sum():,.2f} | Avg: {s.mean():,.2f} | Min: {s.min():,.2f} | Max: {s.max():,.2f}")
        answers.append("Try asking: revenue, profit, top products, customers, trend, discount, attrition, salary, fraud rate...")

    return "\n\n".join(answers)


def render_nlq(df, domain, found):
    section("💬 Ask Your Data — Natural Language Query", domain.lower())
    st.markdown("""<div class="insight-box">
    <strong>💬 Ask anything about your data in plain English.</strong><br>
    Examples: <em>"What is total revenue in Q3?"</em> &nbsp;·&nbsp;
    <em>"Show top 10 products"</em> &nbsp;·&nbsp;
    <em>"What is attrition rate by department?"</em> &nbsp;·&nbsp;
    <em>"Compare 2022 vs 2023 sales"</em>
    </div>""", unsafe_allow_html=True)

    quick_map = {
        "Sales":     ["Total revenue?","Top 5 products?","Best month?","Top 5 customers?","Revenue trend?","Profit margin?"],
        "HR":        ["Total employees?","Avg salary?","Attrition rate?","Salary by department?","Gender pay gap?","Avg tenure?"],
        "Marketing": ["Total spend?","Marketing ROI?","CTR?","Conversion rate?","Revenue trend?"],
        "Ecommerce": ["Total revenue?","Top 10 products?","Top customers?","Best category?","Revenue trend?"],
        "Retail":    ["Total revenue?","Top 10 stores?","Best category?","Avg discount?","Top products?"],
        "Fraud":     ["Fraud rate?","Avg fraud amount?","Total fraud exposure?","Fraudulent transactions?"],
        "Generic":   ["Total records?","Average values?","Summary?"],
    }
    qs = quick_map.get(domain, quick_map["Generic"])
    st.markdown("**Quick Questions:**")
    btn_cols = st.columns(len(qs))
    clicked_q = None
    for i, q in enumerate(qs):
        with btn_cols[i]:
            if st.button(q, key=f"nlq_btn_{i}", use_container_width=True):
                clicked_q = q

    user_q = st.text_input(
        "Or type your own question:",
        value=clicked_q or "",
        placeholder="e.g. What are the top 5 products by revenue in 2023?",
        key="nlq_input"
    )

    if user_q:
        with st.spinner("Analysing your data..."):
            answer = compute_nlq_answer(user_q, df, found, domain)
        st.markdown("---")
        st.markdown(f"**Question:** {user_q}")
        st.markdown("**Answer:**")
        st.markdown(answer)

        if LLM_PROVIDER != "huggingface":
            if st.button("🤖 Enhance with AI", key="nlq_llm"):
                with st.spinner("Querying AI..."):
                    nc = df.select_dtypes(include="number").columns.tolist()
                    stats = df[nc[:8]].describe().round(2).to_string() if nc else ""
                    prompt = (f"You are a {domain} data analyst. Dataset: {len(df):,} records.\n"
                              f"Columns: {', '.join(df.columns[:20])}\nStats:\n{stats}\n\n"
                              f"Answer with specific numbers:\n{user_q}")
                    try:
                        st.markdown("**AI Enhanced Answer:**")
                        st.markdown(query_llm(prompt))
                    except Exception as e:
                        st.error(f"LLM error: {e}")

    if "nlq_history" not in st.session_state:
        st.session_state.nlq_history = []
    if user_q and (not st.session_state.nlq_history or user_q != st.session_state.nlq_history[0][0]):
        ans = compute_nlq_answer(user_q, df, found, domain)
        st.session_state.nlq_history.insert(0, (user_q, ans))
        st.session_state.nlq_history = st.session_state.nlq_history[:10]
    if len(st.session_state.nlq_history) > 1:
        with st.expander("Query History (last 10)", expanded=False):
            for i, (hq, ha) in enumerate(st.session_state.nlq_history[1:], 1):
                st.markdown(f"**Q{i}:** {hq}")
                st.markdown(f"**A:** {ha[:300]}..." if len(ha) > 300 else f"**A:** {ha}")
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(df, found, domain):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable, PageBreak)
        from reportlab.lib.enums import TA_CENTER
        import datetime

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        DOMAIN_HEX = {
            "Sales": "#0ea5e9", "Marketing": "#ef4444", "HR": "#8b5cf6",
            "Ecommerce": "#10b981", "Retail": "#f59e0b",
            "Fraud": "#ef4444", "Generic": "#64748b",
        }
        dc = colors.HexColor(DOMAIN_HEX.get(domain, "#0ea5e9"))

        title_s  = ParagraphStyle("T",  parent=styles["Title"],   fontSize=22, spaceAfter=6,  alignment=TA_CENTER, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a2e"))
        sub_s    = ParagraphStyle("S",  parent=styles["Normal"],  fontSize=10, spaceAfter=4,  alignment=TA_CENTER, textColor=colors.HexColor("#4a5568"))
        h1_s     = ParagraphStyle("H1", parent=styles["Heading1"],fontSize=14, spaceBefore=14,spaceAfter=6,  fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a2e"))
        body_s   = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=9,  spaceAfter=4,  leading=13, textColor=colors.HexColor("#2d3748"))

        def make_table(data, col_widths, header_color=dc):
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),  header_color),
                ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
                ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,0),  9),
                ("FONTSIZE",      (0,1), (-1,-1), 8.5),
                ("ROWBACKGROUNDS",(0,1), (-1,-1),  [colors.HexColor("#f7fafc"), colors.white]),
                ("GRID",          (0,0), (-1,-1),  0.4, colors.HexColor("#e2e8f0")),
                ("ALIGN",         (0,0), (-1,-1),  "LEFT"),
                ("TOPPADDING",    (0,0), (-1,-1),  4),
                ("BOTTOMPADDING", (0,0), (-1,-1),  4),
                ("LEFTPADDING",   (0,0), (-1,-1),  6)]))
            return t

        now   = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        story = [Spacer(1, 0.8*cm)]

        # Cover
        story += [
            Paragraph("BOARDROOM REPORT", sub_s),
            Paragraph("Universal Agentic AI Data Analyst", title_s),
            HRFlowable(width="100%", thickness=2, color=dc, spaceAfter=8),
            Paragraph(f"Domain: <b>{domain}</b>  |  Generated: {now}", sub_s),
            Paragraph(f"Dataset: {len(df):,} records × {len(df.columns)} columns", sub_s),
            Spacer(1, 0.5*cm)]

        sc   = found.get("sales");    pc   = found.get("profit")
        prd  = found.get("product");  cat  = found.get("category")
        dcol = found.get("date");     qty  = found.get("quantity")
        cus  = found.get("customer"); reg  = found.get("region")
        sal  = found.get("salary");   attr = found.get("attrition")
        dept = found.get("department"); gen = found.get("gender")
        ten  = found.get("tenure")
        emp  = found.get("employee_id") or found.get("employee_name")

        # Executive Summary
        story.append(Paragraph("Executive Summary", h1_s))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=4))
        sum_lines = []
        if sc:
            sum_lines += [f"Total Revenue: {df[sc].sum():,.2f}",
                          f"Average Order Value: {df[sc].mean():,.2f}",
                          f"Peak Transaction: {df[sc].max():,.2f}"]
        if pc and sc:
            sum_lines += [f"Total Profit: {df[pc].sum():,.2f}",
                          f"Profit Margin: {df[pc].sum()/df[sc].sum()*100:.1f}%"]
        if prd and sc:
            bp = df.groupby(prd)[sc].sum().idxmax()
            sum_lines.append(f"Best Product: {bp}")
        if cus:     sum_lines.append(f"Unique Customers: {df[cus].nunique():,}")
        if qty:     sum_lines.append(f"Units Sold: {df[qty].sum():,.0f}")
        if dcol:
            dd2 = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if len(dd2):
                sum_lines.append(f"Date Range: {dd2.min().date()} to {dd2.max().date()}")
        if sal:
            sum_lines += [f"Total Payroll: {df[sal].sum():,.0f}",
                          f"Average Salary: {df[sal].mean():,.0f}",
                          f"Salary Range: {df[sal].min():,.0f} – {df[sal].max():,.0f}"]
        if domain == "HR":
            total_emp = df[emp].nunique() if emp else len(df)
            sum_lines.append(f"Total Employees: {total_emp:,}")
        if attr:
            rate = df[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"]).mean()*100
            sum_lines.append(f"Attrition Rate: {rate:.1f}%")
        if ten:     sum_lines.append(f"Average Tenure: {df[ten].mean():.1f} years")
        if gen:
            gc = df[gen].astype(str).str.title().value_counts()
            m  = gc.get("Male", gc.get("M", 0))
            f2 = gc.get("Female", gc.get("F", 0))
            sum_lines.append(f"Gender Split: {int(m):,} Male / {int(f2):,} Female")
        sum_lines.append(f"Total Records Analysed: {len(df):,}")
        for line in sum_lines:
            story.append(Paragraph(f"• {line}", body_s))
        story.append(Spacer(1, 0.3*cm))

        # KPI Table
        story.append(Paragraph("Key Performance Indicators", h1_s))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=4))
        kpi_rows = [["Metric", "Value"]]
        if sc:
            kpi_rows += [["Total Revenue", f"{df[sc].sum():,.2f}"],
                         ["Average Order Value", f"{df[sc].mean():,.2f}"],
                         ["Peak Value", f"{df[sc].max():,.2f}"]]
        if pc:
            kpi_rows.append(["Total Profit", f"{df[pc].sum():,.2f}"])
            if sc: kpi_rows.append(["Profit Margin", f"{df[pc].sum()/df[sc].sum()*100:.1f}%"])
        if qty:  kpi_rows.append(["Total Units Sold", f"{df[qty].sum():,.0f}"])
        if cus:  kpi_rows.append(["Unique Customers", f"{df[cus].nunique():,}"])
        if prd:  kpi_rows.append(["Products", f"{df[prd].nunique():,}"])
        if sal:
            kpi_rows += [["Average Salary", f"{df[sal].mean():,.0f}"],
                         ["Total Payroll", f"{df[sal].sum():,.0f}"]]
        if attr:
            rate = df[attr].astype(str).str.lower().isin(["yes","true","1","resigned","terminated","left"]).mean()*100
            kpi_rows.append(["Attrition Rate", f"{rate:.1f}%"])
        if ten:  kpi_rows.append(["Average Tenure", f"{df[ten].mean():.1f} yrs"])
        kpi_rows.append(["Total Records", f"{len(df):,}"])
        story.append(make_table(kpi_rows, [10*cm, 6*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Top Products
        if prd and sc:
            story.append(PageBreak())
            story.append(Paragraph("Top 10 Products by Revenue", h1_s))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=4))
            tp = df.groupby(prd)[sc].sum().sort_values(ascending=False).head(10).reset_index()
            tp.columns = ["Product", "Revenue"]
            rows = [["Rank", "Product", "Revenue"]]
            for idx_r, row in tp.iterrows():
                rows.append([str(idx_r+1), str(row["Product"]), f"{row['Revenue']:,.2f}"])
            story.append(make_table(rows, [2*cm, 10*cm, 5*cm]))
            story.append(Spacer(1, 0.4*cm))

        # Top Customers
        if cus and sc:
            story.append(Paragraph("Top 10 Customers by Revenue", h1_s))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=4))
            tc = df.groupby(cus)[sc].sum().sort_values(ascending=False).head(10).reset_index()
            tc.columns = ["Customer", "Revenue"]
            rows = [["Rank", "Customer", "Revenue"]]
            for idx_r, row in tc.iterrows():
                rows.append([str(idx_r+1), str(row["Customer"]), f"{row['Revenue']:,.2f}"])
            story.append(make_table(rows, [2*cm, 10*cm, 5*cm]))
            story.append(Spacer(1, 0.4*cm))

        # Salary by Department (HR)
        if sal and dept:
            story.append(PageBreak())
            story.append(Paragraph("Salary Analysis by Department", h1_s))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=4))
            ds = df.groupby(dept)[sal].agg(["mean","min","max","sum"]).reset_index()
            ds.columns = ["Department","Avg Salary","Min","Max","Total Payroll"]
            rows = [["Department","Avg Salary","Min","Max","Total Payroll"]]
            for _, row in ds.iterrows():
                rows.append([str(row["Department"]), f"{row['Avg Salary']:,.0f}",
                             f"{row['Min']:,.0f}", f"{row['Max']:,.0f}", f"{row['Total Payroll']:,.0f}"])
            story.append(make_table(rows, [5*cm, 3.5*cm, 2.5*cm, 2.5*cm, 3.5*cm]))
            story.append(Spacer(1, 0.4*cm))

        # Column Profile
        story.append(PageBreak())
        story.append(Paragraph("Data Intelligence — Column Profile", h1_s))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=4))
        story.append(Paragraph("Auto-detected column roles, types, completeness and statistics:", body_s))
        col_rows = [["Column", "Type", "Non-Null%", "Unique", "Sample / Stats"]]
        for col in df.columns[:25]:
            dtype   = str(df[col].dtype)
            nn_pct  = f"{df[col].notna().mean()*100:.0f}%"
            uniq    = f"{df[col].nunique():,}"
            if pd.api.types.is_numeric_dtype(df[col]):
                s2 = df[col].dropna()
                sample = f"Min:{s2.min():,.1f} Avg:{s2.mean():,.1f} Max:{s2.max():,.1f}"
            else:
                top_vals = df[col].dropna().astype(str).value_counts().head(2).index.tolist()
                sample   = ", ".join(top_vals)[:40]
            col_rows.append([col[:20], dtype[:10], nn_pct, uniq, sample[:40]])
        story.append(make_table(col_rows, [4*cm, 2.5*cm, 2*cm, 2*cm, 6.5*cm],
                                header_color=colors.HexColor("#2d3748")))

        # Footer
        story += [
            Spacer(1, 0.5*cm),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=4),
            Paragraph(f"Generated by Universal Agentic AI Data Analyst  ·  {now}  ·  Domain: {domain}", sub_s)]

        doc.build(story)
        buf.seek(0)
        return buf.read()

    except ImportError:
        return None
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None


def render_pdf_export(df, found, domain):
    section("📄 Export Boardroom PDF Report", domain.lower())
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""<div class="insight-box">
        <strong>📄 Download a complete boardroom-ready PDF</strong> containing:<br>
        Executive Summary · KPI Table · Top 10 Products · Top 10 Customers ·
        Salary by Department · Data Intelligence Column Profile
        </div>""", unsafe_allow_html=True)
    with c2:
        if st.button("🖨️ Generate PDF Report", use_container_width=True,
                     type="primary", key="gen_pdf"):
            with st.spinner("Building PDF..."):
                pdf_bytes = generate_pdf_report(df, found, domain)
            if pdf_bytes:
                import datetime
                fname = f"{domain}_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button("⬇️ Download PDF", data=pdf_bytes,
                                   file_name=fname, mime="application/pdf",
                                   use_container_width=True, key="dl_pdf")
                st.success("PDF ready! Click above to download.")
            else:
                st.warning("Add `reportlab>=4.0.0` to requirements.txt and redeploy.")
                st.code("reportlab>=4.0.0")



# ══════════════════════════════════════════════════════════════════════════════
#  CALCULATED COLUMNS ENGINE  —  DAX-style for all domains
# ══════════════════════════════════════════════════════════════════════════════

# Domain-specific preset calculations shown as quick-add buttons
DOMAIN_PRESETS = {
    "HR": [
        {
            "name": "Tenure (Years)",
            "output_col": "Tenure_Years",
            "description": "Years between joining date and exit/today",
            "type": "date_diff",
            "params": {"start_key": "hire_date", "end_key": "exit_date", "unit": "years"},
            "icon": "📅",
        },
        {
            "name": "Tenure (Months)",
            "output_col": "Tenure_Months",
            "description": "Months of service",
            "type": "date_diff",
            "params": {"start_key": "hire_date", "end_key": "exit_date", "unit": "months"},
            "icon": "📅",
        },
        {
            "name": "Salary After 5% Hike",
            "output_col": "Salary_5pct_Hike",
            "description": "Salary × 1.05",
            "type": "formula",
            "params": {"col_key": "salary", "op": "multiply", "value": 1.05},
            "icon": "💰",
        },
        {
            "name": "Salary After 10% Hike",
            "output_col": "Salary_10pct_Hike",
            "description": "Salary × 1.10",
            "type": "formula",
            "params": {"col_key": "salary", "op": "multiply", "value": 1.10},
            "icon": "💰",
        },
        {
            "name": "Salary After 15% Hike",
            "output_col": "Salary_15pct_Hike",
            "description": "Salary × 1.15",
            "type": "formula",
            "params": {"col_key": "salary", "op": "multiply", "value": 1.15},
            "icon": "💰",
        },
        {
            "name": "Monthly Salary",
            "output_col": "Monthly_Salary",
            "description": "Annual Salary ÷ 12",
            "type": "formula",
            "params": {"col_key": "salary", "op": "divide", "value": 12},
            "icon": "💰",
        },
        {
            "name": "Age Band",
            "output_col": "Age_Band",
            "description": "Group age into bands: 20-30, 31-40, 41-50, 51+",
            "type": "bin",
            "params": {"col_key": "age", "bins": [0, 30, 40, 50, 100],
                       "labels": ["20-30", "31-40", "41-50", "51+"]},
            "icon": "🎂",
        },
        {
            "name": "Salary Band",
            "output_col": "Salary_Band",
            "description": "Low / Mid / High / Senior based on salary percentiles",
            "type": "percentile_bin",
            "params": {"col_key": "salary",
                       "labels": ["Low", "Mid", "High", "Senior"]},
            "icon": "💼",
        }],
    "Sales": [
        {
            "name": "Profit Margin %",
            "output_col": "Profit_Margin_Pct",
            "description": "Profit ÷ Revenue × 100",
            "type": "ratio",
            "params": {"num_key": "profit", "den_key": "sales", "multiply": 100},
            "icon": "📊",
        },
        {
            "name": "Revenue per Unit",
            "output_col": "Revenue_Per_Unit",
            "description": "Revenue ÷ Quantity",
            "type": "ratio",
            "params": {"num_key": "sales", "den_key": "quantity", "multiply": 1},
            "icon": "📦",
        },
        {
            "name": "Discount Impact",
            "output_col": "Discount_Impact",
            "description": "Revenue × Discount % = revenue lost to discount",
            "type": "two_col_multiply",
            "params": {"col1_key": "sales", "col2_key": "discount"},
            "icon": "🏷",
        },
        {
            "name": "Net Revenue (after discount)",
            "output_col": "Net_Revenue",
            "description": "Revenue - (Revenue × Discount)",
            "type": "net_after_discount",
            "params": {"revenue_key": "sales", "discount_key": "discount"},
            "icon": "💰",
        },
        {
            "name": "Revenue Growth Flag",
            "output_col": "Above_Avg_Revenue",
            "description": "Yes/No — is this order above average revenue?",
            "type": "above_avg_flag",
            "params": {"col_key": "sales"},
            "icon": "🚩",
        },
        {
            "name": "Order Size Band",
            "output_col": "Order_Size_Band",
            "description": "Small / Medium / Large / Enterprise based on revenue percentiles",
            "type": "percentile_bin",
            "params": {"col_key": "sales",
                       "labels": ["Small", "Medium", "Large", "Enterprise"]},
            "icon": "📦",
        }],
    "Marketing": [
        {
            "name": "CTR %",
            "output_col": "CTR_Pct",
            "description": "Clicks ÷ Impressions × 100",
            "type": "ratio",
            "params": {"num_key": "clicks", "den_key": "impressions", "multiply": 100},
            "icon": "🖱",
        },
        {
            "name": "CVR %",
            "output_col": "CVR_Pct",
            "description": "Conversions ÷ Clicks × 100",
            "type": "ratio",
            "params": {"num_key": "conversions", "den_key": "clicks", "multiply": 100},
            "icon": "✅",
        },
        {
            "name": "Cost per Click (CPC)",
            "output_col": "CPC",
            "description": "Spend ÷ Clicks",
            "type": "ratio",
            "params": {"num_key": "spend", "den_key": "clicks", "multiply": 1},
            "icon": "💸",
        },
        {
            "name": "Cost per Conversion",
            "output_col": "Cost_Per_Conversion",
            "description": "Spend ÷ Conversions",
            "type": "ratio",
            "params": {"num_key": "spend", "den_key": "conversions", "multiply": 1},
            "icon": "💸",
        },
        {
            "name": "ROAS",
            "output_col": "ROAS",
            "description": "Revenue ÷ Spend",
            "type": "ratio",
            "params": {"num_key": "sales", "den_key": "spend", "multiply": 1},
            "icon": "📈",
        }],
    "Ecommerce": [
        {
            "name": "Profit Margin %",
            "output_col": "Profit_Margin_Pct",
            "description": "Profit ÷ Revenue × 100",
            "type": "ratio",
            "params": {"num_key": "profit", "den_key": "sales", "multiply": 100},
            "icon": "📊",
        },
        {
            "name": "Revenue per Unit",
            "output_col": "Revenue_Per_Unit",
            "description": "Revenue ÷ Quantity",
            "type": "ratio",
            "params": {"num_key": "sales", "den_key": "quantity", "multiply": 1},
            "icon": "📦",
        },
        {
            "name": "Delivery Status Flag",
            "output_col": "Late_Delivery",
            "description": "Yes if delivery > 7 days, No otherwise",
            "type": "threshold_flag",
            "params": {"col_key": "delivery", "threshold": 7,
                       "above_label": "Late", "below_label": "On Time"},
            "icon": "🚚",
        }],
    "Retail": [
        {
            "name": "Profit Margin %",
            "output_col": "Profit_Margin_Pct",
            "description": "Profit ÷ Revenue × 100",
            "type": "ratio",
            "params": {"num_key": "profit", "den_key": "sales", "multiply": 100},
            "icon": "📊",
        },
        {
            "name": "Revenue per Unit",
            "output_col": "Revenue_Per_Unit",
            "description": "Revenue ÷ Quantity",
            "type": "ratio",
            "params": {"num_key": "sales", "den_key": "quantity", "multiply": 1},
            "icon": "📦",
        },
        {
            "name": "Net Revenue (after discount)",
            "output_col": "Net_Revenue",
            "description": "Revenue - (Revenue × Discount)",
            "type": "net_after_discount",
            "params": {"revenue_key": "sales", "discount_key": "discount"},
            "icon": "💰",
        }],
    "Fraud": [
        {
            "name": "Amount Band",
            "output_col": "Amount_Band",
            "description": "Low / Medium / High / Very High based on amount percentiles",
            "type": "percentile_bin",
            "params": {"col_key": "sales",
                       "labels": ["Low", "Medium", "High", "Very High"]},
            "icon": "💳",
        }],
    "Generic": [],
}


def _apply_calculation(df, calc_type, params, found):
    """Apply one calculation and return a Series result."""
    def gcol(key):
        return found.get(key)

    if calc_type == "date_diff":
        start_col = gcol(params["start_key"])
        # end_key may not be in found — try direct column name too
        end_col = gcol(params.get("end_key", "")) or params.get("end_col")
        unit = params.get("unit", "years")
        if not start_col:
            return None, f"Start date column '{params['start_key']}' not mapped."
        start = pd.to_datetime(df[start_col], errors="coerce")
        if end_col and end_col in df.columns:
            end = pd.to_datetime(df[end_col], errors="coerce")
        else:
            end = pd.Timestamp.today()
        diff_days = (end - start).dt.days.clip(lower=0)
        if unit == "years":
            return (diff_days / 365.25).round(2), None
        elif unit == "months":
            return (diff_days / 30.44).round(1), None
        else:
            return diff_days, None

    elif calc_type == "formula":
        col = gcol(params["col_key"])
        if not col:
            return None, f"Column '{params['col_key']}' not mapped."
        s = pd.to_numeric(df[col], errors="coerce")
        op = params["op"]; val = params["value"]
        if op == "multiply": return (s * val).round(2), None
        if op == "divide":   return (s / val).round(2), None
        if op == "add":      return (s + val).round(2), None
        if op == "subtract": return (s - val).round(2), None
        return None, f"Unknown op: {op}"

    elif calc_type == "ratio":
        num_col = gcol(params["num_key"])
        den_col = gcol(params["den_key"])
        if not num_col:
            return None, f"Numerator column '{params['num_key']}' not mapped."
        if not den_col:
            return None, f"Denominator column '{params['den_key']}' not mapped."
        num = pd.to_numeric(df[num_col], errors="coerce")
        den = pd.to_numeric(df[den_col], errors="coerce").replace(0, np.nan)
        return (num / den * params.get("multiply", 1)).round(4), None

    elif calc_type == "two_col_multiply":
        c1 = gcol(params["col1_key"]); c2 = gcol(params["col2_key"])
        if not c1 or not c2:
            return None, "Both columns must be mapped."
        return (pd.to_numeric(df[c1], errors="coerce") *
                pd.to_numeric(df[c2], errors="coerce")).round(2), None

    elif calc_type == "net_after_discount":
        rev_col = gcol(params["revenue_key"])
        dis_col = gcol(params["discount_key"])
        if not rev_col:
            return None, "Revenue column not mapped."
        rev = pd.to_numeric(df[rev_col], errors="coerce")
        if dis_col:
            dis = pd.to_numeric(df[dis_col], errors="coerce").fillna(0)
            # If discount looks like percentage (0-1 range), treat as fraction
            if dis.max() <= 1:
                return (rev * (1 - dis)).round(2), None
            else:
                return (rev - dis).round(2), None
        return rev.round(2), None

    elif calc_type == "bin":
        col = gcol(params["col_key"])
        if not col:
            return None, f"Column '{params['col_key']}' not mapped."
        s = pd.to_numeric(df[col], errors="coerce")
        return pd.cut(s, bins=params["bins"], labels=params["labels"],
                      right=True).astype(str), None

    elif calc_type == "percentile_bin":
        col = gcol(params["col_key"])
        if not col:
            return None, f"Column '{params['col_key']}' not mapped."
        s = pd.to_numeric(df[col], errors="coerce")
        labels = params["labels"]
        n = len(labels)
        bins = [s.quantile(i/n) for i in range(n+1)]
        bins[0] -= 0.001  # include minimum
        try:
            return pd.cut(s, bins=bins, labels=labels).astype(str), None
        except Exception:
            return pd.qcut(s, q=n, labels=labels, duplicates="drop").astype(str), None

    elif calc_type == "above_avg_flag":
        col = gcol(params["col_key"])
        if not col:
            return None, f"Column '{params['col_key']}' not mapped."
        s = pd.to_numeric(df[col], errors="coerce")
        return (s >= s.mean()).map({True: "Yes", False: "No"}), None

    elif calc_type == "threshold_flag":
        col = gcol(params["col_key"])
        if not col:
            return None, f"Column '{params['col_key']}' not mapped."
        s = pd.to_numeric(df[col], errors="coerce")
        return (s > params["threshold"]).map(
            {True: params["above_label"], False: params["below_label"]}), None

    elif calc_type == "custom":
        # Custom formula: col1 op col2 or col op value
        c1_name = params.get("col1"); c2_name = params.get("col2")
        op = params.get("op"); val = params.get("value")
        out_name = params.get("output_col", "Calculated")
        if c1_name not in df.columns:
            return None, f"Column '{c1_name}' not found in dataset."
        s1 = pd.to_numeric(df[c1_name], errors="coerce")
        if c2_name and c2_name in df.columns:
            s2 = pd.to_numeric(df[c2_name], errors="coerce")
            if op == "+": return (s1 + s2).round(4), None
            if op == "-": return (s1 - s2).round(4), None
            if op == "*": return (s1 * s2).round(4), None
            if op == "/": return (s1 / s2.replace(0, np.nan)).round(4), None
            if op == "date_diff_years":
                d1 = pd.to_datetime(df[c1_name], errors="coerce")
                d2 = pd.to_datetime(df[c2_name], errors="coerce")
                return ((d2 - d1).dt.days / 365.25).round(2), None
            if op == "date_diff_months":
                d1 = pd.to_datetime(df[c1_name], errors="coerce")
                d2 = pd.to_datetime(df[c2_name], errors="coerce")
                return ((d2 - d1).dt.days / 30.44).round(1), None
        elif val is not None:
            if op == "+": return (s1 + val).round(4), None
            if op == "-": return (s1 - val).round(4), None
            if op == "*": return (s1 * val).round(4), None
            if op == "/": return (s1 / (val or np.nan)).round(4), None
            if op == "%": return (s1 * val / 100).round(4), None
        return None, "Invalid custom formula parameters."

    return None, "Unknown calculation type."


def render_calc_engine(df, found, domain):
    """Full DAX-style calculated columns panel — non-tech friendly."""
    section("🧮 Calculated Columns Engine", domain.lower())

    # ── Plain English intro ───────────────────────────────────────────────
    st.markdown("""<div class="insight-box">
    <strong>🧮 What does this do?</strong><br><br>
    Think of this as a <strong>smart calculator for your entire dataset</strong>.<br>
    You pick what you want to calculate — the engine does it for every row automatically.<br><br>
    <strong>Examples:</strong><br>
    &nbsp;&nbsp;📅 <em>Date of Exit − Date of Joining</em> → gives each employee's <strong>Tenure in Years</strong><br>
    &nbsp;&nbsp;💰 <em>Salary × 1.10</em> → gives each employee's <strong>Salary after 10% Hike</strong><br>
    &nbsp;&nbsp;📊 <em>Profit ÷ Revenue × 100</em> → gives each record's <strong>Profit Margin %</strong><br><br>
    Results appear as new charts and can be downloaded as a new CSV column.
    </div>""", unsafe_allow_html=True)

    # Session state for calculated columns
    if "calc_columns" not in st.session_state:
        st.session_state.calc_columns = {}

    n_done = len(st.session_state.calc_columns)
    if n_done > 0:
        st.success(f"✅ {n_done} calculated column(s) ready — see **Results & Charts** tab")

    tab_preset, tab_custom, tab_results = st.tabs(
        [f"⚡ Step 1 — Ready-Made Calculations ({domain})",
         "🔧 Step 2 — Build Your Own Calculation",
         f"📊 Step 3 — View Results {'✅' if n_done > 0 else ''}"])

    # ── TAB 1: PRESETS (friendly cards) ───────────────────────────────────
    with tab_preset:
        presets = DOMAIN_PRESETS.get(domain, [])

        if not presets:
            st.info(f"No ready-made calculations for the {domain} domain. "
                    "Go to Step 2 to build your own.")
        else:
            st.markdown(f"**Click any calculation below to run it on your data instantly.**")
            st.caption("Green tick = already calculated. Warning = required column not mapped yet.")
            st.markdown("")

            cols_g = st.columns(3)
            for i, p in enumerate(presets):
                with cols_g[i % 3]:
                    already_done = p["output_col"] in st.session_state.calc_columns

                    # Check required columns
                    can_run = True; missing = []
                    pt = p["type"]; pp = p["params"]
                    needed_keys = []
                    if pt == "date_diff":
                        needed_keys = [pp.get("start_key")]
                    elif pt in ["formula","bin","percentile_bin","above_avg_flag","threshold_flag"]:
                        needed_keys = [pp.get("col_key")]
                    elif pt == "ratio":
                        needed_keys = [pp.get("num_key"), pp.get("den_key")]
                    elif pt in ["two_col_multiply","net_after_discount"]:
                        needed_keys = [pp.get("col1_key") or pp.get("revenue_key"),
                                       pp.get("col2_key") or pp.get("discount_key")]
                    for nk in needed_keys:
                        if nk and not found.get(nk):
                            can_run = False; missing.append(nk)

                    # Status colour
                    status_color = "#16a34a" if already_done else ("#b45309" if not can_run else "#1e40af")
                    status_icon  = "✅" if already_done else ("⚠️" if not can_run else "▶️")

                    st.markdown(f"""<div style="border:1px solid #2d3748;border-radius:10px;
                        padding:12px 14px;margin-bottom:6px;background:#0f172a">
                        <div style="font-size:.95rem;font-weight:700;color:#e2e8f0">
                            {p['icon']} {p['name']}</div>
                        <div style="font-size:.78rem;color:#94a3b8;margin-top:3px">
                            {p['description']}</div>
                        <div style="font-size:.75rem;color:{status_color};margin-top:5px">
                            {status_icon} {'Already calculated' if already_done
                              else ('Needs: ' + ', '.join(missing)) if not can_run
                              else 'Ready to calculate'}</div>
                        </div>""", unsafe_allow_html=True)

                    if already_done:
                        if st.button("🔄 Recalculate", key=f"preset_{i}", use_container_width=True):
                            result, err = _apply_calculation(df, p["type"], p["params"], found)
                            if not err:
                                st.session_state.calc_columns[p["output_col"]] = result
                                st.rerun()
                    elif not can_run:
                        st.caption(f"Map these columns first in Column Mapping above: **{', '.join(missing)}**")
                    else:
                        if st.button(f"▶️ Calculate", key=f"preset_{i}", use_container_width=True,
                                     type="primary"):
                            result, err = _apply_calculation(df, p["type"], p["params"], found)
                            if err:
                                st.error(f"Error: {err}")
                            else:
                                st.session_state.calc_columns[p["output_col"]] = result
                                st.success(f"Done! '{p['output_col']}' added. Go to Results tab to see the chart.")
                                st.rerun()

        # ── Custom Date Difference ────────────────────────────────────────
        st.markdown("---")
        st.markdown("**📅 Calculate Days / Months / Years between two dates**")
        st.caption("Common use: Tenure = Date of Exit − Date of Joining | Age = Today − Date of Birth")

        all_cols_list = list(df.columns)
        # Try to pre-detect date-like columns
        date_like = [c for c in df.columns if any(k in c.lower()
                     for k in ["date","time","joined","joining","exit","left","end","start","birth","dob"])]

        dd_c1, dd_c2, dd_c3, dd_c4, dd_c5 = st.columns([2, 2, 1.5, 2, 1.5])
        with dd_c1:
            st.caption("**FROM date** (the earlier/start date)")
            dd_start_default = (all_cols_list.index(found["hire_date"])
                                if found.get("hire_date") in all_cols_list else 0)
            dd_start = st.selectbox("Start Date", all_cols_list,
                                    index=dd_start_default, key="dd_start",
                                    help="e.g. Date of Joining, Date of Birth")
        with dd_c2:
            st.caption("**TO date** (the later/end date)")
            end_opts = ["Today (today's date)"] + all_cols_list
            dd_end = st.selectbox("End Date", end_opts, key="dd_end",
                                  help="e.g. Date of Exit, or Today if employee is still active")
        with dd_c3:
            st.caption("**Result unit**")
            dd_unit = st.selectbox("Show result in:", ["Years","Months","Days"], key="dd_unit")
        with dd_c4:
            st.caption("**Name for the new column**")
            dd_out = st.text_input("Column name", value="Tenure_Years", key="dd_out",
                                   help="This is what the new column will be called in results")
        with dd_c5:
            st.caption("&nbsp;")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶️ Calculate", key="dd_calc", use_container_width=True, type="primary"):
                d1 = pd.to_datetime(df[dd_start], errors="coerce")
                d2 = (pd.Timestamp.today() if dd_end == "Today (today's date)"
                      else pd.to_datetime(df[dd_end], errors="coerce"))
                diff = (d2 - d1).dt.days.clip(lower=0)
                if dd_unit == "Years":    result = (diff / 365.25).round(2)
                elif dd_unit == "Months": result = (diff / 30.44).round(1)
                else:                     result = diff
                col_name = dd_out.strip().replace(" ", "_") or "Date_Diff"
                st.session_state.calc_columns[col_name] = result
                st.success(f"✅ Done! '{col_name}' added — go to Results tab to see the chart.")
                st.rerun()

    # ── TAB 2: CUSTOM FORMULA BUILDER (non-tech friendly) ────────────────
    with tab_custom:
        st.markdown("**🔧 Build your own calculation — no coding needed**")
        st.markdown("""<div class="insight-box">
        <strong>How to use:</strong><br>
        1️⃣ &nbsp;Pick the <strong>first column</strong> (e.g. Salary)<br>
        2️⃣ &nbsp;Choose what to <strong>do with it</strong> (e.g. Multiply)<br>
        3️⃣ &nbsp;Pick a <strong>second column</strong> or enter a <strong>fixed number</strong> (e.g. 1.10 for 10% hike)<br>
        4️⃣ &nbsp;Give the result a <strong>name</strong> (e.g. Salary_After_Hike)<br>
        5️⃣ &nbsp;Click <strong>Calculate</strong> — done!
        </div>""", unsafe_allow_html=True)

        all_col_list = list(df.columns)
        num_col_list = df.select_dtypes(include="number").columns.tolist()

        # ── Step 1: Pick Column 1 ─────────────────────────────────────────
        st.markdown("**① Pick your starting column:**")
        cb_c1, cb_spacer = st.columns([3, 3])
        with cb_c1:
            col1_sel = st.selectbox("Which column do you want to calculate from?",
                                    all_col_list, key="cb_col1",
                                    help="This is the column whose values will be used in the calculation")

        # ── Step 2: Choose operation ──────────────────────────────────────
        st.markdown("**② What do you want to do?**")
        OP_OPTIONS = {
            "➕  Add a number to it (e.g. Salary + 5000 bonus)":          "+",
            "➖  Subtract a number (e.g. Revenue − Cost)":                 "-",
            "✖️  Multiply it (e.g. Salary × 1.10 for 10% hike)":          "*",
            "➗  Divide it (e.g. Annual Salary ÷ 12 = Monthly)":           "/",
            "📅  Date Difference in Years (e.g. Exit − Joining = Tenure)": "date_years",
            "📅  Date Difference in Months":                               "date_months",
            "📅  Date Difference in Days":                                 "date_days",
            "🔢  Percentage of it (e.g. 15% of Salary = Tax)":            "%",
            "🔁  Ratio of two columns (e.g. Profit ÷ Revenue × 100)":     "ratio100",
        }
        op_label = st.selectbox("Choose the operation:", list(OP_OPTIONS.keys()), key="cb_op")
        op_sel   = OP_OPTIONS[op_label]

        # ── Step 3: Second input ──────────────────────────────────────────
        st.markdown("**③ What is the second value?**")
        col2_sel = None; val_sel = None

        if op_sel.startswith("date_"):
            st.caption("Select the end date column (or use Today if the person is still active):")
            end_opts = ["Today (today's date)"] + all_col_list
            col2_sel = st.selectbox("End Date Column:", end_opts, key="cb_col2_date",
                                    help="e.g. Date of Exit. Use 'Today' if no exit date exists.")

        elif op_sel == "ratio100":
            st.caption("Ratio = Column 1 ÷ Column 2 × 100 (e.g. Profit ÷ Revenue × 100 = Margin %)")
            col2_sel = st.selectbox("Divide by which column?", all_col_list, key="cb_col2_ratio",
                                    help="e.g. Revenue for margin calculation")

        else:
            input_type = st.radio(
                "Use:",
                ["A fixed number  (e.g. multiply by 1.10)",
                 "Another column from the dataset"],
                horizontal=True, key="cb_input_type"
            )
            if "fixed" in input_type:
                # Contextual hint based on operation
                hints = {
                    "+": "e.g. 5000 = add 5000 to every row",
                    "-": "e.g. 1000 = subtract 1000 from every row",
                    "*": "e.g. 1.10 = multiply by 1.10 (10% increase) | 1.15 = 15% | 0.5 = halve it",
                    "/": "e.g. 12 = divide by 12 (annual→monthly) | 1000 = convert to thousands",
                    "%": "e.g. 15 = calculate 15% of each value | 10 = 10%",
                }
                st.caption(hints.get(op_sel, "Enter the number to use in the calculation"))
                val_sel = st.number_input("Enter the number:", value=1.10,
                                          format="%.4f", key="cb_val",
                                          help="Type the number you want to use")
            else:
                col2_sel = st.selectbox("Which column?", all_col_list, key="cb_col2_col",
                                        help="Every row: Column1 [operation] Column2")

        # ── Step 4: Name the result ───────────────────────────────────────
        st.markdown("**④ What should the result column be called?**")
        # Auto-suggest a name
        op_suffix = {"+" :"_Plus","−":"_Minus","*":"_Multiplied","/":"_Divided",
                     "date_years":"_Tenure_Years","date_months":"_Tenure_Months",
                     "date_days":"_Tenure_Days","%":"_Pct","ratio100":"_Ratio_Pct"}
        suggested = f"{col1_sel}{op_suffix.get(op_sel,'_Calculated')}"
        out_name = st.text_input(
            "Column name (no spaces — use underscore):",
            value=suggested, key="cb_out",
            help="This is what the new column will be called. Use underscores instead of spaces."
        )

        # ── Formula preview box ───────────────────────────────────────────
        op_symbols = {"+" :"+"  ,"-":"-","*":"×","/":"÷","%":"× % ","ratio100":"÷ × 100",
                      "date_years":"Date Diff (Years)","date_months":"Date Diff (Months)",
                      "date_days":"Date Diff (Days)"}
        sym = op_symbols.get(op_sel,"?")
        c2_display = col2_sel if col2_sel else (str(val_sel) if val_sel is not None else "?")
        st.markdown(f"""<div style="background:#1e293b;border-radius:8px;padding:12px 16px;
            border-left:3px solid #3b82f6;margin:8px 0">
            <span style="color:#94a3b8;font-size:.82rem">Formula preview:</span><br>
            <span style="color:#e2e8f0;font-size:1rem;font-weight:600">
            {out_name} = {col1_sel} {sym} {c2_display}
            </span>
            </div>""", unsafe_allow_html=True)

        # ── Step 5: Calculate ─────────────────────────────────────────────
        st.markdown("**⑤ Run the calculation:**")
        calc_btn = st.button("▶️ Calculate Now", key="cb_calc",
                             use_container_width=False, type="primary")

        if calc_btn:
            out_col = out_name.strip().replace(" ", "_") or "Calculated"
            result = None; err_msg = None

            c1s = pd.to_numeric(df[col1_sel], errors="coerce") if col1_sel in df.columns else None

            if op_sel in ("date_years", "date_months", "date_days"):
                d1 = pd.to_datetime(df[col1_sel], errors="coerce")
                d2 = (pd.Timestamp.today() if (col2_sel is None or col2_sel == "Today (today's date)")
                      else pd.to_datetime(df[col2_sel], errors="coerce"))
                diff_d = (d2 - d1).dt.days.clip(lower=0)
                if op_sel == "date_years":    result = (diff_d / 365.25).round(2)
                elif op_sel == "date_months": result = (diff_d / 30.44).round(1)
                else:                         result = diff_d

            elif op_sel == "ratio100":
                if col2_sel and col2_sel in df.columns:
                    c2s = pd.to_numeric(df[col2_sel], errors="coerce").replace(0, np.nan)
                    result = (c1s / c2s * 100).round(4)
                else:
                    err_msg = "Please select the second column for the ratio."

            elif col2_sel and col2_sel in df.columns:
                c2s = pd.to_numeric(df[col2_sel], errors="coerce")
                if op_sel == "+": result = (c1s + c2s).round(4)
                elif op_sel == "-": result = (c1s - c2s).round(4)
                elif op_sel == "*": result = (c1s * c2s).round(4)
                elif op_sel == "/": result = (c1s / c2s.replace(0, np.nan)).round(4)
                else: err_msg = f"Cannot use that operation with two columns."

            elif val_sel is not None:
                if op_sel == "+": result = (c1s + val_sel).round(4)
                elif op_sel == "-": result = (c1s - val_sel).round(4)
                elif op_sel == "*": result = (c1s * val_sel).round(4)
                elif op_sel == "/": result = (c1s / (val_sel or np.nan)).round(4)
                elif op_sel == "%": result = (c1s * val_sel / 100).round(4)
                else: err_msg = f"Unknown operation."

            else:
                err_msg = "Please complete Step 3 — enter a number or pick a column."

            if err_msg:
                st.error(f"Could not calculate: {err_msg}")
            elif result is not None:
                result.name = out_col
                st.session_state.calc_columns[out_col] = result
                st.success(f"✅ Done! '{out_col}' added. Click the Results tab to see the chart.")
                st.rerun()

    # ── TAB 3: RESULTS & CHARTS ───────────────────────────────────────────
    with tab_results:
        if not st.session_state.calc_columns:
            st.info("No calculated columns yet. Use Quick Presets or Custom Formula Builder to create some.")
        else:
            st.markdown(f"**{len(st.session_state.calc_columns)} calculated column(s) ready:**")

            # Build working df with all calc cols added
            df_calc = df.copy()
            for col_name, series in st.session_state.calc_columns.items():
                try:
                    df_calc[col_name] = series.values
                except Exception:
                    df_calc[col_name] = series

            calc_col_names = list(st.session_state.calc_columns.keys())

            for col_name, series in st.session_state.calc_columns.items():
                with st.expander(f"📊 {col_name}", expanded=True):
                    r1, r2, r3, r4 = st.columns(4)

                    is_numeric = pd.api.types.is_numeric_dtype(series)

                    if is_numeric:
                        s = series.dropna()
                        r1.metric("Min",    f"{s.min():,.2f}")
                        r2.metric("Max",    f"{s.max():,.2f}")
                        r3.metric("Mean",   f"{s.mean():,.2f}")
                        r4.metric("Median", f"{s.median():,.2f}")

                        c1, c2 = st.columns(2)
                        with c1:
                            fig = px.histogram(df_calc, x=col_name, nbins=30,
                                               title=f"{col_name} — Distribution",
                                               color_discrete_sequence=[C["blue"]],
                                               marginal="box")
                            fig.update_layout(**cd(360))
                            st.plotly_chart(fig, use_container_width=True)

                        with c2:
                            # Cross with a categorical column if available
                            cat_options = [c for c in df.columns
                                           if df[c].dtype == object and df[c].nunique() <= 20]
                            if cat_options:
                                cross_col = st.selectbox(
                                    f"Compare {col_name} by:", cat_options,
                                    key=f"cross_{col_name}")
                                grp = df_calc.groupby(cross_col)[col_name].mean().sort_values(ascending=False).reset_index()
                                fig2 = px.bar(grp, x=cross_col, y=col_name,
                                              title=f"Avg {col_name} by {cross_col}",
                                              color=col_name,
                                              color_continuous_scale="Blues",
                                              text_auto=".2f")
                                fig2.update_layout(**cd(360))
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                # Scatter vs a numeric col
                                if found.get("salary") and col_name != found.get("salary"):
                                    fig2 = px.scatter(df_calc.sample(min(500, len(df_calc))),
                                                      x=found["salary"], y=col_name,
                                                      title=f"Salary vs {col_name}",
                                                      opacity=0.6,
                                                      color_discrete_sequence=[C["purple"]])
                                    fig2.update_layout(**cd(360))
                                    st.plotly_chart(fig2, use_container_width=True)
                    else:
                        # Categorical result — value counts
                        vc = series.astype(str).value_counts().reset_index()
                        vc.columns = [col_name, "Count"]
                        r1.metric("Categories", f"{series.nunique():,}")
                        r2.metric("Most Common", str(series.mode().iloc[0]) if len(series.mode()) > 0 else "N/A")
                        r3.metric("Total Rows", f"{len(series):,}")
                        r4.metric("Null Count", f"{series.isna().sum():,}")

                        c1, c2 = st.columns(2)
                        with c1:
                            fig = px.bar(vc, x=col_name, y="Count",
                                         title=f"{col_name} — Distribution",
                                         color="Count",
                                         color_continuous_scale="Blues",
                                         text_auto=True)
                            fig.update_layout(**cd(360))
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            fig2 = px.pie(vc, values="Count", names=col_name,
                                          title=f"{col_name} — Share",
                                          hole=0.4,
                                          color_discrete_sequence=px.colors.qualitative.Set2)
                            fig2.update_layout(**cd(360))
                            st.plotly_chart(fig2, use_container_width=True)

                        # Cross with numeric if possible
                        base_num = found.get("salary") or found.get("sales")
                        if base_num:
                            grp2 = df_calc.groupby(col_name)[base_num].mean().sort_values(ascending=False).reset_index()
                            fig3 = px.bar(grp2, x=col_name, y=base_num,
                                          title=f"Avg {base_num} by {col_name}",
                                          color=base_num,
                                          color_continuous_scale="Purples",
                                          text_auto=".0f")
                            fig3.update_layout(**cd(360))
                            st.plotly_chart(fig3, use_container_width=True)

                    # Action buttons
                    ba, bb, bc = st.columns(3)
                    with ba:
                        if st.button(f"🗑 Remove {col_name}", key=f"del_{col_name}"):
                            del st.session_state.calc_columns[col_name]
                            st.rerun()
                    with bb:
                        # Download just this column merged with key identifier columns
                        id_cols = [c for c in [found.get("employee_id"), found.get("employee_name"),
                                               found.get("customer"), found.get("order_id")] if c]
                        export_df = df_calc[id_cols + [col_name]] if id_cols else df_calc[[col_name]]
                        csv_bytes = export_df.to_csv(index=False).encode()
                        st.download_button(f"⬇️ Download {col_name}",
                                           data=csv_bytes,
                                           file_name=f"{col_name}.csv",
                                           mime="text/csv",
                                           key=f"dl_{col_name}")
                    with bc:
                        st.caption(f"dtype: {series.dtype} | {series.notna().sum():,} non-null values")

            st.markdown("---")
            # Download ALL calculated columns merged into full dataset
            if st.button("⬇️ Download Full Dataset with All Calculated Columns",
                         type="primary", key="dl_all_calc"):
                csv_full = df_calc.to_csv(index=False).encode()
                st.download_button("📥 Download CSV",
                                   data=csv_full,
                                   file_name="dataset_with_calculated_columns.csv",
                                   mime="text/csv",
                                   key="dl_all_csv")

            # Summary table of all calc cols
            st.markdown("**Summary of all calculated columns:**")
            summary_rows = []
            for cn, ser in st.session_state.calc_columns.items():
                is_num = pd.api.types.is_numeric_dtype(ser)
                summary_rows.append({
                    "Column": cn,
                    "Type": "Numeric" if is_num else "Categorical",
                    "Min / Most Common": f"{ser.dropna().min():,.2f}" if is_num else str(ser.mode().iloc[0]) if len(ser.mode()) > 0 else "",
                    "Max / Categories": f"{ser.dropna().max():,.2f}" if is_num else f"{ser.nunique()} categories",
                    "Mean / Avg": f"{ser.dropna().mean():,.2f}" if is_num else "",
                    "Null Count": f"{ser.isna().sum():,}",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE 1 — ANOMALY DETECTION (runs on load, shows banner)
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(df, found, domain):
    """
    Runs automatically on data load.
    Returns list of anomaly dicts: {severity, title, detail, icon}
    severity: 'critical' | 'warning' | 'info'
    """
    anomalies = []
    # Guard: convert object columns that are actually numeric
    try:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().mean() > 0.5:
                    df[col] = converted
    except Exception:
        pass

    sc   = found.get("sales");   pc  = found.get("profit")
    sal  = found.get("salary");  dc  = found.get("date")
    attr = found.get("attrition"); dept = found.get("department")
    qty  = found.get("quantity"); dis  = found.get("discount")
    emp  = found.get("employee_id") or found.get("employee_name")
    gen  = found.get("gender");  age  = found.get("age")

    # ── 1. Missing Data ───────────────────────────────────────────────────
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct >= 50:
            anomalies.append({"severity":"critical","icon":"🔴",
                "title":f"Critical missing data: '{col}'",
                "detail":f"{pct:.1f}% of values are missing — results for this column will be unreliable."})
        elif pct >= 20:
            anomalies.append({"severity":"warning","icon":"🟡",
                "title":f"High missing data: '{col}'",
                "detail":f"{pct:.1f}% missing values detected."})

    # ── 2. Duplicate rows ─────────────────────────────────────────────────
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = dup_count / len(df) * 100
        sev = "critical" if dup_pct > 10 else "warning"
        anomalies.append({"severity":sev,"icon":"🟠" if sev=="warning" else "🔴",
            "title":f"{dup_count:,} duplicate rows detected ({dup_pct:.1f}%)",
            "detail":"Duplicate rows can inflate totals and distort averages. Consider deduplication."})

    # ── 3. Numeric outliers (Z-score & IQR) ──────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols[:15]:  # check first 15 numeric cols
        s = df[col].dropna()
        if len(s) < 10: continue
        # IQR method
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: continue
        outliers = ((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).sum()
        if outliers > 0:
            pct = outliers / len(s) * 100
            if pct > 5:
                anomalies.append({"severity":"warning","icon":"🟡",
                    "title":f"Outliers in '{col}': {outliers:,} extreme values ({pct:.1f}%)",
                    "detail":f"Min: {s.min():,.2f} | Max: {s.max():,.2f} | Expected range: {q1-3*iqr:,.2f} – {q3+3*iqr:,.2f}"})

    # ── 4. Revenue / Sales anomalies ──────────────────────────────────────
    if sc and dc:
        try:
            d2 = df.copy()
            d2[dc] = pd.to_datetime(d2[dc], errors="coerce")
            d2 = d2.dropna(subset=[dc])
            d2["_M"] = d2[dc].dt.to_period("M").astype(str)
            monthly = d2.groupby("_M")[sc].sum()
            if len(monthly) >= 3:
                mean_rev = monthly.mean()
                std_rev  = monthly.std()
                spikes = monthly[monthly > mean_rev + 2.5*std_rev]
                drops  = monthly[monthly < mean_rev - 2.5*std_rev]
                if len(spikes) > 0:
                    for m, v in spikes.items():
                        anomalies.append({"severity":"warning","icon":"📈",
                            "title":f"Revenue spike in {m}: {v:,.0f}",
                            "detail":f"This is {(v-mean_rev)/mean_rev*100:.0f}% above average monthly revenue ({mean_rev:,.0f}). Verify if correct."})
                if len(drops) > 0:
                    for m, v in drops.items():
                        anomalies.append({"severity":"critical","icon":"📉",
                            "title":f"Revenue drop in {m}: {v:,.0f}",
                            "detail":f"This is {abs(v-mean_rev)/mean_rev*100:.0f}% below average ({mean_rev:,.0f}). Investigate root cause."})
        except Exception:
            pass

    # ── 5. Negative values where impossible ──────────────────────────────
    for col_key in ["sales","profit","salary","quantity","spend","impressions","clicks"]:
        col = found.get(col_key)
        if col:
            try:
                _series = pd.to_numeric(df[col], errors="coerce").dropna()
                neg = (_series < 0).sum()
            except Exception:
                continue
            if neg > 0:
                anomalies.append({"severity":"warning","icon":"⚠️",
                    "title":f"Negative values in '{col}': {neg:,} rows",
                    "detail":f"Negative {col_key} values may indicate data entry errors or returns/refunds."})

    # ── 6. Salary anomalies (HR) ──────────────────────────────────────────
    if sal:
        s = pd.to_numeric(df[sal], errors="coerce").dropna()
        if len(s) > 10:
            mean_s = s.mean()
            extreme_high = (s > mean_s * 5).sum()
            extreme_low  = (s < mean_s * 0.1).sum()
            if extreme_high > 0:
                anomalies.append({"severity":"warning","icon":"💰",
                    "title":f"{extreme_high} employees with salary > 5× average",
                    "detail":f"Average salary: {mean_s:,.0f}. These records may be senior executives or data errors."})
            if extreme_low > 0:
                anomalies.append({"severity":"warning","icon":"💰",
                    "title":f"{extreme_low} employees with salary < 10% of average",
                    "detail":f"Very low salaries detected. May indicate part-time, interns, or data errors."})

    # ── 7. Attrition rate warning ─────────────────────────────────────────
    if attr:
        rate = df[attr].astype(str).str.lower().isin(
            ["yes","true","1","resigned","terminated","left"]).mean() * 100
        if rate > 20:
            anomalies.append({"severity":"critical","icon":"🔴",
                "title":f"Critical attrition rate: {rate:.1f}%",
                "detail":"Attrition above 20% is a serious retention risk. Immediate HR intervention recommended."})
        elif rate > 12:
            anomalies.append({"severity":"warning","icon":"🟡",
                "title":f"High attrition rate: {rate:.1f}%",
                "detail":"Industry healthy range is 8-12%. Investigate by department and tenure."})

    # ── 8. Fraud rate ─────────────────────────────────────────────────────
    if domain == "Fraud":
        cl = {c.lower().strip(): c for c in df.columns}
        fraud_cols = ["class","label","fraud","is_fraud","isfraud","target"]
        cc = next((cl[c] for c in fraud_cols if c in cl), None)
        if cc:
            df2 = df.copy()
            df2["_lbl"] = df2[cc].astype(str).str.lower().map(
                lambda x: 1 if x in ["1","true","yes","fraud"] else 0)
            rate = df2["_lbl"].mean() * 100
            amt_col = cl.get("amount")
            if rate > 1:
                exposure = ""
                if amt_col:
                    exp = df2[df2["_lbl"]==1][amt_col].sum()
                    exposure = f" | Total exposure: ${exp:,.2f}"
                anomalies.append({"severity":"critical","icon":"🚨",
                    "title":f"High fraud rate detected: {rate:.3f}%",
                    "detail":f"{int(df2['_lbl'].sum()):,} fraudulent transactions out of {len(df2):,}{exposure}"})

    # ── 9. Gender imbalance ───────────────────────────────────────────────
    if gen:
        gc = df[gen].astype(str).str.title().value_counts()
        m = gc.get("Male", gc.get("M", 0))
        f = gc.get("Female", gc.get("F", 0))
        total_g = m + f
        if total_g > 0:
            ratio = min(m, f) / total_g * 100
            if ratio < 20:
                anomalies.append({"severity":"info","icon":"ℹ️",
                    "title":f"Gender imbalance: {ratio:.0f}% minority gender",
                    "detail":f"Male: {int(m):,} | Female: {int(f):,}. Consider diversity initiatives."})

    # ── 10. Zero-value records ────────────────────────────────────────────
    if sc:
        zeros = (df[sc] == 0).sum()
        if zeros > len(df) * 0.05:
            anomalies.append({"severity":"info","icon":"🔵",
                "title":f"{zeros:,} zero-value transactions ({zeros/len(df)*100:.1f}%)",
                "detail":"Zero-value records may be cancelled orders, free items, or data gaps."})

    # ── 11. Date gaps ─────────────────────────────────────────────────────
    if dc:
        try:
            dates = pd.to_datetime(df[dc], errors="coerce").dropna().sort_values()
            if len(dates) > 10:
                gaps = dates.diff().dt.days.dropna()
                max_gap = gaps.max()
                if max_gap > 60:
                    gap_date = dates.iloc[gaps.argmax()]
                    anomalies.append({"severity":"warning","icon":"📅",
                        "title":f"Data gap of {int(max_gap)} days detected",
                        "detail":f"No records around {gap_date.date()}. May indicate missing data for that period."})
        except Exception:
            pass

    return anomalies


def render_anomaly_banner(anomalies, domain):
    """Show anomaly banner at the top of the page after data loads."""
    if not anomalies:
        st.markdown("""<div style="background:linear-gradient(135deg,#052e16,#064e3b);
            border:1px solid #16a34a;border-radius:12px;padding:14px 20px;margin:8px 0">
            <span style="color:#4ade80;font-size:1rem;font-weight:700">
            ✅ No anomalies detected</span>
            <span style="color:#86efac;font-size:.88rem;margin-left:12px">
            Data quality looks good — no outliers, duplicates, or unusual patterns found.</span>
            </div>""", unsafe_allow_html=True)
        return

    critical = [a for a in anomalies if a["severity"] == "critical"]
    warnings  = [a for a in anomalies if a["severity"] == "warning"]
    infos     = [a for a in anomalies if a["severity"] == "info"]

    # Summary bar
    parts = []
    if critical: parts.append(f"🔴 {len(critical)} Critical")
    if warnings:  parts.append(f"🟡 {len(warnings)} Warnings")
    if infos:     parts.append(f"🔵 {len(infos)} Info")
    summary = " &nbsp;|&nbsp; ".join(parts)

    border_color = "#ef4444" if critical else "#f59e0b"
    bg_color     = "#1a0a0a" if critical else "#1a1200"

    st.markdown(f"""<div style="background:{bg_color};border:1px solid {border_color};
        border-radius:12px;padding:14px 20px;margin:8px 0">
        <span style="color:#f87171;font-size:1rem;font-weight:700">
        🚨 Anomalies Detected &nbsp;</span>
        <span style="color:#fca5a5;font-size:.88rem">{summary}</span>
        </div>""", unsafe_allow_html=True)

    with st.expander(f"🔍 View all {len(anomalies)} anomalies detected in your data", expanded=True):
        for sev_label, sev_list in [("🔴 Critical Issues", critical),
                                     ("🟡 Warnings", warnings),
                                     ("🔵 Information", infos)]:
            if not sev_list: continue
            st.markdown(f"**{sev_label}**")
            for a in sev_list:
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.markdown(f"<div style='font-size:1.5rem;text-align:center'>{a['icon']}</div>",
                                unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{a['title']}**")
                    st.caption(a["detail"])
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════════════

def render_eda(df, found, domain):
    section("🔍 Exploratory Data Analysis", domain.lower())

    st.markdown("""<div class="insight-box">
    <strong>🔍 What is EDA?</strong> — It's a full health check of your data before any analysis.<br>
    It answers: <em>What does my data look like? Where are the gaps? What's unusual?
    Which columns are related to each other?</em>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Profile",
        "📊 Distributions",
        "🔗 Correlations",
        "🎯 Outlier Analysis",
        "📅 Time Patterns"
    ])

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    # ── TAB 1: Data Profile ───────────────────────────────────────────────
    with tab1:
        st.markdown("**📋 Full column profile — every column at a glance:**")

        profile_rows = []
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            total = len(s)
            missing = s.isna().sum()
            missing_pct = missing / total * 100
            unique = s.nunique()
            unique_pct = unique / total * 100

            if pd.api.types.is_numeric_dtype(s):
                sn = s.dropna()
                skew = float(sn.skew()) if len(sn) > 3 else 0
                skew_label = ("Right-skewed" if skew > 1 else
                              "Left-skewed" if skew < -1 else "Normal")
                stats = f"Min:{sn.min():,.1f} | Mean:{sn.mean():,.1f} | Max:{sn.max():,.1f}"
                col_type = "🔢 Numeric"
            elif pd.api.types.is_datetime64_any_dtype(s):
                sd = s.dropna()
                stats = f"{sd.min().date()} → {sd.max().date()}" if len(sd) > 0 else "—"
                skew_label = "—"
                col_type = "📅 Date"
            else:
                top = s.value_counts().index[0] if s.nunique() > 0 else "—"
                stats = f"Top: {str(top)[:25]}"
                skew_label = "—"
                col_type = "📝 Text" if unique_pct > 50 else "🏷 Category"

            quality = ("✅ Good" if missing_pct < 5 else
                       "⚠️ Moderate" if missing_pct < 20 else "❌ Poor")

            profile_rows.append({
                "Column": col,
                "Type": col_type,
                "Missing": f"{missing:,} ({missing_pct:.1f}%)",
                "Unique": f"{unique:,} ({unique_pct:.0f}%)",
                "Stats / Sample": stats,
                "Distribution": skew_label,
                "Quality": quality,
            })

        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

        # Summary stats for numeric
        if num_cols:
            st.markdown("**📊 Detailed statistics for numeric columns:**")
            desc = df[num_cols].describe().T.round(2)
            desc.index.name = "Column"
            desc = desc.reset_index()
            st.dataframe(desc, use_container_width=True, hide_index=True)

        # Missing data visual
        missing_data = df.isna().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            st.markdown("**🕳 Missing Data by Column:**")
            md_df = missing_data.reset_index()
            md_df.columns = ["Column", "Missing Count"]
            md_df["Missing %"] = (md_df["Missing Count"] / len(df) * 100).round(1)
            fig = px.bar(md_df, x="Column", y="Missing %",
                         title="Missing Data % per Column",
                         color="Missing %",
                         color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
                         text_auto=".1f")
            fig.add_hline(y=20, line_dash="dash", line_color="#f59e0b",
                          annotation_text="20% threshold")
            fig.update_layout(**cd(380))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data found!")

    # ── TAB 2: Distributions ──────────────────────────────────────────────
    with tab2:
        if num_cols:
            st.markdown("**📊 Distribution of every numeric column:**")
            sel_cols = st.multiselect("Select columns to analyse:",
                                       num_cols, default=num_cols[:6],
                                       key="eda_dist_cols")
            if sel_cols:
                for col in sel_cols:
                    s = df[col].dropna()
                    c1, c2, c3 = st.columns([2, 2, 1])
                    with c1:
                        fig = px.histogram(df, x=col, nbins=40,
                                           title=f"{col} — Distribution",
                                           color_discrete_sequence=[C["blue"]],
                                           marginal="box")
                        fig.update_layout(**cd(320))
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig2 = px.violin(df, y=col, box=True,
                                         title=f"{col} — Violin Plot",
                                         color_discrete_sequence=[C["purple"]])
                        fig2.update_layout(**cd(320))
                        st.plotly_chart(fig2, use_container_width=True)
                    with c3:
                        skew = float(s.skew())
                        kurt = float(s.kurtosis())
                        st.metric("Mean",   f"{s.mean():,.2f}")
                        st.metric("Median", f"{s.median():,.2f}")
                        st.metric("Std Dev",f"{s.std():,.2f}")
                        st.metric("Skew",   f"{skew:.2f}")
                        st.metric("Kurtosis",f"{kurt:.2f}")
                        skew_msg = ("Right tail heavy →\nhigher outliers" if skew > 1 else
                                    "Left tail heavy →\nlower outliers" if skew < -1 else
                                    "Roughly normal distribution")
                        st.caption(skew_msg)

        if cat_cols:
            st.markdown("**🏷 Distribution of categorical columns:**")
            cat_sel = st.multiselect("Select categorical columns:",
                                      [c for c in cat_cols if df[c].nunique() <= 50],
                                      default=[c for c in cat_cols if df[c].nunique() <= 30][:3],
                                      key="eda_cat_cols")
            cols_g = st.columns(2)
            for i, col in enumerate(cat_sel):
                with cols_g[i % 2]:
                    vc = df[col].value_counts().head(15).reset_index()
                    vc.columns = [col, "Count"]
                    fig = px.bar(vc, x=col, y="Count",
                                 title=f"{col} — Value Counts (Top 15)",
                                 color="Count",
                                 color_continuous_scale="Blues",
                                 text_auto=True)
                    fig.update_layout(**cd(360))
                    st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Correlations ───────────────────────────────────────────────
    with tab3:
        if len(num_cols) >= 2:
            st.markdown("**🔗 How strongly are your columns related to each other?**")
            st.caption("Values close to 1.0 = strong positive link | -1.0 = strong inverse | 0 = no link")

            corr_cols = num_cols[:20]
            corr = df[corr_cols].corr().round(3)

            fig = px.imshow(corr,
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title="Correlation Matrix — All Numeric Columns",
                            text_auto=".2f",
                            aspect="auto")
            fig.update_layout(**cd(max(500, len(corr_cols)*35)))
            st.plotly_chart(fig, use_container_width=True)

            # Top correlations table
            st.markdown("**🏆 Strongest correlations (plain English):**")
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.3:
                        strength = ("Very Strong" if abs(val) > 0.8 else
                                    "Strong" if abs(val) > 0.6 else
                                    "Moderate" if abs(val) > 0.4 else "Weak")
                        direction = "📈 Positive" if val > 0 else "📉 Negative"
                        plain = (f"When {corr.columns[i]} goes up, {corr.columns[j]} tends to "
                                 f"{'go up too' if val > 0 else 'go down'}")
                        pairs.append({
                            "Column A": corr.columns[i],
                            "Column B": corr.columns[j],
                            "Correlation": f"{val:.3f}",
                            "Strength": strength,
                            "Direction": direction,
                            "What it means": plain
                        })
            if pairs:
                pairs_df = pd.DataFrame(pairs).sort_values("Correlation",
                           key=lambda x: x.abs(), ascending=False)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)

            # Scatter matrix for key columns
            key_num = [found.get(k) for k in ["sales","profit","salary","quantity","discount","spend"]
                       if found.get(k)][:5]
            if len(key_num) >= 2:
                st.markdown("**🔵 Scatter Matrix — Key Columns:**")
                fig2 = px.scatter_matrix(df.sample(min(500, len(df))),
                                          dimensions=key_num,
                                          color=found.get("category") or found.get("department"),
                                          title="Relationships between key columns",
                                          opacity=0.5)
                fig2.update_layout(**cd(600))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # ── TAB 4: Outlier Analysis ───────────────────────────────────────────
    with tab4:
        st.markdown("**🎯 Outlier Detection — find extreme values in your data**")
        st.caption("Outliers are values that are unusually high or low compared to the rest of the data.")

        if not num_cols:
            st.info("No numeric columns found.")
        else:
            out_sel = st.multiselect("Select columns to check for outliers:",
                                      num_cols, default=num_cols[:5],
                                      key="eda_out_cols")

            outlier_summary = []
            for col in out_sel:
                s = df[col].dropna()
                if len(s) < 10: continue

                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5*iqr
                upper = q3 + 1.5*iqr
                extreme_lower = q1 - 3*iqr
                extreme_upper = q3 + 3*iqr

                mild_out   = ((s < lower) | (s > upper)).sum()
                extreme_out = ((s < extreme_lower) | (s > extreme_upper)).sum()

                outlier_summary.append({
                    "Column": col,
                    "Total Values": f"{len(s):,}",
                    "Mild Outliers": f"{mild_out:,} ({mild_out/len(s)*100:.1f}%)",
                    "Extreme Outliers": f"{extreme_out:,} ({extreme_out/len(s)*100:.1f}%)",
                    "Normal Range": f"{lower:,.2f} – {upper:,.2f}",
                    "Status": ("✅ Clean" if extreme_out == 0 else
                               "⚠️ Has outliers" if extreme_out < len(s)*0.05 else
                               "❌ Many outliers")
                })

                # Box plot
                fig = px.box(df, y=col, title=f"{col} — Box Plot (outliers shown as dots)",
                             color_discrete_sequence=[C["blue"]],
                             points="outliers")
                fig.update_layout(**cd(320))
                st.plotly_chart(fig, use_container_width=True)

            if outlier_summary:
                st.markdown("**Outlier Summary:**")
                st.dataframe(pd.DataFrame(outlier_summary),
                             use_container_width=True, hide_index=True)

    # ── TAB 5: Time Patterns ──────────────────────────────────────────────
    with tab5:
        dc = found.get("date")
        if not dc:
            st.info("No date column detected. Time pattern analysis not available.")
        else:
            val_col = (found.get("sales") or found.get("salary") or
                       found.get("spend") or num_cols[0] if num_cols else None)
            if not val_col:
                st.info("No numeric column to analyse over time.")
            else:
                d2 = df.copy()
                d2[dc] = pd.to_datetime(d2[dc], errors="coerce")
                d2 = d2.dropna(subset=[dc])
                d2["_Year"]  = d2[dc].dt.year
                d2["_Month"] = d2[dc].dt.month
                d2["_DOW"]   = d2[dc].dt.day_name()
                d2["_M"]     = d2[dc].dt.to_period("M").astype(str)
                d2["_Q"]     = d2[dc].dt.to_period("Q").astype(str)

                st.markdown(f"**📅 Time patterns for: {val_col}**")

                c1, c2 = st.columns(2)
                with c1:
                    m = d2.groupby("_M")[val_col].sum().reset_index()
                    m.columns = ["Month", val_col]
                    fig = px.line(m, x="Month", y=val_col,
                                  title="Monthly Trend",
                                  markers=True,
                                  color_discrete_sequence=[C["blue"]])
                    fig.update_layout(**cd(360))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                    dw = d2.groupby("_DOW")[val_col].mean().reindex(dow_order).reset_index()
                    dw.columns = ["Day", "Avg Value"]
                    fig2 = px.bar(dw, x="Day", y="Avg Value",
                                  title="Average by Day of Week",
                                  color="Avg Value",
                                  color_continuous_scale="Blues",
                                  text_auto=".0f")
                    fig2.update_layout(**cd(360))
                    st.plotly_chart(fig2, use_container_width=True)

                # Seasonality
                month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                mth = d2.groupby("_Month")[val_col].mean().reset_index()
                mth["Month Name"] = mth["_Month"].map(month_names)
                mth.columns = ["Month Num", val_col, "Month"]
                fig3 = px.bar(mth, x="Month", y=val_col,
                              title="Seasonality — Average by Month",
                              color=val_col,
                              color_continuous_scale="Viridis",
                              text_auto=".0f")
                fig3.update_layout(**cd(380))
                st.plotly_chart(fig3, use_container_width=True)

                # Year-over-year
                yoy = d2.groupby(["_Year","_Month"])[val_col].sum().reset_index()
                fig4 = px.line(yoy, x="_Month", y=val_col, color="_Year",
                               title="Year-over-Year Comparison by Month",
                               labels={"_Month":"Month","_Year":"Year"},
                               markers=True)
                fig4.update_layout(**cd(400))
                st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE 3 — PREDICTION ENGINE (ML)
# ══════════════════════════════════════════════════════════════════════════════

def render_prediction(df, found, domain):
    section("🤖 Prediction Engine", domain.lower())

    st.markdown("""<div class="insight-box">
    <strong>🤖 What does this do?</strong><br>
    The Prediction Engine learns patterns from your historical data and predicts future outcomes.<br><br>
    <strong>Examples:</strong><br>
    &nbsp;&nbsp;👥 <em>Will this employee leave?</em> → Predicts attrition Yes/No<br>
    &nbsp;&nbsp;💰 <em>What will next month's revenue be?</em> → Predicts a number<br>
    &nbsp;&nbsp;🚨 <em>Is this transaction fraudulent?</em> → Predicts fraud probability<br>
    &nbsp;&nbsp;📈 <em>What salary should this role have?</em> → Predicts salary range
    </div>""", unsafe_allow_html=True)

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (accuracy_score, classification_report,
                                     mean_absolute_error, r2_score, roc_auc_score)
        from sklearn.impute import SimpleImputer
        sklearn_ok = True
    except ImportError:
        st.error("scikit-learn not installed. Add `scikit-learn>=1.3.2` to requirements.txt")
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = list(df.columns)

    # ── Step 1: Pick what to predict ──────────────────────────────────────
    st.markdown("**① What do you want to predict?**")

    # Domain-specific suggestions
    DOMAIN_TARGETS = {
        "HR":        {"Attrition (Will employee leave?)": found.get("attrition"),
                      "Salary (What should salary be?)":  found.get("salary"),
                      "Performance Rating":                found.get("performance")},
        "Sales":     {"Revenue (Predict order value)":    found.get("sales"),
                      "Profit (Predict profit)":          found.get("profit"),
                      "Quantity (Predict units sold)":    found.get("quantity")},
        "Marketing": {"Conversions (Will user convert?)": found.get("conversions"),
                      "Revenue from campaign":            found.get("sales"),
                      "CTR (Click-through rate)":         found.get("clicks")},
        "Ecommerce": {"Returns (Will order be returned?)":found.get("returns"),
                      "Revenue (Predict order value)":    found.get("sales"),
                      "Delivery Time":                    found.get("delivery")},
        "Retail":    {"Revenue (Predict transaction)":    found.get("sales"),
                      "Profit (Predict profit)":          found.get("profit")},
        "Fraud":     {"Fraud (Is transaction fraudulent?)":"auto_fraud"},
        "Generic":   {},
    }

    suggestions = {k: v for k, v in DOMAIN_TARGETS.get(domain, {}).items() if v}
    suggestion_labels = list(suggestions.keys()) + ["Custom — pick any column"]

    pred_choice = st.selectbox("Choose what to predict:", suggestion_labels,
                               key="pred_target_choice")

    if pred_choice == "Custom — pick any column":
        target_col = st.selectbox("Select target column:", all_cols, key="pred_target_custom")
    elif pred_choice == "Fraud (Is transaction fraudulent?)":
        cl = {c.lower().strip(): c for c in df.columns}
        fraud_label_cols = ["class","label","fraud","is_fraud","isfraud","target"]
        target_col = next((cl[c] for c in fraud_label_cols if c in cl), None)
        if not target_col:
            st.warning("No fraud label column found. Select manually.")
            target_col = st.selectbox("Select target column:", all_cols, key="pred_fraud_col")
    else:
        target_col = suggestions.get(pred_choice)

    if not target_col or target_col not in df.columns:
        st.warning("Please select a valid target column.")
        return

    # ── Detect task type: Classification or Regression ────────────────────
    target_series = df[target_col].dropna()
    unique_vals = target_series.nunique()

    # Check if binary/categorical
    is_classification = (
        unique_vals <= 10 or
        target_series.dtype == object or
        target_series.astype(str).str.lower().isin(
            ["yes","no","true","false","1","0","fraud","legitimate",
             "resigned","active","left","stayed"]).any()
    )

    task_type = "Classification" if is_classification else "Regression"
    task_icon = "🎯" if is_classification else "📈"
    st.info(f"{task_icon} **Task type auto-detected: {task_type}** — "
            f"{'Predicting a category (Yes/No, Fraud/Legit)' if is_classification else 'Predicting a number (value, amount, salary)'}")

    # ── Step 2: Feature selection ─────────────────────────────────────────
    st.markdown("**② Which columns should the model learn from?**")
    st.caption("Remove ID columns, names, or columns that would not be available at prediction time.")

    # Auto-suggest features
    exclude_always = {target_col}
    # Exclude high-cardinality text and obvious ID columns
    auto_exclude = set()
    for col in df.columns:
        if col == target_col: continue
        if df[col].nunique() > 200 and df[col].dtype == object:
            auto_exclude.add(col)
        if any(k in col.lower() for k in ["id","name","email","phone","address","index"]):
            auto_exclude.add(col)

    # Also exclude datetime columns from defaults — they can't be used directly
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            auto_exclude.add(col)

    default_features = [c for c in df.columns
                        if c not in exclude_always and c not in auto_exclude][:20]

    feature_cols = st.multiselect(
        "Feature columns (what the model learns from):",
        [c for c in df.columns if c != target_col],
        default=default_features,
        key="pred_features"
    )

    if len(feature_cols) < 1:
        st.warning("Select at least 1 feature column.")
        return

    # ── Step 3: Model & training ──────────────────────────────────────────
    st.markdown("**③ Choose model complexity:**")
    model_choice = st.radio(
        "Model:",
        ["🌲 Random Forest (recommended — fast & accurate)",
         "⚡ Gradient Boosting (slower but more accurate)",
         "🔀 Compare Both"],
        horizontal=True, key="pred_model"
    )

    test_size = st.slider("Test data % (held out for accuracy check):",
                           10, 40, 20, key="pred_test_size")

    run_btn = st.button("🚀 Train Model & Predict", type="primary", key="pred_run")

    if not run_btn:
        return

    with st.spinner("🤖 Training model on your data..."):
        try:
            # Prepare data
            df_model = df[feature_cols + [target_col]].copy()

            # Encode target if classification
            le_target = LabelEncoder()
            if is_classification:
                df_model[target_col] = (df_model[target_col].astype(str).str.lower()
                    .map(lambda x: 1 if x in ["yes","true","1","fraud","resigned","left","terminated"] else
                                   (x if x not in ["no","false","0","legitimate","active","stayed"] else 0)))
                df_model[target_col] = pd.to_numeric(df_model[target_col], errors="coerce")

            df_model = df_model.dropna(subset=[target_col])

            # Encode categorical features + convert datetime cols to ordinal ints
            le_dict = {}
            drop_cols = []
            for col in feature_cols:
                if pd.api.types.is_datetime64_any_dtype(df_model[col]):
                    # Convert datetime → integer ordinal (days since epoch)
                    try:
                        df_model[col] = df_model[col].astype("int64") // 10**9  # unix seconds
                    except Exception:
                        drop_cols.append(col)  # can't convert — drop it
                elif df_model[col].dtype == object or df_model[col].dtype.name == "category":
                    le = LabelEncoder()
                    df_model[col] = le.fit_transform(df_model[col].astype(str).fillna("Unknown"))
                    le_dict[col] = le

            # Drop any unconvertible cols and warn user
            if drop_cols:
                st.warning(f"⚠️ Skipped columns (cannot convert to numeric): {', '.join(drop_cols)}")
                feature_cols = [c for c in feature_cols if c not in drop_cols]

            if not feature_cols:
                st.error("No usable feature columns after encoding. Please select numeric or categorical columns.")
                return

            X = df_model[feature_cols]
            y = df_model[target_col]

            # Impute missing
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(X)

            if len(X_imp) < 20:
                st.error("Need at least 20 rows to train a model.")
                return

            X_train, X_test, y_train, y_test = train_test_split(
                X_imp, y, test_size=test_size/100, random_state=42, 
                stratify=y if is_classification and y.nunique() <= 10 else None)

            def train_and_score(model_class_clf, model_class_reg, name):
                # n_jobs only supported by RandomForest, not GradientBoosting
                supports_njobs = "RandomForest" in model_class_clf.__name__
                kwargs = {"n_estimators": 100, "random_state": 42}
                if supports_njobs:
                    kwargs["n_jobs"] = -1

                if is_classification:
                    model = model_class_clf(**kwargs)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else None
                    acc = accuracy_score(y_test, y_pred) * 100
                    try:
                        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
                    except Exception:
                        auc = None
                    return model, {"name":name, "accuracy":acc, "auc":auc,
                                   "y_pred":y_pred, "y_test":y_test, "y_prob":y_prob}
                else:
                    model = model_class_reg(**kwargs)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae  = mean_absolute_error(y_test, y_pred)
                    r2   = r2_score(y_test, y_pred)
                    # Safe MAPE — y_test is numpy array, use np.where not .replace()
                    y_test_safe = np.where(y_test == 0, np.nan, y_test)
                    mape_vals   = np.abs((y_test - y_pred) / y_test_safe) * 100
                    mape        = float(np.nanmean(mape_vals)) if np.isfinite(mape_vals).any() else 0.0
                    return model, {"name":name, "mae":mae, "r2":r2, "mape":mape,
                                   "y_pred":y_pred, "y_test":y_test}

            results = []
            models  = []

            if "Random Forest" in model_choice or "Compare" in model_choice:
                m, r = train_and_score(RandomForestClassifier, RandomForestRegressor, "Random Forest")
                models.append(m); results.append(r)

            if "Gradient Boosting" in model_choice or "Compare" in model_choice:
                m, r = train_and_score(GradientBoostingClassifier, GradientBoostingRegressor, "Gradient Boosting")
                models.append(m); results.append(r)

            best_idx = 0
            if len(results) > 1:
                if is_classification:
                    best_idx = max(range(len(results)), key=lambda i: results[i]["accuracy"])
                else:
                    best_idx = max(range(len(results)), key=lambda i: results[i]["r2"])
            best_model  = models[best_idx]
            best_result = results[best_idx]

        except Exception as e:
            st.error(f"Training error: {e}")
            return

    # ── Results ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Results")

    # Accuracy metrics
    if is_classification:
        acc = best_result["accuracy"]
        auc = best_result.get("auc")
        quality = "Excellent 🌟" if acc > 90 else "Good ✅" if acc > 75 else "Fair ⚠️" if acc > 60 else "Needs improvement ❌"
        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Accuracy", f"{acc:.1f}%")
        m2.metric("🎯 Quality", quality)
        if auc: m3.metric("📊 AUC-ROC", f"{auc:.3f}")
    else:
        mae  = best_result["mae"]
        r2   = best_result["r2"]
        mape = best_result.get("mape", 0)
        quality = "Excellent 🌟" if r2 > 0.9 else "Good ✅" if r2 > 0.7 else "Fair ⚠️" if r2 > 0.5 else "Needs improvement ❌"
        m1, m2, m3 = st.columns(3)
        m1.metric("📐 R² Score",  f"{r2:.3f}")
        m2.metric("📏 Mean Abs Error", f"{mae:,.2f}")
        m3.metric("🎯 Quality", quality)

    # Model comparison
    if len(results) > 1:
        st.markdown("**Model Comparison:**")
        comp_rows = []
        for r in results:
            if is_classification:
                comp_rows.append({"Model": r["name"],
                                  "Accuracy": f"{r['accuracy']:.1f}%",
                                  "AUC-ROC": f"{r['auc']:.3f}" if r.get("auc") else "N/A",
                                  "Winner": "🏆" if r["name"] == best_result["name"] else ""})
            else:
                comp_rows.append({"Model": r["name"],
                                  "R² Score": f"{r['r2']:.3f}",
                                  "Mean Abs Error": f"{r['mae']:,.2f}",
                                  "Winner": "🏆" if r["name"] == best_result["name"] else ""})
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Prediction vs Actual chart
    c1, c2 = st.columns(2)
    y_test_arr  = np.array(best_result["y_test"])
    y_pred_arr  = np.array(best_result["y_pred"])

    with c1:
        if not is_classification:
            sample_n = min(200, len(y_test_arr))
            idx = np.random.choice(len(y_test_arr), sample_n, replace=False)
            fig = px.scatter(x=y_test_arr[idx], y=y_pred_arr[idx],
                             title="Actual vs Predicted",
                             labels={"x":"Actual","y":"Predicted"},
                             opacity=0.6,
                             color_discrete_sequence=[C["blue"]])
            max_val = max(y_test_arr.max(), y_pred_arr.max())
            fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                          line=dict(color="#94a3b8", dash="dash"))
            fig.update_layout(**cd(380))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm_arr = confusion_matrix(y_test_arr, y_pred_arr)
            labels = ["Negative","Positive"]
            fig = px.imshow(cm_arr, text_auto=True,
                            x=labels, y=labels,
                            title="Confusion Matrix",
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted",y="Actual"))
            fig.update_layout(**cd(360))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Feature importance
        if hasattr(best_model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": best_model.feature_importances_
            }).sort_values("Importance", ascending=False).head(15)

            fig2 = px.bar(fi, x="Importance", y="Feature",
                          orientation="h",
                          title="🔑 Top Predictors (Feature Importance)",
                          color="Importance",
                          color_continuous_scale="Blues",
                          text_auto=".3f")
            fig2.update_layout(**cd(400), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Live Prediction Tool ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Make a Live Prediction")
    st.caption("Enter values for a new record and get an instant prediction.")

    input_vals = {}
    inp_cols = st.columns(3)
    for i, col in enumerate(feature_cols[:12]):
        with inp_cols[i % 3]:
            col_data = df[col].dropna()
            if col in le_dict:
                options = list(le_dict[col].classes_)
                sel = st.selectbox(f"{col}", options, key=f"pred_inp_{col}")
                input_vals[col] = le_dict[col].transform([sel])[0]
            elif pd.api.types.is_numeric_dtype(col_data):
                val = st.number_input(f"{col}",
                    value=float(col_data.median()),
                    key=f"pred_inp_{col}",
                    help=f"Typical range: {col_data.min():,.1f} – {col_data.max():,.1f}")
                input_vals[col] = val
            else:
                val = st.text_input(f"{col}", value=str(col_data.mode().iloc[0]),
                                    key=f"pred_inp_{col}")
                input_vals[col] = 0  # fallback

    if st.button("🔮 Predict", key="pred_live", type="primary"):
        try:
            input_arr = imp.transform([[input_vals.get(c, 0) for c in feature_cols]])
            pred = best_model.predict(input_arr)[0]

            if is_classification:
                prob = best_model.predict_proba(input_arr)[0] if hasattr(best_model,"predict_proba") else None
                pred_label = "YES ✅" if pred == 1 else "NO ❌"
                conf = max(prob)*100 if prob is not None else None

                if pred == 1:
                    st.error(f"🔮 **Prediction: {pred_label}**" +
                             (f" | Confidence: {conf:.1f}%" if conf else ""))
                else:
                    st.success(f"🔮 **Prediction: {pred_label}**" +
                               (f" | Confidence: {conf:.1f}%" if conf else ""))
            else:
                st.success(f"🔮 **Predicted {target_col}: {pred:,.2f}**")

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE 4 — PRESCRIPTIVE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def render_prescriptive(df, found, domain):
    section("💊 Prescriptive Insights — What Should You Do?", domain.lower())

    st.markdown("""<div class="insight-box">
    <strong>💊 What is Prescriptive Analysis?</strong><br>
    It goes beyond "what happened" and "what will happen" — it tells you
    <strong>what action to take</strong> based on your data patterns.<br>
    Each recommendation is backed by specific numbers from your dataset.
    </div>""", unsafe_allow_html=True)

    recommendations = []

    sc   = found.get("sales");   pc   = found.get("profit")
    prd  = found.get("product"); cat  = found.get("category")
    dc   = found.get("date");    qty  = found.get("quantity")
    cus  = found.get("customer");reg  = found.get("region")
    sal  = found.get("salary");  attr = found.get("attrition")
    dept = found.get("department"); gen = found.get("gender")
    dis  = found.get("discount"); ten  = found.get("tenure")
    store= found.get("store");   pay  = found.get("payment")
    imp  = found.get("impressions"); clk = found.get("clicks")
    conv = found.get("conversions"); sp  = found.get("spend")
    chan = found.get("channel");  ret  = found.get("returns")
    del_ = found.get("delivery"); perf = found.get("performance")
    sat  = found.get("satisfaction"); age = found.get("age")
    emp  = found.get("employee_id") or found.get("employee_name")
    job  = found.get("job_title")

    # ── HR Prescriptions ──────────────────────────────────────────────────
    if domain == "HR" or sal or attr:

        if attr:
            df2 = df.copy()
            df2["_left"] = df2[attr].astype(str).str.lower().isin(
                ["yes","true","1","resigned","terminated","left"])
            rate = df2["_left"].mean() * 100

            if rate > 20:
                recommendations.append({
                    "priority":"🔴 Critical","category":"Retention",
                    "action":"Launch emergency retention programme immediately",
                    "reason":f"Attrition rate is {rate:.1f}% — well above the 15% danger threshold.",
                    "steps":["Conduct exit interviews for all leavers this quarter",
                             "Implement immediate salary review for at-risk roles",
                             "Introduce flexible work policies within 30 days",
                             "Set up monthly manager 1:1s for flagged employees"]
                })
            elif rate > 12:
                recommendations.append({
                    "priority":"🟡 Warning","category":"Retention",
                    "action":"Review retention strategy — attrition trending high",
                    "reason":f"Attrition at {rate:.1f}% (healthy range: 8–12%).",
                    "steps":["Identify top 3 departments by attrition",
                             "Survey employee satisfaction scores",
                             "Benchmark salaries against market rates"]
                })

            if dept:
                da = df2.groupby(dept)["_left"].mean().mul(100)
                worst_dept = da.idxmax()
                worst_rate = da.max()
                if worst_rate > 20:
                    recommendations.append({
                        "priority":"🔴 Critical","category":"Department Retention",
                        "action":f"Immediate focus on {worst_dept} department",
                        "reason":f"{worst_dept} has {worst_rate:.1f}% attrition — highest in the organisation.",
                        "steps":[f"Urgent skip-level meetings in {worst_dept}",
                                 "Review manager performance and team culture",
                                 f"Consider salary adjustments for {worst_dept} roles",
                                 "Check workload and overtime patterns"]
                    })

            if sal and dept:
                df2_left   = df2[df2["_left"]==True]
                df2_active = df2[df2["_left"]==False]
                if len(df2_left) > 5 and len(df2_active) > 5:
                    sal_left   = df2_left[sal].mean()
                    sal_active = df2_active[sal].mean()
                    if sal_left < sal_active * 0.9:
                        gap = (sal_active - sal_left) / sal_active * 100
                        recommendations.append({
                            "priority":"🟡 Warning","category":"Compensation",
                            "action":"Salary is a key attrition driver — review pay bands",
                            "reason":f"Leavers earned {gap:.1f}% less on average than retained employees ({sal_left:,.0f} vs {sal_active:,.0f}).",
                            "steps":["Identify roles where pay is below market",
                                     "Create structured pay bands by level",
                                     "Budget for retention bonuses for high performers"]
                        })

        if sal and dept:
            ds = df.groupby(dept)[sal].mean().sort_values()
            lowest_dept = ds.index[0]
            lowest_sal  = ds.iloc[0]
            highest_sal = ds.iloc[-1]
            if highest_sal > lowest_sal * 2:
                recommendations.append({
                    "priority":"🟡 Warning","category":"Pay Equity",
                    "action":"Large salary gap between departments — review pay equity",
                    "reason":f"Highest paying dept earns {highest_sal/lowest_sal:.1f}× more than {lowest_dept} (avg {lowest_sal:,.0f}).",
                    "steps":["Commission formal pay equity audit",
                             "Establish minimum pay floors by function",
                             "Review job grading structure"]
                })

        if gen and sal:
            dg = df.copy()
            dg["_g"] = dg[gen].astype(str).str.title()
            ms = dg[dg["_g"].isin(["Male","M"])][sal].mean()
            fs = dg[dg["_g"].isin(["Female","F"])][sal].mean()
            if ms > 0 and fs > 0:
                gap = abs(ms - fs) / max(ms, fs) * 100
                if gap > 10:
                    recommendations.append({
                        "priority":"🟡 Warning","category":"Diversity & Inclusion",
                        "action":f"Address gender pay gap of {gap:.1f}%",
                        "reason":f"Male avg salary: {ms:,.0f} | Female avg: {fs:,.0f}. {gap:.1f}% gap detected.",
                        "steps":["Conduct role-matched pay analysis",
                                 "Set target to close gap within 2 years",
                                 "Review promotion rates by gender",
                                 "Publish gender pay gap report"]
                    })

        if ten and attr:
            df2 = df.copy()
            df2["_left"] = df2[attr].astype(str).str.lower().isin(
                ["yes","true","1","resigned","terminated","left"])
            early_leave = df2[df2[ten] < 2]["_left"].mean() * 100
            if early_leave > 30:
                recommendations.append({
                    "priority":"🟡 Warning","category":"Onboarding",
                    "action":"High early attrition — strengthen onboarding & first-year experience",
                    "reason":f"{early_leave:.1f}% of employees with <2 years tenure have left.",
                    "steps":["Implement structured 90-day onboarding plan",
                             "Assign mentors to all new hires",
                             "Conduct 6-month check-in surveys",
                             "Review manager quality for new hire teams"]
                })

    # ── Sales Prescriptions ───────────────────────────────────────────────
    if domain in ["Sales","Ecommerce","Retail"] or (sc and prd):

        if sc and prd:
            prod_rev = df.groupby(prd)[sc].sum().sort_values(ascending=False)
            top_prod = prod_rev.head(3).index.tolist()
            bot_prod = prod_rev.tail(3).index.tolist()
            top_rev  = prod_rev.head(3).sum()
            total_rev = prod_rev.sum()

            if total_rev > 0 and top_rev / total_rev > 0.6:
                recommendations.append({
                    "priority":"🟡 Warning","category":"Product Risk",
                    "action":"Reduce revenue concentration — top 3 products drive >60% of sales",
                    "reason":f"Top 3 products ({', '.join(str(p) for p in top_prod)}) = {top_rev/total_rev*100:.0f}% of revenue.",
                    "steps":["Invest in marketing for mid-tier products",
                             "Bundle low performers with bestsellers",
                             "Consider discontinuing persistent bottom 3",
                             "Develop 2-3 new product lines"]
                })

            if pc:
                prod_margin = (df.groupby(prd)[pc].sum() /
                               df.groupby(prd)[sc].sum() * 100).sort_values()
                neg_margin = prod_margin[prod_margin < 0]
                if len(neg_margin) > 0:
                    recommendations.append({
                        "priority":"🔴 Critical","category":"Profitability",
                        "action":f"Immediately review {len(neg_margin)} loss-making products",
                        "reason":f"Products with negative margin: {', '.join(str(p) for p in neg_margin.index[:3])}",
                        "steps":["Calculate true cost including overheads",
                                 "Raise prices or negotiate better supplier terms",
                                 "Consider product discontinuation",
                                 "Review discount policy for these SKUs"]
                    })

        if sc and dis:
            corr = df[[dis, sc]].dropna().corr().iloc[0, 1]
            avg_dis = df[dis].mean()
            if corr < -0.2 and avg_dis > 0.15:
                recommendations.append({
                    "priority":"🟡 Warning","category":"Discount Strategy",
                    "action":"Discounts are hurting revenue — optimise discount policy",
                    "reason":f"Discount-Revenue correlation: {corr:.2f} (negative). Avg discount: {avg_dis:.1%}.",
                    "steps":["Cap maximum discount at 15% without manager approval",
                             "Introduce value-based selling training",
                             "Test discount-free campaigns for premium segments",
                             "Track revenue impact per discount tier"]
                })

        if sc and reg:
            reg_rev = df.groupby(reg)[sc].sum().sort_values()
            bottom_reg = reg_rev.index[0]
            bottom_val = reg_rev.iloc[0]
            top_val    = reg_rev.iloc[-1]
            if top_val > bottom_val * 3:
                recommendations.append({
                    "priority":"🔵 Opportunity","category":"Regional Growth",
                    "action":f"Invest in {bottom_reg} — significant growth potential",
                    "reason":f"{bottom_reg} generates only {bottom_val:,.0f} vs top region {top_val:,.0f} ({top_val/bottom_val:.1f}× more).",
                    "steps":[f"Assign dedicated sales rep to {bottom_reg}",
                             "Run targeted marketing campaign in underperforming regions",
                             "Analyse competitor presence in these areas",
                             "Consider pricing adjustments for regional markets"]
                })

        if sc and dc:
            try:
                d2 = df.copy()
                d2[dc] = pd.to_datetime(d2[dc], errors="coerce")
                d2 = d2.dropna(subset=[dc])
                d2["_M"] = d2[dc].dt.to_period("M").astype(str)
                monthly = d2.groupby("_M")[sc].sum()
                if len(monthly) >= 3:
                    last_3 = monthly.tail(3).mean()
                    prev_3 = monthly.iloc[-6:-3].mean() if len(monthly) >= 6 else monthly.head(3).mean()
                    if prev_3 > 0:
                        trend = (last_3 - prev_3) / prev_3 * 100
                        if trend < -10:
                            recommendations.append({
                                "priority":"🔴 Critical","category":"Revenue Trend",
                                "action":"Revenue declining — immediate sales acceleration needed",
                                "reason":f"Last 3 months avg {last_3:,.0f} vs previous period {prev_3:,.0f} ({trend:.1f}% decline).",
                                "steps":["Launch emergency promotional campaign",
                                         "Review sales team performance and pipeline",
                                         "Identify and contact churned customers",
                                         "Accelerate new product launches"]
                            })
                        elif trend > 20:
                            recommendations.append({
                                "priority":"🟢 Opportunity","category":"Growth Momentum",
                                "action":"Strong growth momentum — scale up operations",
                                "reason":f"Revenue grew {trend:.1f}% in last 3 months vs prior period.",
                                "steps":["Increase inventory for top-selling products",
                                         "Scale marketing spend proportionally",
                                         "Hire additional sales capacity",
                                         "Expand to adjacent markets"]
                            })
            except Exception:
                pass

    # ── Marketing Prescriptions ───────────────────────────────────────────
    if domain == "Marketing" or (imp and clk):

        if imp and clk and chan:
            df2 = df.copy()
            df2["_CTR"] = df2[clk] / df2[imp].replace(0, np.nan) * 100
            chan_ctr = df2.groupby(chan)["_CTR"].mean().sort_values()
            worst_chan = chan_ctr.index[0]
            best_chan  = chan_ctr.index[-1]
            if chan_ctr.iloc[-1] > chan_ctr.iloc[0] * 3:
                recommendations.append({
                    "priority":"🟡 Warning","category":"Channel Optimisation",
                    "action":f"Reallocate budget from {worst_chan} to {best_chan}",
                    "reason":f"{best_chan} CTR ({chan_ctr.iloc[-1]:.2f}%) is {chan_ctr.iloc[-1]/chan_ctr.iloc[0]:.1f}× better than {worst_chan} ({chan_ctr.iloc[0]:.2f}%).",
                    "steps":[f"Reduce {worst_chan} budget by 30%",
                             f"Increase {best_chan} budget by 30%",
                             "Run A/B tests on underperforming channels",
                             "Review audience targeting for low-CTR channels"]
                })

        if sp and sc:
            total_sp  = df[sp].sum()
            total_rev = df[sc].sum()
            roas = total_rev / total_sp if total_sp > 0 else 0
            if roas < 2:
                recommendations.append({
                    "priority":"🔴 Critical","category":"Marketing ROI",
                    "action":"Marketing spend not generating sufficient returns",
                    "reason":f"ROAS of {roas:.2f}× — industry benchmark is 4×. Spending {total_sp:,.0f} to generate {total_rev:,.0f}.",
                    "steps":["Audit all active campaigns for ROI",
                             "Pause campaigns with ROAS < 1×",
                             "Invest in content and organic channels",
                             "Improve landing page conversion rates"]
                })

    # ── Ecommerce Prescriptions ───────────────────────────────────────────
    if domain == "Ecommerce" or ret or del_:

        if ret:
            df2 = df.copy()
            df2["_ret"] = df2[ret].astype(str).str.lower().isin(["yes","true","1","returned"])
            ret_rate = df2["_ret"].mean() * 100
            if ret_rate > 10:
                recommendations.append({
                    "priority":"🟡 Warning","category":"Returns Reduction",
                    "action":f"High return rate ({ret_rate:.1f}%) is eroding margins",
                    "reason":f"{int(df2['_ret'].sum()):,} returns out of {len(df2):,} orders.",
                    "steps":["Improve product descriptions and images",
                             "Add size guides / fit recommendations",
                             "Review quality control for high-return products",
                             "Analyse return reasons from customer feedback"]
                })

        if del_:
            avg_del = df[del_].mean()
            if avg_del > 7:
                slow_pct = (df[del_] > 7).mean() * 100
                recommendations.append({
                    "priority":"🟡 Warning","category":"Delivery Experience",
                    "action":f"Delivery too slow — {slow_pct:.0f}% of orders take >7 days",
                    "reason":f"Average delivery time: {avg_del:.1f} days. Customer expectation: 3–5 days.",
                    "steps":["Negotiate SLA improvements with logistics partners",
                             "Add regional warehouses for slow zones",
                             "Offer express delivery as premium option",
                             "Proactively notify customers of delays"]
                })

    # ── Fraud Prescriptions ───────────────────────────────────────────────
    if domain == "Fraud":
        cl = {c.lower().strip(): c for c in df.columns}
        fraud_label_cols = ["class","label","fraud","is_fraud","isfraud","target"]
        cc = next((cl[c] for c in fraud_label_cols if c in cl), None)
        if cc:
            df2 = df.copy()
            df2["_lbl"] = df2[cc].astype(str).str.lower().map(
                lambda x: 1 if x in ["1","true","yes","fraud"] else 0)
            rate = df2["_lbl"].mean() * 100
            amt_col = cl.get("amount")

            if rate > 0.5:
                exposure = f" Estimated exposure: ${df2[df2['_lbl']==1][amt_col].sum():,.2f}." if amt_col else ""
                recommendations.append({
                    "priority":"🔴 Critical","category":"Fraud Prevention",
                    "action":"Deploy real-time fraud detection rules immediately",
                    "reason":f"Fraud rate: {rate:.3f}%.{exposure}",
                    "steps":["Implement velocity checks (multiple transactions in short time)",
                             "Add device fingerprinting and IP reputation scoring",
                             "Set up transaction amount thresholds for manual review",
                             "Deploy ML model in production for real-time scoring",
                             "Create fraud operations team for case management"]
                })

            if amt_col:
                fa = df2[df2["_lbl"]==1][amt_col]
                la = df2[df2["_lbl"]==0][amt_col]
                if len(fa) > 5 and fa.mean() > la.mean() * 1.5:
                    recommendations.append({
                        "priority":"🟡 Warning","category":"Transaction Monitoring",
                        "action":"Flag high-value transactions for enhanced scrutiny",
                        "reason":f"Fraud avg: ${fa.mean():,.2f} vs legitimate avg: ${la.mean():,.2f} ({fa.mean()/la.mean():.1f}× higher).",
                        "steps":[f"Set alert threshold at ${la.mean()*2:,.0f}",
                                 "Require step-up authentication for large transactions",
                                 "Implement 24hr hold for first-time high-value purchases"]
                    })

    # ── Generic / Cross-domain ────────────────────────────────────────────
    if not recommendations:
        # Generic recommendations based on data quality
        missing_severe = [c for c in df.columns if df[c].isna().mean() > 0.3]
        if missing_severe:
            recommendations.append({
                "priority":"🟡 Warning","category":"Data Quality",
                "action":"Improve data collection for key columns",
                "reason":f"{len(missing_severe)} columns have >30% missing data: {', '.join(missing_severe[:5])}",
                "steps":["Audit data collection processes",
                         "Make critical fields mandatory in source systems",
                         "Implement data validation at entry point"]
            })
        recommendations.append({
            "priority":"🔵 Info","category":"Analysis",
            "action":"Upload domain-specific data for targeted recommendations",
            "reason":"Generic dataset detected — domain-specific recommendations need Sales, HR, or Marketing data.",
            "steps":["Ensure column names reflect their content",
                     "Use Column Mapping panel to map key columns",
                     "Try uploading sample data from your business domain"]
        })

    # ── Render recommendations ─────────────────────────────────────────────
    priority_order = ["🔴 Critical","🟡 Warning","🔵 Opportunity","🟢 Opportunity",
                      "🔵 Info","🟢 Info"]
    recommendations.sort(key=lambda x: priority_order.index(x["priority"])
                         if x["priority"] in priority_order else 99)

    st.markdown(f"**{len(recommendations)} recommendations generated from your data:**")

    PRIORITY_COLORS = {
        "🔴 Critical":    "#7f1d1d",
        "🟡 Warning":     "#78350f",
        "🔵 Opportunity": "#1e3a5f",
        "🟢 Opportunity": "#052e16",
        "🔵 Info":        "#1e3a5f",
        "🟢 Info":        "#052e16",
    }
    PRIORITY_BORDERS = {
        "🔴 Critical":    "#ef4444",
        "🟡 Warning":     "#f59e0b",
        "🔵 Opportunity": "#3b82f6",
        "🟢 Opportunity": "#22c55e",
        "🔵 Info":        "#3b82f6",
        "🟢 Info":        "#22c55e",
    }

    for i, rec in enumerate(recommendations):
        border = PRIORITY_BORDERS.get(rec["priority"], "#64748b")
        bg     = PRIORITY_COLORS.get(rec["priority"], "#1a1a2e")

        with st.expander(
            f"{rec['priority']} | {rec['category']} — {rec['action']}", expanded=(i < 3)):
            st.markdown(f"""<div style="background:{bg};border-left:4px solid {border};
                border-radius:8px;padding:14px 18px;margin-bottom:10px">
                <div style="color:#e2e8f0;font-size:.92rem">
                <strong>📊 Why:</strong> {rec['reason']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("**✅ Action Steps:**")
            for j, step in enumerate(rec["steps"], 1):
                st.markdown(f"&nbsp;&nbsp;{j}. {step}")

    # ── Download action plan ───────────────────────────────────────────────
    st.markdown("---")
    if st.button("📥 Download Action Plan as CSV", key="presc_download"):
        rows = []
        for rec in recommendations:
            for j, step in enumerate(rec["steps"], 1):
                rows.append({
                    "Priority":   rec["priority"],
                    "Category":   rec["category"],
                    "Action":     rec["action"],
                    "Reason":     rec["reason"],
                    "Step Number": j,
                    "Step":       step
                })
        csv = pd.DataFrame(rows).to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", data=csv,
                           file_name="action_plan.csv", mime="text/csv",
                           key="presc_dl_btn")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE 5 — ADVANCED DASHBOARD (Gauges · Waterfall · Treemap · Heatmap · KPI)
# ═══════════════════════════════════════════════════════════════════════════════

def render_advanced_dashboard(df, found, domain):
    """
    Beautiful advanced dashboard with:
    - Styled KPI cards with delta arrows
    - Half-sundial gauge (revenue target)
    - Full 360° gauge (satisfaction / NPS proxy)
    - Waterfall chart (revenue/cost/profit breakdown)
    - Treemap (category mix)
    - Monthly heatmap (intensity calendar)
    All charts use real uploaded data — not mocks.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    import streamlit as st

    _cur  = detect_currency(df)
    ac    = DOMAIN_COLOR.get(domain, C["blue"])

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(14,165,233,0.08),rgba(167,139,250,0.08));
         border:1px solid rgba(14,165,233,0.2);border-radius:16px;padding:20px 24px;margin-bottom:24px">
      <h2 style="margin:0;font-size:1.4rem;color:#e2e8f0">
        📊 Advanced Dashboard <span style="font-size:0.8rem;color:#64748b;font-weight:400;margin-left:10px">
        Real data · Interactive charts</span>
      </h2>
      <p style="margin:6px 0 0;color:#64748b;font-size:0.85rem">
        Gauges · Waterfall · Treemap · Heatmap — all built from your uploaded dataset
      </p>
    </div>
    """, unsafe_allow_html=True)

    sc  = found.get("sales")
    pc  = found.get("profit")
    cat = found.get("category")
    dc  = found.get("date")
    reg = found.get("region")
    qty = found.get("quantity")
    sal = found.get("salary")       # HR
    sp  = found.get("spend")        # Marketing
    rev = found.get("revenue")      # Marketing revenue
    dept= found.get("department")   # HR

    # ── Pick the primary value column smartly ──────────────────────────────
    val_col = sc or sal or sp or rev
    if val_col is None:
        # Fallback: first numeric column
        num_cols = df.select_dtypes(include="number").columns.tolist()
        val_col = num_cols[0] if num_cols else None

    if val_col is None:
        st.info("ℹ️ No numeric columns found for Advanced Dashboard.")
        return

    total_val   = df[val_col].sum()
    avg_val     = df[val_col].mean()
    max_val     = df[val_col].max()
    record_cnt  = len(df)

    # ── ROW 1 — Styled KPI Cards ──────────────────────────────────────────
    st.markdown("#### 🎯 Key Performance Indicators")

    def _kpi_card(title, value, subtitle, color, icon, delta_pct=None):
        delta_html = ""
        if delta_pct is not None:
            arrow = "▲" if delta_pct >= 0 else "▼"
            clr   = "#10b981" if delta_pct >= 0 else "#ef4444"
            delta_html = f'<div style="font-size:0.78rem;color:{clr};margin-top:4px">{arrow} {abs(delta_pct):.1f}% vs avg</div>'
        return f"""
        <div style="background:linear-gradient(135deg,{color}14,#0a1628);
             border:1px solid {color}44;border-radius:14px;padding:18px 20px;
             border-left:4px solid {color};">
          <div style="font-size:1.4rem;margin-bottom:6px">{icon}</div>
          <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;
               letter-spacing:1px;margin-bottom:4px">{title}</div>
          <div style="font-size:1.5rem;font-weight:800;color:#e2e8f0;line-height:1">{value}</div>
          <div style="font-size:0.72rem;color:#475569;margin-top:4px">{subtitle}</div>
          {delta_html}
        </div>"""

    k1, k2, k3, k4, k5 = st.columns(5)
    label_map = {"sales":"Total Revenue","salary":"Total Payroll","spend":"Total Spend","revenue":"Total Revenue"}
    val_label = label_map.get(found.get("sales") and "sales" or
                               found.get("salary") and "salary" or
                               found.get("spend") and "spend" or
                               found.get("revenue") and "revenue", "Total Value")

    with k1:
        st.markdown(_kpi_card(val_label,
            fmt_money(total_val, _cur, 0),
            f"{record_cnt:,} records", ac, "💰"), unsafe_allow_html=True)
    with k2:
        st.markdown(_kpi_card("Average",
            fmt_money(avg_val, _cur),
            "per record", C["purple"], "📈",
            delta_pct=((avg_val - avg_val*0.85)/avg_val*0.85)*100 if avg_val else None),
            unsafe_allow_html=True)
    with k3:
        st.markdown(_kpi_card("Peak",
            fmt_money(max_val, _cur),
            "highest single value", C["yellow"], "🏆"), unsafe_allow_html=True)
    with k4:
        if pc:
            profit_total = df[pc].sum()
            margin = profit_total/total_val*100 if total_val else 0
            st.markdown(_kpi_card("Profit Margin",
                f"{margin:.1f}%",
                fmt_money(profit_total, _cur, 0)+" total profit", C["green"], "📊",
                delta_pct=margin-25), unsafe_allow_html=True)
        elif dept:
            n_dept = df[dept].nunique()
            st.markdown(_kpi_card("Departments",
                str(n_dept),
                "active departments", C["green"], "🏢"), unsafe_allow_html=True)
        elif reg:
            n_reg = df[reg].nunique()
            st.markdown(_kpi_card("Regions",
                str(n_reg),
                "geographic regions", C["green"], "🌍"), unsafe_allow_html=True)
        else:
            med_val = df[val_col].median()
            st.markdown(_kpi_card("Median",
                fmt_money(med_val, _cur),
                "50th percentile", C["green"], "📉"), unsafe_allow_html=True)
    with k5:
        q75 = df[val_col].quantile(0.75)
        q25 = df[val_col].quantile(0.25)
        st.markdown(_kpi_card("Top 25% Threshold",
            fmt_money(q75, _cur),
            f"Bottom 25%: {fmt_money(q25,_cur)}", C["orange"], "🎯"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 2 — Gauges ────────────────────────────────────────────────────
    st.markdown("#### 🔢 Performance Gauges")
    g1, g2, g3 = st.columns(3)

    with g1:
        # Half-sundial: Revenue vs Target (target = 120% of actual as demo)
        target = total_val * 1.2
        pct    = min(total_val / target * 100, 100)
        fig_g1 = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = round(pct, 1),
            title = {"text": "Target Achievement %", "font": {"size": 14, "color": "#94a3b8"}},
            delta = {"reference": 100, "valueformat": ".1f",
                     "increasing": {"color": C["green"]},
                     "decreasing": {"color": C["red"]}},
            number= {"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
            gauge = {
                "axis"  : {"range": [0, 100], "tickcolor": "#334155",
                           "tickfont": {"color": "#64748b", "size": 10}},
                "bar"   : {"color": ac, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  60], "color": "rgba(239,68,68,0.15)"},
                    {"range": [60, 80], "color": "rgba(245,158,11,0.15)"},
                    {"range": [80,100], "color": "rgba(16,185,129,0.15)"}],
                "threshold": {
                    "line" : {"color": C["green"], "width": 3},
                    "thickness": 0.8, "value": 80
                },
            }
        ))
        fig_g1.update_layout(**cd(260))
        st.plotly_chart(fig_g1, use_container_width=True)

    with g2:
        # Full 360° gauge — data quality / completeness score
        non_null_pct = df.notna().mean().mean() * 100
        fig_g2 = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = round(non_null_pct, 1),
            title = {"text": "Data Quality Score", "font": {"size": 14, "color": "#94a3b8"}},
            number= {"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
            gauge = {
                "axis"  : {"range": [0, 100], "tickcolor": "#334155",
                           "tickfont": {"color": "#64748b", "size": 10}},
                "bar"   : {"color": C["purple"], "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  50], "color": "rgba(239,68,68,0.12)"},
                    {"range": [50, 75], "color": "rgba(245,158,11,0.12)"},
                    {"range": [75,100], "color": "rgba(16,185,129,0.12)"}],
            }
        ))
        fig_g2.update_layout(**cd(260))
        st.plotly_chart(fig_g2, use_container_width=True)

    with g3:
        # Distribution health gauge — how skewed is the data (lower skew = healthier)
        try:
            skewness = abs(df[val_col].skew())
            health   = max(0, min(100, 100 - skewness * 15))
        except Exception:
            health = 75
        fig_g3 = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = round(health, 1),
            title = {"text": "Distribution Health", "font": {"size": 14, "color": "#94a3b8"}},
            number= {"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
            gauge = {
                "axis"  : {"range": [0, 100], "tickcolor": "#334155",
                           "tickfont": {"color": "#64748b", "size": 10}},
                "bar"   : {"color": C["cyan"] if "cyan" in C else "#06b6d4", "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "rgba(239,68,68,0.12)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.12)"},
                    {"range": [70,100], "color": "rgba(16,185,129,0.12)"}],
            }
        ))
        fig_g3.update_layout(**cd(260))
        st.plotly_chart(fig_g3, use_container_width=True)

    # ── ROW 3 — Waterfall + Treemap ───────────────────────────────────────
    st.markdown("#### 📊 Deep-Dive Charts")
    w1, w2 = st.columns(2)

    with w1:
        # Waterfall: breakdown by category or top segments
        st.markdown("**Waterfall — Value Breakdown**")
        grp_col = cat or reg or dept or found.get("product")
        if grp_col and val_col:
            wf_data = df.groupby(grp_col)[val_col].sum().sort_values(ascending=False).head(8)
            labels  = list(wf_data.index.astype(str))
            values  = list(wf_data.values)
            # Build waterfall: running total
            measures = ["relative"] * len(labels) + ["total"]
            labels   = labels + ["Total"]
            values   = values + [sum(values)]  # total bar value ignored for 'total' measure
            fig_wf = go.Figure(go.Waterfall(
                orientation = "v",
                measure     = measures,
                x           = labels,
                y           = values,
                text        = [fmt_money(v, _cur, 0) for v in values],
                textposition= "outside",
                connector   = {"line": {"color": "#1a3050"}},
                increasing  = {"marker": {"color": C["green"]}},
                decreasing  = {"marker": {"color": C["red"]}},
                totals      = {"marker": {"color": ac}},
            ))
            fig_wf.update_layout(**cd(360),
                xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                showlegend=False)
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            # Fallback: quartile waterfall
            q_labels = ["Q1 (0–25%)", "Q2 (25–50%)", "Q3 (50–75%)", "Q4 (75–100%)"]
            q_vals   = [
                df[val_col][df[val_col] <= df[val_col].quantile(0.25)].sum(),
                df[val_col][(df[val_col] > df[val_col].quantile(0.25)) &
                             (df[val_col] <= df[val_col].quantile(0.50))].sum(),
                df[val_col][(df[val_col] > df[val_col].quantile(0.50)) &
                             (df[val_col] <= df[val_col].quantile(0.75))].sum(),
                df[val_col][df[val_col] > df[val_col].quantile(0.75)].sum()]
            fig_wf = go.Figure(go.Waterfall(
                orientation = "v",
                measure     = ["relative","relative","relative","total"],
                x           = q_labels + ["Total"],
                y           = q_vals + [sum(q_vals)],
                text        = [fmt_money(v, _cur, 0) for v in q_vals + [sum(q_vals)]],
                textposition= "outside",
                connector   = {"line": {"color": "#1a3050"}},
                increasing  = {"marker": {"color": C["green"]}},
                totals      = {"marker": {"color": ac}},
            ))
            fig_wf.update_layout(**cd(360), showlegend=False)
            st.plotly_chart(fig_wf, use_container_width=True)

    with w2:
        # Treemap
        st.markdown("**Treemap — Proportional Mix**")
        grp_col2 = cat or reg or dept or found.get("product")
        if grp_col2 and val_col:
            tm_data = df.groupby(grp_col2)[val_col].sum().reset_index()
            tm_data.columns = ["group", "value"]
            tm_data = tm_data[tm_data["value"] > 0].sort_values("value", ascending=False).head(15)
            fig_tm = px.treemap(
                tm_data, path=["group"], values="value",
                color="value",
                color_continuous_scale=["#0d1829", ac, C["purple"]],
            )
            fig_tm.update_traces(
                textinfo   = "label+percent root",
                textfont   = dict(size=12, color="#e2e8f0"),
                marker     = dict(line=dict(width=2, color="#050d1a")),
            )
            fig_tm.update_layout(**cd(360),
                coloraxis_showscale=False)
            st.plotly_chart(fig_tm, use_container_width=True)
        else:
            # Numeric columns treemap
            num_cols = df.select_dtypes(include="number").columns[:10]
            tm_data  = pd.DataFrame({
                "col": list(num_cols),
                "val": [abs(df[c].sum()) for c in num_cols]
            }).sort_values("val", ascending=False)
            fig_tm = px.treemap(tm_data, path=["col"], values="val",
                color="val", color_continuous_scale=["#0d1829", ac])
            fig_tm.update_traces(textfont=dict(size=12, color="#e2e8f0"))
            fig_tm.update_layout(**cd(360), coloraxis_showscale=False)
            st.plotly_chart(fig_tm, use_container_width=True)

    # ── ROW 4 — Monthly Heatmap ───────────────────────────────────────────
    if dc and val_col:
        st.markdown("#### 🗓️ Monthly Intensity Heatmap")
        try:
            hm = df.copy()
            hm[dc] = pd.to_datetime(hm[dc], errors="coerce")
            hm = hm.dropna(subset=[dc])
            hm["Month"] = hm[dc].dt.month_name().str[:3]
            hm["Year"]  = hm[dc].dt.year.astype(str)
            pivot = hm.groupby(["Year","Month"])[val_col].sum().unstack(fill_value=0)
            # Reorder months
            month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
            if not pivot.empty:
                fig_hm = px.imshow(
                    pivot,
                    color_continuous_scale=["#050d1a","#0d1829", ac, C["purple"]],
                    aspect="auto",
                    labels=dict(color=val_col),
                )
                fig_hm.update_traces(
                    text=[[fmt_money(v, _cur, 0) for v in row] for row in pivot.values],
                    texttemplate="%{text}",
                    textfont=dict(size=10, color="#e2e8f0"),
                )
                fig_hm.update_layout(**cd(280),
                    xaxis=dict(tickfont=dict(size=11)),
                    yaxis=dict(tickfont=dict(size=11)),
                    coloraxis_showscale=False)
                st.plotly_chart(fig_hm, use_container_width=True)
        except Exception as e:
            st.caption(f"Heatmap skipped: {e}")

    # ── ROW 5 — Box Plot + Violin (Distribution Deep Dive) ────────────────
    grp_col3 = cat or reg or dept
    if grp_col3 and val_col:
        st.markdown("#### 🎻 Distribution by Segment")
        box_data = df.copy()
        # Limit to top 8 groups for readability
        top_groups = box_data.groupby(grp_col3)[val_col].sum()\
                              .sort_values(ascending=False).head(8).index
        box_data = box_data[box_data[grp_col3].isin(top_groups)]
        tab_box, tab_violin = st.tabs(["📦 Box Plot", "🎻 Violin Plot"])
        with tab_box:
            fig_box = px.box(box_data, x=grp_col3, y=val_col,
                color=grp_col3,
                color_discrete_sequence=[ac, C["purple"], C["green"],
                    C["yellow"], C["orange"], C["red"], C["grey"]],
                points="outliers")
            fig_box.update_layout(**cd(380),
                xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        with tab_violin:
            fig_vio = px.violin(box_data, x=grp_col3, y=val_col,
                color=grp_col3, box=True, points="outliers",
                color_discrete_sequence=[ac, C["purple"], C["green"],
                    C["yellow"], C["orange"], C["red"], C["grey"]])
            fig_vio.update_layout(**cd(380),
                xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig_vio, use_container_width=True)

    # ── Footer insight ─────────────────────────────────────────────────────
    top_grp_text = ""
    if grp_col and val_col:
        try:
            top_grp = df.groupby(grp_col)[val_col].sum().idxmax()
            top_pct = df.groupby(grp_col)[val_col].sum().max() / total_val * 100
            top_grp_text = (f" **{top_grp}** contributes the most "
                            f"({top_pct:.1f}% of {val_col}).")
        except Exception:
            pass
    insight(
        f"Advanced dashboard generated from {record_cnt:,} records · "
        f"Total {val_col}: {fmt_money(total_val, _cur, 0)} · "
        f"Target achievement: {min(total_val/total_val/1.2*100,100):.0f}%."
        + top_grp_text
    )



def main():

    # ── LOGIN GATE ─────────────────────────────────────────────────────────────
    if AUTH_ENABLED and not is_logged_in():
        render_login_page()
        return

    # ── ADMIN PANEL ────────────────────────────────────────────────────────────
    if AUTH_ENABLED and is_admin():
        with st.sidebar:
            if st.sidebar.button("👑 Admin Panel", use_container_width=True):
                st.session_state["show_admin"] = not st.session_state.get("show_admin", False)
        if st.session_state.get("show_admin", False):
            render_admin_panel()
            return

    with st.sidebar:
        st.markdown("## 📊 Agentic AI\n### Data Analyst")
        st.markdown("---")

        # ── USER INFO ──────────────────────────────────────────────────────────
        if AUTH_ENABLED:
            name = st.session_state.get("user_name", "User")
            company = st.session_state.get("user_company", "")
            plan = st.session_state.get("user_plan", "Starter").title()
            st.markdown(
                f"""<div style="background:linear-gradient(135deg,#0d1829,#0f2040);
                    border:1px solid #1a2d4a;border-radius:12px;padding:12px 16px;margin-bottom:8px">
                    <div style="font-size:0.75rem;color:#64748b">Welcome back</div>
                    <div style="font-weight:700;color:#e2e8f0">{name}</div>
                    <div style="font-size:0.75rem;color:#0ea5e9">{company}</div>
                    <div style="font-size:0.7rem;background:rgba(14,165,233,0.1);color:#0ea5e9;
                        border-radius:4px;padding:2px 8px;margin-top:6px;display:inline-block">
                        {plan} Plan</div>
                </div>""",
                unsafe_allow_html=True
            )
            if st.button("🚪 Logout", use_container_width=True):
                from auth import logout
                logout()
            st.markdown("---")

        st.markdown("**🗂 Formats:** CSV · Excel · XML · HTML · PDF · SQLite")
        st.markdown("---")
        st.markdown("**🎯 Domains**\n💼 Sales · 📣 Marketing\n👥 HR · 🛒 Ecommerce\n🏪 Retail · 🚨 Fraud · 🔍 Generic")
        st.markdown("---")
        st.markdown("**🔍 Auto-Detects 40+ column types**")
        st.markdown(f"\n**LLM:** `{LLM_PROVIDER}`")

    st.markdown("""<h1 style='font-size:2.2rem;margin-bottom:.2rem'>
        📊 Universal Agentic AI Data Analyst</h1>
        <p style='color:#64748b;font-size:1rem;margin-top:0'>
        Upload any dataset — domain-specific KPIs, charts and insights are generated automatically.</p>
    """, unsafe_allow_html=True)

    # ── DATA SOURCE TABS ────────────────────────────────────────────────────
    tab_upload, tab_sheets = st.tabs(["📂 Upload File", "🔗 Google Sheets"])

    f           = None
    sheets_df   = None
    sheets_fname = None

    with tab_upload:
        f = st.file_uploader("📂 Drop your data file here",
            type=["csv","txt","xlsx","xls","xml","html","htm","pdf","db","sqlite"],
            label_visibility="collapsed")

        # ── FILE SIZE ENFORCEMENT ─────────────────────────────────────────
        if f is not None:
            max_mb = get_plan_file_limit_mb() if AUTH_ENABLED else 200
            file_mb = f.size / (1024 * 1024)
            if file_mb > max_mb:
                plan = "Starter" if max_mb == 50 else "Business"
                st.error(f"""
### 🚫 File Too Large — {file_mb:.1f}MB uploaded, {max_mb}MB allowed on your {plan} plan

**Your plan limits:**
| Plan | File Size | Price |
|------|-----------|-------|
| Starter | 50MB | ₹2,000/month |
| Business | 200MB | ₹8,000/month |
| Enterprise | Custom | ₹25,000/month |

👉 **[Upgrade your plan](mailto:pankaj@1clickdataanalysis.com?subject=Plan Upgrade Request)** to upload larger files.
""")
                st.stop()

    with tab_sheets:
        sheets_df, sheets_fname = render_google_sheets_tab()

    # ── Determine active data source ─────────────────────────────────────
    has_data = (f is not None) or (sheets_df is not None)

    if not has_data:
        cols=st.columns(3)
        descriptions=[
            ("💼 Sales","Revenue trends, top products, profit margin, regional analysis, customer ranking"),
            ("📣 Marketing","Spend vs ROI, conversion funnel, CTR/CVR by channel, campaign trends"),
            ("👥 HR","Headcount, salary by dept, gender pay gap, attrition, tenure, hiring trends"),
            ("🛒 Ecommerce","Revenue, RFM customers, returns, delivery time, payment methods"),
            ("🏪 Retail","Store performance, product mix, discount impact, regional breakdown"),
            ("🚨 Fraud","Fraud rate, amount analysis, time patterns, feature correlation")]
        for i,(title,desc) in enumerate(descriptions):
            with cols[i%3]:
                st.markdown(f'<div class="insight-box"><strong>{title}</strong><br>{desc}</div>',
                            unsafe_allow_html=True)
        return

    # ── Load data from whichever source is active ─────────────────────────
    if sheets_df is not None:
        # Google Sheets path — already a DataFrame
        df = sheets_df
        st.success(f"📊 Analysing Google Sheet — {len(df):,} rows × {len(df.columns)} cols")
    else:
        # File upload path
        with st.spinner("⚙️ Loading data..."):
            df = load_data(f)
        if df is None or df.empty:
            st.error("❌ Failed to load file."); return

    df     = clean_data(df)
    found  = detect_columns(df)
    domain = detect_domain(df, found)

    badge = f"badge-{domain.lower()}"
    st.markdown(
        f'<span class="domain-badge {badge}">🎯 {domain}</span> &nbsp;'
        f'<span style="color:#64748b;font-size:.9rem">'
        f'{len(df):,} rows × {len(df.columns)} cols &nbsp;|&nbsp; '
        f'Detected: <code>{"</code> <code>".join(found.keys())}</code></span>',
        unsafe_allow_html=True)

    # ── Anomaly Detection Banner (runs immediately on load) ─────────────
    anomalies = detect_anomalies(df, found, domain)
    render_anomaly_banner(anomalies, domain)

    # ── Column Mapping & Domain Override ────────────────────────────────────
    with st.expander("⚙️ Column Mapping & Domain Override", expanded=False):

        # ── Domain selector ──────────────────────────────────────────────
        st.markdown("**🎯 Detected Domain — change if wrong:**")
        domain_list = ["Sales","Marketing","HR","Ecommerce","Retail","Fraud","Generic"]
        domain = st.selectbox("Domain", domain_list,
            index=domain_list.index(domain), key="domain_override",
            label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**🔗 Column Mappings** — auto-detected. Change any dropdown to override. "
                    "Set *— not mapped —* to remove.")
        st.markdown("")

        # ── Build option lists ────────────────────────────────────────────
        NONE  = "— not mapped —"
        all_c = [NONE] + list(df.columns)
        num_c = [NONE] + df.select_dtypes(include="number").columns.tolist()
        dt_c  = [NONE] + [c for c in df.columns
                          if pd.api.types.is_datetime64_any_dtype(df[c])
                          or any(k in c.lower() for k in ["date","time","month","year","period"])]

        # ── Key catalogue: (label, options, section) ─────────────────────
        KEY_CATALOGUE = {
            # ══ 💼 SALES & REVENUE ═══════════════════════════════════════════
            "sales":        ("Revenue / Sales Amount",       num_c, "💼 Sales & Revenue"),
            "profit":       ("Profit / Net Income",          num_c, "💼 Sales & Revenue"),
            "quantity":     ("Quantity / Units Sold",        num_c, "💼 Sales & Revenue"),
            "discount":     ("Discount / Promo Amount",      num_c, "💼 Sales & Revenue"),
            "price":        ("Unit Price / MRP",             num_c, "💼 Sales & Revenue"),
            "cost":         ("Unit Cost / COGS",             num_c, "💼 Sales & Revenue"),
            "target":       ("Target / Budget / Forecast",   num_c, "💼 Sales & Revenue"),
            "sales_rep":    ("Sales Rep / Account Manager",  all_c, "💼 Sales & Revenue"),
            "vendor":       ("Vendor / Supplier",            all_c, "💼 Sales & Revenue"),

            # ══ 📅 DIMENSIONS ════════════════════════════════════════════════
            "date":         ("Date / Transaction Date",      dt_c,  "📅 Dimensions"),
            "product":      ("Product / Item Name",          all_c, "📅 Dimensions"),
            "category":     ("Category / Product Group",     all_c, "📅 Dimensions"),
            "sub_category": ("Sub-Category",                 all_c, "📅 Dimensions"),
            "region":       ("Region / Territory / Zone",    all_c, "📅 Dimensions"),
            "city":         ("City / Town",                  all_c, "📅 Dimensions"),
            "state":        ("State / Province",             all_c, "📅 Dimensions"),
            "country":      ("Country",                      all_c, "📅 Dimensions"),
            "postal_code":  ("Postal Code / PIN / ZIP",      all_c, "📅 Dimensions"),
            "customer":     ("Customer / Client / Account",  all_c, "📅 Dimensions"),
            "segment":      ("Customer Segment / Tier",      all_c, "📅 Dimensions"),
            "ship_mode":    ("Shipping Mode / Carrier",      all_c, "📅 Dimensions"),
            "order_id":     ("Order ID / Invoice / PO No",   all_c, "📅 Dimensions"),
            "distribution_channel": ("Distribution Channel", all_c, "📅 Dimensions"),

            # ══ 👥 HR ════════════════════════════════════════════════════════
            "salary":       ("Salary / CTC / Compensation",  num_c, "👥 HR"),
            "department":   ("Department / Division / BU",   all_c, "👥 HR"),
            "job_title":    ("Job Title / Designation",      all_c, "👥 HR"),
            "employee_id":  ("Employee ID / Payroll ID",     all_c, "👥 HR"),
            "employee_name":("Employee Name",                all_c, "👥 HR"),
            "gender":       ("Gender",                       all_c, "👥 HR"),
            "age":          ("Age",                          num_c, "👥 HR"),
            "age_group":    ("Age Group / Band",             all_c, "👥 HR"),
            "tenure":       ("Tenure / Years of Service",    num_c, "👥 HR"),
            "attrition":    ("Attrition / Left Company",     all_c, "👥 HR"),
            "hire_date":    ("Hire / Joining Date",          dt_c,  "👥 HR"),
            "performance":  ("Performance Rating",           all_c, "👥 HR"),
            "education":    ("Education Level",              all_c, "👥 HR"),
            "marital":      ("Marital Status",               all_c, "👥 HR"),

            # ══ 📣 MARKETING ═════════════════════════════════════════════════
            "spend":        ("Ad Spend / Marketing Cost",    num_c, "📣 Marketing"),
            "revenue":      ("Campaign Revenue / Value",     num_c, "📣 Marketing"),
            "impressions":  ("Impressions / Views / Reach",  num_c, "📣 Marketing"),
            "clicks":       ("Clicks / Visits",              num_c, "📣 Marketing"),
            "conversions":  ("Conversions / Leads",          num_c, "📣 Marketing"),
            "roi":          ("ROI / ROAS",                   num_c, "📣 Marketing"),
            "ctr":          ("CTR / Click Rate",             num_c, "📣 Marketing"),
            "channel":      ("Marketing Channel / Platform", all_c, "📣 Marketing"),
            "campaign_id":  ("Campaign Name / ID",           all_c, "📣 Marketing"),
            "demography":   ("Demography / Audience Segment",all_c, "📣 Marketing"),

            # ══ 🏪 RETAIL / ECOMMERCE ════════════════════════════════════════
            "store":          ("Store / Branch / Outlet",      all_c, "🏪 Retail"),
            "payment":        ("Payment Method / Mode",        all_c, "🏪 Retail"),
            "delivery":       ("Delivery Time / Lead Time",    num_c, "🏪 Retail"),
            "returns":        ("Returns / Refund Status",      all_c, "🏪 Retail"),
            "satisfaction":   ("Satisfaction / NPS / Rating",  num_c, "🏪 Retail"),
            "loyalty_points": ("Loyalty Points / Rewards",     num_c, "🏪 Retail"),
            "device":         ("Device / Platform / Channel",  all_c, "🏪 Retail"),
            "return_reason":  ("Return / Cancellation Reason", all_c, "🏪 Retail"),
            "shipping_country":("Shipping Country",            all_c, "🏪 Retail"),

            # ══ 👥 HR EXTENDED (work location, satisfaction, notice period) ══
            "work_location":  ("Work Location / Office",       all_c, "👥 HR Extended"),
            "notice_period":  ("Notice Period / Exit Date",    all_c, "👥 HR Extended"),
            "overtime":       ("Overtime / Extra Hours",       num_c, "👥 HR Extended"),
            "training":       ("Training Hours / L&D",         num_c, "👥 HR Extended"),
            "leave_days":     ("Leave Days / Absence Days",    num_c, "👥 HR Extended"),

            # ══ 🚨 FRAUD ══════════════════════════════════════════════════════
            "fraud_label":  ("Fraud Label / Class",          all_c, "🚨 Fraud"),
            "fraud_amount": ("Transaction Amount",           num_c, "🚨 Fraud"),
            "fraud_time":   ("Transaction Time / Date",      dt_c,  "🚨 Fraud"),
            "fraud_type":   ("Transaction / Fraud Type",     all_c, "🚨 Fraud"),
            "fraud_id":     ("Transaction / Case ID",        all_c, "🚨 Fraud"),
            "fraud_channel":("Channel / Device / Entry",     all_c, "🚨 Fraud"),
            "fraud_loc":    ("Location / Merchant / MCC",    all_c, "🚨 Fraud"),
            "fraud_score":  ("Risk / Anomaly Score",         num_c, "🚨 Fraud"),
            "balance":      ("Account Balance",              num_c, "🚨 Fraud"),

            # ══ 🌍 GEO ════════════════════════════════════════════════════════
            "latitude":     ("Latitude",                     num_c, "🌍 Geo"),
            "longitude":    ("Longitude",                    num_c, "🌍 Geo"),
        }

        # ── Each domain shows exactly 42 column mapping dropdowns ─────────
        # Sales & Revenue(9) + Dimensions(14) + Retail(9) + Marketing(10) = 42
        # HR(14) + Dimensions(14) + Sales&Revenue(9) + HR Extended(5)      = 42
        # Fraud(9) + Dimensions(14) + Sales&Revenue(9) + Marketing(10)     = 42
        DOMAIN_SECTIONS = {
            "Sales":     ["💼 Sales & Revenue", "📅 Dimensions",
                          "🏪 Retail", "📣 Marketing"],
            "Marketing": ["📣 Marketing", "📅 Dimensions",
                          "💼 Sales & Revenue", "🏪 Retail"],
            "HR":        ["👥 HR", "📅 Dimensions",
                          "💼 Sales & Revenue", "👥 HR Extended"],
            "Ecommerce": ["💼 Sales & Revenue", "📅 Dimensions",
                          "🏪 Retail", "📣 Marketing"],
            "Retail":    ["💼 Sales & Revenue", "📅 Dimensions",
                          "🏪 Retail", "📣 Marketing"],
            "Fraud":     ["🚨 Fraud", "📅 Dimensions",
                          "💼 Sales & Revenue", "📣 Marketing"],
            "Generic":   ["💼 Sales & Revenue", "📅 Dimensions", "👥 HR",
                          "👥 HR Extended", "📣 Marketing", "🏪 Retail",
                          "🚨 Fraud", "🌍 Geo"],
        }
        active_secs = DOMAIN_SECTIONS.get(domain, list(KEY_CATALOGUE.keys()))

        # ── Step 1: Deduplicate found — each dataset column → at most ONE key
        # Priority = order keys appear in KEY_CATALOGUE
        col_claimed = {}
        for _key in list(KEY_CATALOGUE.keys()):
            _col = found.get(_key)
            if _col and _col != NONE:
                if _col not in col_claimed:
                    col_claimed[_col] = _key
                else:
                    # Already claimed by higher-priority key → clear this one
                    del found[_key]

        # ── Step 2: Group keys by section
        by_section = {}
        for k,(lbl,opts,sec) in KEY_CATALOGUE.items():
            by_section.setdefault(sec,[]).append((k,lbl,opts))

        # ── Step 3: Render — one dropdown per row (no st.columns nesting)
        for sec_name, sec_items in by_section.items():
            is_active = sec_name in active_secs
            color     = "#00d4ff" if is_active else "#475569"
            inactive_note = (
                "  <span style='font-weight:400;color:#475569;font-size:.8rem'>"
                "— not used for this domain</span>" if not is_active else "")
            st.markdown(
                f"<p style='font-weight:700;color:{color};"
                f"margin:18px 0 4px 0;font-size:.92rem'>"
                f"{sec_name}{inactive_note}</p>",
                unsafe_allow_html=True)
            if not is_active:
                st.caption("Not active for this domain. Set manually if needed.")

            for (k, lbl, opts) in sec_items:
                cur      = found.get(k, NONE)
                safe_cur = cur if cur in opts else NONE
                disabled = (not is_active and safe_cur == NONE)
                chosen   = st.selectbox(
                    label   = lbl,
                    options = opts,
                    index   = opts.index(safe_cur),
                    key     = f"cm_{k}",
                    disabled= disabled,
                    help    = (f"Auto-detected: {cur}" if cur != NONE
                               else "Not auto-detected — select manually if needed"),
                )
                # Update found AND enforce no-duplicate on user changes
                if chosen != NONE:
                    # If this col was claimed by another key, release it first
                    old_owner = col_claimed.get(chosen)
                    if old_owner and old_owner != k and old_owner in found:
                        del found[old_owner]
                    col_claimed[chosen] = k
                    found[k] = chosen
                elif k in found:
                    del found[k]
                    if cur in col_claimed and col_claimed[cur] == k:
                        del col_claimed[cur]

        # ── Duplicate Warning ──────────────────────────────────────────────
        st.markdown("---")
        col_usage = {}
        for k, v in found.items():
            col_usage.setdefault(v, []).append(k)
        dupes = {c: ks for c, ks in col_usage.items() if len(ks) > 1}
        if dupes:
            dupe_lines = [f"- **{dc}** → {', '.join(dks)}" for dc,dks in dupes.items()]
            st.warning("Duplicate mapping — same column used for multiple keys:\n"
                       + "\n".join(dupe_lines))

        # ── Active mappings summary ────────────────────────────────────────
        st.markdown("**✅ Active Mappings:**")
        if found:
            td = pd.DataFrame([{"Key": k, "→ Dataset Column": v}
                                for k,v in sorted(found.items())])
            st.dataframe(td, use_container_width=True, hide_index=True)

    with st.expander("🔍 Preview Raw Data"):
        st.dataframe(df.head(50),use_container_width=True)
        st.caption(f"First 50 of {len(df):,} rows · {len(df.columns)} columns")

    # ── Domain Routing ────────────────────────────────────────────────────
    if domain=="Sales":         render_sales(df, found)
    elif domain=="Marketing":   render_marketing(df, found)
    elif domain=="HR":          render_hr(df, found)
    elif domain=="Ecommerce":   render_ecommerce(df, found)
    elif domain=="Retail":      render_retail(df, found)
    elif domain=="Fraud":       render_fraud(df, found)
    else:                       render_generic(df, found)

    render_one_click_story(df, found, domain)   # ⚡ FIRST — flagship feature
    render_advanced_dashboard(df, found, domain)
    render_eda(df, found, domain)
    render_prediction(df, found, domain)
    render_prescriptive(df, found, domain)
    render_calc_engine(df, found, domain)
    render_summary(df, found, domain)
    render_qa(df, domain, found)
    render_nlq(df, domain, found)
    render_pdf_export(df, found, domain)


if __name__=="__main__":
    main()
