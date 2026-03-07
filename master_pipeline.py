# ============================================================
#  master_pipeline.py  —  Universal Agentic AI Data Analyst
#  v3 — Fully Data-Driven Domain-Specific KPIs & Charts
#  Domains: Sales · Marketing · HR · Ecommerce · Retail · Fraud · Generic
# ============================================================

import os, io, re, tempfile, sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dask.dataframe as dd
import chardet

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
        from transformers import pipeline
        return pipeline("text-generation",model="sshleifer/tiny-gpt2")(prompt,max_new_tokens=200)[0]["generated_text"]

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
                with tempfile.NamedTemporaryFile(delete=False,suffix=".csv") as t:
                    t.write(raw); tp=t.name
                try:
                    df = dd.read_csv(tp,encoding=enc,assume_missing=True,dtype="object").compute()
                    for c in df.columns:
                        df[c] = pd.to_numeric(df[c],errors="ignore")
                finally:
                    os.unlink(tp)
                return df
            return pd.read_csv(io.BytesIO(raw),encoding=enc,low_memory=False)
        elif fname.endswith((".xlsx",".xls")): return pd.read_excel(io.BytesIO(raw))
        elif fname.endswith(".xml"):           return pd.read_xml(io.BytesIO(raw))
        elif fname.endswith((".html",".htm")): return pd.read_html(io.BytesIO(raw))[0]
        elif fname.endswith(".pdf"):
            import tabula
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as t:
                t.write(raw); tp=t.name
            try: dfs=tabula.read_pdf(tp,pages="all",multiple_tables=True)
            finally: os.unlink(tp)
            return dfs[0] if dfs else None
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
            try: df[col]=pd.to_datetime(df[col],errors="coerce")
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
        "date":        ["order date","order_date","transaction date","sale date","invoice date",
                        "created date","purchase date","ship date","date","period","order month","month"],
        "sales":       ["sales","revenue","total revenue","net revenue","gross revenue","total sales",
                        "net sales","gross sales","sale amount","total price","order value","order total",
                        "invoice amount","transaction amount","total amount","amount","net amount",
                        "selling price","sales amount","turnover"],
        "profit":      ["profit","net profit","gross profit","operating profit","net income",
                        "earnings","ebit","ebitda","margin amount"],
        "product":     ["product name","productname","product title","product","item name","item",
                        "sku","product code","product id","goods","merchandise","article"],
        "sub_category":["sub category","sub-category","subcategory","product sub category"],
        "category":    ["category","product category","product group","product type","brand","class","line"],
        "region":      ["region","territory","area","zone","market","geography"],
        "city":        ["city","town","municipality"],
        "state":       ["state","province","county"],
        "country":     ["country","nation","country name"],
        "quantity":    ["quantity","qty","units sold","quantity sold","quantity ordered","units",
                        "order qty","items sold","volume","no of units","number of units"],
        "customer":    ["customer name","customer id","customerid","customer","client name","client id",
                        "client","buyer","consumer","account","user"],
        "employee_id": ["employee id","emp id","employee_id","emp_id","staff id","worker id","empid"],
        "employee_name":["employee name","emp name","staff name","worker name","full name","name"],
        "salary":      ["salary","annual salary","monthly salary","base salary","wage","compensation",
                        "pay","ctc","total compensation","gross salary","net salary","income"],
        "department":  ["department","dept","division","team","function","business unit","org unit"],
        "gender":      ["gender","sex","employee gender"],
        "age":         ["age","employee age"],
        "age_group":   ["age group","age band","age bracket","age range"],
        "tenure":      ["tenure","years of service","experience","years at company","seniority",
                        "service years","employment duration","years in company"],
        "hire_date":   ["hire date","joining date","start date","date of joining","doj","date joined",
                        "employment date","onboard date"],
        "attrition":   ["attrition","left company","churn","resigned","turnover","exit",
                        "is churned","employee status","employment status","status","left"],
        "job_title":   ["job title","designation","position","role","job role","title","job level"],
        "performance": ["performance","performance rating","perf rating","rating","appraisal",
                        "performance score","review score"],
        "education":   ["education","qualification","degree","education level"],
        "marital":     ["marital status","marital","relationship status"],
        "spend":       ["ad spend","marketing spend","campaign cost","ad cost","media spend",
                        "advertising spend","spend","marketing budget"],
        "channel":     ["marketing channel","ad channel","traffic source","utm source",
                        "campaign type","acquisition channel","ad platform","media channel"],
        "distribution_channel":["channel","distribution channel","sales channel","order channel"],
        "discount":    ["discount","discount amount","discount %","discount percent",
                        "promo","rebate","coupon","markdown"],
        "returns":     ["return status","returned","return","returns","refund","refunded","is returned"],
        "delivery":    ["delivery time","shipping days","days to ship","lead time",
                        "fulfillment time","days to deliver","shipping time","transit days"],
        "payment":     ["payment method","payment type","payment mode","pay method","mode of payment"],
        "store":       ["store name","store id","store","branch","outlet","shop","point of sale","pos"],
        "satisfaction":["satisfaction score","satisfaction","nps","csat","review score",
                        "feedback score","star rating","rating score"],
        "impressions": ["impressions","impression","views","reach","page views"],
        "clicks":      ["clicks","click","click through","link clicks"],
        "conversions": ["conversions","conversion","leads","sign ups","signups","purchases","goals"],
        "roi":         ["roi","roas","return on investment","return on ad spend"],
        "price":       ["unit price","selling price","list price","mrp","price","rate","retail price"],
        "cost":        ["unit cost","cost of goods","cogs","purchase price","product cost","cost"],
        "order_id":    ["order id","order no","orderid","transaction id","invoice id","order number"],
        "ship_mode":   ["ship mode","shipping mode","delivery mode","shipment type","ship method"],
        "segment":     ["customer segment","market segment","customer type","business segment"],
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
    if "sales" not in found and num_cols:
        found["sales"] = max(num_cols, key=lambda c: df[c].sum())

    # Pass 5: date fallback
    if "date" not in found:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                found["date"]=col; break
        if "date" not in found:
            for col in df.columns:
                if any(k in norm(col) for k in ["date","time","month","year","period"]):
                    found["date"]=col; break

    return found

# ─── Fraud Check ──────────────────────────────────────────────────────────────
def is_fraud_dataset(df):
    cl = [c.lower().strip() for c in df.columns]
    fraud_cols = ["class","label","fraud","is_fraud","isfraud","target","is fraud"]
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

    if "sales"    in keys: scores["Sales"]+=3
    if "profit"   in keys: scores["Sales"]+=3
    if "product"  in keys: scores["Sales"]+=2
    if "quantity" in keys: scores["Sales"]+=2
    if "discount" in keys: scores["Sales"]+=1
    if "customer" in keys: scores["Sales"]+=1
    if any(k in h for k in ["order","invoice","sales","revenue","profit"]): scores["Sales"]+=2

    if "impressions" in keys: scores["Marketing"]+=4
    if "clicks"      in keys: scores["Marketing"]+=4
    if "conversions" in keys: scores["Marketing"]+=3
    if "channel"     in keys: scores["Marketing"]+=3
    if "spend"       in keys: scores["Marketing"]+=3
    if any(k in h for k in ["campaign","ctr","cpm","roas","utm"]): scores["Marketing"]+=3
    if "distribution_channel" in keys and "impressions" not in keys: scores["Marketing"]-=2

    if "salary"     in keys: scores["HR"]+=5
    if "attrition"  in keys: scores["HR"]+=5
    if "department" in keys: scores["HR"]+=3
    if "tenure"     in keys: scores["HR"]+=3
    if "gender"     in keys: scores["HR"]+=3
    if "age"        in keys: scores["HR"]+=2
    if "employee_id"in keys: scores["HR"]+=4
    if "job_title"  in keys: scores["HR"]+=3
    if "hire_date"  in keys: scores["HR"]+=3
    if any(k in h for k in ["employee","headcount","payroll","hire","appraisal"]): scores["HR"]+=3

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
        kpis += [("💰 Total Revenue",   f"{s.sum():,.2f}", None),
                 ("📈 Avg Order Value",  f"{s.mean():,.2f}", None),
                 ("🔝 Max Order",        f"{s.max():,.2f}", None)]
    if pc:
        p = df[pc].dropna()
        kpis.append(("🏆 Total Profit", f"{p.sum():,.2f}", None))
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
        kpis += [("💰 Total Payroll",   f"{s.sum():,.0f}", None),
                 ("📈 Avg Salary",      f"{s.mean():,.0f}", None),
                 ("🔝 Highest Salary",  f"{s.max():,.0f}", None),
                 ("🔽 Lowest Salary",   f"{s.min():,.0f}", None),
                 ("📊 Median Salary",   f"{s.median():,.0f}", None)]

    if gen:
        gc = df[gen].astype(str).str.title().value_counts()
        male   = gc.get("Male",   gc.get("M", 0))
        female = gc.get("Female", gc.get("F", 0))
        kpis += [("👨 Male Employees",   f"{int(male):,}", None),
                 ("👩 Female Employees", f"{int(female):,}", None)]
        if sal and male>0 and female>0:
            df2=df.copy(); df2["_gen"]=df2[gen].astype(str).str.title()
            m_sal=df2[df2["_gen"].isin(["Male","M"])][sal].mean()
            f_sal=df2[df2["_gen"].isin(["Female","F"])][sal].mean()
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
    dom = "Marketing"; ac = C["red"]
    sp   = found.get("spend");   rev  = found.get("sales")
    chan = found.get("channel"); imp  = found.get("impressions")
    clk  = found.get("clicks");  conv = found.get("conversions")
    roi  = found.get("roi");     dc   = found.get("date")
    cat  = found.get("category")

    section("📌 Marketing KPIs", dom)
    kpis=[]
    if sp:   kpis.append(("💸 Total Spend",      f"{df[sp].sum():,.2f}", None))
    if rev:  kpis.append(("💰 Total Revenue",    f"{df[rev].sum():,.2f}", None))
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
    dom="Ecommerce"; ac=C["green"]
    sc=found.get("sales");    prd=found.get("product"); cat=found.get("category")
    cus=found.get("customer");pc=found.get("profit");   dc=found.get("date")
    ret=found.get("returns"); pay=found.get("payment"); del_=found.get("delivery")
    dis=found.get("discount");qty=found.get("quantity");reg=found.get("region")

    section("📌 Ecommerce KPIs", dom)
    kpis=[]
    if sc:   kpis+=[("💰 Total Revenue",f"{df[sc].sum():,.2f}",None),
                    ("📈 Avg Order",f"{df[sc].mean():,.2f}",None)]
    if pc:   kpis.append(("🏆 Total Profit",f"{df[pc].sum():,.2f}",None))
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
    dom="Retail"; ac=C["yellow"]
    sc=found.get("sales"); store=found.get("store"); prd=found.get("product")
    cat=found.get("category"); pc=found.get("profit"); dis=found.get("discount")
    reg=found.get("region"); pay=found.get("payment"); qty=found.get("quantity")

    section("📌 Retail KPIs", dom)
    kpis=[]
    if sc:    kpis+=[("💰 Total Revenue",f"{df[sc].sum():,.2f}",None),
                     ("📈 Avg Transaction",f"{df[sc].mean():,.2f}",None)]
    if pc:    kpis.append(("🏆 Total Profit",f"{df[pc].sum():,.2f}",None))
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

    section("🚨 Fraud Detection KPIs", dom)
    kpis=[("📋 Total Transactions",f"{total:,}",None),
          ("✅ Legitimate",f"{legit_ct:,}",None),
          ("🚨 Fraudulent",f"{fraud_ct:,}",None),
          ("⚠️ Fraud Rate",f"{fraud_pct:.4f}%",None)]
    if amt_col:
        fa=df2[df2["_lbl"]==1][amt_col].dropna()
        la=df2[df2["_lbl"]==0][amt_col].dropna()
        if len(fa)>0: kpis+=[("💳 Avg Fraud Amt",f"${fa.mean():,.2f}",None),
                              ("💚 Avg Legit Amt",f"${la.mean():,.2f}",None),
                              ("💸 Total Fraud Exposure",f"${fa.sum():,.2f}",None)]
    show_metrics(kpis)

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
        total_emp=df[emp].nunique() if emp else len(df)
        if domain=="HR": lines.append(f"- **Total Employees:** {total_emp:,}")
        if dept and domain=="HR":
            lines.append(f"- **Departments:** {df[dept].nunique():,}")
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
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown("## 📊 Agentic AI\n### Data Analyst")
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

    f = st.file_uploader("📂 Drop your data file here",
        type=["csv","txt","xlsx","xls","xml","html","htm","pdf","db","sqlite"])

    if not f:
        cols=st.columns(3)
        descriptions=[
            ("💼 Sales","Revenue trends, top products, profit margin, regional analysis, customer ranking"),
            ("📣 Marketing","Spend vs ROI, conversion funnel, CTR/CVR by channel, campaign trends"),
            ("👥 HR","Headcount, salary by dept, gender pay gap, attrition, tenure, hiring trends"),
            ("🛒 Ecommerce","Revenue, RFM customers, returns, delivery time, payment methods"),
            ("🏪 Retail","Store performance, product mix, discount impact, regional breakdown"),
            ("🚨 Fraud","Fraud rate, amount analysis, time patterns, feature correlation"),
        ]
        for i,(title,desc) in enumerate(descriptions):
            with cols[i%3]:
                st.markdown(f'<div class="insight-box"><strong>{title}</strong><br>{desc}</div>',
                            unsafe_allow_html=True)
        return

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

    # ── Override Panel ────────────────────────────────────────────────────
    with st.expander("⚙️ Override Column Mapping & Domain (if charts missing)", expanded=False):
        all_c = ["— not mapped —"]+list(df.columns)
        num_c = ["— not mapped —"]+df.select_dtypes(include="number").columns.tolist()
        oc1,oc2=st.columns(2)
        with oc1:
            domain=st.selectbox("🎯 Domain",
                ["Sales","Marketing","HR","Ecommerce","Retail","Fraud","Generic"],
                index=["Sales","Marketing","HR","Ecommerce","Retail","Fraud","Generic"].index(domain))
        key_fields={
            "sales":("💰 Revenue/Sales col",num_c), "profit":("🏆 Profit col",num_c),
            "quantity":("📦 Quantity col",num_c),   "discount":("🏷 Discount col",num_c),
            "date":("📅 Date col",all_c),            "product":("🛒 Product col",all_c),
            "category":("🗂 Category col",all_c),    "region":("🌍 Region col",all_c),
            "customer":("👥 Customer col",all_c),    "department":("🏢 Department col",all_c),
            "salary":("💰 Salary col",num_c),        "gender":("⚧ Gender col",all_c),
            "attrition":("📉 Attrition col",all_c),  "tenure":("📅 Tenure col",num_c),
        }
        rc=st.columns(3)
        for i,(key,(lbl,opts)) in enumerate(key_fields.items()):
            cur=found.get(key,"— not mapped —")
            idx=opts.index(cur) if cur in opts else 0
            chosen=rc[i%3].selectbox(lbl,opts,index=idx,key=f"ov_{key}")
            if chosen!="— not mapped —": found[key]=chosen
            elif key in found and chosen=="— not mapped —": del found[key]
        st.markdown("---")
        st.markdown("**Detected Mappings:**")
        st.dataframe(pd.DataFrame({"Key":list(found.keys()),"Column":list(found.values())}),
                     use_container_width=True,hide_index=True)
        st.markdown("**All Columns & Types:**")
        td=pd.DataFrame({"Column":df.columns,
                          "Type":[str(df[c].dtype) for c in df.columns],
                          "Sample":[str(df[c].dropna().iloc[0]) if len(df[c].dropna())>0 else "" for c in df.columns]})
        st.dataframe(td,use_container_width=True,hide_index=True)

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

    render_summary(df, found, domain)
    render_qa(df, domain, found)


if __name__=="__main__":
    main()
