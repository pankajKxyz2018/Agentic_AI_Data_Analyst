# master_pipeline.py

# --- Imports ---
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import dask.dataframe as dd
import sqlite3
import chardet

# --- LLM Provider Wrapper ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")

def query_llm(prompt):
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif LLM_PROVIDER == "cohere":
        import cohere
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
        response = co.chat(model="command-r", message=prompt)
        return response.text

    elif LLM_PROVIDER == "huggingface":
        # FIX: replaced 7GB Mistral model with lightweight model safe for Streamlit Cloud
        from transformers import pipeline
        generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")
        response = generator(prompt, max_new_tokens=200)
        return response[0]["generated_text"]

    else:
        raise ValueError("Unsupported LLM provider")

# --- Universal Data Loader ---
def load_data(uploaded_file):
    filename = uploaded_file.name.lower()
    df = None
    try:
        if filename.endswith((".csv", ".txt")):
            raw_bytes = uploaded_file.read()
            size_mb = len(raw_bytes) / (1024 * 1024)
            uploaded_file.seek(0)
            if size_mb > 100:  # threshold for large files
                df = dd.read_csv(uploaded_file)
                df = df.compute()  # convert to Pandas for downstream functions
            else:
                result = chardet.detect(raw_bytes)
                encoding = result["encoding"] if result["encoding"] else "utf-8"
                df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith(".xml"):
            df = pd.read_xml(uploaded_file)
        elif filename.endswith((".html", ".htm")):
            tables = pd.read_html(uploaded_file)
            df = tables[0]
        elif filename.endswith(".pdf"):
            import tabula
            dfs = tabula.read_pdf(uploaded_file, pages="all", multiple_tables=True)
            df = dfs[0] if dfs else None
        elif filename.endswith((".db", ".sqlite")):
            conn = sqlite3.connect(uploaded_file.name)
            df = pd.read_sql_query("SELECT * FROM my_table", conn)
            conn.close()
    except Exception as e:
        st.error(f"Error loading file {filename}: {e}")
    return df

# --- Data Cleaning ---
def clean_data(df):
    df = df.drop_duplicates()
    # FIX: replaced deprecated fillna(method=...) with ffill()/bfill() for pandas 2.x
    df = df.ffill().bfill()
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df

# --- Domain Detection ---
def detect_domain(df):
    headers = " ".join(df.columns).lower()
    if "employee" in headers or "salary" in headers:
        return "HR"
    elif any(x in headers for x in ["sales", "price", "order", "revenue"]):
        return "Sales / Retail / Ecommerce"
    elif "campaign" in headers or "clicks" in headers or "impressions" in headers:
        return "Marketing"
    else:
        return "Generic"

# --- Domain-specific Aggregations ---
def sales_aggregations(df):
    report = "Executive Summary: Sales Performance\n\n"
    if "Date" in df.columns and "Sales" in df.columns:
        df["Year"] = pd.to_datetime(df["Date"]).dt.year
        df["Quarter"] = pd.to_datetime(df["Date"]).dt.to_period("Q")
        df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M")

        total_sales = df["Sales"].sum()
        yearly = df.groupby("Year")["Sales"].sum()
        quarterly = df.groupby("Quarter")["Sales"].sum()

        report += f"- Total Sales: {total_sales:.2f}\n"
        report += f"- Yearly Breakdown: {yearly.to_dict()}\n"
        report += f"- Quarterly Breakdown: {quarterly.to_dict()}\n"

        if len(yearly) > 1:
            growth = (yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100
            report += f"- YoY Growth: {growth:.2f}%\n"

        report += "\nPrescriptive Insight: Focus on sustaining growth in high-performing quarters while addressing dips in weaker months.\n"
    return report

def marketing_aggregations(df):
    report = "Executive Summary: Marketing Performance\n\n"
    if "Spend" in df.columns and "Revenue" in df.columns:
        total_spend = df["Spend"].sum()
        total_revenue = df["Revenue"].sum()
        roi = (total_revenue - total_spend) / total_spend * 100

        report += f"- Total Spend: {total_spend:.2f}\n"
        report += f"- Total Revenue: {total_revenue:.2f}\n"
        report += f"- Overall ROI: {roi:.2f}%\n"

        if "Date" in df.columns:
            df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M")
            monthly_roi = df.groupby("Month").apply(
                lambda x: (x["Revenue"].sum() - x["Spend"].sum()) / x["Spend"].sum() * 100
            )
            report += f"- Monthly ROI Trend: {monthly_roi.to_dict()}\n"

        report += "\nPrescriptive Insight: Campaigns with ROI below 10% should be re-evaluated; allocate more budget to high-ROI channels.\n"
    return report

def hr_aggregations(df):
    report = "Executive Summary: HR Performance\n\n"
    if "Salary" in df.columns:
        avg_salary = df["Salary"].mean()
        report += f"- Average Salary: {avg_salary:.2f}\n"
    if "Promotions" in df.columns:
        total_promotions = df["Promotions"].sum()
        report += f"- Total Promotions: {total_promotions}\n"
    if "TimeAssociated" in df.columns:
        avg_tenure = df["TimeAssociated"].mean()
        report += f"- Average Tenure: {avg_tenure:.2f} years\n"

    report += "\nPrescriptive Insight: Monitor attrition closely; invest in retention programs if promotions are low relative to workforce size.\n"
    return report

# --- Storytelling ---
def boardroom_story(df, domain):
    if domain == "Sales / Retail / Ecommerce":
        return sales_aggregations(df)
    elif domain == "Marketing":
        return marketing_aggregations(df)
    elif domain == "HR":
        return hr_aggregations(df)
    else:
        return "Generic dataset. Showing distributions and counts."

# --- Visualizations ---
def plot_charts(df, domain):
    if domain == "Sales / Retail / Ecommerce" and "Date" in df.columns and "Sales" in df.columns:
        df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)
        monthly = df.groupby("Month")["Sales"].sum().reset_index()
        st.plotly_chart(px.line(monthly, x="Month", y="Sales", title="Monthly Sales Trend"))
    elif domain == "Marketing" and "Spend" in df.columns and "Revenue" in df.columns:
        st.plotly_chart(px.scatter(df, x="Spend", y="Revenue", title="Spend vs Revenue"))
    else:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                st.plotly_chart(px.histogram(df, x=col, nbins=50, title=f"{col} Distribution"))
            else:
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "count"]
                st.plotly_chart(px.bar(counts, x=col, y="count", title=f"{col} Counts"))

# --- Basic LLM Q&A ---
def generate_numeric_questions(df):
    schema = ", ".join(df.columns)
    sample = df.head(3).to_dict()
    prompt = f"""
You are a corporate data analyst. The dataset has columns: {schema}.
Sample rows: {sample}
Generate exactly 5 numeric questions (averages, totals, YoY comparisons) that a boardroom executive would ask.
Return only the questions as a numbered list.
"""
    try:
        return query_llm(prompt)
    except Exception as e:
        return f"LLM Q&A unavailable: {e}"

def answer_questions(df, questions_text):
    results = []
    lines = [l.strip() for l in questions_text.split("\n") if l.strip() and l[0].isdigit()]
    for q in lines:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        context = df[numeric_cols].describe().to_string() if numeric_cols else "No numeric columns."
        prompt = f"""
You are a data analyst. Answer this question using the dataset statistics below.
Question: {q}
Dataset stats:
{context}
Provide a concise 1-2 sentence answer with numbers.
"""
        try:
            answer = query_llm(prompt)
        except Exception as e:
            answer = f"Could not answer: {e}"
        results.append((q, answer))
    return results

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Agentic AI Data Analyst", layout="wide")
    st.title("🤖 Universal Agentic AI Data Analyst")
    st.markdown("Upload your dataset and get boardroom-ready insights, visualizations, and AI-powered Q&A.")

    uploaded_file = st.file_uploader(
        "Upload your data file (CSV, Excel, XML, HTML, PDF, SQLite)",
        type=["csv", "txt", "xlsx", "xls", "xml", "html", "htm", "pdf", "db", "sqlite"]
    )

    if uploaded_file:
        with st.spinner("Loading and cleaning data..."):
            df = load_data(uploaded_file)

        if df is not None:
            df = clean_data(df)
            domain = detect_domain(df)

            st.success(f"✅ Dataset loaded — **{len(df):,} rows × {len(df.columns)} columns** | Domain: **{domain}**")

            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(20))

            st.subheader("📊 Boardroom Story")
            story = boardroom_story(df, domain)
            st.text(story)

            st.subheader("📈 Visualizations")
            plot_charts(df, domain)

            st.subheader("🧠 AI-Powered Q&A")
            if st.button("Generate & Answer 5 Boardroom Questions"):
                with st.spinner("Thinking..."):
                    questions = generate_numeric_questions(df)
                    st.markdown("**Generated Questions & Answers:**")
                    answers = answer_questions(df, questions)
                    for i, (q, a) in enumerate(answers, 1):
                        st.markdown(f"**Q{i}: {q}**")
                        st.write(a)
        else:
            st.error("Failed to load the file. Please check the format and try again.")

if __name__ == "__main__":
    main()
