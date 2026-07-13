
import io
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(
    page_title="Shareholder Letter Intelligence | Management Narrative Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDUSTRIES = {
    "Technology": ["AI & automation", "Cloud", "Cybersecurity", "R&D", "Recurring revenue", "Talent"],
    "Banking & Financial Services": ["Net interest margin", "Credit quality", "Capital adequacy", "Digital banking", "Cost efficiency", "Regulation"],
    "Consumer & Retail": ["Pricing", "Volume growth", "Brand investment", "Store expansion", "E-commerce", "Supply chain"],
    "Healthcare & Pharmaceuticals": ["Pipeline", "Clinical trials", "Regulation", "Patent exposure", "Market access", "R&D"],
    "Energy": ["Production growth", "Commodity prices", "Capital discipline", "Energy transition", "Reserves", "Safety"],
    "Industrials & Manufacturing": ["Backlog", "Capacity", "Automation", "Input costs", "Supply chain", "Aftermarket"],
    "Telecommunications": ["Subscribers", "ARPU", "Network investment", "Churn", "Spectrum", "Convergence"],
    "Real Estate": ["Occupancy", "Rental growth", "Development pipeline", "Interest rates", "Asset sales", "Leverage"],
}

TONE_LEXICONS = {
    "Optimism": ["growth", "opportunity", "confident", "strong", "momentum", "excited", "record", "leading", "improve", "resilient"],
    "Caution": ["uncertain", "challenging", "risk", "volatile", "pressure", "headwind", "cautious", "difficult", "constraint", "slowdown"],
    "Accountability": ["we acknowledge", "we learned", "our mistake", "we fell short", "we did not", "we must improve", "responsibility", "corrective"],
    "Specificity": ["%", "$", "million", "billion", "target", "by 20", "basis points", "margin", "cagr", "deadline"],
    "Long-term orientation": ["long term", "long-term", "multi-year", "sustainable value", "durable", "over time", "future", "decade", "compounding"],
}

STRATEGY_KEYWORDS = {
    "Growth": ["growth", "expand", "market share", "new market", "customer acquisition"],
    "Margin expansion": ["margin", "productivity", "efficiency", "cost reduction", "operating leverage"],
    "Innovation": ["innovation", "research", "development", "new product", "platform"],
    "Digital transformation": ["digital", "automation", "artificial intelligence", "ai", "cloud", "data"],
    "Capital allocation": ["capital allocation", "dividend", "buyback", "repurchase", "acquisition", "debt reduction"],
    "Customer": ["customer experience", "retention", "loyalty", "customer satisfaction", "service"],
    "People": ["employees", "talent", "culture", "workforce", "leadership"],
    "Sustainability": ["sustainability", "climate", "emissions", "renewable", "esg"],
    "Risk & resilience": ["risk management", "resilience", "cybersecurity", "supply chain", "compliance"],
}

COMMITMENT_PATTERNS = [
    r"we (?:expect|aim|plan|intend|target|will|seek) to[^.]{15,220}\.",
    r"our target is[^.]{10,220}\.",
    r"by 20\d{2}[^.]{10,220}\.",
]

CSS = """
<style>
:root {
    --ice: #f5faff;
    --sky: #dceeff;
    --blue: #3478c9;
    --deep: #173b67;
    --ink: #16304d;
    --soft: #eaf4ff;
}
.stApp {
    background: linear-gradient(135deg, #ffffff 0%, #f5faff 48%, #eaf4ff 100%);
    color: var(--ink);
}
.block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
h1, h2, h3 {color: var(--deep); letter-spacing: -0.02em;}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #edf7ff 0%, #ffffff 100%);
    border-right: 1px solid #d7e9fb;
}
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 24px;
    background: rgba(255,255,255,.86);
    border: 1px solid #d9ebfb;
    box-shadow: 0 14px 36px rgba(52,120,201,.10);
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(3rem, 5vw, 4.4rem);}
.hero-sub {font-size: 1.03rem; color: #52708f; max-width: 850px;}
.kpi {
    background: rgba(255,255,255,.9);
    border: 1px solid #dcecfb;
    border-radius: 20px;
    padding: 2.5rem 2.8rem;
    box-shadow: 0 8px 24px rgba(44,101,160,.08);
    min-height: 118px;
}
.kpi-label {color:#6c86a1; font-size:.82rem; font-weight:700; text-transform:uppercase;}
.kpi-value {color:#173b67; font-size:1.65rem; font-weight:800; margin-top:.2rem;}
.kpi-note {color:#6d839a; font-size:.82rem;}
.insight {
    padding: 1rem 1.1rem;
    background: #ffffff;
    border: 1px solid #dcecfb;
    border-left: 5px solid #6ba7df;
    border-radius: 16px;
    margin: .55rem 0;
}
.highlight-positive {background:#dff4ff; border-radius:5px; padding:1px 3px;}
.highlight-risk {background:#fff0d9; border-radius:5px; padding:1px 3px;}
.highlight-commitment {background:#e8e1ff; border-radius:5px; padding:1px 3px;}
.note-chip {
    display:inline-block; margin:.2rem .3rem .2rem 0; padding:.28rem .58rem;
    border-radius:999px; background:#e7f3ff; color:#2a5e91; font-size:.78rem;
}
.small-muted {font-size:.83rem;color:#6f879f;}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.88);
    border: 1px solid #dcecfb;
    padding: .7rem;
    border-radius: 16px;
}
.stButton>button, .stDownloadButton>button {
    border-radius: 12px;
    border: 1px solid #9dc6eb;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@dataclass
class Letter:
    company: str
    year: int
    industry: str
    text: str
    filename: str


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        pages.append(f"\n[PAGE {idx}]\n{txt}")
    return "\n".join(pages)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text(text)) if len(s.strip()) > 25]


def heuristic_score(text: str, terms: List[str]) -> int:
    low = text.lower()
    count = sum(low.count(term) for term in terms)
    words = max(len(low.split()), 1)
    density = count / words * 1000
    return int(np.clip(35 + density * 8, 5, 95))


def sentence_evidence(text: str, terms: List[str], n: int = 3) -> List[str]:
    scored = []
    for sentence in split_sentences(text):
        score = sum(sentence.lower().count(t) for t in terms)
        if score:
            scored.append((score, sentence))
    return [s for _, s in sorted(scored, reverse=True)[:n]]


def extract_commitments(text: str, n: int = 8) -> List[str]:
    found = []
    clean = clean_text(text)
    for pattern in COMMITMENT_PATTERNS:
        found.extend(re.findall(pattern, clean, flags=re.I))
    unique = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique[:n]


def analyze_heuristic(letter: Letter) -> Dict:
    text = letter.text
    tones = {}
    tone_evidence = {}
    for name, terms in TONE_LEXICONS.items():
        tones[name] = heuristic_score(text, terms)
        tone_evidence[name] = sentence_evidence(text, terms)

    priorities = []
    for theme, terms in STRATEGY_KEYWORDS.items():
        score = heuristic_score(text, terms)
        evidence = sentence_evidence(text, terms, 2)
        priorities.append({"theme": theme, "score": score, "evidence": evidence})
    priorities = sorted(priorities, key=lambda x: x["score"], reverse=True)[:6]

    sentences = split_sentences(text)
    risks = sentence_evidence(text, TONE_LEXICONS["Caution"], 5)
    commitments = extract_commitments(text)

    return {
        "company": letter.company,
        "year": letter.year,
        "industry": letter.industry,
        "tone_scores": tones,
        "tone_evidence": tone_evidence,
        "strategic_priorities": priorities,
        "commitments": commitments,
        "risks": risks,
        "summary": " ".join(sentences[:4]) if sentences else "No readable text found.",
        "mode": "Heuristic demo",
    }


def analyze_with_openai(letter: Letter, api_key: str, model: str) -> Dict:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    client = OpenAI(api_key=api_key)
    trimmed = letter.text[:110000]
    schema_instruction = """
Return valid JSON only with this structure:
{
  "summary": "string",
  "tone_scores": {
    "Optimism": 0-100,
    "Caution": 0-100,
    "Accountability": 0-100,
    "Specificity": 0-100,
    "Long-term orientation": 0-100
  },
  "tone_evidence": {"Optimism":["quote"],"Caution":["quote"],"Accountability":["quote"],"Specificity":["quote"],"Long-term orientation":["quote"]},
  "strategic_priorities":[{"theme":"string","score":0-100,"evidence":["quote"]}],
  "commitments":["verbatim or close paraphrase"],
  "risks":["string"]
}
Use evidence grounded only in the supplied letter. Scores must be internally consistent.
"""
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a rigorous equity research analyst. Never invent facts."},
            {"role": "user", "content": f"{schema_instruction}\n\nCompany: {letter.company}\nYear: {letter.year}\nIndustry: {letter.industry}\n\nLETTER:\n{trimmed}"}
        ],
        text={"format": {"type": "json_object"}},
    )
    result = json.loads(response.output_text)
    result.update({
        "company": letter.company,
        "year": letter.year,
        "industry": letter.industry,
        "mode": "AI analysis",
    })
    return result


def highlighted_letter(text: str, analysis: Dict) -> str:
    excerpts = []
    for category, items in analysis.get("tone_evidence", {}).items():
        for item in items[:2]:
            excerpts.append((item, "positive" if category in ["Optimism", "Long-term orientation"] else "risk", category))
    for item in analysis.get("commitments", [])[:5]:
        excerpts.append((item, "commitment", "Commitment"))

    shown = text[:18000]
    for excerpt, cls, label in sorted(excerpts, key=lambda x: len(x[0]), reverse=True):
        if excerpt and excerpt in shown:
            shown = shown.replace(
                excerpt,
                f'<span class="highlight-{cls}" title="{label}">{excerpt}</span>',
                1,
            )
    shown = shown.replace("[PAGE ", "<br><br><b>Page ").replace("]", "</b><br>")
    return shown


def tone_radar(analyses: List[Dict]):
    categories = list(TONE_LEXICONS.keys())
    fig = go.Figure()
    for a in analyses:
        values = [a["tone_scores"].get(c, 0) for c in categories]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=f'{a["company"]} {a["year"]}',
            opacity=.65,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        margin=dict(l=40,r=40,t=50,b=40),
        height=470,
        title="Management Tone Fingerprint",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-.12),
    )
    return fig


def strategy_constellation(analyses: List[Dict]):
    rows = []
    for a in analyses:
        for p in a["strategic_priorities"]:
            rows.append({
                "company_year": f'{a["company"]} {a["year"]}',
                "theme": p["theme"],
                "importance": p["score"],
                "specificity": a["tone_scores"].get("Specificity", 50),
                "long_term": a["tone_scores"].get("Long-term orientation", 50),
            })
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="specificity",
        y="long_term",
        size="importance",
        color="company_year",
        hover_name="theme",
        hover_data={"importance": True, "specificity": True, "long_term": True},
        size_max=42,
        labels={
            "specificity": "Strategic specificity",
            "long_term": "Long-term orientation",
            "company_year": "Company / year",
        },
        title="Strategy Constellation",
    )
    fig.add_vline(x=50, line_dash="dot", opacity=.3)
    fig.add_hline(y=50, line_dash="dot", opacity=.3)
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.65)")
    return fig


def priority_ribbon(analyses: List[Dict]):
    rows = []
    for a in analyses:
        for p in a["strategic_priorities"]:
            rows.append({"company_year": f'{a["company"]} {a["year"]}', "theme": p["theme"], "score": p["score"]})
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="theme", columns="company_year", values="score", aggfunc="mean", fill_value=0)
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale=["#f6fbff", "#b9dcf7", "#4f91cf", "#173b67"],
        title="Strategic Emphasis Map",
        labels=dict(color="Importance"),
    )
    fig.update_layout(height=440, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def recommendation(analyses: List[Dict]) -> str:
    if len(analyses) < 2:
        a = analyses[0]
        strongest = max(a["tone_scores"], key=a["tone_scores"].get)
        weakest = min(a["tone_scores"], key=a["tone_scores"].get)
        return (
            f"Management communication is strongest in **{strongest.lower()}** "
            f"but weakest in **{weakest.lower()}**. Treat the current narrative as more credible "
            f"when strategic claims are accompanied by quantified targets and later financial verification."
        )

    ranked = sorted(
        analyses,
        key=lambda a: (
            a["tone_scores"].get("Accountability", 0)
            + a["tone_scores"].get("Specificity", 0)
            + a["tone_scores"].get("Long-term orientation", 0)
        ) / 3,
        reverse=True,
    )
    leader, laggard = ranked[0], ranked[-1]
    return (
        f"**{leader['company']} {leader['year']}** presents the strongest evidence-backed management narrative, "
        f"driven by higher accountability, specificity, and long-term orientation. "
        f"**{laggard['company']} {laggard['year']}** should be treated more cautiously: verify its major claims "
        f"against subsequent margins, free cash flow, ROIC, leverage, and shareholder returns before drawing an investment conclusion."
    )


def performance_chart(perf_df: pd.DataFrame, metric: str):
    fig = px.line(
        perf_df,
        x="Year",
        y=metric,
        color="Company",
        markers=True,
        title=f"Long-term {metric}",
    )
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.65)")
    return fig

st.html("""
<div class="hero">
    <div class="eyebrow">AI-ASSISTED EQUITY RESEARCH</div>

    <div class="hero-title">
        Shareholder Letter Intelligence
    </div>

    <div class="hero-headline">
        Turn management narratives into structured investment insight.
    </div>

    <div class="hero-sub">
        Upload shareholder letters to evaluate management tone, strategic priorities,
        commitments and long-term credibility — with every conclusion grounded in
        the source document.
    </div>

    <div class="hero-features">
        Management tone • Strategic priorities • Evidence-backed analysis
    </div>

    </div>
</div>
""")

with st.sidebar:
    st.header("Analysis workspace")
    industry = st.selectbox("Industry", list(INDUSTRIES.keys()))
    st.caption("Industry vocabulary helps contextualise strategic priorities.")
    st.markdown("**Industry lenses**")
    st.markdown(" ".join([f'<span class="note-chip">{x}</span>' for x in INDUSTRIES[industry]]), unsafe_allow_html=True)

    st.divider()
    mode = st.radio("Analysis mode", ["Demo / heuristic", "OpenAI API"])
    api_key = ""
    model = "gpt-4.1-mini"
    if mode == "OpenAI API":
        api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model = st.text_input("Model", value="gpt-4.1-mini")
    st.caption("The demo mode is useful for interface testing. AI mode provides materially better interpretation.")

tabs = st.tabs(["Upload & Analyse", "Visual Comparison", "Annotated Letter", "Long-term Performance", "Methodology"])

with tabs[0]:
    st.subheader("Upload shareholder letters")
    col1, col2, col3 = st.columns([1.5,1,1])
    with col1:
        company = st.text_input("Company name", placeholder="e.g., Microsoft")
    with col2:
        year = st.number_input("Fiscal year", min_value=1990, max_value=2100, value=2025, step=1)
    with col3:
        uploaded = st.file_uploader("Annual letter PDF", type=["pdf"], accept_multiple_files=False)

    if "letters" not in st.session_state:
        st.session_state.letters = []
    if "analyses" not in st.session_state:
        st.session_state.analyses = []

    if st.button("Add and analyse letter", type="primary", use_container_width=True):
        if not company or uploaded is None:
            st.error("Add a company name and PDF.")
        elif mode == "OpenAI API" and not api_key:
            st.error("Add an OpenAI API key or switch to demo mode.")
        else:
            try:
                raw_text = extract_pdf_text(uploaded)
                letter = Letter(company, int(year), industry, raw_text, uploaded.name)
                with st.spinner("Analysing management narrative..."):
                    result = analyze_with_openai(letter, api_key, model) if mode == "OpenAI API" else analyze_heuristic(letter)
                st.session_state.letters.append(letter)
                st.session_state.analyses.append(result)
                st.success(f"Analysed {company} {year}.")
            except Exception as exc:
                st.exception(exc)

    if st.session_state.analyses:
        latest = st.session_state.analyses[-1]
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Optimism", latest["tone_scores"].get("Optimism", 0), "Future confidence"),
            ("Accountability", latest["tone_scores"].get("Accountability", 0), "Ownership of outcomes"),
            ("Specificity", latest["tone_scores"].get("Specificity", 0), "Measurable detail"),
            ("Long-term", latest["tone_scores"].get("Long-term orientation", 0), "Durable value focus"),
        ]
        for col, (label, value, note) in zip([c1,c2,c3,c4], metrics):
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}/100</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

        st.markdown("### Executive interpretation")
        st.markdown(f'<div class="insight">{latest["summary"]}</div>', unsafe_allow_html=True)

        st.markdown("### Strategic priorities")
        pcols = st.columns(3)
        for i, p in enumerate(latest["strategic_priorities"][:6]):
            with pcols[i % 3]:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">Priority {i+1}</div>'
                    f'<div class="kpi-value" style="font-size:1.1rem">{p["theme"]}</div>'
                    f'<div class="kpi-note">Importance {p["score"]}/100</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("### Commitments detected")
        if latest["commitments"]:
            for item in latest["commitments"]:
                st.markdown(f'<div class="insight">{item}</div>', unsafe_allow_html=True)
        else:
            st.info("No explicit commitments were detected in the current analysis.")

with tabs[1]:
    if not st.session_state.get("analyses"):
        st.info("Upload at least one letter first. Two or more letters unlock meaningful comparison.")
    else:
        analyses = st.session_state.analyses
        st.plotly_chart(tone_radar(analyses), use_container_width=True)
        st.plotly_chart(strategy_constellation(analyses), use_container_width=True)
        if len(analyses) >= 2:
            st.plotly_chart(priority_ribbon(analyses), use_container_width=True)

        st.markdown("### Investment-research outcome")
        st.markdown(f'<div class="insight">{recommendation(analyses)}</div>', unsafe_allow_html=True)

        export = pd.DataFrame([
            {
                "Company": a["company"],
                "Year": a["year"],
                **a["tone_scores"],
                "Top priority": a["strategic_priorities"][0]["theme"] if a["strategic_priorities"] else "",
                "Commitments detected": len(a["commitments"]),
            }
            for a in analyses
        ])
        st.download_button(
            "Download comparison data",
            export.to_csv(index=False).encode("utf-8"),
            "shareholder_letter_comparison.csv",
            "text/csv",
        )

with tabs[2]:
    if not st.session_state.get("letters"):
        st.info("Upload a letter to create the annotated reading view.")
    else:
        choices = [f"{l.company} {l.year}" for l in st.session_state.letters]
        selected = st.selectbox("Letter", range(len(choices)), format_func=lambda i: choices[i])
        letter = st.session_state.letters[selected]
        analysis = st.session_state.analyses[selected]

        st.markdown(
            '<span class="highlight-positive">Opportunity / long-term evidence</span> '
            '<span class="highlight-risk">Risk / caution evidence</span> '
            '<span class="highlight-commitment">Management commitment</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="hero" style="max-height:680px;overflow:auto;line-height:1.75">{highlighted_letter(letter.text, analysis)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("The MVP highlights exact matched passages. A production version should use page-level coordinates for PDF overlays.")

with tabs[3]:
    st.subheader("Connect narrative to actual results")
    st.write("Paste or edit annual financial data. The app will compare management messaging with long-term performance.")

    default_df = pd.DataFrame({
        "Company": ["Example Co"] * 5,
        "Year": [2021, 2022, 2023, 2024, 2025],
        "Revenue": [100, 108, 116, 121, 130],
        "Operating Margin": [15.0, 15.8, 14.9, 13.7, 14.2],
        "Free Cash Flow": [12, 13, 12, 10, 13],
        "ROIC": [11.0, 12.0, 10.5, 9.7, 10.2],
        "Net Debt / EBITDA": [2.1, 1.9, 2.3, 2.6, 2.2],
        "TSR": [8.0, -4.0, 12.0, 5.0, 14.0],
    })
    perf_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)
    numeric_metrics = [c for c in perf_df.columns if c not in ["Company", "Year"]]
    metric = st.selectbox("Performance metric", numeric_metrics)
    st.plotly_chart(performance_chart(perf_df, metric), use_container_width=True)

    if len(perf_df) >= 2:
        first, last = perf_df.iloc[0], perf_df.iloc[-1]
        try:
            change = float(last[metric]) - float(first[metric])
            direction = "improved" if change > 0 else "declined"
            st.markdown(
                f'<div class="insight"><b>Performance outcome:</b> {metric} {direction} by '
                f'{abs(change):.2f} between {int(first["Year"])} and {int(last["Year"])}. '
                f'Compare this direction with changes in optimism and specificity before treating management language as predictive.</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

with tabs[4]:
    st.subheader("How the MVP works")
    st.markdown("""
1. **PDF extraction:** text is retained with page markers.
2. **Tone analysis:** five management-communication dimensions are scored.
3. **Strategic-priority analysis:** repeated themes are ranked by emphasis.
4. **Commitment extraction:** forward-looking promises and targets are identified.
5. **Comparison:** radar, constellation and emphasis-map views replace a primitive peer table.
6. **Evidence view:** supporting passages are highlighted in the letter.
7. **Performance layer:** financial trends are compared with management narrative.

**Important:** Demo mode uses transparent keyword heuristics and is intended for interface testing. Use AI mode for semantic interpretation, then manually validate material investment conclusions.
""")

TONE_LEXICONS = {
    "Optimism": ["growth", "opportunity", "confident", "strong", "momentum", "excited", "record", "leading", "improve", "resilient"],
    "Caution": ["uncertain", "challenging", "risk", "volatile", "pressure", "headwind", "cautious", "difficult", "constraint", "slowdown"],
    "Accountability": ["we acknowledge", "we learned", "our mistake", "we fell short", "we did not", "we must improve", "responsibility", "corrective"],
    "Specificity": ["%", "$", "million", "billion", "target", "by 20", "basis points", "margin", "cagr", "deadline"],
    "Long-term orientation": ["long term", "long-term", "multi-year", "sustainable value", "durable", "over time", "future", "decade", "compounding"],
}

STRATEGY_KEYWORDS = {
    "Growth": ["growth", "expand", "market share", "new market", "customer acquisition"],
    "Margin expansion": ["margin", "productivity", "efficiency", "cost reduction", "operating leverage"],
    "Innovation": ["innovation", "research", "development", "new product", "platform"],
    "Digital transformation": ["digital", "automation", "artificial intelligence", "ai", "cloud", "data"],
    "Capital allocation": ["capital allocation", "dividend", "buyback", "repurchase", "acquisition", "debt reduction"],
    "Customer": ["customer experience", "retention", "loyalty", "customer satisfaction", "service"],
    "People": ["employees", "talent", "culture", "workforce", "leadership"],
    "Sustainability": ["sustainability", "climate", "emissions", "renewable", "esg"],
    "Risk & resilience": ["risk management", "resilience", "cybersecurity", "supply chain", "compliance"],
}

COMMITMENT_PATTERNS = [
    r"we (?:expect|aim|plan|intend|target|will|seek) to[^.]{15,220}\.",
    r"our target is[^.]{10,220}\.",
    r"by 20\d{2}[^.]{10,220}\.",
]

CSS = """
<style>
:root {
    --ice: #f5faff;
    --sky: #dceeff;
    --blue: #3478c9;
    --deep: #173b67;
    --ink: #16304d;
    --soft: #eaf4ff;
}
.stApp {
    background: linear-gradient(135deg, #ffffff 0%, #f5faff 48%, #eaf4ff 100%);
    color: var(--ink);
}
.block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
h1, h2, h3 {color: var(--deep); letter-spacing: -0.02em;}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #edf7ff 0%, #ffffff 100%);
    border-right: 1px solid #d7e9fb;
}
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 24px;
    background: rgba(255,255,255,.86);
    border: 1px solid #d9ebfb;
    box-shadow: 0 14px 36px rgba(52,120,201,.10);
    margin-bottom: 1.2rem;
}
.hero-title {font-size: 2.35rem; font-weight: 800; color: #173b67;}
.hero-sub {font-size: 1.03rem; color: #52708f; max-width: 850px;}
.kpi {
    background: rgba(255,255,255,.9);
    border: 1px solid #dcecfb;
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(44,101,160,.08);
    min-height: 118px;
}
.kpi-label {color:#6c86a1; font-size:.82rem; font-weight:700; text-transform:uppercase;}
.kpi-value {color:#173b67; font-size:1.65rem; font-weight:800; margin-top:.2rem;}
.kpi-note {color:#6d839a; font-size:.82rem;}
.insight {
    padding: 1rem 1.1rem;
    background: #ffffff;
    border: 1px solid #dcecfb;
    border-left: 5px solid #6ba7df;
    border-radius: 16px;
    margin: .55rem 0;
}
.highlight-positive {background:#dff4ff; border-radius:5px; padding:1px 3px;}
.highlight-risk {background:#fff0d9; border-radius:5px; padding:1px 3px;}
.highlight-commitment {background:#e8e1ff; border-radius:5px; padding:1px 3px;}
.note-chip {
    display:inline-block; margin:.2rem .3rem .2rem 0; padding:.28rem .58rem;
    border-radius:999px; background:#e7f3ff; color:#2a5e91; font-size:.78rem;
}
.small-muted {font-size:.83rem;color:#6f879f;}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.88);
    border: 1px solid #dcecfb;
    padding: .7rem;
    border-radius: 16px;
}
.stButton>button, .stDownloadButton>button {
    border-radius: 12px;
    border: 1px solid #9dc6eb;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@dataclass
class Letter:
    company: str
    year: int
    industry: str
    text: str
    filename: str


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        pages.append(f"\n[PAGE {idx}]\n{txt}")
    return "\n".join(pages)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text(text)) if len(s.strip()) > 25]


def heuristic_score(text: str, terms: List[str]) -> int:
    low = text.lower()
    count = sum(low.count(term) for term in terms)
    words = max(len(low.split()), 1)
    density = count / words * 1000
    return int(np.clip(35 + density * 8, 5, 95))


def sentence_evidence(text: str, terms: List[str], n: int = 3) -> List[str]:
    scored = []
    for sentence in split_sentences(text):
        score = sum(sentence.lower().count(t) for t in terms)
        if score:
            scored.append((score, sentence))
    return [s for _, s in sorted(scored, reverse=True)[:n]]


def extract_commitments(text: str, n: int = 8) -> List[str]:
    found = []
    clean = clean_text(text)
    for pattern in COMMITMENT_PATTERNS:
        found.extend(re.findall(pattern, clean, flags=re.I))
    unique = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique[:n]


def analyze_heuristic(letter: Letter) -> Dict:
    text = letter.text
    tones = {}
    tone_evidence = {}
    for name, terms in TONE_LEXICONS.items():
        tones[name] = heuristic_score(text, terms)
        tone_evidence[name] = sentence_evidence(text, terms)

    priorities = []
    for theme, terms in STRATEGY_KEYWORDS.items():
        score = heuristic_score(text, terms)
        evidence = sentence_evidence(text, terms, 2)
        priorities.append({"theme": theme, "score": score, "evidence": evidence})
    priorities = sorted(priorities, key=lambda x: x["score"], reverse=True)[:6]

    sentences = split_sentences(text)
    risks = sentence_evidence(text, TONE_LEXICONS["Caution"], 5)
    commitments = extract_commitments(text)

    return {
        "company": letter.company,
        "year": letter.year,
        "industry": letter.industry,
        "tone_scores": tones,
        "tone_evidence": tone_evidence,
        "strategic_priorities": priorities,
        "commitments": commitments,
        "risks": risks,
        "summary": " ".join(sentences[:4]) if sentences else "No readable text found.",
        "mode": "Heuristic demo",
    }


def analyze_with_openai(letter: Letter, api_key: str, model: str) -> Dict:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    client = OpenAI(api_key=api_key)
    trimmed = letter.text[:110000]
    schema_instruction = """
Return valid JSON only with this structure:
{
  "summary": "string",
  "tone_scores": {
    "Optimism": 0-100,
    "Caution": 0-100,
    "Accountability": 0-100,
    "Specificity": 0-100,
    "Long-term orientation": 0-100
  },
  "tone_evidence": {"Optimism":["quote"],"Caution":["quote"],"Accountability":["quote"],"Specificity":["quote"],"Long-term orientation":["quote"]},
  "strategic_priorities":[{"theme":"string","score":0-100,"evidence":["quote"]}],
  "commitments":["verbatim or close paraphrase"],
  "risks":["string"]
}
Use evidence grounded only in the supplied letter. Scores must be internally consistent.
"""
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a rigorous equity research analyst. Never invent facts."},
            {"role": "user", "content": f"{schema_instruction}\n\nCompany: {letter.company}\nYear: {letter.year}\nIndustry: {letter.industry}\n\nLETTER:\n{trimmed}"}
        ],
        text={"format": {"type": "json_object"}},
    )
    result = json.loads(response.output_text)
    result.update({
        "company": letter.company,
        "year": letter.year,
        "industry": letter.industry,
        "mode": "AI analysis",
    })
    return result


def highlighted_letter(text: str, analysis: Dict) -> str:
    excerpts = []
    for category, items in analysis.get("tone_evidence", {}).items():
        for item in items[:2]:
            excerpts.append((item, "positive" if category in ["Optimism", "Long-term orientation"] else "risk", category))
    for item in analysis.get("commitments", [])[:5]:
        excerpts.append((item, "commitment", "Commitment"))

    shown = text[:18000]
    for excerpt, cls, label in sorted(excerpts, key=lambda x: len(x[0]), reverse=True):
        if excerpt and excerpt in shown:
            shown = shown.replace(
                excerpt,
                f'<span class="highlight-{cls}" title="{label}">{excerpt}</span>',
                1,
            )
    shown = shown.replace("[PAGE ", "<br><br><b>Page ").replace("]", "</b><br>")
    return shown


def tone_radar(analyses: List[Dict]):
    categories = list(TONE_LEXICONS.keys())
    fig = go.Figure()
    for a in analyses:
        values = [a["tone_scores"].get(c, 0) for c in categories]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=f'{a["company"]} {a["year"]}',
            opacity=.65,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        margin=dict(l=40,r=40,t=50,b=40),
        height=470,
        title="Management Tone Fingerprint",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-.12),
    )
    return fig


def strategy_constellation(analyses: List[Dict]):
    rows = []
    for a in analyses:
        for p in a["strategic_priorities"]:
            rows.append({
                "company_year": f'{a["company"]} {a["year"]}',
                "theme": p["theme"],
                "importance": p["score"],
                "specificity": a["tone_scores"].get("Specificity", 50),
                "long_term": a["tone_scores"].get("Long-term orientation", 50),
            })
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="specificity",
        y="long_term",
        size="importance",
        color="company_year",
        hover_name="theme",
        hover_data={"importance": True, "specificity": True, "long_term": True},
        size_max=42,
        labels={
            "specificity": "Strategic specificity",
            "long_term": "Long-term orientation",
            "company_year": "Company / year",
        },
        title="Strategy Constellation",
    )
    fig.add_vline(x=50, line_dash="dot", opacity=.3)
    fig.add_hline(y=50, line_dash="dot", opacity=.3)
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.65)")
    return fig


def priority_ribbon(analyses: List[Dict]):
    rows = []
    for a in analyses:
        for p in a["strategic_priorities"]:
            rows.append({"company_year": f'{a["company"]} {a["year"]}', "theme": p["theme"], "score": p["score"]})
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="theme", columns="company_year", values="score", aggfunc="mean", fill_value=0)
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale=["#f6fbff", "#b9dcf7", "#4f91cf", "#173b67"],
        title="Strategic Emphasis Map",
        labels=dict(color="Importance"),
    )
    fig.update_layout(height=440, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def recommendation(analyses: List[Dict]) -> str:
    if len(analyses) < 2:
        a = analyses[0]
        strongest = max(a["tone_scores"], key=a["tone_scores"].get)
        weakest = min(a["tone_scores"], key=a["tone_scores"].get)
        return (
            f"Management communication is strongest in **{strongest.lower()}** "
            f"but weakest in **{weakest.lower()}**. Treat the current narrative as more credible "
            f"when strategic claims are accompanied by quantified targets and later financial verification."
        )

    ranked = sorted(
        analyses,
        key=lambda a: (
            a["tone_scores"].get("Accountability", 0)
            + a["tone_scores"].get("Specificity", 0)
            + a["tone_scores"].get("Long-term orientation", 0)
        ) / 3,
        reverse=True,
    )
    leader, laggard = ranked[0], ranked[-1]
    return (
        f"**{leader['company']} {leader['year']}** presents the strongest evidence-backed management narrative, "
        f"driven by higher accountability, specificity, and long-term orientation. "
        f"**{laggard['company']} {laggard['year']}** should be treated more cautiously: verify its major claims "
        f"against subsequent margins, free cash flow, ROIC, leverage, and shareholder returns before drawing an investment conclusion."
    )


def performance_chart(perf_df: pd.DataFrame, metric: str):
    fig = px.line(
        perf_df,
        x="Year",
        y=metric,
        color="Company",
        markers=True,
        title=f"Long-term {metric}",
    )
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.65)")
    return fig


st.markdown("""
<div class="hero">
  <div class="hero-title">Shareholder Letter Intelligence</div>
  <div class="hero-sub">
    Compare management tone, strategic priorities, promises and long-term performance.
    Every conclusion remains connected to the source letter.
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis workspace")
    industry = st.selectbox(
    "Industry",
    list(INDUSTRIES.keys()),
    key="sidebar_industry"
)
    st.caption("Industry vocabulary helps contextualise strategic priorities.")
    st.markdown("**Industry lenses**")
    st.markdown(" ".join([f'<span class="note-chip">{x}</span>' for x in INDUSTRIES[industry]]), unsafe_allow_html=True)

    st.divider()
    mode = st.radio("Analysis mode", ["Demo / heuristic", "OpenAI API"])
    api_key = ""
    model = "gpt-4.1-mini"
    if mode == "OpenAI API":
        api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model = st.text_input("Model", value="gpt-4.1-mini")
    st.caption("The demo mode is useful for interface testing. AI mode provides materially better interpretation.")

tabs = st.tabs(["Upload & Analyse", "Visual Comparison", "Annotated Letter", "Long-term Performance", "Methodology"])

with tabs[0]:
    st.subheader("Upload shareholder letters")
    col1, col2, col3 = st.columns([1.5,1,1])
    with col1:
        company = st.text_input("Company name", placeholder="e.g., Microsoft")
    with col2:
        year = st.number_input("Fiscal year", min_value=1990, max_value=2100, value=2025, step=1)
    with col3:
        uploaded = st.file_uploader("Annual letter PDF", type=["pdf"], accept_multiple_files=False)

    if "letters" not in st.session_state:
        st.session_state.letters = []
    if "analyses" not in st.session_state:
        st.session_state.analyses = []

    if st.button("Add and analyse letter", type="primary", use_container_width=True):
        if not company or uploaded is None:
            st.error("Add a company name and PDF.")
        elif mode == "OpenAI API" and not api_key:
            st.error("Add an OpenAI API key or switch to demo mode.")
        else:
            try:
                raw_text = extract_pdf_text(uploaded)
                letter = Letter(company, int(year), industry, raw_text, uploaded.name)
                with st.spinner("Analysing management narrative..."):
                    result = analyze_with_openai(letter, api_key, model) if mode == "OpenAI API" else analyze_heuristic(letter)
                st.session_state.letters.append(letter)
                st.session_state.analyses.append(result)
                st.success(f"Analysed {company} {year}.")
            except Exception as exc:
                st.exception(exc)

    if st.session_state.analyses:
        latest = st.session_state.analyses[-1]
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Optimism", latest["tone_scores"].get("Optimism", 0), "Future confidence"),
            ("Accountability", latest["tone_scores"].get("Accountability", 0), "Ownership of outcomes"),
            ("Specificity", latest["tone_scores"].get("Specificity", 0), "Measurable detail"),
            ("Long-term", latest["tone_scores"].get("Long-term orientation", 0), "Durable value focus"),
        ]
        for col, (label, value, note) in zip([c1,c2,c3,c4], metrics):
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}/100</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

        st.markdown("### Executive interpretation")
        st.markdown(f'<div class="insight">{latest["summary"]}</div>', unsafe_allow_html=True)

        st.markdown("### Strategic priorities")
        pcols = st.columns(3)
        for i, p in enumerate(latest["strategic_priorities"][:6]):
            with pcols[i % 3]:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">Priority {i+1}</div>'
                    f'<div class="kpi-value" style="font-size:1.1rem">{p["theme"]}</div>'
                    f'<div class="kpi-note">Importance {p["score"]}/100</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("### Commitments detected")
        if latest["commitments"]:
            for item in latest["commitments"]:
                st.markdown(f'<div class="insight">{item}</div>', unsafe_allow_html=True)
        else:
            st.info("No explicit commitments were detected in the current analysis.")

with tabs[1]:
    if not st.session_state.get("analyses"):
        st.info("Upload at least one letter first. Two or more letters unlock meaningful comparison.")
    else:
        analyses = st.session_state.analyses
        st.plotly_chart(tone_radar(analyses), use_container_width=True)
        st.plotly_chart(strategy_constellation(analyses), use_container_width=True)
        if len(analyses) >= 2:
            st.plotly_chart(priority_ribbon(analyses), use_container_width=True)

        st.markdown("### Investment-research outcome")
        st.markdown(f'<div class="insight">{recommendation(analyses)}</div>', unsafe_allow_html=True)

        export = pd.DataFrame([
            {
                "Company": a["company"],
                "Year": a["year"],
                **a["tone_scores"],
                "Top priority": a["strategic_priorities"][0]["theme"] if a["strategic_priorities"] else "",
                "Commitments detected": len(a["commitments"]),
            }
            for a in analyses
        ])
        st.download_button(
            "Download comparison data",
            export.to_csv(index=False).encode("utf-8"),
            "shareholder_letter_comparison.csv",
            "text/csv",
        )

with tabs[2]:
    if not st.session_state.get("letters"):
        st.info("Upload a letter to create the annotated reading view.")
    else:
        choices = [f"{l.company} {l.year}" for l in st.session_state.letters]
        selected = st.selectbox("Letter", range(len(choices)), format_func=lambda i: choices[i])
        letter = st.session_state.letters[selected]
        analysis = st.session_state.analyses[selected]

        st.markdown(
            '<span class="highlight-positive">Opportunity / long-term evidence</span> '
            '<span class="highlight-risk">Risk / caution evidence</span> '
            '<span class="highlight-commitment">Management commitment</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="hero" style="max-height:680px;overflow:auto;line-height:1.75">{highlighted_letter(letter.text, analysis)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("The MVP highlights exact matched passages. A production version should use page-level coordinates for PDF overlays.")

with tabs[3]:
    st.subheader("Connect narrative to actual results")
    st.write("Paste or edit annual financial data. The app will compare management messaging with long-term performance.")

    default_df = pd.DataFrame({
        "Company": ["Example Co"] * 5,
        "Year": [2021, 2022, 2023, 2024, 2025],
        "Revenue": [100, 108, 116, 121, 130],
        "Operating Margin": [15.0, 15.8, 14.9, 13.7, 14.2],
        "Free Cash Flow": [12, 13, 12, 10, 13],
        "ROIC": [11.0, 12.0, 10.5, 9.7, 10.2],
        "Net Debt / EBITDA": [2.1, 1.9, 2.3, 2.6, 2.2],
        "TSR": [8.0, -4.0, 12.0, 5.0, 14.0],
    })
    perf_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)
    numeric_metrics = [c for c in perf_df.columns if c not in ["Company", "Year"]]
    metric = st.selectbox("Performance metric", numeric_metrics)
    st.plotly_chart(performance_chart(perf_df, metric), use_container_width=True)

    if len(perf_df) >= 2:
        first, last = perf_df.iloc[0], perf_df.iloc[-1]
        try:
            change = float(last[metric]) - float(first[metric])
            direction = "improved" if change > 0 else "declined"
            st.markdown(
                f'<div class="insight"><b>Performance outcome:</b> {metric} {direction} by '
                f'{abs(change):.2f} between {int(first["Year"])} and {int(last["Year"])}. '
                f'Compare this direction with changes in optimism and specificity before treating management language as predictive.</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

with tabs[4]:
    st.subheader("How the MVP works")
    st.markdown("""
1. **PDF extraction:** text is retained with page markers.
2. **Tone analysis:** five management-communication dimensions are scored.
3. **Strategic-priority analysis:** repeated themes are ranked by emphasis.
4. **Commitment extraction:** forward-looking promises and targets are identified.
5. **Comparison:** radar, constellation and emphasis-map views replace a primitive peer table.
6. **Evidence view:** supporting passages are highlighted in the letter.
7. **Performance layer:** financial trends are compared with management narrative.

**Important:** Demo mode uses transparent keyword heuristics and is intended for interface testing. Use AI mode for semantic interpretation, then manually validate material investment conclusions.
""")
