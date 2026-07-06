import streamlit as st
import fitz  # PyMuPDF
import re
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF
from collections import Counter
import numpy as np

# --- 1. UI CONFIG ---
st.set_page_config(page_title="CareerAI Intelligence System", layout="wide")

@st.cache_resource
def load_nlp_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_nlp_models()

# --- 2. DATASETS & ARCHITECTURAL CONFIG ---
MARKET_DEMAND = {
    "python": "Very High Demand", "sql": "Very High Demand", "react": "High Demand",
    "aws": "High Demand", "cloud": "High Demand", "excel": "Moderate Demand",
    "powerbi": "High Demand", "tableau": "High Demand", "machine": "Growing Demand",
    "learning": "Growing Demand", "nlp": "Growing Demand", "java": "Stable Demand",
    "javascript": "Very High Demand", "figma": "Moderate Demand"
}

SKILL_WEIGHTS = {
    "python": 5, "sql": 5, "react": 5, "aws": 5,
    "javascript": 4, "machine": 4, "learning": 4, "nlp": 4, "cloud": 4,
    "powerbi": 3, "tableau": 3, "excel": 3,
    "communication": 2, "leadership": 2, "powerpoint": 1
}

SKILL_WHITELIST = set(SKILL_WEIGHTS.keys()).union({
    'html', 'css', 'modeling', 'visualization', 'database', 'git', 
    'cleaning', 'dashboard', 'pandas', 'numpy', 'scipy', 'adobe xd', 'testing'
})

SECTION_PATTERNS = {
    "education": ["education", "academic background", "qualification"],
    "skills": ["skills", "technical skills", "competencies"],
    "projects": ["projects", "academic projects"],
    "experience": ["experience", "work experience", "internship"],
    "certifications": ["certifications", "licenses"]
}

# --- 3. PARSING HEURISTICS ---
def clean_text(text):
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_text(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    except Exception:
        return ""

def get_keywords(text):
    return set(re.findall(r'\b\w{3,}\b', text.lower()))

def get_youtube_link(skill_name):
    query = skill_name.replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}+tutorial+for+beginners"

def detect_resume_sections(text):
    text_lower = text.lower()
    found_sections = []
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                found_sections.append(section)
                break
    return found_sections

def calculate_ats_score(text, sections):
    score = 0
    recommendations = []
    if re.search(r'\S+@\S+', text): score += 10
    else: recommendations.append("Add professional contact email address")
    
    if "linkedin" in text.lower(): score += 10
    else: recommendations.append("Missing professional LinkedIn alignment")
    
    if "github" in text.lower(): score += 10
    else: recommendations.append("Missing active GitHub technical portfolio")

    for section in ["education", "skills", "projects", "experience"]:
        if section in sections: score += 10
        else: recommendations.append(f"Missing distinct structural '{section.title()}' segmentation")

    word_count = len(text.split())
    if 300 <= word_count <= 900: score += 20
    else: recommendations.append("Structural density alert: Optimize word metrics between 300-900 words")

    action_verbs = ["developed", "built", "created", "designed", "implemented", "optimized", "analyzed"]
    found_verbs = sum(1 for verb in action_verbs if verb in text.lower())
    if found_verbs >= 3: score += 20
    else: recommendations.append("Diction alert: Integrate analytical action verbs (e.g., 'optimized', 'engineered')")

    return min(score, 100), recommendations

def calculate_weighted_score(matched_skills):
    return sum(SKILL_WEIGHTS.get(skill.lower(), 1) for skill in matched_skills)

def calculate_interview_probability(match_score, ats_score, matched_skills):
    skill_factor = min(len(matched_skills) * 3, 25)
    probability = (match_score * 0.5) + (ats_score * 0.3) + skill_factor
    return round(min(probability, 100), 1)

def create_pdf_report(percentage, gaps, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "CareerAI Architecture Analytics Report", ln=True, align='C')
    safe_summary = summary.encode('ascii', 'ignore').decode('ascii')
    pdf.ln(10)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, f"Executive Analytical Context:\n{safe_summary}")
    return pdf.output(dest='S').encode('latin-1')

# Initialize Session States
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'percentage' not in st.session_state: st.session_state.percentage = 0.0
if 'skill_gaps' not in st.session_state: st.session_state.skill_gaps = []
if 'narrative' not in st.session_state: st.session_state.narrative = ""
if 'ats_score' not in st.session_state: st.session_state.ats_score = 0
if 'interview_probability' not in st.session_state: st.session_state.interview_probability = 0.0
if 'resume_suggestions' not in st.session_state: st.session_state.resume_suggestions = []
if 'matched_skills' not in st.session_state: st.session_state.matched_skills = []
if 'weighted_skill_score' not in st.session_state: st.session_state.weighted_skill_score = 0

# --- 4. SIDEBAR NAVIGATION & BACKGROUND ENGINE ---
st.sidebar.title("🌑 CareerAI Platform v1.2")
page = st.sidebar.radio("Systems Diagnostic Hub:", ["🔍 Predictive Career Mapping", "🎯 Precision Profile Matching"])

WHITE_BG = "https://github.com/KeshavaAditi04/CareerAI-Intelligence-System/raw/refs/heads/main/ChatGPT%20Image%20Jul%207,%202026,%2002_42_59%20AM.png"
GREEN_BG = "https://github.com/KeshavaAditi04/CareerAI-Intelligence-System/raw/refs/heads/main/ChatGPT%20Image%20Jul%207,%202026,%2002_46_38%20AM.png"

bg_url = GREEN_BG if page == "🔍 Predictive Career Mapping" else WHITE_BG

if page == "🎯 Precision Profile Matching":
    text_color = "#0F172A"       
    label_color = "#1E293B"      
    header_color = "#8B0000"     
    card_bg = "rgba(255, 255, 255, 0.75)" 
    card_border = "rgba(15, 23, 42, 0.15)"
else:
    text_color = "#F8FAFC"       
    label_color = "#F8FAFC"
    header_color = "#D4AF37"     
    card_bg = "rgba(13, 22, 18, 0.8)" 
    card_border = "rgba(212, 175, 55, 0.2)"

css_template = """
    <style>
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    .main, 
    .stApp,
    [data-testid="stAppViewBlockContainer"] {
        background-image: url("VAR_BG_URL") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        animation: none !important;
    }

    [data-testid="stWidgetLabel"] p, label, p, span {
        color: VAR_LABEL_COLOR !important;
    }
    
    .stMarkdown p {
        color: VAR_TEXT_COLOR !important;
    }

    h1, h2, h3, .stSubheader {
        color: VAR_HEADER_COLOR !important;
        font-weight: 700;
    }

    [data-testid="stForm"], .stAlert, .gap-box-critical, .gap-box-optimize, .roadmap-card {
        background-color: VAR_CARD_BG !important;
        border: 1px solid VAR_CARD_BORDER !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }

    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #050807 !important; 
        border-right: 1px solid rgba(212, 175, 55, 0.15) !important;
    }
    </style>
"""

custom_css = (css_template
              .replace("VAR_BG_URL", bg_url)
              .replace("VAR_LABEL_COLOR", label_color)
              .replace("VAR_TEXT_COLOR", text_color)
              .replace("VAR_HEADER_COLOR", header_color)
              .replace("VAR_CARD_BG", card_bg)
              .replace("VAR_CARD_BORDER", card_border))

st.markdown(custom_css, unsafe_allow_html=True)

with st.sidebar.container():
    st.caption("⚡ SYSTEM DEVELOPER")
    st.subheader("Aditi Das")
    st.markdown("🎓 **Project Matrix Domain**")
    st.code("BCA Final Year Project | 2026", language="text")
    st.markdown("🧠 **Core Architecture**")
    st.caption("SentenceTransformer NLP")

st.sidebar.divider()

# --- 5. PAGE 1: CAREER DISCOVERY ---
if page == "🔍 Predictive Career Mapping":
    st.title("🔍 Predictive Career Mapping System")
    st.write("Processing linguistic parameters to predict structural alignment across industry vectors.")
    
    benchmarks = {
        "Data Analytics": "SQL, Python, Power BI, data visualization, and statistical modeling.",
        "Web Development": "HTML, CSS, JavaScript, React, Node.js, and web architecture.",
        "UI/UX Design": "Figma, Adobe XD, user research, wireframing, and prototypes.",
        "Software Engineering": "C++, Java, algorithms, system design, and testing.",
        "Machine Learning": "Python, machine learning, deep learning, NLP, TensorFlow, modeling.",
        "Cloud Engineering": "AWS, cloud computing, DevOps, Docker, Kubernetes.",
        "Business Analysis": "Excel, Power BI, stakeholder communication, analytics, dashboards.",
        "Cybersecurity": "network security, ethical hacking, Linux, SIEM, risk analysis."
    }

    res_file = st.file_uploader("Insert Academic Document Profile (PDF)", type=["pdf"])
    
    if res_file:
        with st.status("Initializing Predictive Mapping Framework...", expanded=True) as status:
            st.write("Extracting structural corpus layout...")
            resume_text = clean_text(extract_text(res_file.getvalue()))
            time.sleep(0.4)
            
            if not resume_text.strip():
                st.error("Execution Deficit: Unable to compile structural text context.")
                status.update(label="Framework Execution Fault", state="error")
            else:
                st.write("Projecting token embeddings into vector workspace...")
                res_vec = model.encode(resume_text, convert_to_tensor=True)
                time.sleep(0.4)
                
                st.write("Running cross-functional similarity indices...")
                results = []
                for role, desc in benchmarks.items():
                    bench_vec = model.encode(clean_text(desc), convert_to_tensor=True)
                    score = util.cos_sim(res_vec, bench_vec).item()
                    results.append({"Target Domain": role, "Similarity Weight": round(score * 100, 1)})
                
                df = pd.DataFrame(results).sort_values(by="Similarity Weight", ascending=False)
                status.update(label="Dynamic Role Mapping Compiled Successfully!", state="complete", expanded=False)
                
                st.success(f"💥 Dominant Alignment Predicted: **{df.iloc[0]['Target Domain']}**")
                
                fig_bar = px.bar(df, x='Similarity Weight', y='Target Domain', orientation='h', 
                                 color='Similarity Weight', color_continuous_scale='Greens', text='Similarity Weight')
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", plot_bgcolor="rgba(0,0,0,0)", height=450)
                st.plotly_chart(fig_bar, use_container_width=True)

# --- 6. PAGE 2: PRECISION MATCH ---
elif page == "🎯 Precision Profile Matching":
    st.title("🎯 Precision Semantic Alignment Pipeline")
    
    with st.form("computational_matrix_form"):
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Upload Profile Signature (PDF)", type=["pdf"])
        with col2:
            jd_text = st.text_area("Target Job Vector Requirements (Paste JD)", height=200)
        
        sandbox_input = st.text_input("🧬 Predictive Skill Sandbox Insertion Hub (Type skills to simulate real-time updates)")
        submit_button = st.form_submit_button("🚀 INITIATE SEMANTIC AGGREGATION")

    if submit_button:
        if uploaded_file and jd_text:
            with st.status("Synchronizing Distributed Vector Arrays...", expanded=True) as status:
                st.write("Extracting deep layout content from profile array...")
                resume_raw = extract_text(uploaded_file.getvalue())
                clean_jd = clean_text(jd_text)
                clean_res = clean_text(resume_raw)
                time.sleep(0.5)
                
                st.write("Parsing structural resume segments and metadata rules...")
                sections_found = detect_resume_sections(resume_raw)
                ats_score, ats_rec = calculate_ats_score(resume_raw, sections_found)
                time.sleep(0.4)
                
                st.write("Generating neural tensor embeddings for alignment scoring...")
                res_vec = model.encode(clean_res, convert_to_tensor=True)
                jd_vec = model.encode(clean_jd, convert_to_tensor=True)
                base_score = util.cos_sim(res_vec, jd_vec).item()
                
                resume_keywords = get_keywords(resume_raw)
                jd_keywords = get_keywords(clean_jd)
                
                if sandbox_input.strip():
                    st.write(f"Injecting simulated parameters from sandbox: '{sandbox_input}'...")
                    resume_keywords.update(get_keywords(sandbox_input))
                    sand_vec = model.encode(clean_text(sandbox_input), convert_to_tensor=True)
                    sand_match = util.cos_sim(sand_vec, jd_vec).item()
                    final_score = min(base_score + (sand_match * 0.12), 1.0)
                    time.sleep(0.3)
                else:
                    final_score = base_score

                missing_raw = jd_keywords - resume_keywords
                st.session_state.skill_gaps = [m.upper() for m in missing_raw if m.lower() in SKILL_WHITELIST]
                st.session_state.percentage = round(final_score * 100, 1)
                
                st.session_state.matched_skills = list(resume_keywords.intersection(jd_keywords).intersection(SKILL_WHITELIST))
                st.session_state.weighted_skill_score = calculate_weighted_score(st.session_state.matched_skills)
                
                st.session_state.ats_score = ats_score
                st.session_state.resume_suggestions = ats_rec
                st.session_state.interview_probability = calculate_interview_probability(
                    st.session_state.percentage, ats_score, st.session_state.matched_skills
                )
                
                if st.session_state.percentage > 80:
                    st.session_state.narrative = "Profile exhibits elite structural affinity. Semantic distribution closely maps to high-tier target metrics."
                elif 40 <= st.session_state.percentage <= 80:
                    st.session_state.narrative = f"Competitive logic matrix found. Profile contains foundational patterns but remains under-indexed for major target structures."
                else:
                    st.session_state.narrative = "High dimensional variance observed. Vector configuration suggests a significant professional re-indexing pathway is required."
                
                status.update(label="Semantic Compilation Complete! Vector Matrix Populated.", state="complete", expanded=False)
                st.session_state.analyzed = True
            if st.session_state.percentage >= 75:
                st.snow()

    if st.session_state.analyzed:
        # --- COMPACT HIGH-CONTRAST GAUGES ---
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            fig_1 = go.Figure(go.Indicator(mode="gauge+number", value=st.session_state.percentage, 
                title={'text': "Semantic Affinity Score", 'font': {'color': "#FFFFFF", 'size': 14, 'weight': 'bold'}},
                gauge={'bar': {'color': "#D4AF37"}, 'bgcolor': "rgba(255,255,255,0.05)"}))
            fig_1.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFFFFF"}, height=220, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig_1, use_container_width=True)
            
        with col_g2:
            fig_2 = go.Figure(go.Indicator(mode="gauge+number", value=st.session_state.ats_score, 
                title={'text': "ATS Layout Structural Index", 'font': {'color': "#FFFFFF", 'size': 14, 'weight': 'bold'}},
                gauge={'bar': {'color': "#0D9488"}, 'bgcolor': "rgba(255,255,255,0.05)"}))
            fig_2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFFFFF"}, height=220, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig_2, use_container_width=True)
            
        with col_g3:
            fig_3 = go.Figure(go.Indicator(mode="gauge+number", value=st.session_state.interview_probability, 
                title={'text': "Interview Convocation Likelihood", 'font': {'color': "#FFFFFF", 'size': 14, 'weight': 'bold'}},
                gauge={'bar': {'color': "#475569"}, 'bgcolor': "rgba(255,255,255,0.05)"}))
            fig_3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFFFFF"}, height=220, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig_3, use_container_width=True)

        # --- NARRATIVE COMPLIANCE SUMMARY ---
        st.info(f"🧠 **System Executive Context:** {st.session_state.narrative}")

        # --- STRATEGIC AI RESUME ENHANCEMENTS ---
        st.write("") 
        st.markdown("### 👑 Strategic AI Resume Enhancements")

        if st.session_state.skill_gaps:
            st.markdown("<p style='color:#A3B8CC;'>Our semantic parsing engine detected that the target job description heavily weights specific keywords that are missing or weak in your profile signature. Adding these will instantly boost your alignment vector.</p>", unsafe_allow_html=True)
            for skill in sorted(st.session_state.skill_gaps):
                st.markdown(f"✨ **Missing Target Vector:** Consider integrating the phrase `{skill.upper()}` into your professional profile or experience summaries.")
        else:
            st.success("🏆 **Perfect Skill Alignment:** Excellent structural integrity! Your profile matrix already maps cleanly to all critical industry domain keywords found within the target requirements vector.")

        # --- STRATEGIC COMPETENCY GAPS ---
        st.divider()
        st.subheader("⚔️ Automated Competency Deficit Matrix")
        
        if not st.session_state.skill_gaps:
            st.success("✨ Dimensional completeness achieved. No technical skill missing links discovered.")
        else:
            half = max(1, len(st.session_state.skill_gaps) // 2)
            crit_items = st.session_state.skill_gaps[:half]
            opt_items = st.session_state.skill_gaps[half:]
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🛑 Primary Structural Discrepancies")
                for gap in crit_items[:3]:
                    st.markdown(f'<div class="gap-box-critical">☠️ Primary Deficit Token: <b>{gap}</b></div>', unsafe_allow_html=True)
            with c2:
                st.markdown("#### ⚠️ Secondary Vector Optimizations")
                for gap in (opt_items[:3] if opt_items else crit_items[3:6]):
                    st.markdown(f'<div class="gap-box-optimize">⚡ Optimization Priority: <b>{gap}</b></div>', unsafe_allow_html=True)

        # --- ADAPTIVE DEVELOPMENT ROADMAP ---
        st.divider()
        st.subheader("📅 Heuristic Adaptive Development Sequence (3-Week Matrix)")
        
        if st.session_state.skill_gaps:
            steps = [st.session_state.skill_gaps[i] if i < len(st.session_state.skill_gaps) else "Advanced Optimization" for i in range(3)]
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="roadmap-card">⏳ <b>Week 1: Foundations</b><br><p style="margin-bottom:8px; color:#94A3B8;">Isolate and process:</p><b style="font-size:1.15rem; color:#FFF;">{steps[0]}</b><br><a class="yt-link" href="{get_youtube_link(steps[0])}" target="_blank">🎥 Open Curated Repository ↗</a></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="roadmap-card">⏳ <b>Week 2: Integration</b><br><p style="margin-bottom:8px; color:#94A3B8;">Deploy implementations for:</p><b style="font-size:1.15rem; color:#FFF;">{steps[1]}</b><br><a class="yt-link" href="{get_youtube_link(steps[1])}" target="_blank">🎥 Open Curated Repository ↗</a></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="roadmap-card">⏳ <b>Week 3: Production</b><br><p style="margin-bottom:8px; color:#94A3B8;">Refine scalable pipelines for:</p><b style="font-size:1.15rem; color:#FFF;">{steps[2]}</b><br><a class="yt-link" href="{get_youtube_link(steps[2])}" target="_blank">🎥 Open Curated Repository ↗</a></div>', unsafe_allow_html=True)

        # --- RESUME AUDIT SUGGESTIONS ---
        st.divider()
        st.subheader("🛠️ Structural Portfolio Audit Engineering")
        if not st.session_state.resume_suggestions:
            st.success("Structure and metadata standards are fully optimized.")
        else:
            for rec in st.session_state.resume_suggestions[:4]:
                st.markdown(f'<div class="gap-box-optimize">⚙️ <b>Optimization Rule:</b> {rec}</div>', unsafe_allow_html=True)

        # --- MARKET ACCELERATION INSIGHTS ---
        st.divider()
        st.subheader("📈 Real-Time Macro-Market Demand Projections")
        if st.session_state.matched_skills:
            m_cols = st.columns(min(len(st.session_state.matched_skills), 4))
            for i, skill in enumerate(st.session_state.matched_skills[:4]):
                with m_cols[i]:
                    demand = MARKET_DEMAND.get(skill.lower(), "Stable Index")
                    st.markdown(f'<div class="gap-box-optimize" style="text-align:center;">📊 <b>{skill.upper()}</b><br><span style="color:#D4AF37; font-size:0.9rem;">{demand}</span></div>', unsafe_allow_html=True)

        # --- CONFIDENCE EXPLAINABILITY BAR CHART ---
        st.divider()
        st.subheader("📊 Architectural Multi-Criteria Confidence Metrics")
        
        breakdown_df = pd.DataFrame({
            "Evaluation Dimension": ["Semantic Proximity Match", "ATS Layout Structural Quality", "Weighted Core Competency Volumetrics", "Static Context Heuristics", "Keyword Architecture Index"],
            "System Score Score": [st.session_state.percentage, st.session_state.ats_score, min(st.session_state.weighted_skill_score * 5, 100), 75, 82]
        })
        fig_break = px.bar(breakdown_df, x='Evaluation Dimension', y='System Score Score', color='System Score Score', color_continuous_scale='Greens')
        fig_break.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_break, use_container_width=True)

        # --- PDF SUMMARY REPORT EXPORT ---
        st.divider()
        st.subheader("📥 Data Vector Reporting Export")
        pdf_bytes = create_pdf_report(st.session_state.percentage, st.session_state.skill_gaps, st.session_state.narrative)
        st.download_button("📄 Download Deep System Analysis Logs (PDF)", pdf_bytes, "System_Analysis_Logs.pdf", "application/pdf")
