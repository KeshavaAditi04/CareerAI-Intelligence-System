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
st.set_page_config(page_title="CareerAI Computer Science Career Assistant", layout="wide")

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

# --- 3. HELPER FUNCTIONS & RESET ENGINE ---
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
    pdf.cell(200, 10, "CareerAI Analytics Summary Report", ln=True, align='C')
    safe_summary = summary.encode('ascii', 'ignore').decode('ascii')
    pdf.ln(10)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, f"Analysis Summary:\n{safe_summary}")
    return pdf.output(dest='S').encode('latin-1')
def create_radar_chart(match_score, ats_score, interview_prob):
    categories = ['<b>Skill Alignment</b>', '<b>Resume Readiness</b>', '<b>Callback Likelihood</b>', '<b>Match Score</b>']
    values = [match_score, ats_score, interview_prob, match_score]

    fig = go.Figure()

    # User Score Polygon
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Profile',
        line=dict(width=2)
    ))

    # Target Benchmark Polygon (100% ideal)
    fig.add_trace(go.Scatterpolar(
        r=[100, 100, 100, 100],
        theta=categories,
        fill='toself',
        name='Ideal Target',
        fillcolor='rgba(148, 163, 184, 0.15)', # Light grey background target
        line=dict(color='#94A3B8', width=1, dash='dash')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#1E293B"),
            angularaxis=dict(
                color="#1E293B", 
                tickfont=dict(size=12)  # Replaced invalid font/weight keys
            )
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E293B"),
        height=380,
        margin=dict(l=40, r=40, t=30, b=30)
    )
    return fig
    
def reset_analysis():
    # 1. Reset metrics and flags
    st.session_state.analyzed = False
    st.session_state.percentage = 0.0
    st.session_state.skill_gaps = []
    st.session_state.narrative = ""
    st.session_state.ats_score = 0
    st.session_state.interview_probability = 0.0
    st.session_state.resume_suggestions = []
    st.session_state.matched_skills = []
    

    st.session_state["reset_counter"] = st.session_state.get("reset_counter", 0) + 1
    
    for key in ["p1_uploader", "p2_uploader"]:
        if key in st.session_state:
            del st.session_state[key]
            
    st.rerun()

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

# --- SIDEBAR BACKGROUND STYLING ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-image: linear-gradient(rgba(10, 15, 25, 0.75), rgba(10, 15, 25, 0.75)), url("https://raw.githubusercontent.com/KeshavaAditi04/CareerAI-Intelligence-System/main/ChatGPT%20Image%20Jul%2024,%202026,%2009_08_46%20PM.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR NAVIGATION ("Choose a tool") ---
st.sidebar.title("👩🏻‍💻 Choose your CareerAI tool :")
page = st.sidebar.radio(
    "Select Assistant Mode:", 
    ["💻 CS Career Explorer", "📄 Resume Match Analyzer"], 
    key="navigation_radio"
)

st.sidebar.divider()

# --- NEW PROFESSIONAL ABOUT SECTION ---
st.sidebar.markdown("### 🎯 About CareerAI")
st.sidebar.markdown("CareerAI helps Computer Science students:")
st.sidebar.write("✔ Find suitable career paths")
st.sidebar.write("✔ Compare resumes")
st.sidebar.write("✔ Improve ATS score")
st.sidebar.write("✔ Identify missing skills")
st.sidebar.write("✔ Receive AI recommendations")

st.sidebar.markdown("---")

# --- EXPANDED MATURED DEVELOPER CARD ---
st.sidebar.markdown("### 👩🏻‍💻 Developer Matrix")
st.sidebar.write("**Developed By:** Aditi Das")
st.sidebar.write("**Version:** `v2.0`")
st.sidebar.write("**Institution:** JCRE Global College")
st.sidebar.caption("Academic Year: 2022-2026")

st.sidebar.divider()

# --- BACKGROUND IMAGE INJECTION ENGINE ---
WHITE_BG = "https://github.com/KeshavaAditi04/CareerAI-Intelligence-System/raw/refs/heads/main/ChatGPT%20Image%20Jul%207,%202026,%2002_42_59%20AM.png"
GREEN_BG = "https://github.com/KeshavaAditi04/CareerAI-Intelligence-System/raw/refs/heads/main/ChatGPT%20Image%20Jul%207,%202026,%2002_46_38%20AM.png"

bg_url = GREEN_BG if page == "💻 CS Career Explorer" else WHITE_BG

text_color = "#1E293B"       
label_color = "#0F172A"      
card_bg = "rgba(255, 255, 255, 0.85)" 
card_border = "rgba(15, 23, 42, 0.15)"

if page == "📄 Resume Match Analyzer":
    header_color = "#991B1B"     
else:
    header_color = "#1E3A8A"     

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
        display: block !important;
    }

    .stMarkdown p, .stMarkdown li, div, p, span, .stText p {
        color: VAR_TEXT_COLOR !important;
    }
    
    [data-testid="stWidgetLabel"] p, label, [data-testid="stFileUploadDropzone"] div {
        color: VAR_LABEL_COLOR !important;
        font-weight: 700 !important;
    }

    h1, h2, h3, .stSubheader, [data-testid="stHeader"] h1, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, [data-testid="stHeadingWithLink"] h1 {
        color: VAR_HEADER_COLOR !important;
        font-weight: 900 !important;
    }

    [data-testid="stForm"], .stAlert, .gap-box-critical, .gap-box-optimize, .roadmap-card, [data-testid="stFileUploadDropzone"] {
        background-color: VAR_CARD_BG !important;
        border: 2px solid VAR_CARD_BORDER !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }

    [data-testid="stSidebar"] {
        background-color: #050807 !important; 
        border-right: 1px solid rgba(245, 158, 11, 0.15) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h1 span {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h3 span {
        color: #F59E0B !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] caption,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] + div label span,
    [data-testid="stSidebar"] [data-testid="stRadio"] div p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    </style>
"""

custom_css = (css_template
              .replace("VAR_BG_URL", bg_url)
              .replace("VAR_TEXT_COLOR", text_color)
              .replace("VAR_LABEL_COLOR", label_color)
              .replace("VAR_HEADER_COLOR", header_color)
              .replace("VAR_CARD_BG", card_bg)
              .replace("VAR_CARD_BORDER", card_border))

st.markdown(custom_css, unsafe_allow_html=True)

# --- 5. PAGE 1: CS CAREER EXPLORER ---
if page == "💻 CS Career Explorer":
    st.title("CareerAI Computer Science Career Assistant")
    st.write("### Scope: Optimized for Computer Science, IT, and Software Engineering pathways.")
    st.write("Upload your resume to discover which technical career domain matches your skills best.")
    
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

    res_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="p1_uploader")
    
    if res_file:
        with st.status("Analyzing Resume...", expanded=True) as status:
            st.write("Reading Resume...")
            resume_text = clean_text(extract_text(res_file.getvalue()))
            time.sleep(0.4)
            
            if not resume_text.strip():
                st.error("Error: Could not extract text content from the file.")
                status.update(label="Analysis Failed", state="error")
            else:
                st.write("Checking Resume Format...")
                res_vec = model.encode(resume_text, convert_to_tensor=True)
                time.sleep(0.4)
                
                st.write("Comparing Resume with Job Description...")
                results = []
                for role, desc in benchmarks.items():
                    bench_vec = model.encode(clean_text(desc), convert_to_tensor=True)
                    score = util.cos_sim(res_vec, bench_vec).item()
                    results.append({"Target Domain": role, "Match Strength (%)": round(score * 100, 1)})
                
                df = pd.DataFrame(results).sort_values(by="Match Strength (%)", ascending=False)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                st.success(f"💥 Top Recommended Track: **{df.iloc[0]['Target Domain']}**")
                
                # --- GREEN + GOLD THEME FOR CS CAREER EXPLORER ---
                fig_bar = px.bar(
                    df, 
                    x='Match Strength (%)', 
                    y='Target Domain', 
                    orientation='h', 
                    color='Match Strength (%)', 
                    color_continuous_scale=[[0, '#1E4620'], [1, '#D4AF37']], # Deep Forest Green to Rich Gold
                    text='Match Strength (%)'
                )
                
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font_color="#1E293B", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    height=450
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.divider()
                if st.button("🔄 Upload New Resume", use_container_width=True):
                    reset_analysis()

# --- 6. PAGE 2: RESUME MATCH ANALYZER ---
else:
    st.title("CareerAI Computer Science Career Assistant")
    st.write("### Scope: Optimized for Computer Science, IT, and Software Engineering pathways.")
    st.write("Analyze how well your current resume aligns with a targeted job specification.")
    if "reset_counter" not in st.session_state:
        st.session_state["reset_counter"] = 0

    reset_id = st.session_state["reset_counter"]

    with st.form("alignment_matrix_form"):
        res_file = st.file_uploader(
            "Upload Resume (PDF)", 
            type=["pdf"], 
            key=f"p2_uploader_{reset_id}"
        )
        
        job_desc = st.text_area(
            "Paste Job Description", 
            key=f"job_input_box_{reset_id}"
        )
        
        submit_btn = st.form_submit_button("Analyze Resume")

    if submit_btn and res_file and job_desc:
        with st.status("Analyzing Resume...", expanded=True) as status:
            st.write("Reading Resume...")
            raw_resume_text = extract_text(res_file.getvalue())
            resume_text = clean_text(raw_resume_text)
            job_clean = clean_text(job_desc)
            time.sleep(0.3)
            
            if not resume_text.strip() or not job_clean.strip():
                st.error("Incomplete execution data. Ensure both fields contain valid information.")
                status.update(label="Analysis Failed", state="error")
            else:
                st.write("Checking Resume Format...")
                resume_words = get_keywords(resume_text)
                job_words = get_keywords(job_clean)
                
                matched_skills = list(job_words.intersection(resume_words).intersection(SKILL_WHITELIST))
                all_requested_skills = job_words.intersection(SKILL_WHITELIST)
                gaps = list(all_requested_skills.difference(resume_words))
                time.sleep(0.3)
                
                st.write("Comparing Resume with Job Description...")
                res_vec = model.encode(resume_text, convert_to_tensor=True)
                job_vec = model.encode(job_clean, convert_to_tensor=True)
                match_percentage = round(util.cos_sim(res_vec, job_vec).item() * 100, 1)
                
                sections = detect_resume_sections(raw_resume_text)
                ats_score, suggestions = calculate_ats_score(raw_resume_text, sections)
                
                st.session_state.percentage = match_percentage
                st.session_state.skill_gaps = gaps
                st.session_state.ats_score = ats_score
                st.session_state.resume_suggestions = suggestions
                st.session_state.matched_skills = matched_skills
                st.session_state.weighted_skill_score = calculate_weighted_score(matched_skills)
                st.session_state.interview_probability = calculate_interview_probability(match_percentage, ats_score, matched_skills)
                
                if match_percentage < 50:
                    st.session_state.narrative = "Your resume currently has limited alignment with the selected job description. Adding relevant technical skills and project experience can significantly improve your match score."
                elif match_percentage < 75:
                    st.session_state.narrative = "Your resume has a good foundation but can be improved by including more relevant technical keywords."
                else:
                    st.session_state.narrative = "Excellent! Your resume closely matches the selected job description."
                    
                st.session_state.analyzed = True
                status.update(label="Analysis Complete!", state="complete", expanded=False)
if st.session_state.analyzed:
        st.subheader("📊 Alignment Metrics Hub")
        st.info(st.session_state.narrative)
        
        # --- TOP LEVEL GAUGES ---
        col1, col2, col3 = st.columns(3)
        chart_text_color = "#1E293B" 
        
        def create_gauge(title, score, bar_color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={'font': {'size': 38, 'color': chart_text_color}},
                title={'text': title, 'font': {'size': 16, 'color': chart_text_color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': chart_text_color},
                    'bar': {'color': bar_color}, 
                    'bgcolor': "#E2E8F0",
                    'bordercolor': "rgba(0,0,0,0.1)"
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font=dict(color=chart_text_color),
                height=220, 
                margin=dict(l=20, r=20, t=40, b=10)
            )
            return fig

        with col1:
            st.plotly_chart(create_gauge("Profile Match Rating", st.session_state.percentage, "#1E3A8A"), use_container_width=True)
        with col2:
            st.plotly_chart(create_gauge("Layout Check Rating", st.session_state.ats_score, "#475569"), use_container_width=True)
        with col3:
            st.plotly_chart(create_gauge("Interview Call Chance", st.session_state.interview_probability, "#0F172A"), use_container_width=True)

        st.divider()

        # --- NEW RADAR CHART COMPARISON SECTION ---
        st.subheader("🕸️ Profile Alignment Radar")
        st.plotly_chart(
            create_radar_chart(
                st.session_state.percentage, 
                st.session_state.ats_score, 
                st.session_state.interview_probability
            ), 
            use_container_width=True
        )

        st.divider()

        # --- SUGGESTIONS & MISSING SKILLS ---
        st.subheader("💡 Suggested Improvements")
        if st.session_state.resume_suggestions:
            for rec in st.session_state.resume_suggestions:
                st.error(f"❌ {rec}")
        else:
            st.success("✅ Structural verification checks passed cleanly!")

        st.subheader("🎯 Missing Technical Competencies")
        if st.session_state.skill_gaps:
            cols = st.columns(len(st.session_state.skill_gaps) if len(st.session_state.skill_gaps) < 4 else 4)
            for idx, gap in enumerate(st.session_state.skill_gaps):
                with cols[idx % 4]:
                    st.info(f"🔍 **{gap.upper()}**")
                    st.markdown(f"[Find Study Guides ↗]({get_youtube_link(gap)})")
        else:
            st.success("No missing core technical competency gaps recognized.")

        st.divider()
    # ==========================================
# PHASE 1: DYNAMIC AI RESUME REWRITER
# ==========================================
st.divider()
st.markdown("### ⚡ AI Resume Power-Rewriter")
st.caption("Transform generic bullet points into high-impact, quantified ATS achievements.")

with st.expander("✨ View Instant Bullet Point Optimizations", expanded=True):
    # Check if dynamic rewrites exist in session state
    if "ai_rewrites" in st.session_state and st.session_state.ai_rewrites:
        for idx, item in enumerate(st.session_state.ai_rewrites, 1):
            st.markdown(f"#### {idx}. Bullet Point Optimization")
            st.markdown(f"❌ **Original:** *{item.get('original', 'N/A')}*")
            st.markdown(f"✅ **AI Optimized:** *{item.get('optimized', 'N/A')}*")
            st.markdown(f"💡 **Why it works:** {item.get('reason', 'N/A')}")
            if idx < len(st.session_state.ai_rewrites):
                st.divider()
    else:
        st.info("👆 Run the resume analysis above to view dynamic AI bullet point optimizations!")
    
        # --- ACTION BUTTONS (REPORT DOWNLOAD & RESET) ---
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            pdf_bytes = create_pdf_report(
                st.session_state.percentage, 
                st.session_state.skill_gaps, 
                st.session_state.narrative
            )
            st.download_button(
                label="📥 Download Summary Report (PDF)", 
                data=pdf_bytes, 
                file_name="CareerAI_Analysis.pdf", 
                mime="application/pdf", 
                use_container_width=True
            )

        with col_btn2:
            if st.button("🔄 Upload New Resume", use_container_width=True):
                reset_analysis()
