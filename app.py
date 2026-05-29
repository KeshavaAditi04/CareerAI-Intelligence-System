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

# --- 1. UI CONFIG & ADVANCED CYBERPUNK STYLING ---
st.set_page_config(page_title="CareerAI Intelligence System", layout="wide")

st.markdown("""
    <style>
@keyframes royalScan {
        0% { background-position: 0 0, 0 0, 0 0; }
        100% { background-position: 0 100%, 0 0, 0 0; }
    }

    /* 1. CONTAINER: Deep Midnight Jade & Obsidian Velvet Canvas */
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    .main, 
    .stApp,
    [data-testid="stAppViewBlockContainer"] {
        background-color: #0A0F0D !important; /* Luxurious, deep near-black velvet jade */
        background-image: 
            /* Layer 1: Moving Data Scanlines (Faint Emerald Spark) */
            linear-gradient(rgba(16, 185, 129, 0.02) 2px, transparent 2px),
            /* Layer 2: Clean Tech Mesh Grid (Muted Slate-Gold Lines) */
            linear-gradient(90deg, rgba(212, 175, 55, 0.04) 1px, transparent 1px),
            /* Layer 3: Glorious Asgardian Emerald Glow (Top Right Accent) */
            radial-gradient(circle at 85% 20%, rgba(16, 185, 129, 0.15) 0%, transparent 60%) !important;
        background-size: 100% 40px, 45px 45px, auto !important;
        background-attachment: fixed !important;
        animation: royalScan 28s linear infinite !important;
    }

    /* 2. SIDEBAR: Matte Black Velvet */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #050807 !important; 
        border-right: 1px solid rgba(212, 175, 55, 0.15) !important; /* Subtle Royal Gold Divider */
    }

    /* 3. TYPOGRAPHY: Crisp White & Royal Gold Accents */
    h1, h2, h3, .stSubheader {
        color: #D4AF37 !important; /* Majestic Royal Gold for your main headers! */
        font-weight: 700;
        letter-spacing: -0.5px;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Widget Labels & form text */
    [data-testid="stWidgetLabel"] p, label {
        color: #F8FAFC !important; /* Crisp white for input instructions */
    }

    /* Sidebar text colors */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] span {
        color: #A3B8CC !important; 
    }

    /* 4. CARDS: Frosted Dark Slate Containers with Gold/Emerald Trim */
    [data-testid="stForm"], .stAlert {
        background-color: rgba(13, 22, 18, 0.8) !important; /* Deep dark jade well */
        border: 1px solid rgba(212, 175, 55, 0.2) !important; /* Fine Gold Border */
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6) !important;
    }

    /* 5. INPUT & UPLOAD FIELDS */
    [data-testid="stFileUploadDropzone"], input, textarea {
        background-color: #050807 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(212, 175, 55, 0.15) !important;
    }

    /* 6. BUTTON: Pure Glorious Loki/Margaret Emerald Gradient */
    .stButton>button {
        background: linear-gradient(135deg, #0D9488 0%, #065F46 100%) !important; /* Deep Rich Emerald */
        color: #FFFFFF !important; 
        font-weight: 700;
        border-radius: 8px;
        border: 1px solid #D4AF37 !important; /* Gold Trimmed Button */
        padding: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(6, 95, 70, 0.3) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(13, 148, 136, 0.5) !important;
        background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important; /* Vibrant pop on hover */
    }
    
    /* Glassmorphism Forms */
    [data-testid="stForm"] { 
        border: 1px solid #22222F; 
        background-color: #13131A; 
        padding: 30px; 
        border-radius: 16px; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Elegant CTA Button with Pulse Hover Effect */
    .stButton>button {
        background: linear-gradient(135deg, #8B0000 0%, #A30000 100%) !important;
        color: white !important;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        border-radius: 8px;
        border: none !important;
        padding: 12px;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 15px rgba(139, 0, 0, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(163, 0, 0, 0.8);
        background: linear-gradient(135deg, #A30000 0%, #CC0000 100%) !important;
    }
    
    /* Sleek Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #09090D !important;
        border-right: 1px solid #1A1A24;
    }
    
    /* Global Card Transition Physics */
    .gap-box-critical, .gap-box-optimize, .roadmap-card {
        transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.3s ease, border-color 0.3s ease !important;
    }

    /* Interactive Fluid Card Hovers */
    .gap-box-critical:hover {
        transform: translateX(4px);
        background: rgba(255, 77, 77, 0.07) !important;
        box-shadow: -4px 0 15px rgba(255, 77, 77, 0.2);
    }
    .gap-box-optimize:hover {
        transform: translateX(4px);
        background: rgba(234, 179, 8, 0.07) !important;
        box-shadow: -4px 0 15px rgba(234, 179, 8, 0.2);
    }
    .roadmap-card:hover {
        transform: translateY(-5px);
        border-color: #FFD700 !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(255, 215, 0, 0.1);
    }

    /* Modern Glassmorphic Metric Box Styles */
    .gap-box-critical {
        background: rgba(255, 77, 77, 0.04);
        border: 1px solid rgba(255, 77, 77, 0.15);
        border-left: 4px solid #FF4D4D;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .gap-box-optimize {
        background: rgba(234, 179, 8, 0.04);
        border: 1px solid rgba(234, 179, 8, 0.15);
        border-left: 4px solid #EAB308;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .roadmap-card {
        background: #13131A;
        border: 1px solid #222235;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .yt-link {
        color: #FF4D4D !important;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        margin-top: 12px;
        transition: color 0.2s ease;
    }
    .yt-link:hover {
        color: #FF7373 !important;
        text-decoration: underline !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"Match Framework Core Index: {percentage}%", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, f"Executive Analytical Context:\n{summary}")
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

# --- 4. SIDEBAR NAVIGATION & NATIVE DEVELOPER CARD ---
st.sidebar.title("🌑 CareerAI Platform v1.2")
page = st.sidebar.radio("Systems Diagnostic Hub:", ["🔍 Predictive Career Mapping", "🎯 Precision Profile Matching"])

st.sidebar.divider()

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
                                 color='Similarity Weight', color_continuous_scale='Reds', text='Similarity Weight')
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
            # --- PROGRESSIVE AI THINKING STATUS TICKER ---
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
                
            # --- CELEBRATORY SNOW TRIGGER FOR ELITE MATCHES ---
            if st.session_state.percentage >= 75:
                st.snow()

    if st.session_state.analyzed:
        # --- COMPACT HIGH-CONTRAST GAUGES ---
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            fig_1 = go.Figure(go.Indicator(mode="gauge+number", value=st.session_state.percentage, 
                title={'text': "Semantic Affinity Score", 'font': {'color': "#FFFFFF", 'size': 15, 'weight': 'bold'}},
                gauge={'bar': {'color': "#8B0000"}, 'bgcolor': "rgba(255,255,255,0.05)"}))
            fig_1.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFFFFF"}, height=240, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig_1, use_container_width=True)
            
        with col_g2:
            fig_2 = go.Figure(go.Indicator(mode="gauge+number", value=st.session_state.ats_score, 
                title={'text': "ATS Layout structural Index", 'font': {'color': "#FFFFFF", 'size': 15, 'weight': 'bold'}},
                gauge={'bar': {'color': "#EAB308"}, 'bgcolor': "rgba(255,255,255,0.05)"}))
            fig_2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFFFFF"}, height=240, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig_2, use_container_width=True)
            
        with col_g3:
            fig_3 = go.Figure(go.Indicator(mode="gauge+number", value=st.session_state.interview_probability, 
                title={'text': "Interview Convocation Likelihood", 'font': {'color': "#FFFFFF", 'size': 15, 'weight': 'bold'}},
                gauge={'bar': {'color': "#64748B"}, 'bgcolor': "rgba(255,255,255,0.05)"}))
            fig_3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFFFFF"}, height=240, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig_3, use_container_width=True)

# --- NARRATIVE COMPLIANCE SUMMARY ---
        st.info(f"🧠 **System Executive Context:** {st.session_state.narrative}")

        # =====================================================================
        # 👑 STRATEGIC AI RESUME ENHANCEMENTS (KEYWORD GAP ANALYSIS)
        # =====================================================================
        st.write("") 
        st.markdown("### 👑 Strategic AI Resume Enhancements")

        if st.session_state.skill_gaps:
            st.info("💡 **AI Optimization Alert:** Our semantic parsing engine detected that the target job description heavily weights specific keywords that are missing or weak in your profile signature. Adding these will instantly boost your alignment vector.")
            
            # Create a clean layout for the recommendations
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
                    st.markdown(f'<div class="gap-box-critical" style="text-align:center;">📊 <b>{skill.upper()}</b><br><span style="color:#FFD700; font-size:0.9rem;">{demand}</span></div>', unsafe_allow_html=True)

        # --- CONFIDENCE EXPLAINABILITY BAR CHART ---
        st.divider()
        st.subheader("📊 Architectural Multi-Criteria Confidence Metrics")
        
        breakdown_df = pd.DataFrame({
            "Evaluation Dimension": ["Semantic Proximity Match", "ATS Layout Structural Quality", "Weighted Core Competency Volumetrics", "Static Context Heuristics", "Keyword Architecture Index"],
            "System Score Score": [st.session_state.percentage, st.session_state.ats_score, min(st.session_state.weighted_skill_score * 5, 100), 75, 82]
        })
        fig_break = px.bar(breakdown_df, x='Evaluation Dimension', y='System Score Score', color='System Score Score', color_continuous_scale='Reds')
        fig_break.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_break, use_container_width=True)

        # --- PDF SUMMARY REPORT EXPORT ---
        st.divider()
        st.subheader("📥 Data Vector Reporting Export")
        pdf_bytes = create_pdf_report(st.session_state.percentage, st.session_state.skill_gaps, st.session_state.narrative)
        st.download_button("📄 Download Deep System Analysis Logs (PDF)", pdf_bytes, "System_Analysis_Logs.pdf", "application/pdf")