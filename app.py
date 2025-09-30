import streamlit as st
import base64
import urllib.parse

# --- Page Config ---
st.set_page_config(
    page_title="AmanteMath by Oluwabukola Ajayi",
    page_icon="🧮",
    layout="wide"
)

# --- Background Styling (Embed image with base64) ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ Call this once with your file
set_background("assets/images/background_amantemath.png")

# --- Logo and Title ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("assets/logo.png", width=220)  # ✅ Bigger logo
with col2:
    st.title("AmanteMath")
    st.subheader("Exploring Analysis, Series, and Transforms — Simplified")
    st.caption("Love for Math ❤️")

st.markdown("---")

# --- Intro ---
st.write("""
**AmanteMath** is a simple, intuitive web app designed to make mathematics easier.  

Instead of just giving direct answers, it explains formulas, steps, and shortcuts,  
helping students truly **understand** and develop a love for math.
""")

# --- Topics Banner with Links ---
topics = {
    "Calculus": "Calculus",
    "Equations": "Equations",
    "Linear Algebra": "Linear-Algebra",
    "Series": "Series",
    "Transforms": "Transforms",
    "Real Analysis": "Real-Analysis",
    "Calculator": "Calculator"
}

banner_html = '<div class="topics-banner">'
for label, page in topics.items():
    page_url = f"/?page={urllib.parse.quote(page)}"
    banner_html += f'<a href="{page_url}" target="_self">{label}</a>'
banner_html += '</div>'

st.markdown(banner_html + """
<style>
.topics-banner {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: center;
    margin: 1.5rem 0;
    padding: 1rem 2rem;
    background: rgba(255,255,255,0.65);
    border-radius: 1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
.topics-banner a {
    text-decoration: none;
    background: #1E3A8A;
    color: #fff;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    border-radius: 2rem;
    transition: all 0.3s ease;
}
.topics-banner a:hover {
    background: #14B8A6;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# --- Topics List ---
st.markdown("### 📚 Available Topics")
st.markdown("""
1. 📈 **Calculus** — differentiation & integration.  
2. 🧮 **Equations** — algebraic, quadratic & polynomial solving.  
3. 🔢 **Linear Algebra** — vectors, matrices, determinants.  
4. 🌊 **Series** — Fourier, Power, and convergence.  
5. ⚡ **Transforms** — Laplace, Fourier transforms.  
6. 📖 **Real Analysis** — abstract mathematics, Lebesgue theory.  
7. 🧾 **Calculator** — basic & scientific calculator tools.  
""")

st.success("👉 Use the left sidebar OR the banner above to navigate between topics!")
