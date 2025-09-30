# 4_Series.py
import streamlit as st
import sympy as sp
import re
from sympy import symbols, summation, oo

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AmanteMath - Series", layout="wide")

# -----------------------------
# Header with Image + Tagline
# -----------------------------
def show_header_image():
    paths = [
        "assets/images/series.png",
        "images/series.png",
        "assets/series.png",
        "series.png",
    ]
    for p in paths:
        try:
            st.image(p, use_container_width=True)
            return
        except Exception:
            continue
    st.warning(
        "Series image not found. Tried: assets/images/series.png, images/series.png, assets/series.png, series.png"
    )

col1, col2 = st.columns([2, 5])
with col1:
    show_header_image()
with col2:
    st.title("Sequences & Series")
    st.caption("Arithmetic, Geometric, Convergence Tests, Power Series, Taylor & Maclaurin Expansions.")

st.markdown("---")

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_input(expr: str) -> str:
    """Fix basic input syntax for Sympy."""
    expr = expr.replace("^", "**")
    expr = re.sub(r'e\^([-\w\d\+\*]+)', r'exp(\1)', expr)
    return expr

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("Series Tools")
tool = st.sidebar.radio(
    "Select a topic:",
    [
        "Learn Series",
        "Arithmetic Series",
        "Geometric Series",
        "Convergence Test",
        "Power Series",
        "Taylor / Maclaurin Expansion",
    ],
)

# -----------------------------
# Learn Series
# -----------------------------
if tool == "Learn Series":
    st.header("Learn Series")

    tabs = st.tabs([
        "Introduction",
        "Arithmetic",
        "Geometric",
        "Convergence",
        "Power Series",
        "Taylor & Maclaurin"
    ])

    with tabs[0]:
        st.subheader("Introduction")
        st.write("A *series* is the sum of the terms of a sequence. Examples include arithmetic series, geometric series, and power series.")
        st.latex(r"S_n = a_1 + a_2 + \dots + a_n")

    with tabs[1]:
        st.subheader("Arithmetic Series")
        st.latex(r"S_n = \frac{n}{2}(a_1 + a_n)")

    with tabs[2]:
        st.subheader("Geometric Series")
        st.latex(r"S_n = a\frac{1-r^n}{1-r}, \quad r \neq 1")

    with tabs[3]:
        st.subheader("Convergence Tests")
        st.write("To check if a series converges, we use tests such as ratio test, root test, comparison test, etc.")

    with tabs[4]:
        st.subheader("Power Series")
        st.latex(r"\sum_{n=0}^\infty c_n (x-a)^n")

    with tabs[5]:
        st.subheader("Taylor & Maclaurin Series")
        st.latex(r"f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x-a)^n")

# -----------------------------
# Arithmetic Series Calculator
# -----------------------------
elif tool == "Arithmetic Series":
    st.header("Arithmetic Series — Calculator & Steps")

    a1 = st.number_input("First term (a1)", value=1.0)
    d = st.number_input("Common difference (d)", value=1.0)
    n = st.number_input("Number of terms (n)", min_value=1, value=5)

    an = a1 + (n - 1) * d
    Sn = n * (a1 + an) / 2

    st.latex(r"a_n = a_1 + (n-1)d")
    st.write(f"n-th term: {an}")
    st.latex(r"S_n = \frac{n}{2}(a_1 + a_n)")
    st.write(f"Sum of series: {Sn}")

# -----------------------------
# Geometric Series Calculator
# -----------------------------
elif tool == "Geometric Series":
    st.header("Geometric Series — Calculator & Steps")

    a = st.number_input("First term (a)", value=1.0)
    r = st.number_input("Common ratio (r)", value=0.5)
    n = st.number_input("Number of terms (n)", min_value=1, value=5)

    if r == 1:
        Sn = a * n
    else:
        Sn = a * (1 - r**n) / (1 - r)

    st.latex(r"S_n = a\frac{1-r^n}{1-r}, \quad r \neq 1")
    st.write(f"Sum of series: {Sn}")

# -----------------------------
# Convergence Test
# -----------------------------
elif tool == "Convergence Test":
    st.header("Convergence Test — Ratio Test")

    n = symbols("n", positive=True)
    expr_str = st.text_input("Enter general term a_n (use n)", value="1/n!")
    try:
        expr_str = preprocess_input(expr_str)
        a_n = sp.sympify(expr_str)
        ratio = sp.limit(abs(a_n.subs(n, n+1) / a_n), n, oo)
        st.write("Ratio Test Limit:")
        st.latex(sp.latex(ratio))
        if ratio < 1:
            st.success("Series converges (ratio < 1).")
        elif ratio > 1:
            st.error("Series diverges (ratio > 1).")
        else:
            st.warning("Inconclusive (ratio = 1).")
    except Exception as e:
        st.error("Error: " + str(e))

# -----------------------------
# Power Series
# -----------------------------
elif tool == "Power Series":
    st.header("Power Series Expansion")

    x = symbols("x")
    expr_str = st.text_input("Enter function (in x)", value="1/(1-x)")
    try:
        expr_str = preprocess_input(expr_str)
        f = sp.sympify(expr_str)
        series_expansion = sp.series(f, x, 0, 6)
        st.write("Power series expansion around 0 (5 terms):")
        st.latex(sp.latex(series_expansion))
    except Exception as e:
        st.error("Error: " + str(e))

# -----------------------------
# Taylor / Maclaurin Expansion
# -----------------------------
elif tool == "Taylor / Maclaurin Expansion":
    st.header("Taylor / Maclaurin Expansion")

    x = symbols("x")
    expr_str = st.text_input("Enter function (in x)", value="exp(x)")
    a = st.number_input("Expand around point a =", value=0)
    try:
        expr_str = preprocess_input(expr_str)
        f = sp.sympify(expr_str)
        series_expansion = sp.series(f, x, a, 6)
        st.write(f"Series expansion around x={a}:")
        st.latex(sp.latex(series_expansion))
    except Exception as e:
        st.error("Error: " + str(e))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("© 2025 AmanteMath | Series Module — Solved & Explained")
