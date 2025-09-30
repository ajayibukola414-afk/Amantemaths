# 5_Transforms.py
import streamlit as st
import sympy as sp
import re
from sympy import symbols, Function, summation, oo, factorial, exp, sin, cos
import math

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AmanteMath - Transforms", layout="wide")

# -----------------------------
# Header with Image + Tagline
# -----------------------------
def show_header_image():
    paths = [
        "assets/images/transforms.png",
        "images/transforms.png",
        "assets/transforms.png",
        "transforms.png",
    ]
    for p in paths:
        try:
            st.image(p, use_container_width=True)  # ✅ FIXED
            return
        except Exception:
            continue
    st.warning(
        "Transforms image not found. Tried: assets/images/transforms.png, images/transforms.png, assets/transforms.png, transforms.png"
    )

col1, col2 = st.columns([2, 5])
with col1:
    show_header_image()
with col2:
    st.title("Transforms")
    st.caption("Laplace, Inverse Laplace, Z-Transform, Fourier, Inverse Fourier — fully explained & solved step-by-step.")

st.markdown("---")

# -----------------------------
# Helper function
# -----------------------------
def latex_display(s):
    """Display LaTeX safely (string or sympy)."""
    if isinstance(s, str):
        st.latex(s)
    else:
        st.latex(sp.latex(s))

def preprocess_input(expr: str) -> str:
    """
    Fix common math input mistakes and convert to sympy-friendly syntax.
    Examples:
      cos^2 t  -> cos(t)**2
      cos**2*t -> cos(t)**2
      e^t      -> exp(t)
    """
    expr = expr.replace("^", "**")  # replace ^ with **
    expr = re.sub(r'cos\*\*(\d+)\*?t', r'cos(t)**\1', expr)
    expr = re.sub(r'sin\*\*(\d+)\*?t', r'sin(t)**\1', expr)
    expr = re.sub(r'e\*\*\(?([^)]+)\)?', r'exp(\1)', expr)  # e**x -> exp(x)
    expr = re.sub(r'e\^([-\w\d\+\*]+)', r'exp(\1)', expr)  # e^x -> exp(x)
    return expr

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("Transforms Tools")
tool = st.sidebar.radio(
    "Select a topic:",
    [
        "Learn Transforms",
        "Laplace Transform",
        "Inverse Laplace Transform",
        "Z-Transform",
        "Fourier Transform",
        "Inverse Fourier Transform",
    ],
)

# -----------------------------
# Learn Transforms (mini textbook)
# -----------------------------
if tool == "Learn Transforms":
    st.header("Learn Transforms")

    tabs = st.tabs([
        "Introduction",
        "Laplace Transform",
        "Inverse Laplace Transform",
        "Z-Transform",
        "Fourier Transform",
        "Inverse Fourier Transform"
    ])

    with tabs[0]:
        st.subheader("Introduction")
        st.write("A *transform* is like a special lens: you change a problem into another space where it’s easier to solve, then sometimes convert it back.")
        st.write("Example analogy: Imagine a messy room (the time domain). If you put items into labeled boxes (transform domain), it becomes easier to sort and work with.")
        st.write("Main transforms we’ll study:")
        st.markdown("- **Laplace Transform**: for solving differential equations.\n- **Inverse Laplace**: getting back the time function.\n- **Z-Transform**: for discrete sequences.\n- **Fourier Transform**: analyzing frequency content.\n- **Inverse Fourier**: returning to the time signal.")

    with tabs[1]:
        st.subheader("Laplace Transform")
        st.latex(r"\mathcal{L}\{f(t)\} = F(s) = \int_0^\infty e^{-st} f(t)\, dt, \quad s > \sigma_0")
        st.write("**Operational rules (short list):**")
        st.markdown(r"""
        - Linearity: $\mathcal{L}\{af+bg\} = aF + bG$  
        - Time shift: $\mathcal{L}\{u(t-t_0)f(t-t_0)\} = e^{-s t_0}F(s)$  
        - Differentiation in time: $\mathcal{L}\{f'(t)\} = sF(s) - f(0^+)$  
        - Multiplication by $t$: $\mathcal{L}\{t f(t)\} = -\frac{d}{ds}F(s)$  
        """)
        st.write("**Worked Example (step-by-step):** Find Laplace of $f(t)=t e^{2t}$.")
        st.write("**Method A (table lookup):** Use formula $\\mathcal{L}\\{t^n e^{at}\\} = \\tfrac{n!}{(s-a)^{n+1}}$. With $n=1, a=2$, result = $1/(s-2)^2$, valid for Re(s)>2.")
        st.write("**Method B (operational rule):** First compute $\\mathcal{L}\\{e^{2t}\\} = 1/(s-2)$. Then apply multiplication-by-t rule: $\\mathcal{L}\\{t e^{2t}\\} = -d/ds(1/(s-2)) = 1/(s-2)^2$.")

    with tabs[2]:
        st.subheader("Inverse Laplace Transform")
        st.write("We recover $f(t)$ from $F(s)$. Often done by partial fractions.")
        st.write("**Worked Example (step-by-step):** Find $f(t)$ if $F(s) = \\tfrac{s+2}{s^2+4s+5}$.")
        st.write("Step 1: Factor denominator: $s^2+4s+5 = (s+2)^2+1$.")
        st.write("Step 2: Rewrite $F(s) = (s+2)/((s+2)^2+1)$.")
        st.write("Step 3: Recognize standard form: $\\mathcal{L}\\{e^{-at}\\cos(bt)\\} = (s+a)/((s+a)^2+b^2)$.")
        st.latex(r"f(t) = e^{-2t}\cos(t)")

    with tabs[3]:
        st.subheader("Z-Transform")
        st.latex(r"Z\{x[n]\} = X(z) = \sum_{n=0}^\infty x[n] z^{-n}")
        st.write("**Worked Example (step-by-step):** $x[n]=(1/2)^n$. Then")
        st.write("Step 1: Write definition: $X(z)=\sum_{n=0}^\infty (1/2)^n z^{-n}$.")
        st.write("Step 2: This is a geometric series with ratio $(1/2)z^{-1}$.")
        st.write("Step 3: Sum = $1/(1-(1/2)z^{-1}) = z/(z-1/2)$, valid for |z|>1/2.")

    with tabs[4]:
        st.subheader("Fourier Transform")
        st.latex(r"\mathcal{F}\{f(t)\} = F(\omega) = \int_{-\infty}^\infty f(t) e^{-i\omega t}\, dt")
        st.write("**Worked Example (step-by-step):** $f(t)=e^{-at}, t\ge 0, a>0$. Then")
        st.write("Step 1: Definition: $F(\\omega)=\\int_0^\\infty e^{-at} e^{-i\\omega t}dt$.")
        st.write("Step 2: Combine exponents: $e^{-(a+i\\omega)t}$.")
        st.write("Step 3: Integral = $1/(a+i\\omega)$, valid for a>0.")

    with tabs[5]:
        st.subheader("Inverse Fourier Transform")
        st.latex(r"f(t) = \frac{1}{2\pi}\int_{-\infty}^\infty F(\omega) e^{i\omega t}\, d\omega")
        st.write("**Worked Example (step-by-step):** If $F(\\omega)=1/(a^2+\\omega^2)$, then")
        st.write("Step 1: Use known pair: Fourier transform of $e^{-a|t|}/(2a)$ is $1/(a^2+\\omega^2)$.")
        st.latex(r"f(t) = \frac{1}{2a} e^{-a|t|}")

# -----------------------------
# Laplace Transform calculator
# -----------------------------
elif tool == "Laplace Transform":
    st.header("Laplace Transform — Calculator & Steps")

    t, s = sp.symbols("t s", positive=True)
    func_str = st.text_input("Enter f(t) (Sympy syntax, e.g. exp(2*t)*cos(t)**2)", value="t*exp(2*t)")

    try:
        func_str = preprocess_input(func_str)
        f = sp.sympify(func_str)
        F = sp.laplace_transform(f, t, s, noconds=True)
        st.write("Laplace Transform:")
        st.latex(sp.latex(F))

        with st.expander("Short Explanation"):
            st.write("We used the definition $\\mathcal{L}\\{f(t)\\} = \\int_0^\\infty e^{-st} f(t) dt$. Sympy computed the integral symbolically.")
    except Exception as e:
        st.error("Error computing Laplace transform: " + str(e))

# -----------------------------
# Inverse Laplace calculator
# -----------------------------
elif tool == "Inverse Laplace Transform":
    st.header("Inverse Laplace Transform — Calculator & Steps")

    t, s = sp.symbols("t s", positive=True)
    func_str = st.text_input("Enter F(s) (Sympy syntax)", value="(s+2)/(s**2+4*s+5)")

    try:
        func_str = preprocess_input(func_str)
        F = sp.sympify(func_str)
        f = sp.inverse_laplace_transform(F, s, t)
        st.write("Inverse Laplace Transform:")
        st.latex(sp.latex(f))

        with st.expander("Short Explanation"):
            st.write("We decomposed into forms matching standard Laplace pairs, often using partial fractions.")
    except Exception as e:
        st.error("Error computing inverse Laplace: " + str(e))

# -----------------------------
# Z-Transform calculator
# -----------------------------
elif tool == "Z-Transform":
    st.header("Z-Transform — Calculator & Steps")

    n, z = sp.symbols("n z")
    func_str = st.text_input("Enter sequence x[n] (Sympy syntax, use n, e.g. (1/2)**n)", value="(1/2)**n")

    try:
        func_str = preprocess_input(func_str)
        x_n = sp.sympify(func_str)
        Xz = sp.summation(x_n * z**(-n), (n, 0, oo))
        st.write("Z-Transform:")
        st.latex(sp.latex(sp.simplify(Xz)))

        with st.expander("Short Explanation"):
            st.write("We used definition $Z\\{x[n]\\} = \\sum x[n] z^{-n}$.")
    except Exception as e:
        st.error("Error computing Z-transform: " + str(e))

# -----------------------------
# Fourier Transform calculator
# -----------------------------
elif tool == "Fourier Transform":
    st.header("Fourier Transform — Calculator & Steps")

    t, w = sp.symbols("t w", real=True)
    func_str = st.text_input("Enter f(t) (Sympy syntax, e.g. exp(-a*t)*(t>=0))", value="exp(-a*t)*(t>=0)")

    try:
        func_str = preprocess_input(func_str)
        f = sp.sympify(func_str)
        F = sp.fourier_transform(f, t, w)
        st.write("Fourier Transform:")
        st.latex(sp.latex(F))

        with st.expander("Short Explanation"):
            st.write("We used definition $\\mathcal{F}\\{f(t)\\} = \\int f(t) e^{-i\\omega t} dt$. Sympy computed it symbolically.")
    except Exception as e:
        st.error("Error computing Fourier transform: " + str(e))

# -----------------------------
# Inverse Fourier Transform calculator
# -----------------------------
elif tool == "Inverse Fourier Transform":
    st.header("Inverse Fourier Transform — Calculator & Steps")

    t, w = sp.symbols("t w", real=True)
    func_str = st.text_input("Enter F(ω) (Sympy syntax)", value="1/(a**2+w**2)")

    try:
        func_str = preprocess_input(func_str)
        F = sp.sympify(func_str)
        f = sp.inverse_fourier_transform(F, w, t)
        st.write("Inverse Fourier Transform:")
        st.latex(sp.latex(f))

        with st.expander("Short Explanation"):
            st.write("We used definition $f(t) = (1/2\pi)\\int F(\\omega)e^{i\\omega t} d\\omega$.")
    except Exception as e:
        st.error("Error computing inverse Fourier transform: " + str(e))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("© 2025 AmanteMath | Transforms Module — Solved & Explained")
