# 6_Real_Analysis.py
import os
from pathlib import Path
import streamlit as st
import sympy as sp
from sympy import symbols, oo
import math

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AmanteMath - Real Analysis", layout="wide")

# -----------------------------
# Header with Image + Tagline
# -----------------------------
def show_header_image():
    # Try a few locations; only call st.image if the file exists to avoid MediaFileStorageError
    paths = [
        "assets/images/real_analysis.png",
        "images/real_analysis.png",
        "assets/real_analysis.png",
        "real_analysis.png",
    ]
    for p in paths:
        if Path(p).exists():
            st.image(p, use_container_width=True)  # patched here
            return
    st.warning(
        "Real Analysis image not found. Tried: assets/images/real_analysis.png, images/real_analysis.png, assets/real_analysis.png, real_analysis.png"
    )

col1, col2 = st.columns([2, 6])
with col1:
    show_header_image()
with col2:
    st.title("Real Analysis")
    st.caption("Foundations: sequences & series, continuity & differentiability, integration, topology basics, Lebesgue measure — deep, step-by-step, beginner-friendly.")

st.markdown("---")

# -----------------------------
# Helper functions & sympy locals
# -----------------------------
SAFE_LOCALS = {
    # common functions
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "pi": sp.pi, "E": sp.E,
    # allow power with symbol names
    "oo": sp.oo, "I": sp.I
}

def safe_sympify(expr_str, locals_map=SAFE_LOCALS):
    """Sympify with a safe locals map and helpful error message."""
    try:
        return sp.sympify(expr_str, locals=locals_map)
    except Exception as e:
        raise ValueError(
            "Could not parse expression. Use sympy-like syntax with ** for powers, e.g., cos(t)**2, exp(2*t). "
            "Make sure functions are lowercase (sin, cos, exp) and use '*' explicitly for multiplication. "
            f"Original error: {e}"
        )

def latex_display(s):
    """Display LaTeX safely (string or sympy)."""
    if isinstance(s, str):
        st.latex(s)
    else:
        st.latex(sp.latex(s))

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("Real Analysis Tools")
tool = st.sidebar.radio(
    "Select a topic:",
    [
        "Learn Real Analysis",
        "Sequences & Series",
        "Continuity & Differentiability",
        "Integration",
        "Open Balls & Open Sets",
        "Lebesgue Measure",
    ],
)

# -----------------------------
# Learn Real Analysis (mini-textbook)
# -----------------------------
if tool == "Learn Real Analysis":
    st.header("Learn Real Analysis — Mini Textbook (detailed & step-by-step)")

    tabs = st.tabs([
        "Introduction & Foundations",
        "Sequences & Series",
        "Continuity & Differentiability",
        "Integration",
        "Open Balls & Open Sets",
        "Lebesgue Measure",
    ])

    # ---------- Intro ----------
    with tabs[0]:
        st.subheader("1 — Introduction & Foundations")
        st.write(
            "Real Analysis is the rigorous study of real numbers, sequences, series, and functions. "
            "It explains why calculus works and builds the formal language for limits, continuity, derivatives, and integrals."
        )

        st.markdown("**Real numbers**")
        st.write(
            "Real numbers (ℝ) include rationals (fractions) and irrationals (like π and √2). "
            "Key property: the *completeness axiom* — every nonempty set that is bounded above has a least upper bound (supremum)."
        )

        st.markdown("**Completeness — step-by-step example**")
        st.write("Set S = { x in ℝ : x^2 < 2 }.")
        st.write("Members of S include 1, 1.4, 1.41, 1.414, … These approach √2 but never exceed √2.")
        st.write("Supremum of S is √2. Completeness guarantees that this least upper bound exists in ℝ.")

        st.markdown("**Why completeness matters**")
        st.write("It ensures limits of Cauchy sequences exist in ℝ (no 'holes'). This is the foundation for rigorous calculus.")

    # ---------- Sequences & Series ----------
    with tabs[1]:
        st.subheader("2 — Sequences & Series (concepts & worked examples)")
        st.write("A sequence is an ordered list (a₁, a₂, a₃, …). A series is the sum of terms, ∑ a_n.")
        st.markdown("**Convergence (sequence)**")
        st.write(
            "Definition: a_n → L means for every ε>0 there exists N such that |a_n − L| < ε for all n ≥ N.\n\n"
            "Example: a_n = 1/n. Show a_n → 0: given ε>0 choose N = ⌈1/ε⌉ so for n ≥ N we have 1/n ≤ 1/N ≤ ε."
        )
        st.markdown("**Series & partial sums (example)**")
        st.write(
            "Series ∑ 1/2^n. Partial sums: S1 = 1/2, S2 = 1/2+1/4=3/4, S3 = 7/8, ... In general S_n = 1 - (1/2)^n → 1 as n→∞."
        )

        st.markdown("**Tests & intuition (short guide)**")
        st.write(
            "- Geometric series: ∑ ar^{n} converges if |r|<1 to a/(1−r).\n"
            "- p-series: ∑ 1/n^p converges iff p>1.\n"
            "- Comparison, ratio, root, alternating (Leibniz) tests — all are tools to decide convergence."
        )

        st.markdown("**Worked problem fully solved**")
        st.write("Series: ∑_{n=1}^\infty 1/2^n. Step 1: recognize geometric with a=1/2, r=1/2. Sum = a/(1-r) = (1/2)/(1/2)=1.")

    # ---------- Continuity & Differentiability ----------
    with tabs[2]:
        st.subheader("3 — Continuity & Differentiability (full explanation)")
        st.write("Continuity at x=a: lim_{x→a} f(x) = f(a).")
        st.write("Differentiability at x=a: the limit of the difference quotient exists:")
        st.latex(r"f'(a)=\lim_{h\to 0}\frac{f(a+h)-f(a)}{h}")
        st.markdown("**Why continuity doesn't always imply differentiability**")
        st.write("Example: f(x)=|x| is continuous everywhere but not differentiable at 0 because slopes from left and right differ.")
        st.markdown("**Worked example: f(x)=x^2 at x=2**")
        st.write("Compute limit: lim_{x→2} x^2 = 4. f(2)=4 → continuous.")
        st.write("Compute derivative: f'(x)=2x, so f'(2)=4. Using difference quotient: ((2+h)^2−4)/h = (4+4h+h^2−4)/h = 4+h → 4 as h→0.")

    # ---------- Integration ----------
    with tabs[3]:
        st.subheader("4 — Integration (Riemann & Improper integrals)")
        st.write("Riemann integral ∫_a^b f(x) dx is the limit of Riemann sums (area under curve).")
        st.markdown("**Fundamental Theorem of Calculus (two parts)**")
        st.write("1) If F is antiderivative of f, ∫_a^b f(x) dx = F(b) − F(a).")
        st.write("2) If f is integrable, d/dx ∫_a^x f(t) dt = f(x).")
        st.markdown("**Worked example: ∫_0^2 x dx**")
        st.write("Antiderivative F(x)=x^2/2. Evaluate: F(2)-F(0)=2.")
        st.markdown("**Improper integral example: ∫_1^∞ 1/x^2 dx**")
        st.write("Compute ∫_1^b 1/x^2 dx = [−1/x]_1^b = 1 − 1/b. Let b→∞ → 1. So convergent, value 1.")

    # ---------- Open Balls & Open Sets ----------
    with tabs[4]:
        st.subheader("5 — Open Balls & Open Sets (topology basics)")
        st.write("Open ball in ℝ^n: B(a,ε) = { x : ||x−a|| < ε }.")
        st.write("A set U is open if for every x∈U there exists ε>0 such that B(x,ε) ⊂ U.")
        st.markdown("**Example: U=(0,1) in ℝ**")
        st.write("Pick x=0.5. Choose ε=0.1. Then B(0.5,0.1)=(0.4,0.6) which is contained in (0,1). Thus (0,1) is open.")
        st.write("Closed sets: complements of open sets. Boundary points: neither interior nor exterior.")

    # ---------- Lebesgue Measure ----------
    with tabs[5]:
        st.subheader("6 — Lebesgue Measure (outer measure, properties, examples)")
        st.write("Lebesgue outer measure m*(E) = inf { ∑ length(I_k) : E ⊆ ⋃ I_k } where I_k are intervals.")
        st.write("Key properties: nonnegativity, monotonicity (A⊂B ⇒ m*(A) ≤ m*(B)), countable subadditivity.")
        st.markdown("**Worked example: E=[0,1]**")
        st.write("Cover [0,1] by two intervals [0,0.5] and [0.5,1] — total length 0.5+0.5=1. The infimum over covers is 1, so m*([0,1])=1.")
        st.write("Carathéodory criterion (sketch): A set E is Lebesgue measurable if for any A, m*(A) = m*(A∩E) + m*(A∩E^c).")

    st.markdown("---")
    st.write("This mini-textbook tab is intentionally long and explicit. Use the other sidebar tabs to solve specific problems with step-by-step computations.")

# -----------------------------
# Sequences & Series (calculator + step-by-step)
# -----------------------------
elif tool == "Sequences & Series":
    st.header("Sequences & Series — Calculator & Step-by-step")

    st.write("Instructions: Enter a sequence formula in variable n, e.g., 1/n, (1/2)**n, n/(n+1). Use sympy syntax; powers with **, functions sin, cos, exp etc.")

    seq_str = st.text_input("Sequence a_n (in n)", value="1/n")
    max_terms = st.number_input("Show first how many terms?", min_value=3, max_value=200, value=10)
    check_limit = st.checkbox("Check limit as n→∞ (symbolic when possible)", value=True)

    try:
        n = symbols("n", integer=True, positive=True)
        a_n = safe_sympify(seq_str)
        st.write("First terms (a_1 ... a_N):")
        terms = [sp.N(a_n.subs(n, i)) for i in range(1, int(max_terms) + 1)]
        st.write(terms)

        if check_limit:
            st.write("Symbolic limit (if computable):")
            try:
                lim = sp.limit(a_n, n, sp.oo)
                st.write("lim_{n→∞} a_n = ", lim)
                if lim.is_real:
                    if lim == 0:
                        st.success("Sequence tends to 0 (useful for series tests).")
                else:
                    st.info("Limit computed: " + str(lim))
            except Exception as e:
                st.error("Could not compute symbolic limit: " + str(e))

        st.markdown("**Series (partial sums) — interactive**")
        series_on = st.checkbox("Also treat as series ∑ a_n and compute partial sums", value=False)
        if series_on:
            N_sum = st.number_input("Compute partial sums up to N", min_value=5, max_value=2000, value=20)
            try:
                partials = []
                S = 0
                for k in range(1, int(N_sum) + 1):
                    S = S + sp.N(a_n.subs(n, k))
                    partials.append(float(S))
                st.write("Partial sums S_1..S_N:")
                st.write(partials[:200])
                st.write("Approximate limit (last partial sum):", partials[-1])
                st.write("If this is a common series (geometric, p-series), check known tests:")
                # quick detection for geometric
                try:
                    ratio_expr = sp.simplify(sp.simplify(a_n.subs(n, 2)/a_n.subs(n,1)))
                except Exception:
                    ratio_expr = None
                st.write("Note: For rigorous decisions, use ratio, root, comparison tests manually if needed.")
            except Exception as e:
                st.error("Error computing partial sums: " + str(e))

        with st.expander("Full worked explanation for the entered sequence (step-by-step)"):
            st.write("We list the symbolic steps used to analyze a sequence/series:")
            st.write("1) We interpret your formula a_n as a function of integer n.")
            st.write("2) We compute first N terms by evaluating a_n at n=1..N.")
            st.write("3) To decide convergence as n→∞, we attempt a symbolic limit (sp.limit).")
            st.write("4) For series, we compute partial sums S_N = ∑_{k=1}^N a_k and inspect S_N as N grows.")
            st.write("Examples & tips: For a geometric sequence (1/2)**n, the series ∑ (1/2)^n converges to 1. For p-series 1/n^p, convergence depends on p>1.")
    except Exception as e:
        st.error("Error parsing sequence: " + str(e))

# -----------------------------
# Continuity & Differentiability (calculator + step-by-step)
# -----------------------------
elif tool == "Continuity & Differentiability":
    st.header("Continuity & Differentiability — Calculator & Steps")
    st.write("Enter f(x) with sympy syntax. Example: x**2, sin(x), Abs(x) (use Abs for absolute value).")
    fx_str = st.text_input("f(x) = ", value="x**2")
    point = st.number_input("Point a to test continuity/differentiability at", value=2.0)

    try:
        x = symbols("x", real=True)
        f = safe_sympify(fx_str)
        # continuity test
        st.write("Compute limit as x→a of f(x):")
        try:
            lim_left = sp.limit(f, x, point, dir='-')
            lim_right = sp.limit(f, x, point, dir='+')
            lim = sp.limit(f, x, point)
            st.write("left limit:", lim_left)
            st.write("right limit:", lim_right)
            st.write("two-sided limit:", lim)
            fa = f.subs(x, point)
            st.write("f(a) =", fa)
            if lim == fa:
                st.success("Function is continuous at a (limit equals function value).")
            else:
                st.error("Function is NOT continuous at a (limit differs from f(a)).")
        except Exception as e:
            st.error("Could not compute limit: " + str(e))

        # differentiability
        with st.expander("Check differentiability (compute derivative and difference quotient)"):
            try:
                derivative = sp.diff(f, x)
                st.write("Symbolic derivative f'(x) =", derivative)
                st.write("Evaluate derivative at a: f'(a) =", derivative.subs(x, point))
                # show difference quotient limit
                h = symbols("h")
                diffq = (f.subs(x, point + h) - f.subs(x, point)) / h
                st.write("Difference quotient (limit h→0):")
                st.write("Expression:", sp.simplify(diffq))
                st.write("Limit of difference quotient:", sp.limit(diffq, h, 0))
                st.write("If this limit equals derivative at a, function is differentiable at a.")
            except Exception as e:
                st.error("Could not compute derivative/difference quotient: " + str(e))

        st.markdown("**Notes & input tips**")
        st.write("- Use `Abs(x)` for absolute value; use `cos(x)**2` for cos²(x).")
        st.write("- If the expression cannot be symbolically differentiated, we attempt numeric checks (not implemented by default).")
    except Exception as e:
        st.error("Error parsing function: " + str(e))

# -----------------------------
# Integration (calculator + step-by-step)
# -----------------------------
elif tool == "Integration":
    st.header("Integration — Calculator & Steps")
    st.write("Enter f(x) in sympy syntax. For definite integrals put bounds a and b. Use 'oo' for infinity if needed.")

    fx_str = st.text_input("f(x) = ", value="x")
    a_str = st.text_input("Lower bound a (use -oo for -∞)", value="0")
    b_str = st.text_input("Upper bound b (use oo for ∞)", value="2")

    try:
        x = symbols("x", real=True)
        f = safe_sympify(fx_str)
        a = safe_sympify(a_str)
        b = safe_sympify(b_str)

        st.write("Attempting integration:")
        try:
            res = sp.integrate(f, (x, a, b))
            st.success("Integral ∫_a^b f(x) dx = " + str(res))
        except Exception as e:
            st.error("Could not compute definite integral: " + str(e))

        with st.expander("Step-by-step reasoning (conceptual)"):
            st.write("1) We attempt symbolic integration using sympy's integrate().")
            st.write("2) For improper integrals (bounds involve oo), sympy handles limits automatically.")
            st.write("3) For elementary functions, antiderivatives are computed. Example: ∫ x dx = x^2/2.")
            st.write("4) If integration fails symbolically, numeric approximations can be used (not implemented here).")

    except Exception as e:
        st.error("Error parsing inputs: " + str(e))

# -----------------------------
# Open Balls & Open Sets (topology checker)
# -----------------------------
elif tool == "Open Balls & Open Sets":
    st.header("Open Balls & Open Sets — Intuition & Test")
    st.write("We demonstrate openness of intervals in ℝ. Enter an interval (a,b).")

    a_val = st.number_input("Left endpoint a", value=0.0)
    b_val = st.number_input("Right endpoint b", value=1.0)
    x_val = st.number_input("Pick a point x in (a,b)", value=0.5)
    eps = st.number_input("Choose ε > 0", value=0.1)

    if a_val < x_val < b_val:
        st.write("Check whether B(x,ε) ⊂ (a,b).")
        left = x_val - eps
        right = x_val + eps
        if left > a_val and right < b_val:
            st.success("Yes! B(x,ε) is contained in (a,b). This shows openness.")
        else:
            st.warning("For this ε, ball extends beyond interval. But since x is interior, you can choose smaller ε.")
    else:
        st.error("x must lie strictly inside (a,b).")

    with st.expander("Conceptual background"):
        st.write("Open ball in ℝ: B(x,ε) = (x−ε, x+ε). A set is open if every point has such an ε-ball inside the set.")

# -----------------------------
# Lebesgue Measure (demonstration)
# -----------------------------
elif tool == "Lebesgue Measure":
    st.header("Lebesgue Measure — Demonstrations")
    st.write("We show Lebesgue outer measure for simple sets like intervals.")

    a_val = st.number_input("Interval left a", value=0.0, key="measure_a")
    b_val = st.number_input("Interval right b", value=1.0, key="measure_b")
    if a_val <= b_val:
        length = b_val - a_val
        st.success(f"Lebesgue measure of interval [{a_val},{b_val}] is {length}.")
    else:
        st.error("a must be ≤ b.")

    with st.expander("Notes & theory"):
        st.write("Lebesgue measure generalizes length to complicated sets. For intervals, measure is simply length.")
        st.write("Key property: countable additivity — if intervals are disjoint, measure of union is sum of measures.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("© 2025 AmanteMath | Real Analysis Module — Solved & Explained")
