# 2_Equations.py

import streamlit as st
import sympy as sp
import re
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AmanteMath - Equations", layout="wide")

# -----------------------------
# Header with Image + Tagline
# -----------------------------
def show_header_image():
    paths = [
        "assets/images/equations.png",
        "images/equations.png",
        "assets/equations.png",
        "equations.png",
    ]
    for p in paths:
        try:
            st.image(p, use_column_width=True)
            return
        except Exception:
            continue
    st.warning(
        "Equations image not found. "
        "Tried: assets/images/equations.png, images/equations.png, assets/equations.png, equations.png"
    )

col1, col2 = st.columns([2, 5])
with col1:
    show_header_image()
with col2:
    st.title("Equations")
    st.caption("Solve linear, quadratic, simultaneous, and polynomial equations — step by step.")

st.markdown("---")

# -----------------------------
# Parser helpers
# -----------------------------
x, y, z = sp.symbols("x y z")
transformations = (standard_transformations + (implicit_multiplication_application,))

_SUPER_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

def replace_superscripts(s: str) -> str:
    """
    Turns x² -> x^2, y³ -> y^3 etc., then later we'll convert ^ to **.
    """
    return re.sub(
        r"([a-zA-Z])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda m: f"{m.group(1)}^{m.group(2).translate(_SUPER_MAP)}",
        s,
    )

# phrases to strip from word problems (longest first)
INSTRUCTION_PHRASES = [
    r"solve\s+for\s+[a-z]",
    r"find\s+the\s+value\s+of\s+[a-z]",
    r"for\s+what\s+value\s+of\s+[a-z]",
    r"value\s+of\s+[a-z]",
    r"determine\s+the\s+value\s+of\s+[a-z]",
    r"find\s+all\s+real\s+solutions",
    r"find\s+all\s+real\s+roots",
    r"find\s+all\s+solutions",
    r"determine\s+[a-z]",
    r"solve\s+for",
    r"work\s+out",
    r"such\s+that",
    r"subject\s+to",
    r"what\s+is",
    r"what's",
    r"satisfying",
    r"for\s+which",
    r"find\s+[a-z]",
    r"determine",
    r"calculate",
    r"evaluate",
    r"compute",
    r"obtain",
    r"solve",
    r"show\s+that",
    r"prove\s+that",
    r"when",
    r"where",
    r"\bdoes\b",
]

# word-to-symbol replacements
WORD_TO_SYMBOL = [
    (r"\bequals?\b", "="),
    (r"\bis\s+equal\s+to\b", "="),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\bmultiplied\s+by\b", "*"),
    (r"\btimes\b", "*"),
    (r"\bdivided\s+by\b", "/"),
    (r"\bover\b", "/"),
    (r"\bto\s+the\s+power\s+of\s+(\d+)\b", r"**\1"),
    (r"\bsquared\b", "**2"),
    (r"\bcubed\b", "**3"),
]

WORDS_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}

def clean_keywords(text: str) -> str:
    """
    Convert a sentence-like math problem to a clean algebraic string.
    - Lowercases
    - Removes instruction phrases (solve, find, when, where, etc.)
    - Replaces number words and operator words
    - Handles unicode superscripts and '^' -> '**'
    """
    if not isinstance(text, str):
        return ""

    s = text.strip().lower()
    s = replace_superscripts(s)

    # remove instruction phrases (longest first to avoid partial overlaps)
    for pat in sorted(INSTRUCTION_PHRASES, key=len, reverse=True):
        s = re.sub(pat, " ", s)

    # replace number words
    for w, n in WORDS_TO_NUM.items():
        s = re.sub(rf"\b{w}\b", n, s)

    # map math words to operators
    for pat, repl in WORD_TO_SYMBOL:
        s = re.sub(pat, repl, s)

    # remove punctuation that isn't math-relevant
    s = re.sub(r"[,:;?]", " ", s)

    # normalize powers: ^ -> **
    s = s.replace("^", "**")

    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_equation_text(text: str) -> str:
    """
    Ensure we have an equation string 'LHS = RHS'.
    If '=' is absent, assume '= 0' (so '2x+5' becomes '2x+5 = 0').
    """
    s = clean_keywords(text)
    if "=" not in s:
        s = s + " = 0"
    parts = s.split("=")
    lhs = parts[0].strip()
    rhs = "=".join(parts[1:]).strip() if len(parts) > 1 else "0"
    return f"{lhs} = {rhs}"

def parse_side(expr_str: str):
    """
    Parse a single algebraic side with implicit multiplication (so '2x' works).
    """
    return parse_expr(
        expr_str,
        transformations=transformations,
        local_dict={"x": x, "y": y, "z": z},
    )

FRIENDLY_ERROR = (
    "Wrong question/syntax. Please enter a valid equation, e.g., 2x + 5 = 0 "
    "or 'solve for x: two x plus five equals zero'."
)

def display_equation_solution(eq_obj, sols, var_symbols):
    """Render the equation and its solutions using LaTeX only."""
    st.latex(sp.latex(eq_obj))

    # Handle systems solution as dict, or list/tuple for single variable
    if isinstance(sols, dict):
        if not sols:
            st.info("No solution found (or infinitely many).")
            return False
        st.success("Solution:")
        for k, v in sols.items():
            st.latex(rf"{sp.latex(k)} = {sp.latex(v)}")
        return True

    # Normalize solution container
    if not isinstance(sols, (list, tuple)):
        sols = [sols]

    if len(sols) == 0:
        st.info("No solution found (or infinitely many).")
        return False

    if len(var_symbols) == 1:
        v = list(var_symbols)[0]
        st.success("Solutions:")
        for s in sols:
            st.latex(rf"{sp.latex(v)} = {sp.latex(s)}")
    else:
        st.success("Solutions:")
        for s in sols:
            st.latex(sp.latex(s))
    return True

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("Equation Tools")
tool = st.sidebar.radio(
    "Select a tool:",
    ["Learn Equations", "Linear Equation", "Quadratic Equation", "Simultaneous Equations", "Polynomial Equation"]
)

# -----------------------------
# Learn Equations (clean LaTeX)
# -----------------------------
if tool == "Learn Equations":
    st.header("Learn Equations")

    tabs = st.tabs([
        "Introduction", "Linear", "Quadratic",
        "Simultaneous", "Polynomial", "Tips & References"
    ])

    with tabs[0]:
        st.subheader("Introduction")
        st.write(
            "Equations assert that two expressions are equal. "
            "Solving an equation means finding the values of the variables that make the equality true."
        )
        st.write("**Common types**")
        st.write("- Linear (degree 1)")
        st.write("- Quadratic (degree 2)")
        st.write("- Simultaneous (systems)")
        st.write("- Polynomial (degree ≥ 3)")

    with tabs[1]:
        st.subheader("Linear Equations")
        st.write("**Standard Form**")
        st.latex(r"ax + b = 0 \quad (a \neq 0)")
        st.write("**Solution**")
        st.latex(r"x = -\frac{b}{a}")
        st.write("**Worked example**")
        st.latex(r"2x + 5 = 0 \;\Rightarrow\; 2x = -5 \;\Rightarrow\; x = -\tfrac{5}{2}")

    with tabs[2]:
        st.subheader("Quadratic Equations")
        st.write("**Standard Form**")
        st.latex(r"ax^2 + bx + c = 0 \quad (a \neq 0)")
        st.write("**Quadratic Formula**")
        st.latex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
        st.write("**Discriminant**")
        st.latex(r"\Delta = b^2 - 4ac")
        st.write("- \( \Delta > 0 \): two real distinct roots")
        st.write("- \( \Delta = 0 \): one real repeated root")
        st.write("- \( \Delta < 0 \): complex conjugate roots")
        st.write("**Worked (factoring)**")
        st.latex(r"x^2 + 5x + 6 = 0 \;\Rightarrow\; (x+2)(x+3)=0 \;\Rightarrow\; x=-2,-3")
        st.write("**Worked (formula)**")
        st.latex(r"3x^2 - 2x - 1 = 0")
        st.latex(r"\Delta = (-2)^2 - 4(3)(-1) = 16")
        st.latex(r"x = \frac{-(-2) \pm \sqrt{16}}{2\cdot 3} = \frac{2 \pm 4}{6} \Rightarrow x=1,\; -\tfrac{1}{3}")

    with tabs[3]:
        st.subheader("Simultaneous Equations")
        st.write("We solve systems like")
        st.latex(r"\begin{cases} a_1 x + b_1 y = c_1 \\ a_2 x + b_2 y = c_2 \end{cases}")

        st.write("**Method 1: Substitution**")
        st.latex(r"\begin{cases} 2x + y = 10 \\ x - y = 3 \end{cases}")
        st.latex(r"x = y + 3")
        st.latex(r"2(y+3) + y = 10 \;\Rightarrow\; 3y + 6 = 10 \;\Rightarrow\; y=\tfrac{4}{3},\; x=\tfrac{13}{3}")

        st.write("**Method 2: Elimination**")
        st.latex(r"\begin{cases} 2x + y = 10 \\ x - y = 3 \end{cases}")
        st.latex(r"\text{Add: } 3x = 13 \Rightarrow x=\tfrac{13}{3},\;\text{ then } y=\tfrac{4}{3}")

        st.write("**Method 3: Matrix / Gaussian Elimination**")
        st.latex(r"\begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 10 \\ 3 \end{bmatrix}")
        st.latex(r"\left[\begin{array}{cc|c}2&1&10\\1&-1&3\end{array}\right] \xrightarrow{R_1\leftrightarrow R_2}"
                 r"\left[\begin{array}{cc|c}1&-1&3\\2&1&10\end{array}\right] \xrightarrow{R_2-2R_1}"
                 r"\left[\begin{array}{cc|c}1&-1&3\\0&3&4\end{array}\right]")
        st.latex(r"3y=4 \Rightarrow y=\tfrac{4}{3},\; x=3+y=\tfrac{13}{3}")

        st.write("**Method 4: Cramer's Rule**")
        st.latex(r"\Delta=\begin{vmatrix}2&1\\1&-1\end{vmatrix} = -3,\quad "
                 r"\Delta_x=\begin{vmatrix}10&1\\3&-1\end{vmatrix}=-13,\quad "
                 r"\Delta_y=\begin{vmatrix}2&10\\1&3\end{vmatrix}=-4")
        st.latex(r"x=\frac{\Delta_x}{\Delta}=\tfrac{13}{3},\quad y=\frac{\Delta_y}{\Delta}=\tfrac{4}{3}")

    with tabs[4]:
        st.subheader("Polynomial Equations")
        st.write("**Cubic example**")
        st.latex(r"x^3 - 6x^2 + 11x - 6 = 0")
        st.write("By inspection, \(x=1\) is a root:")
        st.latex(r"(x-1)(x^2 - 5x + 6) = 0 \Rightarrow (x-1)(x-2)(x-3)=0 \Rightarrow x=1,2,3")
        st.write("**General advice**")
        st.write("- Try rational roots via the Rational Root Theorem")
        st.write("- Factor by grouping if possible")
        st.write("- Otherwise use numerical methods or exact radicals when available")

    with tabs[5]:
        st.subheader("Tips & References")
        st.write("- Reduce to standard form before solving")
        st.write("- Check solutions by substitution")
        st.write("- For quadratics, the discriminant tells the nature of the roots")
        st.write("- For systems, pick the method that keeps arithmetic simple")

# -----------------------------
# Linear Equation Solver
# -----------------------------
elif tool == "Linear Equation":
    st.subheader("Linear Equation")
    expr_input = st.text_input(
        "Enter a linear equation (e.g., 2x + 5 = 0 or 'find x when two x plus five equals zero')",
        value="2x + 5 = 0"
    )
    if expr_input:
        try:
            eq_text = normalize_equation_text(expr_input)
            lhs_str, rhs_str = [s.strip() for s in eq_text.split("=")]
            lhs = parse_side(lhs_str)
            rhs = parse_side(rhs_str)
            eq = sp.Eq(lhs, rhs)
            sol = sp.solve(eq, x)

            # Render solution cleanly in LaTeX
            has_solution = display_equation_solution(eq, sol, {x})

            # Short Explanation (only after solution is shown)
            if has_solution:
                with st.expander("Short Explanation"):
                    st.write("Rearrange to isolate \(x\).")
                    st.latex(r"ax + b = 0 \;\Rightarrow\; ax = -b \;\Rightarrow\; x = -\tfrac{b}{a}")

            # Optional detailed steps (LaTeX)
            if st.button("Show Steps"):
                st.write("General method:")
                st.latex(r"ax + b = 0")
                st.latex(r"ax = -b")
                st.latex(r"x = -\frac{b}{a}")

        except Exception:
            st.error(FRIENDLY_ERROR)

# -----------------------------
# Quadratic Equation Solver
# -----------------------------
elif tool == "Quadratic Equation":
    st.subheader("Quadratic Equation")
    expr_input = st.text_input(
        "Enter a quadratic equation (e.g., x^2 + 5x + 6 = 0 or 'solve x squared plus five x plus six equals zero')",
        value="x^2 + 5x + 6 = 0"
    )
    if expr_input:
        try:
            eq_text = normalize_equation_text(expr_input)
            lhs_str, rhs_str = [s.strip() for s in eq_text.split("=")]
            lhs = parse_side(lhs_str)
            rhs = parse_side(rhs_str)
            eq = sp.Eq(lhs, rhs)

            # Identify quadratic in x when possible
            poly = sp.Poly(sp.together(lhs - rhs), x)
            sols = sp.solve(eq, x)

            has_solution = display_equation_solution(eq, sols, {x})

            if has_solution:
                with st.expander("Short Explanation"):
                    st.write("Use the quadratic formula with the discriminant \(\\Delta=b^2-4ac\).")
                    if poly.total_degree() == 2:
                        a, b_, c = poly.all_coeffs()
                        D = sp.discriminant(poly.as_expr(), x)
                        st.latex(rf"a = {sp.latex(a)},\; b = {sp.latex(b_)},\; c = {sp.latex(c)}")
                        st.latex(rf"\Delta = b^2 - 4ac = {sp.latex(D)}")
                        st.latex(r"x = \frac{-b \pm \sqrt{\Delta}}{2a}")

            if st.button("Show Steps"):
                if poly.total_degree() == 2:
                    a, b_, c = poly.all_coeffs()
                    D = sp.discriminant(poly.as_expr(), x)
                    st.latex(rf"{sp.latex(poly.as_expr())} = 0")
                    st.latex(rf"a = {sp.latex(a)},\; b = {sp.latex(b_)},\; c = {sp.latex(c)}")
                    st.latex(rf"\Delta = b^2 - 4ac = {sp.latex(D)}")
                    st.latex(r"x = \frac{-b \pm \sqrt{\Delta}}{2a}")
                else:
                    st.write("Solved using algebraic methods.")
        except Exception:
            st.error(FRIENDLY_ERROR)

# -----------------------------
# Simultaneous Equations Solver
# -----------------------------
elif tool == "Simultaneous Equations":
    st.subheader("Simultaneous Equations")
    eq1 = st.text_input("Equation 1 (e.g., 2x + y = 10)", value="2x + y = 10")
    eq2 = st.text_input("Equation 2 (e.g., x - y = 3)", value="x - y = 3")
    if eq1 and eq2:
        try:
            e1_text = normalize_equation_text(eq1)
            e2_text = normalize_equation_text(eq2)

            l1s, r1s = [s.strip() for s in e1_text.split("=")]
            l2s, r2s = [s.strip() for s in e2_text.split("=")]
            l1 = parse_side(l1s); r1 = parse_side(r1s)
            l2 = parse_side(l2s); r2 = parse_side(r2s)

            E1 = sp.Eq(l1, r1)
            E2 = sp.Eq(l2, r2)

            sol_list = sp.solve((E1, E2), (x, y), dict=True)
            sol = sol_list[0] if sol_list else {}
            has_solution = display_equation_solution(E1, sol, {x, y})
            st.latex(sp.latex(E2))

            # Short Explanation (only after solution is shown)
            if has_solution:
                with st.expander("Short Explanation"):
                    st.write("You can solve by **Substitution**, **Elimination**, **Gaussian elimination**, or **Cramer's Rule**.")
                    st.write("For example, with substitution:")
                    try:
                        x_expr = sp.solve(E2, x)[0]
                        st.latex(rf"x = {sp.latex(x_expr)}")
                    except Exception:
                        try:
                            y_expr = sp.solve(E2, y)[0]
                            st.latex(rf"y = {sp.latex(y_expr)}")
                        except Exception:
                            st.write("(Automatic symbolic rearrangement not straightforward for a short display.)")

            if st.button("Show Steps (Substitution & Elimination)"):
                # Substitution
                st.write("**Substitution**")
                try:
                    try:
                        x_expr = sp.solve(E2, x)[0]
                        subst_eq = sp.Eq(E1.lhs.subs(x, x_expr), E1.rhs.subs(x, x_expr))
                        st.latex(rf"x = {sp.latex(x_expr)}")
                        st.latex(rf"{sp.latex(subst_eq.lhs)} = {sp.latex(subst_eq.rhs)}")
                        y_val = sp.solve(subst_eq, y)[0]
                        x_val = x_expr.subs(y, y_val)
                        st.latex(rf"y = {sp.latex(y_val)},\quad x = {sp.latex(x_val)}")
                    except Exception:
                        y_expr = sp.solve(E2, y)[0]
                        subst_eq = sp.Eq(E1.lhs.subs(y, y_expr), E1.rhs.subs(y, y_expr))
                        st.latex(rf"y = {sp.latex(y_expr)}")
                        st.latex(rf"{sp.latex(subst_eq.lhs)} = {sp.latex(subst_eq.rhs)}")
                        x_val = sp.solve(subst_eq, x)[0]
                        y_val = y_expr.subs(x, x_val)
                        st.latex(rf"x = {sp.latex(x_val)},\quad y = {sp.latex(y_val)}")
                except Exception:
                    st.write("Substitution steps not available for this system.")

                # Elimination
                st.write("**Elimination**")
                try:
                    # Move RHS to LHS to read coefficients
                    L1 = sp.together(E1.lhs - E1.rhs)
                    L2 = sp.together(E2.lhs - E2.rhs)
                    a1 = sp.Eq(L1, 0).lhs.as_coefficients_dict().get(x, 0)
                    b1 = sp.Eq(L1, 0).lhs.as_coefficients_dict().get(y, 0)
                    a2 = sp.Eq(L2, 0).lhs.as_coefficients_dict().get(x, 0)
                    b2 = sp.Eq(L2, 0).lhs.as_coefficients_dict().get(y, 0)

                    # Align y-coefficients by multiplying Eq1 by b2 and Eq2 by b1, then subtract
                    m1, m2 = b2, b1
                    E1s = sp.Eq(m1*E1.lhs, m1*E1.rhs)
                    E2s = sp.Eq(m2*E2.lhs, m2*E2.rhs)
                    elim = sp.Eq(E1s.lhs - E2s.lhs, E1s.rhs - E2s.rhs)

                    st.latex(rf"{sp.latex(m1)}\cdot({sp.latex(E1.lhs)} = {sp.latex(E1.rhs)})")
                    st.latex(rf"{sp.latex(m2)}\cdot({sp.latex(E2.lhs)} = {sp.latex(E2.rhs)})")
                    st.latex(r"\text{Subtract to eliminate } y:")
                    st.latex(rf"{sp.latex(elim.lhs)} = {sp.latex(elim.rhs)}")

                    x_val2 = sp.solve(elim, x)[0]
                    y_val2 = sp.solve(E2.subs(x, x_val2), y)[0]
                    st.latex(rf"x = {sp.latex(x_val2)},\quad y = {sp.latex(y_val2)}")
                except Exception:
                    st.write("Elimination steps not available for this system.")
        except Exception:
            st.error(FRIENDLY_ERROR)

# -----------------------------
# Polynomial Equation Solver
# -----------------------------
elif tool == "Polynomial Equation":
    st.subheader("Polynomial Equation")
    expr_input = st.text_input(
        "Enter a polynomial equation (e.g., x^3 - 6x^2 + 11x - 6 = 0 or 'x cubed minus 6x squared plus 11x minus 6 equals zero')",
        value="x^3 - 6x^2 + 11x - 6 = 0"
    )
    if expr_input:
        try:
            eq_text = normalize_equation_text(expr_input)
            lhs_str, rhs_str = [s.strip() for s in eq_text.split("=")]
            lhs = parse_side(lhs_str)
            rhs = parse_side(rhs_str)
            eq = sp.Eq(lhs, rhs)

            # pick a target symbol (prefer x)
            syms = sorted((lhs - rhs).free_symbols, key=lambda s: s.name)
            target = syms[0] if syms else x

            sols = sp.solve(eq, target)
            has_solution = display_equation_solution(eq, sols, {target})

            if has_solution:
                with st.expander("Short Explanation"):
                    st.write("For higher-degree polynomials, try factoring; if not possible, use rational root tests or numerical methods.")
                    st.latex(r"(x-1)(x^2-5x+6)=0 \Rightarrow (x-1)(x-2)(x-3)=0 \Rightarrow x=1,2,3")

            if st.button("Show Steps"):
                st.write("General guidance:")
                st.latex(r"\text{1) Try factoring by inspection}")
                st.latex(r"\text{2) Use Rational Root Theorem}")
                st.latex(r"\text{3) If needed, use numerical methods or exact radicals}")
        except Exception:
            st.error(FRIENDLY_ERROR)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("© 2025 AmanteMath | Powered by Python & Streamlit")
