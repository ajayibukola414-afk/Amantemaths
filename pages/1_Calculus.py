import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ---------------------------
# Natural Math Parsing
# ---------------------------
transformations = (standard_transformations + (implicit_multiplication_application,))

def parse_math(expr_str: str):
    """Friendly parser: supports ^ for powers and implicit multiplication like 3x."""
    expr_str = expr_str.replace("^", "**")
    return parse_expr(expr_str, transformations=transformations)

def symbols_in(expr):
    return sorted(list(expr.free_symbols), key=lambda s: s.name)

def choose_symbol(expr, default="x"):
    syms = symbols_in(expr)
    if not syms:
        return sp.Symbol(default)
    if len(syms) == 1:
        return syms[0]
    names = [s.name for s in syms]
    pick = st.selectbox("Variable to use:", names, index=names.index(default) if default in names else 0)
    return sp.Symbol(pick)

def bullet(lines):
    return "\n".join([f"- {ln}" for ln in lines])

# ---------------------------
# Derivative: Tutor Steps
# ---------------------------
def derivative_tutor_steps(expr, x):
    steps = []
    expr_s = sp.simplify(expr)
    terms = expr_s.as_ordered_terms()
    partials = []

    if len(terms) > 1:
        steps.append("**Linearity:** Differentiate each term separately and then add results.")

    for t in terms:
        # Quotient Rule
        num, den = sp.fraction(sp.together(t))
        is_rational = den != 1
        if is_rational and (num.has(x) or den.has(x)):
            u, v = num, den
            du, dv = sp.diff(u, x), sp.diff(v, x)
            dt = sp.simplify((du*v - u*dv)/v**2)
            steps.append(rf"""
**Quotient Rule** on ${sp.latex(t)}=\frac{{{sp.latex(u)}}}{{{sp.latex(v)}}}$  
Formula: $\displaystyle \left(\frac{{u}}{{v}}\right)'=\frac{{u'v-uv'}}{{v^2}}$  
Substitute: $u'={sp.latex(du)},\ v'={sp.latex(dv)} \Rightarrow {sp.latex(dt)}$
""")
            partials.append(dt)
            continue

        # Product Rule (two or more x-dependent factors)
        if isinstance(t, sp.Mul) and sum(1 for a in t.args if a.has(x)) >= 2:
            xdeps = [a for a in t.args if a.has(x)]
            others = [a for a in t.args if not a.has(x)]
            u = xdeps[0]
            v = sp.simplify(sp.Mul(*(xdeps[1:] + others))) if len(xdeps) > 1 else sp.Mul(*others)
            du = sp.diff(u, x)
            dv = sp.diff(v, x)
            dt = sp.simplify(du*v + u*dv)
            steps.append(rf"""
**Product Rule** on ${sp.latex(t)} = u\cdot v$  
Choose $u={sp.latex(u)}$, $v={sp.latex(v)}$  
Formula: $(uv)'=u'v+uv'$  
Compute: $u'={sp.latex(du)},\ v'={sp.latex(dv)}$  
Combine: ${sp.latex(du)}\cdot {sp.latex(v)} + {sp.latex(u)}\cdot {sp.latex(dv)} = {sp.latex(dt)}$
""")
            partials.append(dt)
            continue

        # Chain / log-diff for powers of functions
        chain_detected = False
        if isinstance(t, sp.Pow) and t.has(x) and (t.base != x or not t.exp.is_Number):
            if t.base.has(x) and t.exp.is_Number:
                g, n = t.base, t.exp
                dg = sp.diff(g, x)
                dt = sp.simplify(n * g**(n-1) * dg)
                steps.append(rf"""
**Chain Rule (Power of inner)** on ${sp.latex(t)}$  
Let $g(x)={sp.latex(g)}$, then $\frac{{d}}{{dx}}[g(x)]^{{{sp.latex(n)}}}={sp.latex(n)}[g(x)]^{{{sp.latex(n-1)}}}g'(x)$  
Compute $g'(x)={sp.latex(dg)}$ → {sp.latex(dt)}
""")
                partials.append(dt)
                chain_detected = True
            elif t.base.has(x) and t.exp.has(x):
                f, g = t.base, t.exp
                df, dg = sp.diff(f, x), sp.diff(g, x)
                dt = sp.simplify(t * (dg*sp.log(f) + g*df/f))
                steps.append(rf"""
**Logarithmic Differentiation** on ${sp.latex(t)} = {sp.latex(f)}^{sp.latex(g)}$  
$\ln y = {sp.latex(g)}\ln({sp.latex(f)}) \Rightarrow \frac{{y'}}{{y}} = {sp.latex(dg)}\ln({sp.latex(f)}) + {sp.latex(g)}\frac{{{sp.latex(df)}}}{{{sp.latex(f)}}}$  
Multiply by $y={sp.latex(t)}$ → {sp.latex(dt)}
""")
                partials.append(dt)
                chain_detected = True

        # Elementary functions (chain if inner)
        if not chain_detected and any(func in t.atoms(sp.Function) for func in [sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.asin, sp.acos, sp.atan]):
            dt = sp.diff(t, x)
            steps.append(rf"""
**Known Derivatives & Chain Rule** on ${sp.latex(t)}$  
Use derivative table and chain rule for inner functions.  
Result: {sp.latex(dt)}
""")
            partials.append(dt)
            continue

        # Power Rule x^n
        if isinstance(t, sp.Pow) and t.base == x and t.exp.is_Number:
            dt = sp.diff(t, x)
            steps.append(rf"**Power Rule** on ${sp.latex(t)}$: ${sp.latex(dt)}$")
            partials.append(dt)
            continue

        # Fallback
        dt = sp.diff(t, x)
        steps.append(rf"Differentiating ${sp.latex(t)}$ directly: {sp.latex(dt)}")
        partials.append(dt)

    result = sp.simplify(sum(partials))
    if len(terms) > 1:
        steps.append(rf"**Add & simplify:** {sp.latex(sp.simplify(sum(partials)))} = {sp.latex(result)}")
    return steps, result

# ---------------------------
# Integral: Tutor Steps
# ---------------------------
def is_linear(axb, x):
    a = sp.diff(axb, x)
    b = sp.simplify(axb - a*x)
    if a.free_symbols - {x} == set() and b.free_symbols - {x} == set():
        return (a, b)
    return None

def integral_tutor_steps(expr, x):
    steps = []
    f = sp.simplify(expr)

    # Rational → division + partial fractions
    num, den = sp.fraction(sp.together(f))
    if den != 1 and (num.has(x) or den.has(x)):
        steps.append("**Rational function** detected → make proper (if needed), then **partial fractions**.")
        deg_num = sp.degree(num, gen=x)
        deg_den = sp.degree(den, gen=x)
        work = sp.simplify(num/den)
        if deg_num is not None and deg_den is not None and deg_num >= deg_den:
            q, r = sp.div(num, den, domain='QQ')
            steps.append(rf"Division: $\dfrac{{{sp.latex(num)}}}{{{sp.latex(den)}}} = {sp.latex(q)} + \dfrac{{{sp.latex(r)}}}{{{sp.latex(den)}}}$")
            work = sp.simplify(q + r/den)
        apart = sp.apart(sp.together(work), x)
        if apart != work:
            steps.append(rf"Partial fractions: ${sp.latex(work)} = {sp.latex(apart)}$")
            work = apart
        terms = work.as_ordered_terms()
        parts = []
        if len(terms) > 1:
            steps.append("Integrate term-by-term (linearity):")
        for t in terms:
            It = sp.integrate(t, x)
            steps.append(rf"$\int {sp.latex(t)}\,dx = {sp.latex(It)}$")
            parts.append(It)
        F = sp.simplify(sum(parts))
        steps.append(rf"Simplify and add **+C**: {sp.latex(F)} + C")
        return steps, F

    # Integration by parts: poly × (exp/trig/log)
    if isinstance(f, sp.Mul):
        polys = [a for a in f.args if a.as_poly(x) is not None]
        if polys:
            poly = max(polys, key=lambda p: sp.degree(sp.simplify(p), gen=x))
            others = [a for a in f.args if a is not poly]
            other = sp.simplify(sp.Mul(*others)) if others else 1
            if other.has(sp.exp) or other.has(sp.sin) or other.has(sp.cos) or other.has(sp.log):
                u = sp.simplify(poly)
                dv = sp.simplify(other)
                du = sp.diff(u, x)
                v = sp.integrate(dv, x)
                steps.append(rf"""
**Integration by Parts** (poly × exp/trig/log).  
Choose $u={sp.latex(u)}$ (polynomial), $dv={sp.latex(dv)}\,dx \Rightarrow du={sp.latex(du)}\,dx,\ v={sp.latex(v)}$  
Formula: $\int u\,dv = uv - \int v\,du$
""")
                part_res = sp.simplify(u*v - sp.integrate(v*du, x))
                steps.append(rf"Apply: $uv - \int v\,du = {sp.latex(u*v)} - \int {sp.latex(v*du)}\,dx = {sp.latex(part_res)}$")
                steps.append("Add **+ C**.")
                return steps, sp.simplify(part_res)

    # Linear inside exp/sin/cos: substitution
    if f.has(sp.exp) or f.has(sp.sin) or f.has(sp.cos):
        for fun in [sp.exp, sp.sin, sp.cos]:
            for node in f.atoms(fun):
                inner = list(node.args)[0]
                lin = is_linear(inner, x)
                if lin:
                    a, b = lin
                    if f == node:
                        if fun == sp.exp:
                            steps.append(rf"$\int e^{{{sp.latex(a)}x+{sp.latex(b)}}}\,dx=\frac1{{{sp.latex(a)}}}e^{{{sp.latex(a)}x+{sp.latex(b)}}}+C$")
                            F = sp.simplify(node/a)
                        elif fun == sp.sin:
                            steps.append(rf"$\int \sin({sp.latex(a)}x+{sp.latex(b)})\,dx=-\frac1{{{sp.latex(a)}}}\cos({sp.latex(a)}x+{sp.latex(b)})+C$")
                            F = sp.simplify(-sp.cos(inner)/a)
                        else:
                            steps.append(rf"$\int \cos({sp.latex(a)}x+{sp.latex(b)})\,dx=\frac1{{{sp.latex(a)}}}\sin({sp.latex(a)}x+{sp.latex(b)})+C$")
                            F = sp.simplify(sp.sin(inner)/a)
                        steps.append("Add **+ C**.")
                        return steps, F
                    if isinstance(f, sp.Mul) and node in f.args:
                        F = sp.integrate(f, x)
                        steps.append(r"Use **u-substitution**: let $u=$ inner, match $du$, constants factor out.")
                        steps.append(rf"Result: {sp.latex(F)} + C")
                        return steps, F

    # Polynomial
    if f.as_poly(x) is not None:
        steps.append("**Polynomial integrand** → reverse Power Rule term-by-term.")
        parts = []
        for t in sp.expand(f).as_ordered_terms():
            It = sp.integrate(t, x)
            steps.append(rf"$\int {sp.latex(t)}\,dx = {sp.latex(It)}$")
            parts.append(It)
        F = sp.simplify(sum(parts))
        steps.append(rf"Simplify and add **+ C**: {sp.latex(F)} + C")
        return steps, F

    # Fallback
    F = sp.integrate(f, x)
    steps.append("Apply symbolic integration rules.")
    steps.append(rf"Result: {sp.latex(F)} + C")
    return steps, F

# ---------------------------
# Limit: Tutor Steps
# ---------------------------
def limit_tutor_steps(expr, x, a, dir_=None):
    steps = []
    show_dir = "" if not dir_ else ("^+" if dir_=="+" else "^-")
    steps.append(f"Target: $\\displaystyle \\lim_{{x\\to {sp.latex(a)}{show_dir}}} {sp.latex(expr)}$")

    # Direct substitution
    try:
        sub = sp.simplify(expr.subs(x, a))
    except Exception:
        sub = sp.nan
    if sub.is_real or sub.is_number:
        steps.append("**Direct substitution** works (continuity at the point).")
        steps.append(rf"Value: ${sp.latex(sub)}$")
        return steps, sub

    # Simplify/factor
    simp = sp.simplify(expr)
    if simp != expr:
        steps.append(rf"**Simplify/Factor:** ${sp.latex(expr)} \rightarrow {sp.latex(simp)}$")
    try:
        sub2 = sp.simplify(simp.subs(x, a))
    except Exception:
        sub2 = sp.nan
    if sub2.is_real or sub2.is_number:
        steps.append("After simplification, substitution succeeds.")
        steps.append(rf"Value: ${sp.latex(sub2)}$")
        return steps, sub2

    # 0/0 → L'Hôpital
    num, den = sp.fraction(sp.together(simp))
    if num.has(x) and den.has(x):
        try:
            n_at = sp.simplify(num.subs(x, a))
            d_at = sp.simplify(den.subs(x, a))
        except Exception:
            n_at = d_at = sp.nan
        if (n_at == 0) and (d_at == 0):
            steps.append("Detected **0/0** → apply **L'Hôpital’s Rule**:")
            dnum, dden = sp.diff(num, x), sp.diff(den, x)
            steps.append(rf"$\displaystyle \lim \frac{{{sp.latex(num)}}}{{{sp.latex(den)}}} = \lim \frac{{{sp.latex(dnum)}}}{{{sp.latex(dden)}}}$")
            lh = sp.simplify(dnum/dden)
            val = sp.limit(lh, x, a, dir=dir_) if dir_ else sp.limit(lh, x, a)
            steps.append(rf"Evaluate new limit: ${sp.latex(val)}$")
            return steps, val

    # Special trig near 0
    if a == 0 and (simp.has(sp.sin) or simp.has(sp.tan) or simp.has(sp.cos)):
        steps.append("Check **special trig limits** near 0 (e.g., $\\lim \\sin x / x = 1$).")
        val = sp.limit(simp, x, a, dir=dir_) if dir_ else sp.limit(simp, x, a)
        steps.append(rf"Value: ${sp.latex(val)}$")
        return steps, val

    # At infinity
    if a in (sp.oo, -sp.oo):
        steps.append("At infinity → compare **dominant growth**.")
        val = sp.limit(simp, x, a, dir=dir_) if dir_ else sp.limit(simp, x, a)
        steps.append(rf"Value: ${sp.latex(val)}$")
        return steps, val

    # Fallback
    val = sp.limit(simp, x, a, dir=dir_) if dir_ else sp.limit(simp, x, a)
    steps.append("Apply general symbolic limit rules.")
    steps.append(rf"Value: ${sp.latex(val)}$")
    return steps, val

# ======================================================
# ===============  UI (UNCHANGED LAYOUT)  ==============
# ======================================================

# --- Page Config (unchanged) ---
st.set_page_config(page_title="AmanteMath - Calculus", page_icon="📈", layout="wide")

# --- Header with Image (unchanged) ---
col1, col2 = st.columns([2, 5])
with col1:
    st.image("assets/images/calculus.png", use_container_width=True)
with col2:
    st.title("📈 Calculus")
    st.caption("Explore derivatives, integrals, and limits — step by step.")

st.markdown("---")

# --- Sidebar (unchanged) ---
st.sidebar.header("🔢 Calculus Tools")
calc_tool = st.sidebar.radio(
    "Select a tool:",
    ["Derivative", "Integral", "Limit", "📘 Learn Calculus"]
)

# ===============================
# DERIVATIVE TOOL
# ===============================
if calc_tool == "Derivative":
    st.subheader("🔹 Differentiation")
    expr_input = st.text_input(
        "Enter a function (one at a time) e.g., (x+1)^x * sin(x^2) or (x^2+1)/(x-1)",
        value="(x+1)^x * sin(x^2)"
    )
    order = st.number_input("Derivative order (n):", min_value=1, max_value=10, value=1, step=1)

    if expr_input:
        try:
            if "," in expr_input:
                st.error("Please enter **one expression at a time** (no commas).")
            else:
                expr = parse_math(expr_input)
                xvar = choose_symbol(expr, default="x")
                deriv = sp.diff(expr, xvar, order)
                st.latex(rf"\frac{{d^{order}}}{{d{sp.latex(xvar)}^{order}}}\left({sp.latex(expr)}\right) = {sp.latex(deriv)}")
                st.success(f"✅ Result: {deriv}")

                if st.checkbox("📌 Short Explanation"):
                    st.info("Identify structure (sum/product/quotient/composition). Apply Power, Product, Quotient, or Chain Rule accordingly. Simplify at the end.")

                if st.button("📖 Show Step-by-Step Solution"):
                    steps, final = derivative_tutor_steps(expr, xvar)
                    st.markdown("### Step-by-step (Tutor Explanation)")
                    for s in steps:
                        st.markdown(s)
                    if order > 1:
                        st.info("Higher-order derivatives are obtained by differentiating repeatedly.")
                    st.success(rf"**Final simplified derivative:** $\displaystyle {sp.latex(sp.simplify(sp.diff(expr, xvar, order)))}$")

        except Exception as e:
            st.error(f"Error: {e}")

# ===============================
# INTEGRAL TOOL
# ===============================
elif calc_tool == "Integral":
    st.subheader("🔹 Integration")
    expr_input = st.text_input(
        "Enter an integrand (e.g., (x^2+1)*exp(x), (2x)/(x^2-1), x*cos(x))",
        value="(x^2+1)*exp(x)"
    )
    definite = st.checkbox("Compute a definite integral?")
    a_text = b_text = None
    if definite:
        a_text = st.text_input("Lower bound a (e.g., 0, -oo):", value="0")
        b_text = st.text_input("Upper bound b (e.g., pi, oo):", value="1")

    if expr_input:
        try:
            if "," in expr_input and not definite:
                st.error("Please enter **one integrand** at a time (no commas).")
            else:
                expr = parse_math(expr_input)
                xvar = choose_symbol(expr, default="x")

                if definite:
                    a = sp.sympify(a_text); b = sp.sympify(b_text)
                    val = sp.integrate(expr, (xvar, a, b))
                    st.latex(rf"\int_{sp.latex(a)}^{sp.latex(b)} {sp.latex(expr)}\, d{sp.latex(xvar)} = {sp.latex(val)}")
                    st.success(f"✅ Definite integral: {val}")
                    if st.checkbox("📌 Also show numeric approximation"):
                        st.info(f"≈ {sp.N(val)}")
                else:
                    F = sp.integrate(expr, xvar)
                    st.latex(rf"\int {sp.latex(expr)}\, d{sp.latex(xvar)} = {sp.latex(F)} + C")
                    st.success(f"✅ Antiderivative: {F} + C")

                if st.checkbox("📌 Short Explanation"):
                    st.info("Technique choice: substitution for compositions; parts for poly×(exp/trig/log); partial fractions for rationals; reverse power rule for polynomials.")

                if st.button("📖 Show Step-by-Step Solution"):
                    steps, F_sym = integral_tutor_steps(expr, xvar)
                    st.markdown("### Step-by-step (Tutor Explanation)")
                    for s in steps:
                        st.markdown(s)
                    if definite:
                        try:
                            a = sp.sympify(a_text); b = sp.sympify(b_text)
                            F_b = sp.simplify(F_sym.subs(xvar, b))
                            F_a = sp.simplify(F_sym.subs(xvar, a))
                            st.markdown(rf"Apply **FTC**: $F(b)-F(a) = {sp.latex(F_b)} - {sp.latex(F_a)} = {sp.latex(sp.simplify(F_b-F_a))}$")
                        except Exception:
                            st.info("FTC evaluation symbolic step not available; definite integral value already computed.")
                    else:
                        check = sp.simplify(sp.diff(F_sym, xvar) - sp.simplify(sp.expand(expr)))
                        if check == 0:
                            st.success("Verified: differentiating the antiderivative returns the original integrand.")
                    st.success(rf"**Final antiderivative:** $\displaystyle {sp.latex(F_sym)}$" + ("" if not definite else ""))

        except Exception as e:
            st.error(f"Error: {e}")

# ===============================
# LIMIT TOOL
# ===============================
elif calc_tool == "Limit":
    st.subheader("🔹 Limits")
    expr_input = st.text_input("Enter an expression (e.g., (1-cos x)/x^2)", value="(1-cos x)/x^2")
    point = st.text_input("As x approaches (e.g., 0, oo, -oo):", value="0")
    direction = st.selectbox("Direction:", ["two-sided", "from the right (+)", "from the left (-)"], index=0)

    if expr_input:
        try:
            expr = parse_math(expr_input)
            xvar = choose_symbol(expr, default="x")
            a = sp.sympify(point)
            dir_ = None
            if direction.endswith("(+)"): dir_ = "+"
            elif direction.endswith("(-)"): dir_ = "-"

            lim_val = sp.limit(expr, xvar, a, dir=dir_) if dir_ else sp.limit(expr, xvar, a)
            st.latex(rf"\lim_{{{sp.latex(xvar)} \to {sp.latex(a)}{'' if not dir_ else '^'+dir_}}} {sp.latex(expr)} = {sp.latex(lim_val)}")
            st.success(f"✅ Limit: {lim_val}")

            if st.checkbox("📌 Short Explanation"):
                st.info("Try substitution; if indeterminate, simplify/factor. Use special trig limits near 0 and **L'Hôpital** for 0/0 or ∞/∞. At infinity, compare dominant growth.")

            if st.button("📖 Show Step-by-Step Solution"):
                steps, val = limit_tutor_steps(expr, xvar, a, dir_=dir_)
                st.markdown("### Step-by-step (Tutor Explanation)")
                for s in steps:
                    st.markdown(f"- {s}")
                st.success(rf"**Final limit value:** $\displaystyle {sp.latex(val)}$")

        except Exception as e:
            st.error(f"Error: {e}")

# ===============================
# 📘 LEARN CALCULUS — Study Guide with Tabs
# ===============================
elif calc_tool == "📘 Learn Calculus":
    st.subheader("📘 Comprehensive Calculus Study Guide")
    st.markdown("This guide summarizes key ideas, **formulas**, and **worked examples** you’ll need for **Differentiation, Integration, and Limits**.")

    tab1, tab2, tab3 = st.tabs(["✂️ Differentiation", "∫ Integration", "→ Limits"])

    # --- Differentiation Tab ---
    with tab1:
        st.markdown("### What is Differentiation?")
        st.write(
            "Differentiation measures the **instantaneous rate of change** of a function — "
            "the slope of the tangent line. Formally:"
        )
        st.latex(r"f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}")

        st.markdown("### Core Rules & Formulas")
        st.latex(r"\frac{d}{dx}(c)=0,\quad \frac{d}{dx}(cx)=c,\quad \frac{d}{dx}(x^n)=nx^{n-1}\ (n\neq 0)")
        st.latex(r"(u\pm v)'=u'\pm v',\quad (cu)'=c\,u'")
        st.latex(r"(uv)'=u'v+uv',\quad \left(\frac{u}{v}\right)'=\frac{u'v-uv'}{v^2}")
        st.latex(r"\frac{d}{dx}f(g(x))=f'(g(x))\cdot g'(x)\quad \text{(Chain Rule)}")
        st.latex(r"\frac{d}{dx}\,e^{x}=e^{x},\ \frac{d}{dx}\,a^{x}=a^{x}\ln a,\ \frac{d}{dx}\,\ln x=\frac{1}{x}")
        st.latex(r"\frac{d}{dx}\sin x=\cos x,\ \frac{d}{dx}\cos x=-\sin x,\ \frac{d}{dx}\tan x=\sec^2 x")

        st.markdown("### Worked Examples")
        st.markdown("**1) Power Rule**")
        st.latex(r"\frac{d}{dx}(3x^4)=3\cdot 4 x^{3}=12x^3")
        st.markdown("**2) Product Rule**")
        st.latex(r"\frac{d}{dx}\left[(x^2+1)\sin x\right]=2x\sin x+(x^2+1)\cos x")
        st.markdown("**3) Quotient Rule**")
        st.latex(r"\frac{d}{dx}\left(\frac{x^2+1}{x-1}\right)=\frac{(2x)(x-1)-(x^2+1)}{(x-1)^2}=\frac{x^2-2x-1}{(x-1)^2}")
        st.markdown("**4) Chain Rule**")
        st.latex(r"\frac{d}{dx}\left(\sin(x^2)\right)=\cos(x^2)\cdot 2x")

        st.markdown("### Tips & Common Mistakes")
        st.markdown(bullet([
            "Don’t forget Chain Rule when there’s an inner function.",
            "Group constants; factor common terms to simplify.",
            "For $y=f(x)^{g(x)}$, use **logarithmic differentiation**.",
            "Higher-order derivatives: differentiate repeatedly."
        ]))

        st.markdown("**Learn more online:**")
        st.markdown("- Paul’s Online Math Notes — Derivatives")
        st.markdown("- Khan Academy — Differentiation")
        st.markdown("- MIT OCW Single Variable Calculus (Lecture notes)")

    # --- Integration Tab ---
    with tab2:
        st.markdown("### What is Integration?")
        st.write(
            "Integration finds **accumulated change** (area under a curve) and **antiderivatives**."
        )
        st.latex(r"\int f(x)\,dx = F(x)+C\quad \text{where}\quad F'(x)=f(x)")
        st.markdown("**Fundamental Theorem of Calculus (FTC):**")
        st.latex(r"\frac{d}{dx}\left(\int_a^x f(t)\,dt\right)=f(x),\quad \int_a^b f(x)\,dx = F(b)-F(a)")

        st.markdown("### Core Formulas")
        st.latex(r"\int x^n\,dx=\frac{x^{n+1}}{n+1}+C\ (n\neq -1),\quad \int \frac{1}{x}\,dx=\ln|x|+C")
        st.latex(r"\int e^x\,dx=e^x+C,\quad \int a^x\,dx=\frac{a^x}{\ln a}+C")
        st.latex(r"\int \sin x\,dx=-\cos x + C,\quad \int \cos x\,dx=\sin x + C")
        st.latex(r"\int \frac{1}{x^2+a^2}\,dx=\frac{1}{a}\arctan\frac{x}{a}+C")
        st.latex(r"\int \frac{1}{ax+b}\,dx=\frac{1}{a}\ln|ax+b|+C")

        st.markdown("### Techniques")
        st.markdown("**Substitution (u-sub):** choose $u=g(x)$ so integrand becomes simpler.")
        st.latex(r"\int f(g(x))g'(x)\,dx=\int f(u)\,du")
        st.markdown("**Integration by Parts:**")
        st.latex(r"\int u\,dv=uv-\int v\,du")
        st.markdown("**Partial Fractions:** for proper rational functions.")
        st.markdown("**Trig Integrals / Substitutions:** use identities or $x=a\sin\theta$, etc.")

        st.markdown("### Worked Examples")
        st.markdown("**1) Substitution**")
        st.latex(r"\int \cos(3x)\,dx = \frac{1}{3}\sin(3x)+C")
        st.markdown("**2) Integration by Parts**")
        st.latex(r"\int x e^x\,dx = x e^x - \int 1\cdot e^x\,dx = x e^x - e^x + C = (x-1)e^x + C")
        st.markdown("**3) Partial Fractions**")
        st.latex(r"\int \frac{2x}{x^2-1}\,dx = \int \left(\frac{1}{x-1}+\frac{1}{x+1}\right)dx = \ln|x-1|+\ln|x+1|+C")

        st.markdown("### Tips & Common Mistakes")
        st.markdown(bullet([
            "Check if a quick antiderivative exists before using techniques.",
            "For products: if one factor simplifies by differentiating, try **parts**.",
            "Rational functions → make proper with division, then **partial fractions**.",
            "Always add **+C** for indefinite integrals."
        ]))

        st.markdown("**Learn more online:**")
        st.markdown("- Paul’s Online Math Notes — Integrals")
        st.markdown("- Khan Academy — Integration techniques")
        st.markdown("- MIT OCW — Techniques of Integration")

    # --- Limits Tab ---
    with tab3:
        st.markdown("### What is a Limit?")
        st.write(
            "A limit describes the value a function approaches as the input approaches some number."
        )
        st.latex(r"\lim_{x\to a} f(x) = L")
        st.markdown("**Continuity:** $f$ is continuous at $a$ if $\\lim_{x\\to a} f(x) = f(a)$.")

        st.markdown("### Rules & Strategies")
        st.markdown(bullet([
            "Try **direct substitution** first.",
            "If undefined (e.g., 0/0), **simplify/factor**.",
            "Use **rationalization** for surds, special trig identities near 0.",
            "**L’Hôpital’s Rule** for 0/0 or ∞/∞: differentiate top & bottom."
        ]))
        st.latex(r"\lim_{x\to 0}\frac{\sin x}{x}=1,\quad \lim_{x\to 0}\frac{1-\cos x}{x^2}=\frac{1}{2}")
        st.latex(r"\lim_{x\to \infty}\frac{ax^n+...}{bx^m+...}=\begin{cases}0&n<m\\ \frac{a}{b}&n=m\\ \infty&n>m\end{cases}")

        st.markdown("### Worked Examples")
        st.markdown("**1) Algebraic factoring**")
        st.latex(r"\lim_{x\to 2}\frac{x^2-4}{x-2}=\lim_{x\to 2}\frac{(x-2)(x+2)}{x-2}=4")
        st.markdown("**2) Trig special limit**")
        st.latex(r"\lim_{x\to 0}\frac{\sin x}{x}=1")
        st.markdown("**3) L’Hôpital’s Rule**")
        st.latex(r"\lim_{x\to 0}\frac{\ln(1+x)}{x}=\lim_{x\to 0}\frac{1/(1+x)}{1}=1")

        st.markdown("### Tips & Common Mistakes")
        st.markdown(bullet([
            "Confirm indeterminate forms before using L’Hôpital.",
            "For infinity, compare dominant growth (exp ≫ poly ≫ log).",
            "Be careful with one-sided limits and absolute values."
        ]))

        st.markdown("**Learn more online:**")
        st.markdown("- Paul’s Online Math Notes — Limits")
        st.markdown("- Khan Academy — Limits & continuity")
        st.markdown("- MIT OCW — Limits and Continuity")

# --- Footer (unchanged) ---
st.markdown("---")
st.caption("© 2025 AmanteMath | Powered by Python & Streamlit")
