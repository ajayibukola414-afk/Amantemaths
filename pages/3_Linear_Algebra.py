import streamlit as st
import sympy as sp

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AmanteMath - Linear Algebra", layout="wide")

# -----------------------------
# Header with Image + Tagline
# -----------------------------
def show_header_image():
    paths = [
        "assets/images/linear_algebra.png",
        "images/linear_algebra.png",
        "assets/linear_algebra.png",
        "linear_algebra.png",
    ]
    for p in paths:
        try:
            st.image(p, use_container_width=True)
            return
        except Exception:
            continue
    st.warning(
        "Linear Algebra image not found. "
        "Tried: assets/images/linear_algebra.png, images/linear_algebra.png, "
        "assets/linear_algebra.png, linear_algebra.png"
    )

col1, col2 = st.columns([2, 5])
with col1:
    show_header_image()
with col2:
    st.title("Linear Algebra")
    st.caption("Work with matrices, determinants, eigenvalues, and linear systems step by step.")

st.markdown("---")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("Linear Algebra Tools")
tool = st.sidebar.radio(
    "Select a tool:",
    [
        "Learn Linear Algebra",            # <-- PATCH: new 'textbook' section
        "Matrix Operations",
        "Determinant",
        "Inverse",
        "Rank",
        "Row Reduction (RREF)",            # <-- PATCH: new RREF tool
        "Eigenvalues & Eigenvectors",
        "Solve Ax = b",
    ],
)

# -----------------------------
# Tool: Learn Linear Algebra  (PATCH: new)
# -----------------------------
if tool == "Learn Linear Algebra":
    st.subheader("📘 Learn Linear Algebra")

    st.markdown("### 1) Linear Systems and the Matrix Form")
    st.write(
        "A linear system with variables \(x_1,\dots,x_n\) can be written as \(A\\,x=b\), "
        "where \(A\) is an \(m\\times n\) matrix, \(x\) is an \(n\\times 1\) vector of variables, "
        "and \(b\) is an \(m\\times 1\) constants vector."
    )
    st.latex(r"""
    \begin{bmatrix}
    a_{11} & a_{12} & \cdots & a_{1n}\\
    a_{21} & a_{22} & \cdots & a_{2n}\\
    \vdots & \vdots & \ddots & \vdots\\
    a_{m1} & a_{m2} & \cdots & a_{mn}
    \end{bmatrix}
    \begin{bmatrix}
    x_1\\x_2\\\vdots\\x_n
    \end{bmatrix}
    =
    \begin{bmatrix}
    b_1\\b_2\\\vdots\\b_m
    \end{bmatrix}
    """)

    st.markdown("**Solving methods:** Gaussian elimination (row operations), \(x=A^{-1}b\) when \(A\) is square and invertible, or using RREF.")

    st.markdown("### 2) Determinant (what it means)")
    st.write(
        "The determinant measures how a linear transformation scales area/volume. "
        "For \(2\\times2\):"
    )
    st.latex(r"""
    \det\begin{bmatrix}a & b\\ c & d\end{bmatrix} = ad - bc
    """)
    st.write(
        "If \(\det(A)=0\), the transformation squashes space (not invertible); if nonzero, \(A\) is invertible."
    )

    st.markdown("### 3) Inverse")
    st.write(
        "A square matrix \(A\) has an inverse \(A^{-1}\) iff \(\det(A)\neq 0\). "
        "For \(2\times2\):"
    )
    st.latex(r"""
    A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b\\ -c & a\end{bmatrix}
    """)
    st.write("For larger matrices, form the augmented matrix \([A\mid I]\) and row-reduce to \([I\mid A^{-1}]\).")

    st.markdown("### 4) Rank (intuition)")
    st.write(
        "Rank is the number of independent rows/columns (number of pivots in RREF). "
        "It tells how many directions the transformation truly acts on."
    )

    st.markdown("### 5) Eigenvalues & Eigenvectors (how to find)")
    st.write(
        "Eigenvalues \(\\lambda\) and eigenvectors \(v\\neq0\) satisfy \(A v = \\lambda v\). "
        "Steps: form the characteristic polynomial \(\\det(A-\\lambda I)\), solve for \(\\lambda\), "
        "then for each \(\\lambda\) solve \( (A-\\lambda I)v=0\) to get eigenvectors (the nullspace)."
    )

# -----------------------------
# Tool: Matrix Operations (UNCHANGED)
# -----------------------------
elif tool == "Matrix Operations":
    st.subheader("Matrix Addition, Subtraction, Multiplication")

    rows = st.number_input("Number of rows", min_value=1, max_value=5, value=2)
    cols = st.number_input("Number of columns", min_value=1, max_value=5, value=2)

    st.write("Enter Matrix A:")
    A_entries = []
    for i in range(rows):
        row = st.text_input(f"A row {i+1} (comma separated)", value=",".join(["0"] * cols))
        A_entries.append([sp.sympify(x.strip()) for x in row.split(",")])

    st.write("Enter Matrix B:")
    B_entries = []
    for i in range(rows):
        row = st.text_input(f"B row {i+1} (comma separated)", value=",".join(["0"] * cols))
        B_entries.append([sp.sympify(x.strip()) for x in row.split(",")])

    try:
        A = sp.Matrix(A_entries)
        B = sp.Matrix(B_entries)

        st.latex(r"A = " + sp.latex(A))
        st.latex(r"B = " + sp.latex(B))

        st.markdown("**Addition:**")
        st.latex(sp.latex(A + B))

        st.markdown("**Subtraction:**")
        st.latex(sp.latex(A - B))

        if cols == rows:
            st.markdown("**Multiplication (A × B):**")
            st.latex(sp.latex(A * B))
        else:
            st.info("Matrix multiplication requires A's columns = B's rows.")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Tool: Determinant (PATCH: more explanation)
# -----------------------------
elif tool == "Determinant":
    st.subheader("Determinant of a Matrix")

    n = st.number_input("Matrix size (n x n)", min_value=2, max_value=5, value=2)

    st.write("Enter Matrix:")
    A_entries = []
    for i in range(n):
        row = st.text_input(f"Row {i+1} (comma separated)", value=",".join(["0"] * n))
        A_entries.append([sp.sympify(x.strip()) for x in row.split(",")])

    try:
        A = sp.Matrix(A_entries)
        st.latex(r"A = " + sp.latex(A))
        detA = A.det()
        st.markdown("**Determinant:**")
        st.latex(sp.latex(detA))

        # Mini-textbook explanation (expanded)
        st.markdown("📘 **Step-by-step expansion:**")
        if n == 2:
            a,b,c,d = A[0,0],A[0,1],A[1,0],A[1,1]
            st.latex(r"\det\begin{bmatrix}a & b\\ c & d\end{bmatrix} = ad - bc")
            st.latex(rf"= ({sp.latex(a)})({sp.latex(d)}) - ({sp.latex(b)})({sp.latex(c)}) = {sp.latex(detA)}")
        elif n == 3:
            a,b,c = A[0,0],A[0,1],A[0,2]
            d,e,f = A[1,0],A[1,1],A[1,2]
            g,h,i = A[2,0],A[2,1],A[2,2]
            st.latex(r"\det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)")
            st.latex(
                rf"= {sp.latex(a)}\!\left({sp.latex(e*i)} - {sp.latex(f*h)}\right) "
                rf"- {sp.latex(b)}\!\left({sp.latex(d*i)} - {sp.latex(f*g)}\right) "
                rf"+ {sp.latex(c)}\!\left({sp.latex(d*h)} - {sp.latex(e*g)}\right)"
            )
            st.latex(rf"= {sp.latex(detA)}")
        else:
            st.write("For \(n>3\), use cofactor expansion or row-reduction to upper triangular (product of diagonal entries).")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Tool: Inverse (PATCH: deeper steps)
# -----------------------------
elif tool == "Inverse":
    st.subheader("Inverse of a Matrix")

    n = st.number_input("Matrix size (n x n)", min_value=2, max_value=5, value=2)

    st.write("Enter Matrix:")
    A_entries = []
    for i in range(n):
        row = st.text_input(f"Row {i+1} (comma separated)", value=",".join(["0"] * n))
        A_entries.append([sp.sympify(x.strip()) for x in row.split(",")])

    try:
        A = sp.Matrix(A_entries)
        st.latex(r"A = " + sp.latex(A))
        if A.det() == 0:
            st.error("Matrix is singular (det = 0), no inverse exists.")
        else:
            st.markdown("**Inverse:**")
            invA = A.inv()
            st.latex(sp.latex(invA))

            # Mini-textbook explanation (expanded)
            st.markdown("📘 **How it’s computed:**")
            if n == 2:
                a,b,c,d = A[0,0],A[0,1],A[1,0],A[1,1]
                st.latex(r"A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b\\ -c & a\end{bmatrix}")
                st.latex(
                    rf"= \frac{{1}}{{{sp.latex(a*d-b*c)}}}"
                    rf"\begin{bmatrix} {sp.latex(d)} & -{sp.latex(b)}\\ -{sp.latex(c)} & {sp.latex(a)}\end{bmatrix}"
                )
                st.latex(rf"= {sp.latex(invA)}")
            else:
                aug = A.row_join(sp.eye(n))
                st.latex(r"\text{Augmented }[A\,|\,I] = " + sp.latex(aug))
                rref = aug.rref()[0]
                st.latex(r"\text{Row-reduce }[A\,|\,I] \to [I\,|\,A^{-1}] = " + sp.latex(rref))
                st.write("Right block is \(A^{-1}\).")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Tool: Rank (UNCHANGED, plus brief explanation)
# -----------------------------
elif tool == "Rank":
    st.subheader("Rank of a Matrix")

    rows = st.number_input("Rows", min_value=1, max_value=5, value=2)
    cols = st.number_input("Columns", min_value=1, max_value=5, value=2)

    st.write("Enter Matrix:")
    A_entries = []
    for i in range(rows):
        row = st.text_input(f"Row {i+1} (comma separated)", value=",".join(["0"] * cols))
        A_entries.append([sp.sympify(x.strip()) for x in row.split(",")])

    try:
        A = sp.Matrix(A_entries)
        st.latex(r"A = " + sp.latex(A))
        st.markdown("**Rank:**")
        st.latex(sp.latex(A.rank()))

        st.markdown("📘 **Explanation:**")
        st.write("Rank is the number of pivots after row-reduction (independent rows/columns).")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Tool: Row Reduction (RREF)  (PATCH: new)
# -----------------------------
elif tool == "Row Reduction (RREF)":
    st.subheader("Row Reduction (RREF)")

    rows = st.number_input("Rows", min_value=1, max_value=7, value=3)
    cols = st.number_input("Columns", min_value=1, max_value=7, value=3)

    st.write("Enter Matrix:")
    A_entries = []
    for i in range(rows):
        row = st.text_input(f"Row {i+1} (comma separated)", value=",".join(["0"] * cols))
        A_entries.append([sp.sympify(x.strip()) for x in row.split(",")])

    try:
        A = sp.Matrix(A_entries)
        st.latex(r"A = " + sp.latex(A))

        R, pivots = A.rref()
        st.markdown("**RREF:**")
        st.latex(sp.latex(R))
        st.write(f"Pivot columns (0-indexed): {pivots}")

        st.markdown("📘 **How to do it by hand (outline):**")
        st.write("- Find the leftmost nonzero column (pivot column).")
        st.write("- Swap a nonzero row into the pivot row if needed.")
        st.write("- Scale the pivot row so the pivot entry is 1.")
        st.write("- Use row operations to make other entries in that column 0.")
        st.write("- Move right and down to the next pivot and repeat.")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Tool: Eigenvalues & Eigenvectors (PATCH: clearer steps)
# -----------------------------
elif tool == "Eigenvalues & Eigenvectors":
    st.subheader("Eigenvalues and Eigenvectors")

    n = st.number_input("Matrix size (n x n)", min_value=2, max_value=5, value=2)

    st.write("Enter Matrix:")
    A_entries = []
    for i in range(n):
        row = st.text_input(f"Row {i+1} (comma separated)", value=",".join(["0"] * n))
        A_entries.append([sp.sympify(x.strip()) for x in row.split(",")])

    try:
        A = sp.Matrix(A_entries)
        st.latex(r"A = " + sp.latex(A))

        lam = sp.symbols("lambda")
        char_poly = A.charpoly(lam)
        st.markdown("**Characteristic Polynomial:**")
        st.latex(r"\det(A - \lambda I) = " + sp.latex(char_poly.as_expr()))

        # Roots as a dict may not preserve order; show nicely:
        roots = char_poly.roots()
        st.markdown("**Eigenvalues (with multiplicities):**")
        for val, mult in roots.items():
            st.latex(rf"\lambda = {sp.latex(val)} \;\;(\text{{mult}}={mult})")

        st.markdown("**Eigenvectors (by nullspace of } A-\lambda I \text{):**")
        for val, mult, vecs in A.eigenvects():
            st.latex(rf"A - ({sp.latex(val)})I = " + sp.latex(A - val*sp.eye(n)))
            rref = (A - val*sp.eye(n)).rref()[0]
            st.latex(r"\text{RREF}(A-\lambda I) = " + sp.latex(rref))
            ns = (A - val*sp.eye(n)).nullspace()
            if ns:
                for k, v in enumerate(ns, start=1):
                    st.latex(rf"\text{{Eigenvector basis }}v_{k} = " + sp.latex(v))
            else:
                st.write("No nonzero eigenvector (this shouldn’t happen for an eigenvalue).")
            st.markdown("---")

        st.markdown("📘 **Reading the result:**")
        st.write(
            "Each eigenvalue solves the characteristic equation. For each one, any nonzero vector in the nullspace "
            "of \(A-\lambda I\) is an eigenvector. Sympy shows a basis of eigenvectors."
        )

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Tool: Solve Ax = b (PATCH: deeper steps)
# -----------------------------
elif tool == "Solve Ax = b":
    st.subheader("Solve a Linear System (Ax = b)")

    n = st.number_input("Number of variables", min_value=2, max_value=5, value=2)

    st.write("Enter coefficient matrix **A**:")
    A_entries = []
    for i in range(n):
        row = st.text_input(f"Row {i+1} (comma separated)", value=",".join(["0"] * n), key=f"Ax{i}")
        A_entries.append([sp.sympify(x.strip()) for x in row.split(",")])

    st.write("Enter constants vector **b**:")
    b_entries = st.text_input("Enter constants (comma separated)", value=",".join(["0"] * n), key="b")
    b_entries = [sp.sympify(x.strip()) for x in b_entries.split(",")]

    try:
        A = sp.Matrix(A_entries)
        b = sp.Matrix(b_entries)

        st.latex(r"A = " + sp.latex(A))
        st.latex(r"b = " + sp.latex(b))

        # Always show RREF of augmented for pedagogy
        aug = A.row_join(b)
        st.markdown("**Augmented matrix:**")
        st.latex(sp.latex(aug))
        R, piv = aug.rref()
        st.markdown("**RREF of augmented matrix:**")
        st.latex(sp.latex(R))

        if A.det() == 0:
            st.warning("Coefficient matrix A is singular (det = 0). System may have no or infinite solutions.")
            sol = sp.linsolve((A, b))

            if not sol:
                st.error("❌ No solution (inconsistent).")
                st.write("In RREF, a row like \\([0\;0\;\dots\;0\mid c]\\) with \(c\\neq 0\) indicates inconsistency.")
            else:
                st.markdown("**Solution set (parametric):**")
                for s in list(sol):
                    st.latex(sp.latex(s))
                st.markdown("📘 **Interpretation:** Free variables correspond to non-pivot columns; assign parameters and express pivot variables in terms of them.")
        else:
            sol = A.LUsolve(b)
            st.markdown("**Unique Solution:**")
            for i, val in enumerate(sol):
                st.latex(rf"x_{{{i+1}}} = {sp.latex(val)}")

            # Step-by-step explanation (expanded)
            st.markdown("📘 **Step-by-step (Gaussian elimination idea):**")
            st.write("- Form the augmented matrix and perform row operations to reach RREF.")
            st.write("- Identify pivot columns → leading variables; others → free variables (if any).")
            st.write("- Back-substitute (RREF already gives the solution directly).")
            st.latex(rf"x = {sp.latex(sol)}")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("© 2025 AmanteMath — Linear Algebra Module")
