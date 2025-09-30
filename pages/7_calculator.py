import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AmanteMath - Calculator", layout="centered")

# --- THEMES ---
themes = {
    "Default": {
        "bg": "#0e1117",
        "button": "#262730",
        "text": "white",
        "display_bg": "#1f2229"
    },
    "AmanteMath": {
        "bg": "#007C91",
        "button": "#0097A7",
        "text": "white",
        "display_bg": "#004D60"
    }
}

if "theme" not in st.session_state:
    st.session_state.theme = "AmanteMath"

theme_choice = st.selectbox("Theme", list(themes.keys()),
                            index=list(themes.keys()).index(st.session_state.theme))
st.session_state.theme = theme_choice
th = themes[st.session_state.theme]

mode = st.radio("Calculator Mode", ["Basic", "Scientific"], index=0, horizontal=True)

calculator_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    background-color: {th['bg']};
    color: {th['text']};
    font-family: Arial, sans-serif;
    display: flex;
    justify-content: center;
  }}
  .calc {{
    background: {th['display_bg']};
    padding: 20px;
    border-radius: 16px;
    width: 320px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }}
  #display {{
    width: 100%;
    height: 60px;
    font-size: 24px;
    text-align: right;
    margin-bottom: 12px;
    padding: 8px;
    border-radius: 10px;
    border: none;
    background: {th['bg']};
    color: {th['text']};
  }}
  .buttons {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }}
  button {{
    height: 55px;
    font-size: 18px;
    border: none;
    border-radius: 10px;
    background: {th['button']};
    color: {th['text']};
    cursor: pointer;
    transition: all 0.2s ease-in-out;
  }}
  button:hover {{
    background: #00bcd4;
  }}
  button:active {{
    transform: scale(0.95);
  }}
</style>
</head>
<body>
  <div class="calc">
    <input type="text" id="display" disabled>
    <div class="buttons" id="button-grid"></div>
  </div>

<script>
  let display = document.getElementById('display');

  function appendChar(char) {{
    display.value += char;
  }}

  function clearDisplay() {{
    display.value = '';
  }}

  function delChar() {{
    display.value = display.value.slice(0, -1);
  }}

  function factorial(n) {{
    if (n < 0) return NaN;
    if (n === 0) return 1;
    let res = 1;
    for (let i = 1; i <= n; i++) res *= i;
    return res;
  }}

  function calculate() {{
    try {{
      let expr = display.value;

      // Replace powers
      expr = expr.replace(/(\\d+)(\\^)(\\d+)/g, "Math.pow($1,$3)");

      // Replace square/cube
      expr = expr.replace(/(\\d+)²/g, "Math.pow($1,2)");
      expr = expr.replace(/(\\d+)³/g, "Math.pow($1,3)");

      // Trig functions in degrees
      expr = expr.replace(/sin\\(?([0-9.]+)\\)?/g, "Math.sin(($1)*Math.PI/180)");
      expr = expr.replace(/cos\\(?([0-9.]+)\\)?/g, "Math.cos(($1)*Math.PI/180)");
      expr = expr.replace(/tan\\(?([0-9.]+)\\)?/g, "Math.tan(($1)*Math.PI/180)");

      // Square root
      expr = expr.replace(/√\\(?([0-9.]+)\\)?/g, "Math.sqrt($1)");

      // Logarithms
      expr = expr.replace(/log10\\(?([0-9.]+)\\)?/g, "Math.log10($1)");
      expr = expr.replace(/ln\\(?([0-9.]+)\\)?/g, "Math.log($1)");

      // Factorial using "!"
      expr = expr.replace(/(\\d+)!/g, "factorial($1)");

      // Replace constants
      expr = expr.replace(/π/g, "Math.PI");
      expr = expr.replace(/e/g, "Math.E");

      display.value = eval(expr);
    }} catch (e) {{
      display.value = 'Error';
    }}
  }}

  const basicButtons = [
    ["AC","⌫","(",")"],
    ["7","8","9","/"],
    ["4","5","6","*"],
    ["1","2","3","-"],
    ["0",".","=","+"],
    ["x²","√","x³","xʸ"]
  ];

  const sciButtons = [
    ["AC","⌫","(",")"],
    ["sin","cos","tan","√"],
    ["log10","ln","π","e"],
    ["!","x²","x³","xʸ"],
    ["7","8","9","/"],
    ["4","5","6","*"],
    ["1","2","3","-"],
    ["0",".","=","+"]
  ];

  let layout = ("{mode}" === "Scientific") ? sciButtons : basicButtons;
  let grid = document.getElementById("button-grid");

  layout.forEach(row => {{
    row.forEach(label => {{
      let btn = document.createElement("button");
      btn.textContent = label;
      btn.onclick = function() {{
        if(label === "AC") clearDisplay();
        else if(label === "⌫") delChar();
        else if(label === "=") calculate();
        else if(label === "x²") appendChar("^2");
        else if(label === "x³") appendChar("^3");
        else if(label === "xʸ") appendChar("^");
        else if(label === "√") appendChar("√");
        else if(label === "sin") appendChar("sin");
        else if(label === "cos") appendChar("cos");
        else if(label === "tan") appendChar("tan");
        else if(label === "log10") appendChar("log10");
        else if(label === "ln") appendChar("ln");
        else if(label === "!") appendChar("!");
        else if(label === "π") appendChar("π");
        else if(label === "e") appendChar("e");
        else appendChar(label);
      }};
      grid.appendChild(btn);
    }});
  }});
</script>
</body>
</html>
"""

components.html(calculator_html, height=650)
