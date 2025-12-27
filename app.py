import streamlit as st
import torch
import numpy as np
import time
import os

# --- IMPORTS ---
from model.final_structure import RecursiveTransformer 
from datasets.dataset_4x4 import GeneralTokenizer, generate_sudoku_level as gen_4x4
from datasets.dataset_9x9 import Tokenizer9x9, generate_9x9_level as gen_9x9

# --- CONFIG ---
DEVICE = "cpu"
PATH_4x4 = os.path.join("trained_models", "brain_4x4_final.pth")
PATH_9x9 = os.path.join("trained_models", "brain_9x9_final.pth")

st.set_page_config(page_title="TRM: Tiny Recursive Model", page_icon="🧠", layout="wide")

# --- CSS STYLES ---
st.markdown("""
<style>
    /* Grid Styling */
    .grid-4x4 { display: grid; grid-template-columns: repeat(4, 50px); gap: 4px; margin-bottom: 10px; }
    .cell-4x4 { width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 4px; font-weight: bold; border: 2px solid #333; font-size: 20px; background-color: white; color: black; }
    
    .grid-9x9 { display: grid; grid-template-columns: repeat(9, 24px); gap: 1px; margin-bottom: 10px; padding: 2px; }
    .cell-9x9 { width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; border-radius: 2px; font-weight: bold; border: 1px solid #888; font-size: 12px; background-color: white; color: black; }
    .cell-9x9:nth-child(9n+3), .cell-9x9:nth-child(9n+6) { border-right: 2px solid #000; margin-right: 3px; }
    .grid-9x9 > div:nth-child(27n) { margin-bottom: 3px; }
    
    /* Text Styling */
    h1, h2, h3 { font-family: 'Helvetica', sans-serif; }
    .paper-text { font-family: 'Georgia', serif; font-size: 18px; line-height: 1.6; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- LOADING LOGIC ---
@st.cache_resource
def load_model(mode):
    if mode == "4x4":
        model = RecursiveTransformer(vocab_size=35, num_layers=4, d_model=128, num_heads=4, d_ff=512, max_len=64, dropout=0.0)
        path = PATH_4x4
    else:
        model = RecursiveTransformer(vocab_size=20, num_layers=6, d_model=256, num_heads=4, d_ff=512, max_len=128, dropout=0.0)
        path = PATH_9x9
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except: return None

def get_mode_tools(mode):
    return (GeneralTokenizer(), gen_4x4) if mode == "4x4" else (Tokenizer9x9(), gen_9x9)

def draw_grid(grid_str, mode, title="Grid"):
    st.caption(title)
    clean = grid_str.replace("|", "").replace(" ", "")
    if mode == "4x4":
        if len(clean) > 16: clean = clean[:16]
        css_grid, css_cell = "grid-4x4", "cell-4x4"
    else:
        if len(clean) > 81: clean = clean[:81]
        css_grid, css_cell = "grid-9x9", "cell-9x9"
    html = f'<div class="{css_grid}">'
    for char in clean:
        is_empty = (char == "0" or char == "?")
        val, bg = ("" if is_empty else char), ("#f8f9fa" if is_empty else "#d1e7dd")
        html += f'<div class="{css_cell}" style="background:{bg};">{val}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("TRM Explorer")
mode = st.sidebar.radio("Dataset", ["4x4", "9x9"])
loops = st.sidebar.slider("Inference Loops (T)", 1, 32, 16 if mode == "4x4" else 24)
model = load_model(mode)
tokenizer, gen_func = get_mode_tools(mode)

# --- TABS ---
tab_paper, tab_demo = st.tabs(["TRM details", "Demo"])

# === TAB 1: THE PAPER ===
with tab_paper:
    st.title("Tiny Recursive Models (TRM)")
    st.markdown("**Based on: *Less is More: Recursive Reasoning with Tiny Networks* (arXiv:2510.04871v1)**")
    
    st.info("💡 **Core Thesis:** A single tiny network (2 layers), recursing on its own output, can outperform massive LLMs on logic tasks by effectively 'simulating' depth.")

    # --- SECTION 1: INTRODUCTION ---
    st.header("1. Introduction: The Problem with Feed-Forward")
    st.markdown("""
    <div class="paper-text">
    Standard Large Language Models (LLMs) generate answers autoregressively. This is brittle: a single wrong token can ruin the logic chain. 
    While Chain-of-Thought (CoT) helps, it requires generating expensive tokens.
    
    **Hierarchical Reasoning Models (HRM)** proposed using two networks (f_L, f_H) to recurse at different frequencies. 
    However, HRM is complex, relying on questionable biological arguments and Fixed-Point Theorems.
    
    **We propose TRM (Tiny Recursive Model).** It simplifies HRM by:
    1.  Using a **Single Network** instead of two.
    2.  Removing the Fixed-Point Theorem requirement.
    3.  Recursively updating two latent variables: the **Answer (y)** and the **Reasoning (z)**.
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 2: ARCHITECTURE ---
    st.header("2. The TRM Architecture")
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown(r"""
        <div class="paper-text">
        TRM defines the state of the system with three variables:
        <ul>
            <li><b>x</b>: The <b>Input Question</b> (e.g., the unsolved Sudoku).</li>
            <li><b>y</b> (or z_H): The <b>Predicted Answer</b> (the filled grid).</li>
            <li><b>z</b> (or z_L): The <b>Latent Reasoning</b> (a dense feature map).</li>
        </ul>
        
        Unlike standard transformers that map x -> y once, TRM maps (x, y_t, z_t) -> (y_{t+1}, z_{t+1}) repeatedly.
        <br><br>
        <b>Why 2 Variables (y and z)?</b>
        <ul>
            <li><b>y</b> is the <b>Explicit Memory</b>: It stores the current best hypothesis.</li>
            <li><b>z</b> is the <b>Implicit Memory</b>: It acts like a "scratchpad" or Chain-of-Thought, storing intermediate calculations that don't fit in the grid.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # IMAGE FOR UPDATE RULE FORMULA
        st.markdown("**Update Rule:**")
        if os.path.exists("assets/formula_update.png"):
            st.image("assets/formula_update.png", width=300)
        else:
            st.code("(y_t+1, z_t+1) = f(x, y_t, z_t)")
            
    with c2:
        if os.path.exists("assets/trm_diagram.png"):
            st.image("assets/trm_diagram.png", caption="Fig 1: TRM Architecture")
        else:
            st.warning("⚠️ Missing 'assets/trm_diagram.png'")

    # --- SECTION 3: THE ALGORITHM ---
    st.header("3. The Recursive Algorithm")
    st.markdown("The core innovation is **Deep Recursion** without gradients, followed by a gradient step.")
    
    st.code("""
def latent_recursion(x, y, z, n=6):
    # Inner loop: Refine the "Reasoning" (z) multiple times
    for i in range(n): 
        z = net(x, y, z)
    
    # Outer loop: Update the "Answer" (y) once based on refined reasoning
    y = net(y, z) 
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    # Phase 1: Free Thinking (No Gradients)
    # We let the model "think" for T-1 loops without optimizing weights.
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)
            
    # Phase 2: Learned Correction (With Gradients)
    y, z = latent_recursion(x, y, z, n)
    
    return y, z
    """, language="python")

    st.markdown("""
    **Key Hyperparameters:**
    * **n=6**: The number of latent updates per answer update.
    * **T=3**: The number of macro-loops.
    * **Total Effective Depth:** T * (n+1) = 21 passes per inference step.
    """)

    # --- SECTION 4: DEEP SUPERVISION ---
    st.header("4. Deep Supervision & Training")
    st.markdown(r"""
    <div class="paper-text">
    Standard BPTT (Backpropagation Through Time) is too expensive for hundreds of layers. 
    TRM uses **Deep Supervision** to train efficiently.
    Instead of unrolling the entire loop, we use a "Stop-Gradient" trick.
    <br><br>
    </div>
    """, unsafe_allow_html=True)
    
    # IMAGE FOR LOSS FORMULA
    st.markdown("**Loss Function:**")
    if os.path.exists("assets/formula_loss.png"):
        st.image("assets/formula_loss.png", width=300)
    else:
        st.code("Loss = Sum(CrossEntropy(y_hat_k, y_true))")

    st.markdown(r"""
    <div class="paper-text">
    <ul>
        <li>We run this for <b>N_sup=16</b> supervision steps during training.</li>
        <li>This forces the model to be correct <i>at every stage</i> of thinking, not just at the end.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 5: WHY "TINY"? ---
    st.header("5. Why 'Less is More'?")
    st.markdown("""
    The authors found a counter-intuitive result: **Smaller networks generalized better.**
    
    | Model Size | Sudoku Accuracy |
    | :--- | :--- |
    | **HRM (27M Params)** | 55.0% |
    | **TRM (7M Params, Att)** | 74.7% |
    | TRM (5M/19M, MLP) | **87.4%** |
    
    **The Theory:** On hard logic tasks with limited data (1k examples), large models **overfit**. They memorize the training set. A tiny, recursive model *cannot* memorize; it is forced to learn the **general algorithm** (the rules of Sudoku) to minimize the loss across 16 recursive steps.
    """)
    
    if os.path.exists("assets/latent_viz.png"):
        st.image("assets/latent_viz.png", caption="Fig 6: Visualization of z_H (Solution) vs z_L (Latent)")

# === TAB 2: INTERACTIVE DEMO ===
with tab_demo:
    st.header(f"🤖 Interactive TRM Solver ({mode})")
    
    if 'grid' not in st.session_state: st.session_state['grid'] = ""
    if 'last_mode' not in st.session_state: st.session_state['last_mode'] = mode
    
    if st.session_state['last_mode'] != mode:
        st.session_state['grid'], _ = gen_func("easy")
        st.session_state['last_mode'] = mode

    c1, c2, c3 = st.columns(3)
    if c1.button("Gen Easy"): st.session_state['grid'], _ = gen_func("easy")
    if c2.button("Gen Medium"): st.session_state['grid'], _ = gen_func("medium")
    if c3.button("Gen Hard"): st.session_state['grid'], _ = gen_func("hard")
    
    val = st.text_input("Input Puzzle String", st.session_state['grid'])
    
    col_in, col_out = st.columns(2)
    with col_in: draw_grid(val, mode, "Input State (x)")
    
    if st.button("🧠 Execute Recursive Reasoning"):
        if model:
            with st.spinner(f"Reasoning for {loops} loops..."):
                x = tokenizer.encode(val).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    all_outs = model(x, loops=loops)
                
                final = tokenizer.decode(torch.argmax(all_outs[:,-1], dim=-1)[0].tolist())
                with col_out: draw_grid(final, mode, f"Final Prediction (Loop {loops})")
                
                st.divider()
                st.subheader("Thinking Process (Trace of y)")
                st.markdown("We can visualize the **Explicit Memory ($y$)** evolving over time.")
                
                r1 = st.columns(4)
                steps = np.linspace(0, loops-1, 4, dtype=int)
                for i, s in enumerate(steps):
                    snap = tokenizer.decode(torch.argmax(all_outs[:,s], dim=-1)[0].tolist())
                    with r1[i]: 
                        draw_grid(snap, mode, f"Loop {s+1}")