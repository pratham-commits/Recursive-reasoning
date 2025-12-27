# 🧠 Recursive Sudoku Reasoner 

> **A Transformer that "thinks" before it answers.**
> **Implementation of "Less is More: Recursive Reasoning with Tiny Networks"**
> *[Read the Paper on arXiv](https://arxiv.org/abs/2510.04871)*

## 📖 Overview
Standard AI models (like ChatGPT) often struggle with logical constraints because they try to predict the answer in a single pass. This project implements a **Recursive Transformer (TRM)**—a specialized architecture that re-processes its own output multiple times ("thinking loops") to refine its logic and solve Sudoku puzzles without external search algorithms.

The model was trained using **Curriculum Learning** (Easy → Medium → Hard) and over different grids :- 
* **4x4 Grid**
* **9x9 Grid** 

## 🚀 Key Features
* **Recursive Architecture:** The model's output is fed back into itself for `N` loops (16-32 steps), allowing it to "backtrack" and correct mistakes internally.
* **Hybrid Solver:** Supports both "Toy" (4x4) and "Grand Challenge" (9x9) puzzles.
* **Thinking Visualization:** A dashboard that visualizes the model's internal state at Loop 1, Loop 8, Loop 16, etc., showing how the solution converges.

## ⚙️ Implementation: The TRM Structure
This codebase faithfully implements the **Recursive Transformer (TRM)** architecture, ensuring that the model "thinks" in latent space before committing to a final answer.

| TRM Concept | Implementation in Code | Description |
| :--- | :--- | :--- |
| **Recurrent Depth** | `loops` parameter in `forward()` | Instead of stacking 100 physical layers, we use 6 layers and loop them 24 times. This shares weights across "time." |
| **State Injection** | `x = x + self.dropout(...)` | The output of Loop $t$ becomes the query input for Loop $t+1$, allowing the model to see its own previous "thought." |
| **Positional Awareness** | `model/final_structure.py` | Custom positional embeddings ensure the model understands the 9x9 grid geometry (rows, cols, sub-grids). |
| **Dynamic Compute** | `app.py` Slider | We can dynamically adjust the recursion depth (e.g., 16 loops for Easy, 32 for Hard) at inference time without retraining. |

## 🛠️ Model Architecture Details
The core logic resides in `model/final_structure.py`:
* **Input:** Tokenized Grid String (e.g., `0 2 0 4 | ...`)
* **Embedding:** Learned Positional Embeddings + Value Embeddings.
* **Architecture Specs:**
    * *4x4 Model:* 4 Layers, 128 Hidden Dim (Small Brain).
    * *9x9 Model:* 6 Layers, 256 Hidden Dim (Big Brain).
* **Inference Strategy:** Iterative refinement. The probability distribution output of Loop $t$ informs the input of Loop $t+1$.

## 📊 Benchmarks & Performance 
*> 100 random samples for each type.*

*> Note: The 9x9 model demonstrates high logical understanding but occasionally fails strict "All-or-Nothing" checks on Hard puzzles due to single-digit errors.*
| Difficulty | 4x4 Accuracy | 9x9  Accuracy |
| :--- | :---: | :---: |
| **Easy** | 100% | 87% |
| **Medium** | 100% | 46% |
| **Hard** | 99% | 20% |


## 💻 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/recursive-sudoku.git](https://github.com/yourusername/recursive-sudoku.git)
    cd recursive-sudoku
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Dashboard**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
The repository follows a modular structure to separate architecture, data, and training logic:

```text
├── datasets/
│   ├── dataset_4x4.py      # Tokenizer & Generator for 4x4
│   └── dataset_9x9.py      # Tokenizer & Generator for 9x9
├── model/
│   ├── final_structure.py  # CORE TRM ARCHITECTURE (Recursive Loop)
│   └── layers.py           # Custom Multi-Head Attention Blocks
├── train_logs/             # Metrics storage
│   ├── 4x4_logs/           # logs for 4x4 model
│   └── 9x9_logs/           # logs for 9x9 model
├── train_scripts/
│   ├── train_4x4.py        # Training pipeline for 4x4
│   └── train_9x9.py        # Training pipeline for 9x9 
├── trained_models/         # Trained models
├── app.py                  # Streamlit Interactive Dashboard
├── test_parameters.py      # Sanity check for model parameters
└── requirements.txt
└── README.md               # Documentation
