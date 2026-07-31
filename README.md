## U.S. Association for Computational Mechanics (USACM) 2026 Short Course  
# The Phase-Field Approach to Brittle Fracture: Theory and Numerical Implementation

Welcome to the companion repository for the USACM 2026 short course: **“The Phase-Field Approach to Brittle Fracture: Theory and Numerical Implementation.”**

---

## 📚 Repository Contents

### 🔬 Jupyter Notebooks

This repository includes two primary Jupyter notebooks for single-edge notch test simulations:

1. **AT1 Model Implementation**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farhadkama/USACM-2026-Short-Course-The-Phase-Field-Approach-to-Brittle-Fracture/blob/main/codes/Single_edge_notch_test_at1.ipynb)

2. **Phase-Field Model Implementation**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farhadkama/USACM-2026-Short-Course-The-Phase-Field-Approach-to-Brittle-Fracture/blob/main/codes/Single_edge_notch_test_phasefield.ipynb)

These notebooks are designed to:
- run seamlessly in Google Colab (as demonstrated in the course),
- execute in local Jupyter environments,
- provide step-by-step implementation guidance, and
- include detailed comments and explanations.

---

### 🛠️ Installation Guide

We provide the installation guide in two formats:

- **PDF:**  
  [`installation-guide/FEniCSx_Installation_Guide.pdf`](installation-guide/FEniCSx_Installation_Guide.pdf)

- **Markdown (AI/agent-readable):**   
  [`installation-guide/FEniCSx_Installation_Guide.md`](installation-guide/FEniCSx_Installation_Guide.md)

The guide covers:
- FEniCSx installation on **Windows 11** using Docker
- FEniCSx installation on **macOS** using Conda
- **ParaView** installation for post-processing and visualization

> **Note:** You do **not** need to install FEniCSx to run the notebooks in this repository.  
> You can run them directly in **Google Colab** by clicking the **“Open in Colab”** badges above,  
> or by opening notebook files in your own Google account via [Google Colab](https://colab.research.google.com/).

---

---

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)
1. Click an **“Open in Colab”** badge above.
2. Follow the setup cells in the notebook.
3. Run cells sequentially.

### Option 2: Local Installation
1. Follow the installation guide for your operating system.
2. Clone this repository:
   ```bash
   git clone https://github.com/farhadkama/USACM-2026-Short-Course-The-Phase-Field-Approach-to-Brittle-Fracture.git
   ```
3. Navigate to the repository directory.
4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
5. Open and run the desired notebook.

---

## 🔗 Related Resources

### Educational Phase-Field Repository
**[FEniCSx_Kamarei_Kumar_Lopez-Pamies](https://github.com/farhadkama/FEniCSx_Kamarei_Kumar_Lopez-Pamies)**  
An educational repository focused on learning phase-field modeling from fundamentals to advanced concepts [1,2].

### Benchmark Problems Repository
**[FEniCSx_Kamarei_Lopez-Pamies](https://github.com/farhadkama/FEniCSx_Kamarei_Lopez-Pamies)**  
A specialized repository containing benchmark problems and advanced modeling approaches [2].

---

## 📩 Contact

For inquiries, please contact:  
- [kamarei2@illinois.edu](mailto:kamarei2@illinois.edu)

You may also reach out to my Ph.D. advisor:  
- [pamies@illinois.edu](mailto:pamies@illinois.edu)

---

## 📖 References

[1] Kumar, A., Francfort, G.A., Lopez-Pamies, O. (2018). *Fracture and healing of elastomers: A phase-transition theory and numerical implementation*. Journal of the Mechanics and Physics of Solids, 112, 523–551. [PDF](http://pamies.cee.illinois.edu/assets/pdf/JMPS2018.pdf)

[2] Kamarei, F., Kumar, A., Lopez-Pamies, O. (2024). *The poker-chip experiments of synthetic elastomers explained*. Journal of the Mechanics and Physics of Solids, 188, 105683. [PDF](http://pamies.cee.illinois.edu/Publications_files/JMPS2004b.pdf)

[3] Kamarei, F., Zeng, B., Dolbow, J.E., Lopez-Pamies, O. (2026). *Nine circles of elastic brittle fracture: A series of challenge problems to assess fracture models*. Computer Methods in Applied Mechanics and Engineering, 448, 118449. [PDF](http://pamies.cee.illinois.edu/Publications_files/CMAME2026.pdf)
