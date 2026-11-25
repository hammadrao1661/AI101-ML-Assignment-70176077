# AI-101 Machine Learning Assignment: Training Machine Learning Models on Three Data Modalities

**The University of Lahore**  
**Department of Intelligent Systems**  
**Course:** Introduction to Artificial Intelligence (AI-101) | Fall 2026  
**Course Instructor:** Dr. Muhammad Awais, AI Architect  

---

## Student Information

| Student Name | Roll Number | Section | GitHub Repository |
| :--- | :--- | :--- | :--- |
| **Muhammad Ali Abid** | **70175812** | **B** | [AI101-ML-Assignment-70175812](https://github.com/hammadrao1661/AI101-ML-Assignment-70175812) |

---

## Tasks Summary (All 9 Tasks Completed for Full Marks + Bonus)

All 9 tasks across the three data modalities (Numerical, Mixed Numerical + Text, and Sequence-to-Sequence) were completed, including executed Jupyter notebooks, figures, data documentation, LaTeX source code, and 2-page double-column IEEE conference reports.

### Primary Graded Tasks (One per Pool)
- **Pool A (Numerical):** Task A2 — Concrete Compressive Strength Prediction
- **Pool B (Mixed Numerical + Text):** Task B1 — Fraudulent Job Postings Detection
- **Pool C (Unstructured Seq2Seq):** Task C1 — English-to-Urdu Neural Machine Translation

### Bonus Tasks (All 6 Tasks Attempted)
- **Pool A:** Task A1 (Breast Cancer Diagnosis) & Task A3 (Dry Bean Variety Recognition)
- **Pool B:** Task B2 (Wine Critic Score Prediction) & Task B3 (Clothing Review Star Rating)
- **Pool C:** Task C2 (Abstractive News Summarisation) & Task C3 (Image Captioning)

---

## Task Deliverables Matrix

| Pool | Task ID | Task Title | Primary Metric | Executed Notebook | Report PDF (2-Page IEEE) | LaTeX Source |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **A** | **A1** | Breast Cancer Diagnosis | Recall & F1, ROC-AUC | [Notebook](A1_Breast_Cancer_Diagnosis/A1_notebook.ipynb) | [PDF Report](A1_Breast_Cancer_Diagnosis/report/70175812_A1_Report.pdf) | [LaTeX Source](A1_Breast_Cancer_Diagnosis/report/source/70175812_A1_Report.tex) |
| **A** | **A2** | Concrete Compressive Strength *(Graded)* | RMSE, MAE, R² | [Notebook](A2_Concrete_Compressive_Strength/A2_notebook.ipynb) | [PDF Report](A2_Concrete_Compressive_Strength/report/70175812_A2_Report.pdf) | [LaTeX Source](A2_Concrete_Compressive_Strength/report/source/70175812_A2_Report.tex) |
| **A** | **A3** | Dry Bean Variety Recognition | Macro-F1 | [Notebook](A3_Dry_Bean_Variety/A3_notebook.ipynb) | [PDF Report](A3_Dry_Bean_Variety/report/70175812_A3_Report.pdf) | [LaTeX Source](A3_Dry_Bean_Variety/report/source/70175812_A3_Report.tex) |
| **B** | **B1** | Fraudulent Job Postings Detection *(Graded)* | F1 (fraud), PR-AUC | [Notebook](B1_Fraudulent_Job_Postings/B1_notebook.ipynb) | [PDF Report](B1_Fraudulent_Job_Postings/report/70175812_B1_Report.pdf) | [LaTeX Source](B1_Fraudulent_Job_Postings/report/source/70175812_B1_Report.tex) |
| **B** | **B2** | Wine Critic-Score Prediction | MAE, RMSE, R² | [Notebook](B2_Wine_Critic_Score_Prediction/B2_notebook.ipynb) | [PDF Report](B2_Wine_Critic_Score_Prediction/report/70175812_B2_Report.pdf) | [LaTeX Source](B2_Wine_Critic_Score_Prediction/report/source/70175812_B2_Report.tex) |
| **B** | **B3** | Clothing Review Star Ratings | Macro-F1, Ordinal MAE | [Notebook](B3_Clothing_Review_Star_Rating/B3_notebook.ipynb) | [PDF Report](B3_Clothing_Review_Star_Rating/report/70175812_B3_Report.pdf) | [LaTeX Source](B3_Clothing_Review_Star_Rating/report/source/70175812_B3_Report.tex) |
| **C** | **C1** | English→Urdu Machine Translation *(Graded)* | sacreBLEU, chrF | [Notebook](C1_English_to_Urdu_Machine_Translation/C1_notebook.ipynb) | [PDF Report](C1_English_to_Urdu_Machine_Translation/report/70175812_C1_Report.pdf) | [LaTeX Source](C1_English_to_Urdu_Machine_Translation/report/source/70175812_C1_Report.tex) |
| **C** | **C2** | Abstractive News Summarisation | ROUGE-1/2/L | [Notebook](C2_Abstractive_News_Summarisation/C2_notebook.ipynb) | [PDF Report](C2_Abstractive_News_Summarisation/report/70175812_C2_Report.pdf) | [LaTeX Source](C2_Abstractive_News_Summarisation/report/source/70175812_C2_Report.tex) |
| **C** | **C3** | Image Captioning (CNN-LSTM) | BLEU-1...4 | [Notebook](C3_Image_Captioning/C3_notebook.ipynb) | [PDF Report](C3_Image_Captioning/report/70175812_C3_Report.pdf) | [LaTeX Source](C3_Image_Captioning/report/source/70175812_C3_Report.tex) |

---

## Repository Structure

```
AI101-ML-Assignment-70175812/
|-- README.md
|-- AI_DISCLOSURE.md
|-- requirements.txt
|-- .gitignore
|-- A1_Breast_Cancer_Diagnosis/
|   |-- A1_notebook.ipynb
|   |-- report/
|   |   |-- 70175812_A1_Report.pdf
|   |   `-- source/
|   |       `-- 70175812_A1_Report.tex
|   |-- figures/
|   `-- data/README.md
|-- A2_Concrete_Compressive_Strength/
|   |-- A2_notebook.ipynb
|   |-- report/
|   |   |-- 70175812_A2_Report.pdf
|   |   `-- source/
|   |       `-- 70175812_A2_Report.tex
|   |-- figures/
|   `-- data/README.md
|-- A3_Dry_Bean_Variety/
|-- B1_Fraudulent_Job_Postings/
|-- B2_Wine_Critic_Score_Prediction/
|-- B3_Clothing_Review_Star_Rating/
|-- C1_English_to_Urdu_Machine_Translation/
|-- C2_Abstractive_News_Summarisation/
`-- C3_Image_Captioning/
```

---

## Instructions to Run the Code

### 1. Environment Setup
Install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Executing the Notebooks
Each folder contains a standalone Jupyter notebook (`<TaskID>_notebook.ipynb`). You can execute any notebook directly via Jupyter Lab / Notebook or run it headlessly:
```bash
jupyter nbconvert --to notebook --execute A2_Concrete_Compressive_Strength/A2_notebook.ipynb
```

### 3. Data Download
All datasets are accessed programmatically via `ucimlrepo`, `sklearn.datasets`, or `datasets` (Hugging Face). Detailed instructions and mirror URLs are documented in each task's `data/README.md`.
