# Task 1: News Topic Classifier Using BERT

An end-to-end Natural Language Processing (NLP) pipeline that fine-tunes a pre-trained **BERT (`bert-base-uncased`)** model to classify news headlines into four distinct categories using the Hugging Face `transformers` library, evaluates its performance, and provides an interactive web application built with **Gradio**.

---

## 🎯 Objective
The primary goal of this project is to apply transfer learning techniques to a sequence classification task. Specifically:
* Fine-tune the pre-trained `bert-base-uncased` model on the **AG News Dataset** (loaded from the official `SetFit/ag_news` repository).
* Classify news headlines into one of four topics:
  1. **World** (Class 0)
  2. **Sports** (Class 1)
  3. **Business** (Class 2)
  4. **Sci/Tech** (Class 3)
* Evaluate the model's accuracy, precision, recall, and F1-score.
* Deploy the model using a lightweight, interactive Gradio interface for live real-time inference.

---

## ⚙️ Technologies & Tools Used
* **Deep Learning Framework:** PyTorch (`torch`)
* **Hugging Face Suite:** `transformers` (Tokenizers, Sequence Classification Model, Trainer API), `datasets`, `evaluate`
* **Machine Learning & Evaluation:** `scikit-learn` (for class distribution analysis and classification report generation)
* **Deployment & UI:** `gradio`
* **Environment:** Google Colab / Jupyter Notebooks (with T4 GPU acceleration)

---

## 🛠️ Methodology & Approach

### 1. Environment Configuration
To support efficient transformer training, the execution environment utilizes a hardware-accelerated runtime (specifically Google Colab's **T4 GPU**). Essential libraries are installed silently to keep the notebook clean:
```bash
pip install transformers[torch] datasets evaluate scikit-learn accelerate gradio
