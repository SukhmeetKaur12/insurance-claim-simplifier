---

# **Simplifying Healthcare Insurance Claims with AI**

In today’s complex healthcare system, navigating insurance claims can be a daunting task for patients. Policies are filled with technical jargon, coverage clauses are often ambiguous, and the approval process can feel like a gamble. The result? Patients either delay treatment or face unexpected financial burdens.

This is the problem we set out to solve with the **Insurance Claim Simplifier for Healthcare** — a smart, AI-powered assistant that not only explains policies but also predicts claim approval chances.

---

## **The Problem**

Healthcare insurance policies are notorious for being complicated. Patients struggle to:

* Understand what their policy actually covers.
* Identify which medical procedures are approved.
* Navigate the claim submission process.
* Predict whether a claim will be accepted or denied.

For clinics and Third Party Administrators (TPAs), this translates into:

* Longer processing times.
* Increased workload answering repetitive policy questions.
* Dissatisfied patients.

---

## **The Solution**

We built a **two-part AI system** combining:

1. **A RAG-based Chatbot** — Retrieval-Augmented Generation ensures the chatbot provides factually grounded answers by searching through actual insurance documents before generating a response.
2. **A Claim Prediction Model** — A machine learning model that forecasts whether a claim will be approved, based on patient and claim data.

This solution is designed as a **B2B HealthTech tool**, allowing clinics and TPAs to:

* Offer instant, accurate policy guidance to patients.
* Reduce back-and-forth with insurance companies.
* Improve patient trust.

---

## **How It Works**

### **1. RAG for Policy Understanding**

* **Data Input**: PDF insurance policy documents (from providers or patients).
* **Processing**:

  * Extracts text using PyMuPDF.
  * Splits text into smaller, meaningful chunks.
  * Converts chunks into embeddings using Hugging Face models.
  * Stores these embeddings in a FAISS vector database.
* **Query Handling**:
  When a user asks a question, the system retrieves the most relevant chunks and feeds them into a language model (`flan-t5`) to generate a clear, policy-specific answer.

---

### **2. Machine Learning for Claim Approval Prediction**

* **Dataset**: Contains patient demographics, health metrics (BMI, blood pressure), lifestyle factors (smoker/diabetic), claim amount, and approval status.
* **Processing**:

  * Missing values are handled.
  * Categorical values are encoded.
  * Data is split into training and testing sets.
* **Model**: A `RandomForestClassifier` trained to predict approval status with high accuracy.
* **Usage**: Clinics can input a patient’s details to instantly get an approval likelihood.

---

## **Tech Stack**

* **Frontend**: Streamlit (interactive web UI).
* **Backend**: Python.
* **RAG Pipeline**: LangChain + Hugging Face models.
* **ML Model**: Scikit-learn (Random Forest).
* **Vector Database**: FAISS.
* **PDF Parsing**: PyMuPDF.
* **Environment**: Google Colab for training, Streamlit Cloud for deployment.

---

## **Benefits**

* **For Patients**: Instant clarity on policy coverage and claim status.
* **For Clinics & TPAs**: Reduced administrative work, faster processing, and improved satisfaction.
* **For Insurers**: Fewer resubmissions, improved customer experience.

---

## **Future Scope**

* Multilingual policy explanation for global reach.
* Integration with hospital management systems.
* Real-time claim tracking.
* Larger, more diverse datasets for improved accuracy.

---

This project represents a step towards **making healthcare insurance more transparent, predictable, and user-friendly**. By combining RAG for document understanding with ML for predictive analytics, we’ve created a tool that bridges the gap between insurance complexity and patient clarity.

---


