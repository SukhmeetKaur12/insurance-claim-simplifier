import os

pdf_folder = r"docs/"

os.makedirs(pdf_folder, exist_ok=True)
import fitz  # PyMuPDF

def extract_text_from_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            doc_path = os.path.join(folder_path, filename)
            pdf_doc = fitz.open(doc_path)
            text = ""
            for page in pdf_doc:
                text += page.get_text()
            documents.append({
                "filename": filename,
                "content": text
            })
    return documents

raw_docs = extract_text_from_pdfs(pdf_folder)
print(f"Extracted {len(raw_docs)} documents")
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Apply to all docs
rag_chunks = []
for doc in raw_docs:
    chunks = chunk_text(doc["content"])
    for idx, chunk in enumerate(chunks):
        rag_chunks.append({
            "text": chunk,
            "source": doc["filename"],
            "chunk_id": idx
        })

print(f"Total chunks prepared for RAG: {len(rag_chunks)}")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [chunk["text"] for chunk in rag_chunks]
embeddings = model.encode(texts, show_progress_bar=True)
import faiss
import numpy as np

embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)

# Add embeddings to index
index.add(np.array(embeddings))
# Save metadata for each chunk alongside the FAISS index
metadata = [
    {
        "text": rag_chunks[i]["text"],
        "source": rag_chunks[i]["source"],
        "chunk_id": rag_chunks[i]["chunk_id"]
    }
    for i in range(len(rag_chunks))
]
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Load a publicly available LLM (lightweight model example)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
text_gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Use the previously loaded SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


rag_pipeline = pipeline("text2text-generation", model=text_gen_model, tokenizer=tokenizer, max_length=512)

def retrieve_context(query, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [metadata[i]["text"] for i in indices[0]]

def generate_answer(query, k=5):
    context_chunks = retrieve_context(query, k)
    context = "\n\n".join(context_chunks)

    prompt = f"""You are an insurance expert assistant. Based on the insurance policy documents below, answer the user’s question.

Policy Documents:
{context}

User Question: {query}

Answer:"""

    response = rag_pipeline(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    return response
query = "Is cancer treatment covered under my policy?"
answer = generate_answer(query)
print("Answer:", answer)
# Ensure embedding_model is defined and accessible
try:
    embedding_model.encode(["test"])
    print("embedding_model is correctly defined and has the encode method.")
except AttributeError:
    print("Error: embedding_model is not defined or does not have the encode method.")
except NameError:
    print("Error: embedding_model is not defined.")

# Now call the generate_answer function
query = "Is cancer treatment covered under my policy?"
answer = generate_answer(query)
print("Answer:", answer)

#TILL ABOVE WE HAVE BUILT THE RAG PIPELINE NOW WE SHALL BE PREDICTING THE INSURANCE CLAIM PREDICTER ML MODEL AS WELL

import pandas as pd

# Load dataset
df = pd.read_csv(r"data/insurance_data.csv")

# Preview
print(df.head())
import numpy as np
# Generate a synthetic 'Approved' column
# Example logic (customize as needed)
def is_approved(row):
    if row['claim'] > 15000:
        return 0
    if row['smoker'] == 'Yes' and row['claim'] > 10000:
        return 0
    if row['bmi'] > 40 or row['bmi'] < 16:
        return 0
    if row['age'] > 60 and row['claim'] > 12000:
        return 0
    if row['diabetic'] == 'Yes' and row['claim'] > 13000:
        return 0
    if row['bloodpressure'] > 140:
        return 0
    return 1

df['approved'] = df.apply(is_approved, axis=1)


# df["approved"] = df.apply(approve, axis=1) # This line caused the error
df.head(5)
import os

# Create the directory if it doesn't exist
os.makedirs("/mnt/data", exist_ok=True)

# Save updated data to new CSV
df.to_csv("/mnt/data/insurance_data_with_approval.csv", index=False)
df['age'].fillna(df['age'].median(), inplace=True)
df['region'].fillna(df['region'].mode()[0], inplace=True)

'''from sklearn.preprocessing import LabelEncoder

# Create a copy to avoid modifying original
df_encoded = df.copy()

# Drop columns that shouldn't be used for prediction
X = df.drop(['approved', 'claim'], axis=1)  # 'claim' is correlated and can cause data leakage
y = df['approved']

categorical_columns = ['gender', 'diabetic', 'smoker', 'region']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)


# Encode each column
le = LabelEncoder()
for col in categorical_columns:
    # Check if the column exists in the DataFrame before encoding
    if col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    else:
        print(f"Warning: Column '{col}' not found in df_encoded. Skipping encoding for this column.")
# Drop columns that shouldn't be used for prediction
X = df.drop(['approved', 'claim'], axis=1)  # 'claim' is correlated and can cause data leakage
y = df['approved']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming df is loaded already
# Drop irrelevant or leaking columns
X = df.drop(columns=['approved', 'PatientID', 'index', 'claim'])  # Also dropped 'index' as irrelevant
y = df['approved']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['gender', 'diabetic', 'smoker', 'region'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate predictions
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

folder_path = r"docs/"
documents = []

# Load all PDF files
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        print("Loading:", filename)
        pdf_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

print("Total documents loaded:", len(documents))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Split {len(documents)} PDFs into {len(chunks)} chunks.")
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Use HuggingFace embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Embed chunks and store in FAISS
db = FAISS.from_documents(chunks, embedding_model)

# Save the vector store to disk (optional but recommended)
db.save_local("faiss_insurance_claim_db")

print("✅ Vector store created and saved.")
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
from transformers import pipeline
import torch

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1 # Explicitly use CPU if GPU is not available
)
def get_claim_answer(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = llm(prompt, max_length=512, do_sample=False)[0]["generated_text"]
    return response
query = "Is MRI covered under the standard insurance policy?"
answer = get_claim_answer(query)

print("Answer:", answer)
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model_name = "google/flan-t5-base"  # or small/large based on resources
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# User query
question = "What does the insurance cover for surgery?"

# Retrieved chunks from FAISS
retrieved_context = "Your insurance covers hospitalization, surgical expenses, and pre-post hospitalization."

# Prepare prompt
prompt = f"Answer the question based on the context.\nContext: {retrieved_context}\nQuestion: {question}"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
output = model.generate(**inputs, max_length=100)
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print("Answer:", answer)
# main.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings # Still need to import to initialize embedding_model
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize embedding model (ensure this is done before loading the vectorstore)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load FAISS vectorstore - Pass the embedding function here
vectorstore = FAISS.load_local("faiss_insurance_claim_db", embeddings=embedding_model, allow_dangerous_deserialization=True)

# Load Flan-T5 model
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def get_context_from_query(query, k=3):
    """Retrieve top-k similar chunks from FAISS"""
    results = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

def generate_answer_flan_t5(query):
    """Generate answer using Flan-T5"""
    context = get_context_from_query(query)
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {query}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    output = model.generate(**inputs, max_length=128)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# --- Example Use ---
if __name__ == "__main__":
    query = input("Ask your insurance question: ")
    answer = generate_answer_flan_t5(query)
    print(f"\n✅ Answer: {answer}")

import streamlit as st
import pandas as pd
import joblib
import torch

# Import necessary components for RAG
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline


# Import necessary components for ML prediction
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier # Added import for the model used

# ------------------ Load RAG components ------------------
@st.cache_resource
def load_vectorstore():
    """Loads the FAISS vector store with insurance policy documents."""
    try:
        # Ensure allow_dangerous_deserialization=True is used for loading from pickle
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_insurance_claim_db", embeddings=embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

@st.cache_resource
def load_rag_pipeline():
    """Loads the Flan-T5 model and creates a HuggingFace pipeline for RAG."""
    try:
        model_name = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        # Using device map 'auto' to automatically handle GPU/CPU placement
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.3, # Corrected temperature parameter usage
            do_sample=True # Added do_sample for temperature to have effect
        )
        # Wrap the pipeline in LangChain's HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {e}")
        return None

# ------------------ Load ML Model ------------------
@st.cache_resource
def load_ml_model():
    """Loads the trained ML model for claim approval prediction."""
    try:
        # Ensure the model file exists and is correctly named
        # Assuming the model was saved as 'claim_model.pkl'
        model = joblib.load("claim_model.pkl")
        return model
    except FileNotFoundError:
        st.error("ML model file 'claim_model.pkl' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="🧠 Insurance Claim Simplifier", layout="centered")
st.title("🧠 Insurance Claim Simplifier")

# Radio button for feature selection
mode = st.sidebar.radio("Select Feature", ["📄 Ask Insurance Questions", "📊 Predict Claim Approval"])

# ------------------ RAG Tab ------------------
if mode == "📄 Ask Insurance Questions":
    st.subheader("📄 Ask a question about your insurance policy")

    vectorstore = load_vectorstore()
    flan_llm = load_rag_pipeline()

    if vectorstore and flan_llm:
        # Corrected RAG chain implementation using LangChain Expression Language (LCEL)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Define the prompt template
        prompt_template = PromptTemplate(
            template="Answer the question based on the context:\n{context}\n\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"],
        )

        # Construct the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | flan_llm
            | StrOutputParser()
        )

        user_query = st.text_input("Enter your question:")
        if user_query:
            with st.spinner("Searching your insurance policy..."):
                try:
                    # Use the invoke method for the LCEL chain
                    response = rag_chain.invoke(user_query)
                    st.success("Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.warning("RAG components could not be loaded. Please check the setup.")


# ------------------ Prediction Tab ------------------
else:
    st.subheader("📊 Predict Claim Approval")

    ml_model = load_ml_model()

    if ml_model:
        # Input fields for ML prediction
        # These input fields should match the features used to train the model
        # based on the notebook cells, the features were:
        # 'age', 'bmi', 'bloodpressure', 'children', 'gender_male',
        # 'diabetic_Yes', 'smoker_Yes', 'region_northwest', 'region_southeast', 'region_southwest'
        # We need to collect the raw values and then one-hot encode them
        age = st.number_input("Age", min_value=18, max_value=100.0, value=30.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=90)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        gender = st.selectbox("Gender", ["male", "female"])
        diabetic = st.selectbox("Diabetic?", ["Yes", "No"])
        smoker = st.selectbox("Smoker?", ["Yes", "No"])
        region = st.selectbox("Region", ["southeast", "northwest", "northeast", "southwest"]) # Assuming these are the regions in your data


        # When Predict button clicked
        if st.button("Predict Claim Approval"):
            try:
                # Create input DataFrame matching the training data structure
                input_data = pd.DataFrame([{
                    "age": age,
                    "bmi": bmi,
                    "bloodpressure": bloodpressure,
                    "children": children,
                    "gender": gender,
                    "diabetic": diabetic,
                    "smoker": smoker,
                    "region": region
                }])

                # One-hot encode categorical features to match training data
                # Ensure columns match the training data columns after one-hot encoding
                categorical_cols = ['gender', 'diabetic', 'smoker', 'region']
                input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

                # Align columns with training data columns
                # This is crucial to avoid errors during prediction if input columns don't match
                # You'll need to have access to the columns from your X_train after one-hot encoding
                # For demonstration, let's assume X_train_cols exists and contains the column names in the correct order
                # In a real scenario, you would save and load the list of training columns
                # For now, let's construct a plausible list based on previous steps
                # Ensure the order and names exactly match
                # Example plausible column order after one-hot encoding and dropping original/extra columns:
                training_columns = ['age', 'bmi', 'bloodpressure', 'children',
                                    'gender_male', 'diabetic_Yes', 'smoker_Yes',
                                    'region_northwest', 'region_southeast', 'region_southwest']

                # Add missing columns with default value 0 and reorder columns
                for col in training_columns:
                    if col not in input_data_encoded.columns:
                        input_data_encoded[col] = 0

                # Reorder columns to match the training data
                input_data_aligned = input_data_encoded[training_columns]


                # Make prediction
                prediction = ml_model.predict(input_data_aligned)[0]
                result = "✅ Likely Approved" if prediction == 1 else "❌ Likely Rejected"
                st.info(f"Result: {result}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("ML model could not be loaded. Prediction is not available.")
