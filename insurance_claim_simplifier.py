import os

pdf_folder = r"docs/docs/"

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
def chunk_text(text, chunk_size=500, overlap=50):
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
os.makedirs("data", exist_ok=True)

# Save updated data to new CSV

df.to_csv("data/insurance_data_with_approval.csv.csv", index=False)
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

folder_path = r"docs/docs/"
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

