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
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10, max_value=60, value=25)
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
