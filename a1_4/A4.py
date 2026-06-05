import os
import json
import urllib.request
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
import warnings  # <-- ADD THIS IMPORT
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# LangChain imports
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import re

DATA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json"
DATA_FILE = "ori_pqal.json"

def download_data():
    """
    Part 1: Task 1.1 - Downloading the PubMedQA dataset
    """
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print("Dataset file already exists.")

def prepare_datasets():
    """
    Part 1: Collect and clean the 'documents' and 'questions' dataframes
    """
    print("Preparing datasets...")
    tmp_data = pd.read_json(DATA_FILE).T
    
    # Filter out 'maybe' and keep only yes/no decisions
    tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

    # Build contexts + long answer text representations
    documents = pd.DataFrame({
        "abstract": tmp_data.apply(lambda row: " ".join(row.CONTEXTS + [row.LONG_ANSWER]), axis=1),
        "year": tmp_data.YEAR
    })
    
    questions = pd.DataFrame({
        "question": tmp_data.QUESTION,
        "year": tmp_data.YEAR,
        "gold_label": tmp_data.final_decision,
        "gold_context": tmp_data.LONG_ANSWER,
        "gold_document_id": documents.index
    })
    print(f"Successfully loaded {len(documents)} documents and {len(questions)} questions.")
    return documents, questions

def configure_llm():
    """
    Part 2: Task 2.1 - Select and configure the language model
    """
    print("Configuring language model...")
    # Using Llama-3.2-1B-Instruct as an example (requires HF_TOKEN environment variable).
    # If you run out of memory or want an open alternative, you can switch to "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_id = "Qwen/Qwen3.5-4B" 
    
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device execution target: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

    model = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
            "max_length": None,
            "temperature": 0.1,
            "return_full_text": False
        },
        device=device
    )
    return model

def setup_vector_store(documents):
    """
    Part 3: Tasks 3.1, 3.2, & 3.3 - Embeddings, Chunking, and Chroma DB Setup
    """
    print("Setting up vector store...")
    # Task 3.1: Define Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Embeddings dimension shape verification check
    sample_emb = embedding_model.embed_query("What is programmed cell death?")
    print(f"Embedding dimensions shape verified: ({len(sample_emb)},)")

    # Task 3.2: Chunking documents with original IDs saved to metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    metadatas = [{"id": idx} for idx in documents.index]
    chunks = text_splitter.create_documents(
        texts=documents.abstract.tolist(),
        metadatas=metadatas
    )
    print(f"Documents converted into {len(chunks)} structural text chunks.")

    # Task 3.3: Instantiate Chroma Vector Store with explicit Cosine Similarity space
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store

def build_rag_pipeline(model, vector_store):
    """
    Part 4: Task 4.1 - Implementing the System using Option B (LCEL Chain)
    """
    print("Building RAG workflow pipeline (Option B: LCEL)...")
    # Configure retriever to fetch only 1 document per prompt as per assignment instructions
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # System template designed for exact classification outputs
    template = """You are a medical research expert. Analyze the context and answer the question.
    You MUST provide your final binary choice wrapped inside square brackets, like [yes] or [no].

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Internal sequential extraction chain
    chain = (
        prompt
        | model
        | StrOutputParser()
    )
    
    # Main runnable parallel layout mapping inputs and retaining source contexts
    runnable_parallel_object = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    
    rag_chain = runnable_parallel_object.assign(answer=chain)
    return rag_chain

def parse_answer(text):
    match = re.search(r"\[(yes|no)\]", text)
    
    if match:
        return match.group(1)  
        
    if text.startswith("yes") or text.endswith("yes"):
        return "yes"
    elif text.startswith("no") or text.endswith("no"):
        return "no"
    
    return "invalid"

def evaluate_system(rag_chain, model, questions, sample_size=30):
    """
    Part 5: Tasks 5.1 & 5.2 - High-level evaluation and Detailed inspection
    """
    print(f"Starting pipeline evaluation execution loop on a random sample of {sample_size} records...")
    eval_sample = questions.sample(n=sample_size, random_state=42)
    
    rag_results = []
    baseline_results = []
    gold_labels = []
    gold_docs_fetched = []
    
    baseline_template = """You are a medical research expert. Analyze the context and answer the question.
    You MUST provide your final binary choice wrapped inside square brackets, like [yes] or [no].

    Question:
    {question}

    Answer:"""


    for idx, row in eval_sample.iterrows():
        q_text = row['question']
        gold_ans = row['gold_label'].strip().lower()
        gold_doc_id = row['gold_document_id']
        
        fetched_id = None
        final_rag_ans = "invalid"
        
        # --- 1. Evaluate RAG Pipeline ---
        try:
            res = rag_chain.invoke(q_text)
            
            retrieved_docs = res['context']
            if retrieved_docs:
                fetched_id = retrieved_docs[0].metadata.get('id')
                
            ans = res['answer'].strip().lower()
            final_rag_ans = parse_answer(ans)
            
        except Exception as e:
            print(f"Execution boundary error on RAG Row {idx}: {e}")            

        # Securely track retrieval quality for this step
        gold_docs_fetched.append(1 if fetched_id == gold_doc_id else 0)
        # --- 2. Evaluate Baseline Pipeline (No Context) ---
        try:
            base_prompt = baseline_template.format(question=q_text)
            base_ans_raw = model.invoke(base_prompt).strip().lower()
            final_base_ans = parse_answer(base_ans_raw)
        except Exception as e:
            final_base_ans = "invalid"
            print(f"Execution boundary error on Baseline Row {idx}: {e}")
            
        rag_results.append(final_rag_ans)
        baseline_results.append(final_base_ans)
        gold_labels.append(gold_ans)
        
    # Collate into summary structure
    df_res = pd.DataFrame({
        "gold": gold_labels,
        "rag": rag_results,
        "baseline": baseline_results,
        "gold_fetched": gold_docs_fetched
    })
    
    # Task 5.1: Calculate performance exclusively over conforming valid outputs
    valid_rag = df_res[df_res.rag.isin(["yes", "no"])]
    valid_base = df_res[df_res.baseline.isin(["yes", "no"])]
    
    print("\n" + "="*24 + " FINAL BENCHMARK PERFORMANCE " + "="*24)
    print(f"Total Sample Size: {sample_size}")
    
    print(f"\n[RAG Pipeline System]")
    print(f"  Valid Binary Responses: {len(valid_rag)} / {sample_size}")
    if len(valid_rag) > 0:
        print(f"  Accuracy Score (on valid): {accuracy_score(valid_rag.gold, valid_rag.rag):.4f}")
        print(f"  Macro F1-Score (on valid): {f1_score(valid_rag.gold, valid_rag.rag, average='macro'):.4f}")
        
    print(f"\n[Baseline Pipeline (Without Context)]")
    print(f"  Valid Binary Responses: {len(valid_base)} / {sample_size}")
    if len(valid_base) > 0:
        print(f"  Accuracy Score (on valid): {accuracy_score(valid_base.gold, valid_base.baseline):.4f}")
        print(f"  Macro F1-Score (on valid): {f1_score(valid_base.gold, valid_base.baseline, average='macro'):.4f}")
        
    print(f"\n[Retrieval Quality Inspection (Task 5.2)]")
    retrieval_recall = sum(gold_docs_fetched) / len(gold_docs_fetched)
    print(f"  Gold Document Top-1 Retrieval Accuracy (Recall@1): {retrieval_recall:.4f}")
    print("="*77 + "\n")

def main():
    # Execute structural sequence
    download_data()
    documents, questions = prepare_datasets()
    model = configure_llm()
    vector_store = setup_vector_store(documents)
    rag_chain = build_rag_pipeline(model, vector_store)
    
    # Benchmark evaluation execution block. 
    # Adjust sample_size as necessary depending on local machine processing capacities.
    evaluate_system(rag_chain, model, questions, sample_size=30)

if __name__ == "__main__":
    main()

#Task 3.2:
#Chunking breaks down long document abstracts into smaller, manageable segments using a tool like RecursiveCharacterTextSplitter 
#to accommodate the strict token limits of embedding models. These design choices directly dictate the overall quality of a 
#RAG system: chunks that are too large introduce irrelevant noise and dilute semantic precision during vector search, 
#while chunks that are too small fragment