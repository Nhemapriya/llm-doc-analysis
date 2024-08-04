from typing import Dict

import pandas as pd
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, BertForQuestionAnswering, pipeline
from snowflake.snowpark.session import Session

from io import BytesIO

st.title("Code - Text & Everything next !")


# @st.cache(allow_output_mutation=True)
def extract_text_from_pdfs(pdf_files):
    df = pd.DataFrame(columns=["file", "text"])
    for pdf_file in pdf_files:
        with BytesIO(pdf_file.read()) as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text
            df = df.append({"file": pdf_file.name, "text": text}, ignore_index=True)
    # Return the data frame
    return df


def preprocess_text(text_list):
    processed_text = []
    for text in text_list:
        num_words = len(text.split(" "))
        if num_words > 10:  
            processed_text.append(text)
    return processed_text


def remove_short_sentences(df):
    df["sentences"] = df["sentences"].apply(preprocess_text)
    return df


# @st.cache_data(allow_output_mutation=True)
def get_relevant_texts(df, topic):
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    model_embedding.save("all-MiniLM-L6-v2")
    cosine_threshold = 0.1 
    queries = topic  
    results = []
    for i, document in enumerate(df["sentences"]):
        sentence_embeddings = model_embedding.encode(document)
        query_embedding = model_embedding.encode(queries)
        for j, sentence_embedding in enumerate(sentence_embeddings):
            distance = cosine_similarity(
                sentence_embedding.reshape((1, -1)), query_embedding.reshape((1, -1))
            )[0][0]
            sentence = df["sentences"].iloc[i][j]
            results += [(i, sentence, distance)]
    results = sorted(results, key=lambda x: x[2], reverse=True)
    del model_embedding
    texts = []
    for idx, sentence, distance in results:
        if distance > cosine_threshold:
            print('yes')
            text = sentence
            texts.append(text)
    context = "".join(texts)
    return context


@st.cache_data
def get_pipeline():
    modelname = "deepset/bert-base-cased-squad2"
    model_qa = BertForQuestionAnswering.from_pretrained(modelname)
    # model_qa.save_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained("tokenizer-deepset")
    # tokenizer.save_pretrained("tokenizer-" + modelname)
    qa = pipeline("question-answering", model=model_qa, tokenizer=tokenizer)
    return qa


def answer_question(pipeline, question: str, context: str) -> Dict:
    input = {"question": question, "context": context}
    return pipeline(input)


@st.cache_data
def create_context(df):
    df["sentences"] = df["text"].apply(
        lambda long_str: long_str.replace("\n", " ").split(".")
    )
    df = remove_short_sentences(df)

    context = get_relevant_texts(df, topic)
    return context


@st.cache_data
def start_app():
    with st.spinner("Loading model. Please wait..."):
        # context = create_context()
        pipeline = get_pipeline()
    return pipeline


pdf_files = st.file_uploader(
    "Upload pdf file", type=["pdf"], accept_multiple_files=True
)

connection_parameters = {
    #define the params
   }
session = Session.builder.configs(connection_parameters).create()

if pdf_files:
    with st.spinner("processing pdf..."):
        df = extract_text_from_pdfs(pdf_files)
    topic = st.text_input("Topic to search for : ")
    question = st.text_input("Question in here : ")
    # topic = 'quarter closure'
    # question = 'when did the quarter end for the company ?'

    if question != "":
        with st.spinner("Searching..."):
            year = pdf_files[0].name[:8]
            context = create_context(df)
            qa_pipeline = start_app()
            answer = answer_question(qa_pipeline, question, context)
            print(answer)
            load_data =[(topic, question, answer, answer["answer"], context[answer["start"]-20:answer["end"]+20], year)]
            load_df = pd.DataFrame(load_data, columns=["prompt_topic", "question", "prompt_response", "answer", "prompt_context", "DATA_ID"])
            st.write(answer)
            session.write_pandas(load_df, table_name='RAW_FINANCE_DATA', auto_create_table=True, overwrite=False)
        del qa_pipeline
        del context