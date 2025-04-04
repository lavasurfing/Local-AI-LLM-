# vector search , vector store, data supply for llm 

# This model will take txt and conver it into embedding model
from langchain_ollama import OllamaEmbeddings

# vector store
from langchain_chroma import Chroma

# Doument for database 
from langchain_core.documents import Document

import os

import pandas as pd


# bring in cs file
df = pd.read_csv("realistic_restaurant_reviews.csv")

# create embedding for DB
embeddibngs = OllamaEmbeddings(model="mxbai-embed-large")

# location of local db
db_location = "./chroma_langchain_db"

# checking location db
add_document = not os.path.exists(db_location)

# converting into documents
if add_document:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + ""  + row["Review"],
            metadata = {
                "rating" : row["Rating"], "date" : row["Date"]
            },
            id=str(i)  
        )
        ids.append(str(i))
        documents.append(document)
        
# creating vector store

vector_store = Chroma(
    collection_name='restaurant_reviews',
    persist_directory=db_location,
    embedding_function=embeddibngs
)
# adding data if data not avilable
if add_document:
    vector_store.add_documents(documents=documents, ids=ids)
    
    
# Retriver of data 
retriver = vector_store.as_retriever(
    search_kwargs = {"k": 5}
)



