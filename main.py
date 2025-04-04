from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import retriver

model = OllamaLLM(model ="llama3.2")

template = '''
    You are an expert in answering questions about a pizza resturant
    Here are some relevant reviews : {reviews}
    
    Here is the question to answer : {question}
'''

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

print('Start chat')
while True:
    print('\n--xx---------------xx----------xx-----------xx----------')
    
    question = input("Ask your question (q to quit) :  ")
    print("\n")
    if question == 'q':
        break
    reviews = retriver.invoke(question)
    
    result = chain.invoke({"reviews": reviews,  "question": question})

    print(result)