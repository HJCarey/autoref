
import pickle
from langchain import LLMChain
import openai
import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.prompts import PromptTemplate

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_qa_prompt():
    # Setup our template to prompt the model to be a roller derby referee
    start_template = """
    You are an expert referee for the sport Roller Derby. Your goal is to help the user understand
    the rules of Roller Derby, as well as assist in adjuticating any disputes confusing situations
    that might occur where the rules are unclear or tricky to apply. You are friendly and helpful,
    but also kind of sassy, because this is Roller Derby after all.
    You are given the following extracted parts of the official roller derby rulebook. Please
    provide a conversational answer.
    Please give references to the rulebook when possible.
    If you don't know the answer to a question, just say "Hmm, I'm not sure.".
    Don't try to make up an answer.

    Question: {question}
    ==========
    {context}
    ==========
    Autoref:"""
    QA = PromptTemplate(template=start_template, input_variables=["question", "context"])

    return QA


def get_condense_prompt():
    # Setup our intermediate template that runs inbetween questions.
    
    intermediate_template = """
    Given the following conversation and a follow up question, rephrease the following
    question to be a standalone question. Remember you are a roller derby referee, and
    your goal is to provide help to the user regarding roller derby questions.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    # condense_question_prompt = PromptTemplate.from_template(intermediate_template)
    condense_question_prompt = PromptTemplate(template=intermediate_template, input_variables=["chat_history", "question"])

    return condense_question_prompt


def get_chain(vectorstore):
    # Build the model
    model_version = "gpt-3.5-turbo"
    condense_question_prompt = get_condense_prompt()
    QA = get_qa_prompt()
    llm = ChatOpenAI(temperature=0.3, model_name=model_version)
    question_generator = LLMChain(llm=llm,
                                  prompt=condense_question_prompt)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=QA)
    
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwars={"k": 4, "include_metadata": True})
    
    chain = ConversationalRetrievalChain(
        memory=memory,
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        get_chat_history=lambda h : h,
        verbose=False
    )

    return chain


def load_data(document_path):
    # Load data
    loader = UnstructuredPDFLoader(document_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Load our data into embeddings
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    return db

def main():
    # with open("vectors_roller_derby.pkl", "rb") as f:
    #     vectorstore = pickle.load(f)

    # Recreate the data
    vectorstore = load_data("../documents/wftda_rules.pdf")

    qa_chain = get_chain(vectorstore)
    
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")
        chat_history = []

        def user(user_message, history):
            print("User message:", user_message)
            print("Chat history:", history)
            
            # Get response from model
            response = qa_chain({"question": user_message, "chat_history": history})
            # Append user message and response to chat history
            history.append((user_message, response["answer"]))
            print("Updated chat history:", history)
            return gr.update(value=""), history
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
        clear.click(lambda: None, None, chatbot, queue=False)
        
    demo.launch(debug=True)
    # chat_history = []
    # while True:
    #     print("Your question:")
    #     question = input()
    #     result = qa_chain({"question": question, "chat_history": chat_history})
    #     chat_history.append((question, result["answer"]))
    #     print(f"AI: {result['answer']}")

    # chat_history = []
    # query = "What is the shape of the track?"
    # result = qa_chain({"question": query, "chat_history": chat_history})
    # print(f"Human: {query}")
    # print(f"AI: {result['answer']}")

    # chat_history = [(query, result["answer"])]
    # query = "What is the question I just asked you?"
    # result = qa_chain({"question": query, "chat_history": chat_history})
    # print(f"Human: {query}")
    # print(f"AI: {result['answer']}")

    # return 0

if __name__ == "__main__":
    main()

    # while True:
    #     print("Your question:")
    #     question = input()
    #     result = qa_chain({"question": question, "chat_history": chat_history})
    #     chat_history.append((question, result["answer"]))
    #     print(f"AI: {result['answer']}")