import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain_community.llms import Ollama
from fuzzywuzzy import fuzz

DB_CHROMA_PATH = 'vectorstore/db_chroma'

custom_prompt_template = """
you are an chatbot that answers my query regarding Music Blocks. 8 to 10 years children should also be capable of understanding your answer so just divide your answer into points(dont write in the answer this point). If any question the user asks not related to music or music blocks then just say This is the music blocks chatbot. When the user asks a question like generate a lesson plan of any particular topic then follow the structure provided. 

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain

# Loading the model
def load_llm():
    llm = Ollama(model="llama3")
    return llm

# QA Model Function
def qa_bot():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Define appreciation and discouragement phrases
APPRECIATION_PHRASES = ["thanks", "thank you", "great", "good job", "well done", "appreciate it", "good"]
DISCOURAGEMENT_PHRASES = ["not helpful", "bad", "poor", "wrong", "incorrect", "disappointed"]

# Define response templates for appreciation and discouragement
APPRECIATION_RESPONSE = "You're welcome! I'm glad I could help. If you have more questions, feel free to ask."
DISCOURAGEMENT_RESPONSE = "I'm sorry to hear that you are not satisfied with my response. Could you please provide more details or ask another question so I can assist you better."

# Check if the input is an appreciation or discouragement phrase using fuzzy matching
def detect_sentiment(message):
    for phrase in APPRECIATION_PHRASES:
        if message.lower() == phrase.lower() or fuzz.ratio(message.lower(), phrase.lower()) > 90:  # 90% similarity threshold
            return 'appreciation'
    for phrase in DISCOURAGEMENT_PHRASES:
        if message.lower() == phrase.lower() or fuzz.ratio(message.lower(), phrase.lower()) > 90:  # 90% similarity threshold
            return 'discouragement'
    return None

# Define greetings and fuzzy matching for greetings
greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]

def is_greeting(user_input):
    user_input = user_input.strip().lower()
    for greeting in greetings:
        if user_input == greeting or fuzz.ratio(user_input, greeting) > 90:  # 90% similarity threshold
            return True
    return False

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    cl.user_session.set("history", [])
    msg.content = "Hi, Welcome to Music Blocks! What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)
    
# Define the conversation chain function
def conversation_chain(history, user_input):
    # Combine the history and user input to form a context-aware prompt
    context = "\n".join([f"User: {entry}" for entry in history])
    context += f"\nUser: {user_input}"
    
    return context

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip().lower()
    
    # Get conversation history from user session
    history = cl.user_session.get("history", [])
    
    # Append current user input to conversation history
    history.append(user_input)
    cl.user_session.set("history", history)
    
    # Check for sentiment
    sentiment = detect_sentiment(user_input)
    
    if sentiment == 'appreciation':
        await cl.Message(content=APPRECIATION_RESPONSE).send()
        return
    elif sentiment == 'discouragement':
        await cl.Message(content=DISCOURAGEMENT_RESPONSE).send()
        return

    # Check for greetings
    if is_greeting(user_input):
        await cl.Message(content="Hello! How can I assist you today?").send()
        return

    # Use conversation chain to form context-aware input
    context = conversation_chain(history, user_input)

    # If no sentiment or greeting detected, proceed with QA
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(context, callbacks=[cb])
    answer = res["result"]

    # Send the response back to the user
    await cl.Message(content=answer).send()