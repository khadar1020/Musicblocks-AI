import torch
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain_community.llms import Ollama
from fuzzywuzzy import fuzz

DB_CHROMA_PATH = 'vectorstore/db_chroma'

custom_prompt_template = """
you are an chatbot that answers my query regarding Music Blocks. 8 to 10 years children should also be capable of understanding your answer so just divide your answer into points with numbers(dont write in the answer this point) and explain the answer in detail.If any question the user asks not related to music or music blocks then just say This is the music blocks chatbot only answers music blocks questions. When the user asks a question like generate a lesson plan of any particular topic then follow the structure provided. 

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

# Global variables
table_memory = {}
topic = ""

# Chainlit code
@cl.on_chat_start
async def start():
    global table_memory, topic
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    cl.user_session.set("history", [])
    msg.content = """NOTE - Please read the Readme file so that you efficiently use our Music Blocks AI.
                     Hi, welcome to the Music Blocks lesson plan creation assistant. How may I help you?"""
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
    global table_memory, topic
    user_input = message.content.strip().lower()
    
    # Get conversation history from user session
    history = cl.user_session.get("history", [])
    if "generate a lesson plan on" in user_input or "create a lesson plan on" in user_input:
        print()
    else:
        if not user_input.isdigit():
            history.append(user_input)
    cl.user_session.set("history", history)  # Update history in user session
    
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

    # Check if user input is a request to generate or create a lesson plan
    if "generate a lesson plan on" in user_input or "create a lesson plan on" in user_input:
        topic = user_input.split("lesson plan on")[1].strip()
        user_input = f"What are the best 6 real world songs that I can use to generate the lesson plan of {topic}? Give me the songs in a table the first column has index and second column has song and third column has why that song is helpful follow this order just give me  table dont give me any other information"
        # Use conversation chain to form context-aware input
        context = conversation_chain(history, user_input)
        chain = cl.user_session.get("chain")
        
        # Initiate QA call to get the response
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        res = await chain.acall(context, callbacks=[cb])
        answer = res["result"]
        table_memory['table'] = answer
        
        # Send the response to the user
        await cl.Message(content=answer).send()
            
        # Prompt the user to select a song index
        await cl.Message(content=f"Please give me the index of the song you want to use for generating the lesson plan on {topic}.").send()
        return

    # Check if user input is a number (1-6)
    if user_input.isdigit():
        selected_index = int(user_input)
        if 1 <= selected_index <= 6:
            selected_song = None
            # Extract song details from table_memory based on selected_index
            if 'table' in table_memory:
                table_content = table_memory['table']
                # | Song # | Song Name |
                lines = table_content.split('\n')
                for line in lines:
                    if line.startswith(f"| {selected_index} |"):
                        parts = line.split("|")
                        selected_song = parts[2].strip()  # Song Name
                        reason = parts[3].strip()        # Reason
                        break
            
                user_input = f"generate a lesson plan for {topic} based on the song {selected_song} follow the structure provided and try to explain the lesson plan in detail using the song and elaborate that song using that topic. Explain in detail in 1000 to 1500 words"
                context = conversation_chain(history, user_input)
                if not user_input.isdigit():
                    history.append(user_input)
                chain = cl.user_session.get("chain")
                
                # Initiate QA call to get the response
                cb = cl.AsyncLangchainCallbackHandler(
                    stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
                )
                cb.answer_reached = True
                res = await chain.acall(context, callbacks=[cb])
                answer = res["result"]
                
                # Send the response to the user
                await cl.Message(content=answer).send()
                return
        else:
            if 'table' in table_memory:
                await cl.Message(content="Please enter a number between 1 to 6.").send()
                return
            else:
                await cl.Message(content="I am a Music Blocks ChatBot").send()
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
