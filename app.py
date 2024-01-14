from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import openai
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from utils import *


st.set_page_config(
    page_title='HarmonyHaven Web App',
    layout='wide'
)

st.header(":blue[HarmonyHaven Chatbot]")
with st.sidebar:
    selected = option_menu(menu_title='Main menu',options=['About', 'Chatbot','Community Forum'], 
    icons=['house-fill', 'chat-fill','globe'],
    menu_icon="cast", default_index=0,)
    
if selected == "About":
    st.write(" ")
    st.header(":blue[Project Background]")
    st.write("""
             Mental health disorders, typically characterized by distress or impairment in vital areas of functioning, encompass
             a diverse range of conditions. Notably, our focus centers on three prevalent categories; Depression, Anxiety 
             Disorder and Bipolar Disorder. As of 2019, a substantial 301 million individuals grappled with anxiety 
             disorders, inclusive of 58 million children and adolescents. Depression affected 280 million people , with 23 
             million being children and adolescents, while 40 million individuals coped with bipolar disorder.
             Des pite the existence of effective prevention and treatment modalities , access to quality care remains limited
             for a significant portion of the global population. A concerning reality persists wherein stigma and discrimination 
             are frequently encountered by individulas navigating mental health challenges.
             
             """)
    
    st.header(":blue[Problem Statement]")
    st.write("""
             A myriad of individuals globally grapple with mental health challenges, particularly depression, 
             anxiety disorders, and bipolar disorder. However, the widespread prevalence of these issues is compounded 
             by the limited accessibility to affordable mental healthcare. Barriers such as stigma, prohibitive costs, 
             and a shortage of mental health professionals contribute to the existing challenges, highlighting the 
             imperative for comprehensive and accessible mental health services on a global scale.

             """)
    
    st.write(" ")
    
    
    
    
    
elif selected == "Chatbot":
    
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I help you today?"]
        
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])

    if 'buffer_memory' not in st.session_state:
                st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


    system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
    response_container = st.container()
# container for text box
    textcontainer = st.container()


    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                st.code(conversation_string)
                refined_query = query_refiner(conversation_string, query)
                #st.subheader("Refined Query:")
                #st.write(refined_query)
                context = find_match(refined_query)
                #print(context)  
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 
    with response_container:
        if st.session_state['responses']:

            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


st.markdown(
    "`Created by` Team RevolveX | 2024 | \
    `Code:` [Github](https://github.com/Adebesin-Aramide/HarmonyHaven_chatbot)"
)