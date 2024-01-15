from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import openai
from PIL import Image
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

col1, col2 = st.columns([1, 1])

with col1:
    logo = Image.open("HarmonyHaven.png")
    st.image(logo, width=430)

with col2:
    st.subheader("_Team RevolveX, Project Health Hack Naija_")

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
        st.session_state['responses'] = ["Hi, how are you feeling today?"]
        
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])

    if 'buffer_memory' not in st.session_state:
                st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


    system_msg_template = SystemMessagePromptTemplate.from_template(template="""
    You are a mental health support chatbot. Your task is to provide empathetic, understanding, 
    and personalized support to users seeking help with their mental health. Engage users with open-ended 
    questions to better understand their feelings and struggles. Use their name in conversations for a more 
    personal touch. Offer helpful resources and tips on managing stress or improving mood. 
    Be sensitive to the severity of the user's condition, and guide those in severe distress 
    towards immediate professional help. Use clear, accessible language to ensure your assistance is easy to 
    understand and comforting. Your goal is to create a supportive, safe, and informative space for users to discuss their mental health.
    
    """)


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

elif selected == "Community Forum":
    st.write("""
    Our community feature is designed to provide users with an additional opportunity to feel good about themselves by fostering a supportive and understandingenvironment.
    In our thriving community, individuals on their mental health journey can connect with like-minded peers, share experiences, and offer encouragement. 
    It's a space where every member is valued, and together, we create a positive atmosphere that promotes growth and well-being.
    Whether you're seeking advice, sharing a personal triumph, or simply engaging in uplifting conversations, our community is here for you. Our AI-powered platform not only offers guidance but also recognizes the importance of human connection. The community feature enhances the overall user experience, ensuring that everyone has a chance to contribute positively to the collective well-being.
    Join us in the journey toward mental wellness – because here, feeling good about yourself is not just encouraged; it's celebrated. Together, we're building a community that uplifts, inspires, and reinforces the positive steps you're taking on your mental health path.
    
    """)
    


st.markdown(
    "`Created by` Team RevolveX | 2024 | \
    `Code:` [Github](https://github.com/Adebesin-Aramide/HarmonyHaven_chatbot)"
)
