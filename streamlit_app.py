import streamlit as st
from openai import OpenAI
import pandas as pd
import os
import utils.prompt_utils as prompt_utils
import openpyxl

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import LLMChain, ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from models import MODEL_CONFIGS
from utils.prompt_utils import target_styles, definitions, survey_items
from utils.utils import response_generator

# Show title and description.
st.title(" Therapist Chatbot Evaluation 👋")

# Get participant ID 
user_PID = st.text_input("What is your participant ID?")

style_id = 0


# Create a dropdown selection box
# target_style = st.selectbox('Choose a communication st:', styles)

# Display the selected option
# st.write(f'You selected: {target_style}')

# Retrieve api key from secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

    # therapist agent
    therapist_model_config = MODEL_CONFIGS['Therapist']
    therapyagent_prompt_template = ChatPromptTemplate.from_messages([
        ("system", therapist_model_config['prompt']),
        MessagesPlaceholder(variable_name="history"), # dynamic insertion of past conversation history
        ("human", "{input}"),
    ])
    # Communication style modifier prompt
    modifier_model_config = MODEL_CONFIGS['Modifier']
    csm_prompt_template = PromptTemplate(
        variables=["communication_style", "chat_history", "unadapted_response"], template=modifier_model_config['prompt']
    )

    # set up streamlit history memory
    msgs = StreamlitChatMessageHistory(key="chat_history")

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            # Prewritten first turn
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hello there! How are you feeling today?"},
        ]

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if user_input := st.chat_input("Enter your input here."):
        
        # create a therapy chatbot llm chain
        therapyagent_chain = therapyagent_prompt_template | llm
        therapy_chain_with_history = RunnableWithMessageHistory(
            therapyagent_chain,
            lambda session_id: msgs,  # Always return the instance created earlier
            input_messages_key="input",
            # output_messages_key="content",
            history_messages_key="history",
        )

        # create a csm chain
        csmagent_chain = LLMChain(
            llm=llm,
            prompt=csm_prompt_template,
            verbose=False,
            output_parser=StrOutputParser()
        )


        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)


        config = {"configurable": {"session_id": "any"}}
        unada_response = therapy_chain_with_history.invoke({"input": user_input}, config)
        unada_bot_response = unada_response.content

        target_style = target_styles[style_id]
        definition = definitions[style_id]
        survey_item = survey_items[style_id]
        ada_response = csmagent_chain.predict(communication_style=target_style,
                                            definition=definition,
                                            survey_item=survey_item,
                                            unadapted_chat_history= st.session_state.messages,
                                            unadapted_response=unada_bot_response)

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(response = unada_bot_response))
            # response = st.write_stream(response_generator(response = ada_response))
        st.session_state.messages.append({"role": "assistant", "content": unada_bot_response})
