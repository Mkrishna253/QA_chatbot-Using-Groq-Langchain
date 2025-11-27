import streamlit as st
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import os

# page configuration

st.set_page_config(
    page_title="ChatGroq LLM Chatbot",page_icon=":robot_face:"
)

# title
st.title("simple langchain chat with Groq")
st.markdown("This is a simple chatbot application using Groq LLM integrated with Langchain and Streamlit")


with st.sidebar:
    st.header("Settings")
    st.markdown(
        """

        **Developed by:** Krishnanjali M

    """)
    
    # API KEY
    api_key = st.text_input("Groq API Key", type="password", help="GET your Groq API key from https://groq.com")

    # model selection
    model_name = st.selectbox(
        "Select Groq Model",
        options=[
            "llama-3.1-8b-instant",
            "gemma2-9b-it"
        ], index=0
    )

    # clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()



# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# initialize Groq LLM
@st.cache_resource
def get_groq_llm(api_key, model_name):
    if not api_key:
       return None

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        streaming=True
        )
            
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the user queries accurately and clearly."),
        ("user", "Question: {question}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain

# get the Groq LLM chain
chain = get_groq_llm(api_key, model_name)

if not chain:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.markdown("Get your Groq API key from [Groq](https://groq.com)")

else:
    # display chat messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # chat input
    if question := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:

                # stream response from Groq LLM
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + " ")

                message_placeholder.markdown(full_response)

                # append assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

### Examples:

st.markdown("-----")
st.markdown("### Example Questions:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("what is langchain?")
    st.markdown("what is the technology behind groq?")
with col2:
    st.markdown("how to integrate groq with langchain?")
    st.markdown("explain groq architecture?")

# footer
st.markdown("-----")
st.markdown("Developed with ❤️ using [Langchain](https://langchain.com/) and [Groq](https://groq.com/)")