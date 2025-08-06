from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_completion_tokens=100)


st.title("LangChain HuggingFace Chatbot")

# took the data from users
cricketer_input = st.selectbox("Select a cricketer:", ["Sachin Tendulkar", "Virat Kohli", "MS Dhoni", "Rohit Sharma"])
style_input = st.selectbox("Select a style:", ["Casual", "Formal", "Humorous"])
length_input = st.selectbox("Select response length:", ["Short", "Medium", "Long"])

# Create a prompt template
template = PromptTemplate(
    input_variables=["cricketer", "style", "length"],
    template="You are a cricket expert. Provide a {style} response about {cricketer} in {length} length."
)

# Generate the prompt using the template and user inputs
prompt = template.invoke({
    "cricketer": cricketer_input,
    "style": style_input,
    "length": length_input
})

if st.button("Submit"):
    result = model.invoke(prompt)
    st.write("Response:")
    st.write(result.content)






