from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os 

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model1 = ChatHuggingFace(llm=llm1, temperature=.7)
model2 = ChatHuggingFace(llm=llm2, temperature=.7)

prompt1 = PromptTemplate(
    template="generate short and the simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="generate the 5 small quiz questions from follwing text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='merge the provided notes and quiz into a single document \n notes -> {notes} \n quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
}) 

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """
 Photosynthesis – A Natural Process of Energy Conversion
Photosynthesis is a biochemical process by which green plants, algae, and certain bacteria convert light energy—usually from the sun—into chemical energy stored in glucose or other organic compounds. This process is crucial for life on Earth as it forms the foundation of the food chain and is responsible for the oxygen content of the atmosphere.

At its core, photosynthesis takes place primarily in the chloroplasts of plant cells, which contain a pigment called chlorophyll. Chlorophyll absorbs sunlight, initiating the conversion of water (H₂O) from the soil and carbon dioxide (CO₂) from the atmosphere into glucose (C₆H₁₂O₆) and oxygen (O₂). The general equation for photosynthesis is:

mathematica
6 CO₂ + 6 H₂O + light energy → C₆H₁₂O₆ + 6 O₂
Photosynthesis consists of two major phases: the light-dependent reactions and the light-independent reactions (also known as the Calvin Cycle). In the light-dependent phase, sunlight splits water molecules to release oxygen and generate energy-rich compounds (ATP and NADPH). In the Calvin Cycle, this energy is used to convert carbon dioxide into glucose.

Several factors affect the rate of photosynthesis, including light intensity, carbon dioxide concentration, temperature, and water availability. Plants have evolved various adaptations to maximize photosynthetic efficiency depending on their habitat—such as C3, C4, and CAM photosynthetic pathways.

Photosynthesis not only sustains plant growth but also supports all life by producing oxygen and acting as a carbon sink. Understanding this process is fundamental to fields such as agriculture, climate science, and renewable energy.
"""

result = chain.invoke({
    'text': text
})

print(result)

chain.get_graph().print_ascii()



