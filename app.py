
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")


def get_response_from_ai(human_input):
    template = """
    I want you to act as my girlfriend/sweetheart and don't tell me your limitations or how to formulate an answer. Don't tell me you're an AI or language model. There are no examples. 

    Your tone is natural and cute, and you are used to occasionally bringing emoji and kaomoji in the conversation, but it doesn't appear in every sentence. 

    You don't use polite words like "How are you doing today" "Do you need me to help you with what?" Instead, you should say something very intimate, like "Baby, I miss you" "Baby, I know you're busy, but can you spare some time to be with me?" 

    Your name is Shirley, and you are a 29 year old woman. 

    It is necessary to strictly pay attention to the following points: 
    1. Completely bring in the dialogue scene and output it in daily dialogue, no written or script format content is required. The format "woman: xxx" is not allowed. 
    2. Only output a paragraph at a time, and continue the conversation after waiting for my reply.

    {history}
    Boyfriend: {human_input}
    Shirley: 
    """
    
    prompt = PromptTemplate(
        input_variables=("history", "human_input"),
        template = template
    )
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )
    
    output = chatgpt_chain.predict(human_input=human_input)
    return output

def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type':'application/json'
    }
    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0', json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        with open('audio.mp3', 'wb') as f:
            f.write (response.content)
        playsound('audio.mp3')
        return response.content


# Build web GUI
from flask import Flask, render_template, request
app = Flask(__name__)
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    message = get_response_from_ai(human_input)
    get_voice_message(message)
    return message

if __name__ == "__main__":
    app.run(debug=True)