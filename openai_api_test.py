import os
from openai import OpenAI
import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
base64_image = encode_image("image_path")
client = OpenAI(
    api_key="", # your api key
    base_url="https://api.openai.com/v1/chat/completions",
    # # https://docs.openkey.cloud/api/model-chat
    # api_key="sk-N7hXtAFOVWiHROgy7b5810F25b6b4998998fBfAbE46533F5",
    # base_url="https://openkey.cloud/v1"
)
chat_completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ],
    temperature=0.7,
    max_tokens=256,
)
print (chat_completion)
response = chat_completion.choices[0].message.content
response = response.strip()
print(response)