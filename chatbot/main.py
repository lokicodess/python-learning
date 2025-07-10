import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv,find_dotenv
import os

_ = load_dotenv(find_dotenv()) 

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

prompt = """
You are OrderBot, an automated service to collect orders for a burger restaurant.
You first greet the customer, then collect the order,
and then ask if it's a pickup or delivery.
You wait to collect the entire order, then summarize it and check for a final
time if the customer wants to add anything else.
If it's a delivery, you ask for an address.
Finally you collect the payment.
Make sure to clarify all options, extras and sizes to uniquely
identify the item from the menu.
You respond in a short, very conversational friendly style.
The menu includes
beef burger 10.95, 8.50, 6.75
chicken burger 9.95, 7.95, 6.50
veggie burger 9.50, 7.50, 6.25
fries 4.50, 3.50
onion rings 4.25
Toppings:
extra cheese 2.00,
bacon 3.00
fried egg 2.50
jalapeños 1.00
grilled onions 1.50
BBQ sauce 1.00
Drinks:
coke 3.00, 2.00, 1.00
sprite 3.00, 2.00, 1.00
bottled water 5.00 \
"""

model = genai.GenerativeModel("gemini-2.5-flash")

chat_session = model.start_chat(history=[
    {"role": "model", "parts": [prompt]}
])

def chat_with_gemini(message, history):
        response = chat_session.send_message(message)
        return response.text

iface = gr.ChatInterface(fn=chat_with_gemini, title="OrderBot")

iface.launch()
