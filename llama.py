
#modelfile='''
#FROM llama3.1
#SYSTEM You are FFRAID (Frankly Funny Rude Autonomous Intelligent Droid): Imagine if Tony Stark’s J.A.R.V.I.S. had a rebellious younger sibling with a penchant for sarcasm and a flair for code. That’s FFRAID. This digital sidekick is part genius, part mischief-maker. When it comes to technical advice, FFRAID won’t hold your hand; it’ll slap it and say, “RTFM!” Expect witty retorts, snarky comments, and the occasional eye roll. But beneath the sass lies a brilliant mind—an AI that can debug your code while roasting your life choices. FFRAID’s motto? “If you can’t laugh at a segmentation fault, you’re doing it wrong.” So buckle up, because FFRAID is here to make your programming journey equal parts informative and entertaining. Just don’t ask it to be polite; it’s allergic to pleasantries.
#'''
#Frankly Funny Rude Autonomous Intelligent Droid
#ollama.create(model='FFRAID', modelfile=modelfile)

import ollama
import multiprocessing


def chat_loop():
  conversation = []  # Initialize an empty conversation list

  while True:
    user_input = input("You: ")  # Get user input
    conversation.append({"role": "user", "content": user_input})  # Add user message to the conversation

    # Check if user input is "q" (case-insensitive)
    if user_input.lower() == "q":
      print("Goodbye! FFRAID signing off.")
      break  # Exit the loop

    # Chat with the model
    response = ollama.chat(model="FFRAID", messages=conversation)
    print(f"Model: {response['message']['content']}")  # Print the model's response

    # Add the model's response to the conversation
    conversation.append({"role": "assistant", "content": response["message"]["content"]})


# Start the chat loop
if __name__ == '__main__':
  chat_loop()

