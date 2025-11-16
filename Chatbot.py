from openai import OpenAI
from tkinter import *
import json
import subprocess
client = OpenAI(base_url="http://localhost:1234/v1",
                api_key="lm-studio")


from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
def open_zalo():
    subprocess.run(["powershell", "start zalo:"], shell=True)

def open_browser():
    subprocess.run(["powershell", "start microsoft-edge:"], shell=True)

def chatbot():
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that controls applications.\n"
                    "If the user says anything about opening Zalo, respond ONLY with: open_zalo\n"
                    "If the user says anything about opening a browser, respond ONLY with: open_browser\n"
                    "Otherwise, answer normally.\n"
                    "NEVER output anything except exactly the command when a command is required."
                )
            },
            {"role": "user", "content": message + attachread}
        ],
        temperature=0.0,
        max_tokens=50
    )

    # Print the model's response
    global chatbotmessage
    chatbotmessage = completion.choices[0].message.content
    if chatbotmessage == "open_zalo":
        open_zalo()
    elif chatbotmessage == "open_browser":
        open_browser()


def clicked():
    global message
    message = chat1.get()
    global attachread
    attachread = ""
    try:
        attach = open(chat2.get(), "r")
        attachread = attach.read()
    except:


        chat0.config(state = 'normal')
        chat0.insert(END, "You: "+ message + "\n")
        chatbot()
        chat0.insert(END, "Reponse: " + chatbotmessage + "\n")
        chat0.config(state='disabled')
        chat1.delete(0, END)
        chat2.delete(0, END)
    else:
        chat0.config(state = 'normal')
        chat0.insert(END, "You: "+ message + "\n")
        chatbot()
        chat0.insert(END, "Reponse: " + chatbotmessage + "\n")
        chat0.config(state='disabled')
        chat1.delete(0, END)
        chat2.delete(0, END)
root = Tk()
stringvar = StringVar()
lbl0 = Label(root, text = "Chat Room", font =("Arial Bold", 20))
lbl0.grid(column = 0, row = 0, sticky = "")
lbl1 = Label(root, text= "Message")
lbl1.grid(row = 1, column = 0, sticky = 'w')
lbl2 = Label(root, text= "Attachment")
lbl2.grid(row = 1, column = 1, sticky = 'w')
chat0 = Text(root, state = 'disabled')
chat0.grid(row = 7, column = 0, sticky = "w")
chat1 = Entry(root)
chat1.grid(row = 2, column = 0, sticky = "w",)
chat2 = Entry(root)
chat2.grid(row = 2, column = 1, sticky = "w",)
rad1 = Button(root, text = "Send", command = clicked)
rad1.grid(row = 7, column = 1, sticky = "w")

root.mainloop()



