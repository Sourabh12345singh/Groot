import os
import gradio as gr
import time
import Groot
import sys
import webview
import shutil
import logging
from colorama import Fore, Back, Style
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
unrestricted = False

def trash(k):
    yield k

def handleCheck(checked):
    global unrestricted
    unrestricted = checked

def fileUpload(history, file, request: gr.Request):
    with open(os.path.join(os.path.dirname(file),os.path.basename(file)),"r") as f:
        Data = f.read()
    Groot.processSample(Data, os.path.basename(file), unrestricted)
    history = history + [(None, "File added to vector database: " + os.path.basename(file) + " as " + ("unrestricted" if unrestricted else "restricted") + ".")]
    return history

with gr.Blocks(css=".gradio-container {max-width: 800px !important}") as demo:
    gr.Markdown(
    """
    # Groot!
    RAG enhanced GPT-3 model.
    """)
    chatbot = gr.Chatbot(avatar_images=[None,"https://i.imgur.com/ixEn0m3.png"])
    
    with gr.Column():
        msg = gr.Textbox(
            show_label=False,
            placeholder="Talk to groot...",
            container=False,
        )
        with gr.Row():
            gr.Interface(handleCheck, gr.Checkbox(label="Unrestricted"),outputs=None, live=True, clear_btn="")
            btn = gr.UploadButton("Dataset File", scale=3, file_types=["Text"])
    clear = gr.ClearButton([msg, chatbot],value="Clear Chat")

    def respond(prompt, chat_history):
        if prompt == "exit":
            sys.exit(0)
        similarChunks = Groot.queryDatabase(prompt, unrestricted)
        print("Similar Chunks: ", similarChunks)
        resp = Groot.generateResponse(True, similarChunks, prompt)
        chat_history.append((prompt, resp))
        time.sleep(2)
        return "", chat_history
    file_msg = btn.upload(fileUpload, [chatbot, btn], [chatbot], queue=False).then(
        trash, chatbot, chatbot
    )

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    print(Fore.YELLOW, "\n  * Initializing Groot\n", Style.RESET_ALL)
    files = [f for f in os.listdir(os.path.join("Dataset")) if (os.path.isfile(os.path.join("Dataset",f)) and not f.endswith(".json"))]
    if len(files) > 0:
        logging.info(f"Files to process: {len(files)}")
    for file in files:
        with open(os.path.join("Dataset",file), "r") as f:
            Data = f.read()
            Groot.processSample(Data,file,True)
        shutil.move(os.path.join("Dataset",file), os.path.join("Dataset/Embedded",file))

    
    demo.launch()