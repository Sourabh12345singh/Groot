import os
import shutil
import Groot
import logging
from colorama import Fore, Back, Style

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    print(Fore.YELLOW, "\n  * Initializing Groot\n", Style.RESET_ALL)
    files = [f for f in os.listdir(os.path.join("Dataset")) if (os.path.isfile(os.path.join("Dataset",f)) and not f.endswith(".json"))]
    if len(files) > 0:
        logging.info(f"Files to process: {len(files)}")
    for file in files:
        with open(os.path.join("Dataset",file), "r") as f:
            Data = f.read()
            Groot.processSample(Data,file)
        shutil.move(os.path.join("Dataset",file), os.path.join("Dataset/Embedded",file))
    
    while True:
        print(Fore.YELLOW, "\n  * Groot is ready to chat. \n  * Type 'exit' to end the conversation.\n", Style.RESET_ALL)
        prompt = input("User: ")
        if prompt == "exit":
            break
        if "<//>" not in prompt:
            similarChunks = Groot.queryDatabase(prompt)
            print("Similar Chunks: ", similarChunks)
            resp = Groot.generateResponse(True, similarChunks, prompt)
        else:
            prompt.replace("<//>","")
            resp = Groot.generateResponse(False, [], prompt)

        print("Groot: ", resp)