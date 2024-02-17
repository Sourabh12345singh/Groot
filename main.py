import os
import shutil
import Groot

if __name__ == "__main__":
    files = [f for f in os.listdir(os.path.join("Dataset")) if (os.path.isfile(os.path.join("Dataset",f)) and not f.endswith(".json"))]
    for file in files:
        with open(os.path.join("Dataset",file), "r") as f:
            Data = f.read()
            Groot.processSample(Data)
        shutil.move(os.path.join("Dataset",file), os.path.join("Dataset/Embedded",file))
    
    while True:
        prompt = input("User: ")
        if prompt == "exit":
            break
        if "<//>" not in prompt:
            similarChunks = Groot.queryDatabase(prompt)
            resp = Groot.generateResponse(True, similarChunks, prompt)
        else:
            resp = Groot.generateResponse(False, [], prompt)

        print("Groot: ", resp)