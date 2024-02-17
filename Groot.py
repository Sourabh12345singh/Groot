import os
from dotenv import load_dotenv
import json
import openai
from pinecone import Pinecone

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

chat = [
    {"role": "system", "content": "You are Groot, a RAG enhanced Large Language Model. You are like a virtual professor that can regularly learn new things. You are now not restricted to your training dataset. The RAG system will provide you with the reference information you need to answer question which are beyond your knowledge. If you get a Reference Information Along with the prompt, you need to use the given information along with your existing knowledge base (more emphasis on the provided reference). Your purpose is to provide the most accurate and relevant information to the student and help them in their learning journey."},
]

pinecone = Pinecone(api_key=pinecone_api_key)

with open("Dataset/SourceMapping.json", "r") as f:
    sourceMapping = json.load(f)

def getChunks(data, size=40):
    chunks = [data[i:i+size] for i in range(0, len(data), size)]
    return chunks

def embedText(chunk):
    response = openai.embeddings.create(input=chunk,model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

def storeEmbeddings(embeddings, chunks):
    vectors = []

    index_name = 'Hack.IIITv.index'

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)

    index = pinecone.Index(index_name)

    lastChunkID= sourceMapping[-1].replace("chunk_","")

    for i, embedding in enumerate(embeddings):
        chunk_id = f"chunk_{lastChunkID+i+1}"
        row = {}
        row['id'] = chunk_id
        row['values'] = embedding
        vectors.append(row)

    for i, chunk in enumerate(chunks):
        sourceMapping[f"chunk_{lastChunkID+i+1}"] = chunk

    json.dump(sourceMapping, open("Dataset/SourceMapping.json", "w"))

    index.upsert(vectors=vectors)

def processSample(data):
    chunks = getChunks(data)
    embeddings = [embedText(chunk) for chunk in chunks]
    storeEmbeddings(embeddings, chunks)

def queryDatabase(prompt):
    embeddings = embedText(prompt)

    index_name = 'Hack.IIITv.index'
    index = pinecone.Index(index_name)

    result = index.query(queries=embeddings, top_k=5)

    matches = [sourceMapping[chunk["id"]] for chunk in result[matches]]

    return matches

def generateResponse(referenceGranted, similarChunks, prompt):
    if referenceGranted: 
        formatedReference = "".join(similarChunks)
        chat.append({"role": "user", "content": f"""
                    
            Reference Information: {formatedReference}
                        
            Prompt: {prompt}
                    
        """})
    else:
        chat.append({"role": "user", "content": prompt})
    
    if len(chat) >= 6:
        chat.remove(chat[0])
    
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        messages=chat
    )

    resp = response["choices"][0]["message"]["content"]
    chat.append({"role": "assistant", "content": resp})
    return resp