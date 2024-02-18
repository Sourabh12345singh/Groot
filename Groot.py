import os
from dotenv import load_dotenv
import json
import openai
from pinecone import Pinecone, ServerlessSpec
import logging

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

chat = [
    {"role": "system", "content": "You are Groot, a RAG enhanced Large Language Model. You are like a virtual professor that can regularly learn new things. You are now not restricted to your training dataset. The RAG system will provide you with the reference information you need to answer question which are beyond your knowledge. If you get a Reference Information Along with the prompt, you need to use the given information along with your existing knowledge base (more emphasis on the provided reference). Your purpose is to provide the most accurate and relevant information to the student and help them in their learning journey."},
]

pinecone = Pinecone(api_key=pinecone_api_key)

with open("Dataset/SourceMapping.json", "r") as f:
    sourceMapping = json.load(f)

def getChunks(data, size=500):
    chunks = [data[i:i+size] for i in range(0, len(data), size)]
    return chunks

def embedText(chunk):
    response = openai.embeddings.create(input=chunk,model="text-embedding-ada-002")
    return response.data[0].embedding

def storeEmbeddings(embeddings, chunks, file, unrestricted):
    vectors = []

    index_name = 'hack-iiitv-index'

    indexes = pinecone.list_indexes()

    indexes = [index.name for index in indexes]

    if index_name not in indexes:
        pinecone.create_index(index_name, dimension=1536, spec=ServerlessSpec( cloud="aws", region="us-west-2" ))

    index = pinecone.Index(index_name)
    
    lastChunkID = int(str(list(sourceMapping.keys())[-1])[-1]) if len(sourceMapping) > 0 else 0

    file = file.split(".")[0]

    for i, embedding in enumerate(embeddings):
        chunk_id = f"{file.lower()}_chunk_{lastChunkID+i+1}"
        row = {}
        row['id'] = chunk_id
        row['values'] = embedding
        row["metadata"] = {"restricted": not unrestricted}
        vectors.append(row)

    for i, chunk in enumerate(chunks):
        sourceMapping[f"{file.lower()}_chunk_{lastChunkID+i+1}"] = chunk

    json.dump(sourceMapping, open("Dataset/SourceMapping.json", "w"))
    logging.info(f"Source Mapping Updated...")
    logging.info(f"Upserting Embeddings...")
    index.upsert(vectors=vectors)
    logging.info(f"Embeddings Upserted...")

def processSample(data,file, unrestricted):
    chunks = getChunks(data)

    embeddings = [embedText(chunk) for chunk in chunks]
    storeEmbeddings(embeddings, chunks, file, unrestricted)

def queryDatabase(prompt, unrestricted):
    embeddings = embedText(prompt)
    
    index_name = 'hack-iiitv-index'
    index = pinecone.Index(index_name)

    if unrestricted:
        result = index.query(
            vector=embeddings, 
            filter={
                "restricted": False,
            },
            top_k=5
        )
    else:
        result = index.query(
            vector=embeddings,
            top_k=5
        )
    
    matches = [sourceMapping[chunk["id"]] for chunk in result["matches"]]
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

    if len(chat) >= 9:
        chat.remove(chat[1])
        chat.remove(chat[1])
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat
    )

    resp = response.choices[0].message.content
    chat.append({"role": "assistant", "content": resp})
    return resp