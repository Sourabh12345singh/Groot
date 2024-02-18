
# Groot

An enhanced AI marvel by Team Vision. Expanding beyond GPT-3's limits, Groot leverages RAG to dynamically learn from external files, revolutionizing personalized learning experiences.

## In Simple Words

The answering capability of every LLM is constrained by its training dataset. For example, it's common to hear ChatGPT say, "I don't know this because my data is limited to January 2022." Retraining such a massive model is both time-consuming and economically unfeasible. Therefore, we employ a technique known as Retrieval-Augmented Generation (RAG). RAG integrates retrieval (finding relevant information) with generation (creating new content) to enhance comprehension and produce more precise and varied responses by retrieving information from an extensive database and leveraging it to improve the generation process.

In essence, Groot can provide information that ChatGPT cannot. 😜

## Groot in Action

<div align="center">
  <img src="https://i.imgur.com/w292p2d.png" alt="screenshot" height="500px"/>
</div>

The loooong word shown in this screenshot is merely a hypothetical theory formulated by our team of esteemed scientists. It has no bearing on the real world, but Groot had to mention it because we fed him the entire 4,000 character theory. 😜
## Installation
Clone the Repository

```bash
$ git clone https://github.com/thedevyashsaini/Groot
$ cd Groot
```

Install venv

```bash
$ pip install virtualenv
```

Setup python virtual environment


```bash
$ python -m venv Vision
$ Vision\Scripts\activate
```

Install required libraries
```bash
(Vision) $ pip install -r requirements.txt
```

### Environment Variables
These environment variables are necessary for the script to run:

```bash
 OPENAI_API_KEY=your-openai-api-key
 PINECONE_API_KEY=your-pinecone-api-key
```

Store these variables in a .env file.




    
## Usage
Run the main file inside the virtual environment

```bash
(Vision) $ python main.py
```

Then open the following address in your browser to access the GUI:
```bash
http://127.0.0.1:7860
```

On this page you can chat with Groot. The files you upload as Dataset will be used by the model to generate responses.

## Authors
##### Team Vision 
  
- [@TheDevyashSaini](https://www.github.com/thedevyashsaini)
- [@DeepRao03](https://www.github.com/deeprao03)
- [@Sourabh12345Singh](https://www.github.com/Sourabh12345singh)
- [@WhoIsAtul](https://www.github.com/whoisatul)
