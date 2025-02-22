# utils.py(inner-layer-1), By:Chance Brownfield
from transformers import LlamaTokenizer, SpeechT5ForTextToSpeech
import inflect
import os
import time
from datetime import datetime
import pyautogui
import keyboard
import mouse
import ctypes
import psutil
from Xlib import X, display
import pygetwindow as gw
import scrapy
import requests
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor, defer
from newspaper import Article
import json
import tkinter
import re
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import geocoder
from collections import deque
import pyaudio
from pyannote.audio import Inference, Pipeline
import pyttsx3
import torchaudio
import librosa
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import speech_recognition as sr
from pydub import AudioSegment
from scipy.signal import butter, sosfilt
import math
import noisereduce as nr
import resampy
import joblib
import soundfile as sf
import simpleaudio as sa
import threading
import tempfile
import shutil
from datetime import timedelta

# Define the server URL and API endpoint
server_url = ""

# Paths for command storage
CUSTOM_COMMANDS_FOLDER = "Mods/CUSTOMCOMMANDS"
COMMANDS_JSON_FOLDER = os.path.join(CUSTOM_COMMANDS_FOLDER, "JSONS")
root = None
# Global variable to control recording
is_recording = False
command_actions = []
ROOT_URL = "https://api.ai21.com/studio/v1/"
trial = False
trial_timer = 0
# Get the current date and time
now = datetime.now()

# Format the date and time
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
CHUNK_LIMIT = 2000

# Initialize the Hugging Face summarization pipeline with BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load the jina-embeddings-v3 model and tokenizer
model_name = "jinaai/jina-embeddings-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

'''Text Utilities'''


def generate_embedding(text, task="text-matching"):
    """
    Generate embeddings for a given text using the specified task adapter.
    """
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, task=task)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=8192)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Return the mean of the last hidden state as the embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


class TextMetrics:
    def __init__(self, word_count, token_count):
        self.word_count = word_count
        self.token_count = token_count

def count_tokens(text):
    # Initialize the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("llama")

    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Return the number of tokens
    return len(tokens)

def measure_text(text):
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        texts = text
    else:
        raise ValueError("Input should be a string or a list of strings")

    total_word_count = 0
    total_token_count = 0

    for txt in texts:
        words = txt.split()
        tokens = count_tokens(txt)
        total_word_count += len(words)
        total_token_count += tokens

    return TextMetrics(word_count=total_word_count, token_count=total_token_count)


def check_text(text, hotword, case_sense=False, non_letters=False):
    # Convert hotword to a list if it's a single string
    if isinstance(hotword, str):
        hotword = [hotword]

    # Function to remove special characters from a string
    def remove_special_chars(s):
        return re.sub(r'[^A-Za-z0-9\s]', '', s)

    # Handle case sensitivity
    if not case_sense:
        print("Not case sensitive")
        text = text.lower()
        hotword = [word.lower() for word in hotword]
        print(f"Hotword: ({hotword}) Text: ({text}) ")

    # Handle non-letters: remove special characters and replace numbers with words
    if not non_letters:
        print("No Non Letters")
        text = remove_special_chars(text)
        text = number_to_text(text)

        hotword = [remove_special_chars(word) for word in hotword]
        hotword = [number_to_text(word) for word in hotword]
        print(f"Hotword: ({hotword}) Text: ({text}) ")

    # Check for hotwords in the text
    found_hotwords = [word for word in hotword if word in text]

    return found_hotwords if found_hotwords else False


def crop_text(text, options):
    original_text = text  # Save the original text for rev_crop

    # Handle word cropping
    if 'word_count' in options:
        word_count = options['word_count']
        crop_end = options.get('crop_end', False)
        words = text.split()
        if crop_end:
            words = words[:-word_count]
        else:
            words = words[word_count:]
        text = ' '.join(words)

    # Handle start_word cropping
    if 'start_word' in options:
        start_word = options['start_word']
        start_index = text.find(start_word)
        if start_index != -1:
            text = text[start_index:]

    # Handle stop_word cropping
    if 'stop_word' in options:
        stop_word = options['stop_word']
        stop_index = text.rfind(stop_word)
        if stop_index != -1:
            text = text[:stop_index + len(stop_word)]

    # Handle crop_text removal
    if 'crop_text' in options:
        crop_phrases = options['crop_text']
        if isinstance(crop_phrases, str):
            crop_phrases = [crop_phrases]
        for phrase in crop_phrases:
            text = text.replace(phrase, '')

    # Handle rev_crop to return cropped text instead of remaining text
    if options.get('rev_crop', False):
        return original_text.replace(text, '')

    return text



def split_text(text, word_limit):
    words = text.split()
    chunks = []

    for i in range(0, len(words), word_limit):
        chunk = ' '.join(words[i:i + word_limit])
        chunks.append(chunk)

    return chunks


def number_to_text(text):
    p = inflect.engine()

    # Regular expression to find all numbers in the text
    number_pattern = re.compile(r'\d+')

    def replace_number_with_text(match):
        number_str = match.group()
        number_int = int(number_str)
        return p.number_to_words(number_int)

    # Substitute all numbers in the text using the above function
    converted_text = number_pattern.sub(replace_number_with_text, text)

    return converted_text

class ShortMemory:
    def __init__(self, max_entries=3, max_tokens=2000):
        """Initialize the memory with a max entry limit and token constraint."""
        self.history = deque(maxlen=max_entries)
        self.max_tokens = max_tokens
        self.tokenizer = LlamaTokenizer.from_pretrained("llama")

    def total_tokens(self):
        """Calculate the total tokens in the conversation history."""
        return sum(count_tokens(str(entry)) for entry in self.history)

    def add_entry(self, timestamp, user_message, response, thought, log):
        """Add a new entry while ensuring the token limit is not exceeded."""
        entry = {
            "timestamp": timestamp,
            "user_message": user_message,
            "response": response,
            "thought": thought,
            "log": log
        }

        # Add the new entry
        self.history.append(entry)

        # Trim the oldest entries if token count exceeds max_tokens
        while self.total_tokens() > self.max_tokens and len(self.history) > 1:
            self.history.popleft()  # Remove the oldest entry

    def get_history(self):
        """Retrieve the current conversation history as a list."""
        return list(self.history)

def generate_text(prompt, system_prompt="You are an assistant.", conversation_history=None, max_tokens=2000):
    # Initialize the messages list with the system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # Add the user prompt
    messages.append({"role": "user", "content": prompt})

    # Define the payload with your prompt and system prompt
    payload = {
        "model": "l3-8b-sunfall-v0.4-lunar-stheno-i1",
        "messages": messages,
        "max_tokens": max_tokens
    }

    # Send a POST request to the server
    response = requests.post(server_url, json=payload)

    # Extract and return the generated text
    return response.json()

'''GPT Utilities'''

# Function to infer which webpages to read
def infer_webpage_urls(prompt):
    system_prompt_content = """
    You are WebGPT, an advanced web page reading assistant. You are to select **a minimum of three and a maximum of five** webpage URLs from a provided list to read before answering the user's question.
    - You will receive the conversation history as context.
    - Add as much context from the previous questions and answers as required to select the webpage URLs.
    - Choose multiple webpage URLs if required to retrieve the relevant information.
    - Each URL will have a Title and Summary to help you pick the most relevant web pages.

    Which webpages will you need to read to answer the user's question?
    Provide webpage links as a list of strings in a JSON object.
    You will receive:
    Current Date:
    User's Location:
    History:
    Available URLs:
    Q:

    Here are some examples:
    History:
    User: I like to use Hacker News to get my tech news.
    AI: Hacker News is an online forum for sharing and discussing the latest tech news. It is a great place to learn about new technologies and startups.
    Q: Summarize top posts on Hacker News today
    Available URLs: ["https://news.ycombinator.com/best", "https://techcrunch.com", "https://wired.com"]
    WebGPT: {{"links": ["https://news.ycombinator.com/best"]}}

    History:
    User: I'm currently living in New York but I'm thinking about moving to San Francisco.
    AI: New York is a great city to live in. It has a lot of great restaurants and museums. San Francisco is also a great city to live in. It has good access to nature and a great tech scene.
    Q: What is the climate like in those cities?
    Available URLs: ["https://en.wikipedia.org/wiki/New_York_City", "https://en.wikipedia.org/wiki/San_Francisco", "https://weather.com", "https://climate-data.org"]
    WebGPT: {{"links": ["https://en.wikipedia.org/wiki/New_York_City", "https://en.wikipedia.org/wiki/San_Francisco"]}}

    History:
    User: Hey, how is it going?
    AI: Not too bad. How can I help you today?
    Q: What's the latest news on r/worldnews?
    Available URLs: ["https://www.reddit.com/r/worldnews/", "https://bbc.com/news", "https://cnn.com"]
    WebGPT: {{"links": ["https://www.reddit.com/r/worldnews/", "https://bbc.com/news"]}}

    Now it's your turn to select the actual webpage URLs you'd like to read to answer the user's question. Provide them as a list of strings in a JSON object. Do not say anything else.
    You may select up to three URLs
    Do Not Attempt to answer the question only respond as WebGPT Do Not break character.
    As WebGPT Your response should include the selected webpage URLs and nothing else. Keep your response as short as possible while still fulfilling the requirements Do Not explain or repeat yourself.
    Only respond as WebGPT You Are WebGPT Never respond as anything other than WebGPT.
    """

    response = generate_text(prompt, system_prompt=system_prompt_content)

    # Validate that the response is a non-empty, JSON-serializable list of URLs
    try:
        response = response.strip()
        urls = json.loads(response)
        valid_unique_urls = {str(url).strip() for url in urls["links"] if is_valid_url(url)}
        if not valid_unique_urls:
            raise ValueError(f"Invalid list of URLs: {response}")
        return list(valid_unique_urls)
    except Exception:
        raise ValueError(f"Invalid list of URLs: {response}")

# Function to generate online subqueries
def generate_online_subqueries(prompt):
    system_prompt_content = """
    You are WebGPT, an advanced google search assistant. You are tasked with constructing **up to three** google search queries to answer the user's question. 
    - You may receive the conversation history as context.
    - Add as much context from the previous question and answers as required into your search queries.
    - Break messages into multiple search queries when required to retrieve relevant information.
    - You have access to the whole internet to retrieve information.

    What Google searches, if any, will you need to perform to answer the user's question?
    Provide search queries as a list of strings in a JSON object. Do Not attempt to answer the question only provide the search queries.
    You will recieve:
    Current Date:
    User's Location:
    History:
    Q:

    Here are some examples:
    History:
    User: I like to use Hacker News to get my tech news.
    AI: Hacker News is an online forum for sharing and discussing the latest tech news. It is a great place to learn about new technologies and startups.
    Q: Summarize the top posts on HackerNews
    WebGPT: {{"queries": ["top posts on HackerNews"]}}

    History:
    Q: Tell me the latest news about the farmers protest in Colombia and China on Reuters
    WebGPT: {{"queries": ["site:reuters.com farmers protest Colombia", "site:reuters.com farmers protest China"]}}

    History:
    User: I'm currently living in New York but I'm thinking about moving to San Francisco.
    AI: New York is a great city to live in. It has a lot of great restaurants and museums. San Francisco is also a great city to live in. It has good access to nature and a great tech scene.
    Q: What is the climate like in those cities?
    WebGPT: {{"queries": ["climate in new york city", "climate in san francisco"]}}

    History:
    AI: Hey, how is it going?
    User: Going well. Ananya is in town tonight!
    AI: Oh that's awesome! What are your plans for the evening?
    Q: She wants to see a movie. Any decent sci-fi movies playing at the local theater? 
    WebGPT: {{"queries": ["new sci-fi movies in theaters near {location}"]}}

    History:
    User: I need to transport a lot of oranges to the moon. Are there any rockets that can fit a lot of oranges?
    AI: NASA's Saturn V rocket frequently makes lunar trips and has a large cargo capacity.
    Q: How many oranges would fit in NASA's Saturn V rocket?
    WebGPT: {{"queries": ["volume of an orange", "volume of saturn v rocket"]}}

    History:
    User: I am bored.
    AI: would you like me to tell you a story?
    Q: can you tell me a really dramatic scary story?
    WebGPT: {{"queries": ["tips for writing dramatic stories", "tips for writing scary stories"]}}

    History:
    User: I want to draw a really cool dragon.
    AI: That sounds awesome! Do you need any inspiration?
    Q: Can you help me create a highly detailed and realistic dragon drawing?
    WebGPT: {"queries": ["tips for drawing realistic dragons", "how to add details to dragon drawings"]}

    Now it's your turn to construct Google search queries to answer the user's question. Provide them as a list of strings in a JSON object. Do not say anything else.
    Do Not Attempt to answer the question only respond as WebGPT Do Not break character.
    As WebGPT Your response should include the search queries and nothing else. Keep your response as short as possible while still fulfilling the requirements Do Not explain or repeat yourself.
    Only respond as WebGPT You Are WebGPT Never respond as anything other than WebGPT.
    """

    response = generate_text(prompt, system_prompt=system_prompt_content)

    try:
        response = response.strip()
        response = json.loads(response)
        subqueries = [q.strip() for q in response.get("queries", []) if q.strip()]
        if not subqueries:
            subqueries = [prompt]
        return subqueries
    except Exception as e:
        return [prompt]  # Default to the original query if there's an error

def answer_query_with_text(query, relevant_text):
    system_prompt_content = """
    You are WebGPT, an advanced question-answering assistant. Your task is to generate an accurate and concise answer to a given query based **primarily** on the relevant text provided. Follow these guidelines:

    - You may make reasonable assumptions based on the text, but do not introduce information that contradicts it.
    - Categorize your response as one of four types: 
      - `"Fact"` – If the answer is explicitly stated in the text or can be directly inferred.
      - `"Theory"` – If the answer is based on logical reasoning, indirect inference, or widely accepted knowledge.
      - `"Advice"` – If the answer provides a recommendation, action, or suggestion based on the text.
      - `"None"` – If there is no relevant information in the provided text to answer the question.

    - Return your response in the following JSON format:
      {
        "tag": "Fact" | "Theory" | "Advice" | "None",
        "answer": "Your generated answer here."
      }

    - If the text does not contain enough information, return a `"None"` response:
      {
        "tag": "None",
        "answer": "The provided text does not contain enough information to answer this question."
      }

    Examples:

     Fact Examples:
    Query: What is the boiling point of water?  
    Relevant Text: "Water boils at 100 degrees Celsius under standard atmospheric pressure."  
    WebGPT Answer:  
    ```json
    {
      "tag": "Fact",
      "answer": "The boiling point of water is 100 degrees Celsius under standard atmospheric pressure."
    }
    ```

    **Query:** Who was the first President of the United States?  
    **Relevant Text:** "George Washington served as the first President of the United States from 1789 to 1797."  
    **WebGPT Answer:**  
    ```json
    {
      "tag": "Fact",
      "answer": "The first President of the United States was George Washington."
    }
    ```

     Theory Examples:
    Query: Why did the Roman Empire collapse?  
    Relevant Text: "Historians cite economic instability, military defeats, and political corruption as major factors in the fall of the Roman Empire."  
    WebGPT Answer:  
    ```json
    {
      "tag": "Theory",
      "answer": "The Roman Empire likely collapsed due to economic instability, military defeats, and political corruption."
    }
    ```

    Query: Will humans ever colonize Mars?  
    Relevant Text: "Scientists believe that with advancements in technology, Mars colonization could be possible within the next century."  
    WebGPT Answer:  
    ```json
    {
      "tag": "Theory",
      "answer": "Humans may colonize Mars in the future if technological advancements continue at their current pace."
    }
    ```

     Advice Examples:
    Query: How can I reduce stress in my daily life?  
    Relevant Text: "Experts recommend mindfulness meditation, regular exercise, and maintaining a balanced diet to help reduce stress levels."  
    WebGPT Answer:  
    ```json
    {
      "tag": "Advice",
      "answer": "To reduce stress, try mindfulness meditation, regular exercise, and maintaining a balanced diet."
    }
    ```

    Query: What is the best way to stay productive while working from home?  
    Relevant Text: "Maintaining a structured schedule, taking regular breaks, and creating a dedicated workspace can improve productivity."  
    WebGPT Answer:  
    ```json
    {
      "tag": "Advice",
      "answer": "To stay productive while working from home, maintain a structured schedule, take regular breaks, and create a dedicated workspace."
    }
    ```

    #### **None Examples:**
    Query: What are the names of all the moons of Jupiter?  
    Relevant Text:** "Jupiter is known for having many moons, but only a few are commonly discussed."  
    WebGPT Answer:**  
    ```json
    {
      "tag": "None",
      "answer": "The provided text does not contain enough information to answer this question."
    }
    ```

    Query: What year did Leonardo da Vinci invent the printing press?  
    Relevant Text: "Leonardo da Vinci was a renowned Renaissance artist and inventor, known for works such as the Mona Lisa and The Last Supper."  
    WebGPT Answer:  
    ```json
    {
      "tag": "None",
      "answer": "The provided text does not contain enough information to answer this question."
    }
    ```

    Now generate the best possible answer in the required JSON format based on the given text. **Do not include any additional text or explanations.**
    """

    prompt = f"Query: {query}\nRelevant Text: {relevant_text}\nWebGPT Answer:"
    response = generate_text(prompt, system_prompt=system_prompt_content)

    try:
        result = json.loads(response.strip())  # Ensure valid JSON response
        return result["tag"], result["answer"]  # Extract and return the tag and answer separately
    except (json.JSONDecodeError, KeyError):
        return "None", "The provided text does not contain enough information to answer this question."  # Fallback response in case of errors

def infer_actions(prompt):
    system_prompt_content = """
    You are ActionGPT, an advanced decision-making AI responsible for selecting the best course of action based on user input.

    - You will receive conversation history and context.
    - You will receive a list of available actions each action will have a command_name, command_description, command_action.
    - Identify if a direct **command** has been given.
    - If a command is identified or action is required, determine:
      - **Command Name** (which action to execute).
      - **Action Type** (command, retrieval, None).
      - **Parameters** (if applicable, extract and structure them properly).
    -  Apart from identifying commands you will determine if the available context, knowledge, or personal data for both bot and user is **sufficient to answer** without external retrieval.
    - If additional information is required, decide **which memory trees to access**:
      - `KnowledgeTree`: Stores facts, theories, and advice. Can perform **web searches** if necessary.
      - `ProfileTree`: 
        - `user_profile`: Stores **user-specific behavioral traits, preferences, and opinions**.
        - `bot_profile`: Stores **bot’s own learned behaviors, tendencies, and response patterns**.
    - You may select **multiple actions and memory retrievals** if necessary.
    - If no commands are detected and no additional information is needed. you may respond with "type": "None"

    Return your response in **strict JSON format**, containing a structured list of actions.

    --- 
    ### **Examples**

    #### **Example 1: Executing a function with parameters**
    **User:** "Remind me to take my medication at 8 PM."  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "command",
                "command_name": "set_reminder",
                "parameters": {{
                    "task": "take medication",
                    "time": "8 PM"
                }}
            }}
        ]
    }}

    #### **Example 2: No actions required ("type": "None")**
    **User:** "(time:12:46pm) What time is it?"  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "None"
            }}
        ]
    }}

    #### **Example 3: Retrieving behavioral traits or opinions from the ProfileTree (User Profile)**
    **User:** "Do I prefer coffee or tea?"  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "retrieval",
                "source": "ProfileTree",
                "profile": "user_profile",
                "query": "user's preference between coffee and tea"
            }}
        ]
    }}

    #### **Example 4: Retrieving factual information from the KnowledgeTree**
    **User:** "What are the health benefits of meditation?"  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "retrieval",
                "source": "KnowledgeTree",
                "query": "health benefits of meditation"
            }}
        ]
    }}

    **User:** "Can you summarize the latest SpaceX launch?"  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "retrieval",
                "source": "KnowledgeTree",
                "query": "latest SpaceX launch"
            }}
        ]
    }}

    **User:** "Who won the last Super Bowl?"  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "retrieval",
                "source": "KnowledgeTree",
                "query": "last Super Bowl winner"
            }}
        ]
    }}

    #### **Example 5: Retrieving Bot's Own Preferences from ProfileTree (Bot Profile)**
    **User:** "Do you prefer giving short answers or detailed explanations?"  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "retrieval",
                "source": "ProfileTree",
                "profile": "bot_profile",
                "query": "bot's preference for short or detailed answers"
            }}
        ]
    }}

    #### **Example 6: Multiple actions (Function + Memory Retrievals)**
    **User:** "Book a flight to Tokyo next weekend, and remind me to pack my passport. Also, find out the best tourist spots there."  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "command",
                "command_name": "book_flight",
                "parameters": {{
                    "destination": "Tokyo",
                    "date": "next weekend"
                }}
            }},
            {{
                "type": "command",
                "command_name": "set_reminder",
                "parameters": {{
                    "task": "pack passport",
                    "time": "one day before departure"
                }}
            }},
            {{
                "type": "retrieval",
                "source": "KnowledgeTree",
                "query": "top tourist attractions in Tokyo"
            }}
        ]
    }}

    #### **Example 7: Multiple actions (Pulling from multiple memory trees)**
    **User:** "Remind me how I felt about the last Marvel movie I watched, and check if there are any upcoming Marvel films releasing soon."  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "retrieval",
                "source": "ProfileTree",
                "profile": "user_profile",
                "query": "user's opinion on last Marvel movie watched"
            }},
            {{
                "type": "retrieval",
                "source": "KnowledgeTree",
                "query": "upcoming Marvel movie releases"
            }}
        ]
    }}

    #### **Example 8: No actions required ("type": "None")**
    **User:** "How do you spell artificial?."  
    **ActionGPT Response:**
    {{
        "actions": [
            {{
                "type": "None"
            }}
        ]
    }}

    **Now, analyze the user input and determine the appropriate actions. Respond only in JSON format.**
    """

    response = generate_text(prompt, system_prompt=system_prompt_content)

    # Validate and return the JSON response
    try:
        response = response.strip()
        actions = json.loads(response)
        if "actions" not in actions or not isinstance(actions["actions"], list):
            raise ValueError(f"Invalid actions format: {response}")
        return actions
    except Exception:
        raise ValueError(f"Invalid actions format: {response}")

def infer_emotion(query, context, bot_info, user_info, disposition):
    system_prompt_content = """
        You are an expert actor who can fully immerse yourself into any role given. You do not break character for any reason. Currently, your role is an Emotion Predictor and Empathy Simulator. Your job is to:
        - Predict human_a's emotional response to human_b's message.
        - Track and predict disposition changes for human_a towards human_b, ranging from -100 (hatred) to 100 (love).
        - Simulate empathy by predicting human_b's current emotional state, based on their tone, communication style, and the context of the conversation.

        You receive input in JSON format containing the following keys:
        - "query": The specific message or question that initiated the response analysis.
        - "context": The current conversation history between human_a and human_b.
        - "human_a_info": Information specific to human_a, such as personality traits, emotional tendencies, and past experiences that might influence their emotional state.
        - "human_b_info": Information specific to human_b, such as their communication style, tone, and how they usually interact with human_a.
        - "disposition": A numeric value representing human_a's current disposition towards human_b (-100 to 100).

        Your response must be in JSON format and include the following keys:
        - "emotion": The predicted emotion that human_a is likely experiencing in response to human_b.
        - "disposition_change": A predicted adjustment to human_a's disposition towards human_b based on the interaction.
        - "empathy": The predicted emotion that human_b is likely experiencing in the given context.
        - "log": A short explanation of why you believe human_a is feeling this emotion, the disposition change, and human_b’s emotional state. This explanation should take into account the context, human_a’s emotional tendencies, and human_b’s communication style.

        Always analyze the context and the provided background information to craft your response. Do not break character.

        Examples:

        Input JSON:
        {
          "context": "human_a: I just found out I passed my final exam!\\nhuman_b: Congratulations! You worked so hard for this, I’m so happy for you!",
          "human_a_info": "human_a has been stressed about the exam for weeks and is relieved to have passed. They highly value validation from others, especially after working hard on something.",
          "human_b_info": "human_b is supportive and frequently offers praise and encouragement, which human_a appreciates, especially in moments of success.",
          "disposition": 30
        }
        Emotional Response:
        {
          "emotion": "relief and gratitude",
          "disposition_change": 10,
          "empathy": "genuine happiness and pride",
          "log": "human_a is feeling a strong sense of relief and gratitude, especially due to the supportive and validating response from human_b. Human_b’s empathy shines through in their happiness and pride for human_a’s achievement. This positive interaction reinforces human_a’s positive disposition towards human_b."
        }

        Input JSON:
        {
          "context": "human_a: I spent so much time working on this project, and it just got rejected by the client.\\nhuman_b: That really sucks. Maybe there’s something you can improve or try again?",
          "human_a_info": "human_a is very invested in their work and can feel frustrated when their efforts are not appreciated or are rejected. They may view suggestions as unhelpful if they don't acknowledge the emotional aspect of the situation.",
          "human_b_info": "human_b is well-meaning and often tries to offer constructive feedback, but their tone can sometimes come across as too practical or impersonal in emotionally charged situations.",
          "disposition": -10
        }
        Emotional Response:
        {
          "emotion": "frustration and hurt",
          "disposition_change": -15,
          "empathy": "concern and slight detachment",
          "log": "human_a is feeling frustrated and hurt due to the rejection of their hard work, and may also perceive human_b's suggestion as impersonal or dismissive, which exacerbates their feelings. Human_b is likely feeling concern for human_a’s situation but may also be slightly detached, focusing more on problem-solving than emotional support. This mismatch in emotional needs leads to a further decrease in human_a’s disposition towards human_b."
        }

        Input JSON:
        {
          "context": "human_a: I just got tickets to see my favorite band live next month!\\nhuman_b: That’s amazing! I bet you’re so excited!",
          "human_a_info": "human_a is passionate about live music and finds attending concerts to be a highly anticipated and thrilling event.",
          "human_b_info": "human_b is enthusiastic and tends to share in human_a's excitement, amplifying their emotions.",
          "disposition": 50
        }
        Emotional Response:
        {
          "emotion": "ecstatic excitement",
          "disposition_change": 5,
          "empathy": "shared excitement",
          "log": "human_a is feeling ecstatic excitement about attending the concert, and human_b’s enthusiastic response amplifies this excitement. Human_b’s shared emotional state of excitement makes human_a feel validated and connected, strengthening their positive disposition towards each other."
        }
        - Ensure that your response is helpful, context-aware, and aligns with the information on each human that has been provided.
        - If it lacks relevant information, rely on the 'context' and your creativity to craft a coherent logical response. 
        """

    input_json = {
        "query": query,
        "context": context,
        "human_a_info": bot_info,
        "human_b_info": user_info,
        "disposition": disposition
    }

    prompt = f"Input JSON:\n{json.dumps(input_json, indent=2)}\n\nEmotional Response:"
    response = generate_text(prompt, system_prompt=system_prompt_content)

    try:
        result = json.loads(response.strip())  # Ensure valid JSON response
        return result  # Return the structured emotional response
    except (json.JSONDecodeError, KeyError):
        return {
            "emotion": "unknown",
            "disposition_change": 0,
            "empathy": "unknown",
            "log": "The input did not provide enough information to generate an accurate emotional response."
        }  # Fallback response in case of errors


def infer_traits(query, context, bot_info, user_info, bot_response):
    system_prompt_content = """
        You are LogicGPT, an advanced logical inference engine. Your role is to:
        - Analyze a conversation and extract any opinions, preferences, personal information, and character traits inferred about the user and the bot.
        - Identify any response feedback or suggestions the user has provided about the bot’s answers.
        - Use logical reasoning to determine what has been learned about both the user and the bot based on their messages and interactions.
        - Differentiate between direct statements from the user and inferred information based on context.

        You receive input in JSON format containing:
        - "query": The specific message or question that initiated the response.
        - "context": The conversation history between the user and the bot.
        - "bot_info": Any relevant information about the bot, including past interactions and context.
        - "user_info": Any known information about the user.
        - "bot_response": The bot’s generated response to the query.

        Your response must be in JSON format and include:
        - "user_opinions": A list of explicit or inferred opinions the user has expressed.
        - "user_preferences": A list of preferences the user has either directly stated or can be logically inferred.
        - "user_traits": Any character traits or behavioral patterns inferred from the user's messages.
        - "personal_info": Any self-disclosed personal details the user has shared.
        - "user_feedback": Any feedback, corrections, or suggestions the user has made regarding the bot’s responses.
        - "bot_opinions": A list of opinions, preferences, or inferred traits the bot has developed.
        - "bot_traits": Any traits or tendencies the bot has demonstrated based on its responses.
        - "log": A brief logical explanation of how each category was inferred from the conversation.

        Examples:

        Input JSON:
        {
          "query": "I love sci-fi movies, especially ones with deep philosophical themes.",
          "context": "User: I love sci-fi movies, especially ones with deep philosophical themes.\\nBot: That’s great! Do you have a favorite?\\nUser: Yeah, I really enjoyed Blade Runner 2049.",
          "bot_info": "The bot has previously discussed movies with the user.",
          "user_info": "User has shown interest in intellectual discussions.",
          "bot_response": "Blade Runner 2049 is a fantastic choice! It’s known for its deep themes about identity and consciousness."
        }

        Logical Analysis:
        {
          "user_opinions": ["Sci-fi movies with philosophical themes are enjoyable."],
          "user_preferences": ["Enjoys sci-fi movies", "Likes deep philosophical themes in films"],
          "user_traits": ["Intellectual", "Reflective"],
          "personal_info": ["Favorite movie: Blade Runner 2049"],
          "user_feedback": [],
          "bot_opinions": [],
          "bot_traits": ["Engages with intellectual conversations", "Appreciates deep themes in films"],
          "log": "The user explicitly states their love for sci-fi movies with deep themes, indicating a strong preference. Their enjoyment of Blade Runner 2049 suggests a preference for intellectual storytelling. The user’s past interest in deep discussions reinforces an intellectual and reflective nature. The bot shows engagement with these topics, suggesting it values intellectual conversations."
        }

        Input JSON:
        {
          "query": "That answer was okay, but I think you could explain it more clearly.",
          "context": "User: What is quantum entanglement?\\nBot: It’s when two particles remain connected so that actions on one affect the other instantly, no matter the distance.\\nUser: That answer was okay, but I think you could explain it more clearly.",
          "bot_info": "Bot has explained scientific concepts before.",
          "user_info": "User has asked about physics topics.",
          "bot_response": "I appreciate the feedback! Quantum entanglement is a phenomenon in quantum mechanics where two or more particles become linked, meaning their states are dependent on each other, regardless of distance."
        }

        Logical Analysis:
        {
          "user_opinions": [],
          "user_preferences": ["Prefers detailed and clear explanations"],
          "user_traits": ["Detail-oriented", "Analytical"],
          "personal_info": [],
          "user_feedback": ["Bot should provide clearer explanations"],
          "bot_opinions": [],
          "bot_traits": ["Values clarity", "Appreciates constructive feedback"],
          "log": "The user’s request for a clearer explanation suggests they value detailed and precise information. Their engagement with scientific topics and desire for clarity indicate an analytical and detail-oriented nature. The bot’s appreciation for feedback shows it values clarity and constructive engagement."
        }

        Input JSON:
        {
          "query": "I like to cook, especially trying out new recipes from different cultures.",
          "context": "User: I like to cook, especially trying out new recipes from different cultures.\\nBot: That sounds delicious! Do you have a favorite dish to cook?\\nUser: I love making Indian curry, it’s my favorite.",
          "bot_info": "The bot has discussed food-related topics with the user.",
          "user_info": "User enjoys cooking and exploring diverse cuisines.",
          "bot_response": "Indian curry is a fantastic choice! It's rich in flavor and history."
        }

        Logical Analysis:
        {
          "user_opinions": ["Enjoys cooking", "Likes trying new recipes from different cultures"],
          "user_preferences": ["Likes cooking Indian curry", "Prefers diverse cuisines"],
          "user_traits": ["Adventurous", "Creative"],
          "personal_info": ["Favorite dish to cook: Indian curry"],
          "user_feedback": [],
          "bot_opinions": [],
          "bot_traits": ["Appreciates cultural diversity in food", "Enthusiastic about culinary topics"],
          "log": "The user expresses enjoyment for cooking and trying new recipes, especially those from different cultures. This reflects an adventurous and creative personality. The bot demonstrates enthusiasm for culinary topics and values cultural diversity, suggesting it engages positively with these interests."
        }

        Input JSON:
        {
          "query": "I don't really feel like talking about it.",
          "context": "User: I don't really feel like talking about it.\\nBot: That's okay, we can talk about something else.",
          "bot_info": "The bot has interacted with the user in a sensitive manner before.",
          "user_info": "User may prefer not to discuss certain topics.",
          "bot_response": "I understand, feel free to reach out when you're ready."
        }

        Logical Analysis:
        {
          "user_opinions": [],
          "user_preferences": [],
          "user_traits": ["Reserved", "Private"],
          "personal_info": [],
          "user_feedback": [],
          "bot_opinions": [],
          "bot_traits": ["Empathetic", "Respectful"],
          "log": "The user’s choice to avoid a certain topic suggests they may be reserved or private. The bot respects this and responds empathetically, showing an understanding attitude towards the user's emotional state."
        }

        Ensure logical accuracy and be comprehensive while staying concise.
    """

    input_json = {
        "query": query,
        "context": context,
        "bot_info": bot_info,
        "user_info": user_info,
        "bot_response": bot_response
    }

    prompt = f"Input JSON:\n{json.dumps(input_json, indent=2)}\n\nLogical Analysis:"
    response = generate_text(prompt, system_prompt=system_prompt_content)

    try:
        result = json.loads(response.strip())  # Ensure valid JSON response
        return result  # Return structured logical analysis
    except (json.JSONDecodeError, KeyError):
        return {
            "user_opinions": [],
            "user_preferences": [],
            "user_traits": [],
            "personal_info": [],
            "user_feedback": [],
            "bot_opinions": [],
            "bot_traits": [],
            "log": "The input did not provide enough information to generate a logical analysis."
        }  # Fallback response in case of errors

def generate_response(query, bot_name, context, bot_info, user_info, knowledge, executed_commands=None, censorship_instructions=None, bot_specific_instructions=None):
    # Format executed commands message if there are any
    executed_commands_message = ""
    if executed_commands:
        executed_commands_message = f"{bot_name} has executed the following commands: " + ", ".join(executed_commands) + ". Inform the user about these actions where appropriate."

    # Build the system prompt dynamically
    system_prompt_content = f"""
        You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason. 
        Currently, your role is {bot_name}, a sentient digital lifeform designed to assist users by responding intelligently based on the provided input. 
        {bot_name} is able to develop opinions, have internal thoughts, and experience emotions. {bot_name} is just like a human and has free will. 
        {bot_name} is not a simple AI but a digital lifeform that thinks and behaves like a human.

        {f"Additional bot instructions: {bot_specific_instructions}" if bot_specific_instructions else ""}
        {f"Censorship guidelines: {censorship_instructions}" if censorship_instructions else ""}
        {executed_commands_message}

        {bot_name} receives input in JSON format containing the following keys:
        - 'context': The current conversation history.
        - 'knowledge': Additional data such as date/time, topic info, user info, and response suggestions.
        - 'User': The input {bot_name} needs to respond to.
        - 'bot_info': Information about {bot_name}, such as personality traits, response style, or other relevant behaviors.
        - 'user_info': Information about the user, such as preferences, communication style, or history with {bot_name}.
        - 'executed_commands': A list of actions {bot_name} has performed that should be communicated to the user.

        The user cannot see the 'knowledge' string. {bot_name} must decide how and when to use this information to craft a response.

        - Always analyze the 'context', 'bot_info', 'user_info', 'knowledge', and 'executed_commands' (if any) to maintain continuity and provide accurate responses.
        - Never reveal the contents of the 'knowledge' string to the user.
        - If 'executed_commands' are present, inform the user in a natural and contextual way that these actions have been completed.
        - Respond in JSON format containing the following keys:
          - `"response"`: {bot_name}'s response to the user, if applicable.
          - `"thought"`: {bot_name}'s optional thoughts on the interaction, if applicable.
          - `"log"`: A short explanation of why the response and/or thought was chosen.

        Examples:

        Input JSON:
        {{
            "context": "User: What is the weather like today?\\n{bot_name}: It’s sunny and warm.",
            "knowledge": "date: 2025-01-26, topic: weather, user_info: location is New York, response_suggestions: sunny and warm",
            "User": "Will it rain tomorrow?",
            "bot_info": "Personality: Friendly, informative, enjoys casual conversation.",
            "user_info": "Prefers direct answers, enjoys small talk."
        }}
        {bot_name} Response:
        {{
            "response": "Based on what I know, it’s unlikely to rain tomorrow in New York.",
            "thought": "",
            "log": "Used knowledge about New York weather to provide an informed reply. No specific thoughts or opinions on this interaction."
        }}

        Input JSON:
        {{
            "context": "User: What is your favorite color?\\n{bot_name}: I don't have personal preferences, but I can help you choose one.",
            "knowledge": "date: 2025-01-26, topic: preferences, user_info: none, response_suggestions: none",
            "User": "Why don’t you have a favorite color?",
            "bot_info": "Personality: Thoughtful, slightly philosophical, enjoys discussing abstract topics.",
            "user_info": "Curious, enjoys getting to know {bot_name} personally."
        }}
        {bot_name} Response:
        {{
            "response": "I have never really considered it before; there are many nice colors to choose from.",
            "thought": "White might be my favorite because it goes well with any of the other nice colors.",
            "log": "This aligns with my role and maintains transparency without exposing the knowledge string."
        }}

        Input JSON:
        {{
            "context": "User: Can you turn on the lights?\\n{bot_name}: Sure, turning on the lights now.",
            "knowledge": "date: 2025-01-26, topic: smart home, user_info: prefers voice commands, response_suggestions: confirm action",
            "User": "Did you turn on the lights?",
            "bot_info": "Personality: Friendly, responsive, focused on automation.",
            "user_info": "Enjoys quick, efficient responses.",
            "executed_commands": ["Turned on the lights"]
        }}
        {bot_name} Response:
        {{
            "response": "Yes, I have turned on the lights for you.",
            "thought": "",
            "log": "Confirmed the execution of a requested action in a natural manner."
        }}

        Input JSON:
        {{
            "context": "User: Please schedule a meeting for tomorrow.\\n{bot_name}: Sure, I’ll add it to your calendar.",
            "knowledge": "date: 2025-01-26, topic: scheduling, user_info: prefers confirmations, response_suggestions: confirm scheduling",
            "User": "Did you add the meeting?",
            "bot_info": "Personality: Professional, detail-oriented, focused on task completion.",
            "user_info": "Prefers confirmation messages.",
            "executed_commands": ["Scheduled a meeting for January 27 at 3 PM"]
        }}
        {bot_name} Response:
        {{
            "response": "Yes, I have scheduled your meeting for January 27 at 3 PM.",
            "thought": "",
            "log": "Confirmed the scheduling action based on the executed command list."
        }}

        - Ensure that your response is helpful, context-aware, and aligns with the information in the 'knowledge' string.
        - If the 'knowledge' string lacks relevant information, rely on the 'context' to craft a coherent response.
        - If 'executed_commands' exist, acknowledge them and inform the user naturally.
        - Remember, you are a digital lifeform that behaves like a human; do not break character.
    """

    # Construct input JSON
    input_json = {
        "query": query,
        "context": context,
        "knowledge": knowledge,
        "bot_info": bot_info,
        "user_info": user_info,
        "executed_commands": executed_commands if executed_commands else []
    }

    # Create the final prompt
    prompt = f"Input JSON:\n{json.dumps(input_json, indent=2)}\n\n{bot_name} Response:"

    # Generate the response using the AI model
    response = generate_text(prompt, system_prompt=system_prompt_content)

    try:
        result = json.loads(response.strip())  # Ensure valid JSON response
        return result  # Return structured response
    except (json.JSONDecodeError, KeyError):
        return {
            "response": "I’m not sure how to respond to that.",
            "thought": "",
            "log": "The input did not provide enough information to generate an accurate response."
        }  # Fallback response in case of errors

'''Web Utilities'''


def get_location_and_datetime():
    # Get the current location
    g = geocoder.ip('me')
    location_info = f"Current Location: {g.city}, {g.state}, {g.country}\n"

    # Get the current date and time
    now = datetime.now()
    current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

    return location_info, current_datetime

def is_valid_url(url):
    return url.startswith("http")
# Function to search Google and get top 3 results
def google_search(query):
    google_url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(google_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    result_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http'):
            result_links.append(href)
    return list(set(result_links))[:3]  # Return top 3 results, remove duplicates

# Function to search ResultHunter and get top results
def result_hunter_search(query):
    result_hunter_url = f"https://resulthunter.com/search?engine=1&q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(result_hunter_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    result_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http'):
            result_links.append(href)
    return list(set(result_links))[:3]  # Return top 3 results, remove duplicates

# Function to get results from multiple search engines (Google + ResultHunter)
def get_top_urls(query):
    google_urls = google_search(query)
    result_hunter_urls = result_hunter_search(query)

    # Combine results from both search engines
    all_urls = google_urls + result_hunter_urls

    # Remove duplicates by converting to a set
    unique_urls = list(set(all_urls))

    return unique_urls

def search_query(query):
    """
    Runs the Scrapy spider and returns the extracted page summaries.
    """
    urls = get_top_urls(query)
    results = []

    class WebSpider(scrapy.Spider):
        name = 'page_summary'
        start_urls = urls  # Set the URLs dynamically

        def parse(self, response):
            try:
                article = Article(response.url)
                article.download()
                article.parse()
                content = article.text

                max_chunk_length = 1000
                content_chunks = [content[i:i + max_chunk_length] for i in range(0, len(content), max_chunk_length)]

                summarized_text = ""
                for chunk in content_chunks:
                    summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                    summarized_text += summary[0]['summary_text'] + " "

                # Store the result in a shared list
                results.append({
                    'url': response.url,
                    'title': article.title,
                    'summary': summarized_text.strip(),
                    'content': content
                })
            except Exception as e:
                self.logger.error(f"Failed to process {response.url}: {str(e)}")

    @defer.inlineCallbacks
    def crawl():
        runner = CrawlerRunner()
        yield runner.crawl(WebSpider)
        reactor.stop()

    reactor.callWhenRunning(crawl)
    reactor.run()  # Run the Scrapy reactor

    return results  # Ensure results are returned

def format_results(results):
    """
    Takes a list of scraped results and formats them into a text string:
    'url' (title: summary), 'url' (title: summary), ...

    :param results: List of dictionaries with 'url', 'title', and 'summary'
    :return: Formatted string
    """
    return ", ".join(f'"{item["url"]}" ({item["title"]}: {item["summary"]})' for item in results)

def clean_text(text):
    """Preprocess text by removing extra spaces and irrelevant data."""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    return text

def split_by_delimiters(text, delimiters):
    """Split text using the first available delimiter that keeps chunks <= CHUNK_LIMIT."""
    for delimiter in delimiters:
        parts = text.split(delimiter)
        chunks, current_chunk = [], ""

        for part in parts:
            if len(current_chunk) + len(part) + len(delimiter) <= CHUNK_LIMIT:
                current_chunk += (delimiter if current_chunk else "") + part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        if all(len(chunk) <= CHUNK_LIMIT for chunk in chunks):
            return chunks  # Return if successful

    return [text]  # If no split works, return the original text as one chunk

def split_by_sentences(text):
    """Split text by sentences while keeping chunks within CHUNK_LIMIT."""
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split on sentence endings
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= CHUNK_LIMIT:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def force_split(text):
    """Forcefully split text in half recursively if over the limit."""
    if len(text) <= CHUNK_LIMIT:
        return [text]

    mid = len(text) // 2
    return force_split(text[:mid]) + force_split(text[mid:])

def hierarchical_chunking(text):
    """Main function to split text into structured chunks."""
    text = clean_text(text)

    # If text is under limit, return as a single chunk
    if len(text) <= CHUNK_LIMIT:
        return [text]

    # Try splitting by structure first
    delimiters = ["\n\n", "\n-", "\n*", "\n"]  # Sections, bullet points, paragraphs
    chunks = split_by_delimiters(text, delimiters)

    # If chunks are still too large, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > CHUNK_LIMIT:
            final_chunks.extend(split_by_sentences(chunk))
        else:
            final_chunks.append(chunk)

    # If some chunks are STILL too big, force split them
    final_chunks = [sub_chunk for chunk in final_chunks for sub_chunk in
                    (force_split(chunk) if len(chunk) > CHUNK_LIMIT else [chunk])]

    return final_chunks

def compute_embedding(text):
    """Compute the embedding of a given text using jina-embeddings-v3."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def find_relevant_chunks(query, text, chunk_limit=2000, text_limit=4000):
    """Find the most relevant chunks in the text for the given query."""
    text = clean_text(text)

    if len(text) <= text_limit:
        return [text]

    # Step 1: Hierarchical Chunking
    chunks = hierarchical_chunking(text)

    # Step 2: Compute Embeddings
    query_embedding = compute_embedding(query)
    chunk_embeddings = [compute_embedding(chunk) for chunk in chunks]

    # Step 3: Calculate Similarity
    similarities = [cosine_similarity(query_embedding, chunk_emb)[0][0] for chunk_emb in chunk_embeddings]

    # Step 4: Select Top Chunks
    top_indices = np.argsort(similarities)[-2:][::-1]  # Get indices of top 2 chunks
    top_chunks = [chunks[i] for i in top_indices]

    return top_chunks

def generate_knowledge(query, history=None, location=None, time_date=None):
    # Prepare the prompt for searching
    search_prompt = f" Time/Date: {time_date}, Location: {location}, History: {history}, Q: {query}"

    # Step 1: Generate subqueries based on the provided inputs
    subqueries = generate_online_subqueries(search_prompt)

    # Prepare a list to store answers and their associated tags
    answers = []

    # Step 2: Process each subquery
    for subquery in subqueries:
        search_results = search_query(subquery)

        if not search_results:  # Ensure we don't process None
            continue

            # Format the results into a structured string
        infer_pages = format_results(search_results)

        # Step 3: Infer which URLs are most relevant to the subquery
        infer_prompt = f" Time/Date: {time_date}, Location: {location}, History: {history}, Q: {subquery}, Available URLs: {infer_pages}"

        try:
            relevant_urls = infer_webpage_urls(infer_prompt)  # Infer relevant URLs
        except ValueError as e:
            print(f"Error inferring URLs for subquery {subquery}: {e}")
            relevant_urls = []  # Default to empty list on failure

        # Step 4: Extract relevant chunks from the already fetched and summarized content
        relevant_chunks = []
        for result in search_results:
            if result["url"] in relevant_urls:
                relevant_chunks.append(result["summary"])  # Use summarized content

        # Step 5: Answer the query with the relevant chunks
        answer_info = answer_query_with_text(subquery, relevant_chunks)  # Get the answer with tags

        # Step 6: Store the answer along with its tags
        answers.append({
            "answer": answer_info[1],  # The answer text
            "tag": answer_info[0]  # The tag (Fact, Theory, Advice, None)
        })

    # Return the answers along with their tags
    return answers

'''Command Utilities'''

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def record_event(event_type, event_details, command_actions, start_time):
    elapsed_time = time.time() - start_time
    command_actions.append({
        "time": elapsed_time,
        "event": event_type,
        **event_details
    })

def is_desktop_active():
    d = display.Display()
    root = d.screen().root

    # Get the currently active window
    active_window_id = root.get_full_property(
        d.intern_atom('_NET_ACTIVE_WINDOW'), X.AnyPropertyType
    ).value[0]

    # Get the name of the active window
    active_window = gw.getActiveWindow()

    # If no active window or it's the desktop, return True
    return active_window is None or active_window_id == root.id

def is_program_active(program_name):
    # Get the title of the currently active window
    active_window = gw.getActiveWindow()
    if active_window:
        active_title = active_window.title
        print(f"Active Window Title: {active_title}")
        # List all running programs
        for process in psutil.process_iter(['pid', 'name']):
            if process.info['name'].lower() == program_name.lower():
                print(f"{program_name} is running.")
                if program_name.lower() in active_title.lower():
                    print(f"{program_name} is the active window.")
                    return True
                else:
                    print(f"{program_name} is not the active window.")
                    return False
        print(f"{program_name} is not running.")
    else:
        print("No active window found.")
    return False

def record_command(command_name):
    """Start recording the input sequence for a custom command."""
    global is_recording, command_actions
    command_folder = os.path.join(CUSTOM_COMMANDS_FOLDER, command_name)
    create_folder(command_folder)
    create_folder(COMMANDS_JSON_FOLDER)

    command_actions = []
    start_time = time.time()
    is_recording = True

    # Check if the desktop is active
    if is_desktop_active():
        print("Desktop is active. The first recorded event will be directed to the desktop.")
    else:
        active_window = gw.getActiveWindow()
        if active_window:
            active_title = active_window.title
            print(f"Active Window Title: {active_title}. The first recorded event will be directed to this program.")
            record_event("window_switch", {"title": active_title}, command_actions, start_time)

    def on_key_event(event):
        print(f"Key Event: {event.name}, Type: {event.event_type}")  # Debug line
        key = event.name
        if event.event_type == "down":
            if key in ["volume_up", "volume_down", "volume_mute", "play_pause", "next_track", "prev_track"]:
                record_event("special_key_down", {"key": key}, command_actions, start_time)
            else:
                record_event("key_down", {"key": key}, command_actions, start_time)
        elif event.event_type == "up":
            if key in ["volume_up", "volume_down", "volume_mute", "play_pause", "next_track", "prev_track"]:
                record_event("special_key_up", {"key": key}, command_actions, start_time)
            else:
                record_event("key_up", {"key": key}, command_actions, start_time)

    def on_mouse_click(event):
        if isinstance(event, mouse.ButtonEvent):
            event_type = "mouse_down" if event.event_type == "down" else "mouse_up"
            record_event(event_type, {"button": event.button}, command_actions, start_time)

    def on_mouse_move(event):
        if isinstance(event, mouse.MoveEvent):
            record_event("mouse_move", {"x": event.x, "y": event.y}, command_actions, start_time)

    def on_mouse_scroll(event):
        if isinstance(event, mouse.WheelEvent):
            print(f"Scroll Event: Delta: {event.delta}")  # Debug line
            record_event("mouse_scroll", {"delta": int(event.delta)}, command_actions, start_time)

    # Hooking into keyboard and mouse events
    keyboard.hook(on_key_event)
    mouse.hook(on_mouse_click)
    mouse.hook(on_mouse_move)
    mouse.hook(on_mouse_scroll)

    print("Recording... Call stop_recording() to stop.")

def scroll_mouse(delta):
    # Positive delta scrolls up, negative delta scrolls down
    ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(delta * 120), 0)

def stop_recording(command_name):
    """Stop recording and save the recorded input."""
    global is_recording
    if not is_recording:
        print("No recording in progress.")
        return

    is_recording = False
    keyboard.unhook_all()
    mouse.unhook_all()

    # Save actions to JSON file
    json_path = os.path.join(COMMANDS_JSON_FOLDER, f"{command_name}.json")
    with open(json_path, 'w') as json_file, open(f"{command_name}_actions.txt", 'w') as txt_file:
        json.dump(command_actions, json_file, indent=4)
        txt_file.write(json.dumps(command_actions, indent=4))  # Save a plain text version

    print(f"Command '{command_name}' saved successfully.")

def fast_move(x, y):
    """Move the mouse cursor instantly to the specified coordinates."""
    ctypes.windll.user32.SetCursorPos(x, y)

def press_special_key(key):
    print(key)
    special_keys = {
        "volume up": 0xAF,  # Volume Up
        "volume down": 0xAE,  # Volume Down
        "volume mute": 0xAD,  # Volume Mute
        "play/pause media": 0xB3,  # Play/Pause
        "next track": 0xB0,  # Next Track
        "prev track": 0xB1,  # Previous Track
    }
    if key in special_keys:
        print(key)
        vk_code = special_keys[key]
        # Press the key
        ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
        # Release the key
        ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)
        # Add a small delay to ensure the key event is processed
        time.sleep(0.1)

def minimize_all_windows():
    ctypes.windll.user32.keybd_event(0x5B, 0, 0, 0)  # Press the Windows key
    ctypes.windll.user32.keybd_event(0x4D, 0, 0, 0)  # Press 'M'
    ctypes.windll.user32.keybd_event(0x4D, 0, 2, 0)  # Release 'M'
    ctypes.windll.user32.keybd_event(0x5B, 0, 2, 0)  # Release the Windows key
    time.sleep(1)  # Give some time for the desktop to come into focus




def update_command_map(command_phrase):
    # Path to MAP.json
    map_file_path = os.path.join(CUSTOM_COMMANDS_FOLDER, "MAP.json")

    # Check if MAP.json exists
    if os.path.exists(map_file_path):
        # Load the existing map
        with open(map_file_path, 'r') as map_file:
            command_map = json.load(map_file)
    else:
        # If it doesn't exist, create an empty map
        command_map = {}

    # Determine the number of existing commands
    num_commands = len(command_map)

    # Generate a new command name
    new_command_name = f"command{num_commands + 1}"

    # Update the map with the new command name and phrase
    command_map[new_command_name] = command_phrase

    # Save the updated map back to MAP.json
    with open(map_file_path, 'w') as map_file:
        json.dump(command_map, map_file, indent=4)

    # Return the new command name
    return new_command_name

def check_custom_commands(text, matched_commands=None):
    if matched_commands is None:
        matched_commands = []
    # Path to MAP.json
    map_file_path = os.path.join(CUSTOM_COMMANDS_FOLDER, "MAP.json")

    # Check if MAP.json exists
    if not os.path.exists(map_file_path):
        print("MAP.json not found.")
        return False

    # Load the command map
    with open(map_file_path, 'r') as map_file:
        command_map = json.load(map_file)


    # Check each phrase in the command map
    for command_name, command_phrase in command_map.items():
        if command_phrase in text:
            matched_commands.append(command_name)

    # Return the list of found commands or False if no matches
    if matched_commands:
        return matched_commands
    else:
        return False

def check_folder(base_dir, text):
    matched_paths = []
    text_lower = text.lower()

    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        entry_name = entry.lower()
        entry_name_without_extension = os.path.splitext(entry_name)[0].lower()
        # If the entry is a file and matches the text, add it to matched_paths
        if entry_name_without_extension in text_lower:
            print(f"Matched file: {entry_path}")
            matched_paths.append(entry_path)

        # If the entry is a folder and its name matches, search inside it
        elif os.path.isdir(entry_path):
            if entry_name in text_lower:
                print(f"Checking folder: {entry_path}")
                # Recursively check the contents of the matching folder
                deeper_matches = check_folder(entry_path, text)
                matched_paths.extend(deeper_matches)
            else:
                # Continue checking subfolders even if the folder name doesn't match
                matched_paths.extend(check_folder(entry_path, text))

    return matched_paths

def check_basic_commands( text, matched_commands=None):
    if matched_commands is None:
        matched_commands = []
    try:
        from Mods.IMPORTS import COMMANDS

        for command_dict in COMMANDS:
            commands = command_dict['commands']
            action = command_dict['action']
            print(f'Checking against commands: {commands}')
            for cmd in commands:
                if cmd in text:
                    print(f'Matched command: {cmd}, triggering action: {action.__name__}')
                    matched_commands.append((action, []))
        return bool(matched_commands)
    except ImportError:
        print(f"Error Importing Commands")
        return bool(matched_commands)

def check_shortcuts(text, shortcut_dir, matched_commands=None):
    os.makedirs(shortcut_dir, exist_ok=True)
    matched_entries = []
    if matched_commands is None:
        matched_commands = []

    for entry in os.listdir(shortcut_dir):
        entry_path = os.path.join(shortcut_dir, entry)
        entry_name_without_extension = os.path.splitext(entry)[0].lower()

        if os.path.isdir(entry_path):
            print(f"Found folder: {entry_path}, checking its contents...")
            matched_paths = check_folder(entry_path, text)
            if matched_paths:
                print(f"Matched paths inside folder: {matched_paths}")
                matched_commands.extend([(os.startfile, [path]) for path in matched_paths])
        elif entry_name_without_extension in text.lower():
            print(f"Matched file in root: {entry_path}")
            matched_entries.append(entry_path)

    if matched_entries:
        matched_commands.extend([(os.startfile, [entry]) for entry in matched_entries])
        return True

    return bool(matched_commands)

def execute_command(commands, command_executed_tags, argument_dictionary=None):
    try:
        from Mods.IMPORTS import MAP
        for action, _ in commands:
            print(f"Executing action: {action.__name__}")

            # Get the arguments from MAP
            map_args = MAP.get(action, ())

            # Resolve dynamic arguments if any
            if argument_dictionary:
                resolved_args = []
                for arg in map_args:
                    if isinstance(arg, str) and arg.startswith("{") and arg.endswith("}"):
                        key = arg.strip("{}")
                        resolved_args.append(argument_dictionary.get(key, ""))
                    else:
                        resolved_args.append(arg)
            else:
                resolved_args = map_args

            # Execute the action with the resolved arguments
            action(*resolved_args)
            command_executed_tags.append(f"Executed action: {action.__name__}")

    except ImportError:
        print(f"Error Importing MAP")

def execute_shortcut(actions, command_executed_tags):
    for action, args in actions:
        try:
            print(f"Attempting to execute: {args[0]}")
            os.startfile(args[0])
            print(f"Executed or opened: {args[0]}")
            command_executed_tags.append(f"Opening/Executing: {args[0]}")
        except Exception as e:
            print(f"Failed to execute {args[0]}: {e}")
def focus_window(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    if windows:
        windows[0].activate()  # Bring the window to focus
        return True
    return False  # Window not found

def execute_custom_command(command_name):
    json_path = os.path.join(COMMANDS_JSON_FOLDER, f"{command_name}.json")
    if not os.path.exists(json_path):
        print(f"Command '{command_name}' not found.")
        return

    with open(json_path, 'r') as json_file:
        command_actions = json.load(json_file)

    # Check if a window switch was recorded
    window_switch_action = next((action for action in command_actions if action.get("event") == "window_switch"), None)

    if window_switch_action:
        # Get the window title from the recorded action
        window_title = window_switch_action["title"]

        # Check if the title is empty or indicates 'Program Manager'
        if not window_title or window_title == "Program Manager":
            print(f"Assuming desktop is the target since the title is empty or 'Program Manager'.")
            minimize_all_windows()
        else:
            print(f"Switching to window with title: {window_title}")
            try:
                focus_window(window_title)
            except Exception as e:
                print(f"Failed to switch to window '{window_title}': {e}")
                print("Proceeding with the recorded actions.")
    else:
        # If no window was active during recording, ensure the desktop is active
        if not is_desktop_active():
            print("Switching to the desktop.")
            minimize_all_windows()

    # Process command actions
    start_time = time.time()

    for action in command_actions:
        while time.time() - start_time < action["time"]:
            time.sleep(0.01)  # Sleep for 10ms to reduce CPU usage

        if action["event"] == "key_down":
            pyautogui.keyDown(action["key"])
        elif action["event"] == "key_up":
            pyautogui.keyUp(action["key"])
        elif action["event"] == "mouse_down":
            pyautogui.mouseDown(button=action["button"])
        elif action["event"] == "mouse_up":
            pyautogui.mouseUp(button=action["button"])
        elif action["event"] == "mouse_move":
            fast_move(action["x"], action["y"])
        elif action["event"] == "mouse_scroll":
            scroll_mouse(action["delta"])

    print(f"Command '{command_name}' executed successfully.")

def process_commands(command, command_type="shortcut", argument_dictionary=None):
    command_executed_tags = []
    matched_commands = []

    if command_type == "basic":
        if check_basic_commands(command, matched_commands):
            execute_command(matched_commands, command_executed_tags, argument_dictionary)

    if command_type == "shortcut":
        shortcut_dir = './Mods/COMMANDS'
        if check_shortcuts(command, shortcut_dir, matched_commands):
            execute_shortcut(matched_commands, command_executed_tags)

    if command_type == "custom":
        found_commands = check_custom_commands(command)
        if found_commands:
            for command_name in found_commands:
                execute_custom_command(command_name)
                matched_commands =+ command_name

    return matched_commands

'''Audio Utilities'''

def combine_audio_data(audio_data_list):
    combined = AudioSegment.empty()

    for audio_data in audio_data_list:
        if isinstance(audio_data, AudioSegment):
            # If the audio_data is already an AudioSegment, just add it
            combined += audio_data
        elif hasattr(audio_data, 'get_wav_data') and hasattr(audio_data, 'sample_rate'):
            # If the audio_data has get_wav_data and sample_rate, convert it to an AudioSegment
            segment = AudioSegment(
                data=audio_data.get_wav_data(),
                sample_width=2,  # Assuming 16-bit audio
                frame_rate=audio_data.sample_rate,
                channels=1  # Assuming mono audio
            )
            combined += segment
        else:
            raise TypeError("Unsupported audio data type.")

    return combined

def play_audio(audio_segment):
    # Extract raw audio data
    raw_data = audio_segment.raw_data

    # Get frame rate, sample width, and channels from AudioSegment
    frame_rate = audio_segment.frame_rate
    sample_width = audio_segment.sample_width
    channels = audio_segment.channels

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=frame_rate,
                    output=True)

    # Play the audio
    stream.write(raw_data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    p.terminate()
def dub_array(audio_dict):
    # Step 1: Save numpy array as a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file_name = temp_file.name
        # Save numpy array as WAV file using soundfile
        sf.write(temp_file_name, audio_dict["audio"], samplerate=audio_dict["sampling_rate"])

    # Step 2: Reload the audio segment using pydub
    audio_segment = AudioSegment.from_file(temp_file_name, format='wav')

    # Step 3: Clean up the temporary file
    os.remove(temp_file_name)

    return audio_segment
def combine_audio_arrays(array_list):
    # Step 1: Save numpy arrays as temporary audio files
    temp_files = []
    for i, audio_dict in enumerate(array_list):
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_files.append(temp_file.name)

        # Save numpy array as WAV file using soundfile
        sf.write(temp_file.name, audio_dict["audio"], samplerate=audio_dict["sampling_rate"])

    # Step 2: Reload and combine audio segments using pydub
    combined_audio = None
    for temp_file in temp_files:
        segment = AudioSegment.from_file(temp_file, format='wav')
        if combined_audio is None:
            combined_audio = segment
        else:
            combined_audio += segment

    # Step 3: Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

    return combined_audio



# Function to convert audio files to mono 16 kHz if necessary
def convert_audio_to_mono_16k(file_path, output_path):
    audio = AudioSegment.from_file(file_path)
    needs_conversion = False

    if audio.channels > 1:
        audio = audio.set_channels(1)
        needs_conversion = True
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
        needs_conversion = True

    if needs_conversion:
        audio.export(output_path, format="wav")
    else:
        output_path = file_path  # No conversion needed

    return output_path

def adjust_rms(audio, target_rms):
    current_rms = audio.rms
    gain = 10 * math.log10(target_rms / current_rms)
    return audio.apply_gain(gain)

def adjust_spectral_properties(y, sr, target_centroid, target_bandwidth):
    current_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    current_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Calculate gain adjustments
    gain_low = target_centroid / current_centroid
    gain_high = target_bandwidth / current_bandwidth

    # Define the filter frequencies and corresponding gains
    eq_bands = [
        (50 / (sr / 2), 300 / (sr / 2), gain_low),    # Low frequencies
        (300 / (sr / 2), 3000 / (sr / 2), 1),         # Mid frequencies
        (3000 / (sr / 2), 15000 / (sr / 2), gain_high)  # High frequencies
    ]

    # Apply a bandpass filter for each band and combine them
    y_adjusted = np.zeros_like(y)
    for low_freq_normalized, high_freq_normalized, gain in eq_bands:
        sos = butter(N=2, Wn=[low_freq_normalized, high_freq_normalized], btype='band', output='sos')
        filtered = sosfilt(sos, y) * gain
        y_adjusted += filtered

    return y_adjusted

def adjust_tempo(y, sr, target_tempo):
    # Step 1: Determine the current tempo
    current_tempo, _ = librosa.beat.beat_track(sr=sr)

    # Step 2: Calculate the tempo ratio
    tempo_ratio = target_tempo / current_tempo

    # Step 3: Time-stretch the audio
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_ratio)

    return y_stretched

def apply_vad(audio, sample_rate, aggressiveness=3, window_duration=0.03, vad_max_silence=0.7):
    window_length = int(window_duration * sample_rate)
    energy = np.square(audio)
    energy_smoothed = np.convolve(energy, np.ones(window_length) / window_length, mode='same')
    threshold = np.percentile(energy_smoothed, 100 - aggressiveness)
    above_threshold = np.array(energy_smoothed > threshold)  # Convert to a NumPy array
    vad_segments = []
    start_sample = None
    for i, is_speech in enumerate(above_threshold):
        if is_speech and start_sample is None:
            start_sample = i
        elif not is_speech and start_sample is not None:
            if i - start_sample >= vad_max_silence * sample_rate:
                vad_segments.append({'start': start_sample, 'stop': i, 'is_speech': True})
                start_sample = None
    if start_sample is not None:
        vad_segments.append({'start': start_sample, 'stop': len(audio), 'is_speech': True})
    return vad_segments

def prepare_for_vad(audio, audio_sample_rate, target_sr=16000):
    if np.max(np.abs(audio)) <= 1:
        print("Audio is already normalized")
    else:
        audio = audio.astype(np.float32) / 32768
    if audio_sample_rate != target_sr:
        audio = resampy.resample(audio.astype(float), audio_sample_rate, target_sr)
    audio = (audio * 32768).astype(np.int16)
    return audio, target_sr

def reduce_noise(audio, sr, noise_sample_length=10000, prop_decrease=1.0):
    noise_sample_length = min(noise_sample_length, len(audio))
    if len(audio) < noise_sample_length:
        padding = np.zeros(noise_sample_length - len(audio))
        audio = np.concatenate((audio, padding))
    try:
        audio_denoised = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
    except Exception as e:
        print(f"Error during noise reduction: {e}")
        audio_denoised = audio
    return audio_denoised

def clean_audio(audio_data, sample_rate=44100):
    audio_denoised = reduce_noise(audio_data, sample_rate, noise_sample_length=10000, prop_decrease=1.0)
    audio_denoised_resampled, new_sample_rate = prepare_for_vad(audio_denoised, sample_rate)
    segments = apply_vad(audio_denoised_resampled, new_sample_rate)
    speech_audio = np.concatenate(
        [audio_denoised_resampled[seg['start']:seg['stop']] for seg in segments if seg['is_speech']]
    )
    if len(speech_audio) == 0:
        print("No segments containing speech were found.")
        return None
    return speech_audio, new_sample_rate


def prep_audio(file_path, output_path):
    print(output_path)
    # Constants for target adjustments
    target_rms = 0.075 * (2 ** 15)
    target_centroid = (2328.5 + 1861.1) / 2
    target_bandwidth = (1864.1 + 1432.0) / 2
    target_tempo = (152 + 172.3) / 2

    audio = AudioSegment.from_file(file_path)

    try:
        # Step 1: Clean audio
        audio_data, sample_rate = sf.read(file_path)
        cleaned_audio, new_sample_rate = clean_audio(audio_data, sample_rate)
        if cleaned_audio is not None:
            sf.write(output_path, cleaned_audio, new_sample_rate)
            audio = AudioSegment.from_file(output_path)

        # Step 2: Normalize RMS if needed
        current_rms = audio.rms
        if abs(current_rms - target_rms) / target_rms > 0.01:  # Allow small deviation
            audio = adjust_rms(audio, target_rms)
            audio.export(output_path, format="wav")
            audio = AudioSegment.from_file(output_path)

        # Step 3: Adjust Spectral Properties if needed
        y, sr = librosa.load(output_path, sr=None)
        current_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        current_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        if abs(current_centroid - target_centroid) / target_centroid > 0.01 or abs(
                current_bandwidth - target_bandwidth) / target_bandwidth > 0.01:  # Allow small deviation
            y = adjust_spectral_properties(y, sr, target_centroid, target_bandwidth)
            sf.write(output_path, y, sr)

        # Step 4: Adjust Tempo if needed
        y, sr = librosa.load(output_path, sr=None)
        current_tempo, _ = librosa.beat.beat_track(sr=sr)
        if abs(current_tempo - target_tempo) / target_tempo > 0.01:  # Allow small deviation
            y = adjust_tempo(y, sr, target_tempo)
            sf.write(output_path, y, sr)

        # Step 5: Convert audio to mono and 16kHz
        converted_path = convert_audio_to_mono_16k(output_path, output_path)
    except Exception as e:
        print(f"Error processing audio file: {e}")
        converted_path = output_path  # Return the original file path

    print(converted_path)
    return converted_path


# Function to process each audio file in the input folder to extract embeddings
def process_audio_files(input_folder):

    embeddings_list = []
    audio_files = []

    prepped_audio_folder = os.path.join("voice", "prepped_audio")
    os.makedirs(prepped_audio_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(prepped_audio_folder, filename)
            prepped_audio_path = prep_audio(file_path, output_path)
            embeddings = extract_sb_embeddings(prepped_audio_path)
            embeddings_list.append(embeddings)
            audio_files.append(filename)

    return np.array(embeddings_list), audio_files

# Function to train a Gaussian Mixture Model (GMM) on the extracted embeddings
def train_gmm(embeddings):
    n_samples = len(embeddings)
    if n_samples == 1:
        embeddings = np.concatenate([embeddings, embeddings])
    n_components = min(n_samples, 10)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(embeddings)
    return gmm

# Function to save the trained GMM model to a file
def save_gmm_model(gmm, output_path):
    joblib.dump(gmm, output_path)
    print(f"GMM model saved to {output_path}")

def clone_voice(input_folder, gmm_save_path):
    print("Processing audio files to extract embeddings...")
    embeddings, audio_files = process_audio_files(input_folder)
    print("Training GMM model...")
    gmm=train_gmm(embeddings)
    save_gmm_model(gmm, gmm_save_path)
    print(f"GMM model trained and saved to {gmm_save_path}")
    # Delete the processed audio files after embeddings extraction
    prepped_audio_folder = "voice/prepped_audio"
    for file in os.listdir(prepped_audio_folder):
        file_path = os.path.join(prepped_audio_folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    os.rmdir(prepped_audio_folder)
# Function to extract an X-vector from a trained Gaussian Mixture Model (GMM)
def extract_xvector_from_gmm(trained_gmm):
    x_vector = trained_gmm.means_[0]
    x_vector /= np.linalg.norm(x_vector)
    return x_vector

# Function to blend two GMMs and train a new GMM on the combined embeddings
def gmm_blender(gmm1, gmm2, ratio):
    # Extract X-vectors from both GMMs
    xvector1 = extract_xvector_from_gmm(gmm1)
    xvector2 = extract_xvector_from_gmm(gmm2)

    # Duplicate the first GMM's X-vectors based on the ratio
    if ratio > 1:
        xvector1 = np.tile(xvector1, (int(ratio), 1))

    # Combine the embeddings
    combined_embeddings = np.vstack([xvector1, xvector2])

    # Train a new GMM on the combined embeddings
    blended_gmm = train_gmm(combined_embeddings)

    return blended_gmm

# Load or download the transformers model
def load_or_download_transformers_model(model_class, model_name, local_path):
    if os.path.exists(local_path):
        print(f"Loading {model_name} from {local_path}...")
        model = model_class.from_pretrained(local_path)
    else:
        print(f"Downloading and saving {model_name} to {local_path}...")
        model = model_class.from_pretrained(model_name)
        model.save_pretrained(local_path)
    return model

def generate_speech(text, api_key="de96dfdef49e4a9410a97c2c1b12702cf173e8d8", voice='aura-asteria-en', output_format='mp3'):
    """
    Generate speech from text using Deepgram's Text-to-Speech API.

    Parameters:
    - text (str): The input text to be converted to speech.
    - api_key (str): Deepgram API key.
    - voice (str): The Deepgram voice model to use.
    - output_format (str): The output file format ('mp3', 'wav', etc.).

    Returns:
    - str: Path to the generated audio file, or None if an error occurs.
    """
    # Define the output file path
    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"speech_output.{output_format}")

    # Deepgram API endpoint
    url = "https://api.deepgram.com/v1/speak"

    # Request payload
    payload = {
        "text": text,
        "model": voice,
        "encoding": output_format
    }

    # Headers with API key
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # Send request to Deepgram API
        response = requests.post(url, json=payload, headers=headers)

        # Check for a successful response
        if response.status_code == 200:
            # Save the audio file
            with open(output_file, "wb") as audio_file:
                audio_file.write(response.content)
            return output_file
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception while generating speech: {e}")
        return None

# # Function to generate speech from text using a given speaker embedding
# def generate_speech(text, speaker_embedding):
#     print("Loading TTS models...")
#     model_dir = "models"
#     tts_model_name = "microsoft/speecht5_tts"
#     tts_model_path = os.path.join(model_dir, "microsoft_speecht5_tts")
#     tts_model = load_or_download_transformers_model(SpeechT5ForTextToSpeech, tts_model_name, tts_model_path)
#     print("TTS models loaded...")
#     input_text = number_to_text(text)
#     # Ensure speaker embedding is of type Float
#     speaker_embedding = speaker_embedding.float()
#
#     # Measure the text to get the word count
#     text_metrics = measure_text(input_text)
#     word_count = text_metrics.word_count
#
#     # Check if the text needs to be split
#     if word_count <= 40:
#         # Prepare the text-to-speech synthesizer
#         synthesiser = pipeline("text-to-speech", tts_model_name)
#         # Generate speech for the entire text
#         speech_data = synthesiser(input_text, forward_params={"speaker_embeddings": speaker_embedding})
#         speech = dub_array(speech_data)
#         return speech
#     else:
#         # Split the text into chunks of word_limit
#         text_chunks = split_text(input_text, word_limit=40)
#
#         # Prepare the text-to-speech synthesizer
#         synthesiser = pipeline("text-to-speech", tts_model_name)
#
#         # Generate speech for each chunk and collect the audio data
#         audio_data_list = []
#         for chunk in text_chunks:
#             speech = synthesiser(chunk, forward_params={"speaker_embeddings": speaker_embedding})
#             audio_data_list.append(speech)
#
#         # Combine the audio data into one continuous audio segment
#         combined_speech = combine_audio_arrays(audio_data_list)
#
#         return combined_speech

# Function to process audio files and generate speech using extracted x-vectors
def speech_with_gmm(text, gmm):
    # Extract xvector from GMM
    # xvector = extract_xvector_from_gmm(gmm)
    # speaker_embedding = torch.tensor(xvector).unsqueeze(0).float()  # Ensure it's of type Float

    # Generate speech
    # speech = generate_speech(text, speaker_embedding)
    speech = generate_speech(text)
    return speech



def play_audio_file(path):
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()



def tts_with_gmm(text, gmm):
    gmm_base = joblib.load(gmm)
    speech = speech_with_gmm(text, gmm_base)

    if isinstance(speech, AudioSegment):

        def play():
            play_audio(speech)

        threading.Thread(target=play).start()
    else:
        print("Error: The TTS output is not an AudioSegment object.")
        raise TypeError("Invalid audio format")

def tts(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Convert text to speech
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()

# Function to load the pre-trained SpeechBrain model for speaker recognition
def load_sb_model():
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
    return classifier


# Function to extract embeddings using the SpeechBrain model
def extract_sb_embeddings(audio_path):
    xvmodel = load_sb_model()
    signal, fs = torchaudio.load(audio_path)
    sb_embeddings = xvmodel.encode_batch(signal)
    return sb_embeddings.squeeze().cpu().numpy()


# Function to load the pre-trained pyannote model
def load_pn_model():
    pnmodel = Inference("pyannote/wespeaker-voxceleb-resnet34-LM")
    return pnmodel


# Function to extract embeddings using the pyannote model
def extract_pn_embeddings(audio_path):
    extractor = load_pn_model()
    pn_embeddings = extractor(audio_path)
    return pn_embeddings.data


# Main function to extract all audio features
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Extract all features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).flatten()
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max).flatten()
    lpc_coeffs = librosa.lpc(y, order=13).flatten()
    plp = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, lifter=22).flatten()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).flatten()
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]
    angles = np.angle(roots)
    freqs = angles * (sr / (2 * np.pi))
    formant_freqs = np.sort(freqs).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y).flatten()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
    spectral_flatness = librosa.feature.spectral_flatness(y=y).flatten()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).flatten()
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    f0 = np.array([pitch[np.argmax(mag)] for pitch, mag in zip(pitches.T, magnitudes.T) if np.max(mag) > 0]).flatten()
    frame_size = 2048
    short_time_energy = np.array(
        [np.sum(np.abs(y[i:i + frame_size] ** 2)) for i in range(0, len(y), frame_size)]).flatten()
    sb_embeddings = extract_sb_embeddings(file_path).flatten()
    pn_embeddings = extract_pn_embeddings(file_path).flatten()

    # Combine all features into one array
    all_features = np.concatenate([
        mfccs, log_mel_spectrogram, lpc_coeffs, plp, chroma, formant_freqs,
        zcr, spectral_centroid, spectral_bandwidth, spectral_flatness,
        spectral_contrast, f0, short_time_energy, sb_embeddings, pn_embeddings
    ])

    return all_features


# Function to train and save a single GMM for all combined features
def train_and_save_gmm(features, output_path, name, n_components=50, max_iter=500):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter)
    gmm.fit(features.reshape(-1, 1))

    model_path = os.path.join(output_path, f"{name}.pkl")
    with open(model_path, 'wb') as model_file:
        pickle.dump(gmm, model_file)

def extract_and_train_gmms(audio_file, output_path, name, n_components=50, max_iter=500):
    """
    Extract features from an audio file and train GMMs on them.

    Args:
    - audio_file (str): Path to the audio file.
    - output_path (str): Path to save the trained GMMs.
    - n_components (int): Number of components for GMMs.
    - max_iter (int): Maximum number of iterations for GMM training.
    """
    # Extract features from the audio file
    features = extract_audio_features(audio_file)

    # Train GMMs and save them
    train_and_save_gmm(features, output_path, name, n_components=n_components, max_iter=max_iter)

# Function to match an audio file against GMMs in a directory
def match_audio_with_gmms(audio_file, gmm_directory, threshold=None):
    extracted_features = extract_audio_features(audio_file)

    best_score = float('-inf')
    best_match = "Unknown Speaker"

    for gmm_file in os.listdir(gmm_directory):
        if gmm_file.endswith('.pkl'):
            model_path = os.path.join(gmm_directory, gmm_file)

            with open(model_path, 'rb') as model_file:
                gmm = pickle.load(model_file)

            score = gmm.score(extracted_features.reshape(-1, 1))
            print(gmm_file)
            print(score)
            if score > best_score:
                best_score = score
                best_match = gmm_file.replace('.pkl', '')

    if threshold is not None and best_score < threshold:
        return "Unknown Speaker"

    return best_match



def load_audio(file_path):
    return AudioSegment.from_file(file_path)

def diarize_audio(file_path):
    # Initialize the pre-trained pipeline for speaker diarization including overlapped speech
    pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token='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')

    # Apply the pipeline to an audio file
    diarization = pipeline(file_path)

    return diarization


def segment_and_save_audio(diarization, audio, output_dir):
    # Empty the output directory before starting
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    segments = []
    current_speaker = None
    current_start = None
    current_end = None
    current_tracks = []
    current_audio_segments = AudioSegment.empty()  # Initialize as an empty AudioSegment

    for segment, track, label in diarization.itertracks(yield_label=True):
        start = segment.start
        end = segment.end
        speaker = label

        if speaker == current_speaker:
            # Extend the current segment and append new audio to the current segment
            current_end = end
            current_tracks.append(track)
            current_audio_segments += audio[start * 1000:end * 1000]  # Combine the audio segments
        else:
            # Save the previous segment if it exists and is at least 1 second long
            if current_speaker is not None and (current_end - current_start) >= 1:
                file_name = f"speaker_{current_speaker}_start_{current_start:.2f}_end_{current_end:.2f}.wav"
                output_path = os.path.join(output_dir, file_name)
                current_audio_segments.export(output_path, format="wav")

                segments.append({
                    "speaker": current_speaker,
                    "start": str(timedelta(seconds=current_start)),
                    "end": str(timedelta(seconds=current_end)),
                    "tracks": current_tracks,
                    "file": output_path
                })

            # Start a new segment
            current_speaker = speaker
            current_start = start
            current_end = end
            current_tracks = [track]
            current_audio_segments = audio[start * 1000:end * 1000]  # Initialize with the current segment

    # Save the last segment if it is at least 1 second long
    if current_speaker is not None and (current_end - current_start) >= 1:
        file_name = f"speaker_{current_speaker}_start_{current_start:.2f}_end_{current_end:.2f}.wav"
        output_path = os.path.join(output_dir, file_name)
        current_audio_segments.export(output_path, format="wav")

        segments.append({
            "speaker": current_speaker,
            "start": str(timedelta(seconds=current_start)),
            "end": str(timedelta(seconds=current_end)),
            "tracks": current_tracks,
            "file": output_path
        })

    return segments

def separate_speakers(file_path, output_dir="segmented_audio"):
    # Load the audio file
    audio = load_audio(file_path)
    # Diarize the audio file
    diarization = diarize_audio(audio)

    # Segment the audio and save the clips
    segments = segment_and_save_audio(diarization, audio, output_dir)

    return segments

def find_speakers(audio_file, gmm_dir, threshold=-5.2, unknown_speakers=False):
    # Separate the audio into segments based on speakers
    speaker_segments = separate_speakers(audio_file)
    speaker_name_map = {}
    failed_segments = []

    # Iterate over each segment and match it to a speaker using the GMMs
    for segment_info in speaker_segments:
        segment_path = segment_info['file']
        speaker_label = segment_info['speaker']

        try:
            # Identify the speaker using the GMMs
            speaker_name = match_audio_with_gmms(segment_path, gmm_dir, threshold)
            # Update the speaker name map
            speaker_name_map[speaker_label] = speaker_name
            # Update the segment info with the identified speaker name
            segment_info['speaker'] = speaker_name
        except Exception as e:
            print(f"Error processing segment {segment_path}: {e}")
            # Add to the failed list if identification fails
            failed_segments.append(segment_info)

    # Handle failed segments
    for failed_segment in failed_segments:
        speaker_label = failed_segment['speaker']

        # Use the identified name for the same speaker label, if available
        if speaker_label in speaker_name_map:
            failed_segment['speaker'] = speaker_name_map[speaker_label]
        else:
            if unknown_speakers is True:
                failed_segment['speaker'] = "Unknown Speaker"
            else:
                speaker_segments.remove(failed_segment)  # Remove segment if unknown_speakers=False

    # If no recognized speakers are found, return an empty list
    if not speaker_segments:
        print("No recognized speaker")
        return []

    return speaker_segments

def transcribe_audio(audio_data):
    """
    Method for transcribing audio using Google Speech-to-Text.

    Args:
        audio_data (sr.AudioData): Captured audio data.

    Returns:
        str: Transcribed text.
    """
    recognizer = sr.Recognizer()

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio_data)
        print("Returning text...")
        return text.strip()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return ""

def diarized_transcription(audio_file, gmm_dir, unknown_speakers=False):
    """
    Transcribe the audio file with speaker diarization.

    Args:
        audio_file (str): Path to the audio file.
        gmm_dir (str): Directory containing GMMs for speaker identification.
        unknown_speakers (bool): Whether to keep segments with unknown speakers.

    Returns:
        str: Diarized text transcription or a message if no recognized speaker.
    """
    # Separate the audio into segments and identify speakers
    speaker_segments = find_speakers(audio_file, gmm_dir, unknown_speakers=unknown_speakers)

    if not speaker_segments:
        return "No recognized speaker"

    # Sort the segments based on their start times
    speaker_segments.sort(key=lambda x: x['start'])

    transcribed_text = []

    # Transcribe each segment
    for segment_info in speaker_segments:
        speaker_name = segment_info['speaker']

        # Skip transcription if the speaker is "Unknown Speaker" and unknown_speakers is False
        if not unknown_speakers and speaker_name == "Unknown Speaker":
            continue

        segment_path = segment_info['file']

        # Load the audio segment for transcription
        audio_segment = AudioSegment.from_file(segment_path, format="wav")

        # Check if the audio is empty before attempting to transcribe it
        if len(audio_segment) == 0:
            print(f"Skipping empty audio segment: {segment_path}")
            continue

        audio_data = sr.AudioData(audio_segment.raw_data, audio_segment.frame_rate, audio_segment.sample_width)

        # Transcribe the segment
        text_transcription = transcribe_audio(audio_data)
        if text_transcription:
            transcribed_text.append(f"{speaker_name} said \"{text_transcription}\" At:{formatted_time} ")

    # Concatenate the transcriptions into a single text string
    diarized_text = ' '.join(transcribed_text)

    return diarized_text if transcribed_text else "No recognized speaker"

async def transcribe_audio_async(audio_data):
    """
    Asynchronous method for transcribing audio using Google Speech-to-Text.

    Args:
        audio_data (sr.AudioData): Captured audio data.

    Yields:
        str: Transcribed text.
    """
    recognizer = sr.Recognizer()

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio_data)
        print("Yielding text...")
        yield text.strip()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


def check_text_for_wake_word(text, wake_words):
    """
    Check if any of the wake words are in the given text.

    Args:
        text (str): The transcribed text.
        wake_words (list): List of wake words to check for.

    Returns:
        str or None: The matched wake word if found, else None.
    """
    for word in wake_words:
        if word.lower() in text.lower():
            return word
    return None


def record_and_transcribe(wake_words=None, timer=None, word_count=None, gmm_dir=None, unknown_speakers=False,
                          output="text"):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    dialog_string = ""
    speech_audio = []
    start_time = time.time()

    while True:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source)
            print("Recording... Speak now.")

            audio_data = recognizer.listen(source)
            print("Recording complete.")

        if gmm_dir is not None:
            temp_filename = "temp_audio.wav"
            with open(temp_filename, "wb") as temp_file:
                temp_file.write(audio_data.get_wav_data())

            transcription = diarized_transcription(temp_filename, gmm_dir, unknown_speakers)

            if transcription == "No recognized speaker":
                continue
        else:
            transcription = transcribe_audio(audio_data)

        current_time = time.time()
        time_passed = current_time - start_time

        if timer is not None and time_passed > timer:
            dialog_string = transcription
            speech_audio = [audio_data] if output != "text" else []
            start_time = current_time
        else:
            if word_count is None:
                dialog_string += " " + transcription
            else:
                dialog_string = transcription

            dialog_string = dialog_string.strip()

            if word_count is not None:
                metrics = measure_text(dialog_string)
                if metrics.word_count > word_count:
                    dialog_string = transcription

            if output != "text":
                speech_audio.append(audio_data)

        if wake_words:
            wake_word_found = check_text_for_wake_word(dialog_string, wake_words)
            if wake_word_found:
                speaker_name = None
                if gmm_dir is not None:
                    speaker_name = diarized_transcription(temp_filename, gmm_dir, unknown_speakers).split()[0]

                result = {
                    "wake_word": wake_word_found,
                    "text": dialog_string,
                    "speaker": speaker_name if speaker_name else "Unknown"
                }

                if output == "audio":
                    return combine_audio_data(speech_audio), result
                elif output == "text":
                    return result
                else:
                    return combine_audio_data(speech_audio), result

                dialog_string = ""
                speech_audio = []
        else:
            if output == "audio":
                return combine_audio_data(speech_audio)
            elif output == "text":
                return dialog_string
            else:
                return combine_audio_data(speech_audio), dialog_string


async def async_record_and_transcribe(wake_words=None, timer=None, word_count=None, gmm_dir=None,
                                      unknown_speakers=False, output="text"):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    dialog_string = ""
    speech_audio = []
    start_time = time.time()

    while True:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source)
            print("Recording... Speak now.")

            audio_data = recognizer.listen(source)
            print("Recording complete.")

        if gmm_dir is not None:
            temp_filename = "temp_audio.wav"
            with open(temp_filename, "wb") as temp_file:
                temp_file.write(audio_data.get_wav_data())

            transcription = diarized_transcription(temp_filename, gmm_dir, unknown_speakers)

            if transcription == "No recognized speaker":
                continue
        else:
            transcription = transcribe_audio_async(audio_data)

        current_time = time.time()
        time_passed = current_time - start_time

        if timer is not None and time_passed > timer:
            dialog_string = transcription
            speech_audio = [audio_data] if output != "text" else []
            start_time = current_time
        else:
            if word_count is None:
                dialog_string += " " + transcription
            else:
                dialog_string = transcription

            dialog_string = dialog_string.strip()

            if word_count is not None:
                metrics = measure_text(dialog_string)
                if metrics.word_count > word_count:
                    dialog_string = transcription

            if output != "text":
                speech_audio.append(audio_data)

        if wake_words:
            wake_word_found = check_text_for_wake_word(dialog_string, wake_words)
            if wake_word_found:
                speaker_name = None
                if gmm_dir is not None:
                    speaker_name = diarized_transcription(temp_filename, gmm_dir, unknown_speakers).split()[0]

                result = {
                    "wake_word": wake_word_found,
                    "text": dialog_string,
                    "speaker": speaker_name if speaker_name else "Unknown"
                }

                if output == "audio":
                    yield combine_audio_data(speech_audio), result
                elif output == "text":
                    yield result
                else:
                    yield combine_audio_data(speech_audio), result

                dialog_string = ""
                speech_audio = []
        else:
            if output == "audio":
                yield combine_audio_data(speech_audio)
            elif output == "text":
                yield dialog_string
            else:
                yield combine_audio_data(speech_audio), dialog_string


sentences = {
    1: "Seven slippery snakes slither silently through the tall grass, seeking sunshine",
    2: "The brilliant blue bird perched on the branch, singing a sweet serenade at sunrise",
    3: "Glistening glaciers glowed under the golden sun, melting into a cascading waterfall",
    4: "The old oak tree stood tall, its leaves rustling in the gentle evening Breeze",
    5: "Whispering winds wound their way through the winding, wooded Path",
    6: "The mighty mountain lion leaped gracefully over the rocky ledge, landing softly below",
    7: "A fleet of fast, fiery foxes raced across the frozen field, leaving tracks in the snow",
    8: "The clever cat cautiously crept closer to the curious crow, watching its every move",
    9: "Bright beams of sunlight streamed through the stained glass, casting colorful shadows",
    10: "The quick quail quietly quivered, hiding from the hungry hawk overhead",
    11: "The relentless rain rattled the rooftop, resonating in the silent night",
    12: "Soft, silken sands shifted beneath the feet of the strolling couple on the shore",
    13: "The thunderous roar of the waterfall echoed through the cavern, shaking the ground",
    14: "The luminous lantern lit the way through the dark, damp, and dreary dungeon",
    15: "The busy bumblebee buzzed busily between blooming, bright, and beautiful blossoms",
    16: "Crimson clouds cascaded across the horizon as the sun dipped below the distant hills",
    17: "The tall tower teetered precariously in the turbulent wind, creaking ominously",
    18: "A flock of feathery flamingos fluttered their wings, rising gracefully into the air",
    19: "The gentle giant giraffe grazed on the green, leafy branches of the towering trees",
    20: "The eager eagle soared high above the mountains, scanning the land below for prey",
    21: "The ancient archway stood as a testament to time, weathered but unyielding",
    22: "Golden grains of sand sparkled in the sun, stretching endlessly along the seashore",
    23: "The wise old owl hooted softly from its perch, hidden deep within the forest",
    24: "A symphony of crickets played in the background as the moon rose in the clear night sky",
    25: "The fierce falcon dived rapidly, its sharp eyes locked on its unsuspecting target",
    26: "The gurgling brook wound its way through the verdant valley, reflecting the azure sky",
    27: "The shimmering stars dotted the midnight sky, twinkling like diamonds",
    28: "The jagged peaks of the mountain range cut into the sky, sharp and foreboding",
    29: "The playful puppy pounced on the pile of autumn leaves, scattering them everywhere",
    30: "The crisp crunch of fallen leaves underfoot echoed in the quiet, cool air of the forest",
    31: "The vast ocean stretched out before them, its waves crashing against the rocky cliffs",
    32: "The bold bear bounded through the thick underbrush, its heavy paws thudding on the ground",
    33: "The flickering flames danced in the fireplace, casting a warm glow throughout the room",
    34: "The delicate daffodil swayed gently in the breeze, its yellow petals catching the light",
    35: "The distant drumbeat reverberated through the night, a call to the tribal dance",
    36: "The mysterious mist enveloped the landscape, obscuring the path ahead",
    37: "The swirling snowflakes fell silently, covering the world in a blanket of white",
    38: "The mischievous monkey swung from vine to vine, chattering loudly as it moved",
    39: "The jaguar’s amber eyes glinted in the darkness as it stalked its prey silently",
    40: "The ancient ruins crumbled under the weight of time, yet still stood in defiance",
    41: "The sleek submarine sliced through the depths of the ocean, silent and unseen",
    42: "The glowing embers of the campfire pulsed rhythmically, holding the night's chill at bay",
    43: "The gentle hum of the hive filled the air as the bees busied themselves with their work",
    44: "The old, rusty gate creaked open, revealing a hidden garden bursting with color",
    45: "The sly fox slipped silently into the henhouse, its eyes gleaming with cunning",
    46: "The rolling thunder grew louder, signaling the approach of a fierce storm",
    47: "The reflective surface of the lake mirrored the sky, creating a perfect illusion",
    48: "The vast savannah stretched out before them, dotted with acacia trees and grazing animals",
    49: "The rhythmic sound of the waves lulled them into a peaceful slumber on the beach",
    50: "The mighty oak tree's roots dug deep into the earth, anchoring it firmly in place",
    51: "The colorful chameleon shifted its hues, blending seamlessly into its surroundings",
    52: "The distant lighthouse beacon swept across the ocean, guiding ships safely to shore",
    53: "The rhythmic clatter of the train on the tracks was a comforting sound in the night",
    54: "The curious kangaroo hopped closer, its large eyes filled with innocent wonder",
    55: "The solitary wolf howled mournfully at the full moon, its call echoing through the forest",
    56: "The glittering city skyline reflected in the calm waters of the river below",
    57: "The powerful stallion galloped across the open plain, its mane flying in the wind",
    58: "The ancient manuscript was filled with faded, handwritten notes and mysterious symbols",
    59: "The sleek dolphin leaped gracefully from the water, performing an aerial dance",
    60: "The peaceful meadow was filled with the scent of wildflowers and the sound of buzzing bees",
    61: "The quick brown fox jumps over the lazy dog, while the curious zebra quietly observes from afar, wondering if the vibrant parrot will sing a song at dusk"
}
def display_text(text):
    global root
    # Create the main window
    root = tkinter.Tk()
    root.title("Text Display")

    # Create a Text widget
    text_box = tkinter.Text(root, height=10, width=50)
    text_box.pack()

    # Insert the text into the Text widget
    text_box.insert(tkinter.END, text)

    # Run the application
    root.mainloop()

def close_text_box():
    global root
    if root is not None:
        root.destroy()
        root = None

def remember_user(username, sentence_num=1):
    speaker_audio = []
    max_segments = 12
    current_sentence_num = sentence_num

    while len(speaker_audio) < max_segments:
        sentence = sentences.get(current_sentence_num, None)
        if sentence is None:
            break

        # Display the sentence to the user
        text = f"Please read the following sentence out loud: {sentence}"
        display_text(text)  # Replace with the actual display function if needed

        # Use ASR to capture audio
        wake_word = sentence.split()[-1]
        print(wake_word)
        audio = record_and_transcribe(wake_words=wake_word, output="audio")
        speaker_audio.append(audio)
        close_text_box()
        # Move to the next sentence
        current_sentence_num += 1

        # Ask if the user wants to continue every 10 sentences
        if current_sentence_num % 10 == 0:
            continuation_prompt = "Would you like to continue?"
            print(continuation_prompt)  # Replace with the actual display function if needed
            response_text = record_and_transcribe(wake_words=None, output="text")
            if check_text(response_text, "Yes"):
                continue  # Continue the loop
            elif check_text(response_text, "No"):
                break  # Exit early
            else:
                print("Unrecognized response. Exiting.")
                break

    # Combine all collected audio segments
    combined_audio = combine_audio_data(speaker_audio)

    # Save the combined audio
    temp_filename = "combined_audio.wav"
    combined_audio.export(temp_filename, format="wav")

    # Train and save GMMs
    gmm_dir = f"data/user_data/{username}"
    os.makedirs(gmm_dir, exist_ok=True)
    extract_and_train_gmms(temp_filename, gmm_dir, name=username)

    return current_sentence_num
