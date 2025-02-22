import threading

from .outer.layer1.SpeechRecognition import SpeechRecognition
from .outer.layer2.SpeechGeneration import SpeechGeneration
from .inner.layer2.RAG import Rag
from .inner.layer1.GATOR import Gator
from Brain.utils import *

class Cerebrum:
    def __init__(self, client, user_dir="data/user_data"):
        """
        Initialize the Brain class with necessary components.
        """
        self.state = "stopped"
        self.state_lock = threading.Lock()

        # Core AI Components
        self.recognizer = SpeechRecognition()
        self.speech_generator = SpeechGeneration()
        self.generate = Rag()
        self.gator = Gator()

        # Profile management
        self.user_dir = user_dir
        self.bot_profiles = self.gator.ProfileTree(client, profile_type="bot")
        self.user_profiles = self.gator.ProfileTree(client, profile_type="user")

        # Dynamically build wake word list from bot profiles
        self.wake_word_map = self._build_wake_word_map()

    def _build_wake_word_map(self):
        """
        Fetch bot profile names and map them to bot IDs.
        """
        bot_profile_data = self.bot_profiles.collection.get()
        return {
            entry["base_data"]["name"].lower(): entry["bot_id"]
            for entry in bot_profile_data["metadatas"] if "name" in entry["base_data"]
        }

    def _get_user_id(self, speaker_name):
        """
        Retrieve the user ID from user profiles based on the speaker name.
        """
        user_profile_data = self.user_profiles.collection.get()
        user_name_map = {
            entry["base_data"]["name"].lower(): entry["user_id"]
            for entry in user_profile_data["metadatas"] if "name" in entry["base_data"]
        }
        return user_name_map.get(speaker_name.lower(), None)

    def listen(self):
        """
        Start the listening loop, process speech input, and generate responses.
        """
        with self.state_lock:
            self.state = "listening"

        while True:
            with self.state_lock:
                if self.state != "listening":
                    break  # Exit loop if state changes

            result = self.recognizer.ASR(wake_words=list(self.wake_word_map.keys()), gmm_dir=self.user_dir, unknown_speakers=True)

            if not result or not isinstance(result, dict):
                print("No valid input detected, retrying...")
                continue  # Retry if no valid input

            # Extract wake word, spoken text, and speaker name
            used_wake_word = result.get("wake_word", "").lower()
            spoken_text = result.get("text", "")
            speaker_name = result.get("speaker", "Unknown").lower()

            # Match wake word to bot profile ID
            bot_id = self.wake_word_map.get(used_wake_word, None)

            # Retrieve user profile ID based on speaker name
            user_id = self._get_user_id(speaker_name)

            if bot_id:
                response = self.generate.process(query=spoken_text, user_id=user_id, bot_id=bot_id)
                print(response)
                self.speech_generator.Speech(response)

    def stop(self):
        """
        Stop the listening loop.
        """
        with self.state_lock:
            self.state = "stopped"

