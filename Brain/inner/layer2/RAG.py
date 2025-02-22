# RAG.py, By: Chance Brownfield
from Brain.inner.layer1.GATOR import *

# Retrieval Augmented Generation
class Rag:
    def __init__(self):
        location_data, timedate_data = get_location_and_datetime()
        self.location = location_data
        self.timedate = timedate_data
        self.gator = Gator()
        self.history = ShortMemory()

    def process(self, query, user_id, bot_id):
        # Retrieve user and bot profile attributes
        user_profile = self.gator.profile_tree
        bot_profile = self.gator.profile_tree

        user_info = {
            "name": user_profile.get_base_attribute(user_id, "name"),
            "age": user_profile.get_base_attribute(user_id, "age"),
            "gender": user_profile.get_base_attribute(user_id, "gender"),
            "bio": user_profile.get_base_attribute(user_id, "bio"),
            "disposition": user_profile.get_base_attribute(user_id, "disposition"),
            "censorship": user_profile.get_base_attribute(user_id, "censorship")
        }

        bot_info = {
            "name": bot_profile.get_base_attribute(bot_id, "name"),
            "gender": bot_profile.get_base_attribute(bot_id, "gender"),
            "bio": bot_profile.get_base_attribute(bot_id, "bio"),
            "special_instruction": bot_profile.get_base_attribute(bot_id, "special_instruction")
        }

        disposition = bot_profile.get_base_attribute(bot_id, user_id) or 0

        def get_disposition_level(value):
            if value >= 100:
                return "love"
            elif value >= 80:
                return "adoration"
            elif value >= 60:
                return "fondness"
            elif value >= 40:
                return "liking"
            elif value >= 20:
                return "mild appreciation"
            elif value >= 0:
                return "neutral"
            elif value >= -20:
                return "mild disdain"
            elif value >= -40:
                return "dislike"
            elif value >= -60:
                return "extreme disdain"
            elif value >= -80:
                return "loathing"
            else:
                return "hatred"

        emotion_result = infer_emotion(query, context="", bot_info=bot_info, user_info=user_info,
                                       disposition=disposition)

        new_disposition = max(-100, min(100, disposition + emotion_result["disposition_change"]))
        disposition_level = get_disposition_level(new_disposition)
        disposition_str = f"{bot_info['name']} currently feels {disposition_level} towards {user_info['name']}"

        bot_profile.update_base(bot_id, user_id, new_disposition)

        bot_info["current_emotion"] = emotion_result["emotion"]
        user_info["perceived_emotion"] = emotion_result["empathy"]

        relevant_context = self.gator.process_conversation(query, bot_id, type="input")
        short_memory = self.history.get_history()

        action_results = self.gator.process_actions(
            query=query,
            user_id=user_id,
            bot_id=bot_id,
            history="\n".join([f"- {entry['user_message']} -> {entry['response']}" for entry in short_memory]),
            location=self.location,
            time_date=self.timedate
        )

        long_memory_str = "Relevant Context from Conversation Tree:\n"
        long_memory_str += "\n".join(
            [f"- User: {branch['input']}\n  Bot: {branch['output']}\n  (Score: {branch['score']:.2f})" for branch in
             relevant_context]
        ) if relevant_context else "No relevant past interactions found.\n"

        short_memory_str = "Short-Term Memory:\n"
        short_memory_str += "\n".join(
            [
                f"- [{entry['timestamp']}] User: {entry['user_message']}\n  Bot: {entry['response']}\n  Thought: {entry['thought']}\n  Log: {entry['log']}"
                for entry in short_memory]
        ) if short_memory else "No recent conversation history.\n"

        executed_commands = action_results.get("commands", [])
        knowledge = action_results.get("retrieved_data", {})

        response_data = generate_response(
            query=query,
            bot_name=bot_info["name"],
            context=f"{short_memory_str}\n\n{long_memory_str}\n\n{disposition_str}",
            bot_info=bot_info,
            user_info=user_info,
            knowledge=knowledge,
            executed_commands=executed_commands,
            censorship_instructions=user_info.get("censorship"),
            bot_specific_instructions=bot_info.get("special_instruction")
        )

        response = response_data.get("response", "")
        thought = response_data.get("thought", "")
        log = response_data.get("log", "")

        # Update conversation tree with output response
        self.gator.process_conversation(f"Response: {response} Thoughts: {thought}", bot_id, type="output")

        # Update short-term memory
        self.history.add_entry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_message=query,
            response=response,
            thought=thought,
            log=log
        )
        traits_context = f"{short_memory_str}\n\n{long_memory_str}\n\n{disposition_str}"
        traits_result = infer_traits(
            query=query,
            context=traits_context,
            bot_info=bot_info,
            user_info=user_info,
            bot_response=response
        )

        # For each non-empty key (except 'log') in traits_result, add as a leaf to the corresponding profile
        # User-related keys
        for key in ["user_opinions", "user_preferences", "user_traits", "personal_info", "user_feedback"]:
            if key in traits_result and traits_result[key]:
                for item in traits_result[key]:
                    # Add to the user's profile branch
                    user_profile.add_leaf(user_id, f"{key}: {item}")

        # Bot-related keys
        for key in ["bot_opinions", "bot_traits"]:
            if key in traits_result and traits_result[key]:
                for item in traits_result[key]:
                    # Add to the bot's profile branch
                    bot_profile.add_leaf(bot_id, f"{key}: {item}")

        return response

