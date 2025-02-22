# GATOR.py, By: Chance Brownfield
from Brain.utils import *
import chromadb
from chromadb.config import Settings
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Generation Augmented Tree Organized Retrieval
class Gator:
    def __init__(self, chroma_path="chroma_db"):
        self.client = chromadb.PersistentClient(Settings(path=chroma_path))
        self.profile_tree = self.ProfileTree(self.client)
        self.conversation_tree = self.ConversationTree(self.client)
        self.knowledge_tree = self.KnowledgeTree(self.client)
        self.command_tree = self.CommandTree(self.client)


    def process_knowledge(self, query, history="", location="", time_date=""):
        answers = generate_knowledge(query, history, location, time_date)
        results = self.knowledge_tree.process_answers_and_search(query, answers)
        return results

    def process_conversation(self, query, id, type="input"):
        """
        Processes a conversation query, retrieving relevant history and responding.
        """
        if type == "input":
            relevant_context = self.conversation_tree.add_input_leaf(query, id)
            return relevant_context
        else:
            self.conversation_tree.add_output_leaf(query, id)

    def process_actions(self, query, user_id, bot_id, history="", location="", time_date=""):
        """
        Determines the appropriate actions based on user input.

        Args:
            query (str): The user input query.
            user_id (str): The ID of the user for profile retrieval.
            bot_id (str): The ID of the bot for profile retrieval.
            history (str): Past conversation or interaction history.
            location (str): User's current location (if relevant).
            time_date (str): Current date and time (if relevant).

        Returns:
            dict: A dictionary containing the executed commands and retrieved data.
        """

        # Step 1: Retrieve relevant commands
        relevant_commands = self.command_tree.search_relevant_commands(query, top_k=3)
        available_actions = json.dumps(relevant_commands)

        # Step 2: Infer actions from the query
        prompt = f"""            
           Here are the available actions:
           {available_actions}

           Contextual Information:
           - History: {history}
           - Location: {location}
           - Date/Time: {time_date}

           User Input: {query}

           Respond only in strict JSON format.
           """

        try:
            response = infer_actions(prompt)
            actions = response.get("actions", [])
        except Exception as e:
            raise ValueError(f"Error inferring actions: {str(e)}")

        matched_commands = []
        retrieved_data = {}

        for action in actions:
            action_type = action.get("type")

            if action_type == "command":
                command_name = action.get("command_name")
                parameters = action.get("parameters", {})

                # Find matching command metadata
                command_metadata = next((cmd for cmd in relevant_commands if cmd["name"] == command_name), None)
                if command_metadata:
                    command_type = command_metadata["type"]
                    matched = process_commands(command_name, command_type, parameters)
                    matched_commands.extend(matched)

            elif action_type == "retrieval":
                source = action.get("source")
                profile = action.get("profile", None)
                retrieval_query = action.get("query")

                if source == "KnowledgeTree":
                    # Only search if explicitly required
                    retrieved_data["knowledge"] = self.process_knowledge(
                        retrieval_query, history, location, time_date
                    )
                elif source == "ProfileTree" and profile:
                    if profile == "user_profile":
                        retrieved_data["user_profile"] = self.profile_tree.search_leaves(user_id, retrieval_query)
                    elif profile == "bot_profile":
                        retrieved_data["bot_profile"] = self.profile_tree.search_leaves(bot_id, retrieval_query)

        # Step 3: Process shortcuts separately
        shortcut_matches = process_commands(query, "shortcut")
        matched_commands.extend(shortcut_matches)

        # Step 4: Consolidate the final response
        final_commands = " ".join(matched_commands) if matched_commands else None

        return {
            "commands": final_commands,
            "retrieved_data": retrieved_data
        }

    class CommandTree:
        def __init__(self, client):
            self.client = client
            self.collection = client.get_or_create_collection("command_tree")

        def add_command_branch(self, command_name, command_action, command_type, description):
            """
            Adds a new command branch with base attributes and leaves (descriptions).
            """
            metadata = {
                "type": "command",
                "command_name": command_name,
                "command_action": command_action,
                "command_type": command_type,
                "descriptions": [description]
            }
            embedding = generate_embedding(text=description, task="retrieval.passage").numpy()

            self.collection.add(
                documents=[f"Command: {command_name}"],
                metadatas=[metadata],
                embeddings=[embedding.tolist()]
            )

        def update_command_description(self, command_name, new_description):
            """
            Updates a command branch by adding a new description as a leaf.
            """
            command_data = self.collection.get()
            for meta in command_data["metadatas"]:
                if meta["command_name"] == command_name:
                    meta["descriptions"].append(new_description)
                    embedding = generate_embedding(new_description, task="retrieval.passage").numpy()
                    self.collection.update(
                        ids=[meta["id"]],
                        metadatas=[meta],
                        embeddings=[embedding.tolist()]
                    )
                    return

        def search_relevant_commands(self, query, top_k=3):
            """
            Searches for relevant commands based on a semantic query.
            """
            query_embedding = generate_embedding(query, task="retrieval.query").numpy()
            results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

            if not results or "results" not in results or not results["results"]:
                return []

            relevant_commands = []
            for res in results["results"]:
                metadata = res["metadata"]
                if metadata:
                    relevant_commands.append({
                        "name": metadata["command_name"],
                        "description": metadata["descriptions"],
                        "action": metadata["command_action"],
                        "type": metadata["command_type"]
                    })
            return relevant_commands


    class ConversationTree:
        def __init__(self, client):
            self.client = client
            self.collection = client.get_or_create_collection("conversation_tree")
            self.temp_input_leaf = None  # Temporarily hold input leaf before storing

        def add_input_leaf(self, text, id):
            """
            Temporarily stores the input leaf (user message), clusters it,
            and retrieves relevant past conversation branches.
            """
            self.temp_input_leaf = {"text": text, "id": id}  # Save input temporarily

            # Find similar input leaves
            relevant_branches = self.search_relevant_branches(text, id)
            return relevant_branches  # Return whole relevant branches (input + output)

        def add_output_leaf(self, text, id):
            """
            Stores both the input leaf and output leaf as a single branch in the conversation tree.
            """
            if not self.temp_input_leaf:
                print("Error: No input leaf found to link output.")
                return

            input_text = self.temp_input_leaf["text"]

            input_embedding = generate_embedding(input_text, task="retrieval.passage").numpy()
            output_embedding = generate_embedding(text, task="retrieval.passage").numpy()

            # Store input + output as a single branch
            self.collection.add(
                documents=[f"Input: {input_text} | Output: {text}"],
                metadatas=[{"id": id, "type": "branch", "input": input_text, "output": text}],
                embeddings=[input_embedding.tolist()]  # Store input embedding (used for clustering)
            )

            # Reset temporary input
            self.temp_input_leaf = None

        def search_relevant_branches(self, text, id, top_k=3):
            query_embedding = generate_embedding(text, task="retrieval.query").numpy()

            results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

            if not results or "results" not in results or not results["results"]:
                return []  # Return empty list if no relevant results

            relevant_branches = []

            for res in results["results"]:
                branch_metadata = res["metadata"]
                if not branch_metadata:
                    continue

                branch_input = branch_metadata.get("input")
                branch_output = branch_metadata.get("output")

                if not branch_input or not branch_output:
                    continue  # Skip incomplete branches

                # Perform a second similarity check
                full_text = f"Input: {branch_input} | Output: {branch_output}"
                branch_embedding = generate_embedding(full_text, task="retrieval.passage").numpy()

                similarity_score = \
                cosine_similarity(query_embedding.reshape(1, -1), branch_embedding.reshape(1, -1))[0][0]

                if similarity_score > 0.7:
                    relevant_branches.append(
                        {"input": branch_input, "output": branch_output, "score": similarity_score})

            return relevant_branches



    class ProfileTree:
        def __init__(self, client, profile_type="user"):
            """
            Initializes the ProfileTree (for bot or user) based on the provided profile_type.
            """
            self.client = client
            self.profile_type = profile_type  # "user" or "bot"
            self.collection = client.get_or_create_collection(f"{profile_type}_tree")

        def create_branch(self, profile_id, base_data):
            """
            Create a new profile branch with base attributes and an empty list for leaves (behaviors or preferences).
            """
            metadata = {"type": self.profile_type, f"{self.profile_type}_id": profile_id, "base_data": base_data,
                        "leaves": []}
            self.collection.add(
                documents=[f"{self.profile_type.capitalize()} Profile: {base_data['name']}"],
                metadatas=[metadata],
                ids=[profile_id]
            )

        def update_base(self, profile_id, key, value):
            """
            Update a specific attribute in the profile's base data.
            """
            profile_data = self.collection.get(ids=[profile_id])
            if not profile_data["metadatas"]:
                print(f"{self.profile_type.capitalize()} not found.")
                return

            metadata = profile_data["metadatas"][0]
            metadata["base_data"][key] = value

            self.collection.update(
                ids=[profile_id],
                metadatas=[metadata]
            )

        def add_leaf(self, profile_id, text):
            """
            Add a new leaf (opinion, preference, learned behavior) to the profile.
            """
            profile_data = self.collection.get(ids=[profile_id])
            if not profile_data["metadatas"]:
                print(f"{self.profile_type.capitalize()} not found.")
                return

            metadata = profile_data["metadatas"][0]
            metadata["leaves"].append(text)

            embedding = generate_embedding(text, task="retrieval.passage").numpy()
            self.collection.update(
                ids=[profile_id],
                metadatas=[metadata],
                embeddings=[embedding.tolist()]
            )

        def get_base_attribute(self, profile_id, key):
            """
            Retrieve a specific attribute from the profile's base dictionary.
            """
            profile_data = self.collection.get(ids=[profile_id])
            if not profile_data["metadatas"]:
                print(f"{self.profile_type.capitalize()} not found.")
                return None

            return profile_data["metadatas"][0]["base_data"].get(key, None)

        def search_leaves(self, profile_id, query, top_k=3):
            """
            Search the profile's learned behaviors or preferences for relevant information.
            """
            profile_data = self.collection.get(ids=[profile_id])
            if not profile_data["metadatas"]:
                print(f"{self.profile_type.capitalize()} not found.")
                return []

            leaves = profile_data["metadatas"][0]["leaves"]
            leaf_embeddings = [generate_embedding(leaf, task="retrieval.passage").numpy() for leaf in leaves]
            query_embedding = generate_embedding(query, task="retrieval.query").numpy()

            similarities = [cosine_similarity(query_embedding.reshape(1, -1), leaf.reshape(1, -1))[0][0] for leaf in
                            leaf_embeddings]
            sorted_leaves = sorted(zip(leaves, similarities), key=lambda x: x[1], reverse=True)

            return [leaf for leaf, score in sorted_leaves[:top_k] if score > 0.7]  # Filter by relevance threshold

    class KnowledgeTree:
        def __init__(self, client):
            self.client = client
            self.collection = client.get_or_create_collection("knowledge_tree")

        def store_tagged_content(self, tagged_data, collection_name="knowledge_tree"):
            """
            Store tagged content in the RAPTOR tree.
            """
            passages = [entry["answer"] for entry in tagged_data]
            tags = [entry["tag"] for entry in tagged_data]

            # Add tags as metadata for better retrieval
            for passage, tag in zip(passages, tags):
                embedding = generate_embedding(passage, task="retrieval.passage").numpy()
                self.collection.add(
                    documents=[passage],
                    metadatas=[{"tag": tag, "source": "knowledge_tree"}],
                    embeddings=[embedding.tolist()]
                )
            print(f"Stored {len(passages)} tagged entries in {collection_name}.")

        def cluster_branches(self, collection_name="knowledge_tree", num_clusters=5):
            collection = self.collection
            docs = collection.get()["documents"]

            if not docs:
                print("No documents found for clustering.")
                return {}

            # Batch process all embeddings at once
            passages = [doc["document"] for doc in docs]
            embeddings = generate_embedding(passages, task="text-matching").numpy()

            # Perform clustering
            kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)))
            kmeans.fit(embeddings)

            clustered_branches = {i: [] for i in range(num_clusters)}
            for idx, label in enumerate(kmeans.labels_):
                clustered_branches[label].append(passages[idx])

            return clustered_branches

        def search_relevant_content(self, query, collection_name="knowledge_tree", top_k=5):
            """
            Search for relevant content using the query and return matching passages.
            """
            collection = self.collection
            query_embedding = generate_embedding(query, task="retrieval.query").numpy()

            # Perform search
            results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

            return [{"passage": res["document"], "score": res["score"], "tag": res["metadata"].get("tag")} for res in
                    results["results"]]

        def process_answers_and_search(self, query, answers, collection_name="knowledge_tree", num_clusters=5):
            """
            Process and store the query + answers as a single knowledge branch,
            then retrieve the most relevant data.
            """

            # Ensure query is stored alongside its answers
            tagged_data = [{"answer": query, "tag": "query", "parent_query": None}]

            for answer_info in answers:
                if answer_info["tag"] is not None:
                    tagged_data.append(
                        {"answer": answer_info["answer"], "tag": answer_info["tag"], "parent_query": query})

            self.store_tagged_content(tagged_data, collection_name=collection_name)

            # Step 2: Dynamically update clusters instead of re-clustering everything
            clusters = self.cluster_branches(collection_name=collection_name, num_clusters=num_clusters)

            # Step 3: Retrieve relevant knowledge based on the query
            relevant_branches = []
            search_results = self.search_relevant_content(query, collection_name=collection_name)

            for result in search_results:
                relevant_branches.append(
                    f"[{result['tag']}] {result['passage']} (Source: {result.get('parent_query', 'N/A')})")

            return "\n".join(relevant_branches)