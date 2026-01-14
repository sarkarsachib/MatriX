import collections
import datetime
import json
import os
from typing import Dict, Any, List, Optional

class ShortTermMemory:
    """Manages active conversation and local threads (RAM)."""
    def __init__(self, max_size: int = 100):
        self.memory = collections.deque(maxlen=max_size)

    def add_entry(self, entry: Dict[str, Any]):
        """Adds a new entry to short-term memory."""
        self.memory.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "content": entry
        })

    def get_recent_entries(self, num_entries: int = 10) -> List[Dict[str, Any]]:
        """Retrieves the most recent entries."""
        return list(self.memory)[-num_entries:]

    def clear(self):
        """Clears all short-term memory entries."""
        self.memory.clear()

class LongTermMemory:
    """Stores high-confidence concepts, facts, and style (LTM)."""
    def __init__(self, storage_path: str = "./ltm_storage.json"):
        self.storage_path = storage_path
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """Loads long-term memory from a JSON file."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_memory(self):
        """Saves long-term memory to a JSON file."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def add_concept(self, concept_id: str, data: Dict[str, Any]):
        """Adds or updates a concept in long-term memory."""
        self.memory[concept_id] = {
            "last_updated": datetime.datetime.now().isoformat(),
            "data": data
        }
        self._save_memory()

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a concept from long-term memory."""
        return self.memory.get(concept_id, None)

    def delete_concept(self, concept_id: str):
        """Deletes a concept from long-term memory."""
        if concept_id in self.memory:
            del self.memory[concept_id]
            self._save_memory()

    def search_concepts(self, keyword: str) -> List[Dict[str, Any]]:
        """Searches for concepts containing a keyword (simple search)."""
        results = []
        for concept_id, entry in self.memory.items():
            if keyword.lower() in json.dumps(entry["data"]).lower():
                results.append({"concept_id": concept_id, **entry["data"]})
        return results

class SelfHealingLayer:
    """Corrects itself over time from web truth and internal consistency checks."""
    def __init__(self, ltm: LongTermMemory):
        self.ltm = ltm

    def verify_and_correct(self, concept_id: str, new_data: Dict[str, Any]) -> bool:
        """Verifies new data against existing LTM and corrects if necessary.
        Returns True if correction occurred, False otherwise.
        """
        existing_concept = self.ltm.get_concept(concept_id)
        if existing_concept:
            # Simple correction logic: if new data contradicts existing, prefer new data if it's more recent
            # In a real system, this would involve more sophisticated truth comparison (e.g., multiple sources)
            existing_timestamp = existing_concept.get("last_updated", "")
            new_timestamp = new_data.get("timestamp", datetime.datetime.now().isoformat())

            if new_timestamp > existing_timestamp: # Simplified logic
                print(f"Correcting concept {concept_id} with newer data.")
                self.ltm.add_concept(concept_id, new_data)
                return True
            else:
                print(f"Existing data for {concept_id} is more recent or same. No correction needed.")
                return False
        else:
            print(f"Concept {concept_id} not found in LTM. Adding new data.")
            self.ltm.add_concept(concept_id, new_data)
            return True

class UserPersonalization:
    """Remembers user style, tone, and bias (optional)."""
    def __init__(self, storage_path: str = "./user_profiles.json"):
        self.storage_path = storage_path
        self.profiles = self._load_profiles()

    def _load_profiles(self) -> Dict[str, Any]:
        """Loads user profiles from a JSON file."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_profiles(self):
        """Saves user profiles to a JSON file."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)

    def update_profile(self, user_id: str, preferences: Dict[str, Any]):
        """Updates a user's personalization profile."""
        if user_id not in self.profiles:
            self.profiles[user_id] = {}
        self.profiles[user_id].update(preferences)
        self._save_profiles()

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a user's personalization profile."""
        return self.profiles.get(user_id, None)

# Example Usage
if __name__ == "__main__":
    # Short-Term Memory Example
    stm = ShortTermMemory(max_size=5)
    stm.add_entry({"user": "Alice", "message": "Hello AI!"})
    stm.add_entry({"user": "AI", "message": "Hello Alice! How can I help you?"})
    stm.add_entry({"user": "Alice", "message": "Tell me about AI."})
    print("\nShort-Term Memory:", stm.get_recent_entries())

    # Long-Term Memory Example
    ltm = LongTermMemory("sathik_ltm.json")
    ltm.add_concept("AI_definition", {"text": "Artificial intelligence is the simulation of human intelligence processes by machines.", "source": "Wikipedia"})
    print("\nLong-Term Memory (AI_definition):", ltm.get_concept("AI_definition"))

    # Self-Healing Layer Example
    shl = SelfHealingLayer(ltm)
    new_ai_data = {"text": "AI is a broad field of computer science that gives computers the ability to perform human-like tasks.", "source": "IBM", "timestamp": "2025-01-01T00:00:00Z"}
    shl.verify_and_correct("AI_definition", new_ai_data)
    print("\nLong-Term Memory (AI_definition after self-healing):", ltm.get_concept("AI_definition"))

    # User Personalization Example
    up = UserPersonalization("sathik_user_profiles.json")
    up.update_profile("Alice", {"tone": "friendly", "preferred_topic": "technology"})
    print("\nUser Profile (Alice):", up.get_profile("Alice"))


