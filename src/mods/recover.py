# src/mods/recover.py

import json, os

from ..providers.base import BaseResponder

class ConversationRecoveryManager:
    def __init__(self, persistence_file, log_file_path):
        self.persistence_file = persistence_file
        self.log_file_path = log_file_path

    def find_last_incomplete_aichat_state(self):
        if not self.persistence_file.exists():
            return None, None
        try:
            with open(self.persistence_file, "r", encoding="utf-8") as f:
                snapshots = json.load(f)
            # Scan in reverse for the last AIChat node with incomplete_conversation True
            for snap in reversed(snapshots):
                node_id = snap.get("node", {}).get("node_id")
                state = snap.get("state", {})
                status = snap.get("status")
                if (
                    node_id == "AIChat"
                    and state.get("incomplete_conversation", False)
                    and (status == "running" or status is None)
                ):
                    return node_id, state
        except Exception as e:
            print(f"Error reading persistence file: {e}")
        return None, None

    def print_conversation_history_from_log(self, convo_id):
        if not convo_id:
            print("[No conversation ID found for recovery log.]")
            return
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                log = json.load(f)
            convo = log.get(convo_id)
            if not convo or "history" not in convo:
                print("[No conversation history found for this convo ID.]")
                return
            for msg in convo["history"]:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    print(f"\nMe:\n> {msg['content']}")
                elif msg["role"] == "assistant":
                    print(f"\nAI:\n> {msg['content']}")
        except Exception as e:
            print(f"[Error reading conversation log: {e}]")

    def print_conversation_preview(self, convo_id):
        """Print the title, model, and summary for a conversation preview."""
        if not convo_id:
            print("[No conversation ID found for preview.]")
            return
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                log = json.load(f)
            convo = log.get(convo_id)
            if not convo:
                print("[No conversation found for this convo ID.]")
                return
            title = convo.get("title") or "[No Title]"
            summary = convo.get("summary") or "[No Summary]"
            model_tag = convo.get("model_tag")
            model_name = None
            if model_tag:
                # Use the static method
                model_details = BaseResponder.get_model_details(model_tag)
                model_name = model_details["name"] if model_details and "name" in model_details else model_tag
            # Strip <thinking>...</thinking> tags from summary
            import re
            summary_clean = re.sub(r"<thinking>.*?</thinking>", "", summary, flags=re.DOTALL | re.IGNORECASE).strip()
            print('\n--- Conversation Preview ---\n')
            print(f"Title: {title}")
            if model_name:
                print(f"Model: {model_name}")
            print('\nSummary: """')
            print(f'{summary_clean}\n"""\n')
        except Exception as e:
            print(f"[Error reading conversation log for preview: {e}]")

    def purge_empty_conversations(self):
        """
        Reads the conversation log, removes empty conversations, and rewrites the log.
        An "empty" conversation has no history, no entries, and zero total tokens.
        """
        if not os.path.exists(self.log_file_path):
            return # No log file to purge

        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    print("\n[RECOVERY] Conversation log is empty or corrupt. Skipping purge.")
                    return # Empty or corrupt log

            if not isinstance(log_data, dict): # Ensure it's a dictionary
                print("\n[RECOVERY] Conversation log is not in the expected format. Skipping purge.")
                return

            original_convo_count = len(log_data)
            convos_to_keep = {}

            for convo_id, convo_details in log_data.items():
                # Define what constitutes an "empty" conversation more precisely
                is_history_empty = not convo_details.get("history") # Empty list or None
                is_entries_empty = not convo_details.get("entries") # Empty list or None
                # Check totals, especially total_tokens
                totals = convo_details.get("totals", {})
                is_tokens_zero = totals.get("total_tokens", 0) == 0
                
                # A stricter check: if history is empty, it's likely an abandoned chat start
                if not is_history_empty: # Keep if there's any history
                    convos_to_keep[convo_id] = convo_details
                # Optional: Add more conditions if needed, e.g., if you want to keep it even if history is empty but title was somehow set
                # For now, focusing on no actual interaction (no history)
            
            purged_count = original_convo_count - len(convos_to_keep)

            if purged_count > 0:
                with open(self.log_file_path, 'w', encoding='utf-8') as f:
                    json.dump(convos_to_keep, f, indent=2, ensure_ascii=False)
                print(f"\n[RECOVERY] Removed {purged_count} empty conversation(s) from the log.")
            else:
                # print("\n[RECOVERY] No empty conversations found to remove.") # Optional: for verbosity
                pass

        except IOError as e:
            print(f"[Purge Error] Could not read/write conversation log: {e}")
        except Exception as e:
            print(f"[Purge Error] An unexpected error occurred during purge: {e}")
