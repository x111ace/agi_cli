# src/mods/logging.py

import json, os

from .__init__ import CONVERSATION_LOG_FILE, LOGS_DIR

class ResponseLog:
    def __init__(self):
        self.log_file = CONVERSATION_LOG_FILE
        self.raw_history_dir = os.path.join(LOGS_DIR, 'raw_history.json')
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True) # Ensure data dir exists

    def _load_log_data(self) -> dict:
        """Loads conversation log data from the JSON file."""
        if not os.path.exists(self.log_file):
            return {}
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[Error] Log file '{self.log_file}' is corrupted or not valid JSON. Returning empty log.")
            return {}
        except IOError as e:
            print(f"[Error] Could not read log file '{self.log_file}': {e}. Returning empty log.")
            return {}

    def _save_log_data(self, data: dict) -> None:
        """Saves conversation log data to the JSON file."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[Error] Could not write to log file '{self.log_file}': {e}")

    def _get_or_initialize_convo_entry(self, log_data: dict, convo_id: str, initial_metadata: dict | None = None) -> dict:
        """
        Retrieves a conversation entry from log_data or initializes a new one.
        Ensures existing entries have all necessary keys with defaults if missing.
        """
        if convo_id not in log_data:
            # Create a new entry
            log_data[convo_id] = {
                "title": None,
                "summary": None,
                "history": [],
                "entries": [],
                "totals": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": "0.00000000" # Consistent string format for cost
                },
                "model_tag": initial_metadata.get("model_tag") if initial_metadata else None,
                "provider_name": initial_metadata.get("provider_name") if initial_metadata else None,
                "thinking_mode_active_on_exit": False,
                "current_reasoning_effort": None
            }
        else:
            # Ensure existing entry has all keys, adding defaults if missing (for backward compatibility)
            entry = log_data[convo_id]
            entry.setdefault("title", None)
            entry.setdefault("summary", None)
            entry.setdefault("history", [])
            entry.setdefault("entries", [])
            entry.setdefault("totals", {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": "0.00000000"
            })
            entry.setdefault("model_tag", None)
            entry.setdefault("provider_name", None)
            entry.setdefault("thinking_mode_active_on_exit", False)
            entry.setdefault("current_reasoning_effort", None) 
            
            # Further ensure sub-keys in totals exist
            if not isinstance(entry["totals"], dict): # if totals was somehow not a dict
                entry["totals"] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": "0.00000000"
                }
            entry["totals"].setdefault("input_tokens", 0)
            entry["totals"].setdefault("output_tokens", 0)
            entry["totals"].setdefault("total_tokens", 0)
            entry["totals"].setdefault("cost", "0.00000000")


        return log_data[convo_id]

    def calculate_price(self, model_tag, input_tokens, output_tokens):
        # Import BaseResponder here, inside the method
        from ..providers.base import BaseResponder
        
        model_details = BaseResponder.get_model_details(model_tag)
        if model_details and "pricing_input" in model_details and "pricing_output" in model_details:
            # Ensure pricing values are not None before calculation
            pricing_input = model_details["pricing_input"]
            pricing_output = model_details["pricing_output"]
            if pricing_input is not None and pricing_output is not None:
                input_cost = (input_tokens / 1_000_000) * pricing_input
                output_cost = (output_tokens / 1_000_000) * pricing_output
                return input_cost + output_cost
            else:
                # print(f"Warning: Pricing (input or output) is None for model tag '{model_tag}'. Cost will be 0.")
                return 0
        # print(f"Warning: Pricing not found or incomplete for model tag '{model_tag}'. Cost will be 0.")
        return 0

    def save_to_conversation_log(self, convo_id, new_messages, metadata_current_turn):
        """Saves new messages and metadata for a conversation turn to the log."""
        log_data = self._load_log_data()
        # Pass metadata_current_turn for initialization if convo_id is new
        convo_entry = self._get_or_initialize_convo_entry(log_data, convo_id, initial_metadata=metadata_current_turn)

        convo_entry["history"].extend(new_messages)
        convo_entry["entries"].append(metadata_current_turn)

        # Update totals
        current_totals = convo_entry["totals"]
        
        # Ensure cost is float for calculation, then convert back to string for storage
        try:
            cost_float = float(current_totals.get("cost", "0.0"))
        except ValueError:
            print(f"Warning: Could not parse existing total cost '{current_totals.get('cost')}' for convo_id {convo_id}. Using 0.0.")
            cost_float = 0.0
        
        cost_float += float(metadata_current_turn.get("cost", "0.0")) # Add current turn's cost
        current_totals["cost"] = f"{cost_float:.8f}" # Store as formatted string

        current_totals["input_tokens"] = current_totals.get("input_tokens", 0) + int(metadata_current_turn.get("input_tokens", 0))
        current_totals["output_tokens"] = current_totals.get("output_tokens", 0) + int(metadata_current_turn.get("output_tokens", 0))
        current_totals["total_tokens"] = current_totals.get("total_tokens", 0) + int(metadata_current_turn.get("total_tokens", 0))
        
        # Ensure model_tag and provider_name are updated if this is the first entry or they were missing
        if convo_entry.get("model_tag") is None and metadata_current_turn.get("model_tag"):
            convo_entry["model_tag"] = metadata_current_turn.get("model_tag")
        if convo_entry.get("provider_name") is None and metadata_current_turn.get("provider_name"):
            convo_entry["provider_name"] = metadata_current_turn.get("provider_name")

        # log_data[convo_id] is already updated by reference via convo_entry
        self._save_log_data(log_data)

    def update_conversation_meta(self, convo_id: str, title: str | None = None, summary: str | None = None, model_tag: str | None = None, provider_name: str | None = None, thinking_mode_active_on_exit: bool | None = None, current_reasoning_effort: str | None = None):
        """Updates metadata fields for a given conversation ID in the log."""
        log_data = self._load_log_data()
        # For meta updates, we don't pass initial_metadata unless we intend to set model/provider if missing
        convo_entry = self._get_or_initialize_convo_entry(log_data, convo_id)

        if title is not None:
            convo_entry['title'] = title
        if summary is not None:
            convo_entry['summary'] = summary
        if model_tag is not None:
            convo_entry['model_tag'] = model_tag
        if provider_name is not None:
            convo_entry['provider_name'] = provider_name
        if thinking_mode_active_on_exit is not None:
            convo_entry['thinking_mode_active_on_exit'] = thinking_mode_active_on_exit
        
        # Explicitly set current_reasoning_effort, allowing it to be None
        convo_entry['current_reasoning_effort'] = current_reasoning_effort
        
        # log_data[convo_id] is already updated by reference
        self._save_log_data(log_data)

    def get_cumulative_input_tokens(self, convo_id: str) -> int:
        """Calculates the cumulative input tokens for a given conversation ID."""
        log_data = self._load_log_data()
        convo_entry = log_data.get(convo_id)
        if not convo_entry or not convo_entry.get("entries"):
            return 0
        
        cumulative_input = 0
        # Iterate through each turn's metadata
        for entry_meta in convo_entry.get("entries", []):
            # We sum up 'input_tokens' from each turn as recorded in the log.
            # This assumes 'input_tokens' per entry reflects the tokens sent TO the API for that turn.
            cumulative_input += int(entry_meta.get("input_tokens", 0))
        return cumulative_input
    
    def get_latest_total_tokens(self, convo_id: str) -> int:
        """Returns the total_tokens from the most recent entry in the entries list for the given convo_id, or 0 if none."""
        log_data = self._load_log_data()
        convo_entry = log_data.get(convo_id)
        if not convo_entry or not convo_entry.get("entries"):
            return 0
        last_entry = convo_entry["entries"][-1]
        return int(last_entry.get("total_tokens", 0))

    def save_to_raw_log(self, convo_id, history, model_tag, provider_name, thinking_mode_active_on_exit, current_reasoning_effort):
        """Saves the complete conversation history including system messages to raw_history.json."""
        try:
            # Load existing raw history or create new
            if os.path.exists(self.raw_history_dir):
                with open(self.raw_history_dir, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            else:
                raw_data = {}

            # Update or create entry for this conversation
            raw_data[convo_id] = {
                "history": history,  # Complete history including system messages
                "model_tag": model_tag,
                "provider_name": provider_name,
                "thinking_mode_active_on_exit": thinking_mode_active_on_exit,
                "current_reasoning_effort": current_reasoning_effort
            }

            # Save updated raw history
            with open(self.raw_history_dir, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Error] Could not save raw history: {e}")
