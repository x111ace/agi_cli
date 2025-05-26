# src/mods/graph.py

from __future__ import annotations
import asyncio, uuid, json, os

from typing import Any, List, Tuple, Union, Coroutine
from dataclasses import dataclass, field

from pydantic_graph import Graph, BaseNode, End, GraphRunContext
from pydantic_graph.persistence.file import FileStatePersistence

# Project-internal imports
from ..providers.base import (
    BaseResponder, 
    MODEL_CATALOG, 
    THINK_ON_SYSTEM_PROMPT, 
    THINK_ON_SYSTEM_PROMPT_CONTENT, 
    THINK_OFF_SYSTEM_PROMPT, 
    THINK_OFF_SYSTEM_PROMPT_CONTENT, 
    NATIVE_REASONING_GUIDANCE_PROMPT, 
    NATIVE_REASONING_GUIDANCE_PROMPT_CONTENT, 
    COT_INSTRUCTION_META_TAG, 
    DEFAULT_SYSTEM_PROMPT, 
    load_cot_instructions, 
)
from ..providers.frontier import OpenAIResponder, xAIResponder, GoogleResponder
from .recover import ConversationRecoveryManager
from .logging import ResponseLog
from .__init__ import (
    CONVERSATION_LOG_FILE,
    PERSISTENCE_FILE, 
    COT_PROMPT_FILE,
    CLI_PROMPT_FILE,
)
from .utils import validate_api_key 

recovery_manager = ConversationRecoveryManager(PERSISTENCE_FILE, CONVERSATION_LOG_FILE)


@dataclass
class ConversationState:
    """State to be shared across graph nodes."""
    message_history: list[Any] = field(default_factory=list)
    current_convo_id: str | None = None
    selected_model_tag: str | None = None
    selected_provider_name: str | None = None
    incomplete_conversation: bool = False
    last_context_threshold: int = 0
    turn_count: int = 0
    title_needs_update: bool = False
    thinking_mode: bool = False 
    current_reasoning_effort: str | None = None 

# Forward declarations
class ToggleThinking(BaseNode[ConversationState]): ... 
class ModelSelect(BaseNode[ConversationState]): ...
class ChatHistory(BaseNode[ConversationState]): ...
class MainMenu(BaseNode[ConversationState]): ...
class AIChat(BaseNode[ConversationState]): ...

@dataclass
class ToggleThinking(BaseNode[ConversationState]):
    """Node for managing thinking mode and reasoning effort."""
    effort_options = ["low", "medium", "high"]

    async def run(self, ctx: GraphRunContext[ConversationState]) -> 'AIChat':
        # Determine if the current model is native reasoning
        is_native_reasoning_model = False
        if ctx.state.selected_model_tag:
            model_details = BaseResponder.get_model_details(ctx.state.selected_model_tag)
            if model_details:
                is_native_reasoning_model = model_details.get("native_reasoning", False)

        if is_native_reasoning_model:
            print("\n--- Select Reasoning Effort ---\n"
                  f"\nCurrent reasoning effort: {ctx.state.current_reasoning_effort.capitalize() if ctx.state.current_reasoning_effort else 'Not set'}"
                  "\nSelect an effort level:\n\n"
                  "[0] Cancel")
            
            for i, effort in enumerate(self.effort_options):
                print(f"[{i+1}] {effort.capitalize()}")

            choice_str = await asyncio.to_thread(input, "\n> ")
            
            try:
                choice_idx = int(choice_str)
                if choice_idx == 0:
                    print("\nSelection cancelled. Current options remain unchanged.")
                elif 1 <= choice_idx <= len(self.effort_options):
                    selected_effort = self.effort_options[choice_idx - 1]
                    ctx.state.current_reasoning_effort = selected_effort
                    ctx.state.thinking_mode = True # Native reasoning display is on if an effort is set
                    print(f"\n[i] Reasoning Effort: '{selected_effort.capitalize()}'.")
                else:
                    print("\nInvalid selection. Effort remains unchanged.")
            except ValueError:
                print("\nInvalid input. Effort remains unchanged.")
        else: # For non-native models, toggle custom COT thinking mode
            ctx.state.thinking_mode = not ctx.state.thinking_mode
            if ctx.state.thinking_mode:
                print(f"\n[i] '/think' ACTIVATED.")
            else:
                print(f"\n[i] '/think' DEACTIVATED.")
        
        return AIChat()

@dataclass
class ModelSelect(BaseNode[ConversationState]):
    """Node for selecting the AI model."""
    async def run(self, ctx: GraphRunContext[ConversationState]) -> Union['MainMenu', 'AIChat', 'End']:

        print("\n--- Choose a Model ---\n"
              "\nAdd 'i' to the key (e.g., 'ai') to see its description."
              "\n\n[0] Exit")
        
        available_models: List[Tuple[str, str, str]] = [] # (display_letter, provider_name, model_tag)
        current_option_index = 0

        for provider_name, models in MODEL_CATALOG.items():
            if not models:
                continue
            print(f"\n{provider_name}:")
            for model_info in models:
                # Ensure current_option_index correctly maps to letters then numbers
                if current_option_index < 26: # a-z
                    display_letter = chr(ord('a') + current_option_index)
                else: # 27 onwards, use numbers starting from 27 (arbitrary, just to be unique)
                    display_letter = str(current_option_index + 1) 
                
                display_text = f"  [{display_letter}] {model_info['name']}"
                print(display_text)
                available_models.append((display_letter, provider_name, model_info['tag']))
                current_option_index += 1
        
        if not available_models:
            print("\nNo models available in MODEL_CATALOG. Please configure models.")
            return End(None)
        
        print("\nEnter a model's key to select it.")
        choice = await asyncio.to_thread(input, "\n> ")
        choice = choice.strip().lower() # Standardize to lowercase

        if choice == "0":
            print("\nReturning to previous screen...")
            if ctx.state.message_history and ctx.state.incomplete_conversation:
                return AIChat()
            else:
                return MainMenu()

        # --- Info Display Logic ---
        if choice.endswith('i') and len(choice) > 1:
            model_key_for_info = choice[:-1]
            selected_opt_for_info = next((opt for opt in available_models if opt[0] == model_key_for_info), None)
            
            if selected_opt_for_info:
                _, _, tag_for_info = selected_opt_for_info
                model_details_for_info = BaseResponder.get_model_details(tag_for_info)
                if model_details_for_info and model_details_for_info.get("description"):
                    model_description_header = f"--- Model Description ({model_details_for_info['name']}) ---"
                    model_description_footer = "-"*len(model_description_header)
                    print(f"\n{model_description_header}\n"
                          f"\n{model_details_for_info['description']}\n"
                          f"\n{model_description_footer}")
                else:
                    print(f"\nNo description available for model with key '{model_key_for_info}'.")
                input("\nPress Enter to continue...")
            else:
                print(f"\nInvalid model key '{model_key_for_info}' for description.")
            
            return ModelSelect()

        # --- Model Selection Logic (existing) ---
        selected_option = next((opt for opt in available_models if opt[0] == choice), None)

        if selected_option:
            _, provider, tag = selected_option
            ctx.state.selected_provider_name = provider
            ctx.state.selected_model_tag = tag
            # Use the static method
            model_details = BaseResponder.get_model_details(tag)
            print(f"\nModel selected: {model_details['name'] if model_details else tag} from {provider}")

            # Consolidated logic for reasoning effort and thinking_mode
            is_native = model_details and model_details.get("native_reasoning")
            if is_native:
                # Set default reasoning effort if not already set or if switching to native model
                ctx.state.current_reasoning_effort = ctx.state.current_reasoning_effort or "low"
                # thinking_mode is True unless effort is explicitly "off"
                ctx.state.thinking_mode = ctx.state.current_reasoning_effort != "off"
                print(f"[INFO] Native reasoning (effort: {ctx.state.current_reasoning_effort}). '/think' to adjust.")
            else:
                ctx.state.current_reasoning_effort = None
                ctx.state.thinking_mode = False

            # If we're starting a new chat (message_history is empty or only system prompt), go to chat
            if ctx.state.incomplete_conversation:
                return AIChat()
            else:
                return MainMenu()
        else:
            print("\nInvalid selection. Please try again.")
            return ModelSelect()

@dataclass
class ChatHistory(BaseNode[ConversationState]):
    """Node for displaying and selecting from chat history."""
    async def run(self, ctx: GraphRunContext[ConversationState]) -> Union['AIChat', 'MainMenu']:
        chat_log_path = CONVERSATION_LOG_FILE
        if not os.path.exists(chat_log_path):
            print("\nNo chat history found!")
            return MainMenu()
        
        try:
            with open(chat_log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            print("\nChat history file is corrupted or empty!")
            try:
                with open(chat_log_path, "w", encoding="utf-8") as f_corrupt:
                    json.dump({}, f_corrupt)
                print("[INFO] Corrupted chat history file has been reset.")
            except Exception as e_corrupt:
                print(f"[ERROR] Could not reset corrupted chat history file: {e_corrupt}")
            return MainMenu()

        if not log_data:
            print("\nNo chat history found!")
            return MainMenu()
        
        print("\n--- Chat History ---\n"
              "\nEnter a chat index to resume. Enter 'xx' to delete ALL chat history.\n"
              "\nAdd 'x' to the end of a chat's index number to delete it (e.g., 1x).")
        print("\n[0] Back to Main Menu")
        
        convo_keys = list(log_data.keys())
        for idx, convo_id in enumerate(convo_keys):
            title = log_data[convo_id].get("title") or "[No Title]"
            summary = log_data[convo_id].get("summary") or "[No Summary]"
            model_tag = log_data[convo_id].get("model_tag")
            model_name = None
            if model_tag:
                model_details = BaseResponder.get_model_details(model_tag)
                model_name = model_details["name"] if model_details and "name" in model_details else model_tag
            
            display_text = f"[{idx+1}] {title}"
            if model_name:
                thinking_active = log_data[convo_id].get("thinking_mode_active_on_exit", False)
                reasoning_effort = log_data[convo_id].get("current_reasoning_effort", "low") 
                if model_details and model_details.get("native_reasoning") and thinking_active:
                    display_text += f"  [{model_name} (Effort: {reasoning_effort})]"
                else:
                    display_text += f"  [Model: {model_name}]"
            print(f"\n{display_text}")
            print(f"- {summary}")
            # print(f"    {summary[:100]}{'...' if len(summary) > 100 else ''}")

        selection = await asyncio.to_thread(input, "\n> ")
        selection = selection.strip().lower()

        if selection == "0":
            return MainMenu()

        # --- Delete ALL History Logic ---
        if selection == "xx":
            confirm_delete_all = await asyncio.to_thread(
                input, 
                "\nDelete ALL chat history? This cannot be undone.\n\n(1/0) > "
            )
            if confirm_delete_all.strip().lower() in ['1', 'y']:
                try:
                    # Write an empty JSON object to clear the file
                    with open(chat_log_path, "w", encoding="utf-8") as f:
                        json.dump({}, f)
                    print("\n[SUCCESS] All chat history has been deleted.")
                    # Also clear any in-memory state related to last conversation if necessary
                    ctx.state.current_convo_id = None
                    ctx.state.message_history = []
                    ctx.state.incomplete_conversation = False
                except IOError as e:
                    print(f"\n[ERROR] Could not clear chat history file: {e}")
                except Exception as e_all:
                    print(f"\nAn unexpected error occurred while deleting all history: {e_all}")
                return MainMenu() # Go to main menu after deleting all
            else:
                print("\nDeletion of all history cancelled.")
                return ChatHistory() # Refresh history view

        # --- Single Deletion Logic ---
        if selection.endswith('x') and len(selection) > 1: 
            try:
                num_part = selection[:-1]
                sel_idx_to_delete = int(num_part) - 1
                
                if 0 <= sel_idx_to_delete < len(convo_keys):
                    convo_id_to_delete = convo_keys[sel_idx_to_delete]
                    title_to_delete = log_data[convo_id_to_delete].get("title") or convo_id_to_delete
                    
                    confirm_delete = await asyncio.to_thread(input, f"\nAre you sure you want to delete chat '{title_to_delete}'?\n\n(1/0) > ")
                    if confirm_delete.strip().lower() in ['1', 'y']:
                        del log_data[convo_id_to_delete]
                        try:
                            with open(chat_log_path, "w", encoding="utf-8") as f:
                                json.dump(log_data, f, indent=2, ensure_ascii=False)
                            print(f"\nChat '{title_to_delete}' deleted successfully.")
                        except IOError as e:
                            print(f"\n[ERROR] Could not write changes to log file: {e}")
                    else:
                        print("\nDeletion cancelled.")
                    return ChatHistory() # Refresh the history view
                else:
                    print("\nInvalid number for deletion.")
                    return ChatHistory()
            except ValueError:
                print("\nInvalid format for deletion (e.g., 1x).")
                return ChatHistory()
            except Exception as e:
                print(f"\nAn error occurred during deletion: {e}")
                return ChatHistory()

        # --- Resume Logic (existing) ---
        try:
            sel_idx = int(selection) - 1
            if 0 <= sel_idx < len(convo_keys):
                selected_convo_id = convo_keys[sel_idx]
                ctx.state.current_convo_id = selected_convo_id
                ctx.state.message_history = log_data[selected_convo_id].get("history", [])
                ctx.state.incomplete_conversation = True
                ctx.state.turn_count = len(ctx.state.message_history) // 2
                ctx.state.selected_model_tag = log_data[selected_convo_id].get("model_tag")
                ctx.state.selected_provider_name = log_data[selected_convo_id].get("provider_name")
                # Also load thinking_mode state if it's in the log, otherwise default to False
                ctx.state.thinking_mode = log_data[selected_convo_id].get("thinking_mode_active_on_exit", False)
                ctx.state.current_reasoning_effort = log_data[selected_convo_id].get("current_reasoning_effort", "low")

                print(f"\n[i] Resuming chat: {log_data[selected_convo_id].get('title', selected_convo_id)}")
                recovery_manager.print_conversation_history_from_log(selected_convo_id)
                return AIChat()
            else:
                print("\nInvalid selection.")
                return ChatHistory()
        except ValueError: # Handles non-integer inputs that don't match 'x' pattern
            print("\nInvalid input.")
            return ChatHistory()
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            return ChatHistory()

@dataclass
class MainMenu(BaseNode[ConversationState]):
    """Main menu node that presents options to the user."""
    async def run(self, ctx: GraphRunContext[ConversationState]) -> Union['AIChat', 'ModelSelect', 'ChatHistory', 'End']:
        # Fix: Don't try to access model details when no model is selected
        if ctx.state.selected_model_tag:
            # Use the static method
            selected_model_details = BaseResponder.get_model_details(ctx.state.selected_model_tag)
            selected_model_display_name = selected_model_details["name"] if selected_model_details else ctx.state.selected_model_tag

        print("\n--- Main Menu ---\n"
              "\n[0] Exit"
              "\n[1] New Chat"
              "\n[2] Chat History")

        choice = await asyncio.to_thread(input, "\n> ")
        choice = choice.strip()
        
        if choice == "1":
            ctx.state.message_history = [] 
            ctx.state.current_convo_id = str(uuid.uuid4())
            ctx.state.incomplete_conversation = True
            ctx.state.last_context_threshold = 0
            ctx.state.turn_count = 0
            # If no model is set, prompt for model selection
            if not ctx.state.selected_model_tag or not ctx.state.selected_provider_name:
                return ModelSelect()
            # Fix: Validate API key before entering chat
            if not validate_api_key(ctx.state.selected_provider_name):
                print(f"\n[ERROR] No valid API key found for provider '{ctx.state.selected_provider_name}'. Please set it in your .env file.")
                return MainMenu()
            return AIChat()
        elif choice == "2":
            return ChatHistory()
        elif choice == "0":
            print("\nExiting application...")
            return End(None)
        else:
            print("\nInvalid choice, please try again.")
            return MainMenu()

@dataclass
class AIChat(BaseNode[ConversationState]):
    """Node to handle the conversation loop."""
    async def run(self, ctx: GraphRunContext[ConversationState]) -> Union[MainMenu, ModelSelect, ToggleThinking, End]: 
        ctx.state.incomplete_conversation = True
        if not ctx.state.current_convo_id:
            ctx.state.current_convo_id = str(uuid.uuid4())

        if not ctx.state.selected_model_tag or not ctx.state.selected_provider_name:
            print("No model selected. Redirecting to model selection...")
            return ModelSelect() 

        # Use the static method
        model_details = BaseResponder.get_model_details(ctx.state.selected_model_tag)
        model_display_name = model_details["name"] if model_details else ctx.state.selected_model_tag
        is_native_reasoning_model = model_details.get("native_reasoning", False) if model_details else False

        # --- Default thinking_mode for native reasoning models on AIChat entry ---
        if is_native_reasoning_model: # This applies to xAI's Grok 3 Mini
            if ctx.state.current_reasoning_effort != "off":
                if not ctx.state.thinking_mode: 
                    ctx.state.thinking_mode = True
                    print(f"[INFO] Native reasoning display automatically ENABLED on chat entry (effort: {ctx.state.current_reasoning_effort}). '/think' to adjust.")
            else: 
                if ctx.state.thinking_mode: 
                    ctx.state.thinking_mode = False
                    print(f"[INFO] Native reasoning display turned OFF as effort is 'off'.")
        # For non-native models (including Gemini for now, and OpenAI), 
        # thinking_mode is explicitly toggled by user via /think for custom COT.
        # It will default to False unless resumed from a state where it was True.

        if not validate_api_key(ctx.state.selected_provider_name):
            print(f"\n[ERROR] No valid API key found for provider '{ctx.state.selected_provider_name}'. Please select a different model.")
            return ModelSelect()

        help_text = ""
        help_text += (
            "\nAvailable commands:\n"
            "- '/exit' to exit the application\n"
            "- '/menu' to return to Main Menu\n"
            "- '/help' to show this help menu\n"
            "- '/model' to change the model\n")
        if is_native_reasoning_model == False:
            help_text += (
                "- '/think' to toggle on/off thinking")
        if is_native_reasoning_model == True:
            help_text += (
                "- '/think' to change reasoning effort\n"
                "Current model supports native reasoning; '/think' adjusts effort & display")

        print("\n--- Entering Chat Mode ---")
        if ctx.state.turn_count == 0:
            print(f"{help_text}")
        print(f"\nUsing {model_display_name} from {ctx.state.selected_provider_name}")

        ################################
        # --- INITIALIZE PROMPTING --- #
        ################################

        history = ctx.state.message_history
        convo_id = ctx.state.current_convo_id

        # --- System Prompt and Mode Management ---
        # Current history (potentially from log)
        history = ctx.state.message_history

        # Ensure DEFAULT_SYSTEM_PROMPT is at the start if not already
        if not history or history[0].get("role") != "system" or history[0].get("content") != DEFAULT_SYSTEM_PROMPT:
            temp_history_for_default_prompt = [msg for msg in history if not (msg.get("role") == "system" and msg.get("content") == DEFAULT_SYSTEM_PROMPT)]
            history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + temp_history_for_default_prompt
        
        # Add the CLI prompt right after the system prompt
        if os.path.exists(CLI_PROMPT_FILE):
            with open(CLI_PROMPT_FILE, "r", encoding="utf-8") as f:
                cli_prompt_content = f.read().strip()
            if not any(msg.get("role") == "system" and msg.get("content") == cli_prompt_content for msg in history):
                history.insert(1, {"role": "system", "content": cli_prompt_content})
        
        # --- Enhanced cleanup for CoT instructions and special prompts ---
        # Load both full and stripped CoT instructions for comparison
        COT_INSTRUCTIONS_CONTENT_FULL = load_cot_instructions(COT_PROMPT_FILE)
        COT_INSTRUCTIONS_CONTENT_STRIPPED = load_cot_instructions(COT_PROMPT_FILE, strip_from="IMPORTANT:")

        # Remove ALL messages that are:
        # - meta_tag == COT_INSTRUCTION_META_TAG
        # - OR content matches either full or stripped CoT instructions
        history = [
            msg for msg in history
            if not (
                msg.get("meta_tag", None) == COT_INSTRUCTION_META_TAG or
                msg.get("content") == COT_INSTRUCTIONS_CONTENT_FULL or
                msg.get("content") == COT_INSTRUCTIONS_CONTENT_STRIPPED
            )
        ]

        # Remove all special system prompts
        history = [msg for msg in history if msg.get("content") not in [
            THINK_ON_SYSTEM_PROMPT_CONTENT,     
            THINK_OFF_SYSTEM_PROMPT_CONTENT, 
            NATIVE_REASONING_GUIDANCE_PROMPT_CONTENT 
        ]]

        if ctx.state.thinking_mode:
            if not is_native_reasoning_model: 
                # Standard CoT for non-native models
                if not any(msg.get("meta_tag") == COT_INSTRUCTION_META_TAG for msg in history):
                    COT_INSTRUCTIONS_CONTENT = COT_INSTRUCTIONS_CONTENT_FULL
                    history.append({
                        "role": "system",
                        "content": COT_INSTRUCTIONS_CONTENT,
                        "meta_tag": COT_INSTRUCTION_META_TAG
                    })
                # Then add think on prompt
                if not any(m.get("content") == THINK_ON_SYSTEM_PROMPT_CONTENT for m in history if m.get("role") == "system"):
                    history.append(THINK_ON_SYSTEM_PROMPT)
            else: 
                # Native reasoning model guidance
                if not any(msg.get("meta_tag") == COT_INSTRUCTION_META_TAG for msg in history):
                    COT_INSTRUCTIONS_CONTENT = COT_INSTRUCTIONS_CONTENT_STRIPPED
                    history.append({
                        "role": "system",
                        "content": COT_INSTRUCTIONS_CONTENT,
                        "meta_tag": COT_INSTRUCTION_META_TAG
                    })
                # Then add native reasoning guidance
                if not any(m.get("content") == NATIVE_REASONING_GUIDANCE_PROMPT_CONTENT for m in history if m.get("role") == "system"):
                    history.append(NATIVE_REASONING_GUIDANCE_PROMPT)
        else: 
            # Thinking mode is OFF
            pass

        ctx.state.message_history = history # Update the state
        
        ################################
        # --- INITIALIZE PROVIDERS --- #
        ################################

        openai_responder = OpenAIResponder()
        xai_responder = xAIResponder()
        gemini_responder = GoogleResponder()

        # Initialize response_logger BEFORE using it
        response_logger = ResponseLog()
        
        # Now we can safely use response_logger
        response_logger.update_conversation_meta(
            convo_id=ctx.state.current_convo_id,
            model_tag=ctx.state.selected_model_tag,
            provider_name=ctx.state.selected_provider_name,
            thinking_mode_active_on_exit=ctx.state.thinking_mode,
            current_reasoning_effort=ctx.state.current_reasoning_effort if is_native_reasoning_model else None
        )

        # Log initial system messages
        response_logger.save_to_raw_log(
            convo_id=ctx.state.current_convo_id,
            history=history,
            model_tag=ctx.state.selected_model_tag,
            provider_name=ctx.state.selected_provider_name,
            thinking_mode_active_on_exit=ctx.state.thinking_mode,
            current_reasoning_effort=ctx.state.current_reasoning_effort if is_native_reasoning_model else None
        )

        # Display initial context bar based on the latest total_tokens if resuming/switching model
        # This is crucial for showing correct context usage when entering chat from history or after /model
        initial_latest_total_tokens = response_logger.get_latest_total_tokens(convo_id)
        if initial_latest_total_tokens > 0:  # Only print if there's history
            if ctx.state.thinking_mode:
                print(f"\n[i] '/think' ACTIVATED")
            if ctx.state.selected_provider_name == "OpenAI":
                openai_responder.print_context_window_bar(initial_latest_total_tokens, ctx.state.selected_model_tag)
            elif ctx.state.selected_provider_name == "xAI":
                xai_responder.print_context_window_bar(initial_latest_total_tokens, ctx.state.selected_model_tag)
            elif ctx.state.selected_provider_name == "Google": 
                gemini_responder.print_context_window_bar(initial_latest_total_tokens, ctx.state.selected_model_tag)

        ######################
        # --- USER INPUT --- #
        ######################

        while True:
            user_input = await asyncio.to_thread(input, "\nMe:\n> ")
            if user_input.lower() == '/exit' or user_input.lower() == '/menu':
                if user_input.lower() == '/exit':
                    print("\nExiting application...")
                    return End(None)  # Return End node for clean exit
                else:  # menu
                    print("\nReturning to Main Menu.")
                    ctx.state.incomplete_conversation = False # Mark as complete for recovery logic
                    
                    # Save current thinking_mode state and reasoning effort to the log before exiting
                    current_model_details_on_exit = BaseResponder.get_model_details(ctx.state.selected_model_tag)
                    is_native_model_on_exit = current_model_details_on_exit.get("native_reasoning", False) if current_model_details_on_exit else False

                    response_logger.update_conversation_meta(
                        convo_id=ctx.state.current_convo_id,
                        thinking_mode_active_on_exit=ctx.state.thinking_mode,
                        current_reasoning_effort=ctx.state.current_reasoning_effort if is_native_model_on_exit else None
                    )

                    # Clean up special mode prompts from history if exiting
                    history = [msg for msg in history if msg.get("meta_tag") != COT_INSTRUCTION_META_TAG]
                    history = [msg for msg in history if msg["content"] not in [
                        THINK_ON_SYSTEM_PROMPT_CONTENT,
                        THINK_OFF_SYSTEM_PROMPT_CONTENT,
                        NATIVE_REASONING_GUIDANCE_PROMPT_CONTENT
                    ]]
                    # Ensure DEFAULT_SYSTEM_PROMPT remains if it was there
                    if not any(msg["content"] == DEFAULT_SYSTEM_PROMPT and msg["role"] == "system" for msg in history):
                        if ctx.state.message_history and ctx.state.message_history[0]["content"] == DEFAULT_SYSTEM_PROMPT:
                            history.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
                    elif not history: # if history became empty, re-add default
                        history.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

                    ctx.state.message_history = history
                    # Revoke the model selection when exiting chat
                    ctx.state.selected_model_tag = None
                    ctx.state.selected_provider_name = None
                    return MainMenu()
            if user_input.lower() == '/help':
                print(help_text)
                continue
            if user_input.lower() == '/model':
                return ModelSelect()
            if user_input.lower() == '/think':
                # Use the static method
                current_model_is_native_reasoning = BaseResponder.get_model_details(ctx.state.selected_model_tag).get("native_reasoning", False) if BaseResponder.get_model_details(ctx.state.selected_model_tag) else False

                if current_model_is_native_reasoning:
                    print("\n[i] Adjusting native reasoning effort...")
                    
                    return ToggleThinking() # Go to the new node
                else: # For non-native models, toggle custom COT thinking mode
                    ctx.state.thinking_mode = not ctx.state.thinking_mode
                    # System prompt cleanup and re-application logic
                    # First remove all special mode prompts
                    history = [msg for msg in history if not (
                        msg.get("meta_tag") == COT_INSTRUCTION_META_TAG or  
                        msg.get("content") in [
                            THINK_ON_SYSTEM_PROMPT_CONTENT,
                            THINK_OFF_SYSTEM_PROMPT_CONTENT,
                            NATIVE_REASONING_GUIDANCE_PROMPT_CONTENT
                        ]
                    )]

                    if ctx.state.thinking_mode:
                        print(f"\n[i] '/think' ACTIVATED")
                        # First add COT instructions
                        if not any(m.get("meta_tag") == COT_INSTRUCTION_META_TAG for m in history):
                            if not is_native_reasoning_model:
                                COT_INSTRUCTIONS_CONTENT = COT_INSTRUCTIONS_CONTENT_FULL
                            else:
                                COT_INSTRUCTIONS_CONTENT = COT_INSTRUCTIONS_CONTENT_STRIPPED
                            history.append({
                                "role": "system",
                                "content": COT_INSTRUCTIONS_CONTENT,
                                "meta_tag": COT_INSTRUCTION_META_TAG
                            })
                        # Then add think on prompt
                        if not any(m["content"] == THINK_ON_SYSTEM_PROMPT_CONTENT for m in history if m["role"] == "system"):
                            history.append(THINK_ON_SYSTEM_PROMPT)
                    else:
                        print(f"\n[i] '/think' DEACTIVATED")
                        # Remove all CoT instruction messages (the entire message, not just the meta_tag)
                        history = [msg for msg in history if msg.get("meta_tag", None) != COT_INSTRUCTION_META_TAG]
                        if not any(m.get("content") == THINK_OFF_SYSTEM_PROMPT_CONTENT for m in history if m.get("role") == "system"):
                            history.append(THINK_OFF_SYSTEM_PROMPT)
                    ctx.state.message_history = history
                    # Log system messages after thinking mode change
                    response_logger.save_to_raw_log(
                        convo_id=ctx.state.current_convo_id,
                        history=history,
                        model_tag=ctx.state.selected_model_tag,
                        provider_name=ctx.state.selected_provider_name,
                        thinking_mode_active_on_exit=ctx.state.thinking_mode,
                        current_reasoning_effort=ctx.state.current_reasoning_effort if is_native_reasoning_model else None
                    )
                
                continue

            ctx.state.turn_count += 1

            new_user_message = {"role": "user", "content": user_input}
            
            # --- Prepare history for API: Strip custom tags ---
            api_ready_history = []
            # Start with a copy of the current history
            temp_history_for_api = history + [new_user_message]

            for msg in temp_history_for_api:
                api_msg = {"role": msg["role"], "content": msg["content"]}
                api_ready_history.append(api_msg)
            
            response_tuple = None
            current_reasoning_effort_for_api = ctx.state.current_reasoning_effort if is_native_reasoning_model else None

            if ctx.state.selected_provider_name == "OpenAI":
                response_tuple = await openai_responder.get_response(
                    api_ready_history, convo_id, ctx.state.selected_model_tag # OpenAI doesn't use reasoning_effort
                )
            elif ctx.state.selected_provider_name == "xAI":
                response_tuple = await xai_responder.get_response(
                    api_ready_history, convo_id, ctx.state.selected_model_tag, current_reasoning_effort_for_api
                )
            elif ctx.state.selected_provider_name == "Google": 
                response_tuple = await gemini_responder.get_response(
                    api_ready_history, convo_id, ctx.state.selected_model_tag
                )
            else:
                print(f"Error: Provider '{ctx.state.selected_provider_name}' not supported yet.")
                continue

            if response_tuple is None or response_tuple[0] is None: # Check the SDK object for None
                print("Error: Could not get response from the model provider.")
                continue

            # Unpacking 8 items now
            sdk_completion_object, raw_response_string_unused, updated_hist, token_warning_tuple, out_tokens, in_tokens_this_turn, tot_tokens, cv_id = response_tuple
            token_warning_msg, new_threshold = token_warning_tuple

            # Use the static method
            current_model_details_for_display = BaseResponder.get_model_details(ctx.state.selected_model_tag) # Re-fetch for safety
            is_native_reasoning_model_for_display = current_model_details_for_display.get("native_reasoning", False) if current_model_details_for_display else False


            if model_details: # This is outer scope model_details, ensure it's what we want
                if token_warning_msg:
                    print(token_warning_msg)
                    ctx.state.last_context_threshold = new_threshold # Update state here

            ################################
            # --- DISPLAY THE RESPONSE --- # 
            ################################

            if updated_hist: # This updated_hist contains the final_answer_text as assistant message
                final_answer_text_from_history = updated_hist[-1]['content']

                # Save to raw history log after each turn
                response_logger.save_to_raw_log(
                    convo_id=ctx.state.current_convo_id,
                    history=updated_hist,  # Use updated_hist which includes the latest response
                    model_tag=ctx.state.selected_model_tag,
                    provider_name=ctx.state.selected_provider_name,
                    thinking_mode_active_on_exit=ctx.state.thinking_mode,
                    current_reasoning_effort=ctx.state.current_reasoning_effort if is_native_reasoning_model else None
                )

                print("\nAI:")
                # Check if native reasoning should be displayed
                if is_native_reasoning_model_for_display and ctx.state.thinking_mode and ctx.state.current_reasoning_effort != "off":
                    reasoning_text_native = None
                    # Use sdk_completion_object here
                    if hasattr(sdk_completion_object, "choices") and sdk_completion_object.choices and \
                       hasattr(sdk_completion_object.choices[0], "message") and \
                       hasattr(sdk_completion_object.choices[0].message, "reasoning_content"):
                        reasoning_text_native = sdk_completion_object.choices[0].message.reasoning_content
                    
                    if reasoning_text_native:
                        print(f"<reasoning>\n{reasoning_text_native}\n</reasoning>")
                        print(f"<response>\n{final_answer_text_from_history}\n</response>\n")

                elif ctx.state.thinking_mode: # Custom XML-based thinking mode (non-native models)
                    import re
                    thinking_text_xml = ""
                    answer_text_xml = final_answer_text_from_history # Start with the full response

                    # Fix: Update regex to handle potential malformed tags
                    thinking_match = re.search(r"(?:<|```)thinking(?:>|```)(.*?)(?:</|```)thinking(?:>|```)", answer_text_xml, re.DOTALL | re.IGNORECASE)
                    if thinking_match:
                        thinking_text_xml = thinking_match.group(1).strip()
                        answer_text_xml = re.sub(r"(?:<|```)thinking(?:>|```).*?(?:</|```)thinking(?:>|```)", "", answer_text_xml, flags=re.DOTALL | re.IGNORECASE).strip()
                    
                    if thinking_text_xml:
                        print(f"<thinking>\n{thinking_text_xml}\n</thinking>")
                    
                    response_match = re.search(r"(?:<|```)response(?:>|```)(.*?)(?:</|```)response(?:>|```)", answer_text_xml, re.DOTALL | re.IGNORECASE)
                    if response_match:
                        actual_answer_xml = response_match.group(1).strip()
                        print(f"<response>\n{actual_answer_xml}\n</response>\n")
                    elif "<response>" in answer_text_xml and not "</response>" in answer_text_xml:
                        actual_answer_xml = answer_text_xml.split("<response>", 1)[-1].strip()
                        print(f"<response>\n{actual_answer_xml}\n</response>\n") 
                    else:
                        print(f"{answer_text_xml.strip()}\n")
                else: # Standard non-thinking mode display
                    print(">", final_answer_text_from_history, "\n")

                # Fetch the LATEST context window usage for the bar (use tot_tokens from response_tuple)
                if ctx.state.thinking_mode:
                    if is_native_reasoning_model_for_display:
                        print(f"[INFO] Reasoning Effort: '{ctx.state.current_reasoning_effort}'")
                    else:
                        print(f"[INFO] '/think' ACTIVATED")
                if ctx.state.selected_provider_name == "OpenAI":
                    openai_responder.print_context_window_bar(tot_tokens, ctx.state.selected_model_tag)
                elif ctx.state.selected_provider_name == "xAI":
                    xai_responder.print_context_window_bar(tot_tokens, ctx.state.selected_model_tag)
                elif ctx.state.selected_provider_name == "Google": 
                    gemini_responder.print_context_window_bar(tot_tokens, ctx.state.selected_model_tag)

                if ctx.state.turn_count == 1:
                    current_selected_tag = ctx.state.selected_model_tag 
                    history_for_summary_title = updated_hist 
                    
                    responder_for_utility = None
                    if ctx.state.selected_provider_name == "OpenAI":
                        responder_for_utility = openai_responder
                    elif ctx.state.selected_provider_name == "xAI":
                        # For xAI, summary/title still use OpenAI utility model via BaseResponder's methods
                        responder_for_utility = xai_responder # or openai_responder
                    elif ctx.state.selected_provider_name == "Google": 
                        responder_for_utility = gemini_responder
            
                    if responder_for_utility:
                        asyncio.create_task(responder_for_utility.generate_initial_summary_then_title(convo_id, history_for_summary_title, current_selected_tag))

            history = updated_hist
            convo_id = cv_id
            ctx.state.message_history = history
            ctx.state.current_convo_id = convo_id

            if ctx.state.turn_count == 1 or (ctx.state.turn_count > 1 and (ctx.state.turn_count - 1) % 5 == 0):
                current_title = None
                try:
                    
                    # Use ResponseLog instance directly for path, no need for responder_for_log_check
                    log_file_to_check = response_logger.log_file
                    
                    if os.path.exists(log_file_to_check):
                        with open(log_file_to_check, 'r', encoding='utf-8') as f:
                            log_data_for_title = json.load(f) # Renamed to avoid conflict with outer log_data
                            if log_data_for_title.get(convo_id):
                                current_title = log_data_for_title[convo_id].get("title")
                except Exception as e:
                    print(f"[AIChat Summary] Error reading title from log: {e}") 
                    pass 
                
                if not current_title:
                    current_title = "Chat Conversation"

                if ctx.state.turn_count > 1:
                    responder_for_periodic_summary = None
                if ctx.state.selected_provider_name == "OpenAI":
                        responder_for_periodic_summary = openai_responder
                elif ctx.state.selected_provider_name == "xAI":
                    # Summary/title generation uses OpenAI utility model via BaseResponder methods
                    responder_for_periodic_summary = xai_responder # or openai_responder
                elif ctx.state.selected_provider_name == "Google": 
                    responder_for_periodic_summary = gemini_responder

                if responder_for_periodic_summary:
                    asyncio.create_task(
                        responder_for_periodic_summary.generate_and_save_summary(convo_id, history, current_title, ctx.state.selected_model_tag)
                    )

            # Log system messages after model change
            response_logger.save_to_raw_log(
                convo_id=ctx.state.current_convo_id,
                history=history,
                model_tag=ctx.state.selected_model_tag,
                provider_name=ctx.state.selected_provider_name,
                thinking_mode_active_on_exit=ctx.state.thinking_mode,
                current_reasoning_effort=ctx.state.current_reasoning_effort if is_native_reasoning_model else None
            )

conversation_graph = Graph(
    nodes=(ModelSelect, MainMenu, ChatHistory, AIChat, ToggleThinking), 
    state_type=ConversationState)

async def run_graph_nodes(run: GraphRunContext[ConversationState]):
    """Helper to iterate through graph nodes."""
    while True:
        node = await run.next()
        if isinstance(node, End):
            break

persistence = FileStatePersistence(PERSISTENCE_FILE)
persistence.set_graph_types(conversation_graph)

def patch_persistence_file_status(persistence_file_path):
    """Reset any 'success' statuses to 'created' in the persistence file."""
    if not os.path.exists(persistence_file_path):
        return
    try:
        with open(persistence_file_path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            changed = False
            for snap in data:
                if snap.get("status") == "success":
                    snap["status"] = "created"
                    changed = True
            if changed:
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
    except Exception as e:
        print(f"[Patch] Could not patch persistence file: {e}")