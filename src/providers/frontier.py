# src/providers/frontier.py

from __future__ import annotations
import asyncio, json, os

from openai import AuthenticationError
from httpx import HTTPStatusError
from google import genai

# Project-internal import
from .base import BaseResponder, DEFAULT_SYSTEM_PROMPT

class OpenAIResponder(BaseResponder):
    def __init__(self):
        super().__init__(
            provider_name="OpenAI", 
            api_key_env_var="OPENAI_API_KEY", 
            base_url_env_var="OPENAI_BASE_URL" # Pass the env var name
        )

    async def call_provider_api(self, message_history_for_api, model_to_use, current_reasoning_effort: str | None = None): # Added, though not used by OpenAI
        model_to_use = model_to_use or self.default_model_tag() # Use default if None
        try:
            # Assuming self.client is AsyncOpenAI initialized in BaseResponder
            response = await self.client.chat.completions.create( # Changed from self.client.responses.create
                model=model_to_use,
                messages=message_history_for_api # Changed from input=
            )
            raw_response = response.model_dump_json(indent=2)
            json_response_data = json.loads(raw_response)

            usage_data = {}
            if hasattr(response, "usage") and response.usage:
                usage_data['input_tokens'] = getattr(response.usage, "prompt_tokens", 0)
                usage_data['output_tokens'] = getattr(response.usage, "completion_tokens", 0)
                usage_data['total_tokens'] = getattr(response.usage, "total_tokens", 0)
            elif json_response_data.get("usage"):
                usage_data['input_tokens'] = json_response_data["usage"].get("prompt_tokens", json_response_data["usage"].get("input_tokens", 0))
                usage_data['output_tokens'] = json_response_data["usage"].get("completion_tokens", json_response_data["usage"].get("output_tokens", 0))
                usage_data['total_tokens'] = json_response_data["usage"].get("total_tokens", 0)
            
            json_response_for_base = json_response_data.copy()
            json_response_for_base['usage'] = usage_data

            return response, raw_response, json_response_for_base
        except (AuthenticationError, HTTPStatusError) as e: # More specific error catching
            print(f"[API Error] OpenAI ({model_to_use}): {e}")
            if isinstance(e, AuthenticationError) or (isinstance(e, HTTPStatusError) and e.response.status_code in [401, 403]):
                print("Please ensure your OPENAI_API_KEY and OPENAI_BASE_URL (if used) are correctly set and valid.")
            return None, None, {}
        except Exception as e: # Catch-all for other unexpected errors
            print(f"[Unexpected API Error] OpenAI ({model_to_use}): {e}")
            return None, None, {}

    def extract_output_text(self, response, json_response):
        # Standard OpenAI chat completion response structure
        choices = json_response.get("choices", [])
        if choices and choices[0].get("message") and choices[0]["message"].get("content"):
            return choices[0]["message"]["content"].strip()
        
        # Fallback for older client.responses.create if it was intended (though less likely now)
        if hasattr(response, "output_text") and response.output_text: # Check if it's the custom 'responses' obj
            return response.output_text
        
        print("[Warning] OpenAI: Could not extract output text from response. Structure might be unexpected.")
        return ""

    def default_model_tag(self):
        return "gpt-4.1-nano-2025-04-14"

class GoogleResponder(BaseResponder):
    def __init__(self):
        # Initialize the google.genai client here and pass it to BaseResponder
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("[Critical Error] GEMINI_API_KEY not found in environment. GoogleResponder cannot be initialized.")
            # Handle this case, perhaps by making self.client None and checking in call_provider_api
            # For now, we assume it should be set for the app to function with Google models.
            g_client = None 
        else:
            g_client = genai.Client(api_key=gemini_api_key)
        
        super().__init__(
            provider_name="Google", 
            api_key_env_var="GEMINI_API_KEY", # Still pass for reference/validation if needed
            client_instance=g_client # Pass the initialized google.genai client
        )

    async def call_provider_api(self, message_history_for_api, model_to_use, current_reasoning_effort: str | None = None):
        model_to_use = model_to_use or self.default_model_tag() 

        if not self.client: 
            print("[API Error] Google (Gemini): Client not initialized (GEMINI_API_KEY missing?).")
            return None, None, {}
            
        try:
            # Check if custom COT should be applied (thinking_mode ON, and not a native_reasoning model for Google)
            # For Google, we assume no models have "native_reasoning": True yet for this custom COT purpose.
            # This logic mirrors what's in AIChat for OpenAI/xAI custom COT.
            
            # We need access to ctx.state.thinking_mode here.
            # This is a limitation of the current responder structure.
            # For now, we'll assume if reasoning_effort is None (which it would be for Google models
            # unless we explicitly add native_reasoning support for them), and if we *could* know
            # thinking_mode is on, we'd inject COT.

            # Simplified approach: Gemini's generate_content is often used for single-turn or simple multi-turn.
            # For robust chat with system prompts, model.start_chat() is preferred.
            # To make custom COT work here, we'd ideally prepend COT instructions to the *last user message* content
            # if thinking_mode is on. This is a hacky way to force instructions.

            # Let's refine the message history formatting for Gemini
            processed_history_for_gemini = []
            # The `message_history_for_api` already contains the necessary prompts if thinking_mode is ON
            # (THINK_ON_SYSTEM_PROMPT, COT_INSTRUCTIONS_CONTENT) because AIChat adds them.
            # We just need to format them correctly for Gemini.
            
            for msg in message_history_for_api:
                role = msg.get("role")
                content = msg.get("content", "")
                
                # Gemini uses 'user' and 'model' roles. 'assistant' maps to 'model'.
                # System messages from OpenAI/xAI format need to be handled.
                # If it's our special COT instruction, we can try to prepend it to the next user message,
                # or treat it as a user message itself for simplicity in this stateless call.
                
                if role == "user":
                    processed_history_for_gemini.append({'role': 'user', 'parts': [{'text': content}]})
                elif role == "assistant":
                    processed_history_for_gemini.append({'role': 'model', 'parts': [{'text': content}]})
                elif role == "system":
                    # For generate_content, system prompts are not directly part of the 'contents' list in the same way.
                    # We can try to prepend system instructions to the *next* user message,
                    # or include them as a 'user' turn if they are instructions *for* the model.
                    # This is where model.start_chat(system_instruction=...) is better.
                    # For now, let's try to include system prompts as if they are part of the user's directive,
                    # especially our COT instructions.
                    if content == DEFAULT_SYSTEM_PROMPT: # General system prompt, maybe less critical for direct content
                        pass # Often implicitly understood or set via generation_config
                    else:
                        # For our THINK_ON, COT_INSTRUCTIONS, NORMAL_MODE prompts, treat them as user instructions for now.
                        processed_history_for_gemini.append({'role': 'user', 'parts': [{'text': content}]})


            if not processed_history_for_gemini:
                print("[API Error] Google (Gemini): No content to send after formatting.")
                return None, None, {}

            # The google.genai SDK's generate_content is synchronous
            def sync_gemini_call():
                return self.client.models.generate_content(
                    model=model_to_use, 
                    contents=processed_history_for_gemini 
                )

            response = await asyncio.to_thread(sync_gemini_call)
            
            raw_response = response.model_dump_json(indent=2) 
            json_response_data = json.loads(raw_response)

            usage_metadata = json_response_data.get("usage_metadata", {})
            usage_data = {
                "input_tokens": usage_metadata.get("prompt_token_count", 0),
                "output_tokens": usage_metadata.get("candidates_token_count", 0),
                "total_tokens": usage_metadata.get("total_token_count", 0),
            }
            json_response_for_base = json_response_data.copy()
            json_response_for_base["usage"] = usage_data

            return response, raw_response, json_response_for_base
        except Exception as e:
            print(f"[API Error] Google (Gemini - {model_to_use}): {e}")
            return None, None, {}

    def extract_output_text(self, response, json_response):
        # google.generativeai.types.GenerateContentResponse
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        
        # Fallback from json_response if direct attribute access fails
        candidates = json_response.get("candidates", [])
        if candidates:
            first_candidate = candidates[0]
            if "content" in first_candidate and "parts" in first_candidate["content"]:
                parts = first_candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
        
        print("[Warning] Google (Gemini): Could not extract output text from response.")
        return ""

    def default_model_tag(self):
        return "gemini-2.0-flash"

class xAIResponder(BaseResponder):
    def __init__(self):
        super().__init__(
            provider_name="xAI", 
            api_key_env_var="XAI_API_KEY", 
            base_url_env_var="XAI_BASE_URL"
        )

    async def call_provider_api(self, message_history_for_api, model_to_use, current_reasoning_effort: str | None = None): # Added current_reasoning_effort
        model_to_use = model_to_use or self.default_model_tag() # Use default if None
        try:
            # Use the static method from BaseResponder
            model_details = BaseResponder.get_model_details(model_to_use)
            api_call_params = {
                "model": model_to_use,
                "messages": message_history_for_api
            }

            if model_details and model_details.get("native_reasoning") and current_reasoning_effort:
                # Only add reasoning_effort if the model supports it AND an effort is set
                api_call_params["reasoning_effort"] = current_reasoning_effort
            elif model_details and model_details.get("native_reasoning"):
                 # Default to "low" if native reasoning model but no specific effort from state (e.g. initial state)
                api_call_params["reasoning_effort"] = "low"


            response = await self.client.chat.completions.create(**api_call_params)
            
            raw_response_for_inspection = response.model_dump_json(indent=2)
            
            usage_data = {}
            if hasattr(response, "usage") and response.usage:
                usage_data['input_tokens'] = getattr(response.usage, "prompt_tokens", 0)
                
                completion_tokens_for_answer = getattr(response.usage, "completion_tokens", 0)
                reasoning_tokens_native = 0
                # Check if reasoning was actually performed (e.g. if reasoning_effort was not None or 'off')
                # and if the model supports native reasoning.
                if model_details and model_details.get("native_reasoning") and \
                   hasattr(response.usage, "completion_tokens_details") and \
                   response.usage.completion_tokens_details:
                    reasoning_tokens_native = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0)
                
                usage_data['output_tokens'] = completion_tokens_for_answer + reasoning_tokens_native
                usage_data['total_tokens'] = getattr(response.usage, "total_tokens", 0)
            
            json_response = { "usage": usage_data }
            return response, response.model_dump_json(indent=2), json_response
        except (AuthenticationError, HTTPStatusError, Exception) as e:
            print(f"[API Error] xAI: {e}")
            if isinstance(e, AuthenticationError) or (isinstance(e, HTTPStatusError) and e.response.status_code in [401, 403]):
                print("Please ensure your XAI_API_KEY and XAI_BASE_URL are correctly set in the .env file and are valid.")
            return None, None, {}

    def extract_output_text(self, response, json_response):
        # This method now primarily extracts the *final answer* content.
        # Reasoning content will be handled separately in AIChat if native_reasoning is used.
        choices = getattr(response, "choices", [])
        if choices and hasattr(choices[0], "message") and hasattr(choices[0].message, "content"):
            return getattr(choices[0].message, "content", "")
        print("[Warning] xAI: Could not extract final answer content from response.")
        return ""

    def default_model_tag(self):
        return "grok-3-mini-beta"
