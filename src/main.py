# py_template/src/main.py

from pydantic_graph.persistence.file import FileStatePersistence
from dotenv import load_dotenv
import asyncio, json, os

from .mods.__init__ import (
    FULL_PROJECT_ROOT,
    SOURCE_CODE_DIR, 
    MODULES_DIR, 
    ENV_PATH,
    CONVERSATION_LOG_FILE, 
    PERSISTENCE_FILE, 
)
from .mods.utils import (
    print_file_path as FILEi, 
    print_project_tree as TR33,
    printR, 
)
from .mods.graph import (
    ConversationState, 
    MainMenu, AIChat, 
    conversation_graph, 
    run_graph_nodes, 
    patch_persistence_file_status
)
from .mods.recover import ConversationRecoveryManager

PY_PATH_MAIN = os.path.abspath(__file__)

# Load environment variables from .env file
if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
else:
    print(f"[WARNING] `.env` file not found at. API keys won't be loaded.\nSearched at: {ENV_PATH}")

# --- --- --- --- --- --- --- --- ---

persistence = FileStatePersistence(PERSISTENCE_FILE)
persistence.set_graph_types(conversation_graph)
recovery_manager = ConversationRecoveryManager(PERSISTENCE_FILE, CONVERSATION_LOG_FILE)

async def main_async():

    # --- PATCH: Fix persistence file statuses if needed ---
    patch_persistence_file_status(str(PERSISTENCE_FILE))

    # --- PURGE EMPTY CONVERSATIONS from log ---
    recovery_manager.purge_empty_conversations() # Call before recovery logic

    # --- Recovery logic for incomplete AIChat ---
    node_id, state = recovery_manager.find_last_incomplete_aichat_state()
    initial_node = None # Declare upfront
    initial_state = None # Declare upfront

    if node_id == "AIChat" and state:
        convo_id = state.get("current_convo_id")
        user_messages_exist = False
        if convo_id: # Only proceed if convo_id exists in state
            try:
                # Ensure log file exists before trying to open
                if os.path.exists(CONVERSATION_LOG_FILE):
                    with open(CONVERSATION_LOG_FILE, "r", encoding="utf-8") as f:
                        log_data = json.load(f)
                    convo = log_data.get(convo_id)
                    if convo and "history" in convo:
                        user_messages_exist = any(msg.get("role") == "user" for msg in convo["history"])
            except Exception as e: # Catch potential JSONDecodeError or other file issues
                print(f"[RECOVERY] Could not read or parse conversation log: {e}")
                user_messages_exist = False # Default to false if log is unreadable

        if not user_messages_exist:
            print("\n[RECOVERY] No user messages found in last conversation, starting a new session...")
            initial_node = MainMenu()
            initial_state = ConversationState()
        else: # user_messages_exist is True
            print("\n[RECOVERY] Previous session was interrupted.")
            recovery_manager.print_conversation_preview(convo_id)
            resume_choice = await asyncio.to_thread(input, "Do you want to continue your last convo?\n\n(1/0) > ")
            if resume_choice.strip().lower() in ["1", "y", "yes"]:
                print("\n[RECOVERY] Resuming your last session...")
                recovery_manager.print_conversation_history_from_log(convo_id)
                initial_node = AIChat()
                initial_state = ConversationState(**state) # Load state from persistence
            else:
                print("\n[RECOVERY] Starting a new session...")
                initial_node = MainMenu()
                initial_state = ConversationState() # Fresh state

    else: # No incomplete AIChat state found, or state was invalid
        initial_snapshot = await persistence.load_next()
        if initial_snapshot:
            # print("\n[RECOVERY] No previous session found.")
            if hasattr(initial_snapshot, "status") and initial_snapshot.status == "success":
                initial_snapshot.status = "created"
            initial_node = initial_snapshot.node
            initial_state = initial_snapshot.state
        else:
            # print("\n[RECOVERY] No previous session found.")"
            initial_node = MainMenu()
            initial_state = ConversationState()

    if PERSISTENCE_FILE.exists():
        try:
            os.remove(PERSISTENCE_FILE)
            # print("[RECOVERY] Cleared old graph state persistence file.") # Optional info
        except Exception as e:
            print(f"[RECOVERY ERROR] Could not clear old graph state: {e}")

    async with conversation_graph.iter(initial_node, state=initial_state, persistence=persistence) as run:
        await run_graph_nodes(run)

def main():
    FILEi(
        MOD=MODULES_DIR,
        DIR=SOURCE_CODE_DIR,
        ROOT=FULL_PROJECT_ROOT,
        # output=True
    )
    TR33(
        ROOT=FULL_PROJECT_ROOT
        # output=True
    )
    asyncio.run(main_async())

if __name__ == "__main__":      
    main()