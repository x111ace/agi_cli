# py_template/src/mods/__init__.py

import os
from pathlib import Path

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CODE_DIR = os.path.abspath(os.path.join(MODULES_DIR, '..'))
FULL_PROJECT_ROOT = os.path.abspath(os.path.join(SOURCE_CODE_DIR, '..'))

PROVIDERS_DIR = os.path.abspath(os.path.join(SOURCE_CODE_DIR, 'providers'))
DATA_DIR = os.path.abspath(os.path.join(MODULES_DIR, 'data'))
EXT_DIR = os.path.join(FULL_PROJECT_ROOT, 'ext')

ENV_PATH = os.path.join(FULL_PROJECT_ROOT, '.env')

# New path definitions
AGENTS_DIR = os.path.join(EXT_DIR, 'agtc')
LOGS_DIR = os.path.join(EXT_DIR, 'logs')
CHARACTERS_DIR = os.path.join(AGENTS_DIR, 'characters')
SYSTEM_PROMPT_FILE = os.path.join(AGENTS_DIR, 'sys-p.txt')
COT_PROMPT_FILE = os.path.join(AGENTS_DIR, 'cot-p.txt')
CLI_PROMPT_FILE = os.path.join(AGENTS_DIR, 'cli-p.txt')

# DATA_DIR is already defined above as: os.path.abspath(os.path.join(MODULES_DIR, 'data'))
PERSISTENCE_FILE = Path(os.path.join(DATA_DIR, 'graph_state.json'))
CONVERSATION_LOG_FILE = os.path.join(LOGS_DIR, 'conversation_log.json')

ind1_4 = "    "
ind2_4 = "        "
ind3_4 = "            "
ind4_4 = "                "
