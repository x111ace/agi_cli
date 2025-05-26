# AGI CLI

⚠️ This project is not intended for public use.  
It is a personal or internal tool, and no license has been granted for usage or distribution.

A command-line interface for interacting with multiple AI providers (OpenAI, xAI, Google) with features including: 
- Conversation tracking & last session recovery
- Token pricing & context tracking
- Custom thinking mode

## Features

- Multi-provider support (OpenAI, xAI, Google)
    - Native reasoning display for supported models
    - Custom thinking mode with XML-based reasoning

- Conversation history management
    - Context window tracking
    - Auto pricing calculations

- Auto updated conversation storage
    - Title & summary generation
    - Every {int} turns in the convo

## Setup

```bash
# Clone the repository:
git clone <repository-url>
cd agi_cli

# Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Developer dependencies
# pip install -r required-dev.txt

# Set up environment variables:
# Edit the `.env.exa` file with your API keys:

# Run the application:
venv\Scripts\activate
python run.py
```

### Commands

- `/exit` - Exit the application
- `/menu` - Return to Main Menu
- `/help` - Show help text
- `/model` - Change the AI model
- `/think` - Toggle thinking mode or adjust reasoning effort

### Model Selection

The application supports multiple models from different providers:

- OpenAI: 
    - GPT-4.1 [BASE | MINI | NANO]
    - GPT-4o [BASE | MINI]
- xAI: 
    - GROK 3 [BASE | MINI]
    - GROK 2
- Google: 
    - GEMINI 2.0 FLASH

### Thinking Mode

- For non-reasoning models, `/think` toggles the custom thinking mode.
- For native reasoning models, `/think` will adjust the reasoning effort. 
    - Default reasoning effort is set to "low".
    - Reasoning effort levels: low, medium, high

## Project Structure

```plaintext
agi_cli/
│
├── ext/
│   ├── agtc/
│   │   ├── characters/
│   │   │   └── create.txt
│   │   │
│   │   ├── cli-p.txt
│   │   ├── cot-p.txt
│   │   └── sys-p.txt
│   │    
│   └── logs/
│       ├── concersation_log.json
│       └── raw_history.json
│
├── src/
│   ├── mods/
│   │   ├── data/
│   │   │   └── graph_state.json
│   │   │
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── logging.py
│   │   ├── recover.py
│   │   └── utils.py
│   │   
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── frontier.py
│   │   
│   └── main.py
│
├── tests/
│   └── test_mod.py
│
├── .env.exa
├── .gitignore
├── README.md
├── required-dev.txt
├── requirements.txt
└── run.py
```