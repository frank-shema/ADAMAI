# ADAMAI

## Overview

ADAMAI is an AI-powered project designed to handle various cognitive tasks using modular components structured in different layers. The architecture is divided into **inner** and **outer** layers, each responsible for specific functionalities such as speech recognition, speech generation, and custom commands.

## Project Structure

```
ADAMAI/
│── Brain/
│   ├── inner/
│   │   ├── layer1/
│   │   │   ├── GATOR.py   # Handles primary cognitive processing
│   │   ├── layer2/
│   │   │   ├── RAG.py     # Responsible for retrieval-augmented generation
│   ├── Mods/
│   │   ├── COMMANDS/open/
│   │   │   ├── Youtube.htm  # Custom web-based command
│   │   ├── CUSTOMCOMMANDS/JSONS/
│   │   │   ├── my_custom_command.json # Stores user-defined commands
│   │   ├── Project/
│   │   │   ├── functions.py  # Core functions for project modules
│   │   ├── IMPORTS.py        # Manages imports for various modules
│   ├── outer/
│   │   ├── layer1/
│   │   │   ├── SpeechRecognition.py  # Handles audio input processing
│   │   ├── layer2/
│   │   │   ├── SpeechGeneration.py   # Generates AI responses
│   │   ├── __init__.py               # Initializes outer modules
│   ├── Cerebrum.py    # Main brain module
│   ├── utils.py       # Utility functions for AI processing
│   ├── brain.png      # Project logo/image
│── main.py            # Entry point for the AI system
│── project_log.txt    # General project logs
│── project_log(inner_layers).txt  # Logs specific to inner layers
│── project_log(outer_layers).txt  # Logs specific to outer layers
│── README.md          # Project documentation
│── requirements.txt   # Python dependencies
```

## Features

- **Speech Recognition:** Processes voice commands.
- **Speech Generation:** Converts text into spoken responses.
- **Custom Commands:** Allows users to define personalized commands.
- **Layered Architecture:** Modular design for flexibility and scalability.
- **Integration with External APIs:** Includes web-based commands like YouTube automation.

## Installation

### Prerequisites

- Python 3.12+
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

Run the main script to start the AI system:

```bash
python main.py
```

## Configuration

- Modify `Brain/utils.py` for custom utility functions.
- Store API keys and credentials in an `.env` file to keep them secure.
- Define additional commands in `CUSTOMCOMMANDS/JSONS/my_custom_command.json`.

## Contribution

Feel free to contribute! Follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
