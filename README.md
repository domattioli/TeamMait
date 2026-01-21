# TeamMait

**AI-Powered Therapy Transcript Review Assistant**

TeamMait is a peer-support assistant designed to help clinicians review and analyze therapy session transcripts. It uses RAG (Retrieval-Augmented Generation) to provide evidence-based insights grounded in the actual transcript content.

## Features

- **Transcript Review**: Load and display therapy session transcripts with line numbers
- **AI-Powered Discussion**: Ask questions about therapist techniques, session dynamics, and clinical observations
- **Evidence-Based Responses**: Responses cite specific transcript lines using `[Line X]` format
- **PE Fidelity Framework**: Analysis grounded in Prolonged Exposure therapy fidelity criteria
- **Calibrated Language**: Uses appropriate epistemic markers ("appears", "may indicate")

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

```bash
# Clone or download the repository
cd teammait

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "sk-your-api-key-here"

# Optional: For multi-user deployment
[credentials]
users = [
    { username = "user1", password = "pass1" },
    { username = "user2", password = "pass2" }
]
```

Or use demo mode with `demo`/`demo` credentials and provide your API key at runtime.

### Running

```bash
streamlit run Home.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
teammait/
├── Home.py                 # Main application (chat interface)
├── requirements.txt        # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── session_manager.py  # Session persistence
│   └── input_parser.py     # Message handling
├── doc/
│   └── RAG/
│       ├── 116_P8_conversation.json    # Sample transcript
│       └── supporting_documents/
│           └── 116_P8_metadata.json    # Client/therapist metadata
├── archive/
│   └── experiment_setup/   # Original research study files
│       ├── README.md
│       ├── MODULE_SUMMARY.md
│       └── original_*.py
└── logs/                   # Application logs
```

## Usage

1. **Login**: Use your credentials or demo mode
2. **View Transcript**: Expand the sidebar to see the full session transcript
3. **Ask Questions**: Type questions in the chat input about:
   - Therapist techniques and interventions
   - Session dynamics and flow
   - Clinical observations
   - Fidelity to PE protocol
4. **Request Analysis**: Ask for specific evaluations, e.g.:
   - "Analyze the therapist's SUDS monitoring"
   - "What techniques did the therapist use for processing?"
   - "Were there missed opportunities for engagement?"

## Adding New Transcripts

1. Create a JSON file in `doc/RAG/` with the structure:
```json
{
  "full_conversation": [
    "Therapist: First utterance...",
    "Client: Response...",
    ...
  ]
}
```

2. Optionally add metadata in `doc/RAG/supporting_documents/`:
```json
{
  "client_profile": { ... },
  "therapist_profile": { ... },
  "trauma_info": { ... }
}
```

3. Delete `./rag_store/` to rebuild the vector database on next run

## Research Background

This project originated from a research study at Penn State investigating how AI can enhance clinical supervision and training in therapy settings. The original experiment included:

- Multiple review modules (open chat + guided observations)
- Structured feedback items with different presentation styles
- Session timing and analytics
- Comprehensive data export

The archived experiment setup is available in `archive/experiment_setup/` for reference.

**Original Research Team**:
- Principal Investigator: Dominik Mattioli, Ph.D.
- Institution: Penn State College of Information Sciences and Technology

## License


## Contributing

