# TeamMait
**T**eam **E**xpert **A**I **M**entoring **A**ssistant for **I**ntervention **T**raining

A peer-support assistant designed to help expert clinicians review and analyze PE (Prolonged Exposure) therapy session transcripts through AI-assisted supervision.

## Overview

TeamMait provides four interaction modes for clinical supervision:
1. **Module 1 (Open Chat)**: Free-form conversation about a therapy transcript
2. **Module 2 (Guided Observations)**: Structured observations with evidence and discussion
3. **Survey**: External Qualtrics survey for feedback collection
4. **Finish**: Session completion and data export

## Features

### Core Functionality
- **AI-Powered Analysis**: Uses OpenAI GPT-4o-mini for transcript analysis and summaries
- **RAG (Retrieval-Augmented Generation)**: ChromaDB vector database for contextual responses
- **Preloading**: Background loading of embeddings and ChromaDB on Home page for faster navigation
- **Message Queue**: Sequential message processing prevents input loss during rapid interaction
- **Line-Referenced Citations**: Precise transcript references with line numbers
- **Session Management**: Persistent sessions with 2-hour timeout and 48-hour cleanup
- **Analytics Logging**: Event logging to `logs/session_analytics.jsonl`

### User Interface
- **Consent Integration**: Consent form on landing page with checkbox requirement
- **Professional Styling**: Clean, clinical interface with proper typography
- **Collapsible Evidence**: Expandable evidence boxes with line numbers
- **Progress Tracking**: Visual indicators for observation progress
- **Timestamp Display**: Unobtrusive timestamps throughout conversations

## File Structure

```
TeamMait/
├── Home.py                          # Landing page with login, consent, preloading
├── pages/
│   ├── 1_Module_1.py               # Open chat mode (uses P8 transcript)
│   ├── 2_Qualtrics_Survey.py       # External survey redirect
│   ├── 3_Module_2.py               # Guided observations (uses P10 transcript)
│   └── 4_End.py                    # Session completion and export
├── utils/
│   ├── __init__.py                 
│   ├── analytics.py                # Event logging
│   ├── api_handler.py              # OpenAI API with retry logic
│   ├── input_parser.py             # Navigation intent detection, message buffer
│   ├── module_preload.py           # Background ChromaDB/embedding preloader
│   ├── navigation_validator.py     # Phase transition validation
│   ├── session_manager.py          # Persistent session storage
│   └── streamlit_compat.py         # Compatibility utilities
├── doc/
│   ├── consent_form.md             # Consent form text
│   ├── RAG/
│   │   ├── 116_P8_conversation.json    # Module 1 transcript
│   │   ├── 281_P10_conversation.json   # Module 2 transcript
│   │   └── supporting_documents/       # Additional training materials
│   └── interaction_prompts/
│       └── interaction_prompts.json    # Module 2 observation bank
├── rag_store/                      # ChromaDB persistent storage
├── user_sessions/                  # Persistent session data
├── logs/                           # Analytics logs
├── requirements.txt                # Python dependencies
└── runtime.txt                     # Python version specification
```

## Installation & Setup

### Prerequisites
- Python 3.9+
- OpenAI API key

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or add to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-openai-api-key"
```

### Running the Application
```bash
streamlit run Home.py
```

## Usage

### Authentication
- Users authenticate via username/password (configured in `.streamlit/secrets.toml`)
- **Test Mode**: Use username `test mode` / password `test` to try the app with your own OpenAI API key

### Interaction Modes

#### Module 1: Open Chat
- Natural conversation with TeamMait about the therapy transcript (P8)
- Ask questions about therapist performance, techniques, or observations
- Request evidence and citations from the transcript
- No time limits or structured requirements

#### Module 2: Guided Observations
- TeamMait presents 3 structured observations about a different transcript (P10)
- Each observation includes one of three styles:
  - `evidence_only_reflection`: Evidence excerpts for user reflection
  - `evidence_based_evaluation`: Evaluation with supporting evidence
  - `actionable_training_prescription`: Specific recommendations with evidence
- Evidence includes line numbers referencing the transcript
- Users discuss, skip, or advance through observations using the **⏩ Next** button
- Review phase allows revisiting all observations and requesting summaries

**Observation JSON structure:**
```json
{
  "style": "evidence_based_evaluation",
  "title": "Brief title",
  "assertion": "Main observation",
  "evidence": [{"text": "Quote from transcript", "line": 42}],
  "justification": "Why this matters"
}
```

#### Survey & Completion
- External Qualtrics survey for feedback
- Session completion with data export
- Comprehensive JSON export with conversation history and metadata

## Technical Implementation

### AI System Prompt
TeamMait operates with strict guidelines:
- Focus exclusively on observable therapist skills in transcripts
- Anchor all claims to transcript evidence with line references
- Maintain academic neutrality without emotional language
- Provide evidence-based analysis with specific citations
- Limit scope to transcript analysis (no broader therapy advice)

### Response Handling
- **API Retry Handler**: Exponential backoff (2s, 4s, 8s) for transient failures
- **Message Queue**: Sequential processing prevents message loss during rapid input
- **Near-Duplicate Detection**: Warns users when submitting similar questions (>90% similarity)
- **Navigation Intent**: Detects phrases like "next" and redirects to the Next button

### Session Management
- **Persistent Storage**: Sessions saved to `user_sessions/` directory
- **Session Timeout**: 2-hour inactivity timeout
- **Session Cleanup**: Expired sessions removed after 48 hours
- **Metadata Tracking**: Phase, observation index, message counts

### Data Management
- **Session Persistence**: State maintained across page navigation
- **Analytics Logging**: Events logged to `logs/session_analytics.jsonl`
- **Export Functionality**: JSON format with timestamps, metadata, and conversation history
- **Google Sheets Integration**: Automatic export for non-test users

## Research & Privacy

### Study Purpose
This tool is designed for research into AI-assisted clinical supervision and training. Participants serve as expert evaluators of both therapist performance and AI system effectiveness.

### Data Collection
- All interactions are logged for research analysis
- Data is anonymized in research outputs
- Participation is voluntary with right to withdraw
- No personal client information is collected

### Consent Framework
- Comprehensive informed consent process
- Clear explanation of research purpose and data use
- Professional context appropriate for expert clinicians
- Detailed privacy and security information

## Customization

### Adding Observations
Edit `doc/interaction_prompts/interaction_prompts.json`:
```json
{
  "id": "unique_id",
  "style": "evidence_based_evaluation",
  "title": "Brief title",
  "assertion": "Main observation",
  "evidence": [{"text": "Transcript quote", "line": 42}],
  "justification": "Clinical reasoning"
}
```

### Transcript Management
- Module 1 uses `doc/RAG/116_P8_conversation.json`
- Module 2 uses `doc/RAG/281_P10_conversation.json`
- Supporting documents in `doc/RAG/supporting_documents/` are indexed in ChromaDB

### User Management
Add users to `.streamlit/secrets.toml`:
```toml
[users]
username1 = "password1"
username2 = "password2"
```

## Deployment

### Local Development
```bash
streamlit run Home.py
```

### Production Deployment
- Configure environment variables for OpenAI API
- Ensure proper file permissions for `rag_store/` directory
- Set up secure user authentication
- Configure HTTPS for production use

## Troubleshooting

### Common Issues

**ChromaDB Permission Error**
```bash
chmod -R 755 rag_store/
```

**OpenAI API Key Missing**
- Verify environment variable or secrets.toml configuration
- Check API key validity and billing status

**Document Processing Errors**
- Ensure python-docx is installed for DOCX support
- Check file permissions in supporting_documents folder
- Verify document formats are supported

**Session State Issues**
- Clear browser cache and cookies
- Restart Streamlit server
- Check for JavaScript console errors

### Debug Mode
Enable debug output by checking browser console for detailed error messages and state information.

## Contributing

This is a research tool developed for clinical supervision studies. For issues or improvements, please consult with the research team through institutional channels.

## License

This software is developed for research purposes. Please refer to your institutional guidelines for usage and distribution.