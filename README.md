# TeamMait
**T**eam **E**xpert **A**I **M**entoring **A**ssistant for **I**ntervention **T**raining

A peer-support assistant designed to help expert clinicians review and analyze PE (Prolonged Exposure) therapy session transcripts through AI-assisted supervision.

## Overview

TeamMait provides two interaction modes for clinical supervision:
1. **Open Chat**: Free-form conversation about therapy transcripts
2. **Guided Interaction**: Structured observations and discussion
3. **Survey**: Post-interaction feedback collection
4. **Finish**: Session completion and data export

## Features

### Core Functionality
- **AI-Powered Analysis**: Uses OpenAI GPT-4o-mini for transcript analysis
- **RAG (Retrieval-Augmented Generation)**: ChromaDB vector database for contextual responses
- **DOCX Support**: Automatic processing of Word documents with chunking for large files
- **Multi-format Document Support**: JSON, TXT, and DOCX files in supporting documents
- **Line-Referenced Citations**: Precise transcript references with line numbers
- **Intelligent Response Classification**: LLM-powered user intent detection
- **Inactivity Detection**: 5-minute prompts to maintain engagement
- **Completion Tracking**: Progress monitoring across all interaction phases

### User Interface
- **Professional Styling**: Clean, clinical interface with proper typography
- **Responsive Design**: Optimized for various screen sizes
- **Progress Tracking**: Visual progress bars and completion status
- **Timestamp Display**: Smaller, unobtrusive timestamps throughout conversations
- **Evidence Display**: On-demand citation and supporting evidence
- **Quick Response Buttons**: Thumbs up/down feedback for guided observations

## File Structure

```
TeamMait/
├── Home.py                          # Landing page with login, instructions, consent
├── pages/
│   ├── 1_Open_Chat.py              # Free-form conversation mode
│   ├── 2_Survey.py                 # Post-interaction survey
│   ├── 3_Guided_Interaction.py     # Structured flowchart interaction
│   └── 4_Finish.py                 # Session completion and export
├── utils/
│   ├── __init__.py                 
│   └── streamlit_compat.py         # Compatibility utilities
├── doc/
│   ├── users.json                  # User authentication data
│   ├── RAG/
│   │   ├── 116_P8_conversation.json    # Reference therapy transcript
│   │   └── supporting_documents/       # Additional training materials
│   │       ├── *.docx              # Word documents (auto-processed)
│   │       ├── *.json              # JSON metadata files
│   │       └── *.txt               # Text documents
│   └── interaction_prompts/
│       └── interaction_prompts.json    # Guided interaction question bank
├── rag_store/                      # ChromaDB persistent storage
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version specification
└── README.md                       # This file
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
- Users authenticate via username/password (configured in `doc/users.json`)
- Default test credentials: username=`test`, password=`test`

### Interaction Modes

#### Phase 1: Open Chat
- Natural conversation with TeamMait about the therapy transcript
- Ask questions about therapist performance, techniques, or observations
- Request evidence and citations from the transcript
- No time limits or structured requirements

#### Phase 2: Guided Interaction
- TeamMait presents prepared observations about the transcript
- Users can accept, disagree, request clarification, or skip observations
- Natural discussion around each observation
- Progress tracking through 4 prepared questions
- Automatic completion detection and reminders

#### Survey & Completion
- Brief post-interaction survey about user experience
- Session completion tracking
- Data export functionality

## Technical Implementation

### AI System Prompt
TeamMait operates with strict guidelines:
- Focus exclusively on observable therapist skills in transcripts
- Anchor all claims to transcript evidence with line references
- Maintain academic neutrality without emotional language
- Provide evidence-based analysis with specific citations
- Limit scope to transcript analysis (no broader therapy advice)

### Intelligent Response Handling
- **LLM Classification**: Determines user intent (accept/correct/clarify/disregard)
- **Context-Aware Responses**: Adapts to conversation flow and user engagement
- **Completion Detection**: Recognizes when users have finished their evaluation
- **Inactivity Monitoring**: Suggests progression after 5 minutes of inactivity

### Document Processing
- **DOCX Extraction**: Automatic text extraction from Word documents
- **Chunking Strategy**: Large documents split into 500-word chunks for optimal retrieval
- **Multi-format Support**: Handles JSON, TXT, and DOCX files seamlessly
- **Error Handling**: Graceful handling of corrupted or unreadable files

### Data Management
- **Session Persistence**: State maintained across page navigation
- **Completion Tracking**: Progress saved across interaction phases
- **Export Functionality**: JSON format with timestamps and metadata

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

### Adding Questions
Edit `doc/interaction_prompts/interaction_prompts.json`:
```json
{
  "id": "unique_question_id",
  "assertion": "Observation about the transcript",
  "explanation": "Supporting details and reasoning",
  "invitation": "Question to prompt user discussion"
}
```

### Document Management
- Add supporting documents to `doc/RAG/supporting_documents/`
- Supports DOCX, JSON, and TXT formats
- Files are automatically indexed in the vector database
- Large documents are automatically chunked for optimal retrieval

### User Management
Edit `doc/users.json`:
```json
{
  "users": [
    {"username": "user1", "password": "password1"},
    {"username": "user2", "password": "password2"}
  ]
}
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