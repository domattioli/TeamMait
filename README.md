# TeamMait
Trustworthy, Explainable, and Adaptive Monitoring Machine for AI Teams

# TeamMait Three-Page Implementation

## 📁 File Structure

```
your_app/
├── Home.py                          # Landing page (login, instructions, consent)
├── pages/
│   ├── 1_💬_Open_Chat.py           # Free-form conversation (your current app)
│   └── 2_📋_Guided_Review.py       # Structured flowchart interaction
├── utils/
│   ├── __init__.py                 # Package marker
│   ├── rag_setup.py                # Shared RAG/ChromaDB logic
│   ├── llm_clients.py              # Shared OpenAI client functions
│   └── flowchart_logic.py          # Flowchart state management
├── doc/
│   ├── RAG/
│   │   ├── 116_P8_conversation.json
│   │   └── supporting_documents/
│   └── interaction_prompts/
│       └── interaction_prompts.json
└── requirements.txt
```

## 🚀 How It Works

### Page 1: Home.py (Landing Page)
- **Login dialog** captures username, email, and consent
- **Instructions** explain both interaction modes
- **Navigation buttons** direct users to either:
  - Open Chat (free-form Q&A)
  - Guided Review (structured flowchart)

### Page 2: Open Chat
- Exact replica of your current `teammaitGPT.py`
- Free-form conversation with TeamMait
- RAG-enhanced responses from transcript
- Evidence display on request
- Export functionality

### Page 3: Guided Review (Flowchart)
Implements the flowchart with these stages:

1. **Intro** → Display welcome message
2. **Prompt** → Show random question from bank (Adherence/Procedural/Relational/Structural)
3. **Detect Response Type** → User selects: Accept/Correct/Clarify/Disregard
4. **Active Engagement** → If user elaborates, LLM responds
5. **Anything Else?** → Ask if user wants to discuss more
6. **Loop or Complete** → Either show next question or end session

## 🔄 Flowchart State Machine

The `flowchart_logic.py` manages transitions:

| Current Stage | User Input | Next Stage | Action |
|--------------|------------|------------|---------|
| `intro` | Any | `prompt` | Show first question |
| `prompt` | "Accept" (short) | `anything_else` | Acknowledge, move on |
| `prompt` | "Correct..." (elaborated) | `active_engagement` | Trigger LLM response |
| `prompt` | "Disregard" | `anything_else` | Skip question |
| `active_engagement` | Any | `anything_else` | Ask for more discussion |
| `anything_else` | "No" | `prompt` OR `complete` | Next question or end |
| `anything_else` | "Yes, about..." | `anything_else` | Open-ended LLM chat |

## 🎯 Key Features

### Shared Utilities
- **rag_setup.py**: Single source of truth for ChromaDB and document loading
- **llm_clients.py**: Unified OpenAI client with mode-specific prompts
- **flowchart_logic.py**: State machine for guided review

### Navigation
- Users can switch between pages using sidebar buttons
- `st.switch_page()` provides seamless transitions
- Session state persists across pages

### Data Export
- Both modes have independent export functionality
- JSON format includes metadata, messages, and state
- Google Sheets integration maintained

## 🛠️ Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install streamlit openai chromadb sentence-transformers gspread oauth2client
   ```

2. **Create folder structure:**
   ```bash
   mkdir -p pages utils doc/RAG/supporting_documents doc/interaction_prompts
   ```

3. **Move files:**
   - Place `Home.py` in root directory
   - Place page files in `pages/` folder
   - Place utility files in `utils/` folder

4. **Run the app:**
   ```bash
   streamlit run Home.py
   ```

## 📝 Customization Tips

### Adding More Questions
Edit `doc/interaction_prompts/interaction_prompts.json`:
```json
{
  "id": "new_question_01",
  "label": "Domain Name",
  "assertion": "Your assertion here",
  "explanation": "Why this matters",
  "invitation": "Question to ask user"
}
```

### Adjusting Flowchart Logic
Modify `utils/flowchart_logic.py`:
- `detect_response_type()`: Change keyword detection
- `handle_flowchart_transition()`: Adjust state transitions
- `get_next_question()`: Change selection algorithm (currently random)

### Styling
Add custom CSS in each page file using `st.markdown()` with `unsafe_allow_html=True`

## 🐛 Troubleshooting

**Problem**: "Please log in first" appears on pages
- **Solution**: Ensure `user_info` is in session state before navigating

**Problem**: Import errors in page files
- **Solution**: Check that `utils/__init__.py` exists and path is added to sys.path

**Problem**: ChromaDB not persisting
- **Solution**: Verify `./rag_store` directory has write permissions

**Problem**: Questions repeat in guided review
- **Solution**: Check that `questions_asked` list is being updated in session state

## 📊 Session State Variables

### Global (All Pages)
- `user_info`: {username, email, consent_given, consent_timestamp}
- `username`: String (convenience copy)
- `email`: String (convenience copy)

### Open Chat Page
- `messages`: List of chat messages
- `errors`: List of error logs
- `stream_on`: Boolean for streaming
- `show_timestamps`: Boolean for timestamp display

### Guided Review Page
- `guided_messages`: List of chat messages (separate from open chat)
- `flowchart_state`: {
  - `stage`: Current flowchart stage
  - `questions_asked`: List of asked question IDs
  - `current_question`: Current question object
  - `current_response_type`: User's response type
  - `needs_followup`: Boolean for LLM engagement
  - `all_domains_covered`: Boolean for completion
}

## 🎨 UI Enhancements

The guided review includes:
- **Progress bar** showing questions completed
- **Quick response buttons** for Accept/Correct/Clarify/Disregard
- **Text input** for elaborated responses
- **Completion message** when all questions reviewed
- **Restart functionality** to begin new session

## 🔐 Privacy & Consent

The home page captures:
- Username and email (required)
- Explicit consent checkbox (required)
- Timestamp of consent (automatic)

All exported data includes consent metadata.