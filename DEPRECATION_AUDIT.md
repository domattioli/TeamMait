# Deprecation & Code Cleanup Audit

**Date**: January 9, 2026
**Status**: Analysis Complete

---

## ✅ ACTIVELY USED (No Action Needed)

### `user_sessions/` Directory
- **Status**: ✅ ACTIVELY USED
- **Used By**: `pages/3_Module_2.py` (SessionManager)
- **Purpose**: Persistent storage for guided interaction (Module 2) sessions
- **Details**: Stores session metadata and conversation history per user
- **Action**: Keep

### `completion_status` Session State
- **Status**: ✅ ACTIVELY USED
- **Used By**: All pages (`Home.py`, `1_Module_1.py`, `2_Qualtrics_Survey.py`, `3_Module_2.py`)
- **Purpose**: Tracks which modules/pages have been visited
- **Used In Export**: `4_Finish.py` includes this in session snapshot
- **Action**: Keep

### Google Sheets Integration
- **Status**: ✅ ACTIVELY USED (for non-test users)
- **Used By**: `pages/1_Module_1.py`, `pages/4_Finish.py`
- **Purpose**: Data export to Google Sheet
- **Note**: Skipped for test users (new feature)
- **Action**: Keep

### All Utility Modules
- **Status**: ✅ ACTIVELY USED
- **Files**: `session_manager.py`, `navigation_validator.py`, `input_parser.py`, `analytics.py`, etc.
- **Action**: Keep

---

## ❌ DEPRECATED CODE (Recommend Cleanup)

### 1. **Timer Component** (DEPRECATED)
- **File**: `utils/timer_component_embedded.py`
- **Status**: ❌ NOT USED ANYWHERE
- **History**: Timer was removed from Module 2 in previous session
- **Last Reference**: None found in codebase
- **Action**: **DELETE** - Can be safely removed

### 2. **Guided Messages & Flowchart State** (DEPRECATED)
- **Files**: Exported in `pages/4_Finish.py` lines 79-126
- **Status**: ❌ NEVER POPULATED
- **Details**: 
  - Code attempts to export `st.session_state.get("guided_messages", [])`
  - Code attempts to export `st.session_state.get("flowchart_state", {})`
  - Neither of these are set anywhere in Module 2
  - This appears to be legacy code from an older flowchart-based implementation
- **Current State**: Always returns empty lists/dicts
- **Action**: **CLEAN UP** - Remove this export logic from `build_export()`

### 3. **Brevity Policy & Template Config** (DEPRECATED)
- **File**: `pages/1_Module_1.py` lines 128-134
- **Status**: ❌ COMMENTED OUT, NOT USED
- **Details**: 
  ```python
  # # ---- Brevity policy & templates (1–5) ----
  # BREVITY = {
  #     1: dict(name="Detailed Narrative", ...),
  #     2: dict(name="Structured Summary", ...),
  #     ...
  # }
  ```
- **Why**: System now uses fixed system prompt instead of configurable brevity levels
- **Action**: **DELETE** - Clean up commented code block (lines 128-134)

### 4. **Structure Prompt Function** (DEPRECATED)
- **File**: `pages/1_Module_1.py` lines 360-376
- **Status**: ❌ COMMENTED OUT, NOT USED
- **Details**: 
  ```python
  # def structure_prompt(level:int) -> str:
  #     cfg = BREVITY[level]
  #     if cfg["headline"]:
  #     ...
  ```
- **Why**: Replaced by unified `build_system_prompt()` function
- **Action**: **DELETE** - Clean up commented code block

### 5. **Old System Prompt Function Signature** (DEPRECATED)
- **File**: `pages/1_Module_1.py` line 383
- **Status**: ❌ COMMENTED OUT, NOT USED
- **Details**: 
  ```python
  # def build_system_prompt(empathy_value: int, brevity_level: int) -> str:
  ```
- **Why**: Function signature changed to `build_system_prompt() -> str` (no parameters)
- **Action**: **DELETE** - Clean up comment (line 383)

### 6. **Load Conversation Function** (DEPRECATED)
- **File**: `pages/1_Module_1.py` lines 343-357
- **Status**: ❌ COMMENTED OUT, NOT USED
- **Details**: 
  ```python
  # # ---------- Load JSON Conversation into Vector Store ----------
  # @st.cache_resource
  # def load_conversation_and_seed():
  #     with open("116_P8_conversation.json") as f:
  ```
- **Why**: Replaced by `load_rag_documents()` which handles all document loading
- **Action**: **DELETE** - Clean up commented code block

### 7. **Anthropic Client Function** (DEPRECATED)
- **File**: `pages/1_Module_1.py` lines 467-473
- **Status**: ❌ COMMENTED OUT, NOT USED
- **Details**: 
  ```python
  # def get_anthropic_client():
  #     if anthropic is None:
  #     st.error("anthropic package not installed. Run: pip install anthropic")
  ```
- **Why**: Currently using only OpenAI
- **Note**: Could be useful if Anthropic support is added later, but currently unused
- **Action**: **DELETE** - Clean up commented code (or keep if future support planned)

### 8. **Brevity Sidebar Settings** (DEPRECATED)
- **File**: `pages/1_Module_1.py` lines 146-147
- **Status**: ❌ COMMENTED OUT, NOT USED
- **Details**: 
  ```python
  # st.session_state['empathy'] = empathy
  # st.session_state['brevity'] = brevity
  ```
- **Why**: UI settings for old configurable prompt system
- **Action**: **DELETE** - Clean up (line 146-147)

### 9. **Clear Chat Button** (DEPRECATED)
- **File**: `pages/1_Module_1.py` line 139
- **Status**: ❌ COMMENTED OUT, NOT USED
- **Details**: 
  ```python
  # st.button("Clear chat", type="secondary", on_click=lambda: st.session_state.update(messages=[]))
  ```
- **Why**: Clear button functionality removed in favor of permanent chat history
- **Action**: **DELETE** - Clean up comment

---

## 📊 Cleanup Impact Summary

| Item | Type | Impact | Priority |
|------|------|--------|----------|
| `timer_component_embedded.py` | File | Safe delete | HIGH |
| `guided_messages`/`flowchart_state` export | Code | Remove 50+ lines | HIGH |
| BREVITY config block | Code | Remove 8 lines | MEDIUM |
| `structure_prompt()` | Code | Remove 18 lines | MEDIUM |
| Old prompt function signature | Code | Remove 1 line | LOW |
| `load_conversation_and_seed()` | Code | Remove 16 lines | MEDIUM |
| `get_anthropic_client()` | Code | Remove 7 lines | MEDIUM |
| Sidebar settings | Code | Remove 2 lines | LOW |
| Clear chat button | Code | Remove 1 line | LOW |

---

## 📋 Recommended Cleanup Order

### Phase 1 (High Priority - Remove Dead Code)
1. Delete `utils/timer_component_embedded.py` (unused file)
2. Remove `guided_messages`/`flowchart_state` export logic from `4_Finish.py`
3. Remove deprecated blocks from `pages/1_Module_1.py`:
   - Lines 128-134: BREVITY config
   - Lines 360-376: `structure_prompt()` function
   - Lines 343-357: `load_conversation_and_seed()` function
   - Line 383: Old function signature comment
   - Lines 467-473: `get_anthropic_client()` function
   - Lines 146-147: Sidebar settings
   - Line 139: Clear chat button

### Phase 2 (Clean Up - Remove Single-Line Comments)
1. Line 163 in `1_Module_1.py`: `# st.divider()` comment

---

## ⚠️ Notes

- **No Active Functionality Removed**: All cleanup items are either:
  - Commented-out code that's not executed
  - Standalone utility files not imported anywhere
  - Unused export logic that returns empty data

- **Session State Safety**: `completion_status` and module inclusion flags (`include_open_chat`, etc.) are used and should **NOT** be removed despite not being strictly necessary with current flow

- **User Sessions**: `user_sessions/` directory is essential for Module 2 functionality - keep as-is

- **Test Mode Validated**: New test mode code doesn't use any deprecated patterns

---

## Estimated Cleanup Time
- **High Priority**: ~15 minutes (1 file delete + ~50 lines of code removal)
- **Medium Priority**: ~10 minutes (another ~45 lines of code removal)
- **Low Priority**: ~5 minutes (final cleanup)
- **Total**: ~30 minutes for full cleanup

