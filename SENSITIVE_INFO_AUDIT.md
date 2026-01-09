# Sensitive Information Audit & Secrets Management

## Executive Summary
This document identifies all sensitive information in the TeamMait codebase that should be stored only in `.streamlit/secrets.toml` (which is git-ignored) and NOT committed to the public repository.

**Good News**: The `.streamlit/secrets.toml` file is already properly ignored in `.gitignore`, so committed secrets are NOT currently exposed.

---

## Sensitive Information Found in Code

### CRITICAL: Hardcoded Default Credentials ⚠️
**Location**: `Home.py` lines 36
**Issue**: Default login credentials are visible in source code
```python
password = st.text_input("Password", type="password", value="secureAdminPass123!")
```
**Risk**: While this is a default value for the input field, it should be removed or moved to secrets.
**Action**: REMOVE hardcoded default password value from the code.

---

### 1. API Keys & Authentication Tokens

#### OpenAI API Key
- **File**: `pages/1_Module_1.py` (line 480), `pages/3_Module_2.py` (line 132)
- **Usage**: `st.secrets["OPENAI_API_KEY"]` and `get_secret_then_env("OPENAI_API_KEY")`
- **Status**: ✅ CORRECTLY stored in secrets
- **Format**: Starts with `sk-proj-`
- **Risk Level**: CRITICAL - allows full API access

#### Anthropic API Key
- **File**: `pages/1_Module_1.py` (line 470, commented out)
- **Usage**: `get_secret_then_env("ANTHROPIC_API_KEY")`
- **Status**: ✅ CORRECTLY stored in secrets (currently empty)
- **Risk Level**: CRITICAL if populated

---

### 2. Google Cloud Credentials

#### Google Service Account Credentials
- **File**: `pages/1_Module_1.py` (line 106), `pages/4_Finish.py` (line 227)
- **Usage**: `st.secrets["GOOGLE_CREDENTIALS"]`, `st.secrets.get("GOOGLE_CREDENTIALS")`
- **Status**: ✅ CORRECTLY stored in secrets
- **Contains**:
  - `private_key` - RSA private key for authentication
  - `private_key_id` - Key identifier
  - `client_email` - Service account email
  - `client_id` - Google Cloud project ID
  - `project_id` - `"teammait"`
- **Risk Level**: CRITICAL - allows full Google Cloud API access
- **Note**: This is a complete service account JSON that enables write access to Google Sheets

#### Google Sheet Name
- **File**: `pages/1_Module_1.py` (line 109), `pages/4_Finish.py` (line 228)
- **Usage**: `st.secrets["SHEET_NAME"]`
- **Status**: ✅ CORRECTLY stored in secrets
- **Value**: `"quick_and_clunky_teammait"`
- **Risk Level**: MEDIUM - identifies the target spreadsheet for data export

---

### 3. User Credentials

#### User Authentication Database
- **File**: `.streamlit/secrets.toml` (lines 22-30)
- **Status**: ✅ CORRECTLY stored in secrets (not in code)
- **Contains**: Username/password pairs for test users:
  - admin / n4@w2!kd9p
  - alpha / 6z#3j8!qvx
  - beta / 2b$7!f1ywm
  - gamma / 9k@4$c5hdo
  - delta / 1p!8#r2tse
  - epsilon / 5l$9@3guis
  - zeta / 7a#2!n6jbc
  - eta / 4e$1@8wfky
- **Status**: ✅ CORRECTLY stored in secrets
- **Risk Level**: MEDIUM - allows access to research study
- **Note**: These are test credentials for study participants

---

### 4. External Service URLs & Identifiers

#### Qualtrics Survey URL
- **File**: `pages/2_Qualtrics_Survey.py` (line 16)
- **Value**: `https://pennstate.qualtrics.com/jfe/form/SV_0pPPg0tAmtv31si`
- **Status**: ⚠️ SEMI-SENSITIVE - In code but not a secret
- **Risk Level**: LOW-MEDIUM - Identifies the specific survey form but this is intentional
- **Recommendation**: This is fine as-is (it's a public survey link)

#### Contact Email Address
- **File**: `Home.py` (line 120), `doc/consent_form.md`
- **Value**: `domattioli@psu.edu`
- **Status**: ✅ ACCEPTABLE - This is intentionally public contact info
- **Risk Level**: LOW - This is meant to be visible to users

#### Google Service Account Email
- **File**: `.streamlit/secrets.toml` (line 10)
- **Value**: `domattioli@teammait.iam.gserviceaccount.com`
- **Status**: ✅ CORRECTLY stored in secrets
- **Risk Level**: MEDIUM - Identifies the service account but not usable without private key

---

## Summary: What Should Be in Secrets

The following items **MUST** be stored in `.streamlit/secrets.toml` and **NEVER** committed to git:

### Current Status Table

| Item | File Location | Current Status | Should Be in Secrets | Risk Level |
|------|----------------|-----------------|---------------------|-----------|
| OpenAI API Key | 1_Module_1.py, 3_Module_2.py | ✅ In secrets | YES | CRITICAL |
| Anthropic API Key | 1_Module_1.py (commented) | ✅ In secrets | YES | CRITICAL |
| Google Credentials | 1_Module_1.py, 4_Finish.py | ✅ In secrets | YES | CRITICAL |
| Sheet Name | 1_Module_1.py, 4_Finish.py | ✅ In secrets | YES | MEDIUM |
| User Credentials | Home.py loading logic | ✅ In secrets | YES | MEDIUM |
| Hardcoded Admin Password | Home.py line 36 | ❌ IN CODE | NO | MEDIUM |
| Qualtrics URL | 2_Qualtrics_Survey.py | ✅ In code (public) | NO | LOW |
| Contact Email | Home.py | ✅ In code (public) | NO | LOW |

---

## Issues & Recommendations

### 🔴 ISSUE 1: Hardcoded Default Login Password
**Location**: `Home.py:36`
```python
password = st.text_input("Password", type="password", value="secureAdminPass123!")
```
**Problem**: While it's just a default field value, this password is visible in source code.
**Fix**: Remove the default value or use an empty string:
```python
password = st.text_input("Password", type="password", value="")
```

### 🟢 GOOD: Proper Secret Configuration
- `.streamlit/secrets.toml` is properly listed in `.gitignore`
- All API keys are loaded via `st.secrets.get()` pattern
- Fallback to environment variables is implemented with `get_secret_then_env()`
- Service account credentials stored as complete JSON in secrets

### 🟡 CAUTION: Google Sheet Access
The current setup exports all session data to a Google Sheet. Ensure:
- Service account has minimal necessary permissions
- Sheet is not publicly accessible
- Data is properly access-controlled at Google Cloud level
- Regular audits of who has access to the sheet

---

## Best Practices Checklist

- [x] API keys stored in secrets.toml, not code
- [x] Service account credentials in secrets.toml, not code
- [x] .streamlit/secrets.toml in .gitignore
- [x] get_secret_then_env() pattern for flexible config
- [x] Environment variable fallback implemented
- [ ] Remove hardcoded default password from Home.py
- [ ] Document all required secrets in setup instructions
- [ ] Regular audit of secrets exposure

---

## Files to Review Before Public Release

1. **Home.py** - Remove default password value (line 36)
2. **README.md** - Verify no secrets are documented with real values
3. **.gitignore** - Verify .streamlit/secrets.toml is ignored (✅ Already done)
4. **All page files** - Verify no API keys are hardcoded (✅ Already compliant)

---

## Setup Instructions for Developers

Create `.streamlit/secrets.toml` with:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
ANTHROPIC_API_KEY = ""  # Leave empty if not using Anthropic

GOOGLE_CREDENTIALS = """
{
  "type": "service_account",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "...",
  "client_email": "...",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "...",
  "universe_domain": "googleapis.com"
}
"""

SHEET_NAME = "your-google-sheet-name"

[credentials]
users = [
  { username = "user1", password = "password1" },
  { username = "user2", password = "password2" }
]
```

**NOTE**: Never commit this file. It's git-ignored for security reasons.

---

## Audit Completion

- **Date**: January 9, 2026
- **Status**: Code is mostly compliant with security best practices
- **Action Items**: 1 minor issue (remove hardcoded password default)
- **Security Posture**: GOOD - All critical secrets properly managed

