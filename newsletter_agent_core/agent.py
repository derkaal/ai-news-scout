# newsletter_agent_core/agent.py

import os
import base64
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Google API client imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# NEW: Import google.generativeai directly
import google.generativeai as genai

# Axiom configuration imports
from newsletter_agent_core.config import AxiomConfig
from newsletter_agent_core.axiom_prompts import (
    get_axiom_analysis_prompt,
    get_simple_extraction_prompt,
    get_cluster_validation_prompt,
    get_kill_gate_prompt
)

# Clustering engine imports
try:
    from .clustering import ClusteringOrchestrator, ClusteringConfig
    CLUSTERING_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Clustering engine not available: {e}")
    ClusteringOrchestrator = None
    ClusteringConfig = None
    CLUSTERING_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
print(f"DEBUG: GMAIL_LABEL loaded as: {os.getenv('GMAIL_LABEL')}")


# --- Configuration ---
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/spreadsheets'
]
GMAIL_LABEL = os.getenv("GMAIL_LABEL", "Newsletters")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
YOUR_INTERESTS = os.getenv("YOUR_INTERESTS",
    "Agentic shopping, agentic commerce, AI-driven purchasing, "
    "AI agents in retail, agentic AI, commerce platforms, "
    "marketplace AI, conversational commerce")

# Clustering configuration
ENABLE_CLUSTERING = os.getenv("ENABLE_CLUSTERING", "false").lower() == "true"
CLUSTERING_ALGORITHM = os.getenv("CLUSTERING_ALGORITHM", "hybrid")

# Axiom checker configuration
AXIOM_ENABLED = os.getenv("AXIOM_ENABLED", "true").lower() == "true"
AXIOM_FILTERING_MODE = os.getenv("AXIOM_FILTERING_MODE", "balanced")
AXIOM_INCLUDE_REPAIRS = os.getenv(
    "AXIOM_INCLUDE_REPAIRS", "true"
).lower() == "true"

# Cost optimization configuration
USE_CHEAP_MODEL_FOR_SCREENING = os.getenv(
    "USE_CHEAP_MODEL_FOR_SCREENING", "true"
).lower() == "true"
USE_EXPENSIVE_MODEL_FOR_ANALYSIS = os.getenv(
    "USE_EXPENSIVE_MODEL_FOR_ANALYSIS", "false"
).lower() == "true"

# Google AI API Key Configuration (used by google.generativeai)
# Initialize models based on cost optimization settings
global_model_for_tools = None
cheap_model = None
expensive_model = None

try:
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Initialize cheap model for screening (always available)
        cheap_model = genai.GenerativeModel("gemini-2.0-flash")
        print(
            "Configured cheap model (gemini-2.0-flash) for screening"
        )
        
        # Initialize expensive model if needed
        if USE_EXPENSIVE_MODEL_FOR_ANALYSIS:
            expensive_model = genai.GenerativeModel("gemini-2.5-pro")
            print(
                "Configured expensive model (gemini-2.5-pro) "
                "for detailed analysis"
            )
            global_model_for_tools = expensive_model
        else:
            global_model_for_tools = cheap_model
            print("Using cheap model for all operations (cost optimization)")
    else:
        print(
            "CRITICAL ERROR: GOOGLE_API_KEY environment variable "
            "not set. Cannot initialize AI model."
        )
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it in your .env file."
        )
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Google AI model: {e}")
    raise

# Initialize axiom configuration
axiom_config = None
if AXIOM_ENABLED:
    try:
        axiom_config = AxiomConfig()
        axiom_config.load()
        print(f"Axiom checker enabled (mode: {AXIOM_FILTERING_MODE})")
        print(f"  - Loaded {len(axiom_config.get_axioms())} axioms")
        print(f"  - Configuration version: {axiom_config.get_version()}")
    except Exception as e:
        print(f"WARNING: Failed to initialize axiom configuration: {e}")
        print("  - Falling back to basic extraction")
        axiom_config = None
        AXIOM_ENABLED = False
else:
    print("Axiom checker disabled - using basic extraction")


# --- Helper Functions for Authentication ---
def get_gmail_service():
    """Authenticates and returns a Gmail API service object."""
    creds = None
    token_path = os.path.join(os.path.dirname(__file__), '..', 'token.json')
    credentials_path = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def get_sheets_service():
    """Authenticates and returns a Google Sheets API service object."""
    creds = None
    token_path = os.path.join(os.path.dirname(__file__), '..', 'token.json')
    credentials_path = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('sheets', 'v4', credentials=creds)


# --- Core Logic Functions ---
def fetch_newsletters(
    gmail_label_name: str,
    days_back: int = 7
) -> List[Dict[str, Any]]:
    """
    Fetches newsletter emails from a specified Gmail label within a given number of days.
    """
    try:
        service = get_gmail_service()
        today = datetime.now()
        start_date = today - timedelta(days=days_back)
        query_date = start_date.strftime("%Y/%m/%d")

        query = f"label:{gmail_label_name} after:{query_date}"
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])

        newsletters = []
        if not messages:
            print(f"No new newsletters found in '{gmail_label_name}' since {query_date}.")
        else:
            print(f"Found {len(messages)} potential newsletters in '{gmail_label_name}' since {query_date}.")
            for msg in messages:
                msg_id = msg['id']
                msg_full = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                payload = msg_full['payload']
                headers = payload.get('headers', [])

                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'No Sender')
                
                body_html = ''
                if 'parts' in payload:
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/html' and 'body' in part:
                            data = part['body'].get('data')
                            if data:
                                body_html = base64.urlsafe_b64decode(data).decode('utf-8')
                                break
                        elif 'parts' in part:
                            for sub_part in part['parts']:
                                if sub_part['mimeType'] == 'text/html' and 'body' in sub_part:
                                    data = sub_part['body'].get('data')
                                    if data:
                                        body_html = base64.urlsafe_b64decode(data).decode('utf-8')
                                        break
                            if body_html: break
                elif payload['mimeType'] == 'text/html' and 'body' in payload:
                    data = payload['body'].get('data')
                    if data:
                        body_html = base64.urlsafe_b64decode(data).decode('utf-8')

                if body_html:
                    newsletters.append({
                        'id': msg_id,
                        'subject': subject,
                        'sender': sender,
                        'body_html': body_html
                    })
        return newsletters
    except HttpError as error:
        print(f"An error occurred with Gmail API: {error}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in fetch_newsletters: {e}")
        return []

def extract_clean_text(html_content: str) -> str:
    """
    Extracts and cleans plain text from HTML content.
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')

    for script_or_style in soup(['script', 'style']):
        script_or_style.extract()

    text = soup.get_text(separator='\n', strip=True)

    cleaned_text = re.sub(r'[ \t]+', ' ', text)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    cleaned_text = re.sub(r'View this email in your browser.*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Unsubscribe.*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'If you no longer wish to receive these emails.*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Copyright \d{4}.*reserved\.', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Sent by:.*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Your privacy is important to us.*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Manage Preferences \| Unsubscribe.*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Read this email in your browser\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Having trouble viewing this email\? View it in your browser\.\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Click here to unsubscribe\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'To ensure delivery, please add [^@]+@[\w.-]+ to your address book\.', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Forward to a friend.*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'If this email is not displaying correctly, you can view it here.*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'To view this email as a web page, click here\..*', '', cleaned_text, flags=re.IGNORECASE)
    
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def _sanitize_json_strings(text: str) -> str:
    """
    Fix unescaped newlines, tabs, and control characters inside JSON string
    values. The LLM often produces multi-line strings that break json.loads().

    Works by walking the text character-by-character, tracking whether we're
    inside a JSON string (between unescaped double quotes), and replacing
    raw newlines/tabs with their escaped equivalents.
    """
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]

        if ch == '\\' and in_string and i + 1 < len(text):
            # Escaped character — keep as-is (including \n, \", etc.)
            result.append(ch)
            result.append(text[i + 1])
            i += 2
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if in_string:
            # Replace raw control characters with escaped versions
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            result.append(ch)

        i += 1

    return ''.join(result)


def _parse_json_safe(raw_text: str, expected_type=None):
    """
    Robust JSON parser that handles common LLM output issues:
    - Markdown code fences around JSON
    - Unescaped newlines/quotes inside string values
    - Trailing commas
    - Single objects when list expected

    Args:
        raw_text: Raw LLM response text
        expected_type: Expected top-level type (list, dict, or None for any)

    Returns:
        Parsed JSON object, or None if parsing fails
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    def _try_parse(s):
        """Attempt json.loads, return result or None."""
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None

    def _coerce_type(result):
        """Coerce result to expected_type if needed."""
        if expected_type and not isinstance(result, expected_type):
            if expected_type == list and isinstance(result, dict):
                return [result]
        return result

    # Attempt 1: Direct parse
    result = _try_parse(text)
    if result is not None:
        return _coerce_type(result)

    # Attempt 2: Sanitize unescaped newlines inside strings, then parse
    sanitized = _sanitize_json_strings(text)
    result = _try_parse(sanitized)
    if result is not None:
        return _coerce_type(result)

    # Attempt 3: Extract JSON array/object with regex, then sanitize
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(0)
            result = _try_parse(extracted)
            if result is not None:
                return _coerce_type(result)
            # Try sanitized version of extracted
            result = _try_parse(_sanitize_json_strings(extracted))
            if result is not None:
                return _coerce_type(result)

    # Attempt 4: Remove trailing commas and retry
    no_trailing = re.sub(r',\s*([}\]])', r'\1', sanitized)
    result = _try_parse(no_trailing)
    if result is not None:
        return _coerce_type(result)

    print(f"  - WARNING: _parse_json_safe failed. First 300 chars: {text[:300]}")
    return None


def summarize_and_extract_topics(
    text_content: str,
    interests: str, # This now includes the specific focus areas
    original_subject: str,
    original_sender: str
) -> Dict[str, Any]:
    """
    Analyzes newsletter content to extract relevant headlines, descriptions, source, date,
    regional implications, and LinkedIn angles for agentic shopping. Handles large inputs by chunking.
    """
    print(f"Attempting summarization for '{original_subject}'...")
    if not text_content:
        print("  - No text content provided for summarization.")
        return {
            'extracted_items': [],
            'relevance_score': 0
        }

    if global_model_for_tools is None:
        print("  - ERROR: AI model not initialized for summarization.")
        return {
            'extracted_items': [],
            'relevance_score': 0
        }

    # --- Token Count and Chunking Logic ---
    final_text_for_llm = text_content
    
    # Enhanced token counting with timeout and retry logic
    current_token_count = None
    max_retries = 2
    timeout_seconds = 10
    
    for attempt in range(max_retries + 1):
        try:
            print(f"  - Attempting token count for '{original_subject}' (attempt {attempt + 1}/{max_retries + 1})...")
            
            # Import signal for timeout handling (Unix/Linux) or use threading for cross-platform
            import threading
            import time
            
            # Create a result container for the threaded operation
            result_container = {'result': None, 'error': None}
            
            def count_tokens_with_timeout():
                try:
                    result_container['result'] = global_model_for_tools.count_tokens(text_content)
                except Exception as e:
                    result_container['error'] = e
            
            # Start the token counting in a separate thread
            thread = threading.Thread(target=count_tokens_with_timeout)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                print(f"  - Token counting timed out after {timeout_seconds}s for '{original_subject}' (attempt {attempt + 1})")
                if attempt < max_retries:
                    print(f"  - Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    raise TimeoutError(f"Token counting timed out after {timeout_seconds}s")
            
            if result_container['error']:
                raise result_container['error']
            
            if result_container['result']:
                current_token_count = result_container['result'].total_tokens
                print(f"  - Successfully counted tokens for '{original_subject}': {current_token_count} tokens.")
                break
            else:
                raise ValueError("No result returned from token counting")
                
        except Exception as e:
            print(f"  - Token counting failed for '{original_subject}' (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                print(f"  - Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                print(f"  - All token counting attempts failed for '{original_subject}': {e}. Using character-based estimation.")
                current_token_count = len(text_content) / 4
                break

    MAX_MODEL_INPUT_TOKENS = 1048575 # Max for Gemini 1.5 Flash (1M tokens)
    TARGET_CHUNK_SIZE_TOKENS = 500000 # Aim for chunks around half the context window
    CHUNK_OVERLAP_TOKENS = 1000 # Small overlap

    if current_token_count > (MAX_MODEL_INPUT_TOKENS * 0.9):
        print(f"  - Newsletter '{original_subject}' is too long ({current_token_count} tokens). Chunking and summarizing parts.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TARGET_CHUNK_SIZE_TOKENS * 4,
            chunk_overlap=CHUNK_OVERLAP_TOKENS * 4,
            length_function=len,
        )
        chunks = text_splitter.split_text(text_content)
        print(f"  - Split into {len(chunks)} chunks.")
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            # Safe chunk token counting with timeout
            chunk_token_count = None
            try:
                result_container = {'result': None, 'error': None}
                
                def count_chunk_tokens():
                    try:
                        result_container['result'] = global_model_for_tools.count_tokens(chunk)
                    except Exception as e:
                        result_container['error'] = e
                
                thread = threading.Thread(target=count_chunk_tokens)
                thread.daemon = True
                thread.start()
                thread.join(timeout=5)  # Shorter timeout for chunks
                
                if thread.is_alive():
                    print(f"  - Chunk {i+1} token counting timed out, using character estimation")
                    chunk_token_count = len(chunk) / 4
                elif result_container['error']:
                    print(f"  - Chunk {i+1} token counting failed: {result_container['error']}, using character estimation")
                    chunk_token_count = len(chunk) / 4
                elif result_container['result']:
                    chunk_token_count = result_container['result'].total_tokens
                else:
                    chunk_token_count = len(chunk) / 4
                    
            except Exception as e:
                print(f"  - Error in chunk {i+1} token counting: {e}, using character estimation")
                chunk_token_count = len(chunk) / 4
            
            if chunk_token_count > (MAX_MODEL_INPUT_TOKENS * 0.95):
                print(f"  - WARNING: Chunk {i+1} for '{original_subject}' is still very large ({chunk_token_count} tokens). Truncating for sub-summary.")
                chunk = chunk[:int(MAX_MODEL_INPUT_TOKENS * 0.9 * 4)]
            
            chunk_prompt = f"""
Analyze the following text from a newsletter chunk. Focus on information relevant to: "{interests}" (agentic shopping, agentic commerce, AI-driven purchasing, AI agents in retail, marketplace AI).
Extract only key factual points related to agentic shopping and commerce.

Concise Summary (max 3-5 sentences, strictly factual, no commentary):
---
{chunk}
---
"""
            try:
                chunk_response = global_model_for_tools.generate_content(
                    chunk_prompt,
                    generation_config={"response_mime_type": "text/plain"}
                )
                if chunk_response.text:
                    chunk_summaries.append(chunk_response.text.strip())
                    print(f"  - Successfully summarized chunk {i+1}/{len(chunks)} for '{original_subject}'.")
                else:
                    print(f"  - WARNING: Chunk {i+1} summary for '{original_subject}' was empty.")
            except Exception as e:
                print(f"  - ERROR: summarizing chunk {i+1} for '{original_subject}': {e}. Skipping this chunk.")

        if chunk_summaries:
            combined_chunk_summaries = "\n\n---\n\n".join(chunk_summaries)
            
            # Safe combined token counting with timeout
            combined_token_count = None
            try:
                result_container = {'result': None, 'error': None}
                
                def count_combined_tokens():
                    try:
                        result_container['result'] = global_model_for_tools.count_tokens(combined_chunk_summaries)
                    except Exception as e:
                        result_container['error'] = e
                
                thread = threading.Thread(target=count_combined_tokens)
                thread.daemon = True
                thread.start()
                thread.join(timeout=5)
                
                if thread.is_alive():
                    print(f"  - Combined summaries token counting timed out, using character estimation")
                    combined_token_count = len(combined_chunk_summaries) / 4
                elif result_container['error']:
                    print(f"  - Combined summaries token counting failed: {result_container['error']}, using character estimation")
                    combined_token_count = len(combined_chunk_summaries) / 4
                elif result_container['result']:
                    combined_token_count = result_container['result'].total_tokens
                else:
                    combined_token_count = len(combined_chunk_summaries) / 4
                    
            except Exception as e:
                print(f"  - Error in combined summaries token counting: {e}, using character estimation")
                combined_token_count = len(combined_chunk_summaries) / 4
            
            if combined_token_count > (MAX_MODEL_INPUT_TOKENS * 0.9):
                print(f"  - Combined summaries for '{original_subject}' still too long ({combined_token_count} tokens). Truncating for final pass.")
                final_text_for_llm = combined_chunk_summaries[:int(MAX_MODEL_INPUT_TOKENS * 0.9 * 4)]
            else:
                final_text_for_llm = combined_chunk_summaries
        else:
            print(f"  - No summaries generated from any chunks for '{original_subject}'.")
            return {
                'extracted_items': [],
                'relevance_score': 0
            }
    else:
        final_text_for_llm = text_content


    # --- Main Prompt for Detailed Item Extraction and Analysis ---
    # Select model based on cost optimization
    model_to_use = (cheap_model if USE_CHEAP_MODEL_FOR_SCREENING
                    else global_model_for_tools)
    
    # Generate prompt based on axiom configuration
    if AXIOM_ENABLED and axiom_config:
        axioms_text = axiom_config.get_prompt_text()
        prompt_template = get_axiom_analysis_prompt(
            axioms_text,
            final_text_for_llm
        )
        print(f"  - Using axiom checker (mode: {AXIOM_FILTERING_MODE})")
    else:
        prompt_template = get_simple_extraction_prompt(
            interests,
            final_text_for_llm
        )
        print("  - Using simple extraction mode")

    try:
        print(
            f"  - Sending final structured extraction request to "
            f"{'cheap' if model_to_use == cheap_model else 'expensive'} "
            f"model for '{original_subject}'."
        )
        response = model_to_use.generate_content(
            prompt_template,
            generation_config={"response_mime_type": "application/json"}
        )
        
        summary_data = response.text
        print(f"  - Raw Gemini final response for '{original_subject}':\n---\n{summary_data[:2000] if summary_data else 'EMPTY RESPONSE'}\n---")
        
        json_string = ""
        json_match = re.search(r"(\{.*\})", summary_data, re.DOTALL) # Looks for first {.*} block
        if json_match:
            json_string = json_match.group(1).strip()
            if not json_string.startswith('[') and not json_string.endswith(']'):
                json_string = f"[{json_string}]"
        elif summary_data and summary_data.strip().startswith('[') and summary_data.strip().endswith(']'):
            json_string = summary_data.strip()
        else:
            list_match = re.search(r"\[.*\]", summary_data, re.DOTALL)
            if list_match:
                json_string = list_match.group(0).strip()
            elif summary_data:
                json_string = summary_data.strip()

        if not json_string:
            print(f"  - JSON string is empty after extraction attempt for '{original_subject}'.")
            raise json.JSONDecodeError("Empty or invalid JSON string after extraction attempt", summary_data or "None", 0)
            
        parsed_data = json.loads(json_string)
        
        if not isinstance(parsed_data, list):
            print(f"  - WARNING: Gemini returned non-list JSON: {parsed_data}. Attempting to convert if single object.")
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
            else:
                raise json.JSONDecodeError(f"Expected JSON list of objects, got {type(parsed_data)}", summary_data or "None", 0)

        final_extracted_items = []
        for item in parsed_data:
            # Build required_keys based on axiom mode
            if AXIOM_ENABLED and axiom_config:
                required_keys = [
                    'master_headline', 'headline', 'short_description',
                    'source', 'date', 'companies', 'technologies',
                    'reality_status', 'reality_reason',
                    'axiom_check', 'violations',
                    'regional_implications', 'linkedin_angle'
                ]
            else:
                # Simple extraction mode
                required_keys = [
                    'master_headline', 'headline', 'short_description',
                    'source', 'date', 'companies', 'technologies',
                    'regional_implications', 'linkedin_angle'
                ]

            if isinstance(item, dict) and all(key in item for key in required_keys):
                # Format axiom data for easier sheet writing
                if 'violations' in item and isinstance(item['violations'], dict):
                    item['violation_count'] = item['violations'].get('count', 0)
                    item['top_violations_formatted'] = '; '.join(
                        item['violations'].get('top_violations', [])
                    )
                    item['minimal_repairs_formatted'] = '; '.join(
                        item['violations'].get('minimal_repair', [])
                    )
                # Format regional implications for sheet writing
                if 'regional_implications' in item and isinstance(item['regional_implications'], dict):
                    ri = item['regional_implications']
                    item['region_us'] = ri.get('US') or ''
                    item['region_eu'] = ri.get('EU') or ''
                    item['region_cn'] = ri.get('CN') or ''
                    item['region_asia'] = ri.get('ASIA') or ''
                final_extracted_items.append(item)
            else:
                missing_keys = [k for k in required_keys if k not in item] if isinstance(item, dict) else required_keys
                print(f"  - WARNING: Skipping malformed extracted item. Missing keys: {missing_keys}. Check LLM output format.")
        
        if not final_extracted_items and not parsed_data:
            relevance_score = 0
            brief_summary = "No relevant content found."
        elif not final_extracted_items:
            relevance_score = 0
            brief_summary = "No well-formed relevant items found."
        else:
            relevance_score = 90 if final_extracted_items else 0
            brief_summary = final_extracted_items[0].get('short_description', 'N/A') if final_extracted_items else "N/A"

        return {
            'extracted_items': final_extracted_items,
            'relevance_score': relevance_score,
            'newsletter_summary_brief': brief_summary
        }

    except json.JSONDecodeError as e:
        print(f"  - ERROR: JSON PARSING FAILED for '{original_subject}': {e}. Raw response might not be valid JSON.")
        print(f"  - Full raw response that failed JSON parsing:\n---\n{summary_data}\n---")
        print(f"  - Attempting fallback to simpler text summary...")
        try:
            fallback_prompt = f"""Summarize the following text, focusing on topics related to: {interests}.
            Text: {final_text_for_llm}
            ---
            Concise Summary (max 3-5 sentences):
            """
            fallback_response = global_model_for_tools.generate_content(fallback_prompt)
            fallback_summary = fallback_response.text.strip() if fallback_response.text else "No fallback summary generated."
            print(f"  - Fallback summary generated: {fallback_summary}")
            return {
                'extracted_items': [],
                'relevance_score': 0,
                'newsletter_summary_brief': f"Summary failed to parse as JSON. Fallback: {fallback_summary}"
            }
        except Exception as fallback_e:
            print(f"  - ERROR: Fallback summarization also failed for '{original_subject}': {fallback_e}")
            return {
                'extracted_items': [],
                'relevance_score': 0,
                'newsletter_summary_brief': "Could not summarize due to AI processing error (both JSON and fallback failed)."
            }
    except Exception as e:
        print(f"  - UNEXPECTED SUMMARIZATION ERROR for '{original_subject}': {e}. Returning default error structure.")
        return {
            'extracted_items': [],
            'relevance_score': 0,
            'newsletter_summary_brief': "Could not summarize due to AI processing error."
        }


# Keywords that signal core agentic shopping (transactional, not just AI-adjacent)
# CORE keywords: multi-word phrases that unambiguously signal agentic commerce transactions.
# Single ambiguous words like "order", "bid", "transaction" are NOT included because
# they match too broadly (e.g., "executive order", "bid for attention", "transaction disputes").
# The classifier only searches headline + short_description (NOT linkedin_angle, which the
# LLM writes and can contain any commerce word as commentary).
_CORE_KEYWORDS = [
    # Agent financial infrastructure
    'agent wallet', 'agentic wallet', 'agent payment', 'agent purchase',
    'agent checkout', 'agentic checkout', 'agentic transaction',
    'agentic procurement', 'agentic commerce',
    # Autonomous purchasing
    'autonomous purchase', 'autonomous purchasing', 'autonomous buying',
    'auto-purchase', 'delegated spending', 'spending authority',
    'budget delegation', 'ai purchasing', 'ai shopping',
    # Agent-to-agent
    'agent-to-agent', 'machine-to-machine commerce', 'm2m commerce',
    # Shopping-specific (must reference agents/AI + shopping together)
    'shopping agent', 'shopping ai', 'ai agent.*checkout',
    'ai agent.*purchase', 'ai agent.*transaction',
]


def classify_relevance_tier(item: Dict[str, Any]) -> str:
    """
    Classify an item as CORE or ADJACENT based on whether it directly
    involves transactional agentic shopping mechanics.

    CORE: Directly about payment authority, agent wallets, autonomous transactions,
          agent purchasing decisions, agent-to-agent commerce
    ADJACENT: Related to agentic commerce ecosystem but not directly transactional
              (e.g., platform infrastructure, regulation, capabilities, AI safety)

    Only searches headline + short_description to avoid false positives from
    LLM-generated commentary in linkedin_angle.

    Args:
        item: Extracted newsletter item dict

    Returns:
        "CORE" or "ADJACENT"
    """
    # Only search factual fields — NOT linkedin_angle (LLM commentary)
    searchable = " ".join([
        item.get('headline', ''),
        item.get('short_description', ''),
    ]).lower()

    for kw in _CORE_KEYWORDS:
        if '.*' in kw:
            # Regex pattern for keywords that need proximity matching
            if re.search(kw, searchable):
                return "CORE"
        elif kw in searchable:
            return "CORE"

    return "ADJACENT"


def deduplicate_by_master_headline(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate items that share the same underlying story.

    Uses master_headline (normalized) as the dedup key. When duplicates
    are found, keeps the item with the most violations (richest analysis).
    Falls back to headline similarity check for items without master_headline.

    Args:
        items: List of extracted items, possibly from multiple newsletters

    Returns:
        Deduplicated list, keeping the best version of each story
    """
    if not items:
        return items

    def _normalize(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    # Group items by normalized master_headline
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        key = _normalize(
            item.get('master_headline', item.get('headline', ''))
        )
        if not key:
            key = _normalize(item.get('headline', 'unknown'))
        groups.setdefault(key, []).append(item)

    # Fuzzy matching: merge groups that clearly refer to the same story
    _GENERIC_WORDS = {
        'ai', 'agent', 'agents', 'agentic', 'launches', 'launch',
        'new', 'for', 'the', 'and', 'in', 'on', 'of', 'to', 'with',
        'introduces', 'unveils', 'announces', 'offers', 'platform',
    }

    def _is_same_story(a: str, b: str) -> bool:
        """Check if two normalized headlines refer to the same story."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return False
        intersection = words_a & words_b
        union = words_a | words_b
        jaccard = len(intersection) / len(union) if union else 0.0

        # Need high Jaccard AND shared distinctive words (not just generic terms)
        distinctive_overlap = intersection - _GENERIC_WORDS
        return jaccard >= 0.4 and len(distinctive_overlap) >= 1

    keys = list(groups.keys())
    merged = set()
    for i, k1 in enumerate(keys):
        if k1 in merged:
            continue
        for j, k2 in enumerate(keys):
            if i >= j or k2 in merged:
                continue
            should_merge = False
            # Check if one is a substantial substring of the other
            if (len(k1) > 10 and len(k2) > 10 and
                    (k1 in k2 or k2 in k1)):
                should_merge = True
            # Check word-level similarity with distinctive word requirement
            elif _is_same_story(k1, k2):
                should_merge = True

            if should_merge:
                # Merge k2 into k1
                groups[k1].extend(groups[k2])
                merged.add(k2)

    for k in merged:
        del groups[k]

    # From each group, keep the item with the most violations (richest analysis)
    deduped = []
    for key, group in groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Pick best: most violations > most tensions > first encountered
            best = max(
                group,
                key=lambda x: (
                    x.get('violation_count', 0),
                    sum(1 for v in x.get('axiom_check', {}).values()
                        if isinstance(v, dict) and v.get('judgment') == 'TENSION'),
                )
            )
            deduped.append(best)
            for dup in group:
                if dup is not best:
                    print(
                        f"  - DEDUP: Removed '{dup.get('headline', 'N/A')}' "
                        f"(duplicate of '{best.get('headline', 'N/A')}')"
                    )

    print(f"  - Deduplication: {len(deduped)}/{len(items)} unique items")
    return deduped


def apply_axiom_quality_filter(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-axiom filter with tiered criteria based on relevance tier.

    CORE items (direct transaction mechanics) get a lighter bar:
      - Must have axiom engagement (3+ aligned OR 1+ tension/violation)
      - Must have angle quality (>100 chars, not GENERIC_ANGLE)

    ADJACENT items (ecosystem, not transactional) get a harder bar:
      - Must have at least 1 VIOLATION (not just tensions — tensions are cheap)
      - Must have angle quality (>100 chars, not GENERIC_ANGLE)
      - Must have substance (companies or technologies identified)

    All items:
      - OPINION/ANALYSIS with zero violations → rejected
      - GENERIC_ANGLE → rejected

    Args:
        items: List of extracted items with axiom_check data

    Returns:
        Filtered list of items that pass the quality gate
    """
    if not AXIOM_ENABLED or not axiom_config:
        return items

    filtered = []
    for item in items:
        axiom_check = item.get('axiom_check', {})
        reject_reasons = []
        relevance_tier = item.get('relevance_tier', 'ADJACENT')

        if not axiom_check:
            item['quality_gate'] = 'NO_DATA'
            filtered.append(item)
            continue

        # --- Count axiom judgments ---
        aligned_with_evidence = 0
        tension_count = 0
        violation_count_from_axioms = 0
        na_count = 0

        for axiom_id, axiom_data in axiom_check.items():
            if not isinstance(axiom_data, dict):
                continue
            judgment = axiom_data.get('judgment', 'N/A')
            reason = axiom_data.get('reason', '')

            if judgment == 'ALIGNED' and len(reason) > 10:
                aligned_with_evidence += 1
            elif judgment == 'TENSION':
                tension_count += 1
            elif judgment == 'VIOLATION':
                violation_count_from_axioms += 1
            elif judgment == 'N/A':
                na_count += 1

        tension_violation_count = tension_count + violation_count_from_axioms
        reality_status = item.get('reality_status', '')
        violation_count = item.get('violation_count', 0)
        linkedin_angle = item.get('linkedin_angle', '')
        companies = item.get('companies', [])
        technologies = item.get('technologies', [])

        # --- Universal criteria ---

        # OPINION/ANALYSIS with zero violations = no structural tension
        if reality_status in ('OPINION', 'ANALYSIS') and violation_count == 0:
            reject_reasons.append(
                f"{reality_status} with zero violations — no structural tension"
            )

        # Angle quality
        if linkedin_angle == 'GENERIC_ANGLE':
            reject_reasons.append("GENERIC_ANGLE — no non-obvious take found")
        elif len(linkedin_angle) < 100:
            reject_reasons.append(
                f"weak angle ({len(linkedin_angle)} chars < 100 minimum)"
            )

        # --- Tier-specific criteria ---

        if relevance_tier == 'CORE':
            # CORE: need axiom engagement AND structural substance
            axiom_engaged = (aligned_with_evidence >= 3 or tension_violation_count >= 1)
            if not axiom_engaged:
                reject_reasons.append(
                    f"CORE but weak axiom engagement "
                    f"(aligned={aligned_with_evidence}, tv={tension_violation_count})"
                )
            # CORE with zero violations AND zero tensions = bland consensus
            if violation_count_from_axioms == 0 and tension_count == 0:
                reject_reasons.append(
                    "CORE with zero violations AND zero tensions — no structural tension"
                )
        else:
            # ADJACENT: harder bar — need real violations, not just tensions
            if violation_count_from_axioms < 1:
                reject_reasons.append(
                    f"ADJACENT with no VIOLATIONS "
                    f"(tensions={tension_count} don't count for ADJACENT)"
                )
            # Must have substance
            if not companies and not technologies:
                reject_reasons.append(
                    "ADJACENT with no companies AND no technologies"
                )

        # --- Decision ---
        if not reject_reasons:
            item['quality_gate'] = 'PASS'
            item['quality_gate_detail'] = (
                f"tier={relevance_tier}, aligned={aligned_with_evidence}, "
                f"tensions={tension_count}, violations={violation_count_from_axioms}, "
                f"angle_len={len(linkedin_angle)}"
            )
            filtered.append(item)
        else:
            item['quality_gate'] = 'FAIL'
            item['quality_gate_detail'] = '; '.join(reject_reasons)
            print(
                f"  - QUALITY FILTER: Rejected '{item.get('headline', 'N/A')}' — "
                f"{'; '.join(reject_reasons)}"
            )

    print(f"  - Axiom quality filter: {len(filtered)}/{len(items)} items passed")
    return filtered


# Broader shopping/commerce keywords for pre-kill-gate check (superset of _CORE_KEYWORDS)
# Broader shopping/commerce keywords for pre-kill-gate check.
# These are simple substring matches (no regex) — used to detect any commerce connection.
_SHOPPING_KEYWORDS = [
    # Core transactional terms
    'payment', 'wallet', 'transaction', 'purchase', 'purchasing',
    'checkout', 'buy', 'buying', 'procurement', 'negotiate', 'negotiation',
    'bid', 'bidding', 'auction', 'spending', 'budget',
    # Broader commerce terms
    'shopping', 'commerce', 'retail', 'merchant', 'marketplace',
    'e-commerce', 'ecommerce', 'store', 'cart', 'price', 'pricing',
    'consumer', 'buyer', 'seller', 'vendor', 'supplier',
    'fulfillment', 'delivery', 'logistics', 'supply chain',
    'agent shopping', 'agentic commerce', 'agentic shopping',
    'product search', 'product discovery', 'comparison shopping',
]


def apply_pre_kill_filter(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic pre-kill-gate filter that removes items which clearly
    don't belong before wasting an LLM call.

    Two-tier filtering:

    Tier 1 (hard — any reality status):
      Remove if BOTH: violation_count == 0 AND relevance_tier == ADJACENT
      AND no shopping keywords anywhere in the item.
      These items have no connection to agentic shopping at all.

    Tier 2 (soft — REPORTED/OPINION only):
      Remove if ALL: violation_count == 0, reality_status in {REPORTED, OPINION},
      AND no CORE keywords (the strict transactional set).
      These are weakly-sourced items with no transactional angle.

    Args:
        items: List of items that passed the quality gate

    Returns:
        Filtered list with obviously irrelevant items removed
    """
    filtered = []
    for item in items:
        violation_count = item.get('violation_count', 0)
        reality_status = item.get('reality_status', '')
        relevance_tier = item.get('relevance_tier', 'ADJACENT')

        # Build searchable text from core fields
        searchable = " ".join([
            item.get('headline', ''),
            item.get('short_description', ''),
            " ".join(item.get('technologies', [])),
        ]).lower()

        has_shopping_keyword = any(kw in searchable for kw in _SHOPPING_KEYWORDS)
        has_core_keyword = any(
            re.search(kw, searchable) if '.*' in kw else (kw in searchable)
            for kw in _CORE_KEYWORDS
        )

        # --- Tier 1: Hard filter — no shopping connection at all ---
        if (violation_count == 0
                and relevance_tier == 'ADJACENT'
                and not has_shopping_keyword):
            print(
                f"  - PRE-KILL FILTER (T1): Removed '{item.get('headline', 'N/A')}' — "
                f"zero violations, ADJACENT, no shopping keywords"
            )
            item['kill_gate'] = 'PRE_KILL'
            item['kill_gate_reason'] = (
                'No violations, ADJACENT tier, no shopping/commerce keywords'
            )
            continue

        # --- Tier 2: Soft filter — weak source + no transactional angle ---
        if (violation_count == 0
                and reality_status in ('REPORTED', 'OPINION')
                and not has_core_keyword):
            print(
                f"  - PRE-KILL FILTER (T2): Removed '{item.get('headline', 'N/A')}' — "
                f"zero violations, {reality_status}, no core transactional keywords"
            )
            item['kill_gate'] = 'PRE_KILL'
            item['kill_gate_reason'] = (
                f'No violations, {reality_status}, no core transactional keywords'
            )
            continue

        filtered.append(item)

    print(f"  - Pre-kill filter: {len(filtered)}/{len(items)} items passed")
    return filtered


def validate_clusters_with_llm(
    items: List[Dict[str, Any]],
    clustering_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    LLM validation step after clustering: verify each cluster can support
    a genuine synthesis thesis, not just a thematic list.

    Args:
        items: List of items with cluster metadata
        clustering_result: Full clustering result dict

    Returns:
        Items with cluster_synthesis and cluster_thesis fields added
    """
    if not clustering_result or not clustering_result.get('clustering_applied', False):
        return items

    model_to_use = (cheap_model if USE_CHEAP_MODEL_FOR_SCREENING
                    else global_model_for_tools)
    if model_to_use is None:
        print("  - WARNING: No model available for cluster validation")
        return items

    # Group items by cluster_id
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for item in items:
        cid = item.get('cluster_id', -1)
        if cid >= 0:  # Skip noise items
            clusters.setdefault(cid, []).append(item)

    # Validate each cluster with 2+ items
    cluster_verdicts: Dict[int, Dict[str, Any]] = {}
    for cid, cluster_items in clusters.items():
        if len(cluster_items) < 2:
            cluster_verdicts[cid] = {
                'has_synthesis': False,
                'thesis': '',
                'reason': 'Single-item cluster'
            }
            continue

        # Format cluster items for the prompt
        items_text = "\n".join(
            f"- **{ci.get('headline', 'N/A')}**: {ci.get('short_description', 'N/A')}"
            for ci in cluster_items
        )

        prompt = get_cluster_validation_prompt(items_text)
        try:
            response = model_to_use.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            verdict = _parse_json_safe(response.text or "", expected_type=dict)
            if verdict is None:
                raise ValueError("JSON parsing failed for cluster validation response")
            cluster_verdicts[cid] = verdict
            synth = verdict.get('has_synthesis', False)
            print(
                f"  - Cluster {cid} ({len(cluster_items)} items): "
                f"synthesis={'YES' if synth else 'NO'}"
                f"{' — ' + verdict.get('thesis', '')[:80] if synth else ''}"
            )
        except Exception as e:
            print(f"  - WARNING: Cluster {cid} validation failed: {e}")
            cluster_verdicts[cid] = {
                'has_synthesis': False,
                'thesis': '',
                'reason': f'Validation error: {e}'
            }

    # Apply verdicts to items
    for item in items:
        cid = item.get('cluster_id', -1)
        if cid >= 0 and cid in cluster_verdicts:
            v = cluster_verdicts[cid]
            item['cluster_synthesis'] = v.get('has_synthesis', False)
            item['cluster_thesis'] = v.get('thesis', '')
        else:
            item['cluster_synthesis'] = None
            item['cluster_thesis'] = ''

    return items


def _kill_gate_eval_batch(
    batch_items: List[Dict[str, Any]],
    model
) -> List[Dict[str, Any]]:
    """
    Evaluate a small batch of items through the kill gate LLM.

    Args:
        batch_items: Compact item dicts for evaluation (max ~5)
        model: Gemini model to use

    Returns:
        List of verdict dicts, or None if parsing fails
    """
    # Truncate descriptions and angles to reduce token size and JSON complexity
    compact_batch = []
    for item in batch_items:
        compact_batch.append({
            'headline': item.get('headline', 'N/A'),
            'short_description': (item.get('short_description', '') or '')[:200],
            'reality_status': item.get('reality_status', 'N/A'),
            'violation_count': item.get('violation_count', 0),
            'relevance_tier': item.get('relevance_tier', 'ADJACENT'),
            'linkedin_angle': (item.get('linkedin_angle', '') or '')[:150],
        })

    prompt = get_kill_gate_prompt(json.dumps(compact_batch, indent=2))
    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.1,  # Low temperature = strict, deterministic judgments
        }
    )
    raw = response.text or ""
    result = _parse_json_safe(raw, expected_type=list)
    if result is None:
        # Log the raw response for debugging
        print(f"    - Kill gate batch raw response ({len(raw)} chars): {raw[:300]}...")
    return result


def _kill_gate_eval_single(item_eval: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Evaluate a single item through the kill gate LLM (fallback for batch failures).

    Args:
        item_eval: Compact item dict for evaluation
        model: Gemini model to use

    Returns:
        Verdict dict with headline, verdict, reason
    """
    try:
        verdicts = _kill_gate_eval_batch([item_eval], model)
        if verdicts and isinstance(verdicts, list) and len(verdicts) > 0:
            v = verdicts[0]
            if isinstance(v, dict) and 'verdict' in v:
                return v
    except Exception as e:
        print(f"    - Single-item kill gate also failed for '{item_eval.get('headline', '?')}': {e}")

    return {
        'headline': item_eval.get('headline', ''),
        'verdict': 'NEEDS_REVIEW',
        'reason': 'Kill gate evaluation failed'
    }


def apply_kill_gate(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Final kill gate: remove items where high relevance + no tensions +
    generic angle = actually not worth writing about.

    Processes items in small batches (max 5) to avoid JSON parsing failures
    from oversized LLM responses. Falls back to per-item evaluation if a
    batch fails.

    Args:
        items: List of items that have passed all prior filters

    Returns:
        Filtered list of items that survived the kill gate
    """
    if not items:
        return items

    model_to_use = (cheap_model if USE_CHEAP_MODEL_FOR_SCREENING
                    else global_model_for_tools)
    if model_to_use is None:
        print("  - WARNING: No model available for kill gate")
        return items

    BATCH_SIZE = 5

    # Build compact representations keyed by headline for matching back
    items_for_eval = []
    for item in items:
        items_for_eval.append({
            'headline': item.get('headline', 'N/A'),
            'short_description': item.get('short_description', 'N/A'),
            'reality_status': item.get('reality_status', 'N/A'),
            'violation_count': item.get('violation_count', 0),
            'relevance_tier': item.get('relevance_tier', 'ADJACENT'),
            'linkedin_angle': item.get('linkedin_angle', 'N/A'),
        })

    # Process in batches
    all_verdicts: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(items_for_eval), BATCH_SIZE):
        batch = items_for_eval[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(items_for_eval) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  - Kill gate batch {batch_num}/{total_batches} ({len(batch)} items)...")

        try:
            verdicts = _kill_gate_eval_batch(batch, model_to_use)

            if verdicts is None:
                raise ValueError("Batch JSON parsing returned None")

            # Map verdicts by headline
            for v in verdicts:
                if isinstance(v, dict) and 'headline' in v:
                    all_verdicts[v['headline']] = v

        except Exception as e:
            print(f"  - WARNING: Kill gate batch {batch_num} failed: {e}. Retrying per-item...")
            for single_item in batch:
                v = _kill_gate_eval_single(single_item, model_to_use)
                all_verdicts[v.get('headline', single_item.get('headline', ''))] = v

    # Match verdicts back to items
    survivors = []
    for item in items:
        headline = item.get('headline', 'N/A')

        # Try exact match, then fuzzy substring match
        verdict = all_verdicts.get(headline)
        if verdict is None:
            for vk, vv in all_verdicts.items():
                if vk and (vk in headline or headline in vk):
                    verdict = vv
                    break

        if verdict is None:
            item['kill_gate'] = 'NEEDS_REVIEW'
            item['kill_gate_reason'] = 'No LLM verdict matched'
        else:
            item['kill_gate'] = verdict.get('verdict', 'NEEDS_REVIEW')
            item['kill_gate_reason'] = verdict.get('reason', '')

        if item['kill_gate'] in ('KEEP', 'NEEDS_REVIEW'):
            survivors.append(item)
        else:
            print(
                f"  - KILL GATE: Removed '{headline}' — "
                f"{item.get('kill_gate_reason', 'no reason')}"
            )

    print(f"  - Kill gate: {len(survivors)}/{len(items)} items survived")
    return survivors


# --- Hard Cap Configuration ---
HARD_CAP_MAX_ITEMS = 8  # Absolute maximum after all gates
HARD_CAP_TARGET_ITEMS = 5  # Ideal target number


def _compute_item_score(item: Dict[str, Any]) -> float:
    """
    Compute a composite quality score for ranking items when a hard cap
    is needed. Higher score = better item.

    Scoring factors:
      - CORE tier: +3.0 (vs ADJACENT: +0.0)
      - Violation count: +1.0 per violation (structural tension = interesting)
      - Tension count: +0.3 per tension
      - CONFIRMED reality: +2.0, REPORTED: +1.0, ANALYSIS: +0.5
      - LinkedIn angle quality: +1.0 if >150 chars, +0.5 if >100 chars
      - Companies named: +0.5 if any
      - Kill gate KEEP: +1.0 (vs NEEDS_REVIEW: +0.0)

    Args:
        item: Newsletter item dict with all metadata

    Returns:
        Float score for ranking
    """
    score = 0.0

    # Relevance tier — CORE items get a big bonus
    if item.get('relevance_tier') == 'CORE':
        score += 3.0

    # Axiom engagement — violations are gold, tensions are silver
    axiom_check = item.get('axiom_check', {})
    for axiom_id, axiom_data in axiom_check.items():
        if not isinstance(axiom_data, dict):
            continue
        judgment = axiom_data.get('judgment', 'N/A')
        if judgment == 'VIOLATION':
            score += 1.0
        elif judgment == 'TENSION':
            score += 0.3

    # Reality status — confirmed > reported > opinion/analysis
    reality_status = item.get('reality_status', '')
    if reality_status == 'CONFIRMED':
        score += 2.0
    elif reality_status == 'REPORTED':
        score += 1.0
    elif reality_status == 'ANALYSIS':
        score += 0.5

    # LinkedIn angle quality
    angle = item.get('linkedin_angle', '')
    if angle and angle != 'GENERIC_ANGLE':
        if len(angle) > 150:
            score += 1.0
        elif len(angle) > 100:
            score += 0.5

    # Companies named = concrete, not abstract
    if item.get('companies'):
        score += 0.5

    # Kill gate verdict
    if item.get('kill_gate') == 'KEEP':
        score += 1.0

    return score


def apply_hard_cap(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Final safety net: if all quality gates still let through more than
    HARD_CAP_MAX_ITEMS, rank by composite score and keep only the top N.

    This ensures the newsletter NEVER exceeds the target range regardless
    of how lenient earlier gates were on a particular week.

    Args:
        items: List of items that survived all prior gates

    Returns:
        Top-ranked items, capped at HARD_CAP_MAX_ITEMS
    """
    if len(items) <= HARD_CAP_MAX_ITEMS:
        return items

    # Score every item
    scored = [(item, _compute_item_score(item)) for item in items]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep top items
    survivors = []
    for item, score in scored[:HARD_CAP_MAX_ITEMS]:
        item['hard_cap_score'] = round(score, 2)
        survivors.append(item)

    # Log what got cut
    for item, score in scored[HARD_CAP_MAX_ITEMS:]:
        print(
            f"  - HARD CAP: Removed '{item.get('headline', 'N/A')}' "
            f"(score={score:.2f}, tier={item.get('relevance_tier', '?')})"
        )

    print(
        f"  - Hard cap: {len(survivors)}/{len(items)} items kept "
        f"(max={HARD_CAP_MAX_ITEMS})"
    )
    return survivors


def write_to_google_sheet(
    spreadsheet_id: str,
    data: List[List[str]]
) -> str:
    """
    Writes data to a Google Sheet, appending new rows.
    """
    print(f"Attempting to write {len(data)} rows to Google Sheet ID: {spreadsheet_id}")
    try:
        service = get_sheets_service()
        range_name = 'Sheet1!A:AZ'
        
        body = {
            'values': data
        }
        result = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body=body
        ).execute()
        
        updated_cells = result.get('updates', {}).get('updatedCells')
        print(f"Successfully wrote {updated_cells} cells to Google Sheet.")
        return f"Successfully wrote {updated_cells} cells to Google Sheet."
    except HttpError as error:
        print(f"ERROR writing to Google Sheet: {error}")
        return f"Failed to write to Google Sheet: {error}"
    except Exception as e:
        print(f"UNEXPECTED ERROR writing to Google Sheet: {e}")
        return f"Failed to write to Google Sheet due to unexpected error: {e}"


def has_headers_in_sheet(spreadsheet_id: str) -> bool:
    """Helper to check if the first row of the sheet likely contains the current expected headers."""
    # Get the expected headers based on current configuration
    expected_headers = get_enhanced_headers_with_clustering()
    
    try:
        service = get_sheets_service()
        # Dynamically determine range based on expected header count
        col_count = len(expected_headers)
        if col_count <= 26:
            end_column = chr(ord('A') + col_count - 1)
        else:
            end_column = chr(ord('A') + (col_count - 1) // 26 - 1) + chr(ord('A') + (col_count - 1) % 26)
        range_name = f'Sheet1!A1:{end_column}1'
        
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name
        ).execute()
        values = result.get('values', [])
        
        if not values or not values[0]:
            return False
            
        found_headers = [h.strip().lower() for h in values[0][:len(expected_headers)]]
        expected_headers_lower = [h.lower() for h in expected_headers]
        
        return found_headers == expected_headers_lower
    except HttpError as error:
        print(f"Error checking sheet for headers: {error}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred checking sheet headers: {e}")
        return False


def apply_clustering_to_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply clustering to extracted newsletter items.
    
    Args:
        items: List of extracted newsletter items
        
    Returns:
        Dictionary with clustering results and enhanced items
    """
    if not CLUSTERING_AVAILABLE:
        print("WARNING: Clustering not available, skipping clustering step")
        return {
            "items": items,
            "clustering_applied": False,
            "error": "Clustering dependencies not available"
        }
    
    if not ENABLE_CLUSTERING:
        print("INFO: Clustering disabled via configuration")
        return {
            "items": items,
            "clustering_applied": False,
            "reason": "Clustering disabled in configuration"
        }
    
    if len(items) < 3:
        print(f"INFO: Too few items ({len(items)}) for meaningful clustering")
        return {
            "items": items,
            "clustering_applied": False,
            "reason": "Insufficient items for clustering"
        }
    
    try:
        print(f"Applying {CLUSTERING_ALGORITHM} clustering to {len(items)} items...")

        # Initialize clustering orchestrator with tighter segmentation
        config = ClusteringConfig()
        config.default_algorithm = CLUSTERING_ALGORITHM
        # Ensure granular clusters: small min size, capped max size
        config.hdbscan.min_cluster_size = 2
        config.hdbscan.max_cluster_size = 25
        config.hdbscan.cluster_selection_method = "leaf"
        # For hierarchical: target reasonable cluster count
        target_n = max(len(items) // 8, 3)  # ~8 items per cluster
        config.hierarchical.n_clusters = min(target_n, 20)
        orchestrator = ClusteringOrchestrator(config)

        # Perform clustering
        clustering_result = orchestrator.cluster_items(
            items=items,
            text_field="short_description",
            algorithm=CLUSTERING_ALGORITHM,
            validate_results=True
        )

        # Post-clustering validation: reject oversized clusters
        clustered_items = clustering_result["items"]
        cluster_counts = {}
        for ci in clustered_items:
            cid = ci.get('cluster_id', -1)
            if cid >= 0:
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        oversized = [cid for cid, cnt in cluster_counts.items() if cnt > 25]
        if oversized:
            print(f"  - WARNING: {len(oversized)} oversized cluster(s) detected: "
                  f"{', '.join(f'Cluster {c}={cluster_counts[c]} items' for c in oversized)}. "
                  f"Demoting oversized cluster members to noise.")
            for ci in clustered_items:
                if ci.get('cluster_id', -1) in oversized:
                    ci['cluster_id'] = -1
                    ci['is_noise'] = True
                    ci['cluster_probability'] = 0.0

        print(f"Clustering completed: {clustering_result['total_clusters']} clusters, "
              f"{clustering_result['noise_items']} noise items")

        return {
            "items": clustered_items,
            "clustering_applied": True,
            "clustering_result": clustering_result,
            "performance_stats": orchestrator.get_performance_stats()
        }

    except Exception as e:
        print(f"ERROR: Clustering failed: {e}")
        return {
            "items": items,
            "clustering_applied": False,
            "error": str(e)
        }


def get_enhanced_headers_with_clustering() -> List[str]:
    """Get enhanced headers that include axiom, regional, and clustering metadata."""
    base_headers = [
        "Story Master Headline", "Headline", "Short description",
        "Source", "Date", "Companies", "Technologies"
    ]

    # Add axiom headers if enabled
    if AXIOM_ENABLED and axiom_config:
        axiom_headers = [
            "Reality Status", "Reality Reason", "Violation Count",
            "Top Violations", "Minimal Repairs"
        ]
        base_headers.extend(axiom_headers)

    # Add regional and LinkedIn headers (always present in new schema)
    regional_headers = [
        "Implication: US", "Implication: EU",
        "Implication: China", "Implication: Rest of Asia",
        "LinkedIn Angle"
    ]
    base_headers.extend(regional_headers)

    # Add relevance tier and quality gate headers
    base_headers.extend(["Relevance Tier", "Quality Gate"])

    # Add clustering headers if enabled
    if ENABLE_CLUSTERING and CLUSTERING_AVAILABLE:
        clustering_headers = [
            "Cluster ID", "Cluster Size", "Is Noise",
            "Cluster Probability", "Representative Items",
            "Cluster Synthesis", "Cluster Thesis"
        ]
        base_headers.extend(clustering_headers)

    # Add kill gate headers
    base_headers.extend(["Kill Gate", "Kill Gate Reason"])

    return base_headers


def prepare_sheet_row_with_clustering(
    item: Dict[str, Any],
    cluster_summaries: List[Dict[str, Any]] = None
) -> List[str]:
    """Prepare a sheet row with axiom, regional, and clustering metadata."""
    # Base row data
    master_headline = item.get('master_headline', item.get('headline', 'N/A'))
    headline = item.get('headline', 'N/A')
    short_description = item.get('short_description', 'N/A')
    source = item.get('source', 'N/A')
    date = item.get('date', 'N/A')
    companies = ", ".join(item.get('companies', []))
    technologies = ", ".join(item.get('technologies', []))

    base_row = [
        master_headline, headline, short_description, source,
        date, companies, technologies
    ]

    # Add axiom fields if enabled
    if AXIOM_ENABLED and axiom_config:
        reality_status = item.get('reality_status', 'N/A')
        reality_reason = item.get('reality_reason', 'N/A')
        violation_count = item.get('violation_count', 0)
        top_violations = item.get('top_violations_formatted', 'None')
        minimal_repairs = item.get('minimal_repairs_formatted', 'None')

        base_row.extend([
            reality_status, reality_reason, str(violation_count),
            top_violations, minimal_repairs
        ])

    # Add regional implications and LinkedIn angle
    base_row.extend([
        item.get('region_us', ''),
        item.get('region_eu', ''),
        item.get('region_cn', ''),
        item.get('region_asia', ''),
        item.get('linkedin_angle', '')
    ])

    # Add relevance tier and quality gate fields
    base_row.append(item.get('relevance_tier', 'N/A'))
    base_row.append(item.get('quality_gate', 'N/A'))

    # Add clustering metadata if available
    if ENABLE_CLUSTERING and CLUSTERING_AVAILABLE and 'cluster_id' in item:
        cluster_id = item.get('cluster_id', -1)
        is_noise = item.get('is_noise', False)
        cluster_probability = item.get('cluster_probability', 'N/A')

        # Find cluster size and representative items
        cluster_size = 1
        representative_items = ""

        if cluster_summaries and cluster_id != -1:
            for summary in cluster_summaries:
                if summary.get('cluster_id') == cluster_id:
                    cluster_size = summary.get('size', 1)
                    rep_items = summary.get('representative_items', [])
                    representative_items = "; ".join([
                        rep.get('headline', '')[:50] + "..."
                        for rep in rep_items[:2]
                    ])
                    break

        cluster_synthesis = item.get('cluster_synthesis')
        cluster_thesis = item.get('cluster_thesis', '')

        clustering_row = [
            str(cluster_id) if cluster_id != -1 else "NOISE",
            str(cluster_size),
            "Yes" if is_noise else "No",
            f"{cluster_probability:.3f}" if isinstance(cluster_probability, (int, float)) else str(cluster_probability),
            representative_items,
            str(cluster_synthesis) if cluster_synthesis is not None else "N/A",
            cluster_thesis
        ]

        base_row.extend(clustering_row)

    # Add kill gate fields
    base_row.extend([
        item.get('kill_gate', 'N/A'),
        item.get('kill_gate_reason', '')
    ])

    return base_row


# --- Main Execution Block ---
def main():
    """
    Main function to orchestrate the newsletter aggregation process.
    """
    print("Starting newsletter aggregation process...")
    
    if ENABLE_CLUSTERING and CLUSTERING_AVAILABLE:
        print(f"Clustering enabled using {CLUSTERING_ALGORITHM} algorithm")
    elif ENABLE_CLUSTERING and not CLUSTERING_AVAILABLE:
        print("WARNING: Clustering enabled but dependencies not available")
    else:
        print("Clustering disabled")

    all_items_for_sheet = []

    # Step 1: Fetch newsletters
    print(f"Calling fetch_newsletters directly with label: '{GMAIL_LABEL}'")
    newsletters = fetch_newsletters(gmail_label_name=GMAIL_LABEL, days_back=7)
    print(f"Fetched {len(newsletters)} newsletters.")
    for nl in newsletters:
        print(f"  - Fetched: Subject='{nl['subject']}', Sender='{nl['sender']}'")

    if not newsletters:
        print("No newsletters found to process. Exiting script.")
        return "No newsletters found to process."

    for nl in newsletters:
        current_original_subject = nl['subject']
        current_original_sender = nl['sender']
        current_processing_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Processing newsletter: '{current_original_subject}' from '{current_original_sender}'")
        
        # Step 2: Extract and clean text
        cleaned_text = extract_clean_text(html_content=nl['body_html'])
        print(f"  - Cleaned text length: {len(cleaned_text)} characters.")
        if len(cleaned_text) < 2000:
            print(f"  - Cleaned text (first 500 chars):\n---\n{cleaned_text[:500]}\n---")

        # Step 3: Summarize and extract items
        summary_result = summarize_and_extract_topics(
            text_content=cleaned_text,
            interests=YOUR_INTERESTS,
            original_subject=current_original_subject,
            original_sender=current_original_sender
        )
        
        relevance = summary_result.get('relevance_score', 0)
        extracted_items_from_llm = summary_result.get('extracted_items', [])
        brief_summary_from_llm = summary_result.get('newsletter_summary_brief', 'N/A')

        print(f"  - Summary Result for '{current_original_subject}': Relevance={relevance}, Brief Summary='{brief_summary_from_llm}'")
        print(f"  - Extracted Items Count (from LLM): {len(extracted_items_from_llm)}")

        if relevance >= 50 and extracted_items_from_llm:
            for item in extracted_items_from_llm:
                # Add source fallback logic
                source_from_llm = item.get('source')
                if source_from_llm and source_from_llm != 'N/A':
                    item['source'] = source_from_llm
                elif current_original_sender and '<' in current_original_sender:
                    item['source'] = current_original_sender.split('<')[0].strip()
                else:
                    item['source'] = current_original_sender if current_original_sender else 'N/A'

                # Ensure date is set
                if not item.get('date') or item.get('date') == 'N/A':
                    item['date'] = current_processing_date

                # Classify relevance tier
                item['relevance_tier'] = classify_relevance_tier(item)

                # Store the complete item dictionary (not a row array)
                all_items_for_sheet.append(item)

            print(f"  - Added {len(extracted_items_from_llm)} relevant items from '{current_original_subject}' to master list.")
        else:
            print(f"  - Skipping newsletter '{current_original_subject}' (Relevance < 50 or no valid items extracted by LLM).")

    # --- Gate 0: Deduplication by master_headline ---
    if all_items_for_sheet:
        pre_dedup_count = len(all_items_for_sheet)
        all_items_for_sheet = deduplicate_by_master_headline(all_items_for_sheet)
        if len(all_items_for_sheet) < pre_dedup_count:
            print(f"Deduplication: {len(all_items_for_sheet)}/{pre_dedup_count} unique items")

    # --- Gate 1: Post-Axiom Quality Filter ---
    if all_items_for_sheet:
        pre_filter_count = len(all_items_for_sheet)
        all_items_for_sheet = apply_axiom_quality_filter(all_items_for_sheet)
        print(f"Post-axiom quality filter: {len(all_items_for_sheet)}/{pre_filter_count} items passed")

    # --- Apply Clustering (if enabled and items available) ---
    clustering_result = None
    cluster_summaries = []
    if all_items_for_sheet:
        # all_items_for_sheet already contains item dictionaries, use them directly
        clustering_result = apply_clustering_to_items(all_items_for_sheet)

        if clustering_result.get('clustering_applied', False):
            print(f"Clustering applied successfully: "
                  f"{clustering_result['clustering_result']['total_clusters']} clusters")

            # Update items with clustering metadata
            all_items_for_sheet = clustering_result['items']
            cluster_summaries = clustering_result['clustering_result'].get('cluster_summaries', [])

            # --- Gate 2: LLM Cluster Validation ---
            all_items_for_sheet = validate_clusters_with_llm(
                all_items_for_sheet, clustering_result
            )

    # --- Gate 3a: Pre-Kill Filter (deterministic, no LLM call) ---
    if all_items_for_sheet:
        pre_filter_count = len(all_items_for_sheet)
        all_items_for_sheet = apply_pre_kill_filter(all_items_for_sheet)
        if len(all_items_for_sheet) < pre_filter_count:
            print(f"Pre-kill filter: {len(all_items_for_sheet)}/{pre_filter_count} items passed")

    # --- Gate 3b: Final Kill Gate (LLM-based) ---
    if all_items_for_sheet:
        pre_kill_count = len(all_items_for_sheet)
        all_items_for_sheet = apply_kill_gate(all_items_for_sheet)
        print(f"Kill gate: {len(all_items_for_sheet)}/{pre_kill_count} items survived")

    # --- Gate 4: Hard Cap (deterministic, score-based) ---
    if all_items_for_sheet:
        pre_cap_count = len(all_items_for_sheet)
        all_items_for_sheet = apply_hard_cap(all_items_for_sheet)
        if len(all_items_for_sheet) < pre_cap_count:
            print(f"Hard cap: {len(all_items_for_sheet)}/{pre_cap_count} items kept (max={HARD_CAP_MAX_ITEMS})")

    # --- Prepare sheet rows ---
    if all_items_for_sheet:
        sheet_rows = []
        for item in all_items_for_sheet:
            row = prepare_sheet_row_with_clustering(item, cluster_summaries)
            sheet_rows.append(row)

        print(f"Total items for sheet after all gates: {len(sheet_rows)}")

        # Get headers (with all gate metadata)
        headers = get_enhanced_headers_with_clustering()

        data_to_write = []
        if not has_headers_in_sheet(GOOGLE_SHEET_ID):
            print("Adding headers to Google Sheet...")
            data_to_write.append(headers)

        data_to_write.extend(sheet_rows)

        write_status = write_to_google_sheet(
            spreadsheet_id=GOOGLE_SHEET_ID,
            data=data_to_write
        )

        # Build status message
        status_message = f"Newsletter processing complete. {write_status}."
        if clustering_result and clustering_result.get('clustering_applied', False):
            cr = clustering_result['clustering_result']
            status_message += f" Clustering: {cr['total_clusters']} clusters, {cr['noise_items']} noise items."
        status_message += " Check your Google Sheet."

        print(f"Final newsletter processing status: {status_message}")
        return status_message
    else:
        print("No relevant items survived all quality gates.")
        return "No items passed all quality gates (relevance, axiom quality, kill gate)."
        

if __name__ == "__main__":
    main()