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
    get_simple_extraction_prompt
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
    "Agentic AI, AI (general), CRM technology, European AI policy, "
    "machine learning, cloud computing, data science, Python programming")

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
        cheap_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        print(
            "Configured cheap model (gemini-2.0-flash-exp) for screening"
        )
        
        # Initialize expensive model if needed
        if USE_EXPENSIVE_MODEL_FOR_ANALYSIS:
            expensive_model = genai.GenerativeModel("gemini-2.5-flash")
            print(
                "Configured expensive model (gemini-2.5-flash) "
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


def summarize_and_extract_topics(
    text_content: str,
    interests: str, # This now includes the specific focus areas
    original_subject: str,
    original_sender: str
) -> Dict[str, Any]:
    """
    Analyzes newsletter content to extract relevant headlines, descriptions, source, date,
    and a potential spin for Agentic CRM in Europe. Handles large inputs by chunking.
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
Analyze the following text from a newsletter chunk. Focus on information relevant to: "{interests}" (Agentic AI, AI general, CRM technology, European AI policy, machine learning, cloud computing, data science, Python programming).
Extract only key factual points.

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
                    'axiom_check', 'violations'
                ]
            else:
                # Simple extraction mode
                required_keys = [
                    'master_headline', 'headline', 'short_description',
                    'source', 'date', 'companies', 'technologies'
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
        range_name = 'Sheet1!A:Z'
        
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
        end_column = chr(ord('A') + len(expected_headers) - 1)  # A=0, B=1, etc.
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
        
        # Initialize clustering orchestrator
        config = ClusteringConfig()
        config.default_algorithm = CLUSTERING_ALGORITHM
        orchestrator = ClusteringOrchestrator(config)
        
        # Perform clustering
        clustering_result = orchestrator.cluster_items(
            items=items,
            text_field="short_description",
            algorithm=CLUSTERING_ALGORITHM,
            validate_results=True
        )
        
        print(f"Clustering completed: {clustering_result['total_clusters']} clusters, "
              f"{clustering_result['noise_items']} noise items")
        
        return {
            "items": clustering_result["items"],
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
    """Get enhanced headers that include axiom and clustering metadata."""
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
    else:
        # Legacy mode - just add the CRM angle
        base_headers.append("Potential spin for Marketing, Sales, Service, Consumers")
    
    # Add clustering headers if enabled
    if ENABLE_CLUSTERING and CLUSTERING_AVAILABLE:
        clustering_headers = [
            "Cluster ID", "Cluster Size", "Is Noise",
            "Cluster Probability", "Representative Items"
        ]
        return base_headers + clustering_headers
    
    return base_headers


def prepare_sheet_row_with_clustering(
    item: Dict[str, Any],
    cluster_summaries: List[Dict[str, Any]] = None
) -> List[str]:
    """Prepare a sheet row with sovereignty and clustering metadata."""
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
    else:
        # Legacy mode - add CRM spin
        spin = item.get('potential_spin_for_marketing_sales_service_consumers', 'N/A')
        base_row.append(spin)
    
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
        
        clustering_row = [
            str(cluster_id) if cluster_id != -1 else "NOISE",
            str(cluster_size),
            "Yes" if is_noise else "No",
            f"{cluster_probability:.3f}" if isinstance(cluster_probability, (int, float)) else str(cluster_probability),
            representative_items
        ]
        
        return base_row + clustering_row
    
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
                
                # Store the complete item dictionary (not a row array)
                all_items_for_sheet.append(item)
            
            print(f"  - Added {len(extracted_items_from_llm)} relevant items from '{current_original_subject}' to master list.")
        else:
            print(f"  - Skipping newsletter '{current_original_subject}' (Relevance < 50 or no valid items extracted by LLM).")

    # --- Apply Clustering (if enabled and items available) ---
    clustering_result = None
    if all_items_for_sheet:
        # Convert sheet rows back to item dictionaries for clustering
        items_for_clustering = []
        for row in all_items_for_sheet:
            # Base fields are always at the same positions
            item_dict = {
                'master_headline': row[0] if len(row) > 0 else '',
                'headline': row[1] if len(row) > 1 else '',
                'short_description': row[2] if len(row) > 2 else '',
                'source': row[3] if len(row) > 3 else '',
                'date': row[4] if len(row) > 4 else '',
                'companies': row[5].split(", ") if len(row) > 5 and row[5] else [],
                'technologies': row[6].split(", ") if len(row) > 6 and row[6] else []
            }
            
            # Add axiom or legacy fields depending on configuration
            if AXIOM_ENABLED and axiom_config:
                # Axiom fields: Reality Status, Reality Reason, Violation Count, Top Violations, Minimal Repairs
                if len(row) > 7:
                    item_dict['reality_status'] = row[7]
                if len(row) > 8:
                    item_dict['reality_reason'] = row[8]
                if len(row) > 9:
                    item_dict['violation_count'] = row[9]
                if len(row) > 10:
                    item_dict['top_violations_formatted'] = row[10]
                if len(row) > 11:
                    item_dict['minimal_repairs_formatted'] = row[11]
            else:
                # Legacy mode: CRM spin is at position 7
                if len(row) > 7:
                    item_dict['potential_spin_for_marketing_sales_service_consumers'] = row[7]
            
            items_for_clustering.append(item_dict)
        
        # Apply clustering
        clustering_result = apply_clustering_to_items(items_for_clustering)
        
        if clustering_result.get('clustering_applied', False):
            print(f"Clustering applied successfully: "
                  f"{clustering_result['clustering_result']['total_clusters']} clusters")
            
            # Update items with clustering metadata
            clustered_items = clustering_result['items']
            cluster_summaries = clustering_result['clustering_result'].get('cluster_summaries', [])
            
            # Rebuild sheet rows with clustering data
            all_items_for_sheet = []
            for item in clustered_items:
                row = prepare_sheet_row_with_clustering(item, cluster_summaries)
                all_items_for_sheet.append(row)

    # --- Final Writing to Sheet ---
    if all_items_for_sheet:
        print(f"Total relevant items collected for sheet: {len(all_items_for_sheet)}")
        
        # Get headers (with clustering metadata if enabled)
        headers = get_enhanced_headers_with_clustering()
        
        data_to_write = []
        if not has_headers_in_sheet(GOOGLE_SHEET_ID):
            print("Adding headers to Google Sheet...")
            data_to_write.append(headers)

        data_to_write.extend(all_items_for_sheet)

        write_status = write_to_google_sheet(
            spreadsheet_id=GOOGLE_SHEET_ID,
            data=data_to_write
        )
        
        # Add clustering summary to status message
        status_message = f"Newsletter processing complete. {write_status}."
        if clustering_result and clustering_result.get('clustering_applied', False):
            cr = clustering_result['clustering_result']
            status_message += f" Clustering: {cr['total_clusters']} clusters, {cr['noise_items']} noise items."
        status_message += " Check your Google Sheet."
        
        print(f"Final newsletter processing status: {status_message}")
        return status_message
    else:
        print("No relevant items extracted from any newsletters based on interests.")
        return "No relevant items found or processed based on your interests."
        

if __name__ == "__main__":
    main()