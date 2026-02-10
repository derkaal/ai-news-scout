# Incremental Writing to Google Sheets - Implementation Guide

## Problem
Currently, the newsletter agent processes all newsletters and only writes to Google Sheets at the very end. If the process fails after 1 hour, all work is lost.

## Solution
Implement incremental writing where each newsletter's items are written to the sheet immediately after processing.

## Implementation Steps

### 1. Add Configuration Variable
Add to `.env`:
```
INCREMENTAL_WRITE=true
```

### 2. Modify Main Function Flow

**Current Flow:**
```python
all_items_for_sheet = []
for newsletter in newsletters:
    # Process newsletter
    items = extract_items(newsletter)
    all_items_for_sheet.extend(items)  # Accumulate all items

# Write everything at the end
write_to_sheet(all_items_for_sheet)
```

**New Flow:**
```python
# Initialize sheet with headers once at start
initialize_sheet_with_headers()

for newsletter in newsletters:
    # Process newsletter
    items = extract_items(newsletter)
    
    # Write immediately if incremental mode enabled
    if INCREMENTAL_WRITE:
        write_items_to_sheet(items)  # Append to sheet
        print(f"✓ Wrote {len(items)} items to sheet")
    else:
        all_items_for_sheet.extend(items)

# If not incremental, write at end (legacy behavior)
if not INCREMENTAL_WRITE and all_items_for_sheet:
    write_to_sheet(all_items_for_sheet)
```

### 3. Create Helper Functions

#### A. Initialize Sheet with Headers
```python
def initialize_sheet_with_headers(sheet_id: str):
    """Write headers to sheet at the start."""
    service = get_sheets_service()
    headers = get_enhanced_headers_with_clustering()
    
    # Clear existing content
    service.spreadsheets().values().clear(
        spreadsheetId=sheet_id,
        range='Sheet1!A1:Z'
    ).execute()
    
    # Write headers
    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range='Sheet1!A1',
        valueInputOption='RAW',
        body={'values': [headers]}
    ).execute()
    
    print("✓ Sheet initialized with headers")
```

#### B. Append Items to Sheet
```python
def append_items_to_sheet(sheet_id: str, items: List[Dict[str, Any]]):
    """Append items to sheet immediately."""
    if not items:
        return
    
    service = get_sheets_service()
    
    # Convert items to rows
    rows = []
    for item in items:
        row = prepare_sheet_row_with_clustering(item, cluster_summaries=None)
        rows.append(row)
    
    # Append to sheet
    service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range='Sheet1!A2',  # Start after headers
        valueInputOption='RAW',
        insertDataOption='INSERT_ROWS',
        body={'values': rows}
    ).execute()
    
    print(f"✓ Appended {len(rows)} rows to sheet")
```

### 4. Update Main Function

**Location:** Around line 900-940 in `agent.py`

**Find this section:**
```python
# Store the complete item dictionary (not a row array)
all_items_for_sheet.append(item)

print(f"  - Added {len(extracted_items_from_llm)} relevant items...")
```

**Replace with:**
```python
# Store items
all_items_for_sheet.extend(extracted_items_from_llm)

# Write immediately if incremental mode
if INCREMENTAL_WRITE and GOOGLE_SHEET_ID:
    try:
        append_items_to_sheet(GOOGLE_SHEET_ID, extracted_items_from_llm)
        print(f"  ✓ Wrote {len(extracted_items_from_llm)} items to sheet immediately")
    except Exception as e:
        print(f"  ✗ Failed to write items: {e}")
        print(f"  - Items saved in memory, will retry at end")

print(f"  - Added {len(extracted_items_from_llm)} relevant items...")
```

### 5. Initialize Sheet at Start

**Location:** Around line 870-880 in `main()` function

**Add after fetching newsletters:**
```python
newsletters = fetch_newsletters(GMAIL_LABEL, days_back=7)

# Initialize sheet if incremental writing enabled
if INCREMENTAL_WRITE and GOOGLE_SHEET_ID and newsletters:
    try:
        initialize_sheet_with_headers(GOOGLE_SHEET_ID)
    except Exception as e:
        print(f"WARNING: Failed to initialize sheet: {e}")
        print("Falling back to batch writing at end")
        INCREMENTAL_WRITE = False
```

## Benefits

1. **No Data Loss**: If process fails after 45 minutes, you still have 45 minutes of data
2. **Progress Visibility**: See items appearing in sheet in real-time
3. **Early Error Detection**: Catch sheet writing errors early, not after 1 hour
4. **Memory Efficient**: Don't need to hold all items in memory
5. **Backward Compatible**: Can disable with `INCREMENTAL_WRITE=false`

## Clustering Consideration

**Issue**: Clustering requires all items to be processed first.

**Solution**: 
- In incremental mode, write items without clustering metadata
- After all newsletters processed, run clustering on all items
- Update sheet with clustering columns in a second pass

**Alternative**: Disable clustering when using incremental mode:
```python
if INCREMENTAL_WRITE:
    ENABLE_CLUSTERING = False
    print("Note: Clustering disabled in incremental write mode")
```

## Testing

1. Set `INCREMENTAL_WRITE=true` in `.env`
2. Run with 2-3 newsletters
3. Watch items appear in sheet after each newsletter
4. Simulate failure (Ctrl+C) after 2nd newsletter
5. Verify 2 newsletters worth of data is in sheet

## Rollback

If issues occur, simply set:
```
INCREMENTAL_WRITE=false
```

This reverts to the original batch-write-at-end behavior.
