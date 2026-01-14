# European Sovereignty Filtering System - Technical Architecture

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Filtering Strategy](#filtering-strategy)
4. [Data Model](#data-model)
5. [Prompt Engineering](#prompt-engineering)
6. [Configuration Management](#configuration-management)
7. [Integration Points](#integration-points)
8. [Backwards Compatibility](#backwards-compatibility)
9. [Implementation Phases](#implementation-phases)
10. [Risk Assessment](#risk-assessment)
11. [Performance Considerations](#performance-considerations)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Newsletter Agent                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────────┐                    │
│  │   Gmail API  │─────▶│  Email Fetcher   │                    │
│  └──────────────┘      └────────┬─────────┘                    │
│                                  │                               │
│                                  ▼                               │
│                        ┌─────────────────┐                      │
│                        │  HTML Extractor │                      │
│                        └────────┬────────┘                      │
│                                  │                               │
│                                  ▼                               │
│  ┌──────────────────────────────────────────────────┐          │
│  │         Sovereignty Filtering Engine              │          │
│  ├──────────────────────────────────────────────────┤          │
│  │                                                    │          │
│  │  ┌────────────────┐    ┌──────────────────────┐ │          │
│  │  │ Configuration  │───▶│  Thesis Loader       │ │          │
│  │  │ sovereignty_   │    │  (14 Theses)         │ │          │
│  │  │ theses.json    │    └──────────┬───────────┘ │          │
│  │  └────────────────┘               │             │          │
│  │                                    ▼             │          │
│  │  ┌────────────────┐    ┌──────────────────────┐ │          │
│  │  │ .env Config    │───▶│  Prompt Generator    │ │          │
│  │  │ - Mode         │    │  (Sovereignty-aware) │ │          │
│  │  │ - Threshold    │    └──────────┬───────────┘ │          │
│  │  │ - Flags        │               │             │          │
│  │  └────────────────┘               ▼             │          │
│  │                         ┌──────────────────────┐ │          │
│  │                         │  Gemini 2.5 Flash    │ │          │
│  │                         │  LLM Analysis        │ │          │
│  │                         └──────────┬───────────┘ │          │
│  │                                    │             │          │
│  │                                    ▼             │          │
│  │                         ┌──────────────────────┐ │          │
│  │                         │  Response Parser     │ │          │
│  │                         │  & Validator         │ │          │
│  │                         └──────────┬───────────┘ │          │
│  └────────────────────────────────────┼─────────────┘          │
│                                        │                         │
│                                        ▼                         │
│                         ┌──────────────────────┐                │
│                         │  Threshold Filter    │                │
│                         │  (Mode-based)        │                │
│                         └──────────┬───────────┘                │
│                                    │                             │
│                                    ▼                             │
│  ┌──────────────┐      ┌──────────────────────┐                │
│  │ Google       │◀─────│  Output Formatter    │                │
│  │ Sheets API   │      │  (Sovereignty Fields)│                │
│  └──────────────┘      └──────────────────────┘                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

                    Legacy Fallback Path
                    ═══════════════════
                    (When SOVEREIGNTY_ENABLED=false)
                           │
                           ▼
                  ┌─────────────────┐
                  │  Legacy CRM     │
                  │  Prompt         │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Standard Output│
                  │  (No Sovereignty│
                  │   Fields)       │
                  └─────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Configuration, filtering logic, and output formatting are decoupled
2. **Fail-Safe Design**: Automatic fallback to legacy mode on configuration errors
3. **Extensibility**: New theses can be added without code changes
4. **Transparency**: All filtering decisions are logged and traceable
5. **Backwards Compatibility**: Legacy mode remains fully functional

---

## Architecture Components

### 1. Configuration Layer

**File**: [`newsletter_agent_core/config/sovereignty_config.py`](newsletter_agent_core/config/sovereignty_config.py:1)

**Responsibilities**:
- Load and validate sovereignty theses from JSON
- Provide threshold values for different filtering modes
- Generate formatted prompt text for LLM
- Expose thesis metadata and configuration

**Key Classes**:

```python
class SovereigntyConfig:
    """Configuration loader for sovereignty theses"""
    
    def load(self) -> None:
        """Load configuration from JSON file"""
        
    def get_theses(self) -> List[Dict[str, Any]]:
        """Get all 14 sovereignty theses"""
        
    def get_thesis_by_id(self, thesis_id: int) -> Optional[Dict[str, Any]]:
        """Get specific thesis by ID (1-14)"""
        
    def get_prompt_text(self) -> str:
        """Generate formatted prompt text for LLM"""
        
    def get_threshold(self, mode: str = "balanced") -> float:
        """Get filtering threshold for mode"""
        
    def get_min_aligned_theses(self) -> int:
        """Get minimum required aligned theses"""
```

**Error Handling**:
- `FileNotFoundError`: Configuration file missing
- `json.JSONDecodeError`: Invalid JSON syntax
- `ValueError`: Missing required fields or invalid structure
- `RuntimeError`: Methods called before `load()`

### 2. Thesis Definition

**File**: [`newsletter_agent_core/config/sovereignty_theses.json`](newsletter_agent_core/config/sovereignty_theses.json:1)

**Structure**:
```json
{
  "version": "1.0",
  "last_updated": "2025-12-19",
  "language": "en",
  "theses": [
    {
      "id": 1,
      "title": "Delegation is Power Transfer",
      "text": "Full thesis text...",
      "category": "control",
      "keywords": ["delegation", "power", "automation"]
    }
  ],
  "filtering": {
    "default_threshold": 0.60,
    "min_aligned_theses": 1,
    "modes": {
      "strict": {"threshold": 0.75, "description": "..."},
      "balanced": {"threshold": 0.60, "description": "..."},
      "exploratory": {"threshold": 0.40, "description": "..."}
    }
  }
}
```

**Validation Rules**:
- Each thesis must have: `id`, `title`, `text`, `category`, `keywords`
- IDs must be unique integers 1-14
- Filtering section must define all three modes
- Thresholds must be between 0.0 and 1.0

### 3. Filtering Engine

**File**: [`newsletter_agent_core/agent.py`](newsletter_agent_core/agent.py:1)

**Function**: `summarize_and_extract_topics()`

**Processing Pipeline**:

```
Input Newsletter Text
        │
        ▼
┌───────────────────┐
│ Token Counting    │ ◀── Timeout protection (10s)
│ with Retry Logic  │     Max 2 retries
└────────┬──────────┘
         │
         ▼
    Too Large?
    ┌───┴───┐
   Yes      No
    │        │
    ▼        │
┌───────────────────┐
│ Chunk & Summarize │
│ (500K token chunks)│
└────────┬──────────┘
         │
         ▼        ▼
┌─────────────────────────┐
│ Generate Sovereignty    │
│ or Legacy Prompt        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ LLM Analysis            │
│ (Gemini 2.5 Flash)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Parse JSON Response     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Validate & Format       │
│ - Add formatted fields  │
│ - Validate scores       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Apply Threshold Filter  │
│ (Mode-based)            │
└────────┬────────────────┘
         │
         ▼
    Output Items
```

**Chunking Strategy**:
- **Trigger**: Content > 90% of max model tokens (1M tokens for Gemini 1.5 Flash)
- **Chunk Size**: 500K tokens (~2M characters)
- **Overlap**: 1K tokens (~4K characters)
- **Processing**: Each chunk summarized independently, then combined
- **Final Pass**: Combined summaries analyzed for sovereignty alignment

### 4. Prompt Engineering

**Sovereignty-Aware Prompt Structure**:

```python
prompt_template = f"""
You are an expert content curator specializing in European retail AI sovereignty.

Your task: Analyze newsletter content and extract news items that align with our 
European sovereignty theses for retail AI.

{theses_text}  # All 14 theses with full text

**Filtering Instructions:**
- Relevance threshold: {threshold}/10 (mode: {mode})
- Only extract items scoring ≥{threshold}/10 in sovereignty relevance
- An item must align with at least 1 thesis to be included
- Focus on: European context, data sovereignty, regulatory compliance, local innovation

**For each relevant news item:**
1. Master Headline (3-7 words)
2. Headline (3-10 words)
3. Short Description (1-2 sentences)
4. Source
5. Date
6. Companies (max 3)
7. Technologies (max 3)
8. Aligned Theses (array of integers 1-14)
9. Sovereignty Angle (2-3 sentences referencing specific theses)
10. Sovereignty Relevance Score (0-10)
11. Thesis Scores (optional, individual scores per thesis)
12. CRM Angle (optional, legacy compatibility)

**Output Format:** JSON array of objects. Include ONLY items meeting the threshold.

[Examples with proper formatting...]

**Text for Analysis:**
---
{newsletter_content}
---
"""
```

**Key Prompt Features**:
- **Explicit Theses**: All 14 theses included in full text
- **Clear Threshold**: Mode-specific threshold stated upfront
- **Structured Output**: JSON schema with required fields
- **Examples**: 3 diverse examples showing different alignment patterns
- **Filtering Instruction**: Explicit "return [] if no items meet threshold"

---

## Filtering Strategy

### Three-Tier Filtering Approach

#### Tier 1: LLM-Based Semantic Filtering
- **Location**: Within LLM prompt
- **Method**: LLM evaluates content against 14 theses
- **Output**: Only items meeting threshold are generated
- **Advantage**: Semantic understanding of sovereignty concepts

#### Tier 2: Threshold Enforcement
- **Location**: Post-LLM processing in [`agent.py`](newsletter_agent_core/agent.py:450)
- **Method**: Verify `sovereignty_relevance_score >= threshold`
- **Purpose**: Catch any LLM errors or edge cases
- **Fallback**: Remove items that shouldn't have passed

#### Tier 3: Minimum Thesis Requirement
- **Location**: Configuration validation
- **Method**: Ensure `len(aligned_theses) >= min_aligned_theses`
- **Default**: Minimum 1 thesis alignment required
- **Purpose**: Prevent generic content from passing

### Mode-Specific Behavior

```python
# Strict Mode (0.75 threshold)
# - High precision, low recall
# - Only clearly aligned content
# - Use case: Executive briefings

# Balanced Mode (0.60 threshold) - DEFAULT
# - Balanced precision/recall
# - Moderate filtering
# - Use case: Daily monitoring

# Exploratory Mode (0.40 threshold)
# - Low precision, high recall
# - Broad content discovery
# - Use case: Trend analysis
```

### Threshold Override

Users can override mode defaults:

```bash
# Override balanced mode threshold
SOVEREIGNTY_FILTERING_MODE=balanced
SOVEREIGNTY_THESIS_THRESHOLD=0.70  # Custom threshold
```

**Priority**: `SOVEREIGNTY_THESIS_THRESHOLD` > mode default

---

## Data Model

### Input Data Model

```python
# Newsletter Email
{
    "id": str,              # Gmail message ID
    "subject": str,         # Email subject line
    "sender": str,          # From address
    "date": datetime,       # Received date
    "body_html": str,       # HTML content
    "body_text": str        # Plain text content (extracted)
}
```

### Sovereignty Output Data Model

```python
# Extracted News Item (Sovereignty Mode)
{
    # Standard fields (legacy compatible)
    "master_headline": str,           # 3-7 words
    "headline": str,                  # 3-10 words
    "short_description": str,         # 1-2 sentences
    "source": str,                    # Newsletter name
    "date": str,                      # Publication date
    "companies": List[str],           # Max 3 companies
    "technologies": List[str],        # Max 3 technologies
    
    # Sovereignty-specific fields
    "aligned_theses": List[int],      # Thesis IDs (1-14)
    "aligned_theses_formatted": str,  # "1, 3, 6" (for display)
    "sovereignty_angle": str,         # 2-3 sentences
    "sovereignty_relevance_score": int,  # 0-10
    
    # Optional fields (configurable)
    "thesis_scores": Dict[str, int],  # {"1": 10, "3": 9, "6": 8}
    "potential_spin_for_marketing_sales_service_consumers": str  # Legacy CRM
}
```

### Legacy Output Data Model

```python
# Extracted News Item (Legacy Mode)
{
    "master_headline": str,
    "headline": str,
    "short_description": str,
    "source": str,
    "date": str,
    "companies": List[str],
    "technologies": List[str],
    "potential_spin_for_marketing_sales_service_consumers": str
}
```

### Configuration Data Model

```python
# Sovereignty Configuration
{
    "version": str,                   # "1.0"
    "last_updated": str,              # "2025-12-19"
    "language": str,                  # "en"
    "theses": List[Thesis],           # 14 theses
    "filtering": FilteringConfig
}

# Thesis
{
    "id": int,                        # 1-14
    "title": str,                     # Thesis title
    "text": str,                      # Full thesis text
    "category": str,                  # control, tradeoffs, etc.
    "keywords": List[str]             # Relevant keywords
}

# FilteringConfig
{
    "default_threshold": float,       # 0.60
    "min_aligned_theses": int,        # 1
    "modes": {
        "strict": ModeConfig,
        "balanced": ModeConfig,
        "exploratory": ModeConfig
    }
}

# ModeConfig
{
    "threshold": float,               # 0.40-0.75
    "description": str                # Mode description
}
```

---

## Prompt Engineering

### Prompt Design Principles

1. **Explicit Context**: All 14 theses included verbatim
2. **Clear Instructions**: Unambiguous filtering criteria
3. **Structured Output**: JSON schema with examples
4. **Threshold Transparency**: Mode and threshold stated upfront
5. **European Focus**: Explicit emphasis on European context

### Prompt Components

#### 1. Role Definition
```
You are an expert content curator specializing in European retail AI sovereignty.
```

#### 2. Task Description
```
Your task: Analyze newsletter content and extract news items that align with our 
European sovereignty theses for retail AI.
```

#### 3. Thesis Injection
```python
theses_text = sovereignty_config.get_prompt_text()
# Returns formatted string with all 14 theses
```

#### 4. Filtering Instructions
```
- Relevance threshold: {threshold}/10 (mode: {mode})
- Only extract items scoring ≥{threshold}/10 in sovereignty relevance
- An item must align with at least 1 thesis to be included
- Focus on: European context, data sovereignty, regulatory compliance, local innovation
```

#### 5. Output Schema
```
**For each relevant news item:**
1. Master Headline (3-7 words): Objective, universal identifier
2. Headline (3-10 words): Concise, factual summary
[... 10 more fields with specifications ...]
```

#### 6. Examples
Three diverse examples showing:
- High-alignment regulatory content (score: 9)
- Moderate-alignment technical content (score: 7-8)
- Lower-alignment but relevant content (score: 6)

#### 7. Content Injection
```
**Text for Analysis:**
---
{final_text_for_llm}
---
```

### Prompt Optimization Techniques

**Token Efficiency**:
- Theses loaded once, reused across all newsletters
- Examples are concise but comprehensive
- Instructions use bullet points for clarity

**Consistency**:
- Same prompt structure for all newsletters
- Deterministic thesis ordering (by ID)
- Standardized field names and formats

**Error Prevention**:
- Explicit "return [] if no items" instruction
- JSON format specified with examples
- Field types and constraints clearly stated

---

## Configuration Management

### Environment Variables

**File**: [`.env`](.env:1)

```bash
# Core sovereignty settings
SOVEREIGNTY_ENABLED=true                    # Enable/disable sovereignty filtering
SOVEREIGNTY_FILTERING_MODE=balanced         # strict|balanced|exploratory
SOVEREIGNTY_THESIS_THRESHOLD=               # Optional override (0.0-1.0)

# Output customization
SOVEREIGNTY_INCLUDE_SCORES=true             # Include individual thesis scores
SOVEREIGNTY_LEGACY_CRM_ANGLE=true           # Include legacy CRM perspective

# Other settings (unchanged)
GMAIL_LABEL=Newsletters
GOOGLE_SHEET_ID=...
YOUR_INTERESTS=AI, Agentic, Retail
GOOGLE_API_KEY=...
```

### Configuration Loading Sequence

```
1. Load .env file
   └─▶ dotenv.load_dotenv()

2. Parse environment variables
   └─▶ SOVEREIGNTY_ENABLED, MODE, etc.

3. Initialize SovereigntyConfig
   └─▶ sovereignty_config = SovereigntyConfig()

4. Load thesis configuration
   └─▶ sovereignty_config.load()
   
5. Validate configuration
   ├─▶ Check required fields
   ├─▶ Validate thesis structure
   └─▶ Verify filtering modes

6. Success or Fallback
   ├─▶ Success: Log configuration details
   └─▶ Failure: Log warning, set SOVEREIGNTY_ENABLED=False
```

### Configuration Validation

**Validation Checks**:
1. File exists at expected path
2. Valid JSON syntax
3. Required top-level fields present (`theses`, `filtering`)
4. Each thesis has required fields (`id`, `title`, `text`, `category`, `keywords`)
5. All three modes defined (`strict`, `balanced`, `exploratory`)
6. Thresholds are valid floats

**Validation Errors**:
```python
# File not found
FileNotFoundError: Configuration file not found: [path]

# Invalid JSON
json.JSONDecodeError: Invalid JSON in configuration file: [details]

# Missing fields
ValueError: Configuration must contain 'theses' field
ValueError: Thesis 5 missing fields: ['keywords']

# Invalid mode
ValueError: Unknown mode 'custom'. Available modes: ['strict', 'balanced', 'exploratory']
```

---

## Integration Points

### 1. Gmail API Integration

**Location**: [`agent.py:fetch_newsletters()`](newsletter_agent_core/agent.py:150)

**Flow**:
```
Gmail API → Fetch emails with label → Extract HTML → Clean text → Pass to filtering
```

**No Changes Required**: Sovereignty filtering is transparent to email fetching.

### 2. Google Sheets API Integration

**Location**: [`agent.py:write_to_google_sheet()`](newsletter_agent_core/agent.py:700)

**Changes**:
- Added columns: `Aligned Theses`, `Sovereignty Angle`, `Sovereignty Score`
- Optional columns: `Thesis Scores`, `CRM Angle`
- Automatic column creation if missing
- Backwards compatible with legacy sheets

**Column Mapping**:
```python
sovereignty_columns = {
    "aligned_theses_formatted": "Aligned Theses",
    "sovereignty_angle": "Sovereignty Angle",
    "sovereignty_relevance_score": "Sovereignty Score",
    "thesis_scores": "Thesis Scores",  # Optional
    "potential_spin_for_marketing_sales_service_consumers": "CRM Angle"  # Optional
}
```

### 3. LLM Integration (Gemini)

**Location**: [`agent.py:summarize_and_extract_topics()`](newsletter_agent_core/agent.py:240)

**Model**: `gemini-2.5-flash`

**Configuration**:
```python
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
global_model_for_tools = genai.GenerativeModel("gemini-2.5-flash")
```

**Token Limits**:
- Max input: 1,048,575 tokens (~4M characters)
- Target chunk: 500,000 tokens (~2M characters)
- Overlap: 1,000 tokens (~4K characters)

**Timeout Protection**:
- Token counting: 10s timeout with 2 retries
- Chunk processing: Individual timeouts per chunk
- Fallback: Character-based estimation if token counting fails

### 4. Clustering Engine Integration

**Location**: [`agent.py`](newsletter_agent_core/agent.py:27)

**Status**: Optional, independent of sovereignty filtering

**Compatibility**: Sovereignty fields are preserved through clustering pipeline.

---

## Backwards Compatibility

### Legacy Mode Support

**Trigger Conditions**:
1. `SOVEREIGNTY_ENABLED=false` explicitly set
2. Configuration file missing or invalid
3. Configuration loading fails

**Behavior**:
- Uses original CRM-focused prompt
- Outputs standard fields only
- No sovereignty analysis performed
- Existing workflows unchanged

### Transition Strategy

**Phase 1: Dual Output** (Current)
```bash
SOVEREIGNTY_ENABLED=true
SOVEREIGNTY_LEGACY_CRM_ANGLE=true  # Include both perspectives
```

**Phase 2: Sovereignty Primary**
```bash
SOVEREIGNTY_ENABLED=true
SOVEREIGNTY_LEGACY_CRM_ANGLE=false  # Sovereignty only
```

**Phase 3: Legacy Deprecation**
```bash
# Legacy mode available but not recommended
SOVEREIGNTY_ENABLED=false  # Fallback only
```

### Data Migration

**Existing Sheets**:
- New columns added automatically
- Existing data preserved
- No manual migration required

**Existing Code**:
- All legacy fields still present
- New fields are additive
- No breaking changes to existing integrations

### API Compatibility

**Function Signatures**: Unchanged
```python
# Before and After
summarize_and_extract_topics(
    text_content: str,
    interests: str,
    original_subject: str,
    original_sender: str
) -> Dict[str, Any]
```

**Return Structure**: Extended, not replaced
```python
# Legacy fields always present
{
    "extracted_items": [...],
    "relevance_score": int
}

# Sovereignty fields added to items when enabled
item["aligned_theses"] = [...]
item["sovereignty_angle"] = "..."
```

---

## Implementation Phases

### Phase 1: Foundation (Completed)

**Deliverables**:
- ✅ [`SovereigntyConfig`](newsletter_agent_core/config/sovereignty_config.py:13) class
- ✅ [`sovereignty_theses.json`](newsletter_agent_core/config/sovereignty_theses.json:1) configuration
- ✅ Configuration loading and validation
- ✅ Error handling and fallback logic

**Testing**:
- ✅ Unit tests for configuration loading
- ✅ Validation error handling
- ✅ Thesis retrieval methods

### Phase 2: Filtering Engine (Completed)

**Deliverables**:
- ✅ Sovereignty-aware prompt generation
- ✅ Mode-based threshold filtering
- ✅ Response parsing and validation
- ✅ Output field formatting

**Testing**:
- ✅ Integration tests with mock LLM
- ✅ Threshold enforcement validation
- ✅ Output format verification

### Phase 3: Integration (Completed)

**Deliverables**:
- ✅ Google Sheets column updates
- ✅ Environment variable configuration
- ✅ Logging and monitoring
- ✅ Legacy mode compatibility

**Testing**:
- ✅ End-to-end workflow tests
- ✅ Backwards compatibility verification
- ✅ Error recovery testing

### Phase 4: Documentation (Current)

**Deliverables**:
- ✅ User guide ([`SOVEREIGNTY_FILTERING.md`](SOVEREIGNTY_FILTERING.md:1))
- ✅ Technical architecture (this document)
- 🔄 API documentation
- 🔄 Deployment guide

### Phase 5: Optimization (Future)

**Planned Improvements**:
- [ ] Caching of thesis embeddings
- [ ] Batch processing optimization
- [ ] Advanced thesis weighting
- [ ] Multi-language support
- [ ] Custom thesis definitions

---

## Risk Assessment

### Technical Risks

#### Risk 1: LLM Hallucination
**Severity**: Medium  
**Probability**: Low  
**Impact**: Incorrect thesis alignment or scores

**Mitigation**:
- Explicit examples in prompt
- Structured JSON output format
- Post-processing validation
- Threshold enforcement as safety net

#### Risk 2: Configuration Corruption
**Severity**: High  
**Probability**: Low  
**Impact**: System failure or incorrect filtering

**Mitigation**:
- JSON schema validation
- Automatic fallback to legacy mode
- Configuration version tracking
- Comprehensive error logging

#### Risk 3: Performance Degradation
**Severity**: Medium  
**Probability**: Medium  
**Impact**: Slower processing, higher costs

**Mitigation**:
- Efficient chunking strategy
- Token counting with timeout
- Caching of configuration
- Monitoring of processing times

#### Risk 4: Threshold Misconfiguration
**Severity**: Low  
**Probability**: Medium  
**Impact**: Too many or too few items extracted

**Mitigation**:
- Clear mode descriptions
- Default to balanced mode
- Easy threshold adjustment
- Logging of filtering decisions

### Operational Risks

#### Risk 5: Breaking Changes to Existing Workflows
**Severity**: High  
**Probability**: Very Low  
**Impact**: Disruption to users

**Mitigation**:
- Backwards compatibility maintained
- Legacy mode always available
- Additive changes only
- Comprehensive testing

#### Risk 6: User Confusion
**Severity**: Medium  
**Probability**: Medium  
**Impact**: Incorrect configuration, suboptimal results

**Mitigation**:
- Comprehensive documentation
- Clear error messages
- Sensible defaults
- Example configurations

### Data Risks

#### Risk 7: Sensitive Data in Logs
**Severity**: Medium  
**Probability**: Low  
**Impact**: Privacy concerns

**Mitigation**:
- No newsletter content in logs
- Only metadata logged
- Configurable log levels
- Secure log storage

---

## Performance Considerations

### Token Usage

**Per Newsletter**:
- Prompt overhead: ~2,000 tokens (theses + instructions)
- Content: Variable (1K - 1M tokens)
- Response: ~500-2,000 tokens per item

**Optimization**:
- Theses loaded once, reused
- Efficient chunking for large newsletters
- Minimal prompt overhead

### Processing Time

**Typical Newsletter** (10K tokens):
- Token counting: <1s
- LLM analysis: 2-5s
- Parsing & validation: <1s
- **Total**: 3-7s

**Large Newsletter** (500K tokens):
- Token counting: 1-2s
- Chunking: 1-2s
- LLM analysis (per chunk): 5-10s × chunks
- Combining: 2-5s
- **Total**: 15-60s depending on chunks

### Cost Estimation

**Gemini 2.5 Flash Pricing** (as of Dec 2024):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

**Per Newsletter** (average 50K tokens):
- Input cost: ~$0.004
- Output cost: ~$0.001
- **Total**: ~$0.005 per newsletter

**Monthly** (100 newsletters):
- **Total**: ~$0.50/month

### Scalability

**Current Capacity**:
- Sequential processing: 1 newsletter at a time
- Rate limits: Gemini API limits apply
- Storage: Google Sheets (10M cells limit)

**Scaling Options**:
1. Parallel processing (multiple newsletters)
2. Batch API calls (if available)
3. Caching of common patterns
4. Database backend for large volumes

### Monitoring Metrics

**Key Metrics**:
- Processing time per newsletter
- Token usage per newsletter
- Items extracted per newsletter
- Filtering rate (items passed/total)
- Error rate and types
- API costs

**Logging**:
```python
print(f"Sovereignty filtering enabled (mode: {mode})")
print(f"  - Loaded {len(theses)} sovereignty theses")
print(f"  - Configuration version: {version}")
print(f"  - Newsletter '{subject}' processed in {time}s")
print(f"  - Extracted {count} items (threshold: {threshold})")
```

---

## Testing Strategy

### Unit Tests

**File**: [`test_sovereignty_filtering.py`](test_sovereignty_filtering.py:1)

**Coverage**:
- Configuration loading and validation
- Thesis retrieval methods
- Threshold calculation
- Error handling
- Metadata access

**Example**:
```python
def test_get_threshold_balanced(self):
    """Test getting threshold for balanced mode"""
    self.config.load()
    threshold = self.config.get_threshold("balanced")
    self.assertEqual(threshold, 0.60)
```

### Integration Tests

**Coverage**:
- End-to-end filtering pipeline
- LLM integration (mocked)
-