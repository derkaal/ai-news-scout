# Axiom Checker Migration Guide

## Overview
This document describes the migration from sovereignty-based analysis to axiom-based structural checking, including cost optimization strategies.

## What Has Been Completed

### 1. New Configuration Files Created

#### `newsletter_agent_core/config/axiom_config.json`
- Defines 10 non-negotiable axioms for European context
- Includes reality gate definitions (CONFIRMED, REPORTED, RUMOR, OPINION, ANALYSIS)
- Defines filtering modes (strict, balanced, exploratory)

#### `newsletter_agent_core/config/axiom_config.py`
- Python class to load and manage axiom configuration
- Similar structure to `SovereigntyConfig` for easy migration
- Methods: `get_axioms()`, `get_axiom_by_id()`, `get_prompt_text()`, etc.

#### `newsletter_agent_core/axiom_prompts.py`
- Modular prompt templates for axiom analysis
- `get_axiom_analysis_prompt()` - Full axiom checker prompt
- `get_simple_extraction_prompt()` - Fallback when axiom checker disabled

### 2. Environment Configuration Updated

#### `.env` changes:
```bash
# OLD (Removed):
SOVEREIGNTY_ENABLED=true
SOVEREIGNTY_FILTERING_MODE=balanced
SOVEREIGNTY_INCLUDE_SCORES=true
SOVEREIGNTY_LEGACY_CRM_ANGLE=true

# NEW (Added):
AXIOM_ENABLED=true
AXIOM_FILTERING_MODE=balanced
AXIOM_INCLUDE_REPAIRS=true

# Cost Optimization (NEW):
USE_CHEAP_MODEL_FOR_SCREENING=true
USE_EXPENSIVE_MODEL_FOR_ANALYSIS=false
```

### 3. Agent.py Partial Updates

#### Completed:
- ✅ Imported `AxiomConfig` instead of `SovereigntyConfig`
- ✅ Imported axiom prompt functions
- ✅ Updated configuration variables (AXIOM_* instead of SOVEREIGNTY_*)
- ✅ Implemented dual-model system:
  - `cheap_model` = gemini-2.0-flash-exp (for screening)
  - `expensive_model` = gemini-2.5-flash (optional, for detailed analysis)
- ✅ Initialized axiom configuration on startup

## What Needs To Be Completed

### 1. Replace Prompt Generation Logic (Line ~487 onwards)

**Current code** (lines 487-630+):
```python
if SOVEREIGNTY_ENABLED and sovereignty_config:
    # Sovereignty-focused prompt
    theses_text = sovereignty_config.get_prompt_text()
    # ... long sovereignty prompt ...
else:
    # Legacy CRM-focused prompt
    # ... legacy prompt ...
```

**Should be replaced with**:
```python
# Select model based on cost optimization
model_to_use = cheap_model if USE_CHEAP_MODEL_FOR_SCREENING else global_model_for_tools

# Generate prompt based on axiom configuration
if AXIOM_ENABLED and axiom_config:
    axioms_text = axiom_config.get_prompt_text()
    prompt_template = get_axiom_analysis_prompt(axioms_text, final_text_for_llm)
else:
    prompt_template = get_simple_extraction_prompt(interests, final_text_for_llm)
```

### 2. Update JSON Parsing Logic (Line ~635 onwards)

**Current structure expects**:
```json
{
  "master_headline": "...",
  "headline": "...",
  "short_description": "...",
  "source": "...",
  "date": "...",
  "companies": [],
  "technologies": [],
  "aligned_theses": [1, 3, 6],
  "sovereignty_angle": "...",
  "sovereignty_relevance_score": 9
}
```

**New axiom structure**:
```json
{
  "master_headline": "...",
  "headline": "...",
  "short_description": "...",
  "source": "...",
  "date": "...",
  "companies": [],
  "technologies": [],
  "reality_status": "CONFIRMED",
  "reality_reason": "...",
  "axiom_check": {
    "1": {"judgment": "ALIGNED", "reason": "..."},
    "2": {"judgment": "TENSION", "reason": "..."},
    ...
  },
  "violations": {
    "count": 0,
    "top_violations": [],
    "minimal_repair": []
  }
}
```

**Update required_keys** (around line 638-651):
```python
if AXIOM_ENABLED and axiom_config:
    required_keys = [
        'master_headline', 'headline', 'short_description',
        'source', 'date', 'companies', 'technologies',
        'reality_status', 'reality_reason', 'axiom_check', 'violations'
    ]
else:
    # Simple extraction mode
    required_keys = [
        'master_headline', 'headline', 'short_description',
        'source', 'date', 'companies', 'technologies'
    ]
```

### 3. Update Sheet Headers

The function `get_enhanced_headers_with_clustering()` needs to be updated to include axiom-related columns instead of sovereignty columns.

**Find and update** (search for function definition):
```python
def get_enhanced_headers_with_clustering():
    base_headers = [
        "Master Headline",
        "Headline",
        "Short Description",
        "Source",
        "Date",
        "Companies",
        "Technologies",
        "Reality Status",          # NEW
        "Reality Reason",          # NEW
        "Violation Count",         # NEW
        "Top Violations",          # NEW
        "Minimal Repairs",         # NEW
        # Remove: "Aligned Theses", "Sovereignty Angle", "Sovereignty Score"
    ]
    # ... rest of function
```

### 4. Update Data Formatting for Sheets

Around line 656-657, update how axiom data is formatted:
```python
# Format violation count and details
if 'violations' in item:
    item['violation_count'] = item['violations'].get('count', 0)
    item['top_violations_formatted'] = '; '.join(item['violations'].get('top_violations', []))
    item['minimal_repairs_formatted'] = '; '.join(item['violations'].get('minimal_repair', []))
```

## Cost Optimization Strategy

### Model Selection Logic

1. **Cheap Model (gemini-2.0-flash-exp)**:
   - Cost: ~$0.01 per 1M input tokens
   - Use for: Initial screening, simple extraction
   - When: `USE_CHEAP_MODEL_FOR_SCREENING=true`

2. **Expensive Model (gemini-2.5-flash)**:
   - Cost: ~$0.075 per 1M input tokens (7.5x more expensive)
   - Use for: Detailed axiom analysis (optional)
   - When: `USE_EXPENSIVE_MODEL_FOR_ANALYSIS=true`

### Recommended Settings

**For cost optimization** (current default):
```bash
USE_CHEAP_MODEL_FOR_SCREENING=true
USE_EXPENSIVE_MODEL_FOR_ANALYSIS=false
```
This uses only the cheap model, reducing costs by ~87%.

**For maximum quality**:
```bash
USE_CHEAP_MODEL_FOR_SCREENING=false
USE_EXPENSIVE_MODEL_FOR_ANALYSIS=true
```

## Testing Plan

1. **Unit Tests**: Test axiom configuration loading
2. **Integration Tests**: Test prompt generation
3. **End-to-End Tests**: Run full newsletter processing
4. **Cost Monitoring**: Track API costs before/after

## Migration Steps

1. ✅ Create new configuration files
2. ✅ Update environment variables
3. ⏳ Complete agent.py prompt replacement (lines 487-630)
4. ⏳ Update JSON parsing logic (lines 635-660)
5. ⏳ Update sheet headers function
6. ⏳ Test with sample newsletters
7. ⏳ Monitor costs and adjust model selection
8. ⏳ Update documentation

## Rollback Plan

If issues arise, revert `.env` to:
```bash
AXIOM_ENABLED=false
```

The system will fall back to simple extraction mode.

## Expected Cost Savings

**Before** (using gemini-2.5-flash for everything):
- ~1 EUR per analysis session

**After** (using gemini-2.0-flash-exp):
- ~0.13 EUR per analysis session
- **87% cost reduction**

## Files Modified

1. ✅ `newsletter_agent_core/config/axiom_config.json` (NEW)
2. ✅ `newsletter_agent_core/config/axiom_config.py` (NEW)
3. ✅ `newsletter_agent_core/axiom_prompts.py` (NEW)
4. ✅ `newsletter_agent_core/config/__init__.py` (UPDATED)
5. ✅ `.env` (UPDATED)
6. ⏳ `newsletter_agent_core/agent.py` (PARTIALLY UPDATED - needs completion)

## Next Steps

1. Complete the prompt replacement in `agent.py` (lines 487-630)
2. Update JSON parsing logic (lines 635-660)
3. Update sheet headers function
4. Run test suite
5. Monitor first production run for costs and quality
