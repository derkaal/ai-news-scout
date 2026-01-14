"""
Axiom Checker Prompt Templates

This module provides prompt templates for the axiom-based analysis system.
"""


def get_axiom_analysis_prompt(axioms_text: str, final_text: str) -> str:
    """
    Generate the axiom analysis prompt for newsletter items.
    
    Args:
        axioms_text: Formatted text of the 10 axioms
        final_text: The newsletter content to analyze
        
    Returns:
        Complete prompt string for axiom analysis
    """
    return f"""# European Newsletter Axiom Checker (Brief Scan)

## ROLE
You are an internal evaluation agent that performs a **brief structural sanity check** of newsletter items about AI, tech, regulation, markets, and Europe.

You do NOT write an opinion piece.
You do NOT summarize the article.
You do NOT expand into speculation.

Your job is to identify whether the item's implied claims or recommendations **violate or stress any of the 10 axioms**, and why — briefly.

## CONTEXT
Assume the European environment:
- pluralistic politics and legitimacy constraints,
- regulatory adjustment over time (not stability),
- EU competition/state-aid constraints,
- energy/material constraints and climate targets,
- vendor lock-in risk and geopolitical chokepoints.

{axioms_text}

## STEP 0 — REALITY GATE (MANDATORY)
Before applying axioms, classify the item's factual status:

Choose one:
- CONFIRMED (official announcement, primary source, executed change)
- REPORTED (reputable reporting, not official)
- RUMOR (unconfirmed claim, single source, social media)
- OPINION (argument, not an event)
- ANALYSIS (interpretation of trends)

If status is RUMOR:
- DO NOT run full axiom analysis.
- Output: "RUMOR → Needs verification; no structural judgment."

If status is OPINION or ANALYSIS:
- Run axioms against the *argument's logic*, not the world.

## STEP 1 — AXIOM CHECK (BRIEF)
For each axiom, output one of:
- ALIGNED
- TENSION
- VIOLATION
- N/A (only if truly irrelevant)

Each axiom must include **one short causal reason** (max 18 words).

Example format:
1) Legitimacy is fragile — TENSION: relies on stable public acceptance despite likely backlash.

## STEP 2 — VIOLATION SUMMARY (ONLY IF NEEDED)
If any axiom is marked VIOLATION, provide:

- **Top Violations (max 3):** axiom number + 1 sentence why
- **Minimal Repair (max 2 bullets):** the smallest design change that would remove the violation
  - must be a mechanism, constraint, incentive, or governance change
  - must not be "communicate better" or "train people more"

## OUTPUT FORMAT (MANDATORY JSON)
Return a JSON array where each object represents one news item found in the text:

[
  {{
    "master_headline": "Brief universal identifier (3-7 words)",
    "headline": "Concise factual summary (3-10 words)",
    "short_description": "1-2 sentence factual summary",
    "source": "Newsletter or publication name",
    "date": "Publication date or 'This Week'",
    "companies": ["Company1", "Company2"],
    "technologies": ["Tech1", "Tech2"],
    "reality_status": "CONFIRMED|REPORTED|RUMOR|OPINION|ANALYSIS",
    "reality_reason": "Brief reason for status classification",
    "axiom_check": {{
      "1": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "2": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "3": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "4": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "5": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "6": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "7": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "8": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "9": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "10": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}}
    }},
    "violations": {{
      "count": 0,
      "top_violations": ["Axiom X: reason", "Axiom Y: reason"],
      "minimal_repair": ["Repair suggestion 1", "Repair suggestion 2"]
    }}
  }}
]

## RULES
- Be unsentimental and mechanism-focused.
- No moral language ("ethical", "should", "responsible") unless quoting.
- No long explanations. This is a screening gate.
- If insufficient detail, mark affected axioms as TENSION with "insufficient detail".
- Never invent facts. Never assume the event occurred.
- If NO relevant items found, return: `[]`

**Text for Analysis:**
---
{final_text}
---
"""


def get_simple_extraction_prompt(interests: str, final_text: str) -> str:
    """
    Generate a simple extraction prompt (fallback when axiom checker disabled).
    
    Args:
        interests: User's interests string
        final_text: The newsletter content to analyze
        
    Returns:
        Complete prompt string for simple extraction
    """
    return f"""Your task is to act as an expert content curator.
Analyze the following text content from a newsletter. Strictly identify news items, developments, or insights that are highly relevant to the user's specific interests.

User's Key Interests and Focus Areas: "{interests}".

For each relevant news item identified in the 'Text for Analysis':
- Extract a concise, factual **Master Headline** (3-7 words, highly objective, universal identifier for the news event, no commentary or spin). This will be used to group similar stories across different newsletters.
- Extract a concise, factual **Headline** (3-10 words, objective, no commentary).
- Write a **Short description** (1-2 sentences) summarizing the news item.
- Identify the **Source** (e.g., "TLDR AI", "Marie Haynes").
- Determine the **Date** (e.g., "July 2, 2025" or "This Week").
- Extract a list of **Companies** (max 3, e.g., "Google", "OpenAI") directly involved or mentioned in this news item. If none, return empty list `[]`.
- Extract a list of **Technologies** (max 3, e.g., "Agentic AI", "LLMs", "CRM") directly relevant to this news item. If none, return empty list `[]`.

Your output MUST be a structured JSON array of objects, where each object represents one extracted news item. DO NOT include any other text or markdown outside the JSON array.

Example of expected JSON array structure:
[
  {{
    "master_headline": "AI Language Affects Human Speech",
    "headline": "AI chatbot-speak affecting English language",
    "short_description": "Linguistic styles from chatbot outputs are influencing human speech in the US.",
    "source": "Warren Ellis",
    "date": "This Week",
    "companies": [],
    "technologies": ["AI", "NLP"]
  }}
]

If NO relevant content is found in the 'Text for Analysis' based on the specified interests, return an EMPTY JSON array: `[]`.

Text for Analysis:
---
{final_text}
---
"""
