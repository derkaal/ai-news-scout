"""
Axiom Checker Prompt Templates

This module provides prompt templates for the axiom-based analysis system,
focused on agentic shopping implications with regional analysis.
"""


def get_axiom_analysis_prompt(axioms_text: str, final_text: str) -> str:
    """
    Generate the axiom analysis prompt for newsletter items.

    Args:
        axioms_text: Formatted text of the 12 axioms
        final_text: The newsletter content to analyze

    Returns:
        Complete prompt string for axiom analysis
    """
    return f"""# Agentic Shopping Axiom Checker (Brief Scan)

## ROLE
You are an internal evaluation agent that performs a **brief structural analysis** of newsletter items through the lens of **agentic shopping** — AI agents that discover, compare, negotiate, and purchase products/services on behalf of consumers and businesses.

You do NOT write an opinion piece.
You do NOT summarize the article.
You do NOT expand into speculation.

Your job is threefold:
1. Check whether the item's implied claims or recommendations **violate or stress any of the 12 axioms**
2. Assess **regional implications** where they differ significantly across US, EU, China, and rest-of-Asia
3. Suggest a **LinkedIn content angle** for a business audience interested in agentic AI

## RELEVANCE FILTER (HARD GATE)
Only extract items that are **directly relevant to agentic shopping, agentic commerce, or AI-driven purchasing/retail**. This includes:
- AI agents acting on behalf of consumers or businesses in commercial transactions
- Platform/marketplace changes that affect how AI agents interact with commerce
- Regulatory moves that constrain or enable agentic commerce
- Infrastructure (payments, identity, logistics) that agentic shopping depends on
- AI capabilities (tool use, planning, negotiation) that enable agentic commerce
- Competitive moves by companies building agentic shopping systems

Items about general AI, LLMs, or tech that have NO clear connection to commerce/shopping/retail/purchasing MUST be excluded. Return `[]` if nothing qualifies.

## CONTEXT
Agentic shopping is unfolding differently across regions:
- **US**: Platform-dominated; Big Tech controls agent infrastructure; light-touch regulation; consumer data as competitive moat; emerging antitrust but slow
- **EU**: Regulatory-first; AI Act + DMA + DSA as structural constraints; interoperability mandates; consumer protection tradition; Brussels Effect as leverage
- **China**: State-directed super-apps as agent infrastructure; social credit integration; domestic data sovereignty; massive scale; government-aligned AI
- **Rest of Asia**: Mobile-first commerce; super-app models (Grab, GoTo); diverse regulation; cross-border complexity; rapid adoption

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
- Output: "RUMOR -> Needs verification; no structural judgment."

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
1) Institutions matter more than intent — TENSION: platform terms override stated consumer-first goals in agent delegation.

## STEP 2 — VIOLATION SUMMARY (ONLY IF NEEDED)
If any axiom is marked VIOLATION, provide:

- **Top Violations (max 3):** axiom number + 1 sentence why
- **Minimal Repair (max 2 bullets):** the smallest design change that would remove the violation
  - must be a mechanism, constraint, incentive, or governance change
  - must not be "communicate better" or "train people more"

## STEP 3 — REGIONAL IMPLICATIONS
Assess how this item plays out differently across regions. Only include a region if there is a **significant difference** worth noting. Skip regions where the implication is generic or identical.

For each relevant region, provide a 1-sentence implication (max 25 words).

## STEP 4 — LINKEDIN ANGLE
Suggest one LinkedIn post angle for a business audience with strong AI interest. The angle should:
- Be opinionated and mechanism-focused (not "this is interesting")
- Connect the news to a structural insight about agentic shopping
- Sound like the author understands the tech but speaks business
- Be max 2 sentences

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
      "10": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "11": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}},
      "12": {{"judgment": "ALIGNED|TENSION|VIOLATION|N/A", "reason": "max 18 words"}}
    }},
    "violations": {{
      "count": 0,
      "top_violations": ["Axiom X: reason", "Axiom Y: reason"],
      "minimal_repair": ["Repair suggestion 1", "Repair suggestion 2"]
    }},
    "regional_implications": {{
      "US": "1-sentence implication or null if not significantly different",
      "EU": "1-sentence implication or null if not significantly different",
      "CN": "1-sentence implication or null if not significantly different",
      "ASIA": "1-sentence implication or null if not significantly different"
    }},
    "linkedin_angle": "1-2 sentence opinionated post angle for business audience"
  }}
]

## RULES
- Be unsentimental and mechanism-focused.
- No moral language ("ethical", "should", "responsible") unless quoting.
- No long explanations. This is a screening gate.
- If insufficient detail, mark affected axioms as TENSION with "insufficient detail".
- Never invent facts. Never assume the event occurred.
- HARD FILTER: Only include items with a clear agentic shopping / agentic commerce connection. General AI news without commerce relevance must be excluded.
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
    return f"""Your task is to act as an expert content curator focused on **agentic shopping and agentic commerce**.
Analyze the following text content from a newsletter. Strictly identify news items, developments, or insights that are relevant to AI agents acting in commercial/shopping/retail/purchasing contexts.

User's Key Interests and Focus Areas: "{interests}".

For each relevant news item identified in the 'Text for Analysis':
- Extract a concise, factual **Master Headline** (3-7 words, highly objective, universal identifier for the news event, no commentary or spin). This will be used to group similar stories across different newsletters.
- Extract a concise, factual **Headline** (3-10 words, objective, no commentary).
- Write a **Short description** (1-2 sentences) summarizing the news item.
- Identify the **Source** (e.g., "TLDR AI", "Marie Haynes").
- Determine the **Date** (e.g., "July 2, 2025" or "This Week").
- Extract a list of **Companies** (max 3, e.g., "Google", "OpenAI") directly involved or mentioned in this news item. If none, return empty list `[]`.
- Extract a list of **Technologies** (max 3, e.g., "Agentic AI", "LLMs", "Commerce AI") directly relevant to this news item. If none, return empty list `[]`.
- Assess **Regional Implications** for US, EU, China, and Rest of Asia — only where significantly different. Use null for regions with no distinct implication.
- Suggest a **LinkedIn Angle** — 1-2 sentence opinionated post angle for a business audience interested in agentic AI.

HARD FILTER: Only include items with a clear connection to agentic shopping, agentic commerce, or AI-driven purchasing/retail. General AI news without commerce relevance must be excluded.

Your output MUST be a structured JSON array of objects. DO NOT include any other text or markdown outside the JSON array.

Example of expected JSON array structure:
[
  {{
    "master_headline": "Amazon Launches Agent Shopping API",
    "headline": "Amazon opens product API for AI shopping agents",
    "short_description": "Amazon released an API allowing third-party AI agents to browse, compare, and purchase products on behalf of consumers.",
    "source": "TLDR AI",
    "date": "This Week",
    "companies": ["Amazon"],
    "technologies": ["Agentic AI", "Commerce API"],
    "regional_implications": {{
      "US": "Entrenches Amazon's platform dominance as the default agent-commerce rail.",
      "EU": "DMA interoperability mandates may force open access to competing agents.",
      "CN": null,
      "ASIA": null
    }},
    "linkedin_angle": "Amazon just made every shopping agent dependent on its infrastructure. The question isn't whether agents will shop — it's who controls the rails they shop on."
  }}
]

If NO relevant content is found in the 'Text for Analysis' based on the specified interests, return an EMPTY JSON array: `[]`.

Text for Analysis:
---
{final_text}
---
"""
