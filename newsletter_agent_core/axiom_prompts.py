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

## RELEVANCE FILTER (HARD GATE — STRICT)
Each item must pass **at least one** of these three tests:

1. **Autonomous purchasing**: AI agents making, approving, or executing purchase decisions without human-in-the-loop for each transaction
2. **Transaction authority**: Systems that delegate financial or contractual authority to AI agents — payment rails for agents, procurement delegation, agent wallets, agent-held budgets
3. **Agent-to-agent commerce**: AI agents negotiating, bidding, or transacting with other AI agents on behalf of principals

**EXPLICIT REJECTIONS — these do NOT qualify unless they specifically enable one of the three tests above:**
- General AI capabilities (better reasoning, new models, benchmarks, training breakthroughs)
- AI assistants that recommend but don't transact
- Chatbots, AI search, AI-powered customer service
- Productivity tools, copilots, AI analytics dashboards
- "AI in retail" that is just personalization, demand forecasting, or inventory optimization without agent autonomy
- Vague "agentic AI" announcements without concrete commerce/transaction mechanics

Return `[]` if nothing qualifies. When in doubt, exclude.

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

## STEP 4 — LINKEDIN ANGLE (STRICT QUALITY GATE)
Suggest one LinkedIn post angle for a business audience with strong AI interest. The angle MUST:
- Contain a **specific mechanism, prediction, or named tension** — not "this is worth watching"
- Pass the "I hadn't thought of that" test — if a reader could guess the angle from the headline alone, it's too generic
- Name a **concrete paradox, second-order effect, or structural contradiction**
- Be max 2 sentences

**REJECT these patterns** — if your angle matches any of these, replace it with the literal string `GENERIC_ANGLE`:
- "raises questions about..."
- "has implications for..."
- "could disrupt..."
- "time will tell..."
- "businesses should pay attention to..."
- "this is a sign that..."
- Any angle that merely restates the headline with "and here's why it matters"

If you cannot produce a genuinely non-obvious angle, output `"GENERIC_ANGLE"` — do not force a mediocre one.

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
- HARD FILTER: Only include items that pass at least one of the three relevance tests (autonomous purchasing, transaction authority, agent-to-agent commerce). General AI news without concrete transaction/commerce mechanics must be excluded. Prefer returning `[]` over including marginal items.
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
- Suggest a **LinkedIn Angle** — 1-2 sentence opinionated post angle that names a specific mechanism, paradox, or second-order effect. Must pass the "I hadn't thought of that" test. If you cannot produce a non-obvious angle, output `"GENERIC_ANGLE"`. Do NOT use patterns like "raises questions about...", "has implications for...", "could disrupt...", or "time will tell...".

HARD FILTER: Each item must pass at least one of these three tests:
1. Autonomous purchasing — AI agents making/executing purchase decisions without human-in-the-loop
2. Transaction authority — systems delegating financial/contractual authority to AI agents
3. Agent-to-agent commerce — AI agents negotiating/transacting with other AI agents

General AI capabilities, chatbots, AI search, copilots, recommendation engines, and "AI in retail" without agent transaction autonomy do NOT qualify. Prefer returning [] over including marginal items.

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


def get_cluster_validation_prompt(cluster_items_text: str) -> str:
    """
    Generate a prompt to validate whether a cluster of items can support
    a genuine synthesis thesis, not just a thematic list.

    Args:
        cluster_items_text: Formatted text of cluster items (headlines + descriptions)

    Returns:
        Complete prompt string for cluster validation
    """
    return f"""You are evaluating whether a cluster of newsletter items can support a **genuine synthesis thesis**.

A synthesis thesis is a non-obvious argument that connects the items and produces an insight that none of them contain individually. It is NOT a summary, not a theme label, and not "these are all about X."

## Items in this cluster:
{cluster_items_text}

## Your task:
1. Can these items support a genuine synthesis thesis — a claim that emerges from their combination but isn't stated in any single item?
2. Or are they just a list of thematically similar but intellectually independent items?

## Rules:
- A valid thesis must be **falsifiable** — someone could disagree with it
- "These items show that agentic shopping is growing" is NOT a thesis — it's an observation
- "These items reveal that platform lock-in is accelerating faster than regulatory response, creating a 12-month window where agent infrastructure becomes irreversible" IS a thesis
- If the items are too loosely connected or too similar to generate tension, return has_synthesis: false

Return ONLY valid JSON (no markdown, no explanation outside JSON):
{{
  "has_synthesis": true or false,
  "thesis": "One sentence synthesis thesis if true, empty string if false",
  "reason": "One sentence explaining why synthesis works or fails"
}}"""


def get_kill_gate_prompt(items_json: str) -> str:
    """
    Generate a prompt for the final kill gate that removes items
    that are not genuinely worth writing about.

    Args:
        items_json: JSON string of items to evaluate

    Returns:
        Complete prompt string for kill gate evaluation
    """
    return f"""You are the FINAL editorial kill gate for a highly selective weekly newsletter about **agentic commerce** — AI agents that autonomously purchase, transact, and negotiate on behalf of principals.

## YOUR MANDATE
You are the last line of defense. Earlier filters were too lenient — they let through many items that are tangentially relevant but NOT worth a busy executive's time. Your job is to be RUTHLESS.

**TARGET: KEEP only 1-2 items per batch of 5.** If none deserve to survive, return all as KILL. The final newsletter targets 3-5 items per WEEK from dozens of newsletters. Most items should die here.

**DEFAULT TO KILL.** An item must earn its KEEP by being genuinely exceptional. "Relevant to agentic commerce" is NOT enough — everything reaching you is already relevant. The question is: **is this item worth 3 minutes of a senior executive's reading time THIS WEEK?**

## KILL criteria — flag as KILL if ANY of these apply:
1. **Not transactional**: The item is about AI capabilities, platforms, or infrastructure but does NOT directly involve agents making purchases, holding wallets, executing transactions, or negotiating deals. "AI agents can now do X" is NOT agentic commerce unless X is a financial transaction.
2. **Consensus news**: Everyone in the industry already knows this. No tension, no surprise, no argument to be had.
3. **Angle failure**: The linkedin_angle is GENERIC_ANGLE, or is so obvious it could be guessed from the headline alone.
4. **So what?**: A reader working in agentic commerce would shrug. The item states a fact but reveals no structural tension, no mechanism change, no second-order effect.
5. **Derivative**: A minor update to an already-known story. Not the first time this was interesting.
6. **Aspirational / vaporware**: Plans, intentions, "could", "aims to", "exploring" — without concrete shipped product, executed transaction, or binding commitment.
7. **ADJACENT without teeth**: The item is about the broader AI/commerce ecosystem (regulation, infrastructure, platform moves) but has zero violations of core axioms. It's background noise, not signal.
8. **Announcement-only**: A company announced something but there is no evidence of execution, no user/market response, no structural implication beyond the announcement itself.

## KEEP criteria — flag as KEEP only if it passes ALL of these:
1. **Transactional core**: Directly involves agents executing purchases, holding financial authority, managing wallets, or conducting agent-to-agent negotiation — not just "AI in commerce"
2. **Novel tension**: Contains a genuine structural tension (two forces pulling opposite directions) that isn't obvious from the headline
3. **Mechanism, not announcement**: Shows a concrete mechanism change that has already happened or is verifiably shipping — not future plans
4. **Changes mental models**: A senior reader in agentic commerce would update their thinking or strategy based on this item

## Items to evaluate:
{items_json}

## OUTPUT FORMAT (STRICT)
Return ONLY a valid JSON array. No markdown, no code fences, no explanation outside the JSON.
Every string value must be on a single line — no newlines inside strings.
Escape any quotes inside strings with backslash.
Use the EXACT headline from the input — do not rephrase or truncate it.

[
  {{
    "headline": "exact headline from input",
    "verdict": "KILL",
    "reason": "One sentence, no newlines."
  }}
]"""


def get_weekly_synthesis_prompt(items_json: str) -> str:
    """
    Generate a prompt that takes the top surviving items and produces
    a 3-sentence 'Weekly Synthesis' tying them into a narrative arc.

    Args:
        items_json: JSON string of the top items (max 5) with headline,
                    short_description, relevance_tier, violation_count,
                    and linkedin_angle fields

    Returns:
        Complete prompt string for weekly synthesis generation
    """
    return f"""You are the editor-in-chief of a weekly newsletter about **agentic commerce** — AI agents that autonomously purchase, transact, and negotiate on behalf of principals.

You have just selected this week's top stories. Your task is to write a **Weekly Synthesis** — exactly 3 sentences that weave these items into a single narrative arc.

## Rules:
1. **Exactly 3 sentences.** No more, no less.
2. The first sentence establishes the week's dominant theme or tension.
3. The second sentence connects two or more items to reveal a pattern, contradiction, or acceleration that none of them show individually.
4. The third sentence poses a forward-looking implication or question that a senior executive should be thinking about next week.
5. **Be specific** — name companies, mechanisms, or tensions from the items. No generic "the landscape is evolving" filler.
6. **No bullet points, no headers, no markdown** — just 3 plain sentences.
7. Write for a reader who has NOT yet read the individual items — the synthesis should stand on its own as an executive briefing.

## This week's top items:
{items_json}

## OUTPUT FORMAT (STRICT)
Return ONLY valid JSON. No markdown, no code fences, no explanation outside the JSON.

{{
  "weekly_synthesis": "Exactly 3 sentences. No newlines inside this string.",
  "narrative_theme": "2-5 word label for this week's dominant theme"
}}"""
