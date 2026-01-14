# European Sovereignty Filtering System - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [The 14 European Sovereignty Theses](#the-14-european-sovereignty-theses)
3. [Configuration](#configuration)
4. [Output Format](#output-format)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is Sovereignty Filtering?

The European Sovereignty Filtering System transforms the newsletter agent from a generic CRM-focused content extractor into a specialized tool for analyzing AI and retail news through the lens of **European digital sovereignty**. Instead of extracting all AI-related content, the system now filters and prioritizes content based on its alignment with 14 specific theses about AI agents, sovereignty, and retail in the European context.

### Why Was It Implemented?

European retailers and policymakers face unique challenges in the AI era:

- **Regulatory Compliance**: EU-specific regulations (GDPR, AI Act) require different approaches than US markets
- **Data Sovereignty**: European businesses need to maintain control over their data and AI systems
- **Infrastructure Independence**: Reducing dependency on non-European cloud providers and AI platforms
- **Local Innovation**: Supporting European AI development and competitive positioning

The sovereignty filtering system helps stakeholders identify news and developments that are specifically relevant to these European concerns.

### Key Benefits for European Retailers

- **Focused Intelligence**: Only receive content aligned with European sovereignty priorities
- **Strategic Insights**: Understand how developments impact European retail competitiveness
- **Compliance Awareness**: Stay informed about regulatory changes and requirements
- **Vendor Evaluation**: Identify solutions that support data sovereignty and independence

---

## The 14 European Sovereignty Theses

The filtering system evaluates content against 14 theses organized into strategic categories:

### Control & Power

**Thesis 1: Delegation is Power Transfer**
> "Delegation to agents is not automation; it is the surrender of deterministic policy to stochastic systems."

Focuses on understanding that AI delegation means giving up direct control to probabilistic systems.

**Thesis 7: Update Authority is the Chokepoint**
> "Who controls the weights controls the business."

Emphasizes that control over AI model updates determines business autonomy.

### Tradeoffs & Economics

**Thesis 2: Efficiency ≠ Sovereignty**
> "Performance and control are now divergent variables."

Highlights the tension between optimizing for performance versus maintaining control.

**Thesis 4: Updates are Economic Events**
> "Model updates are third-party market interventions, not maintenance."

Frames AI model updates as external economic forces affecting business operations.

### Ethics & Trust

**Thesis 3: Virtue is Survival**
> "Ethics is an insurance policy against agentic reputational collapse, not a growth lever."

Positions ethical AI as risk management rather than competitive advantage.

### Lock-in & Exit

**Thesis 5: Exit is a Technical Mirage**
> "You can export data, but not reasoning. Distillation is the only partial exit."

Explains the difficulty of switching AI providers once reasoning capabilities are embedded.

### Resilience

**Thesis 6: Survivability > Exit**
> "The metric is 'What runs when cognition degrades?' not 'Can we leave?'"

Shifts focus from exit strategies to operational resilience during AI failures.

### Architecture

**Thesis 8: Hybrid is Structural**
> "Edge = Execution/Privacy. Cloud = Reasoning/Scale. This is permanent."

Defines the permanent architectural split between edge and cloud computing.

### Procurement & Policy

**Thesis 9: Procurement is Enforcement**
> "Only buying power forces architectural sovereignty."

Emphasizes that purchasing decisions are the primary tool for enforcing sovereignty.

**Thesis 12: Democratic Sovereignty is Non-Default**
> "It must be engineered; the state prioritizes industrial survival."

Recognizes that democratic control over AI requires intentional design.

### Adoption & UX

**Thesis 10: Habits Persist by UX**
> "Sovereignty fails if the UX is inferior to the locked-in alternative."

Warns that sovereignty solutions must match or exceed proprietary alternatives in usability.

### Inequality & Access

**Thesis 11: The Agentic Divide**
> "Sovereignty is distributed by technical literacy and infrastructure access."

Highlights how sovereignty capabilities are unevenly distributed across society.

### Retail-Specific

**Thesis 13: Retail = Governor of Cognition**
> "Retailers are now certifiers of agentic outcomes, not just merchants."

Positions retailers as gatekeepers of AI-driven commerce.

**Thesis 14: Context is Shelf Space**
> "Visibility depends on retrieval context, not physical placement."

Explains how AI retrieval systems replace physical shelf placement in determining product visibility.

---

## Configuration

### Environment Variables

Configure the sovereignty filtering system in your [`.env`](.env:1) file:

```bash
# Enable/disable sovereignty filtering
SOVEREIGNTY_ENABLED=true

# Filtering mode: strict, balanced, or exploratory
SOVEREIGNTY_FILTERING_MODE=balanced

# Optional: Override mode threshold (0.0-1.0)
# SOVEREIGNTY_THESIS_THRESHOLD=0.60

# Include individual thesis scores in output
SOVEREIGNTY_INCLUDE_SCORES=true

# Include legacy CRM angle during transition
SOVEREIGNTY_LEGACY_CRM_ANGLE=true
```

### Filtering Modes

The system supports three filtering modes with different thresholds:

#### Strict Mode (`strict`)
- **Threshold**: 0.75/10 (7.5 out of 10)
- **Use Case**: High-confidence filtering for critical decision-making
- **Behavior**: Only extracts content with strong, clear alignment to sovereignty theses
- **Best For**: Executive briefings, policy analysis, strategic planning

```bash
SOVEREIGNTY_FILTERING_MODE=strict
```

#### Balanced Mode (`balanced`) - **Default**
- **Threshold**: 0.60/10 (6 out of 10)
- **Use Case**: Standard filtering for regular monitoring
- **Behavior**: Moderate filtering balancing relevance and coverage
- **Best For**: Daily newsletters, general awareness, team updates

```bash
SOVEREIGNTY_FILTERING_MODE=balanced
```

#### Exploratory Mode (`exploratory`)
- **Threshold**: 0.40/10 (4 out of 10)
- **Use Case**: Broad content discovery and trend monitoring
- **Behavior**: Permissive filtering to capture emerging topics
- **Best For**: Research, trend analysis, competitive intelligence

```bash
SOVEREIGNTY_FILTERING_MODE=exploratory
```

### Enabling/Disabling Sovereignty Filtering

**Enable Sovereignty Filtering** (default):
```bash
SOVEREIGNTY_ENABLED=true
```

**Disable Sovereignty Filtering** (legacy CRM mode):
```bash
SOVEREIGNTY_ENABLED=false
```

When disabled, the system reverts to the original CRM-focused content extraction without sovereignty analysis.

### Backwards Compatibility

The system maintains backwards compatibility through:

1. **Legacy CRM Angle**: Set `SOVEREIGNTY_LEGACY_CRM_ANGLE=true` to include traditional CRM perspectives alongside sovereignty analysis
2. **Graceful Fallback**: If sovereignty configuration fails to load, the system automatically falls back to legacy mode
3. **Optional Thesis Scores**: Control detailed scoring output with `SOVEREIGNTY_INCLUDE_SCORES`

---

## Output Format

### New Fields in Output

When sovereignty filtering is enabled, each extracted news item includes these additional fields:

#### `aligned_theses` (array of integers)
List of thesis IDs (1-14) that the content aligns with.

**Example**: `[1, 3, 6]`

#### `aligned_theses_formatted` (string)
Comma-separated string of aligned thesis IDs for easy display.

**Example**: `"1, 3, 6"`

#### `sovereignty_angle` (string)
2-3 sentence explanation of how the content relates to European retail AI sovereignty, referencing specific aligned theses.

**Example**: 
> "This directly supports Thesis 1 (regulatory compliance) and Thesis 3 (data sovereignty) by enforcing EU-specific AI governance. Thesis 6 (local innovation) benefits as European companies gain competitive advantage through early compliance expertise."

#### `sovereignty_relevance_score` (integer 0-10)
Overall alignment score with European sovereignty goals. Items below the mode threshold are filtered out.

**Example**: `9`

#### `thesis_scores` (object, optional)
Individual alignment scores for each aligned thesis. Only included when `SOVEREIGNTY_INCLUDE_SCORES=true`.

**Example**: `{"1": 10, "3": 9, "6": 8}`

### Google Sheets Column Structure

The system writes to Google Sheets with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Master Headline | Universal identifier (3-7 words) | "EU AI Act Enforcement Begins" |
| Headline | Concise summary (3-10 words) | "EU starts enforcing AI Act regulations" |
| Short Description | Factual summary (1-2 sentences) | "The European Union has begun..." |
| Source | Newsletter/publication name | "EuroNews Tech" |
| Date | Publication date | "Dec 10, 2024" |
| Companies | Organizations mentioned (max 3) | "European Commission" |
| Technologies | Relevant technologies (max 3) | "AI Regulation, GDPR, Compliance" |
| **Aligned Theses** | Comma-separated thesis IDs | "1, 3, 6" |
| **Sovereignty Angle** | European sovereignty context | "This directly supports Thesis 1..." |
| **Sovereignty Score** | Relevance score (0-10) | "9" |
| Thesis Scores (optional) | Individual thesis scores | "1:10, 3:9, 6:8" |
| CRM Angle (optional) | Legacy CRM perspective | "European retailers must audit..." |

**Bold** columns are new sovereignty-specific fields.

### Interpreting Thesis Alignment Scores

**Individual Thesis Scores** (when enabled):
- **8-10**: Strong, direct alignment - core relevance to the thesis
- **6-7**: Moderate alignment - clear but indirect relevance
- **4-5**: Weak alignment - tangential or partial relevance
- **0-3**: Minimal alignment - barely relevant (typically filtered out)

**Overall Sovereignty Relevance Score**:
- **9-10**: Critical for European sovereignty strategy
- **7-8**: Highly relevant, should be monitored
- **5-6**: Moderately relevant, contextual importance
- **4 and below**: Low relevance (filtered in strict/balanced modes)

---

## Usage Examples

### Example 1: High-Alignment Content (Multiple Theses)

**Input Newsletter Content**:
> "The European Commission has begun enforcing the AI Act, requiring all AI systems operating in the EU to register and comply with transparency requirements by Q2 2025. This affects major cloud providers including AWS, Google Cloud, and Microsoft Azure. Penalties up to 6% of global revenue."

**Extracted Output**:
```json
{
  "master_headline": "EU AI Act Enforcement Begins",
  "headline": "EU starts enforcing AI Act regulations",
  "short_description": "The European Union has begun enforcing the AI Act, requiring companies to comply with transparency and data governance rules.",
  "source": "EuroNews Tech",
  "date": "Dec 10, 2024",
  "companies": ["European Commission", "AWS", "Google Cloud"],
  "technologies": ["AI Regulation", "GDPR", "Compliance"],
  "aligned_theses": [1, 3, 6],
  "aligned_theses_formatted": "1, 3, 6",
  "sovereignty_angle": "This directly supports Thesis 1 (Delegation is Power Transfer) by establishing regulatory control over AI systems, Thesis 3 (Virtue is Survival) through mandatory ethical compliance, and Thesis 6 (Survivability > Exit) by ensuring operational standards during AI deployment.",
  "sovereignty_relevance_score": 9,
  "thesis_scores": {"1": 10, "3": 9, "6": 8}
}
```

### Example 2: Edge Computing & Data Sovereignty

**Input Newsletter Content**:
> "Shopify announced edge computing AI features for real-time personalization without cloud dependency. The solution processes customer data locally on edge devices, eliminating cloud data transfers and ensuring GDPR compliance."

**Sovereignty Angle Output**:
> "Aligns with Thesis 2 (Efficiency ≠ Sovereignty) by prioritizing data control over pure performance, Thesis 5 (Exit is a Technical Mirage) by reducing cloud lock-in through edge processing, and Thesis 8 (Hybrid is Structural) by implementing the permanent edge/cloud architectural split."

**Aligned Theses**: `[2, 5, 8]`  
**Sovereignty Score**: `8`

### Example 3: Open Source AI Development

**Input Newsletter Content**:
> "The European Open Source AI Foundation released benchmarks showing open-source models now match proprietary alternatives in retail applications. Mistral-Retail achieved 94% accuracy in product recommendations, comparable to GPT-4."

**Sovereignty Angle Output**:
> "Supports Thesis 11 (The Agentic Divide) by democratizing access to competitive AI through open-source tooling, Thesis 6 (Survivability > Exit) by providing alternatives when proprietary systems fail, and Thesis 4 (Updates are Economic Events) by giving European developers control over model evolution."

**Aligned Theses**: `[11, 6, 4]`  
**Sovereignty Score**: `7`

### Adjusting Filtering Threshold

**Scenario**: You're receiving too many low-relevance items in balanced mode.

**Solution**: Switch to strict mode or set a custom threshold:

```bash
# Option 1: Use strict mode
SOVEREIGNTY_FILTERING_MODE=strict

# Option 2: Custom threshold (0.70 = 7.0/10)
SOVEREIGNTY_THESIS_THRESHOLD=0.70
```

**Scenario**: You're missing potentially relevant content.

**Solution**: Switch to exploratory mode:

```bash
SOVEREIGNTY_FILTERING_MODE=exploratory
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: No Items Being Extracted

**Symptoms**: Newsletter processing completes but returns empty results.

**Possible Causes**:
1. Threshold too high for content
2. Content genuinely not aligned with sovereignty theses
3. Configuration error

**Solutions**:
```bash
# Try exploratory mode
SOVEREIGNTY_FILTERING_MODE=exploratory

# Check logs for filtering decisions
# Look for: "Newsletter 'X' is too long" or "No items meet the threshold"

# Verify configuration loaded successfully
# Look for: "Sovereignty filtering enabled (mode: balanced)"
```

#### Issue: Missing Sovereignty Fields in Output

**Symptoms**: Output contains headlines but lacks `sovereignty_angle` or `aligned_theses`.

**Possible Causes**:
1. Sovereignty filtering disabled
2. System fell back to legacy mode due to configuration error

**Solutions**:
```bash
# Verify sovereignty is enabled
SOVEREIGNTY_ENABLED=true

# Check for configuration errors in logs
# Look for: "WARNING: Failed to initialize sovereignty configuration"

# Ensure sovereignty_theses.json exists
# Location: newsletter_agent_core/config/sovereignty_theses.json
```

#### Issue: Thesis Scores Not Appearing

**Symptoms**: Output has sovereignty fields but no `thesis_scores`.

**Solution**:
```bash
# Enable thesis scores
SOVEREIGNTY_INCLUDE_SCORES=true
```

### Falling Back to Legacy Mode

The system automatically falls back to legacy CRM mode if:

1. `SOVEREIGNTY_ENABLED=false` is set
2. Configuration file [`newsletter_agent_core/config/sovereignty_theses.json`](newsletter_agent_core/config/sovereignty_theses.json:1) is missing
3. Configuration file contains invalid JSON
4. Required fields are missing from configuration

**Log Messages**:
```
WARNING: Failed to initialize sovereignty configuration: [error details]
  - Falling back to legacy CRM angle generation
```

or

```
Sovereignty filtering disabled - using legacy CRM angle generation
```

### Logging and Debugging

**Enable Detailed Logging**:

The system logs sovereignty filtering decisions. Look for these messages:

```
Sovereignty filtering enabled (mode: balanced)
  - Loaded 14 sovereignty theses
  - Configuration version: 1.0
```

**Filtering Decisions**:
```
Newsletter '[Title]' is too long (X tokens). Chunking and summarizing parts.
  - Split into Y chunks.
```

**Threshold Information**:
```
Relevance threshold: 0.60/10 (mode: balanced)
Only extract items scoring ≥0.60/10 in sovereignty relevance
```

### Configuration Validation

**Verify Configuration Loaded**:

Check that the configuration file is valid:

```python
from newsletter_agent_core.config import SovereigntyConfig

config = SovereigntyConfig()
config.load()
print(f"Version: {config.get_version()}")
print(f"Theses: {len(config.get_theses())}")
print(f"Threshold (balanced): {config.get_threshold('balanced')}")
```

Expected output:
```
Version: 1.0
Theses: 14
Threshold (balanced): 0.6
```

### Getting Help

If issues persist:

1. **Check Configuration**: Verify [`.env`](.env:1) settings match your requirements
2. **Review Logs**: Look for WARNING or ERROR messages during newsletter processing
3. **Test Configuration**: Use [`test_sovereignty_filtering.py`](test_sovereignty_filtering.py:1) to validate setup
4. **Fallback Mode**: Temporarily disable sovereignty filtering to isolate the issue:
   ```bash
   SOVEREIGNTY_ENABLED=false
   ```

---

## Related Documentation

- **Technical Architecture**: See [`SOVEREIGNTY_ARCHITECTURE.md`](SOVEREIGNTY_ARCHITECTURE.md:1) for implementation details
- **Configuration Reference**: See [`newsletter_agent_core/config/sovereignty_config.py`](newsletter_agent_core/config/sovereignty_config.py:1)
- **Theses Definition**: See [`newsletter_agent_core/config/sovereignty_theses.json`](newsletter_agent_core/config/sovereignty_theses.json:1)
- **Main Agent**: See [`newsletter_agent_core/agent.py`](newsletter_agent_core/agent.py:1)

---

**Last Updated**: December 19, 2024  
**Version**: 1.0  
**Configuration Version**: 1.0