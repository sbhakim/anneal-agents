from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import re
from pathlib import Path
import time


@dataclass
class DeonticRule:
    """Value constraint (KG^val triple)."""
    action: str
    modality: str  # Obligatory, Prohibited, Permitted
    condition: str
    confidence: float = 1.0
    source: str = 'manual'


@dataclass
class CausalRelation:
    """Causal dependency (KG^cau edge)."""
    cause: str
    effect: str
    relation_type: str  # causes, prevents, enables
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)


class KGBuilder:
    """
    LLM-assisted knowledge graph construction from policy documents.
    Addresses scalability limitation from Section XIII.
    """

    def __init__(self, config: Dict[str, Any], llm_provider=None):
        self.config = config
        self.llm_provider = llm_provider
        self.extraction_stats = {
            'documents_processed': 0,
            'rules_extracted': 0,
            'rules_validated': 0,
            'rules_rejected': 0,
            'extraction_time_seconds': 0,
        }

    def extract_from_policy_document(self, policy_text: str,
                                    domain: str = 'travel') -> List[DeonticRule]:
        """
        Extract deontic rules from natural language policy.
        Uses LLM with structured prompting.
        """

        start_time = time.time()

        prompt = self._build_extraction_prompt(policy_text, domain)

        if self.llm_provider:
            extracted_rules = self._llm_extract(prompt)
        else:
            extracted_rules = self._heuristic_extract(policy_text, domain)

        elapsed = time.time() - start_time
        self.extraction_stats['documents_processed'] += 1
        self.extraction_stats['rules_extracted'] += len(extracted_rules)
        self.extraction_stats['extraction_time_seconds'] += elapsed

        return extracted_rules

    def _build_extraction_prompt(self, policy_text: str, domain: str) -> str:
        """Construct structured prompt for rule extraction."""

        examples = {
            'travel': """
Example policy: "Corporate cards are blocked on blackout dates"
Extracted rule:
{
  "action": "BookHotel",
  "modality": "Prohibited",
  "condition": "payment_method == 'CorporateCard' AND date IN blackout_dates"
}
""",
            'ecommerce': """
Example policy: "Employee discounts require manager approval for orders over $500"
Extracted rule:
{
  "action": "ApplyPromoCode",
  "modality": "Obligatory",
  "condition": "promo_type == 'employee_discount' AND order_total > 500 REQUIRES manager_approval"
}
""",
        }

        return f"""Extract deontic constraints from this policy document.
For each rule, identify:
- action: The operation being constrained
- modality: Obligatory (required), Prohibited (forbidden), or Permitted (allowed)
- condition: When the constraint applies (use simple predicate logic)

Domain: {domain}

{examples.get(domain, examples['travel'])}

Policy Document:
{policy_text}

Output JSON array of rules:
"""

    def _llm_extract(self, prompt: str) -> List[DeonticRule]:
        """Use LLM to extract rules with structured output."""

        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=0.1,  # Low temp for consistency
                max_tokens=1000,
            )

            rules_json = self._parse_json_response(response)
            return [DeonticRule(**rule, source='llm_assisted') for rule in rules_json]

        except Exception as e:
            print(f"KG_BUILDER: LLM extraction failed: {e}")
            return []

    def _heuristic_extract(self, policy_text: str, domain: str) -> List[DeonticRule]:
        """Fallback heuristic extraction using pattern matching."""

        rules = []

        # Pattern: "X is prohibited/required when Y"
        prohibition_patterns = [
            r'(\w+)\s+(?:is|are)\s+(?:prohibited|forbidden|blocked)\s+(?:when|if|on)\s+(.+)',
            r'(?:cannot|must not)\s+(\w+)\s+(?:when|if)\s+(.+)',
        ]

        requirement_patterns = [
            r'(\w+)\s+(?:requires|must have|needs)\s+(.+)',
            r'(?:must|required to)\s+(\w+)\s+(?:when|if)\s+(.+)',
        ]

        for pattern in prohibition_patterns:
            matches = re.finditer(pattern, policy_text, re.IGNORECASE)
            for match in matches:
                action = self._normalize_action(match.group(1), domain)
                condition = match.group(2).strip()

                rules.append(DeonticRule(
                    action=action,
                    modality='Prohibited',
                    condition=condition,
                    confidence=0.7,
                    source='heuristic'
                ))

        for pattern in requirement_patterns:
            matches = re.finditer(pattern, policy_text, re.IGNORECASE)
            for match in matches:
                action = self._normalize_action(match.group(1), domain)
                condition = match.group(2).strip()

                rules.append(DeonticRule(
                    action=action,
                    modality='Obligatory',
                    condition=condition,
                    confidence=0.6,
                    source='heuristic'
                ))

        return rules

    def _normalize_action(self, action_text: str, domain: str) -> str:
        """Map natural language to operator names."""

        mappings = {
            'travel': {
                'book hotel': 'BookHotel',
                'hotel': 'BookHotel',
                'book flight': 'BookFlight',
                'flight': 'BookFlight',
                'payment': 'ProcessPayment',
            },
            'ecommerce': {
                'order': 'PlaceOrder',
                'discount': 'ApplyPromoCode',
                'promo': 'ApplyPromoCode',
                'ship': 'CalculateShipping',
                'refund': 'ProcessRefund',
            },
        }

        action_lower = action_text.lower().strip()
        domain_map = mappings.get(domain, {})

        for key, operator in domain_map.items():
            if key in action_lower:
                return operator

        return action_text.title().replace(' ', '')

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON from LLM response."""

        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try to find individual JSON objects
        objects = re.findall(r'\{[\s\S]*?\}', response)
        parsed = []
        for obj in objects:
            try:
                parsed.append(json.loads(obj))
            except json.JSONDecodeError:
                continue

        return parsed

    def extract_from_veto_event(self, veto_event: Dict[str, Any]) -> Optional[DeonticRule]:
        """
        Infer implicit constraint from human veto.
        Enables incremental KG learning.
        """

        patch = veto_event.get('patch', {})
        reason = veto_event.get('reason', '')

        action = patch.get('target_operator', '')
        if not action:
            return None

        # Infer modality from veto reason
        if 'violates policy' in reason.lower() or 'prohibited' in reason.lower():
            modality = 'Prohibited'
        elif 'requires approval' in reason.lower() or 'must' in reason.lower():
            modality = 'Obligatory'
        else:
            modality = 'Prohibited'  # Conservative default

        # Extract condition from patch context
        condition = self._infer_condition(patch, reason)

        return DeonticRule(
            action=action,
            modality=modality,
            condition=condition,
            confidence=0.8,  # High confidence from human feedback
            source='veto_learning'
        )

    def _infer_condition(self, patch: Dict[str, Any], reason: str) -> str:
        """Infer constraint condition from patch and veto reason."""

        # Extract predicates from patch
        predicates = patch.get('preconditions', [])
        if predicates:
            return ' AND '.join(predicates)

        # Parse reason for constraints
        if 'when' in reason.lower():
            parts = reason.lower().split('when')
            if len(parts) > 1:
                return parts[1].strip()

        # Default to patch target
        target = patch.get('target', '')
        return f"{target} is modified"

    def validate_with_human(self, rules: List[DeonticRule],
                          validation_mode: str = 'interactive') -> Tuple[List[DeonticRule], List[DeonticRule]]:
        """
        Human-in-the-loop validation of extracted rules.
        Returns (accepted, rejected).
        """

        accepted = []
        rejected = []

        if validation_mode == 'interactive':
            for rule in rules:
                print(f"\nProposed rule:")
                print(f"  Action: {rule.action}")
                print(f"  Modality: {rule.modality}")
                print(f"  Condition: {rule.condition}")
                print(f"  Confidence: {rule.confidence:.2f}")
                print(f"  Source: {rule.source}")

                response = input("Accept? [y/n/e(dit)]: ").lower().strip()

                if response == 'y':
                    accepted.append(rule)
                    self.extraction_stats['rules_validated'] += 1
                elif response == 'e':
                    edited = self._edit_rule_interactive(rule)
                    if edited:
                        accepted.append(edited)
                        self.extraction_stats['rules_validated'] += 1
                else:
                    rejected.append(rule)
                    self.extraction_stats['rules_rejected'] += 1

        elif validation_mode == 'confidence_threshold':
            threshold = self.config.get('validation_threshold', 0.8)
            for rule in rules:
                if rule.confidence >= threshold:
                    accepted.append(rule)
                    self.extraction_stats['rules_validated'] += 1
                else:
                    rejected.append(rule)
                    self.extraction_stats['rules_rejected'] += 1

        elif validation_mode == 'auto_accept':
            accepted = rules
            self.extraction_stats['rules_validated'] += len(rules)

        return accepted, rejected

    def _edit_rule_interactive(self, rule: DeonticRule) -> Optional[DeonticRule]:
        """Allow human to edit extracted rule."""

        print(f"\nCurrent: {rule.action} is {rule.modality} when {rule.condition}")

        action = input(f"Action [{rule.action}]: ").strip() or rule.action
        modality = input(f"Modality [{rule.modality}]: ").strip() or rule.modality
        condition = input(f"Condition [{rule.condition}]: ").strip() or rule.condition

        return DeonticRule(
            action=action,
            modality=modality,
            condition=condition,
            confidence=1.0,  # Human-edited = high confidence
            source='human_edited'
        )

    def export_to_kg_format(self, rules: List[DeonticRule], output_file: Path):
        """Export rules to knowledge graph JSON format."""

        output_file.parent.mkdir(parents=True, exist_ok=True)

        kg_triples = []
        for rule in rules:
            kg_triples.append({
                'action': rule.action,
                'modality': rule.modality,
                'condition': rule.condition,
                'confidence': rule.confidence,
                'source': rule.source,
            })

        with open(output_file, 'w') as f:
            json.dump({'value_constraints': kg_triples}, f, indent=2)

        print(f"KG_BUILDER: Exported {len(kg_triples)} rules to {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """Return extraction statistics for paper reporting."""

        total_extracted = self.extraction_stats['rules_extracted']
        validated = self.extraction_stats['rules_validated']
        rejected = self.extraction_stats['rules_rejected']

        return {
            **self.extraction_stats,
            'validation_precision': validated / total_extracted if total_extracted > 0 else 0,
            'rejection_rate': rejected / total_extracted if total_extracted > 0 else 0,
            'avg_time_per_document': (
                self.extraction_stats['extraction_time_seconds'] /
                self.extraction_stats['documents_processed']
                if self.extraction_stats['documents_processed'] > 0 else 0
            ),
        }
