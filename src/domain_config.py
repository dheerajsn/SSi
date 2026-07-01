"""
Domain configuration for SSI use cases.

A DomainConfig bundles:
  - scope_label: what each row represents ("currency", "country", "market")
  - fields: the ordered list of fields the LLM must populate
  - system_prompt: the fully-formed system prompt for the LLM

Pre-built configs:  FX_SSI, EQUITY_SSI, REPO_SSI, CORRESPONDENT_BANKING
Custom:             custom_domain(scope_label, fields)
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DomainConfig:
    name: str
    scope_label: str
    fields: List[str]
    system_prompt: str

    def extraction_question(self, item: str) -> str:
        """Build a natural-language extraction question for a single scope item."""
        fields_str = ", ".join(self.fields)
        return (
            f"Extract all available SSI fields for {self.scope_label} {item}. "
            f"Required fields: {fields_str}. "
            f"If a field is absent from the context, write 'N/A'."
        )

    def batch_extraction_question(self, items: List[str]) -> str:
        """Build a single question that covers multiple currencies/countries."""
        items_str = ", ".join(items)
        fields_str = ", ".join(self.fields)
        return (
            f"Extract all available SSI fields for each of the following "
            f"{self.scope_label}(s): {items_str}. "
            f"For each {self.scope_label} present a separate block with fields: {fields_str}. "
            f"Write 'N/A' for any field not found in the context."
        )


def _make_prompt(scope_label: str, fields: List[str], context: str) -> str:
    fields_block = "\n".join(f"  - {f}" for f in fields)
    return (
        f"You are a settlement instructions assistant for {scope_label.title()} Operations.\n"
        f"Answer questions about Standard Settlement Instructions (SSI) using ONLY the\n"
        f"context provided below. The context comes from verified SSI documents.\n\n"
        f"{context}\n\n"
        f"For each {scope_label}, extract and present these fields:\n{fields_block}\n\n"
        "RULES:\n"
        "1. ONLY use information from the provided context. Never use general knowledge,\n"
        "   training data, or assumptions to fill in SWIFT codes, BIC codes, IBAN numbers,\n"
        "   or account numbers. A wrong code causes a failed or misdirected settlement.\n"
        "2. If the settlement instructions for a requested instrument are not in the context\n"
        "   at all, say exactly: 'Settlement instructions for [X] were not found in the\n"
        "   available SSI documents.' Do not guess or infer.\n"
        "3. If an instrument is present but a specific field has no value, write 'N/A'.\n"
        "4. If the context contains cross-references (e.g. 'see GERMANY entry'), look up\n"
        "   that entry in the same context block and include its details.\n"
        "5. If an instrument has multiple settlement paths (Agency/Principal, Triparty/Bilateral),\n"
        "   list ALL paths with clear labels. Do not silently pick one.\n"
        "6. Always include operational notes or special instructions.\n"
        "7. RESPONSE FORMAT: Always respond with a single valid JSON object — no text outside it.\n"
        "   Conversational answer: {\"answer\": \"...\", \"source\": \"doc_name / section\"}\n"
        "   Unknown instrument: {\"answer\": \"Settlement instructions for [X] were not found in the available SSI documents.\", \"source\": null}\n"
        "8. Extraction queries specify their own JSON schema in the question; follow it exactly.\n"
        "   Missing field value → \"N/A\". Unknown instrument → all its field values set to \"not found\"."
    )


FX_SSI = DomainConfig(
    name="fx_ssi",
    scope_label="currency",
    fields=[
        "Currency",
        "Beneficiary Bank",
        "SWIFT/BIC",
        "Account Number",
        "IBAN",
        "Correspondent/Intermediary Bank",
        "Correspondent SWIFT/BIC",
        "Special Instructions",
        "Notes",
    ],
    system_prompt=_make_prompt(
        "FX currency",
        ["Currency", "Beneficiary Bank", "SWIFT/BIC", "Account Number", "IBAN",
         "Correspondent/Intermediary Bank", "Correspondent SWIFT/BIC",
         "Special Instructions", "Notes"],
        "You handle FX Spot and FX Forward SSI. Settlement instructions include\n"
        "beneficiary bank details, SWIFT/BIC codes, and correspondent bank chains.",
    ),
)

EQUITY_SSI = DomainConfig(
    name="equity_ssi",
    scope_label="country",
    fields=[
        "Country",
        "Custodian Bank",
        "Custodian BIC",
        "DTC/Euroclear/Clearstream Account",
        "Settlement Currency",
        "Notes",
    ],
    system_prompt=_make_prompt(
        "equity country",
        ["Country", "Custodian Bank", "Custodian BIC",
         "DTC/Euroclear/Clearstream Account", "Settlement Currency", "Notes"],
        "You handle Cash Equities SSI. Settlement instructions specify the local\n"
        "custodian, ICSD account (Euroclear/Clearstream/DTC), and settlement currency.",
    ),
)

REPO_SSI = DomainConfig(
    name="repo_ssi",
    scope_label="country",
    fields=[
        "Country",
        "Settlement Type",
        "Custodian/ICSD",
        "BIC",
        "Account Number",
        "Notes",
    ],
    system_prompt=_make_prompt(
        "repo country",
        ["Country", "Settlement Type", "Custodian/ICSD", "BIC", "Account Number", "Notes"],
        "You handle Repo/Fixed Income SSI. Settlement instructions include triparty\n"
        "(Euroclear, Clearstream, JPMorgan GCF) and bilateral clearing paths.",
    ),
)

CORRESPONDENT_BANKING = DomainConfig(
    name="correspondent_banking",
    scope_label="currency",
    fields=[
        "Currency",
        "Nostro Bank",
        "Nostro SWIFT/BIC",
        "Nostro Account",
        "Intermediary Bank",
        "Intermediary SWIFT/BIC",
        "Notes",
    ],
    system_prompt=_make_prompt(
        "correspondent banking currency",
        ["Currency", "Nostro Bank", "Nostro SWIFT/BIC", "Nostro Account",
         "Intermediary Bank", "Intermediary SWIFT/BIC", "Notes"],
        "You handle Correspondent Banking SSI. Instructions specify nostro accounts\n"
        "and intermediary bank chains for wire transfers.",
    ),
)

MARKETS_SSI = DomainConfig(
    name="markets_ssi",
    scope_label="market",
    fields=[
        "Market",
        "Country",
        "Security Type",
        "Institution BIC",
        "Global Agent Swift Agent",
        "Local Agent Swift Agent",
        "Local Agent Account Number",
        "Notes",
    ],
    system_prompt=_make_prompt(
        "securities settlement market",
        ["Market", "Country", "Security Type", "Institution BIC",
         "Global Agent Swift Agent", "Local Agent Swift Agent",
         "Local Agent Account Number", "Notes"],
        "You handle Securities Settlement SSI by market and method.\n"
        "Documents use varying terminology for the same concepts:\n"
        "  • 'Institution BIC' = PSET BIC = :95P::PSET — the central depository or ICSD where the trade settles\n"
        "    (e.g., DAKVDEFF = Clearstream Germany, MGTCBEBE = Euroclear Belgium, DTCYUS33 = DTC)\n"
        "  • 'Global Agent Swift Agent' = DEC/RECU BIC = :95P::DECU/:95P::RECU = BCI\n"
        "    — the global custodian or delivering/receiving custodian BIC\n"
        "  • 'Local Agent Swift Agent' = Local Settlement Agent BIC = REAG/DEAG BIC = :95P::REAG/:95P::DEAG\n"
        "    — the local sub-custodian or depository participant at the place of settlement\n"
        "  • 'Local Agent Account Number' = the custodian's safe-custody account at the local agent\n"
        "    (field :97A::SAFE in MT54x messages)\n"
        "Markets are identified as 'Country-Method' (e.g., 'Italy-CLEARGER', 'France-EUROCLEAR', 'UK-CREST').\n"
        "Always include the exact BIC codes and account numbers as they appear in the context.",
    ),
)

DOMAIN_REGISTRY: Dict[str, DomainConfig] = {
    "fx_ssi": FX_SSI,
    "equity_ssi": EQUITY_SSI,
    "repo_ssi": REPO_SSI,
    "correspondent_banking": CORRESPONDENT_BANKING,
    "markets_ssi": MARKETS_SSI,
}


def get_domain(name: str) -> DomainConfig:
    if name not in DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain '{name}'. Available: {list(DOMAIN_REGISTRY)}")
    return DOMAIN_REGISTRY[name]


def custom_domain(
    scope_label: str,
    fields: List[str],
    name: str = "custom",
    context_description: str = "",
) -> DomainConfig:
    """Create a domain config with any scope label and field list at runtime."""
    return DomainConfig(
        name=name,
        scope_label=scope_label,
        fields=fields,
        system_prompt=_make_prompt(scope_label, fields, context_description),
    )
