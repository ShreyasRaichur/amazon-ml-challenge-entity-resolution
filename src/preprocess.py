"""
Preprocessing module for Business Entity Resolution.
Provides country-agnostic text normalization, diacritic folding, legal entity
suffix stripping, address regex extraction, phone standardization, and acronym generation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from .config import ColumnMapping
except (ImportError, ValueError):
    from src.config import ColumnMapping


class TextNormalizer:
    """Fast, robust Unicode and diacritic normalization."""

    @staticmethod
    def strip_accents(text: str) -> str:
        """Strip accents and diacritics while preserving base characters (e.g., 'Café' -> 'Cafe')."""
        if not text:
            return ""
        # Decompose unicode characters into base char and combining diacritic marks
        nfkd_form = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Standardize text: accent stripping, lowercasing, whitespace collapse, punctuation normalization."""
        if text is None or pd.isna(text):
            return ""
        text = str(text)
        text = TextNormalizer.strip_accents(text)
        text = text.lower()
        # Replace non-alphanumeric (except standard separators) with spaces
        text = re.sub(r"[^\w\s\-\.]", " ", text)
        # Collapse multiple dashes/periods/spaces
        text = re.sub(r"[\s\-\.]+", " ", text).strip()
        return text

    @staticmethod
    def extract_acronym(text: str) -> str:
        """Extract acronym from alphanumeric tokens (e.g. 'Tata Consultancy Services' -> 'tcs')."""
        tokens = [t for t in re.split(r"\s+", text) if t and t[0].isalnum()]
        if len(tokens) >= 2:
            return "".join([t[0] for t in tokens]).lower()
        return ""


class LegalEntityNormalizer:
    """
    Country-agnostic legal entity suffix normalizer covering multi-jurisdiction
    corporate designations (US, India, France, and Global).
    """

    # Comprehensive multi-country corporate designation mapping
    # Maps canonical legal type to list of regex patterns
    LEGAL_PATTERNS: Dict[str, List[str]] = {
        "pvt_ltd": [
            r"\bpvt\s+ltd\b", r"\bprivate\s+limited\b", r"\bp\.?\s*ltd\b",
            r"\bpvt\s+limited\b", r"\bprvt\s+ltd\b"
        ],
        "ltd": [
            r"\bltd\b", r"\blimited\b", r"\bl\.t\.d\b"
        ],
        "llc": [
            r"\bllc\b", r"\bl\.l\.c\b", r"\blimited\s+liability\s+company\b"
        ],
        "inc": [
            r"\binc\b", r"\bincorporated\b", r"\bi\.n\.c\b"
        ],
        "corp": [
            r"\bcorp\b", r"\bcorporation\b", r"\bc\.o\.r\.p\b"
        ],
        "llp": [
            r"\bllp\b", r"\bl\.l\.p\b", r"\blimited\s+liability\s+partnership\b"
        ],
        "co": [
            r"\bco\b", r"\bcompany\b", r"\bc\.o\b"
        ],
        # French corporate designations (Distribution Shift)
        "sarl": [
            r"\bsarl\b", r"\bs\.a\.r\.l\b", r"\bsociete\s+a\s+responsabilite\s+limitee\b"
        ],
        "sas": [
            r"\bsas\b", r"\bs\.a\.s\b", r"\bsociete\s+par\s+actions\s+simplifiee\b"
        ],
        "sasu": [
            r"\bsasu\b", r"\bs\.a\.s\.u\b"
        ],
        "sa": [
            r"\bsa\b", r"\bs\.a\b", r"\bsociete\s+anonyme\b"
        ],
        "eurl": [
            r"\beurl\b", r"\be\.u\.r\.l\b"
        ],
        "snc": [
            r"\bsnc\b", r"\bs\.n\.c\b"
        ],
        "sci": [
            r"\bsci\b", r"\bs\.c\.i\b"
        ],
        # Global corporate designations
        "gmbh": [
            r"\bgmbh\b", r"\bg\.m\.b\.h\b"
        ],
        "ag": [
            r"\bag\b", r"\ba\.g\b"
        ],
        "bv": [
            r"\bbv\b", r"\bb\.v\b"
        ],
        "nv": [
            r"\bnv\b", r"\bn\.v\b"
        ],
        "spa": [
            r"\bspa\b", r"\bs\.p\.a\b"
        ],
    }

    def __init__(self) -> None:
        # Precompile regexes for fast execution
        self.compiled_patterns: List[Tuple[str, re.Pattern[str]]] = []
        for legal_type, patterns in self.LEGAL_PATTERNS.items():
            for pat in patterns:
                self.compiled_patterns.append(
                    (legal_type, re.compile(pat, re.IGNORECASE))
                )

    def parse(self, clean_name: str) -> Tuple[str, str]:
        """
        Extract the root name (without corporate suffix) and legal entity type.
        Returns:
            (root_name, legal_type)
        """
        if not clean_name:
            return "", "none"

        detected_type = "none"
        working_name = clean_name

        for legal_type, regex in self.compiled_patterns:
            if regex.search(working_name):
                detected_type = legal_type
                # Remove legal suffix from name
                working_name = regex.sub(" ", working_name)
                break

        # Clean trailing/leading spaces and punctuation left behind
        root_name = re.sub(r"\s+", " ", working_name).strip()
        # Fallback if stripping emptied the name entirely
        if not root_name:
            root_name = clean_name

        return root_name, detected_type


class AddressNormalizer:
    """Country-agnostic address parser extracting street numbers, units, and postal codes."""

    # International address tokens
    STREET_TYPES = {
        "st": "street", "street": "street",
        "rd": "road", "road": "road",
        "ave": "avenue", "avenue": "avenue", "av": "avenue",
        "blvd": "boulevard", "boulevard": "boulevard",
        "dr": "drive", "drive": "drive",
        "ln": "lane", "lane": "lane",
        "ct": "court", "court": "court",
        "pl": "place", "place": "place",
        "pkwy": "parkway", "parkway": "parkway",
        "sq": "square", "square": "square",
        "rue": "rue", "chemin": "chemin", "allee": "allee", "boulevard": "boulevard",
        "marg": "marg", "rasta": "rasta", "chowk": "chowk", "bazaar": "bazaar",
    }

    UNIT_REGEX = re.compile(
        r"\b(?:apt|apartment|suite|ste|fl|floor|unit|bldg|building|room|rm|no|flat)\s*#?\s*([a-z0-9\-]+)\b",
        re.IGNORECASE
    )

    NUMBER_REGEX = re.compile(r"\b\d+([a-z])?\b", re.IGNORECASE)

    # Agnostic postal code patterns: 5-digit (US/FR), 6-digit (IN), 9-digit (US extended)
    POSTAL_REGEX = re.compile(
        r"\b(\d{5}(?:-\d{4})?|[1-9]\d{5}|\d{4,6}|[A-Z]{1,2}\d[A-Z0-9]?\s*\d[A-Z]{2})\b",
        re.IGNORECASE
    )

    @classmethod
    def extract_street_numbers(cls, address: str) -> List[str]:
        """Extract all numerical components (e.g. '123', '45B') from address."""
        if not address:
            return []
        # Extract whole match strings
        raw_matches = re.finditer(cls.NUMBER_REGEX, address)
        nums = [m.group(0).lower() for m in raw_matches]
        return list(dict.fromkeys(nums))  # preserve order deduplicated

    @classmethod
    def extract_postal_code(cls, text: str) -> str:
        """Find valid postal code in text or address string."""
        if not text:
            return ""
        matches = cls.POSTAL_REGEX.findall(text)
        for m in matches:
            # Handle US 5+4 format: extract first 5 digits
            if "-" in m:
                base = m.split("-")[0].strip()
                if len(base) == 5 and base.isdigit():
                    return base
            cleaned_m = re.sub(r"[^0-9a-zA-Z]", "", m)
            if 4 <= len(cleaned_m) <= 10 and any(c.isdigit() for c in cleaned_m):
                return cleaned_m.lower()
        return ""

    @classmethod
    def normalize_address(cls, address: Optional[str]) -> str:
        """Standardize address text, expanding common abbreviations."""
        if not address or pd.isna(address):
            return ""
        clean_addr = TextNormalizer.clean_text(address)
        tokens = clean_addr.split()
        normalized_tokens = [cls.STREET_TYPES.get(t, t) for t in tokens]
        return " ".join(normalized_tokens)


class CountryNormalizer:
    """
    Open-set country code standardizer.
    Resolves known aliases for US, India, France, and preserves open-set unknown countries.
    """

    KNOWN_MAPPING = {
        "united states": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US", "us": "US",
        "india": "IN", "ind": "IN", "in": "IN", "bharat": "IN",
        "france": "FR", "fra": "FR", "fr": "FR", "republique francaise": "FR",
        "united kingdom": "GB", "uk": "GB", "gb": "GB", "great britain": "GB",
        "germany": "DE", "de": "DE", "deutschland": "DE",
        "canada": "CA", "ca": "CA",
        "australia": "AU", "au": "AU",
    }

    @classmethod
    def normalize_country(cls, country_val: Optional[str]) -> str:
        """Map country to standard 2-letter ISO code or uppercase normalized token."""
        if country_val is None or pd.isna(country_val):
            return "UNKNOWN"
        raw = str(country_val).strip().lower()
        raw = TextNormalizer.strip_accents(raw)
        raw = re.sub(r"[^\w\s]", "", raw).strip()
        if not raw:
            return "UNKNOWN"
        if raw in cls.KNOWN_MAPPING:
            return cls.KNOWN_MAPPING[raw]
        # Open-set dynamic country: return sanitized uppercase string
        return raw.upper().replace(" ", "_")


class PhoneNormalizer:
    """Extract standard digits for phone comparison, stripping formatting and country codes."""

    @staticmethod
    def extract_digits(phone: Optional[str]) -> str:
        if phone is None or pd.isna(phone):
            return ""
        digits = re.sub(r"\D", "", str(phone))
        # Keep trailing 10 digits if longer (strips international prefixes like +1, +91, +33)
        if len(digits) > 10:
            return digits[-10:]
        return digits


class RecordPreprocessor:
    """
    Full pipeline record preprocessor for Source 1, 2, and 3 DataFrames.
    Maps arbitrary schema columns to standardized, feature-ready fields.
    """

    def __init__(self, col_map: Optional[ColumnMapping] = None) -> None:
        self.col_map = col_map or ColumnMapping()
        self.legal_normalizer = LegalEntityNormalizer()

    def resolve_column(self, df: pd.DataFrame, target_col: str) -> Optional[str]:
        """Find best matching column name in DataFrame using alias lookup."""
        if target_col in df.columns:
            return target_col
        # Check aliases
        aliases = self.col_map.aliases.get(target_col, [])
        for alias in aliases:
            if alias in df.columns:
                return alias
            # Case-insensitive check
            for col in df.columns:
                if col.lower() == alias.lower():
                    return col
        # Case-insensitive direct check
        for col in df.columns:
            if col.lower() == target_col.lower():
                return col
        return None

    def process_dataframe(self, df: pd.DataFrame, source_prefix: str) -> pd.DataFrame:
        """
        Transform raw source DataFrame into normalized records with rich metadata.
        Guarantees presence of standard columns.
        """
        processed = pd.DataFrame()

        # 1. Resolve Entity ID
        id_col = self.resolve_column(df, "entity_id")
        if id_col is not None:
            processed["entity_id"] = df[id_col].astype(str).str.strip()
        else:
            # Auto-generate ID if missing
            processed["entity_id"] = [f"{source_prefix}-{i}" for i in range(len(df))]

        # Ensure entity_id has proper source prefix if missing
        processed["entity_id"] = processed["entity_id"].apply(
            lambda x: x if x.startswith(source_prefix) else f"{source_prefix}-{x}"
        )

        # 2. Business Name Normalization & Legal Suffix
        name_col = self.resolve_column(df, "name")
        raw_names = df[name_col].fillna("").astype(str) if name_col else pd.Series([""] * len(df))
        processed["raw_name"] = raw_names
        processed["clean_name"] = raw_names.apply(TextNormalizer.clean_text)

        # Parse root name and legal type
        parsed_legal = processed["clean_name"].apply(self.legal_normalizer.parse)
        processed["root_name"] = [p[0] for p in parsed_legal]
        processed["legal_type"] = [p[1] for p in parsed_legal]
        processed["acronym"] = processed["root_name"].apply(TextNormalizer.extract_acronym)

        # 3. Address Normalization & Street Numbers
        addr_col = self.resolve_column(df, "address")
        raw_addr = df[addr_col].fillna("").astype(str) if addr_col else pd.Series([""] * len(df))
        processed["raw_address"] = raw_addr
        processed["clean_address"] = raw_addr.apply(AddressNormalizer.normalize_address)
        processed["street_numbers"] = processed["clean_address"].apply(
            lambda a: ",".join(AddressNormalizer.extract_street_numbers(a))
        )

        # 4. Postal Code
        post_col = self.resolve_column(df, "postal_code")
        if post_col and post_col in df:
            processed["postal_code"] = (
                df[post_col].fillna("").astype(str).apply(AddressNormalizer.extract_postal_code)
            )
        else:
            # Attempt extraction from address
            processed["postal_code"] = processed["clean_address"].apply(
                AddressNormalizer.extract_postal_code
            )

        # 5. City and State
        city_col = self.resolve_column(df, "city")
        processed["city"] = (
            df[city_col].fillna("").astype(str).apply(TextNormalizer.clean_text)
            if city_col else ""
        )

        state_col = self.resolve_column(df, "state")
        processed["state"] = (
            df[state_col].fillna("").astype(str).apply(TextNormalizer.clean_text)
            if state_col else ""
        )

        # 6. Country Normalization (Agnostic open-set)
        country_col = self.resolve_column(df, "country")
        raw_country = (
            df[country_col].fillna("").astype(str) if country_col else pd.Series(["UNKNOWN"] * len(df))
        )
        processed["country"] = raw_country.apply(CountryNormalizer.normalize_country)

        # 7. Phone digits
        phone_col = self.resolve_column(df, "phone")
        processed["phone_digits"] = (
            df[phone_col].fillna("").astype(str).apply(PhoneNormalizer.extract_digits)
            if phone_col else ""
        )

        # 8. Composite representation for dense blocking and cross-encoder
        processed["composite_text"] = (
            processed["clean_name"] + " " +
            processed["clean_address"] + " " +
            processed["city"] + " " +
            processed["state"] + " " +
            processed["postal_code"]
        ).str.strip()

        return processed
