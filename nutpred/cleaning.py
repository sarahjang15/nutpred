import re
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_top_level(text: str) -> List[str]:
    """Split by top-level commas/periods (ignore those inside () or [])."""
    parts, buf = [], []
    depth_par = depth_brk = 0
    for c in text:
        if c == '(':
            depth_par += 1; buf.append(c)
        elif c == ')':
            depth_par = max(depth_par - 1, 0); buf.append(c)
        elif c == '[':
            depth_brk += 1; buf.append(c)
        elif c == ']':
            depth_brk = max(depth_brk - 1, 0); buf.append(c)
        elif (c in {',','.'}) and depth_par == 0 and depth_brk == 0:
            part = ''.join(buf).strip()
            if part: parts.append(part)
            buf = []
        else:
            buf.append(c)
    last = ''.join(buf).strip()
    if last: parts.append(last)
    return parts

# Lead-ins like "contains 2% or less of:", "and less than 2% of the following:", etc.
_CONTAINS_PREFIX = re.compile(
    r"""(?ix) ^
        \s*
        (?:and\s+)?less\s+than\s+\d+%?\s+of(?:\s+the\s+following)?\s*:? |
        (?:may\s+)?contains?\s+\d+%?\s*(?:or\s+less|or\s+more)?\s*(?:of\s*:?) |
        (?:may\s+)?contains?\s+(?:one\s+or\s+more\s+of\s+the\s+following:\s*) |
        (?:may\s+)?contains?\s*:?
    """
)

# Pattern for "less than 2 of each of the following [ingredient]"
_LESS_THAN_EACH_PATTERN = re.compile(
    r"""(?ix)
        (?:and\s+)?less\s+than\s+\d+\s+of\s+each\s+of\s+the\s+following\s*:?\s*
        (.+)
    """
)

def _strip_contains_prefix(s: str) -> str:
    original = s
    cleaned = _CONTAINS_PREFIX.sub('', s).strip()
    if original != cleaned:
        logger.info(f"Stripped contains prefix: '{original}' -> '{cleaned}'")
    return cleaned

def _clean_less_than_each_pattern(s: str) -> str:
    """Clean patterns like 'less than 2 of each of the following sea salt' -> 'sea salt'"""
    original = s
    match = _LESS_THAN_EACH_PATTERN.match(s)
    if match:
        cleaned = match.group(1).strip()
        logger.info(f"Cleaned 'less than X of each' pattern: '{original}' -> '{cleaned}'")
        return cleaned
    return s

def clean_ingredient_text(text):
    """
    Turn an INGREDIENTS string into a clean list of tokens.
    - drops bracketed sublists but keeps the head phrase (e.g., "... seasoning [salt,...]" -> "... seasoning")
    - strips 'contains/less than 2% of the following' style prefixes
    - handles 'less than X of each of the following [ingredient]' patterns
    """
    text = str(text).strip()
    if not text: 
        logger.debug("Empty text input, returning empty list")
        return []
    
    logger.info(f"Cleaning ingredient text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    
    text = text.lower()
    text = text.replace('{', '(').replace('}', ')')
    text = re.sub(r'^\s*ingredients?\s*:\s*', '', text, flags=re.I)
    parts = split_top_level(text)

    cleaned = []
    for i, part in enumerate(parts):
        ing = part.strip()
        if not ing: continue

        logger.debug(f"Processing part {i+1}: '{ing}'")

        # Clean "less than X of each of the following" patterns
        ing = _clean_less_than_each_pattern(ing)

        # explicit "(contains ...)" → drop
        original_ing = ing
        ing = re.sub(r'\(\s*(?:may\s+)?contains?.*?\)', '', ing, flags=re.I)
        if original_ing != ing:
            logger.info(f"Removed contains clause: '{original_ing}' -> '{ing}'")

        # drop bracketed/parenthetical sublists; keep head
        original_ing = ing
        ing = re.sub(r'\[[^\]]*\]', '', ing)
        ing = re.sub(r'\([^)]*\)', '', ing)
        if original_ing != ing:
            logger.info(f"Removed brackets/parentheses: '{original_ing}' -> '{ing}'")

        # strip leading "contains/less than 2% ..." prefixes
        ing = _strip_contains_prefix(ing)

        # leading connectors
        original_ing = ing
        ing = re.sub(r'^(?:and\/or|and or|and|or|andor)\s+', '', ing)
        if original_ing != ing:
            logger.debug(f"Removed leading connector: '{original_ing}' -> '{ing}'")

        # descriptive lead-ins
        original_ing = ing
        ing = re.sub(r'^\s*(?:made\s+with|made\s+of|made\s+from)\s*:\s*', '', ing)
        ing = re.sub(r'^\s*(?:includes?|including|containing)\s*:\s*', '', ing)
        ing = re.sub(r'^\s*if[^,;]*', '', ing)
        if original_ing != ing:
            logger.debug(f"Removed descriptive lead-in: '{original_ing}' -> '{ing}'")

        # fallback prefix
        original_ing = ing
        ing = re.sub(r'^\s*\d+%?\s*(?:or\s+less|or\s+more)?\s*(?:of\s*:?)\s*', '', ing)
        if original_ing != ing:
            logger.debug(f"Removed fallback prefix: '{original_ing}' -> '{ing}'")

        # normalize
        original_ing = ing
        ing = re.sub(r'[^a-z0-9\s\-&/#]', '', ing).strip()
        ing = re.sub(r'\s+', ' ', ing)
        ing = ing.replace('andor', 'and/or')
        if ing in ('natural flavors', 'natural flavoring'): 
            ing = 'natural flavor'
            logger.debug("Normalized 'natural flavors' -> 'natural flavor'")
        if ing == 'thiamine mononitrate': 
            ing = 'thiamin mononitrate'
            logger.debug("Normalized 'thiamine mononitrate' -> 'thiamin mononitrate'")
        ing = re.sub(r'(?:\s+(?:and|or|and\/or))+$', '', ing)
        
        if original_ing != ing:
            logger.debug(f"Normalized text: '{original_ing}' -> '{ing}'")

        if ing: 
            cleaned.append(ing)
            logger.debug(f"Added cleaned ingredient: '{ing}'")
        else:
            logger.debug(f"Skipped empty ingredient after cleaning")

    logger.info(f"Cleaning complete. Input: {len(parts)} parts, Output: {len(cleaned)} ingredients")
    return cleaned
