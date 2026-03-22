import re
from pathlib import Path
from bs4 import BeautifulSoup


def natural_key(path: Path):
    """
    Sort key that treats digit runs in filenames as integers, so
    _1, _2, ..., _10 instead of _1, _10, _11, _2.

    Works for names like 'X and IP Quiz (Week 10)_12.jpg'.
    """
    name = path.name
    # Split into digit and non-digit chunks: ["X and IP Quiz (Week ", "10", ")_", "12", ".jpg"]
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def clean_html_text(html: str) -> str:
    """
    Clean HTML text by removing unwanted tags and extracting plain text.
    
    Removes <link>, <script>, and <style> tags entirely, then extracts
    clean text content. Useful for cleaning Canvas text submissions that
    contain HTML with stylesheets and scripts.
    
    Args:
        html: Raw HTML string (e.g., from Canvas text submission)
    
    Returns:
        Cleaned plain text with basic formatting preserved
        
    Example:
        Input: '<link rel="stylesheet"...>Yes I do plan...<script>...</script>'
        Output: 'Yes I do plan...'
    """
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script, style, and link tags entirely
    for tag in soup(['script', 'style', 'link']):
        tag.decompose()
    
    # Extract clean text with space separator
    # separator=' ' helps preserve basic formatting between paragraphs
    # strip=True removes leading/trailing whitespace
    return soup.get_text(separator=' ', strip=True)