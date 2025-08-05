"""
Jinja2 templating setup for PDF reports
"""
import os
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

# Get the reports directory path
REPORTS_DIR = Path(__file__).parent
TEMPLATES_DIR = REPORTS_DIR / "templates"
ASSETS_DIR = REPORTS_DIR / "assets"

# Create Jinja2 environment
env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=True
)

def get_asset_path(filename: str) -> str:
    """Get the full path to an asset file"""
    return str(ASSETS_DIR / filename)