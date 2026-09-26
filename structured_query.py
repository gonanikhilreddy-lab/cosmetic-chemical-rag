from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from src.tools.structured_query import *


if __name__ == "__main__":
    results = find_by_category("Skin Care Products")
    print(results.head(20).to_string(index=False))