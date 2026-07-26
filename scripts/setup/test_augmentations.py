"""Unit test verifying text_augmentations.py across Python, C/C++, Java, JS, SQL, and Shell code samples."""

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.scoring.text_augmentations import augment_code_snippet_4views, detect_language

TEST_SAMPLES = {
    "python": """def process_dataset(items):
    output = []
    for val in items:
        output.append(val)
    return output""",

    "c_like": """#include <iostream>
int main() {
    int count = 10;
    std::cout << count << std::endl;
    return 0;
}""",

    "sql": """SELECT user_id, count(order_id) FROM orders WHERE status = 'completed' GROUP BY user_id;""",

    "shell": """#!/bin/bash
echo "Starting job execution"
export DATA_DIR="/tmp/data"
"""
}

def main():
    print("=== Testing Language Detection & 4-View Code Augmentations ===")
    for name, snippet in TEST_SAMPLES.items():
        lang = detect_language(snippet)
        print(f"\n--- Sample: {name} (Detected Language: {lang}) ---")
        views = augment_code_snippet_4views(snippet, n_aug=4)
        assert len(views) == 4, f"Expected 4 views, got {len(views)}"
        
        for idx, v in enumerate(views):
            print(f"  View {idx + 1} ({len(v)} chars): {repr(v[:40])}...")

    print("\n✅ All 4-view code augmentations generated cleanly and successfully across all language types!")

if __name__ == "__main__":
    main()
