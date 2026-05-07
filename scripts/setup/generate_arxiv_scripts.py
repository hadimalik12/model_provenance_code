"""Refined ArXiv-specific experiment scripts generator (Precise Path Fix)."""

import os
import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.join(REPO_ROOT, "scripts", "experiments")

def transform_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    new_content = content
    
    # 1. Dataset Header Transformation
    new_content = new_content.replace("Dataset: MIMIR GitHub", "Dataset: MIMIR ArXiv")
    
    # 2. Specific Path Transformations (Pre-prefixed to avoid double-hits)
    new_content = new_content.replace("data/processed/mimir_github_nonmember_control_seed0", "data/processed/mimir_arxiv_nonmember_control_seed0")
    new_content = new_content.replace("data/processed/nonmember_control_seed0", "data/processed/mimir_arxiv_nonmember_control_seed0")
    
    new_content = new_content.replace("data/scores/mimir_github_nonmember_control_seed0", "data/scores/mimir_arxiv_nonmember_control_seed0")
    new_content = new_content.replace("data/scores/nonmember_control_seed0", "data/scores/mimir_arxiv_nonmember_control_seed0")

    new_content = new_content.replace("outputs/runs/mimir_github_nonmember_control_seed0", "outputs/runs/mimir_arxiv_nonmember_control_seed0")
    new_content = new_content.replace("outputs/runs/nonmember_control_seed0", "outputs/runs/mimir_arxiv_nonmember_control_seed0")

    # 3. Standard Main Path Transformations
    new_content = new_content.replace("mimir_github", "mimir_arxiv")
    
    # 4. Data Preparation Script Transformation
    new_content = new_content.replace("prepare_mimir_github.py", "prepare_mimir_arxiv.py")
    new_content = new_content.replace("prepare_mimir_github_nonmember_control.py", "prepare_mimir_arxiv_nonmember_control.py")
    
    # 5. Parent results cross-reference fix
    new_content = new_content.replace("run_parent_pythia_1_4b.py", "run_parent_pythia_1_4b_arxiv.py")
    new_content = new_content.replace("run_parent_pythia_1b.py", "run_parent_pythia_1b_arxiv.py")
    new_content = new_content.replace("run_parent_pythia_6_9b.py", "run_parent_pythia_6_9b_arxiv.py")
    new_content = new_content.replace("run_parent_pythia_12b.py", "run_parent_pythia_12b_arxiv.py")
    
    # Construct new filename
    base = os.path.basename(file_path)
    name, ext = os.path.splitext(base)
    new_filename = f"{name}_arxiv{ext}"
    new_path = os.path.join(EXP_DIR, new_filename)
    
    with open(new_path, "w") as f:
        f.write(new_content)
    
    print(f"Created: {new_filename}")

def main():
    parent_files = glob.glob(os.path.join(EXP_DIR, "run_parent_*.py"))
    target_files = glob.glob(os.path.join(EXP_DIR, "run_target_*.py"))
    
    all_files = [f for f in parent_files + target_files if "_arxiv.py" not in f]
    
    print(f"Transforming {len(all_files)} scripts with Precise ArXiv Paths...")
    for f in all_files:
        transform_file(f)

if __name__ == "__main__":
    main()
