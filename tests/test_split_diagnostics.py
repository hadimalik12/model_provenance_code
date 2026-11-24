import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.shard_audit.data.split_diagnostics import (
    classifier_two_sample_test,
    image_duplicate_diagnostics,
    lexical_js_divergence,
    numeric_feature_diagnostics,
    resolve_image_path,
    text_duplicate_diagnostics,
)


def test_lexical_and_structural_diagnostics_are_finite():
    lexical = lexical_js_divergence(["alpha alpha beta"], ["gamma gamma beta"])
    assert 0 < lexical["js_divergence_bits"] <= 1
    numeric = numeric_feature_diagnostics({"length": [5, 6, 10, 11]}, [1, 1, 0, 0])
    assert numeric["length"]["member_mean"] == 5.5
    assert numeric["length"]["ks_statistic"] == 1.0
    assert numeric["length"]["ks_pvalue_holm"] == numeric["length"]["ks_pvalue_unadjusted"]


def test_text_duplicates_find_cross_shard_exact_and_near_pairs():
    texts = [
        "one two three four five six seven",
        "one two three four five six seven",
        "unrelated words appear only over here",
        "another completely different short document",
    ]
    result = text_duplicate_diagnostics(texts, [1, 0, 1, 0], threshold=0.8)
    assert result["exact_cross_shard_pairs"] == 1
    assert result["near_cross_shard_pairs"] >= 1


def test_c2st_detects_large_shift_reproducibly():
    rng = np.random.default_rng(3)
    labels = np.repeat([0, 1], 80)
    features = rng.normal(size=(160, 8)) + labels[:, None] * 3.0
    result = classifier_two_sample_test(features, labels, seed=7, permutations=20)
    assert result["balanced_accuracy"] > 0.9
    assert result["permutation_pvalue"] <= 2 / 21


def test_image_duplicate_and_path_resolution(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    arrays = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.full((16, 16, 3), 255, dtype=np.uint8),
    ]
    paths = []
    for i, array in enumerate(arrays):
        path = image_dir / f"{i}.png"
        Image.fromarray(array).save(path)
        paths.append(path)
    result = image_duplicate_diagnostics(paths, [1, 0, 0])
    assert result["exact_cross_shard_pairs"] == 1
    assert resolve_image_path({"id": "x", "image_path": "0.png"}, tmp_path) == paths[0]
