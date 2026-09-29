"""Checks for the keyed intervention and final fixed-threshold statistic."""
import math
import unittest

from experiments.table_08_controlled_provenance.shared.audit import two_sided_randomization_p
from experiments.table_08_controlled_provenance.shared.downstream import normalize, select
from experiments.table_08_controlled_provenance.shared.prepare import (canonical_row, choose,
    choose_global_shuffle, fit_model_window, split_audit)
from experiments.table_08_controlled_provenance.shared.train import tokenize_row


class Table06Tests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"prompt": f"Write code for operation {i} with a stable description",
                      "response": f"def operation_{i}(value):\n    return value + {i}  # unique result"}
                     for i in range(50)]
        self.limits = {"max_prompt_chars": 300, "max_response_chars": 1100}

    def test_keyed_split_is_reproducible_disjoint_and_sensitive_to_key(self):
        s, c, counts = choose(self.rows + [self.rows[3]], b"a" * 32, self.limits, 10)
        s2, c2, _ = choose(reversed(self.rows), b"a" * 32, self.limits, 10)
        ids = lambda rows: [row["id"] for row in rows]
        self.assertEqual(ids(s), ids(s2))
        self.assertEqual(ids(c), ids(c2))
        self.assertFalse(set(ids(s)) & set(ids(c)))
        self.assertEqual(counts["eligible_unique"], 50)
        other_s, _, _ = choose(self.rows, b"b" * 32, self.limits, 10)
        self.assertNotEqual(ids(s), ids(other_s))
        excluded_limits = {**self.limits, "exclude_ids": [s[0]["id"]]}
        without_known, other, counts = choose(self.rows, b"a" * 32, excluded_limits, 10)
        self.assertNotIn(s[0]["id"], ids(without_known) + ids(other))
        self.assertEqual(counts["excluded_known_pile_github_overlap"], 1)
        cal, test = split_audit(s, c, 3, 4)
        self.assertFalse({r["id"] for r in cal} & {r["id"] for r in test})
        self.assertEqual([r["label"] for r in cal].count(1), 3)
        self.assertEqual([r["label"] for r in test].count(0), 4)

    def test_global_shuffle_is_reproducible_disjoint_and_keeps_arm_sizes(self):
        selected, control, counts = choose_global_shuffle(self.rows, b"a" * 32,
            self.limits, 10)
        selected_again, control_again, _ = choose_global_shuffle(self.rows, b"a" * 32,
            self.limits, 10)
        ids = lambda rows: [row["id"] for row in rows]
        self.assertEqual(ids(selected), ids(selected_again))
        self.assertEqual(ids(control), ids(control_again))
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(control), 10)
        self.assertFalse(set(ids(selected)) & set(ids(control)))
        self.assertEqual(counts["s_count"], 10)
        self.assertEqual(counts["s_prime_count"], 10)

    def test_instructcoder_pair_uses_instruction_existing_code_and_edit(self):
        limits = {"format": "instructcoder", "max_prompt_chars": 2200,
                  "max_response_chars": 3000}
        raw = {"instruction": "Replace the operation with multiplication.",
               "input": "def calculate(value):\n    return value + 3",
               "output": "def calculate(value):\n    return value * 3"}
        row = canonical_row(raw, limits)
        self.assertIsNotNone(row)
        self.assertIn(raw["instruction"], row["prompt"])
        self.assertIn(raw["input"], row["prompt"])
        self.assertEqual(raw["output"], row["response"])
        class ToyTokenizer:
            def __call__(self, value, add_special_tokens=False):
                return {"input_ids": [ord(char) for char in value]}

            def decode(self, ids, clean_up_tokenization_spaces=False):
                return "".join(chr(i) for i in ids)
        fitted = fit_model_window([row], ToyTokenizer(), 96)[0]
        self.assertTrue(fitted["prompt"].endswith("### Answer\n"))
        self.assertEqual(fitted["text"], fitted["prompt"] + fitted["response"])
        self.assertLessEqual(len(fitted["text"]), 96)

    def test_randomization_p_is_exact_and_symmetric(self):
        self.assertAlmostEqual(two_sided_randomization_p(0, 0, 4), 1.0)
        self.assertAlmostEqual(two_sided_randomization_p(4, 0, 4), 2 / math.comb(8, 4))
        self.assertAlmostEqual(two_sided_randomization_p(4, 0, 4),
                               two_sided_randomization_p(0, 4, 4))

    def test_train_labels_mask_prompt_and_keep_answer(self):
        class ToyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(c) for c in text]}
        output = tokenize_row({"prompt": "abc", "response": "xyz"}, ToyTokenizer(), 5)
        self.assertEqual(output["input_ids"], [ord(c) for c in "bcxyz"])
        self.assertEqual(output["labels"], [-100, -100] + [ord(c) for c in "xyz"])

    def test_saved_audit_text_matches_training_parts_and_window(self):
        class ToyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(c) for c in text]}

            def decode(self, ids, clean_up_tokenization_spaces=False):
                return "".join(chr(i) for i in ids)
        selected, _, _ = choose(self.rows, b"c" * 32, self.limits, 2)
        fitted = fit_model_window(selected, ToyTokenizer(), 64)
        for row in fitted:
            self.assertEqual(row["text"], row["prompt"] + row["response"])
            self.assertIn("### Task", row["prompt"])
            self.assertLessEqual(len(row["text"]), 64)

    def test_downstream_selection_excludes_shard_and_parses_all_profiles(self):
        hh = normalize("hh_sft", {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello"})
        self.assertEqual(hh["response"], "Hello")
        tulu = normalize("tulu_sft", {"messages": [{"role": "user", "content": "Hi"},
                                                     {"role": "assistant", "content": "Hello"}]})
        self.assertEqual(tulu["response"], "Hello")
        lamini = normalize("lamini_sft", {"instruction": "Say hi", "response": "Hello"})
        self.assertEqual(lamini["response"], "Hello")
        finance = normalize("finance_sft", {"instruction": "Define APR", "input": "loans",
                                               "output": "Annual percentage rate"})
        self.assertIn("loans", finance["prompt"])
        self.assertEqual(finance["response"], "Annual percentage rate")
        rows = [{"chosen": f"\n\nHuman: Question {i}\n\nAssistant: Answer {i}"} for i in range(12)]
        excluded = {normalize("hh_sft", rows[0])["id"]}
        selected = select(rows, "hh_sft", 5, excluded)
        self.assertEqual(len(selected), 5)
        self.assertFalse(excluded & {r["id"] for r in selected})


if __name__ == "__main__":
    unittest.main()
