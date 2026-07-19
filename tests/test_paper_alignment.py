import json
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from prepare_endef_data import encode_endef_label, repair_chinese_labels
from modeling.social_debias import infer_bottleneck_dim
from scripts.compute_nli_p_entail import resolve_nli_label_indices
from scripts.prepare_nrc_emolex import load_long, load_translation_map, write_long
from utils.contrastive_dataloader import ContrastiveFakeNewsDataset
from utils.explanation_metrics import js_divergence, top_k_overlap
from utils.surface_features import SurfaceFeatureExtractor


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        value = len(text)
        return {
            "input_ids": torch.tensor([[value, 0]]),
            "attention_mask": torch.tensor([[1, 0]]),
        }


class PaperAlignmentTests(unittest.TestCase):
    def test_endef_labels_match_language_specific_loaders(self):
        self.assertEqual(encode_endef_label(0, "en"), 0)
        self.assertEqual(encode_endef_label(1, "en"), 1)
        self.assertEqual(encode_endef_label(0, "zh"), "real")
        self.assertEqual(encode_endef_label(1, "zh"), "fake")

    def test_repair_existing_endef_ch_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                (root / f"{split}.json").write_text(
                    json.dumps([{"label": 0}, {"label": 1}]),
                    encoding="utf-8",
                )
            repair_chinese_labels(root)
            repaired = json.loads((root / "val.json").read_text(encoding="utf-8"))
        self.assertEqual([row["label"] for row in repaired], ["real", "fake"])

    def test_top_k_overlap_uses_intersection_over_k(self):
        tokens_a = ["a", "b", "c", "d"]
        tokens_b = ["a", "b", "x", "y"]
        scores = [4.0, 3.0, 2.0, 1.0]
        self.assertEqual(
            top_k_overlap(tokens_a, scores, tokens_b, scores, k=4), 0.5
        )

    def test_js_divergence_does_not_require_common_tokens(self):
        value = js_divergence(
            ["a", "b", "c"], [3.0, 2.0, 1.0],
            ["x", "y", "z"], [3.0, 2.0, 1.0],
        )
        self.assertAlmostEqual(value, 0.0, places=7)

    def test_nli_indices_follow_model_metadata(self):
        indices = resolve_nli_label_indices({0: "neutral", 1: "contradiction", 2: "entailment"})
        self.assertEqual(
            indices, {"entailment": 2, "neutral": 0, "contradiction": 1}
        )

    def test_legacy_and_paper_bottleneck_detection(self):
        legacy = {"fact_classifier.weight": torch.zeros(2, 384)}
        paper = {"fact_classifier.weight": torch.zeros(2, 128)}
        self.assertEqual(infer_bottleneck_dim(legacy), 0)
        self.assertEqual(infer_bottleneck_dim(paper), 128)

    def test_nrc_long_tsv_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nrc.tsv"
            path.write_text(
                "alarm\tfear\t1\nalarm\tsurprise\t1\ncalm\tfear\t0\n",
                encoding="utf-8",
            )
            lexicon = SurfaceFeatureExtractor._load_lexicon(path)
        self.assertEqual(lexicon["alarm"], {"fear", "surprise"})
        self.assertNotIn("calm", lexicon)

    def test_nrc_translation_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            english = root / "english.tsv"
            translations = root / "translations.csv"
            chinese = root / "chinese.tsv"
            english.write_text(
                "alarm\tfear\t1\nalarm\tjoy\t0\n",
                encoding="utf-8",
            )
            translations.write_text(
                "English Word,Chinese-Simplified\nalarm,警报\n",
                encoding="utf-8",
            )
            rows = load_long(english)
            mapping, _, _ = load_translation_map(translations)
            written = write_long(
                [(zh, emotion, flag)
                 for word, emotion, flag in rows
                 for zh in mapping[word.lower()]],
                chinese,
            )
        self.assertEqual(written, [("警报", "fear", "1")])

    def test_p_entail_matches_selected_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig_path = Path(tmp) / "orig.pkl"
            adv_path = Path(tmp) / "adv.pkl"
            with orig_path.open("wb") as f:
                pickle.dump({"news": ["original"], "labels": [1]}, f)
            with adv_path.open("wb") as f:
                pickle.dump({
                    "news": ["rewrite"], "labels": [1], "style": ["neutral"],
                    "orig_idx": [0], "p_entail": [0.37],
                }, f)
            dataset = ContrastiveFakeNewsDataset.from_pkl(
                str(orig_path), str(adv_path), FakeTokenizer()
            )
            sample = dataset[0]
        self.assertAlmostEqual(sample["p_entail"].item(), 0.37, places=6)


if __name__ == "__main__":
    unittest.main()
