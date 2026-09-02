import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    # The fetch calls are mocked in these unit tests. Keep the test suite
    # importable in lightweight CI environments that omit the runtime client.
    sys.modules["requests"] = MagicMock()

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from family_classification import (
    fetch_uniprot_annotation_batch,
    normalize_uniprot_accession,
)


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = {}

    def json(self):
        return self._payload


def _entry(primary, secondary=None):
    return {
        "primaryAccession": primary,
        "secondaryAccessions": secondary or [],
        "keywords": [{"name": "Kinase"}],
        "comments": [{
            "commentType": "SIMILARITY",
            "texts": [{"value": "Belongs to the protein kinase family."}],
        }],
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Test kinase"}},
        },
    }


class UniProtFetchTests(unittest.TestCase):
    def test_accession_validation_accepts_six_and_ten_character_ids(self):
        self.assertEqual(normalize_uniprot_accession(" p04637 "), "P04637")
        self.assertEqual(normalize_uniprot_accession("A0A0A0ABC1"), "A0A0A0ABC1")
        self.assertEqual(normalize_uniprot_accession("P04637-2"), "P04637-2")
        self.assertIsNone(normalize_uniprot_accession("P04637;Q9Y243"))

    @patch("family_classification.time.sleep", return_value=None)
    @patch("family_classification.requests.get")
    def test_http_400_batch_is_split_until_queries_succeed(self, mock_get, _sleep):
        def side_effect(_url, params, headers, timeout):
            accessions = [term.split(":", 1)[1]
                          for term in params["query"].split(" OR ")]
            if len(accessions) > 1:
                return _FakeResponse(400, text="query too long")
            return _FakeResponse(200, {
                "results": [_entry(accessions[0])],
            })

        mock_get.side_effect = side_effect
        annotations, report = fetch_uniprot_annotation_batch(
            ["P04637", "Q9Y243"], batch_size=2, return_report=True,
        )

        self.assertEqual(set(annotations), {"P04637", "Q9Y243"})
        self.assertEqual(report["retrieved"], 2)
        self.assertEqual(report["terminal_failed_ids"], [])

    @patch("family_classification.time.sleep", return_value=None)
    @patch("family_classification.requests.get")
    def test_secondary_accession_is_cached_under_requested_id(self, mock_get, _sleep):
        mock_get.return_value = _FakeResponse(200, {
            "results": [_entry("P04637", secondary=["Q9Y2X3"])],
        })

        annotations = fetch_uniprot_annotation_batch(["Q9Y2X3"])

        self.assertIn("Q9Y2X3", annotations)
        self.assertEqual(annotations["Q9Y2X3"]["keywords"], ["Kinase"])

    @patch("family_classification.requests.get")
    def test_malformed_ids_are_reported_without_api_request(self, mock_get):
        annotations, report = fetch_uniprot_annotation_batch(
            ["P04637;Q9Y243"], return_report=True,
        )

        self.assertEqual(annotations, {})
        self.assertEqual(report["invalid_ids"], ["P04637;Q9Y243"])
        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
