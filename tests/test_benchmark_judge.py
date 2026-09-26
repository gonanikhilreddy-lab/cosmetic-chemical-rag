import unittest
from unittest.mock import patch

from scripts import benchmark_retrieval


class BenchmarkJudgeTests(unittest.TestCase):
    def test_judge_prompt_does_not_receive_gold_labels(self):
        class FakeResponse:
            content = '{"relevant_ids":[0]}'
            response_metadata = {"model": "qwen3:4b", "prompt_eval_count": 10, "eval_count": 2}

        class FakeClient:
            def close(self):
                pass

        class FakeModel:
            last_prompt = ""

            def __init__(self, **_kwargs):
                self._client = FakeClient()

            def invoke(self, prompt):
                FakeModel.last_prompt = prompt
                return FakeResponse()

        rows = [{"query": "wood alcohol", "chemical_name": "Methanol", "relevant": {"Methanol"}}]
        with patch.object(benchmark_retrieval, "ChatOllama", FakeModel):
            judged, usage = benchmark_retrieval._llm_judge(rows)
        self.assertNotIn("expected_relevant", FakeModel.last_prompt)
        judge_input = FakeModel.last_prompt.split("\n\n", 1)[1]
        self.assertNotIn('"relevant":', judge_input)
        self.assertTrue(judged[0]["llm_judged_relevant"])
        self.assertEqual(usage["api_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()