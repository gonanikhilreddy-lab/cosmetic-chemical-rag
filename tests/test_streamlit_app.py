import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_without_exceptions(self):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py", default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Chemical disclosure search" in item.value for item in app.markdown))
        self.assertTrue(any(button.label == "Search disclosures" for button in app.button))
        self.assertTrue(any("Question 1" in option for option in app.selectbox[0].options))
        self.assertIn("question-highlight", "\n".join(item.value for item in app.markdown))

    def test_sample_question_runs_and_renders_evidence(self):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py", default_timeout=60).run()
        app.toggle[0].set_value(False)
        app.selectbox[0].select("Question 3 · Discontinued products")
        next(button for button in app.button if button.label == "Load selected question").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("4 products" in item.value for item in app.caption))
        self.assertTrue(any("Lotion Hand Soap" in item.value for item in app.markdown))
        self.assertTrue(any("Performance · latency and token usage" in item.label for item in app.expander))
        self.assertGreaterEqual(len(app.dataframe), 3)

    def test_chat_retains_turns_and_uses_previous_filters(self):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py", default_timeout=60).run()
        app.toggle[0].set_value(False)
        app.text_input[0].set_value("Which products from New Avon LLC contain Titanium dioxide?")
        next(button for button in app.button if button.label == "Search disclosures").click().run()
        app.text_input[0].set_value("What about products discontinued in 2010?")
        next(button for button in app.button if button.label == "Search disclosures").click().run()
        self.assertFalse(app.exception)
        displayed = "\n".join(item.value for item in app.markdown)
        self.assertIn("Which products from New Avon LLC contain Titanium dioxide?", displayed)
        self.assertIn("What about products discontinued in 2010?", displayed)
        history = app.session_state["chat_history"]
        self.assertEqual(len(history), 2)
        second_result = history[1]["result"]
        self.assertEqual(second_result["query_plan"]["inherited_entities"]["company"], "New Avon LLC")
        self.assertEqual(second_result["query_plan"]["inherited_entities"]["chemical"], "Titanium dioxide")
        self.assertEqual(second_result["query_plan"]["date_from"], "2010-01-01")
        self.assertGreater(second_result["summary"]["counts"]["product_count"], 0)
        self.assertTrue(all("CDPHId" in row and "ChemicalId" in row for row in second_result["evidence"]))

    def test_comparison_query_does_not_invent_lip_brand(self):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py", default_timeout=60).run()
        app.toggle[0].set_value(False)
        app.text_input[0].set_value(
            'Compare brands AVON and MARK for CAS 13463-67-7 in SubCategory '
            '"Lip Color - Lipsticks, Liners, and Pencils", discontinued in 2010.'
        )
        next(button for button in app.button if button.label == "Search disclosures").click().run()
        self.assertFalse(app.exception)
        history = app.session_state["chat_history"]
        result = history[-1]["result"]
        self.assertEqual(result["query_plan"]["comparisons"]["brand"], ["AVON", "MARK"])
        self.assertNotIn("Lip", result["query_plan"]["comparisons"]["brand"])
        self.assertNotIn("The name can refer", " ".join(result["warnings"]))


if __name__ == "__main__":
    unittest.main()