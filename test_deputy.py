import unittest
from unittest.mock import Mock, patch

import deputy


class DeputyTests(unittest.TestCase):
    @patch("deputy.ed.get_ids_data")
    def test_rag_loads_only_important_messages(self, get_ids_data):
        get_ids_data.return_value = {
            "message:1:1": {
                "message_id": 1,
                "source_id": 1,
                "text": "classifier important",
                "show": True,
                "score": 0.7,
                "reference": -1,
            },
            "message:1:2": {
                "message_id": 2,
                "source_id": 1,
                "text": "human important",
                "show": False,
                "score": 0.1,
                "reference": 1,
            },
            "message:1:3": {
                "message_id": 3,
                "source_id": 1,
                "text": "hidden even with high score",
                "show": False,
                "score": 0.99,
                "reference": -1,
            },
            "message:1:4": {
                "message_id": 4,
                "source_id": 1,
                "text": "human rejected",
                "show": True,
                "score": 0.99,
                "reference": 0,
            },
        }

        chunks = deputy.load_important_messages(
            "database", exclude_source_id=1, exclude_message_id=1
        )

        self.assertEqual([chunk["source"] for chunk in chunks], ["message:1:2"])

    @patch("deputy.httpx.post")
    @patch("deputy.search_important_messages")
    def test_answer_uses_selected_model_and_retrieved_context(self, search, post):
        search.return_value = [{"text": "Дедлайн перенесли на вторник"}]
        response = Mock()
        response.json.return_value = {"message": {"content": "Во вторник."}}
        post.return_value = response

        answer = deputy.answer_question("database", "Когда дедлайн?", "llama3.1:8b")

        self.assertEqual(answer, "Во вторник.")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "llama3.1:8b")
        self.assertIn("Дедлайн перенесли", payload["messages"][1]["content"])
        self.assertFalse(payload["stream"])

    @patch("deputy.httpx.post")
    @patch("deputy.search_important_messages", return_value=[])
    def test_no_context_does_not_call_ollama(self, search, post):
        answer = deputy.answer_question("database", "Неизвестный вопрос")

        self.assertEqual(answer, deputy.NO_INFORMATION)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
