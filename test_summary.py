import json
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import summary


def message(message_id, author_id=1, source_id=10, topic_id=None, show=False, score=0.1):
    return {
        "message_id": message_id,
        "author_id": author_id,
        "source_id": source_id,
        "topic_id": topic_id,
        "show": show,
        "score": score,
        "text": f"message {message_id}",
    }


class SummaryTests(unittest.TestCase):
    @patch("summary.ed.get_ids_data")
    def test_load_messages_filters_range_and_sorts(self, get_ids_data):
        get_ids_data.return_value = {
            "message:10:2": {
                "message_id": 2,
                "date": "2026-01-02T12:00:00+00:00",
            },
            "message:10:1": {
                "message_id": 1,
                "date": "2026-01-01T12:00:00",
            },
            "message:10:3": {"message_id": 3, "date": "not-a-date"},
        }

        messages = summary.load_messages(
            "database", datetime(2026, 1, 1), datetime(2026, 1, 3)
        )

        self.assertEqual([item["message_id"] for item in messages], [1, 2])
        get_ids_data.assert_called_once_with("database", "message:")

    def test_context_does_not_cross_source_or_topic(self):
        messages = [
            message(1, source_id=10),
            message(2, source_id=20),
            message(3, source_id=10, show=True),
            message(4, source_id=20),
            message(5, source_id=10, topic_id=7),
        ]

        selected = summary.select_messages(messages)

        self.assertEqual([item["message_id"] for item in selected], [1, 3])

    def test_high_score_false_message_is_selected(self):
        selected = summary.select_messages([message(1, score=0.35)])
        self.assertEqual([item["message_id"] for item in selected], [1])

    def test_message_link_supports_public_and_private_topics(self):
        public = message(10, topic_id=7)
        public["source_username"] = "project_chat"
        private = message(11, source_id=-1001234567890)

        self.assertEqual(
            summary.message_link(public),
            "https://t.me/project_chat/7/10",
        )
        self.assertEqual(
            summary.message_link(private),
            "https://t.me/c/1234567890/11",
        )

    def test_private_dialog_has_no_message_link(self):
        self.assertIsNone(summary.message_link(message(1, source_id=12345)))

    @patch("summary.ask_ollama")
    def test_authors_are_sent_in_separate_requests(self, ask_ollama):
        ask_ollama.side_effect = [
            {
                "tasks": [{"text": "first", "message_ids": [1]}],
                "events": [],
                "messages": [],
            },
            {
                "tasks": [],
                "events": [],
                "messages": [{"text": "second", "message_ids": [2]}],
            },
        ]

        result = summary.summarize_messages(
            [message(1, author_id=10, show=True), message(2, author_id=20, show=True)]
        )

        self.assertEqual(ask_ollama.call_count, 2)
        self.assertEqual(
            result["tasks"],
            [{"text": "first", "message_ids": [1], "author_id": 10}],
        )
        self.assertEqual(
            result["messages"],
            [{"text": "second", "message_ids": [2], "author_id": 20}],
        )
        self.assertNotIn("id=2", ask_ollama.call_args_list[0].args[0])
        self.assertNotIn("id=1", ask_ollama.call_args_list[1].args[0])

    @patch("summary.ask_ollama")
    def test_multiple_chunks_get_one_merge_request(self, ask_ollama):
        messages = [message(index, show=True) for index in range(31)]
        ask_ollama.side_effect = [
            {
                "tasks": [{"text": "part one", "message_ids": [0]}],
                "events": [],
                "messages": [],
            },
            {
                "tasks": [],
                "events": [{"text": "part two", "message_ids": [30]}],
                "messages": [],
            },
            {
                "tasks": [{"text": "merged task", "message_ids": ["0", 999]}],
                "events": [{"text": "merged event", "message_ids": [30]}],
                "messages": [],
            },
        ]

        result = summary.summarize_author(messages)

        self.assertEqual(ask_ollama.call_count, 3)
        self.assertEqual(
            result,
            {
                "tasks": [{"text": "merged task", "message_ids": [0]}],
                "events": [{"text": "merged event", "message_ids": [30]}],
                "messages": [],
            },
        )

    @patch("summary.ask_ollama")
    def test_selected_model_is_used_for_summary(self, ask_ollama):
        ask_ollama.return_value = {"tasks": [], "events": [], "messages": []}

        summary.summarize_author(
            [message(1, show=True)], summary.OLLAMA_MODELS["quality"]
        )

        self.assertEqual(ask_ollama.call_args.args[1], "llama3.1:8b")

    @patch("summary.ask_ollama")
    def test_summary_adds_links_to_source_messages(self, ask_ollama):
        ask_ollama.return_value = {
            "tasks": [{"text": "do it", "message_ids": [1]}],
            "events": [],
            "messages": [],
        }
        source_message = message(1, source_id=-1001234567890, show=True)

        result = summary.summarize_messages([source_message])

        self.assertEqual(
            result["tasks"][0]["message_links"],
            [{"message_id": 1, "url": "https://t.me/c/1234567890/1"}],
        )

    def test_chunking_also_limits_prompt_size(self):
        messages = [message(1, show=True), message(2, show=True)]
        messages[0]["text"] = "a" * 80
        messages[1]["text"] = "b" * 80

        with patch.object(summary, "MAX_CHUNK_CHARS", 150):
            chunks = summary.chunk_messages(messages)

        self.assertEqual([[item["message_id"] for item in chunk] for chunk in chunks], [[1], [2]])

    def test_clean_result_keeps_categories_and_valid_ids(self):
        result = summary.clean_result(
            {
                "tasks": [
                    {"text": "do it", "message_ids": ["1", "invalid", 2]},
                    {"text": "unsupported", "message_ids": [999]},
                ],
                "events": [],
                "messages": [{"text": "notice", "message_ids": [1]}],
            },
            {1},
        )
        self.assertEqual(
            result,
            {
                "tasks": [{"text": "do it", "message_ids": [1]}],
                "events": [],
                "messages": [{"text": "notice", "message_ids": [1]}],
            },
        )

    @patch("summary.httpx.post")
    def test_ollama_uses_json_mode(self, post):
        response = Mock()
        response.json.return_value = {
            "response": json.dumps(
                {
                    "tasks": [{"text": "done", "message_ids": [1]}],
                    "events": [],
                    "messages": [],
                }
            )
        }
        post.return_value = response

        result = summary.ask_ollama("prompt")

        self.assertEqual(result["tasks"][0]["text"], "done")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen3:0.6b")
        self.assertEqual(post.call_args.kwargs["json"]["format"], "json")
        self.assertFalse(post.call_args.kwargs["json"]["stream"])
        self.assertEqual(post.call_args.kwargs["json"]["options"]["temperature"], 0)
        response.raise_for_status.assert_called_once_with()

    @patch("summary.httpx.post")
    def test_ollama_can_use_quality_model(self, post):
        response = Mock()
        response.json.return_value = {
            "response": json.dumps({"tasks": [], "events": [], "messages": []})
        }
        post.return_value = response

        summary.ask_ollama("prompt", summary.OLLAMA_MODELS["quality"])

        self.assertEqual(post.call_args.kwargs["json"]["model"], "llama3.1:8b")


if __name__ == "__main__":
    unittest.main()
