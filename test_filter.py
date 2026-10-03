import unittest
from unittest.mock import patch

import filter as message_filter
from filter import analyze_message


class MessageFilterTests(unittest.TestCase):
    def setUp(self):
        # Юнит-тесты проверяют детерминированный fallback отдельно от большой
        # семантической модели. Полная модель проверяется launcher-командой analyze.
        self.semantics = patch.object(message_filter, "USE_SEMANTICS", False)
        self.semantics.start()
        self.model = patch.object(message_filter, "get_model_probability", return_value=None)
        self.model.start()

    def tearDown(self):
        self.model.stop()
        self.semantics.stop()

    def test_event_with_date_and_time_is_important(self):
        result = analyze_message("Встреча команды 12.10 в 14:30")
        self.assertIn("EVENT", result.tags)
        self.assertTrue(result.show)
        self.assertEqual(result.source, "fallback")

    def test_urgent_task_is_important(self):
        result = analyze_message("Срочно отправь отчет сегодня")
        self.assertIn("DEADLINE", result.tags)
        self.assertIn("ACTION", result.tags)
        self.assertTrue(result.show)

    def test_request_is_detected(self):
        result = analyze_message("Пожалуйста, помоги разобраться?")
        self.assertIn("ACTION", result.tags)
        self.assertEqual(result.features["question_count"], 1)

    def test_regular_message_is_not_important(self):
        result = analyze_message("Спасибо, было интересно")
        self.assertFalse(result.show)

    def test_goodbye_is_not_shown(self):
        result = analyze_message("До встречи")
        self.assertFalse(result.show)

    def test_empty_message_is_safe(self):
        result = analyze_message("")
        self.assertEqual(result.source, "empty")
        self.assertFalse(result.show)


if __name__ == "__main__":
    unittest.main()
