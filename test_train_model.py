import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import easydata as ed
import train_model


class TrainingSamplesTests(unittest.TestCase):
    def test_labeled_messages_extend_builtin_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "messages")
            ed.create_database(database)
            ed.give_id_data(database, "message:1:1", {"text": "важное", "reverence": 1})
            ed.give_id_data(database, "message:1:2", {"text": "шум", "reverence": 0})
            ed.give_id_data(database, "message:1:3", {"text": "без оценки", "reverence": -1})
            ed.give_id_data(database, "message:1:4", {"text": "старое", "reference": 1})
            ed.give_id_data(database, "settings", {"text": "не сообщение", "reverence": 1})

            with patch.object(train_model, "DB_NAME", database):
                samples = train_model.load_training_samples()

        self.assertEqual(samples[:len(train_model.SAMPLES)], train_model.SAMPLES)
        self.assertEqual(samples[len(train_model.SAMPLES):], [
            ("важное", 1),
            ("шум", 0),
            ("старое", 1),
        ])


if __name__ == "__main__":
    unittest.main()
