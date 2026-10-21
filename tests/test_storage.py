from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from edge_runtime.storage import EventStore


class StorageTests(unittest.TestCase):
    def test_submit_performs_no_database_or_file_write(self):
        with TemporaryDirectory() as directory:
            store = EventStore(Path(directory))
            with patch.object(store, "connect", side_effect=AssertionError("synchronous database access")):
                with patch.object(Path, "write_bytes", side_effect=AssertionError("synchronous media write")):
                    self.assertTrue(store.submit("r", "s", "k", {"sequence": 1}, b"jpeg"))
            self.assertTrue(store.close()["flushed"])

    def test_payload_and_media_are_snapshotted_before_start(self):
        with TemporaryDirectory() as directory:
            store = EventStore(Path(directory))
            payload = {"capture": {"sequence": 1}}
            media = bytearray(b"original")
            self.assertTrue(store.submit("roads", "one", "candidate", payload, media))
            payload["capture"]["sequence"] = 999
            media[:] = b"modified"
            store.start()
            store.queue.join()
            outcome = store.close()
            row = store.list()[0]
            self.assertEqual(row["payload"]["capture"]["sequence"], 1)
            self.assertEqual((store.media / row["media_name"]).read_bytes(), b"original")
            self.assertTrue(outcome["flushed"])

    def test_invalid_json_never_creates_media_or_pending_work(self):
        with TemporaryDirectory() as directory:
            store = EventStore(Path(directory), max_media_bytes=10)
            self.assertFalse(store.submit("roads", "one", "candidate", {"bad": float("nan")}, b"jpeg"))
            self.assertEqual(store.status()["lost_events"], 1)
            self.assertEqual(store.status()["pending"], 0)
            self.assertEqual(list(store.media.iterdir()), [])
            store.close()

    def test_item_payload_media_and_total_queue_byte_limits(self):
        with TemporaryDirectory() as directory:
            store = EventStore(Path(directory), max_media_bytes=10)
            self.assertFalse(store.submit("r", "s", "k", {}, b"x" * 11))
            with patch.object(store, "MAX_PAYLOAD_BYTES", 10):
                self.assertFalse(store.submit("r", "s", "k", {"x": "too large"}))
            with patch.object(store, "MAX_QUEUE_BYTES", 150):
                self.assertTrue(store.submit("r", "s", "k", {}))
                self.assertFalse(store.submit("r", "s", "k", {}))
            self.assertLessEqual(store.status()["pending_bytes"], 150)
            self.assertTrue(store.close()["flushed"])

    def test_queue_item_capacity_and_postclose_rejection(self):
        with TemporaryDirectory() as directory:
            store = EventStore(Path(directory))
            for i in range(64):
                self.assertTrue(store.submit("r", "s", "k", {"i": i}))
            self.assertFalse(store.submit("r", "s", "k", {}))
            self.assertTrue(store.close()["flushed"])
            self.assertFalse(store.submit("r", "s", "k", {}))
            self.assertEqual(store.status()["pending"], 0)
            self.assertEqual(store.status()["lost_events"], 2)

    def test_retention_rows_and_media_stay_bounded(self):
        with TemporaryDirectory() as directory:
            store = EventStore(Path(directory), max_events=2, max_media_bytes=10)
            store.start()
            for i in range(4):
                self.assertTrue(store.submit("r", "s", "k", {"i": i}, b"x" * 6))
            store.queue.join()
            store.close()
            rows = store.list()
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["payload"]["i"] for r in rows}, {2, 3})
            self.assertLessEqual(sum(p.stat().st_size for p in store.media.glob("*.jpg")), 10)
            for row in rows:
                if row["media_name"]:
                    self.assertTrue((store.media / row["media_name"]).is_file())

    def test_startup_removes_orphans_temporary_and_missing_references(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = EventStore(root)
            store.submit("r", "s", "k", {}, b"jpeg")
            store.close()
            row = store.list()[0]
            (store.media / row["media_name"]).unlink()
            (store.media / "orphan.jpg").write_bytes(b"orphan")
            (store.media / "interrupted.tmp").write_bytes(b"partial")
            reopened = EventStore(root, max_media_bytes=2)
            self.assertEqual(list(reopened.media.iterdir()), [])
            self.assertIsNone(reopened.list()[0]["media_name"])
            reopened.close()

    def test_sql_failure_cleans_file_and_releases_connection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = EventStore(root)
            with store.connect() as conn:
                conn.execute("CREATE TRIGGER reject_insert BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'forced failure'); END")
            store.submit("r", "s", "k", {}, b"jpeg")
            store.close()
            self.assertEqual(store.status()["lost_events"], 1)
            self.assertEqual(list(store.media.iterdir()), [])
            self.assertEqual(store.list(), [])
            # Especially meaningful on Windows: no gc.collect() needed.
            renamed = root / "renamed.sqlite3"
            store.db.rename(renamed)
            renamed.rename(store.db)

    def test_close_reports_timeout_including_inflight_then_flushes(self):
        with TemporaryDirectory() as directory:
            store = EventStore(Path(directory))
            entered, release = threading.Event(), threading.Event()
            original = store._persist

            def blocked(event, jpeg):
                entered.set()
                release.wait(2)
                return original(event, jpeg)

            store._persist = blocked
            store.submit("r", "s", "k", {})
            store.start()
            self.assertTrue(entered.wait(1))
            with patch.object(store, "CLOSE_TIMEOUT_SECONDS", .01):
                outcome = store.close()
            self.assertTrue(outcome["flush_timed_out"])
            self.assertEqual(outcome["pending"], 1)
            self.assertEqual(outcome["in_flight"], 1)
            self.assertFalse(outcome["flushed"])
            release.set()
            self.assertTrue(store.close()["flushed"])

    def test_start_is_idempotent_and_limits_validate(self):
        with TemporaryDirectory() as directory:
            store = EventStore(Path(directory))
            store.start()
            thread = store._thread
            store.start()
            self.assertIs(thread, store._thread)
            store.close()
            store.start()
            self.assertIs(thread, store._thread)
            with self.assertRaises(ValueError):
                EventStore(Path(directory), max_events=-1)
            with self.assertRaises(ValueError):
                EventStore(Path(directory), max_media_bytes=-1)


if __name__ == "__main__":
    unittest.main()
