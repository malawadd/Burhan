import unittest

import cv2
import numpy as np

from edge_runtime.contracts import FrameContext
from edge_runtime.profiles.roads import RoadsProfile, demo_frame


def context(sequence, timestamp=None, source="road-test", session="one"):
    return FrameContext(sequence, sequence / 10 if timestamp is None else timestamp, source, session)


def asphalt():
    rng = np.random.default_rng(33)
    gray = np.clip(145 + rng.normal(0, 4, (540, 960)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class RoadsTests(unittest.TestCase):
    def test_pixel_candidate_confirmed_once_with_stable_id(self):
        profile = RoadsProfile()
        results = [profile.process(demo_frame(i), context(i)) for i in range(8)]
        self.assertFalse(results[0].emit_event)
        self.assertFalse(results[1].emit_event)
        self.assertTrue(results[2].emit_event)
        self.assertEqual(sum(result.emit_event for result in results), 1)
        self.assertEqual({result.domain_state["tracks"][0]["id"] for result in results}, {1})
        self.assertTrue(all("Experimental" in result.title for result in results))

    def test_pixels_control_result_not_frame_sequence(self):
        blank = asphalt()
        profile = RoadsProfile({"confirm_frames": 1})
        empty = profile.process(blank, context(0))
        self.assertEqual(empty.metrics["candidate_count"], 0)
        cv2.ellipse(blank, (480, 330), (44, 25), 0, 0, 360, (45, 45, 45), -1)
        found = profile.process(blank, context(1))
        self.assertEqual(found.metrics["candidate_count"], 1)
        self.assertTrue(found.emit_event)

    def test_broad_elongated_shadow_is_filtered(self):
        frame = asphalt()
        cv2.ellipse(frame, (480, 385), (170, 9), 0, 0, 360, (65, 65, 65), -1)
        result = RoadsProfile().process(frame, context(0))
        self.assertEqual(result.status, "clear")
        self.assertEqual(result.metrics["candidate_count"], 0)

    def test_compact_shadow_is_known_false_positive(self):
        frame = asphalt()
        cv2.ellipse(frame, (480, 330), (44, 25), 0, 0, 360, (65, 65, 65), -1)
        result = RoadsProfile({"confirm_frames": 1}).process(frame, context(0))
        self.assertEqual(result.status, "review")
        self.assertIn("shadows", result.guidance)

    def test_bad_quality_never_reports_clear_or_event(self):
        for value in (0, 145, 255):
            result = RoadsProfile().process(np.full((540, 960, 3), value, dtype=np.uint8), context(0))
            self.assertEqual(result.status, "rejected")
            self.assertEqual(result.action, "request_retake")
            self.assertFalse(result.emit_event)

    def test_blurred_candidate_frame_requests_retake(self):
        blurred = cv2.GaussianBlur(demo_frame(0), (41, 41), 15)
        result = RoadsProfile().process(blurred, context(0))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.action, "request_retake")
        self.assertIn("blurred", result.domain_state["rejection"])

    def test_roi_excludes_nonroad_region(self):
        frame = asphalt()
        cv2.circle(frame, (100, 100), 40, (20, 20, 20), -1)
        result = RoadsProfile().process(frame, context(0))
        self.assertEqual(result.metrics["candidate_count"], 0)

    def test_timeout_ends_dedup_window(self):
        profile = RoadsProfile({"confirm_frames": 1, "track_timeout_seconds": 0.5})
        first = profile.process(demo_frame(0), context(0, 0))
        duplicate = profile.process(demo_frame(0), context(1, 0.1))
        profile.process(asphalt(), context(2, 1))
        reappeared = profile.process(demo_frame(0), context(3, 1.1))
        self.assertTrue(first.emit_event)
        self.assertFalse(duplicate.emit_event)
        self.assertTrue(reappeared.emit_event)
        self.assertNotEqual(first.event_key, reappeared.event_key)

    def test_nonoverlapping_region_gets_new_identity(self):
        profile = RoadsProfile({"confirm_frames": 1})
        first = profile.process(demo_frame(0), context(0))
        moved = asphalt()
        cv2.ellipse(moved, (650, 440), (44, 25), 0, 0, 360, (45, 45, 45), -1)
        second = profile.process(moved, context(1))
        self.assertTrue(second.emit_event)
        self.assertNotEqual(first.domain_state["tracks"][0]["id"], second.domain_state["tracks"][0]["id"])

    def test_source_change_requires_confirmation_again(self):
        profile = RoadsProfile({"confirm_frames": 2})
        profile.process(demo_frame(0), context(0))
        self.assertTrue(profile.process(demo_frame(1), context(1)).emit_event)
        result = profile.process(demo_frame(2), context(2, source="different"))
        self.assertFalse(result.emit_event)
        self.assertEqual(result.status, "pending")

    def test_rejection_breaks_unconfirmed_streak(self):
        profile = RoadsProfile({"confirm_frames": 2})
        profile.process(demo_frame(0), context(0))
        profile.process(np.zeros((540, 960, 3), dtype=np.uint8), context(1))
        self.assertFalse(profile.process(demo_frame(2), context(2)).emit_event)
        self.assertTrue(profile.process(demo_frame(3), context(3)).emit_event)

    def test_invalid_input_and_roi_fail_explicitly(self):
        with self.assertRaises(ValueError):
            RoadsProfile({"roi": [[0, 0]]})
        with self.assertRaises(ValueError):
            RoadsProfile().process(np.zeros((100, 100), dtype=np.uint8), context(0))


if __name__ == "__main__":
    unittest.main()
