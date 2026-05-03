# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

"""Unit tests for TRAIL scheduling policies."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

import dataclasses
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_batch import Req, TrailState
from sglang.srt.managers.schedule_policy import SchedulePolicy


def make_req(rid, output_ids=None, trail_state=None):
    req = MagicMock(spec=Req)
    req.rid = rid
    req.output_ids = output_ids or []
    req.trail_state = trail_state
    return req


def make_trail_state(initial=100.0, remaining=50.0, count=1):
    return TrailState(
        initial_predicted_len=initial,
        current_predicted_remaining=remaining,
        prediction_count=count,
    )


class TestTrailSorting(unittest.TestCase):
    def test_sjf_sorts_by_initial_predicted_len(self):
        r1 = make_req("a", trail_state=make_trail_state(initial=200))
        r2 = make_req("b", trail_state=make_trail_state(initial=50))
        r3 = make_req("c", trail_state=make_trail_state(initial=100))
        queue = [r1, r2, r3]
        SchedulePolicy._sort_by_trail_sjf(queue)
        self.assertEqual([r.rid for r in queue], ["b", "c", "a"])

    def test_sjf_no_prediction_goes_last(self):
        r1 = make_req("a", trail_state=make_trail_state(initial=100))
        r2 = make_req("b")
        queue = [r2, r1]
        SchedulePolicy._sort_by_trail_sjf(queue)
        self.assertEqual([r.rid for r in queue], ["a", "b"])

    def test_sprpt_sorts_by_current_remaining(self):
        r1 = make_req("a", trail_state=make_trail_state(remaining=80))
        r2 = make_req("b", trail_state=make_trail_state(remaining=20))
        r3 = make_req("c", trail_state=make_trail_state(remaining=50))
        queue = [r1, r2, r3]
        SchedulePolicy._sort_by_trail_sprpt(queue)
        self.assertEqual([r.rid for r in queue], ["b", "c", "a"])

    def test_lrpsprpt_unpreemptible_requests_first(self):
        r1 = make_req(
            "a",
            output_ids=[0] * 90,
            trail_state=make_trail_state(initial=100, remaining=10),
        )
        r2 = make_req(
            "b",
            output_ids=[0] * 10,
            trail_state=make_trail_state(initial=100, remaining=90),
        )
        r3 = make_req(
            "c",
            output_ids=[0] * 50,
            trail_state=make_trail_state(initial=100, remaining=50),
        )
        queue = [r2, r3, r1]
        SchedulePolicy._sort_by_trail_lrpsprpt(queue, preemption_threshold=0.8)
        self.assertEqual([r.rid for r in queue], ["a", "c", "b"])

    def test_lrpsprpt_threshold_boundary(self):
        r1 = make_req(
            "at_boundary",
            output_ids=[0] * 80,
            trail_state=make_trail_state(initial=100, remaining=20),
        )
        r2 = make_req(
            "below",
            output_ids=[0] * 79,
            trail_state=make_trail_state(initial=100, remaining=21),
        )
        queue = [r2, r1]
        SchedulePolicy._sort_by_trail_lrpsprpt(queue, preemption_threshold=0.8)
        # 80/100 == 0.8, uses strict >, so both are still preemptible
        self.assertEqual([r.rid for r in queue], ["at_boundary", "below"])

    def test_lrpsprpt_past_threshold_is_unpreemptible(self):
        r1 = make_req(
            "past",
            output_ids=[0] * 81,
            trail_state=make_trail_state(initial=100, remaining=19),
        )
        r2 = make_req(
            "early",
            output_ids=[0] * 10,
            trail_state=make_trail_state(initial=100, remaining=5),
        )
        queue = [r2, r1]
        SchedulePolicy._sort_by_trail_lrpsprpt(queue, preemption_threshold=0.8)
        self.assertEqual([r.rid for r in queue], ["past", "early"])

    def test_lsprpt_uses_50_percent_threshold(self):
        r1 = make_req(
            "past50",
            output_ids=[0] * 51,
            trail_state=make_trail_state(initial=100, remaining=49),
        )
        r2 = make_req(
            "early",
            output_ids=[0] * 10,
            trail_state=make_trail_state(initial=100, remaining=90),
        )
        queue = [r2, r1]
        SchedulePolicy._sort_by_trail_lsprpt(queue, preemption_threshold=0.5)
        self.assertEqual([r.rid for r in queue], ["past50", "early"])

    def test_rpsprpt_same_as_sprpt(self):
        r1 = make_req("a", trail_state=make_trail_state(remaining=100))
        r2 = make_req("b", trail_state=make_trail_state(remaining=10))
        queue = [r1, r2]
        SchedulePolicy._sort_by_trail_rpsprpt(queue)
        self.assertEqual([r.rid for r in queue], ["b", "a"])


class TestTrailCompare(unittest.TestCase):
    def test_preempt_when_waiting_has_shorter_remaining(self):
        waiting = make_req(
            "w",
            output_ids=[0] * 5,
            trail_state=make_trail_state(initial=50, remaining=20),
        )
        running = make_req(
            "r",
            output_ids=[0] * 5,
            trail_state=make_trail_state(initial=200, remaining=150),
        )
        self.assertGreater(SchedulePolicy.trail_compare(waiting, running), 0)

    def test_no_preempt_when_running_has_shorter_remaining(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=150))
        running = make_req(
            "r",
            output_ids=[0] * 5,
            trail_state=make_trail_state(initial=200, remaining=20),
        )
        self.assertLessEqual(SchedulePolicy.trail_compare(waiting, running), 0)

    def test_no_preempt_when_running_is_unpreemptible(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=5))
        running = make_req(
            "r",
            output_ids=[0] * 85,
            trail_state=make_trail_state(initial=100, remaining=15),
        )
        self.assertLessEqual(
            SchedulePolicy.trail_compare(waiting, running, preemption_threshold=0.8), 0
        )

    def test_no_preempt_when_waiting_has_no_prediction_short_running(self):
        waiting = make_req("w")
        running = make_req(
            "r",
            output_ids=[0] * 5,
            trail_state=make_trail_state(initial=200, remaining=150),
        )
        self.assertLessEqual(SchedulePolicy.trail_compare(waiting, running), 0)

    def test_preempt_when_waiting_has_no_prediction_long_running(self):
        waiting = make_req("w")
        running = make_req(
            "r",
            output_ids=[0] * 5,
            trail_state=make_trail_state(initial=500, remaining=400),
        )
        self.assertGreater(SchedulePolicy.trail_compare(waiting, running), 0)

    def test_no_preempt_when_running_has_no_prediction(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=10))
        running = make_req("r")
        self.assertLessEqual(SchedulePolicy.trail_compare(waiting, running), 0)

    def test_no_preempt_when_equal_remaining(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=50))
        running = make_req(
            "r",
            output_ids=[0] * 5,
            trail_state=make_trail_state(initial=200, remaining=50),
        )
        self.assertLessEqual(SchedulePolicy.trail_compare(waiting, running), 0)

    def test_custom_threshold(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=5))
        running = make_req(
            "r",
            output_ids=[0] * 55,
            trail_state=make_trail_state(initial=100, remaining=45),
        )
        self.assertLessEqual(
            SchedulePolicy.trail_compare(waiting, running, preemption_threshold=0.5), 0
        )
        self.assertGreater(
            SchedulePolicy.trail_compare(waiting, running, preemption_threshold=0.8), 0
        )


class TestTrailState(unittest.TestCase):
    def test_defaults(self):
        ts = TrailState()
        self.assertEqual(ts.initial_predicted_len, 0.0)
        self.assertEqual(ts.current_predicted_remaining, 0.0)
        self.assertEqual(ts.prediction_count, 0)

    def test_update(self):
        ts = TrailState(
            initial_predicted_len=100,
            current_predicted_remaining=100,
            prediction_count=1,
        )
        ts.current_predicted_remaining = 50
        ts.prediction_count += 1
        self.assertEqual(ts.current_predicted_remaining, 50)
        self.assertEqual(ts.prediction_count, 2)
        self.assertEqual(ts.initial_predicted_len, 100)

    def test_dataclass_replace(self):
        ts = TrailState(
            initial_predicted_len=100,
            current_predicted_remaining=75,
            prediction_count=3,
        )
        ts2 = dataclasses.replace(ts)
        self.assertEqual(ts2.initial_predicted_len, 100)
        self.assertEqual(ts2.current_predicted_remaining, 75)
        self.assertEqual(ts2.prediction_count, 3)


if __name__ == "__main__":
    unittest.main()
