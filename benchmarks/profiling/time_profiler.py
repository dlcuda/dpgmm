import csv
import json
import os
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Optional, TypedDict

import pandas as pd


class TimeProfilerEvent(TypedDict):
    name: str
    time: float
    epoch: int


@lru_cache(maxsize=1)
def get_profiler():
    return TimeProfiler()


class TimeProfiler:
    def __init__(self, event_log_path: str = "./event_logs.csv"):
        """
        Default log format: CSV (best for pandas)
        You can still use JSON lines with `write_events_to_file_json` and `read_events_from_file_json`.
        """
        self.event_log_path = event_log_path
        self.events: List[TimeProfilerEvent] = []
        self.current_epoch = 0
        self.wrote_headers = False

    def add_event(self, name: str, time: float) -> None:
        """Add a profiling event to the internal list."""
        self.events.append(
            TimeProfilerEvent(name=name, time=time, epoch=self.current_epoch)
        )

    def write_events_to_file_json(self, path: Optional[str] = None) -> None:
        """Write all events to a file in JSON lines format."""
        path = path or self.event_log_path
        with open(path, "w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
        self.events = []

    def read_events_from_file_json(self, path: Optional[str] = None) -> None:
        """Load events from a JSON lines file."""
        path = path or self.event_log_path
        self.events = []
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.events.append(TimeProfilerEvent(**data))

    def write_events_to_file_csv(self) -> None:
        """Write all recorded events to a CSV file."""
        path = self.event_log_path
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists or not self.wrote_headers:
                writer.writerow(["name", "time", "epoch"])
                self.wrote_headers = True

            for event in self.events:
                writer.writerow([event["name"], event["time"], event["epoch"]])

            self.events = []

    def read_events_from_file_csv(self) -> pd.DataFrame:
        """
        Read events from a CSV file and return them as a pandas DataFrame.
        Also updates self.events internally for consistency.
        """
        if not os.path.exists(self.event_log_path):
            return pd.DataFrame(columns=["name", "time", "epoch"])

        df = pd.read_csv(self.event_log_path)
        df["time"] = df["time"].astype(float)
        self.events = [
            TimeProfilerEvent(name=row["name"], time=row["time"], epoch=row["epoch"])
            for _, row in df.iterrows()
        ]

        return df

    def update_epoch(self, current_epoch: Optional[int] = None) -> None:
        if current_epoch is not None:
            self.current_epoch = current_epoch
        else:
            self.current_epoch += 1


@contextmanager
def with_profiler(name: str):
    """Context manager to measure and record the duration of a code block."""
    profiler = get_profiler()

    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        profiler.add_event(name=name, time=elapsed)
