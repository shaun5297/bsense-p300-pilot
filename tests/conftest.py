from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from pylsl import StreamInfo
import pytest

from bsense_p300_pilot.protocol import Config, HIGHLIGHT, VERSION, build_plan
from bsense_p300_pilot.xdf_writer import XDFWriter


def event_rows(config=Config(1,6,4), subject="P01", session="S01"):
    rows = []
    time = 100.0
    for step in build_plan(subject, session, config):
        row = {"subject_id": subject, "session_id": session, **step.metadata, "event": step.event, "time": time}
        rows.append(row)
        if step.flash is not None:
            rows.append({**row, "event": "p300_flash_off", "time": time+HIGHLIGHT})
        time += step.seconds
    return rows


@pytest.fixture
def synthetic_session(tmp_path):
    """Explicitly synthetic fixture; no participant data or scientific metric claim."""
    config = Config(1,6,4)
    rows = event_rows(config)
    folder = tmp_path/"sub-P01"/"ses-S01"
    folder.mkdir(parents=True)
    context = dict(subject_id="P01", session_id="S01", status="complete", protocol_version=VERSION,
                   config=asdict(config), nominal_srate=250, channel_order=["FP1", "FP2"],
                   clock_offset_count=1, timestamp_inversions=0, synthetic=True)
    (folder/"context.json").write_text(json.dumps(context), encoding="utf-8")
    times = np.arange(99, rows[-1]["time"]+2, 1/250)
    values = np.random.default_rng(8).normal(0, 0.2, (len(times), 2))
    for row in rows:
        if row["event"] == "p300_flash" and row["is_target"]:
            lo = np.searchsorted(times, row["time"])
            hi = np.searchsorted(times, row["time"]+0.8)
            pulse = np.exp(-((times[lo:hi]-row["time"]-0.3)/0.07)**2)
            values[lo:hi] += 2*pulse[:, None]
    writer = XDFWriter(folder/"synthetic_task-m7_p300.xdf")
    eeg = StreamInfo("Synthetic EEG", "EEG", 2, 250, "float32", "synthetic-eeg")
    channels = eeg.desc().append_child("channels")
    for label in ("FP1", "FP2"):
        channels.append_child("channel").append_child_value("label", label)
    marker = StreamInfo("Synthetic Markers", "Markers", 1, 0, "string", "synthetic-markers")
    writer.write_stream_header(1, eeg.as_xml())
    writer.write_stream_header(2, marker.as_xml())
    writer.write_clock_offset(1, 99, 0)
    writer.write_clock_offset(2, 99, 0)
    writer.write_samples(1, times.tolist(), values.tolist(), 2, 1)
    writer.write_samples(2, [r["time"] for r in rows], [[json.dumps(r)] for r in rows], 1, 3)
    writer.write_stream_footer(1, times[0], times[-1], len(times))
    writer.write_stream_footer(2, rows[0]["time"], rows[-1]["time"], len(rows))
    writer.close()
    return tmp_path
