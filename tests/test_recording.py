import json
import threading
import time
import uuid
from collections import deque

import numpy as np
import pyxdf
from pylsl import StreamInfo, StreamOutlet, local_clock, resolve_byprop
import pytest

from bsense_p300_pilot.recording import Recording, channel_order


def health_recorder(age=0, inversions=0, data=None):
    recorder = Recording.__new__(Recording)
    recorder.lock = threading.Lock()
    recorder.recent = deque(data if data is not None else np.column_stack((np.arange(200), np.arange(200))))
    recorder.last_received = time.monotonic()-age
    recorder.receive_started = time.monotonic()-30
    recorder.samples = len(recorder.recent)
    recorder.inversions = inversions
    recorder.errors = []
    return recorder


def test_transient_gap_and_timestamp_history_warn_without_aborting():
    recorder = health_recorder(age=3, inversions=1)
    health = recorder.health()
    assert health["ok"] and len(health["warnings"]) == 2
    recorder.last_received = time.monotonic()
    assert len(recorder.health()["warnings"]) == 1


@pytest.mark.parametrize("data", [np.zeros((200, 2)), np.full((200, 2), np.nan)])
def test_signal_quality_is_a_recorded_warning(data):
    health = health_recorder(data=data).health()
    assert health["ok"] and health["warnings"]


def test_persistent_disconnect_and_recording_errors_still_block():
    recorder = health_recorder(age=10.1)
    assert not recorder.health()["ok"]
    recorder.last_received = time.monotonic()
    recorder.errors = ["disk write failed"]
    assert recorder.health()["reason"] == "disk write failed"


def test_initial_wait_has_ten_second_grace():
    recorder = health_recorder(data=[])
    recorder.last_received = None
    recorder.receive_started = time.monotonic()-3
    assert recorder.health()["ok"]
    recorder.receive_started = time.monotonic()-11
    assert not recorder.health()["ok"]


def eeg_info(labels=("FP1", "FP2")):
    info = StreamInfo(f"P300_TEST_{uuid.uuid4().hex}", "EEG", 2, 250, "float32", uuid.uuid4().hex)
    channels = info.desc().append_child("channels")
    for label in labels:
        channels.append_child("channel").append_child_value("label", label)
    return info


def test_channels_must_be_confirmed_and_non_fp_labels_rejected():
    assert channel_order(eeg_info(("FP2", "FP1")))[0] == ["FP2", "FP1"]
    with pytest.raises(ValueError):
        channel_order(eeg_info(("", "")))
    assert channel_order(eeg_info(("", "")), "FP1,FP2")[1] == "operator_confirmed"
    with pytest.raises(ValueError):
        channel_order(eeg_info(("C3", "C4")), "FP1,FP2")


def test_real_lsl_loopback_xdf_abort_retains_data(tmp_path):
    """Real local LSL transport, with explicitly synthetic samples only."""
    info = eeg_info()
    outlet = StreamOutlet(info)
    done = threading.Event()
    def publish():
        i = 0
        while not done.is_set():
            outlet.push_sample([np.sin(i/7), np.cos(i/9)], timestamp=local_clock())
            i += 1
            done.wait(0.004)
    worker = threading.Thread(target=publish, daemon=True)
    worker.start()
    recording = None
    try:
        resolved = resolve_byprop("source_id", info.source_id(), timeout=3)
        assert len(resolved) == 1
        recording = Recording(tmp_path, "TEST", "S1", resolved[0])
        recording.mark("synthetic_loopback", {"synthetic": True})
        deadline = time.monotonic()+5
        while not recording.health()["ok"] and time.monotonic() < deadline:
            time.sleep(.05)
        assert recording.health()["ok"]
        recording.stop("aborted", "synthetic automated test")
        with pytest.raises(FileExistsError):
            Recording(tmp_path, "TEST", "S1", info)
        streams, _ = pyxdf.load_xdf(str(next(recording.path.glob("*.xdf"))), dejitter_timestamps=False)
        eeg = next(s for s in streams if s["info"]["type"][0] == "EEG")
        markers = next(s for s in streams if s["info"]["type"][0] == "Markers")
        assert len(eeg["time_stamps"]) >= 100
        assert json.loads(markers["time_series"][0][0])["synthetic"] is True
        context = json.loads((recording.path/"context.json").read_text())
        assert context["status"] == "aborted"
        assert context["clock_offset_count"] >= 1
    finally:
        done.set()
        worker.join(timeout=2)
        if recording and recording.worker.is_alive():
            recording.stop("aborted")
