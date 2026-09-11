from pathlib import Path
from types import SimpleNamespace
import time
import tkinter as tk
from unittest.mock import Mock

import pytest

from bsense_p300_pilot.app import App, create_root
from bsense_p300_pilot.protocol import HARD_LIMIT, Step


@pytest.fixture
def app(tmp_path):
    try:
        root = create_root()
    except tk.TclError:
        pytest.skip("Tk display unavailable; native GUI verification is separate")
    root.withdraw()
    app = App(root, tmp_path)
    yield app
    root.destroy()


def test_time_limit_prevents_next_stimulus(app):
    app.recording = Mock()
    app.started = time.monotonic()-HARD_LIMIT-1
    app.finish = Mock()
    app._next()
    app.finish.assert_called_once_with("time_limit", "已达到 20 分钟上限")
    app.recording.mark.assert_not_called()


def test_abort_cancels_flash_and_saves_partial_recording(app):
    app.recording = Mock()
    app.after_step = app.root.after(10000, lambda: None)
    app.after_off = app.root.after(10000, lambda: None)
    app.background = Mock()
    app.abort()
    assert app.finishing
    assert app.after_step is app.after_off is None
    app.recording.mark.assert_called_once_with("recording_end", {"status":"aborted", "reason":"操作者结束"})
    kind, callback = app.background.call_args.args
    assert kind == "stop"
    app.recording.context = {"status":"aborted"}
    app.recording.path = Path("partial")
    callback()
    app.recording.stop.assert_called_once_with("aborted", "操作者结束")


def test_flash_ui_hides_cue_and_marks_actual_onset(app):
    app.recording = Mock()
    app.started = time.monotonic()
    app.plan = [Step("p300_flash", .175, "保持注视", {"target_command":"forward"}, flash=1)]
    app._next()
    assert app.task_text.get() == "保持注视"
    assert all(tile["highlightbackground"] == "#1C2D43" for tile in app.tiles)
    assert app.recording.mark.call_args.args[0] == "p300_flash"
    assert isinstance(app.recording.mark.call_args.args[2], float)
    app.root.after_cancel(app.after_step)
    app.root.after_cancel(app.after_off)


def test_failed_stop_keeps_live_recorder_and_retry_entry(app):
    recorder = Mock()
    recorder.worker.is_alive.return_value = True
    app.recording = recorder
    app.closing = True
    app.pending_completion = ("complete", "协议完成")
    app._handle_stop(None, "thread still alive")
    assert app.recording is recorder
    assert app.busy and app.save_failed and not app.closing
    assert str(app.start_button["state"]) == "disabled"
    app.finish = Mock()
    app.abort()
    app.finish.assert_called_once_with("complete", "协议完成")
