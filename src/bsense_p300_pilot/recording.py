"""Record one explicitly selected EEG outlet plus locally timestamped task markers."""

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from pylsl import StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_byprop

from .protocol import VERSION, Config, identifier, timing_summary
from .xdf_writer import XDFWriter


def channel_order(info, override: str = "auto") -> tuple[list[str], str]:
    if info.channel_count() != 2 or info.nominal_srate() < 100:
        raise ValueError("需要两通道、至少 100 Hz 的 EEG 流。")
    labels = []
    node = info.desc().child("channels").child("channel")
    for _ in range(2):
        labels.append(node.child_value("label").upper().replace(" ", ""))
        node = node.next_sibling()
    if set(labels) == {"FP1", "FP2"}:
        return labels, "lsl_metadata"
    if any(labels):
        raise ValueError(f"EEG 通道标签不匹配：{labels}，需要 FP1/FP2。")
    if override not in ("FP1,FP2", "FP2,FP1"):
        raise ValueError("设备未提供通道标签，请核实后在界面中指定实际顺序。")
    return override.split(","), "operator_confirmed"


def discover():
    return list(resolve_byprop("type", "EEG", timeout=2.0))


class Recording:
    def __init__(self, root: Path, subject: str, session: str, info, override="auto", notes="", config=None):
        config = config or Config()
        config.validate()
        self.path = root.expanduser().resolve()/f"sub-{identifier(subject)}"/f"ses-{identifier(session)}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.context = dict(schema_version=1, protocol_version=VERSION, subject_id=subject,
                            session_id=session, status="starting", notes=notes,
                            started_utc=datetime.now(timezone.utc).isoformat(),
                            timing=timing_summary(config), config=asdict(config), stream_name=info.name(), stream_uid=info.uid(),
                            source_id=info.source_id(), nominal_srate=info.nominal_srate())
        self._save_context()
        self.stop_event = threading.Event()
        self.worker = None
        self.writer = None
        self.inlet = None
        self.events_file = None
        self.outlet = None
        self.samples = 0
        self.first = self.last = 0.0
        self.last_received = None
        self.receive_started = time.monotonic()
        self.offsets = []
        self.errors = []
        self.marker_times = []
        self.inversions = 0
        self.lock = threading.Lock()
        self.recent = deque(maxlen=2000)
        try:
            self.inlet = StreamInlet(info, max_buflen=30, recover=False)
            self.inlet.open_stream(timeout=5)
            full = self.inlet.info(timeout=5)
            labels, provenance = channel_order(full, override)
            self.context.update(channel_order=labels, channel_order_source=provenance,
                                eeg_xml=full.as_xml(), nominal_srate=full.nominal_srate())
            self.writer = XDFWriter(self.path/f"sub-{subject}_ses-{session}_task-m7_p300.xdf")
            self.writer.write_stream_header(1, full.as_xml())
            mi = StreamInfo("BSense P300 Pilot Markers", "Markers", 1, 0, "string", f"p300-{subject}-{session}")
            self.outlet = StreamOutlet(mi)
            self.writer.write_stream_header(2, mi.as_xml())
            now = local_clock()
            self.writer.write_clock_offset(2, now, 0.0)
            # Obtain the remote-to-local correction before the stimulus may start.
            correction = float(self.inlet.time_correction(timeout=5))
            if not math.isfinite(correction):
                raise ValueError("EEG 时钟校正不是有限数。")
            self._offset(correction)
            self.events_file = (self.path/"events.jsonl").open("x", encoding="utf-8")
            self.format = full.channel_format()
            self.context["status"] = "recording"
            self.context["acquisition_quality_policy"] = "warnings_continue_disconnect_10s_v1"
            self._save_context()
            self.receive_started = time.monotonic()
            self.worker = threading.Thread(target=self._record, daemon=True)
            self.worker.start()
        except Exception as error:
            self.context.update(status="start_failed", error=str(error))
            self._save_context()
            if self.writer:
                self.writer.close()
            if self.inlet:
                self.inlet.close_stream()
            raise

    def _save_context(self):
        temporary = self.path/"context.tmp"
        temporary.write_text(json.dumps(self.context, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path/"context.json")

    def _offset(self, correction):
        # XDF stores collection time in the source clock domain.
        pair = (local_clock()-correction, correction)
        self.writer.write_clock_offset(1, *pair)
        self.offsets.append(pair)

    def _record(self):
        next_offset = time.monotonic()+5
        try:
            while not self.stop_event.is_set():
                samples, stamps = self.inlet.pull_chunk(timeout=0.2, max_samples=1024)
                if stamps:
                    self.writer.write_samples(1, stamps, samples, 2, self.format)
                    with self.lock:
                        if self.samples == 0:
                            self.first = stamps[0]
                        for stamp in stamps:
                            if self.samples and stamp <= self.last:
                                self.inversions += 1
                            self.last = stamp
                            self.samples += 1
                        self.recent.extend(samples)
                        self.last_received = time.monotonic()
                if time.monotonic() >= next_offset:
                    correction = float(self.inlet.time_correction(timeout=0.2))
                    if not math.isfinite(correction):
                        raise ValueError("EEG 时钟校正失败")
                    self._offset(correction)
                    self.writer.write_boundary()
                    next_offset = time.monotonic()+5
        except Exception as error:
            self.errors.append(str(error))

    def health(self):
        with self.lock:
            data = np.asarray(list(self.recent), dtype=float)
            age = time.monotonic()-(self.last_received if self.last_received is not None else self.receive_started)
            count = self.samples
            inversions = self.inversions
        reason = "ok"
        warnings = []
        if self.errors:
            reason = self.errors[-1]
        elif age >= 10:
            reason = "EEG 持续 10 秒无新数据，请检查设备与 LSL 发布软件"
        if inversions:
            warnings.append("EEG 曾有时间戳倒序或重复，训练前需检查")
        if age > 2:
            warnings.append("EEG 暂时无新数据，正在等待恢复")
        if len(data) < 100:
            warnings.append("EEG 样本不足，正在等待")
        elif not np.isfinite(data).all():
            warnings.append("EEG 含非有限数，请检查设备")
        elif np.any(data.std(axis=0) < 1e-8):
            warnings.append("EEG 通道平直，请调整电极接触")
        return {"ok": reason == "ok", "reason": reason, "samples": count,
                "warnings": warnings, "age_seconds": age}

    def mark(self, event: str, metadata=None, timestamp=None):
        stamp = local_clock() if timestamp is None else timestamp
        row = {"task": "m7_p300", "subject_id": self.context["subject_id"],
               "session_id": self.context["session_id"], **(metadata or {}),
               "event": event, "timestamp_lsl": stamp}
        text = json.dumps(row, ensure_ascii=False, allow_nan=False)
        self.writer.write_samples(2, [stamp], [[text]], 1, 3)
        self.events_file.write(text+"\n")
        self.events_file.flush()
        self.outlet.push_sample([text], timestamp=stamp)
        self.marker_times.append(stamp)

    def stop(self, status="aborted", reason=""):
        self.stop_event.set()
        self.worker.join(timeout=3)
        if self.worker.is_alive():
            # Keep the file open for the writer; never close beneath a live worker.
            self.context.update(status="stop_failed", error="EEG 录制线程未停止")
            self._save_context()
            raise RuntimeError("EEG 录制线程未停止，记录不能标为完成。")
        self.inlet.close_stream()
        now = local_clock()
        self.writer.write_clock_offset(2, now, 0.0)
        self.writer.write_stream_footer(1, self.first, self.last, self.samples, self.offsets)
        self.writer.write_stream_footer(2, self.marker_times[0] if self.marker_times else 0,
                                        self.marker_times[-1] if self.marker_times else 0,
                                        len(self.marker_times))
        self.writer.close()
        self.events_file.close()
        self.outlet = None
        self.context.update(status="recording_error" if self.errors else status, reason=reason,
                            ended_utc=datetime.now(timezone.utc).isoformat(),
                            eeg_samples=self.samples, marker_count=len(self.marker_times),
                            clock_offset_count=len(self.offsets), timestamp_inversions=self.inversions,
                            errors=self.errors)
        self._save_context()
