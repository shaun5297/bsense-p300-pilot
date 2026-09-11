"""Minimal, thread-safe XDF writer used by the built-in LSL recorder."""

from __future__ import annotations

import struct
import threading
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Sequence


FILE_HEADER = 1
STREAM_HEADER = 2
SAMPLES = 3
CLOCK_OFFSET = 4
BOUNDARY = 5
STREAM_FOOTER = 6
BOUNDARY_UUID = bytes(
    (0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F, 0xB3, 0x0E, 0xD5, 0x46, 0x73, 0x83, 0xCB, 0xE4)
)


def encode_varlen_int(value: int) -> bytes:
    if value < 0:
        raise ValueError("XDF variable-length integers cannot be negative")
    if value < 256:
        return b"\x01" + struct.pack("<B", value)
    if value <= 0xFFFFFFFF:
        return b"\x04" + struct.pack("<I", value)
    return b"\x08" + struct.pack("<Q", value)


class XDFWriter:
    """Write the chunk subset used by LabRecorder-compatible XDF files."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._file: BinaryIO = self.path.open("xb")
        self._lock = threading.Lock()
        self._closed = False
        self._file.write(b"XDF:")
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
        header = f'<?xml version="1.0"?>\n  <info>\n    <version>1.0</version>\n    <datetime>{timestamp}</datetime>\n  </info>'
        self._write_chunk(FILE_HEADER, header.encode("utf-8"))

    def _write_chunk(self, tag: int, content: bytes, stream_id: int | None = None) -> None:
        stream_prefix = b"" if stream_id is None else struct.pack("<I", stream_id)
        chunk_length = 2 + len(stream_prefix) + len(content)
        self._file.write(encode_varlen_int(chunk_length))
        self._file.write(struct.pack("<H", tag))
        self._file.write(stream_prefix)
        self._file.write(content)
        self._file.flush()

    def write_stream_header(self, stream_id: int, xml: str) -> None:
        with self._lock:
            self._write_chunk(STREAM_HEADER, xml.encode("utf-8"), stream_id)

    def write_samples(
        self,
        stream_id: int,
        timestamps: Sequence[float],
        samples: Sequence[Sequence[object]],
        channel_count: int,
        channel_format: int,
    ) -> None:
        if not timestamps:
            return
        if len(timestamps) != len(samples):
            raise ValueError("timestamp/sample count mismatch")
        if channel_count <= 0:
            raise ValueError("channel_count must be positive")

        payload = bytearray(b"\x04" + struct.pack("<I", len(timestamps)))
        numeric_formats = {1: "f", 2: "d", 4: "i", 5: "h", 6: "b", 7: "q"}
        sample_struct = (
            struct.Struct("<" + numeric_formats[channel_format] * channel_count)
            if channel_format in numeric_formats
            else None
        )
        if channel_format != 3 and sample_struct is None:
            raise ValueError(f"unsupported LSL channel format: {channel_format}")

        for timestamp, sample in zip(timestamps, samples, strict=True):
            if len(sample) != channel_count:
                raise ValueError("sample channel count mismatch")
            payload.extend(b"\x08")
            payload.extend(struct.pack("<d", float(timestamp)))
            if channel_format == 3:
                for value in sample:
                    encoded = value if isinstance(value, bytes) else str(value).encode("utf-8")
                    payload.extend(encode_varlen_int(len(encoded)))
                    payload.extend(encoded)
            else:
                assert sample_struct is not None
                payload.extend(sample_struct.pack(*sample))

        with self._lock:
            self._write_chunk(SAMPLES, bytes(payload), stream_id)

    def write_clock_offset(self, stream_id: int, collection_time: float, offset: float) -> None:
        with self._lock:
            self._write_chunk(CLOCK_OFFSET, struct.pack("<dd", collection_time, offset), stream_id)

    def write_boundary(self) -> None:
        with self._lock:
            self._write_chunk(BOUNDARY, BOUNDARY_UUID)

    def write_stream_footer(
        self,
        stream_id: int,
        first_timestamp: float,
        last_timestamp: float,
        sample_count: int,
        clock_offsets: Sequence[tuple[float, float]] = (),
    ) -> None:
        offsets_xml = "".join(
            f"<offset><time>{collection_time:.16g}</time><value>{offset:.16g}</value></offset>"
            for collection_time, offset in clock_offsets
        )
        footer = (
            '<?xml version="1.0"?><info>'
            f"<first_timestamp>{first_timestamp:.16g}</first_timestamp>"
            f"<last_timestamp>{last_timestamp:.16g}</last_timestamp>"
            f"<sample_count>{sample_count}</sample_count>"
            f"<clock_offsets>{offsets_xml}</clock_offsets></info>"
        )
        with self._lock:
            self._write_chunk(STREAM_FOOTER, footer.encode("utf-8"), stream_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._file.flush()
            self._file.close()
            self._closed = True


__all__ = [
    "BOUNDARY",
    "CLOCK_OFFSET",
    "FILE_HEADER",
    "SAMPLES",
    "STREAM_FOOTER",
    "STREAM_HEADER",
    "XDFWriter",
    "encode_varlen_int",
]
