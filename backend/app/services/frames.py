"""WebSocket 二进制帧编解码 / Binary keyframe codec.

设计动机：SPH 数据量大。以 JSON 数组传输 4000 个粒子的位置 + 速度约需
240 KB/帧；同样的数据用 ``float32`` 二进制只需 96 KB，且浏览器端解码
是一次 ``Float32Array`` 视图，没有 JSON.parse 的开销。

Frame layout
------------
::

    +----------------+--------------------------+---------------------------+
    | uint32 header  |  UTF-8 JSON header       |  concatenated float32     |
    | length (LE)    |  (header length bytes)   |  arrays, in header order  |
    +----------------+--------------------------+---------------------------+

``header["arrays"]`` 按顺序声明每个数组的元素个数，客户端据此切片。
所有数组均为小端 ``float32``。

.. important::
   JSON 头会被**填充空格到 4 字节对齐**，这样客户端可以直接用
   ``new Float32Array(buffer, headerEnd, count)`` 建立零拷贝视图。
   尾随空格是合法的 JSON 空白，不影响解析。若不做对齐，
   ``Float32Array`` 会因起始偏移不是 4 的倍数而抛 ``RangeError``。
"""
from __future__ import annotations

import json
import struct
from typing import Any, Iterable

import numpy as np

HEADER_LENGTH_STRUCT = struct.Struct("<I")


def encode_message(header: dict[str, Any], arrays: Iterable[tuple[str, np.ndarray]]) -> bytes:
    """Serialise one header + float32 payload into a single WebSocket message."""
    blocks: list[bytes] = []
    descriptors: list[dict[str, Any]] = []
    for name, array in arrays:
        flat = np.ascontiguousarray(array, dtype="<f4").ravel()
        descriptors.append({"name": name, "count": int(flat.size), "dtype": "f4"})
        blocks.append(flat.tobytes())

    header = {**header, "arrays": descriptors}
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # Pad to a 4-byte boundary so the float32 payload stays aligned.
    padding = (-len(header_bytes)) % 4
    if padding:
        header_bytes += b" " * padding
    return b"".join([HEADER_LENGTH_STRUCT.pack(len(header_bytes)), header_bytes, *blocks])


def encode_json_message(payload: dict[str, Any]) -> str:
    """Control messages (status/progress/completion) travel as plain JSON text."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decode_message(message: bytes | str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Inverse of :func:`encode_message` (used by the test-suite)."""
    if isinstance(message, str):
        return json.loads(message), {}

    (header_length,) = HEADER_LENGTH_STRUCT.unpack_from(message, 0)
    start = HEADER_LENGTH_STRUCT.size
    header = json.loads(message[start:start + header_length].decode("utf-8"))
    offset = start + header_length

    arrays: dict[str, np.ndarray] = {}
    for descriptor in header.get("arrays", []):
        count = int(descriptor["count"])
        block = message[offset:offset + count * 4]
        arrays[descriptor["name"]] = np.frombuffer(block, dtype="<f4")
        offset += count * 4
    return header, arrays
