import os
import struct
import mmap
import msgpack
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# DOCLOOM — UNIVERSE CONTAINER (universe.loom)
# =============================================================================
# The single unified brain-state file. Replaces the loose multi-file scatter
# (universe.metric, atlas.capnp, latent_state.snapshot, cortex_journal.bin,
#  coordinates.bin, velocities.bin, physics_tensors.bin, brain_embeddings.bin,
#  working_memory.bin, causal_links.bin, metadata.bin, replay.bin)
# with one rigid, sequential binary container:
#
#   [Header Block: bytes 0..1024]   magic "LUNI", version, Genesis Master Seed,
#                                   dimension, generation, TOC offset/length.
#   [Segment Body]                  named segments packed contiguously
#                                   (atlas router matrix, cortex state tensors,
#                                    ledger snapshot, ...). Zero-copy readable.
#   [TOC]                           msgpack {name: [offset, length]}.
#   [Journal Tail: TOC end..EOF]    append-only typed records. Appends are pure
#                                   O(1) file appends — no header mutation.
#
# Write model:
#   - rebuild(segments, journal)  = atomic full rewrite (tmp + os.replace),
#     performed only at checkpoint/save cadence.
#   - append_journal(blob)        = O(1) append at EOF between rebuilds.
#   - The mass shard data itself lives in per-crystal .loom extents (append-only
#     logs); this container is the catalog + live brain state that binds them.
#
# Crash model: the header/TOC are only touched by the atomic rebuild; a torn
# journal tail is tolerated by the typed-record parser (stops at first record
# that does not fit).
# =============================================================================

UNIVERSE_MAGIC = b"LUNI"
UNIVERSE_VERSION = 1
UNIVERSE_HEADER_FORMAT = "<4sHQIIIQQ"  # magic, version, seed, dimension, flags, generation, toc_offset, toc_len
UNIVERSE_HEADER_STRUCT = struct.Struct(UNIVERSE_HEADER_FORMAT)
UNIVERSE_HEADER_SIZE = 1024

# ---- Journal record framing: [u8 type][u16 payload_len][payload] ----
JREC_PREFIX = struct.Struct("<BH")
JREC_TXN = 1        # ledger transaction: <Qfd> shard_hash, activation, timestamp
JREC_CRYSTAL = 2    # crystal registration: <BIH> state, num_shards, path_len + path + centroid f32[dim]

TXN_PAYLOAD = struct.Struct("<Qfd")
CRYSTAL_PREFIX = struct.Struct("<BIH")


def pack_txn_record(shard_hash: int, activation: float, timestamp: float) -> bytes:
    payload = TXN_PAYLOAD.pack(shard_hash, activation, timestamp)
    return JREC_PREFIX.pack(JREC_TXN, len(payload)) + payload


def pack_crystal_record(state_byte: int, num_shards: int, path: str, centroid_f32_bytes: bytes) -> bytes:
    path_bytes = path.encode("utf-8")
    payload = CRYSTAL_PREFIX.pack(state_byte, num_shards, len(path_bytes)) + path_bytes + centroid_f32_bytes
    return JREC_PREFIX.pack(JREC_CRYSTAL, len(payload)) + payload


def iter_journal_records(blob: bytes):
    """Yields (rec_type, payload). Stops cleanly at a torn tail."""
    pos = 0
    end = len(blob)
    while pos + JREC_PREFIX.size <= end:
        rec_type, plen = JREC_PREFIX.unpack_from(blob, pos)
        pos += JREC_PREFIX.size
        if pos + plen > end:
            break  # torn record — everything before it is still valid
        yield rec_type, blob[pos:pos + plen]
        pos += plen


class UniverseContainer:
    """Handle to one universe.loom file. Stateless between operations except a
    generation-keyed TOC cache — safe to reopen from any process."""

    def __init__(self, path: str, seed: int = 0, dimension: int = 0, create: bool = False):
        self.path = os.path.abspath(path)
        self._toc_cache: Optional[Dict[str, Tuple[int, int]]] = None
        self._toc_cache_generation = -1

        if os.path.exists(self.path) and os.path.getsize(self.path) >= UNIVERSE_HEADER_SIZE:
            self._read_header()
        elif create:
            self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
            self.dimension = int(dimension)
            self.flags = 0
            self.generation = 0
            self.toc_offset = UNIVERSE_HEADER_SIZE
            self.toc_len = 0
            self.rebuild({}, journal=b"")
        else:
            raise FileNotFoundError(self.path)

    @staticmethod
    def exists(path: str) -> bool:
        try:
            if not os.path.exists(path) or os.path.getsize(path) < UNIVERSE_HEADER_SIZE:
                return False
            with open(path, "rb") as f:
                return f.read(4) == UNIVERSE_MAGIC
        except Exception:
            return False

    # ------------------------------------------------------------------ header

    def _read_header(self) -> None:
        with open(self.path, "rb") as f:
            header = f.read(UNIVERSE_HEADER_SIZE)
        magic, version, seed, dimension, flags, generation, toc_offset, toc_len = \
            UNIVERSE_HEADER_STRUCT.unpack(header[:UNIVERSE_HEADER_STRUCT.size])
        if magic != UNIVERSE_MAGIC:
            raise ValueError(f"Not a universe.loom container (magic={magic!r}): {self.path}")
        self.seed = seed
        self.dimension = dimension
        self.flags = flags
        self.generation = generation
        self.toc_offset = toc_offset
        self.toc_len = toc_len

    def _header_bytes(self) -> bytes:
        header = UNIVERSE_HEADER_STRUCT.pack(
            UNIVERSE_MAGIC, UNIVERSE_VERSION, self.seed, self.dimension,
            self.flags, self.generation, self.toc_offset, self.toc_len
        )
        return header.ljust(UNIVERSE_HEADER_SIZE, b"\x00")

    # ------------------------------------------------------------------ writes

    def rebuild(self, segments: Dict[str, bytes], journal: bytes = b"") -> None:
        """
        Atomically rewrites the container: header + segment body + TOC (+ an
        optional carried-over journal). Called at checkpoint/save cadence only.
        """
        toc: Dict[str, List[int]] = {}
        offset = UNIVERSE_HEADER_SIZE
        for name, blob in segments.items():
            toc[name] = [offset, len(blob)]
            offset += len(blob)
        toc_blob = msgpack.packb(toc, use_bin_type=True)

        self.generation += 1
        self.toc_offset = offset
        self.toc_len = len(toc_blob)

        tmp_path = self.path + ".tmp"
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(self._header_bytes())
            for name in toc:
                f.write(segments[name])
            f.write(toc_blob)
            if journal:
                f.write(journal)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)
        self._toc_cache = None
        self._toc_cache_generation = -1

    def append_journal(self, blob: bytes) -> None:
        """Pure O(1) append at EOF — the journal region is [TOC end, EOF)."""
        if not blob:
            return
        with open(self.path, "ab") as f:
            f.write(blob)

    def truncate_journal(self) -> None:
        with open(self.path, "r+b") as f:
            f.truncate(self.toc_offset + self.toc_len)

    # ------------------------------------------------------------------- reads

    def _toc(self) -> Dict[str, Tuple[int, int]]:
        self._read_header()
        if self._toc_cache is not None and self._toc_cache_generation == self.generation:
            return self._toc_cache
        if self.toc_len == 0:
            toc: Dict[str, Tuple[int, int]] = {}
        else:
            with open(self.path, "rb") as f:
                f.seek(self.toc_offset)
                raw = msgpack.unpackb(f.read(self.toc_len), raw=False)
            toc = {name: (int(off), int(length)) for name, (off, length) in raw.items()}
        self._toc_cache = toc
        self._toc_cache_generation = self.generation
        return toc

    def segment_names(self) -> List[str]:
        return list(self._toc().keys())

    def get_segment(self, name: str) -> Optional[bytes]:
        toc = self._toc()
        if name not in toc:
            return None
        offset, length = toc[name]
        if length == 0:
            return b""
        with open(self.path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return bytes(mm[offset:offset + length])

    def read_journal(self) -> bytes:
        self._read_header()
        journal_start = self.toc_offset + self.toc_len
        size = os.path.getsize(self.path)
        if size <= journal_start:
            return b""
        with open(self.path, "rb") as f:
            f.seek(journal_start)
            return f.read(size - journal_start)

    def stats(self) -> Dict[str, Any]:
        toc = self._toc()
        size = os.path.getsize(self.path)
        journal_start = self.toc_offset + self.toc_len
        return {
            "path": self.path,
            "seed": self.seed,
            "dimension": self.dimension,
            "generation": self.generation,
            "segments": {name: length for name, (_, length) in toc.items()},
            "journal_bytes": max(0, size - journal_start),
            "total_bytes": size,
        }
