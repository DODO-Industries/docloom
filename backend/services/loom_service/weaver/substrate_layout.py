import os
import struct
import mmap
import threading
import msgpack
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Iterator

# =============================================================================
# DOCLOOM — PHYSICAL SUBSTRATE LAYOUT (The .loom Container File)
# =============================================================================
# v2 ("LOM2") — Append-Only Virtualized File System
#
#   .loom      = 64-byte mutable header + pure append-only record log.
#                The log is NEVER rewritten and NEVER truncated. One shard
#                ingest = one seek + one write. Millions of virtual sub-files
#                live inside a single OS file (no file-tree overhead).
#
#   .loom.idx  = disposable acceleration snapshot ("LIX2"): byte offsets,
#                mass array, precomputed norms, a CONTIGUOUS float32 vector
#                matrix, routing leaders, shard ids, and a bucket_id per shard
#                (which leader neighborhood it resolved to at ingest — the
#                resonance-jump candidate narrowing in recall() reads this) —
#                all mmap'd zero-copy. If deleted or stale it is rebuilt from
#                the log (the log is the single source of truth).
#
#   Deep Sleep = dormant shards cost 0 bytes of heap. Reads jump straight to
#                byte offsets via mmap; the OS page cache wakes only the
#                sectors a query actually touches.
#
# v1 ("LOOM") files remain readable through the LoomSubstrate facade and are
# auto-migrated to v2 on first write.
# =============================================================================

# ---- v1 constants (legacy format, read-only support) ----
V1_HEADER_FORMAT = "<4sHQIIIQQQ"  # magic, version, seed, num_leaders, num_shards, vector_dim, off_routing, off_dense, off_journal
V1_MAGIC = b"LOOM"
HEADER_SIZE = 64  # shared by v1 and v2

# ---- v2 constants ----
V2_MAGIC = b"LOM2"
V2_VERSION = 2
V2_HEADER_FORMAT = "<4sHQIIQQ"  # magic, version, seed, vector_dim, flags, record_count, data_end
V2_HEADER_STRUCT = struct.Struct(V2_HEADER_FORMAT)

REC_SHARD = 1
REC_LEADER = 2
REC_LEN_STRUCT = struct.Struct("<I")   # rec_len = bytes after this u32 (type byte + body)
U16 = struct.Struct("<H")
U32 = struct.Struct("<I")
F32 = struct.Struct("<f")

IDX_MAGIC = b"LIX2"
IDX_VERSION = 2  # v2 adds a trailing bucket_ids (uint16 x n) array — see checkpoint()/_try_load_idx()
IDX_HEADER_FORMAT = "<4sHQQII"  # magic, version, covered_data_end, n_records, vector_dim, n_leaders
IDX_HEADER_STRUCT = struct.Struct(IDX_HEADER_FORMAT)

# Sentinel bucket_id for shards ingested before leader-bucket tracking existed,
# or written through a path that doesn't assign one. Never matches a real
# leader index, but recall() always includes sentinel shards as a safety net
# so no shard silently becomes unreachable via the fast candidate-narrowing path.
UNBUCKETED = 0xFFFF

# Fold the RAM tail into the .idx snapshot once it grows past this many records.
DEFAULT_CHECKPOINT_TAIL_LIMIT = 25000


def _np_default(obj):
    """msgpack fallback so numpy scalars/arrays in metadata never poison a write."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot serialize type {type(obj)} into shard metadata")


def _pack_leader(sig: np.ndarray) -> bytes:
    """{-1,1} (>=128 dims) -> 16 packed bytes."""
    sig128 = np.asarray(sig)[:128]
    if len(sig128) < 128:
        sig128 = np.pad(sig128, (0, 128 - len(sig128)), constant_values=1)
    binary = ((sig128 + 1) // 2).astype(np.uint8)
    return np.packbits(binary).tobytes()


def _unpack_leader(chunk: bytes) -> np.ndarray:
    binary = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8))[:128]
    return (binary.astype(np.int8) * 2) - 1


def _encode_shard_record(shard_id: str, vector: np.ndarray, mass: float, meta: Dict[str, Any], vector_dim: int) -> bytes:
    id_bytes = shard_id.encode("utf-8")
    if len(id_bytes) > 0xFFFF:
        raise ValueError("shard_id too long (max 65535 utf-8 bytes)")
    vec = np.asarray(vector, dtype=np.float32)
    if vec.shape != (vector_dim,):
        raise ValueError(f"Vector must have shape ({vector_dim},), got {vec.shape}")
    meta_bytes = msgpack.packb(meta, use_bin_type=True, default=_np_default)
    body = b"".join((
        U16.pack(len(id_bytes)),
        id_bytes,
        F32.pack(float(mass)),
        vec.tobytes(),
        U32.pack(len(meta_bytes)),
        meta_bytes,
    ))
    rec_len = 1 + len(body)  # type byte + body
    return REC_LEN_STRUCT.pack(rec_len) + bytes([REC_SHARD]) + body


def _encode_leader_record(sig: np.ndarray) -> bytes:
    packed = _pack_leader(sig)
    return REC_LEN_STRUCT.pack(1 + 16) + bytes([REC_LEADER]) + packed


class _ParsedRecord:
    __slots__ = ("offset", "rec_type", "next_offset", "shard_id", "mass", "vector_offset", "meta_offset", "meta_len", "leader")

    def __init__(self):
        self.offset = 0
        self.rec_type = 0
        self.next_offset = 0
        self.shard_id = None
        self.mass = 0.0
        self.vector_offset = 0
        self.meta_offset = 0
        self.meta_len = 0
        self.leader = None


def _parse_record(buf, offset: int, end: int, vector_dim: int) -> Optional[_ParsedRecord]:
    """
    Strictly validates and parses one record at `offset`. Returns None if the
    bytes do not form a complete, internally-consistent record (torn tail).
    `buf` is any buffer supporting slicing (mmap or bytes).
    """
    if offset + 5 > end:
        return None
    (rec_len,) = REC_LEN_STRUCT.unpack_from(buf, offset)
    if rec_len < 1 or offset + 4 + rec_len > end:
        return None
    rec_type = buf[offset + 4]
    body_off = offset + 5
    body_len = rec_len - 1
    rec = _ParsedRecord()
    rec.offset = offset
    rec.rec_type = rec_type
    rec.next_offset = offset + 4 + rec_len

    if rec_type == REC_LEADER:
        if body_len != 16:
            return None
        rec.leader = bytes(buf[body_off:body_off + 16])
        return rec

    if rec_type == REC_SHARD:
        if body_len < 2 + 4 + vector_dim * 4 + 4:
            return None
        (id_len,) = U16.unpack_from(buf, body_off)
        pos = body_off + 2
        expected = 2 + id_len + 4 + vector_dim * 4 + 4
        if body_len < expected:
            return None
        try:
            rec.shard_id = bytes(buf[pos:pos + id_len]).decode("utf-8")
        except UnicodeDecodeError:
            return None
        pos += id_len
        (rec.mass,) = F32.unpack_from(buf, pos)
        pos += 4
        rec.vector_offset = pos
        pos += vector_dim * 4
        (meta_len,) = U32.unpack_from(buf, pos)
        pos += 4
        if body_len != expected + meta_len:
            return None
        rec.meta_offset = pos
        rec.meta_len = meta_len
        return rec

    return None


class LoomStore:
    """
    Handle to one .loom v2 crystal. Owns the append path, the mmap read path,
    the RAM tail, and .idx checkpointing. Single-writer-per-process; guarded
    by an internal lock for thread safety.
    """

    def __init__(
        self,
        path: str,
        vector_dim: Optional[int] = None,
        seed: int = 0,
        writable: bool = True,
        fsync: bool = False,
        checkpoint_tail_limit: int = DEFAULT_CHECKPOINT_TAIL_LIMIT,
    ):
        self.path = os.path.abspath(path)
        self.idx_path = self.path + ".idx"
        self.writable = writable
        self.fsync = fsync
        self.checkpoint_tail_limit = max(1, int(checkpoint_tail_limit))
        self._lock = threading.RLock()

        self._f = None
        self._mm: Optional[mmap.mmap] = None
        self._idx_f = None
        self._idx_mm: Optional[mmap.mmap] = None

        # Base (checkpointed) views — zero-copy over .idx mmap
        self._base_n = 0
        self._base_offsets: Optional[np.ndarray] = None
        self._base_mass: Optional[np.ndarray] = None
        self._base_norms: Optional[np.ndarray] = None
        self._base_matrix: Optional[np.ndarray] = None
        self._base_bucket_ids: Optional[np.ndarray] = None
        self._base_leaders: List[bytes] = []
        self._base_ids_blob: Optional[bytes] = None
        self._base_ids_cache: Optional[List[str]] = None

        # RAM tail (records after the checkpoint stamp; bounded by checkpoint_tail_limit)
        self._tail_offsets: List[int] = []
        self._tail_ids: List[str] = []
        self._tail_mass: List[float] = []
        self._tail_vectors: List[np.ndarray] = []
        self._tail_metas: List[bytes] = []          # raw msgpack blobs
        self._tail_leaders: List[bytes] = []

        # Lazy caches
        self._id_index: Optional[Dict[str, int]] = None
        self._mass_cache: Optional[np.ndarray] = None
        self._norms_cache: Optional[np.ndarray] = None
        self._bucket_id_cache: Optional[np.ndarray] = None
        self._tail_matrix_cache: Optional[np.ndarray] = None
        self._leader_matrix_cache: Optional[np.ndarray] = None

        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        if exists:
            magic = self._peek_magic()
            if magic == V1_MAGIC:
                if not writable:
                    raise ValueError("v1 .loom file: open via LoomSubstrate facade or writable LoomStore (auto-migrates)")
                self._migrate_v1()
            elif magic != V2_MAGIC:
                raise ValueError(f"Not a .loom file (magic={magic!r}): {self.path}")
            self._open_existing(vector_dim)
        else:
            if vector_dim is None:
                raise ValueError("vector_dim is required to create a new .loom store")
            if not writable:
                raise FileNotFoundError(self.path)
            self._create_new(vector_dim, seed)

    # ------------------------------------------------------------------ setup

    def _peek_magic(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read(4)

    def _create_new(self, vector_dim: int, seed: int) -> None:
        self.vector_dim = int(vector_dim)
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.flags = 0
        self.record_count = 0
        self.data_end = HEADER_SIZE
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(self._header_bytes())
        self._open_file()

    def _header_bytes(self) -> bytes:
        header = V2_HEADER_STRUCT.pack(
            V2_MAGIC, V2_VERSION, self.seed, self.vector_dim,
            self.flags, self.record_count, self.data_end
        )
        return header.ljust(HEADER_SIZE, b"\x00")

    def _open_file(self) -> None:
        mode = "r+b" if self.writable else "rb"
        self._f = open(self.path, mode)
        self._remap_log()

    def _remap_log(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        self._f.seek(0, os.SEEK_END)
        size = self._f.tell()
        if size > 0:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def _open_existing(self, expected_dim: Optional[int]) -> None:
        with open(self.path, "rb") as f:
            header = f.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError("Truncated .loom header")
        magic, version, seed, vector_dim, flags, record_count, data_end = V2_HEADER_STRUCT.unpack(
            header[:V2_HEADER_STRUCT.size]
        )
        if magic != V2_MAGIC:
            raise ValueError(f"Invalid v2 magic: {magic!r}")
        if expected_dim is not None and expected_dim != vector_dim:
            raise ValueError(f"Dimension mismatch: store has {vector_dim}, expected {expected_dim}")
        self.vector_dim = vector_dim
        self.seed = seed
        self.flags = flags
        self.record_count = record_count
        self.data_end = data_end
        self._open_file()

        file_size = self._mm.size() if self._mm else HEADER_SIZE
        if self.data_end > file_size:
            # Header claims more data than exists (torn header write) — rescan from scratch.
            self.data_end = HEADER_SIZE
            self.record_count = 0

        # Load checkpoint snapshot if present & consistent with the log.
        base_end = HEADER_SIZE
        if self._try_load_idx():
            base_end = self._idx_covered_end
        # Scan the tail (records after the snapshot) into RAM.
        self._scan_tail(base_end, self.data_end)
        # Crash recovery: adopt complete records written past the committed header.
        if self.writable and file_size > self.data_end:
            self._recover_beyond(self.data_end, file_size)
        # A large un-indexed tail (missing/stale .idx) would otherwise sit in
        # RAM — fold it into a fresh snapshot immediately (flat-heap guarantee).
        if self.writable and len(self._tail_ids) >= self.checkpoint_tail_limit:
            self.checkpoint()

    # -------------------------------------------------------------- idx layer

    def _close_idx(self) -> None:
        if self._idx_mm is not None:
            try:
                self._idx_mm.close()
            except Exception:
                pass
            self._idx_mm = None
        if self._idx_f is not None:
            try:
                self._idx_f.close()
            except Exception:
                pass
            self._idx_f = None
        self._base_n = 0
        self._base_offsets = None
        self._base_mass = None
        self._base_norms = None
        self._base_matrix = None
        self._base_bucket_ids = None
        self._base_leaders = []
        self._base_ids_blob = None
        self._base_ids_cache = None
        self._leader_matrix_cache = None

    def _try_load_idx(self) -> bool:
        self._idx_covered_end = HEADER_SIZE
        if not os.path.exists(self.idx_path):
            return False
        try:
            self._idx_f = open(self.idx_path, "rb")
            self._idx_mm = mmap.mmap(self._idx_f.fileno(), 0, access=mmap.ACCESS_READ)
            mm = self._idx_mm
            if len(mm) < IDX_HEADER_STRUCT.size:
                raise ValueError("idx too small")
            magic, version, covered_end, n, dim, n_leaders = IDX_HEADER_STRUCT.unpack_from(mm, 0)
            if magic != IDX_MAGIC or version != IDX_VERSION:
                # Includes the old v1 idx format (no bucket_ids array) — discard
                # and rebuild from the log so every snapshot gains bucket_ids.
                raise ValueError("bad idx magic/version")
            if dim != self.vector_dim:
                raise ValueError("idx dim mismatch")
            if covered_end > self.data_end or n > self.record_count:
                raise ValueError("idx is ahead of log (stale header?)")
            pos = IDX_HEADER_STRUCT.size
            offsets = np.frombuffer(mm, dtype="<u8", count=n, offset=pos); pos += n * 8
            mass = np.frombuffer(mm, dtype="<f4", count=n, offset=pos); pos += n * 4
            norms = np.frombuffer(mm, dtype="<f4", count=n, offset=pos); pos += n * 4
            matrix = np.frombuffer(mm, dtype="<f4", count=n * dim, offset=pos).reshape(n, dim); pos += n * dim * 4
            leaders = [bytes(mm[pos + i * 16: pos + (i + 1) * 16]) for i in range(n_leaders)]; pos += n_leaders * 16
            (ids_len,) = U32.unpack_from(mm, pos); pos += 4
            if pos + ids_len > len(mm):
                raise ValueError("idx ids blob truncated")
            ids_blob = bytes(mm[pos:pos + ids_len])
            pos += ids_len
            bucket_ids = np.frombuffer(mm, dtype="<u2", count=n, offset=pos); pos += n * 2

            self._base_n = n
            self._base_offsets = offsets
            self._base_mass = mass
            self._base_norms = norms
            self._base_matrix = matrix
            self._base_bucket_ids = bucket_ids
            self._base_leaders = leaders
            self._base_ids_blob = ids_blob
            self._idx_covered_end = covered_end
            return True
        except Exception:
            # Stale/corrupt snapshot: discard, rebuild from the log later.
            self._close_idx()
            self._idx_covered_end = HEADER_SIZE
            return False

    def _base_ids(self) -> List[str]:
        if self._base_ids_cache is None:
            if self._base_ids_blob:
                self._base_ids_cache = msgpack.unpackb(self._base_ids_blob, raw=False)
            else:
                self._base_ids_cache = []
        return self._base_ids_cache

    # ------------------------------------------------------------- log scans

    def _scan_tail(self, start: int, end: int) -> None:
        """Sequentially adopt committed records in [start, end) into the RAM tail."""
        if self._mm is None or start >= end:
            return
        offset = start
        while offset < end:
            rec = _parse_record(self._mm, offset, end, self.vector_dim)
            if rec is None:
                # Committed region must parse cleanly; a failure means dim
                # mismatch or corruption. Stop adopting rather than guessing.
                break
            self._adopt_record(rec)
            offset = rec.next_offset

    def _recover_beyond(self, start: int, file_size: int) -> None:
        """Adopt complete, valid records written past the committed header (crash tail)."""
        offset = start
        adopted = 0
        while offset < file_size:
            rec = _parse_record(self._mm, offset, file_size, self.vector_dim)
            if rec is None:
                break
            self._adopt_record(rec)
            offset = rec.next_offset
            adopted += 1
        if adopted:
            self.data_end = offset
            self.record_count += adopted
            self._write_header()

    def _adopt_record(self, rec: _ParsedRecord) -> None:
        if rec.rec_type == REC_LEADER:
            self._tail_leaders.append(rec.leader)
            return
        self._tail_offsets.append(rec.offset)
        self._tail_ids.append(rec.shard_id)
        self._tail_mass.append(rec.mass)
        vec = np.frombuffer(self._mm, dtype="<f4", count=self.vector_dim, offset=rec.vector_offset).copy()
        self._tail_vectors.append(vec)
        self._tail_metas.append(bytes(self._mm[rec.meta_offset:rec.meta_offset + rec.meta_len]))

    # ------------------------------------------------------------ write path

    def _write_header(self) -> None:
        self._f.seek(0)
        self._f.write(self._header_bytes())
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())

    def _append_blob(self, blob: bytes) -> None:
        self._f.seek(self.data_end)
        self._f.write(blob)
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())

    def append_shard(self, shard_id: str, vector: np.ndarray, mass: float, meta: Dict[str, Any]) -> int:
        """Appends one shard record. Returns its global record index."""
        return self.append_batch([(shard_id, vector, mass, meta)])[0]

    def append_batch(self, items: List[Tuple[str, np.ndarray, float, Dict[str, Any]]]) -> List[int]:
        """Appends many shard records with a single write + single header commit."""
        if not self.writable:
            raise IOError("Store opened read-only")
        if not items:
            return []
        with self._lock:
            blobs = []
            offset = self.data_end
            new_offsets = []
            for shard_id, vector, mass, meta in items:
                blob = _encode_shard_record(shard_id, vector, mass, meta, self.vector_dim)
                blobs.append(blob)
                new_offsets.append(offset)
                offset += len(blob)
            payload = b"".join(blobs)
            self._append_blob(payload)

            indices = []
            base_idx = self.n
            for i, (shard_id, vector, mass, meta) in enumerate(items):
                self._tail_offsets.append(new_offsets[i])
                self._tail_ids.append(shard_id)
                self._tail_mass.append(float(mass))
                self._tail_vectors.append(np.asarray(vector, dtype=np.float32).copy())
                self._tail_metas.append(msgpack.packb(meta, use_bin_type=True, default=_np_default))
                if self._id_index is not None:
                    self._id_index[shard_id] = base_idx + i
                indices.append(base_idx + i)

            self.data_end = offset
            self.record_count += len(items)
            self._write_header()
            self._mass_cache = None
            self._norms_cache = None
            self._bucket_id_cache = None
            self._tail_matrix_cache = None

            if len(self._tail_ids) >= self.checkpoint_tail_limit:
                self.checkpoint()
            return indices

    def append_leader(self, sig: np.ndarray) -> None:
        if not self.writable:
            raise IOError("Store opened read-only")
        with self._lock:
            blob = _encode_leader_record(sig)
            self._append_blob(blob)
            self._tail_leaders.append(_pack_leader(sig))
            self.data_end += len(blob)
            self.record_count += 1
            self._write_header()
            self._leader_matrix_cache = None

    def checkpoint(self) -> None:
        """
        Folds base + RAM tail into a fresh .idx snapshot (atomic replace).
        The log itself is untouched — the snapshot is derived, disposable data.
        """
        if not self.writable:
            return
        with self._lock:
            n = self.n
            all_ids = self.get_all_ids()
            all_mass = self.mass_all().astype("<f4", copy=False)
            all_offsets = self._offsets_all()
            leaders = self._base_leaders + self._tail_leaders

            tmp_path = self.idx_path + ".tmp"
            with open(tmp_path, "wb") as out:
                out.write(IDX_HEADER_STRUCT.pack(
                    IDX_MAGIC, IDX_VERSION, self.data_end, n, self.vector_dim, len(leaders)
                ))
                out.write(np.asarray(all_offsets, dtype="<u8").tobytes())
                out.write(all_mass.tobytes())
                # norms + matrix (stream base matrix from mmap, then tail)
                norms = np.empty(n, dtype="<f4")
                if self._base_matrix is not None and self._base_n:
                    norms[:self._base_n] = self._base_norms
                if self._tail_vectors:
                    tail_mat = self._tail_matrix()
                    norms[self._base_n:] = np.linalg.norm(tail_mat, axis=1).astype("<f4")
                out.write(norms.tobytes())
                # Stream the matrix in row blocks — never materialize it whole.
                if self._base_matrix is not None and self._base_n:
                    for start in range(0, self._base_n, 65536):
                        stop = min(start + 65536, self._base_n)
                        out.write(np.ascontiguousarray(self._base_matrix[start:stop], dtype="<f4").tobytes())
                if self._tail_vectors:
                    out.write(self._tail_matrix().astype("<f4", copy=False).tobytes())
                for leader in leaders:
                    out.write(leader)
                ids_blob = msgpack.packb(all_ids, use_bin_type=True)
                out.write(U32.pack(len(ids_blob)))
                out.write(ids_blob)
                out.write(self.bucket_ids_all().astype("<u2", copy=False).tobytes())
                out.flush()
                if self.fsync:
                    os.fsync(out.fileno())

            self._close_idx()
            os.replace(tmp_path, self.idx_path)

            # Reset tail and remap everything against the fresh snapshot.
            self._tail_offsets = []
            self._tail_ids = []
            self._tail_mass = []
            self._tail_vectors = []
            self._tail_metas = []
            self._tail_leaders = []
            self._tail_matrix_cache = None
            self._bucket_id_cache = None
            self._leader_matrix_cache = None
            self._remap_log()
            if not self._try_load_idx():
                raise IOError("Failed to load freshly written .idx snapshot")

    # ------------------------------------------------------------- read path

    @property
    def n(self) -> int:
        return self._base_n + len(self._tail_ids)

    @property
    def num_leaders(self) -> int:
        return len(self._base_leaders) + len(self._tail_leaders)

    def _offsets_all(self) -> np.ndarray:
        if self._base_offsets is not None and self._base_n:
            if self._tail_offsets:
                return np.concatenate([
                    np.asarray(self._base_offsets, dtype=np.uint64),
                    np.asarray(self._tail_offsets, dtype=np.uint64),
                ])
            return np.asarray(self._base_offsets, dtype=np.uint64)
        return np.asarray(self._tail_offsets, dtype=np.uint64)

    def _tail_matrix(self) -> np.ndarray:
        if self._tail_matrix_cache is None:
            if self._tail_vectors:
                self._tail_matrix_cache = np.vstack(self._tail_vectors)
            else:
                self._tail_matrix_cache = np.empty((0, self.vector_dim), dtype=np.float32)
        return self._tail_matrix_cache

    def get_vector(self, idx: int) -> np.ndarray:
        with self._lock:
            if idx < 0 or idx >= self.n:
                raise IndexError("Shard index out of bounds.")
            if idx < self._base_n:
                return np.array(self._base_matrix[idx], dtype=np.float32)
            return self._tail_vectors[idx - self._base_n].copy()

    def get_id(self, idx: int) -> str:
        with self._lock:
            if idx < 0 or idx >= self.n:
                raise IndexError("Shard index out of bounds.")
            if idx >= self._base_n:
                return self._tail_ids[idx - self._base_n]
            rec = self._parse_at(int(self._base_offsets[idx]))
            return rec.shard_id

    def get_meta(self, idx: int) -> Dict[str, Any]:
        with self._lock:
            if idx < 0 or idx >= self.n:
                raise IndexError("Shard index out of bounds.")
            if idx >= self._base_n:
                return msgpack.unpackb(self._tail_metas[idx - self._base_n], raw=False)
            rec = self._parse_at(int(self._base_offsets[idx]))
            blob = bytes(self._mm[rec.meta_offset:rec.meta_offset + rec.meta_len])
            return msgpack.unpackb(blob, raw=False) if blob else {}

    def get_mass(self, idx: int) -> float:
        return float(self.mass_all()[idx])

    def _parse_at(self, offset: int) -> _ParsedRecord:
        rec = _parse_record(self._mm, offset, self.data_end, self.vector_dim)
        if rec is None or rec.rec_type != REC_SHARD:
            raise IOError(f"Corrupt shard record at offset {offset} in {self.path}")
        return rec

    def get_all_ids(self) -> List[str]:
        with self._lock:
            return list(self._base_ids()) + list(self._tail_ids)

    def index_of(self, shard_id: str) -> Optional[int]:
        with self._lock:
            if self._id_index is None:
                self._id_index = {sid: i for i, sid in enumerate(self.get_all_ids())}
            return self._id_index.get(shard_id)

    def get_leaders(self) -> List[np.ndarray]:
        with self._lock:
            return [_unpack_leader(b) for b in (self._base_leaders + self._tail_leaders)]

    def get_leaders_packed(self) -> List[bytes]:
        with self._lock:
            return list(self._base_leaders) + list(self._tail_leaders)

    def get_leader_matrix(self) -> np.ndarray:
        """
        Cached (num_leaders, 16) uint8 matrix of every leader signature —
        rebuilt only when a leader is actually appended, not on every call.
        The leader set is read on every ingest (bucket resolution) and every
        large-crystal recall (resonance-jump entry), so avoiding a fresh
        list-copy + byte-join + reshape each time matters at scale.
        """
        with self._lock:
            if self._leader_matrix_cache is None:
                leaders = self._base_leaders + self._tail_leaders
                if leaders:
                    self._leader_matrix_cache = np.frombuffer(
                        b"".join(leaders), dtype=np.uint8
                    ).reshape(len(leaders), 16)
                else:
                    self._leader_matrix_cache = np.empty((0, 16), dtype=np.uint8)
            return self._leader_matrix_cache

    def mass_all(self) -> np.ndarray:
        with self._lock:
            if self._mass_cache is None:
                parts = []
                if self._base_mass is not None and self._base_n:
                    parts.append(np.asarray(self._base_mass, dtype=np.float32))
                if self._tail_mass:
                    parts.append(np.asarray(self._tail_mass, dtype=np.float32))
                self._mass_cache = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
            return self._mass_cache

    def norms_all(self) -> np.ndarray:
        with self._lock:
            if self._norms_cache is None:
                parts = []
                if self._base_norms is not None and self._base_n:
                    parts.append(np.asarray(self._base_norms, dtype=np.float32))
                if self._tail_vectors:
                    parts.append(np.linalg.norm(self._tail_matrix(), axis=1).astype(np.float32))
                self._norms_cache = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
            return self._norms_cache

    def bucket_ids_all(self) -> np.ndarray:
        """
        The LSH leader-bucket each shard resolved to at ingest time (uint16;
        UNBUCKETED for shards written before this feature existed, or through
        a path that never assigned one). Small enough (2 bytes/shard) to
        always materialize fully — unlike the vector matrix, this is never
        the expensive part of a query.
        """
        with self._lock:
            if self._bucket_id_cache is None:
                parts = []
                if self._base_bucket_ids is not None and self._base_n:
                    parts.append(np.asarray(self._base_bucket_ids, dtype=np.uint16))
                if self._tail_metas:
                    tail_bids = np.full(len(self._tail_metas), UNBUCKETED, dtype=np.uint16)
                    for i, blob in enumerate(self._tail_metas):
                        try:
                            meta = msgpack.unpackb(blob, raw=False)
                            tail_bids[i] = meta.get("bucket_id", UNBUCKETED)
                        except Exception:
                            pass
                    parts.append(tail_bids)
                self._bucket_id_cache = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)
            return self._bucket_id_cache

    def dot_all(self, q: np.ndarray) -> np.ndarray:
        """
        Vectorized dot product of every stored vector against q.
        Base part streams through the mmap'd contiguous matrix (page cache,
        no heap copy); tail part is the small in-RAM matrix.
        """
        with self._lock:
            q = np.asarray(q, dtype=np.float32)
            out = np.empty(self.n, dtype=np.float32)
            if self._base_n:
                out[:self._base_n] = self._base_matrix @ q
            if self._tail_vectors:
                out[self._base_n:] = self._tail_matrix() @ q
            return out

    def gather_vectors(self, idx_array: np.ndarray) -> np.ndarray:
        """
        Fancy-index gather of specific rows — the actual payoff of resonance-
        jump candidate narrowing in recall(): touches only the candidate rows
        through the mmap'd base matrix (OS page cache) instead of scanning
        the whole crystal. Cheap whenever idx_array is a small fraction of n.
        """
        with self._lock:
            idx_array = np.asarray(idx_array, dtype=np.int64)
            out = np.empty((len(idx_array), self.vector_dim), dtype=np.float32)
            if len(idx_array) == 0:
                return out
            base_mask = idx_array < self._base_n
            if np.any(base_mask):
                out[base_mask] = self._base_matrix[idx_array[base_mask]]
            tail_mask = ~base_mask
            if np.any(tail_mask):
                tail_positions = idx_array[tail_mask] - self._base_n
                out[tail_mask] = self._tail_matrix()[tail_positions]
            return out

    def iter_vector_blocks(self, block_size: int = 65536) -> Iterator[Tuple[int, np.ndarray]]:
        """Yields (start_index, float32 block) across base + tail without loading everything."""
        for start in range(0, self._base_n, block_size):
            stop = min(start + block_size, self._base_n)
            yield start, np.asarray(self._base_matrix[start:stop])
        if self._tail_vectors:
            yield self._base_n, self._tail_matrix()

    def get_journal(self) -> List[Dict[str, Any]]:
        """Compat shape: full journal decode — [{shard_id, meta}, ...]. O(n); avoid on hot paths."""
        return [{"shard_id": self.get_id(i), "meta": self.get_meta(i)} for i in range(self.n)]

    # ------------------------------------------------------------- migration

    def _migrate_v1(self) -> None:
        """Rewrites a legacy v1 container as a v2 append-only log (atomic replace)."""
        meta = _v1_read_metadata(self.path)
        leaders = _v1_get_routing_space(self.path)
        journal = _v1_get_journal(self.path)
        dim = meta["vector_dim"]
        seed = meta["seed"]

        tmp_path = self.path + ".v2tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        tmp = LoomStore(tmp_path, vector_dim=dim, seed=seed, writable=True, fsync=self.fsync)
        try:
            items = []
            for idx, entry in enumerate(journal):
                vec = _v1_get_vector_mmap(self.path, idx)
                m = entry.get("meta", {}) or {}
                mass = float(m.get("mass", 1.0))
                items.append((entry["shard_id"], vec, mass, m))
            if items:
                tmp.append_batch(items)
            for leader in leaders:
                tmp.append_leader(leader)
            tmp.checkpoint()
        finally:
            tmp.close()
        os.replace(tmp_path, self.path)
        if os.path.exists(tmp_path + ".idx"):
            os.replace(tmp_path + ".idx", self.idx_path)

    # -------------------------------------------------------------- lifecycle

    def close(self) -> None:
        with self._lock:
            try:
                if self.writable and self._tail_ids:
                    self.checkpoint()
            except Exception:
                pass
            self._close_idx()
            if self._mm is not None:
                try:
                    self._mm.close()
                except Exception:
                    pass
                self._mm = None
            if self._f is not None:
                try:
                    self._f.close()
                except Exception:
                    pass
                self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# =============================================================================
# v1 readers (legacy) — kept verbatim so old crystals stay readable.
# =============================================================================

def _v1_read_metadata(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError("File is too small to contain a valid .loom header.")
    magic, version, seed, num_leaders, num_shards, vector_dim, offset_routing, offset_dense, offset_journal = struct.unpack(
        V1_HEADER_FORMAT, header_bytes[:struct.calcsize(V1_HEADER_FORMAT)]
    )
    if magic != V1_MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic}")
    return {
        "version": version,
        "seed": seed,
        "num_leaders": num_leaders,
        "num_shards": num_shards,
        "vector_dim": vector_dim,
        "offset_routing": offset_routing,
        "offset_dense": offset_dense,
        "offset_journal": offset_journal,
    }


def _v1_get_vector_mmap(path: str, shard_idx: int) -> np.ndarray:
    meta = _v1_read_metadata(path)
    if shard_idx < 0 or shard_idx >= meta["num_shards"]:
        raise IndexError("Shard index out of bounds.")
    vector_dim = meta["vector_dim"]
    offset = meta["offset_dense"] + shard_idx * vector_dim * 4
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.seek(offset)
            vector_bytes = mm.read(vector_dim * 4)
    return np.frombuffer(vector_bytes, dtype=np.float32).copy()


def _v1_get_routing_space(path: str) -> List[np.ndarray]:
    meta = _v1_read_metadata(path)
    num_leaders = meta["num_leaders"]
    if num_leaders == 0:
        return []
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.seek(meta["offset_routing"])
            routing_bytes = mm.read(num_leaders * 16)
    return [_unpack_leader(routing_bytes[i * 16:(i + 1) * 16]) for i in range(num_leaders)]


def _v1_get_journal(path: str) -> List[Dict[str, Any]]:
    meta = _v1_read_metadata(path)
    entries = []
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.seek(meta["offset_journal"])
            file_size = mm.size()
            while mm.tell() < file_size:
                length_bytes = mm.read(4)
                if len(length_bytes) < 4:
                    break
                entry_len = struct.unpack("<I", length_bytes)[0]
                entry_bytes = mm.read(entry_len)
                entries.append(msgpack.unpackb(entry_bytes, raw=False))
    return entries


def detect_version(path: str) -> int:
    """Returns 1 (legacy) or 2 (append-only) for a .loom container."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == V2_MAGIC:
        return 2
    if magic == V1_MAGIC:
        return 1
    raise ValueError(f"Not a .loom file (magic={magic!r}): {path}")


_detect_version = detect_version  # internal alias


class LoomSubstrate:
    """
    ============================================================================
    Static compatibility facade over v1 and v2 .loom containers.
    Existing callers (routes, decoder, tools) keep working unchanged; new
    files are always written as v2. Hot paths should use LoomStore directly.
    ============================================================================
    """

    @staticmethod
    def write(
        path: str,
        leaders: List[np.ndarray],
        shards: List[Tuple[str, np.ndarray, Dict[str, Any]]],
        vector_dim: int,
        seed: int
    ) -> None:
        """Full (re)write — now emits a v2 append-only container."""
        if os.path.exists(path):
            os.remove(path)
        idx_path = path + ".idx"
        if os.path.exists(idx_path):
            os.remove(idx_path)
        store = LoomStore(path, vector_dim=vector_dim, seed=seed, writable=True)
        try:
            items = []
            for shard_id, vector, meta in shards:
                mass = float((meta or {}).get("mass", 1.0))
                items.append((shard_id, vector, mass, meta or {}))
            if items:
                store.append_batch(items)
            for leader in leaders:
                store.append_leader(leader)
            store.checkpoint()
        finally:
            store.close()

    @staticmethod
    def read_metadata(path: str) -> Dict[str, Any]:
        if _detect_version(path) == 1:
            return _v1_read_metadata(path)
        with LoomStore(path, writable=False) as store:
            return {
                "version": V2_VERSION,
                "seed": store.seed,
                "num_leaders": store.num_leaders,
                "num_shards": store.n,
                "vector_dim": store.vector_dim,
                "offset_routing": 0,
                "offset_dense": 0,
                "offset_journal": 0,
            }

    @staticmethod
    def get_vector_mmap(path: str, shard_idx: int) -> np.ndarray:
        if _detect_version(path) == 1:
            return _v1_get_vector_mmap(path, shard_idx)
        with LoomStore(path, writable=False) as store:
            return store.get_vector(shard_idx)

    @staticmethod
    def get_routing_space(path: str) -> List[np.ndarray]:
        if _detect_version(path) == 1:
            return _v1_get_routing_space(path)
        with LoomStore(path, writable=False) as store:
            return store.get_leaders()

    @staticmethod
    def get_journal(path: str) -> List[Dict[str, Any]]:
        if _detect_version(path) == 1:
            return _v1_get_journal(path)
        with LoomStore(path, writable=False) as store:
            return store.get_journal()
