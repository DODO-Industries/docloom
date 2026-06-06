import os
import struct
import mmap
import msgpack
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

HEADER_FORMAT = "<4sHQIIIQQQ"  # magic(4s), version(H), seed(Q), num_leaders(I), num_shards(I), vector_dim(I), offset_routing(Q), offset_dense(Q), offset_journal(Q)
HEADER_SIZE = 64

class LoomSubstrate:
    """
    ============================================================================
    DOCLOOM — PHYSICAL SUBSTRATE LAYOUT (The .loom Container File)
    ============================================================================
    Manages custom binary .loom files with a fixed-width header, leader routing
    space, dense field vectors, and cognitive journals.
    Leverages mmap for zero-copy binary offsets read.
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
        """
        Serializes leaders, dense vectors, and cognitive journals into a binary .loom file.
        Each shard tuple contains (shard_id, dense_vector, journal_metadata).
        """
        num_leaders = len(leaders)
        num_shards = len(shards)

        # Calculate offsets
        offset_routing = HEADER_SIZE
        # Routing space size: num_leaders * 16 bytes (128 bits = 16 bytes)
        routing_size = num_leaders * 16
        offset_dense = offset_routing + routing_size
        # Dense space size: num_shards * vector_dim * 4 bytes (float32)
        dense_size = num_shards * vector_dim * 4
        offset_journal = offset_dense + dense_size

        # Create header
        header = struct.pack(
            HEADER_FORMAT,
            b"LOOM",
            1,  # Version 1
            seed,  # Universe Seed Key (uint64)
            num_leaders,
            num_shards,
            vector_dim,
            offset_routing,
            offset_dense,
            offset_journal
        )
        # Pad header to 64 bytes
        header = header.ljust(HEADER_SIZE, b"\x00")

        # Compile Routing Space
        # Each leader signature is a 128-bit array of {-1, 1}
        # Pack {-1, 1} into 16 bytes (128 bits)
        routing_data = bytearray()
        for leader in leaders:
            leader_128 = leader[:128]
            if len(leader_128) < 128:
                leader_128 = np.pad(leader_128, (0, 128 - len(leader_128)), constant_values=1)
            # Binarize to 0 or 1
            binary = ((leader_128 + 1) // 2).astype(np.uint8)
            packed = np.packbits(binary)
            routing_data.extend(packed.tobytes())

        # Compile Dense Field Space
        dense_data = bytearray()
        for _, vector, _ in shards:
            vector_f32 = vector.astype(np.float32)
            dense_data.extend(vector_f32.tobytes())

        # Compile Cognitive Journal
        journal_data = bytearray()
        for shard_id, _, meta in shards:
            entry = {
                "shard_id": shard_id,
                "meta": meta
            }
            packed_entry = msgpack.packb(entry, use_bin_type=True)
            length_header = struct.pack("<I", len(packed_entry))
            journal_data.extend(length_header)
            journal_data.extend(packed_entry)

        # Write to file
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(header)
            f.write(routing_data)
            f.write(dense_data)
            f.write(journal_data)

    @staticmethod
    def read_metadata(path: str) -> Dict[str, Any]:
        """
        Reads only the fixed-width header metadata from the file.
        """
        with open(path, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
        
        if len(header_bytes) < HEADER_SIZE:
            raise ValueError("File is too small to contain a valid .loom header.")

        magic, version, seed, num_leaders, num_shards, vector_dim, offset_routing, offset_dense, offset_journal = struct.unpack(
            HEADER_FORMAT,
            header_bytes[:struct.calcsize(HEADER_FORMAT)]
        )

        if magic != b"LOOM":
            raise ValueError(f"Invalid magic bytes: {magic}")

        return {
            "version": version,
            "seed": seed,
            "num_leaders": num_leaders,
            "num_shards": num_shards,
            "vector_dim": vector_dim,
            "offset_routing": offset_routing,
            "offset_dense": offset_dense,
            "offset_journal": offset_journal
        }

    @staticmethod
    def get_vector_mmap(path: str, shard_idx: int) -> np.ndarray:
        """
        Jumps directly to the byte offset of the desired dense vector using mmap (zero-copy).
        """
        meta = LoomSubstrate.read_metadata(path)
        if shard_idx < 0 or shard_idx >= meta["num_shards"]:
            raise IndexError("Shard index out of bounds.")

        vector_dim = meta["vector_dim"]
        offset = meta["offset_dense"] + shard_idx * vector_dim * 4
        size_bytes = vector_dim * 4

        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.seek(offset)
                vector_bytes = mm.read(size_bytes)

        return np.frombuffer(vector_bytes, dtype=np.float32).copy()

    @staticmethod
    def get_routing_space(path: str) -> List[np.ndarray]:
        """
        Reads the 128-bit leader signatures from the routing space.
        """
        meta = LoomSubstrate.read_metadata(path)
        num_leaders = meta["num_leaders"]
        offset = meta["offset_routing"]
        size_bytes = num_leaders * 16

        if num_leaders == 0:
            return []

        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.seek(offset)
                routing_bytes = mm.read(size_bytes)

        leaders = []
        for i in range(num_leaders):
            chunk = routing_bytes[i*16 : (i+1)*16]
            binary = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8))[:128]
            leader_sig = (binary.astype(np.int8) * 2) - 1
            leaders.append(leader_sig)
        return leaders

    @staticmethod
    def get_journal(path: str) -> List[Dict[str, Any]]:
        """
        Reads the cognitive journal stream sequentially.
        """
        meta = LoomSubstrate.read_metadata(path)
        offset = meta["offset_journal"]

        entries = []
        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.seek(offset)
                file_size = mm.size()
                while mm.tell() < file_size:
                    length_bytes = mm.read(4)
                    if len(length_bytes) < 4:
                        break
                    entry_len = struct.unpack("<I", length_bytes)[0]
                    entry_bytes = mm.read(entry_len)
                    entry = msgpack.unpackb(entry_bytes, raw=False)
                    entries.append(entry)
        return entries
