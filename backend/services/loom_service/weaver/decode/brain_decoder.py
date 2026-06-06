import os
import sys
import json
import struct
import numpy as np
import msgpack
import zstandard as zstd
from typing import Dict, List, Any, Optional

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    # Resolve paths for standalone CLI execution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from backend.services.loom_service.weaver.atlas_router import GlobalAtlasRouter
    from backend.services.loom_service.weaver.substrate_layout import LoomSubstrate
else:
    from ..atlas_router import GlobalAtlasRouter
    from ..substrate_layout import LoomSubstrate


def hash_shard_id(shard_id: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(shard_id.encode("utf-8")).digest()[:8], byteorder="big")


def decode_atlas(path: str) -> Dict[str, Any]:
    """Decodes a binary atlas file (e.g. atlas.capnp) using GlobalAtlasRouter."""
    router = GlobalAtlasRouter(atlas_path=path)
    crystals_info = {}
    for c_path, info in router.crystals.items():
        crystals_info[os.path.basename(c_path)] = {
            "full_path": c_path,
            "state": info["state"],
            "num_shards": info["num_shards"],
            "centroid": info["centroid"].tolist()
        }
    return {
        "file_type": "GLOBAL_ATLAS_REGISTRY",
        "magic_bytes": "ATLS",
        "version": 1,
        "universe_seed_key": router.seed,
        "num_crystals": len(router.crystals),
        "crystals": crystals_info
    }

def decode_loom(path: str) -> Dict[str, Any]:
    """Decodes a custom binary .loom crystal file."""
    meta = LoomSubstrate.read_metadata(path)
    leaders = LoomSubstrate.get_routing_space(path)
    journal = LoomSubstrate.get_journal(path)
    
    leaders_decoded = [l.tolist() for l in leaders]
    shards_decoded = []
    
    for idx, entry in enumerate(journal):
        sid = entry.get("shard_id")
        meta_info = entry.get("meta", {})
        vec = LoomSubstrate.get_vector_mmap(path, idx)
        
        shards_decoded.append({
            "index": idx,
            "shard_id": sid,
            "vector": vec.tolist(),
            "text": meta_info.get("text", ""),
            "mass": meta_info.get("mass", 0.0),
            "activation": meta_info.get("activation", 0.0),
            "hits": meta_info.get("hits", 0),
            "last_recalled_timestamp": meta_info.get("last_recalled", 0.0),
            "other_metadata": {k: v for k, v in meta_info.items() if k not in ["text", "mass", "activation", "hits", "last_recalled"]}
        })
        
    return {
        "file_type": "PHYSICAL_SUBSTRATE_CONTAINER",
        "magic_bytes": "LOOM",
        "version": meta["version"],
        "universe_seed_key": meta["seed"],
        "num_leaders": meta["num_leaders"],
        "num_shards": meta["num_shards"],
        "vector_dimension": meta["vector_dim"],
        "offsets": {
            "routing": meta["offset_routing"],
            "dense": meta["offset_dense"],
            "journal": meta["offset_journal"]
        },
        "leaders": leaders_decoded,
        "shards": shards_decoded
    }

def try_load_concept_keys(dir_path: str) -> Optional[List[str]]:
    """Tries to read metadata.bin to fetch concept keys for mapping coordinate vectors."""
    meta_path = os.path.join(dir_path, "metadata.bin")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                metadata = msgpack.unpackb(f.read(), raw=False)
            return metadata.get("concept_keys", [])
        except Exception:
            pass
    return None

def decode_brain_file(path: str) -> Dict[str, Any]:
    """Decodes a specific binary/msgpack file saved in .brain_data."""
    filename = os.path.basename(path)
    dir_path = os.path.dirname(path)
    
    # 1. Coordinate / Velocity Raw Float32 Matrices
    if filename in ["coordinates.bin", "velocities.bin"]:
        keys = try_load_concept_keys(dir_path)
        with open(path, "rb") as f:
            data_bytes = f.read()
        
        if not data_bytes:
            return {"file_type": filename.upper(), "data": {}}
            
        array_flat = np.frombuffer(data_bytes, dtype=np.float32)
        if keys and len(keys) > 0:
            dim = len(array_flat) // len(keys)
            matrix = array_flat.reshape(len(keys), dim)
            mapped_data = {}
            for idx, key in enumerate(keys):
                mapped_data[key] = matrix[idx].tolist()
            return {
                "file_type": filename.upper(),
                "dimension": dim,
                "mapped_rows": len(keys),
                "data": mapped_data
            }
        else:
            return {
                "file_type": filename.upper(),
                "warning": "metadata.bin not found or empty in same folder. Displaying raw flat list.",
                "data": array_flat.tolist()
            }
            
    # 2. Zstd-Compressed Episodic Replay logs
    if filename == "replay.bin":
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            compressed = f.read()
        if not compressed:
            return {"file_type": "EPISODIC_REPLAY", "records": []}
        decompressed = dctx.decompress(compressed)
        replay_list = msgpack.unpackb(decompressed, raw=False)
        return {
            "file_type": "EPISODIC_REPLAY",
            "num_records": len(replay_list),
            "records": replay_list
        }
        
    # 3. Special file: universe.metric
    if filename == "universe.metric":
        with open(path, "rb") as f:
            metric_bytes = f.read()
        if len(metric_bytes) == 14:
            magic, dim, seed = struct.unpack("<4sHQ", metric_bytes)
            return {
                "file_type": "UNIVERSE_METRIC_KEY",
                "magic_bytes": magic.decode("ascii", errors="ignore"),
                "hyperdimensional_count": dim,
                "universe_seed_key": seed
            }
        else:
            return {"file_type": "UNIVERSE_METRIC_KEY", "error": f"Invalid metric file size: {len(metric_bytes)} bytes"}

    # 4. Special file: latent_state.snapshot
    if filename == "latent_state.snapshot":
        with open(path, "rb") as f:
            compressed = f.read()
        if not compressed:
            return {"file_type": "LATENT_STATE_SNAPSHOT", "records": []}
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed)
        num_records = len(decompressed) // 16
        records = []
        for i in range(num_records):
            chunk = decompressed[i*16 : (i+1)*16]
            h, act, t = struct.unpack("<QfI", chunk)
            records.append({
                "shard_id_hash": h,
                "activation_energy": act,
                "last_recalled_timestamp": float(t)
            })
        return {
            "file_type": "LATENT_STATE_SNAPSHOT",
            "num_records": len(records),
            "records": records
        }

    # 5. Special file: cortex_journal.bin
    if filename == "cortex_journal.bin":
        with open(path, "rb") as f:
            journal_bytes = f.read()
        if not journal_bytes:
            return {"file_type": "CORTEX_JOURNAL", "transactions": []}
        num_txns = len(journal_bytes) // 32
        txns = []
        for i in range(num_txns):
            chunk = journal_bytes[i*32 : (i+1)*32]
            magic, h, act, t, _ = struct.unpack("<4sQfd8s", chunk)
            txns.append({
                "magic": magic.decode("ascii", errors="ignore").strip("\x00"),
                "shard_id_hash": h,
                "activation_energy": act,
                "timestamp": t
            })
        return {
            "file_type": "CORTEX_JOURNAL",
            "num_transactions": len(txns),
            "transactions": txns
        }

    # 6. Standard MsgPack-Unpacked States
    with open(path, "rb") as f:
        data_bytes = f.read()
    if not data_bytes:
        return {"file_type": filename.upper(), "data": None}
        
    unpacked = msgpack.unpackb(data_bytes, raw=False)
    return {
        "file_type": filename.upper(),
        "data": unpacked
    }

def decode_file(path: str) -> Dict[str, Any]:
    """Detects file format/magic bytes and routes to appropriate decoder."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: '{path}'")
        
    if os.path.isdir(path):
        raise IsADirectoryError(f"Target is a directory: '{path}'")
        
    filename = os.path.basename(path)
    if filename in ["universe.metric", "latent_state.snapshot", "cortex_journal.bin"]:
        try:
            return decode_brain_file(path)
        except Exception as e:
            return {"error": f"Failed to decode brain file structure: {e}"}
            
    # Check magic bytes first
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"ATLS":
            return decode_atlas(path)
        if magic == b"LOOM":
            return decode_loom(path)
    except Exception as e:
        return {"error": f"Failed to check magic bytes: {e}"}
        
    # Handle brain data by filename
    try:
        return decode_brain_file(path)
    except Exception as e:
        return {"error": f"Failed to decode brain file structure: {e}"}


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("DOCLOOM STATE DECODER CLI")
        print("=" * 60)
        print("Usage:")
        print("  python backend/services/loom_service/weaver/decode/brain_decoder.py <file_path>")
        print("  python backend/services/loom_service/weaver/decode/brain_decoder.py <directory_path>")
        print("=" * 60)
        sys.exit(1)
        
    target = sys.argv[1]
    
    valid_extensions = [".bin", ".capnp", ".loom", ".snapshot"]
    valid_files = ["universe.metric", "cortex_journal.bin", "latent_state.snapshot"]

    if os.path.isdir(target):
        print(f"Scanning directory: {target}\n")
        for f in sorted(os.listdir(target)):
            if f.endswith(".json") or f.endswith(".py"):
                continue
            f_path = os.path.join(target, f)
            
            is_valid = False
            if f in valid_files:
                is_valid = True
            else:
                for ext in valid_extensions:
                    if f.endswith(ext):
                        is_valid = True
                        break
                        
            if os.path.isfile(f_path) and is_valid:
                try:
                    res = decode_file(f_path)
                    output_name = f + ".json"
                    output_path = os.path.join(target, output_name)
                    with open(output_path, "w", encoding="utf-8") as out:
                        json.dump(res, out, indent=4, ensure_ascii=False)
                    print(f"  [+] Decoded '{f}' -> saved as '{output_name}'")
                except Exception as e:
                    print(f"  [-] Failed to decode '{f}': {e}")
    else:
        try:
            res = decode_file(target)
            print(json.dumps(res, indent=4, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
