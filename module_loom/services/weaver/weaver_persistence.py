"""
DocLoom Weave Persistence Mixin
Handles Write-Ahead Journaling (.jnl), atomic checkpoint rebuilds,
sub-second boot snapshot loading, and full-fidelity cortex state persistence.
"""
import os
import time
import heapq
import struct
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import zstandard as zstd

from module_loom.config.tuning_config import tuning_manager
from module_loom.services.weaver.weaver_common import (
    hash_shard_id, default_cognitive_fields, np_msgpack_default,
    LEGACY_STATE_FILES, DEAD_CONTAINER_SEGMENTS,
)
from module_loom.services.weaver.atlas_router import STATE_MAP_TO_BYTE, STATE_MAP_FROM_BYTE
from module_loom.services.weaver.substrate_layout import LoomSubstrate, LoomStore, detect_version
from module_loom.services.weaver.universe_container import (
    UniverseContainer, pack_txn_record, pack_crystal_record, iter_journal_records,
    JREC_TXN, JREC_CRYSTAL, TXN_PAYLOAD, CRYSTAL_PREFIX,
)
from module_loom.services.cortex.latent_field_cognition.attractor_basin_compilation import CognitiveAssembly


class WeavePersistenceMixin:
    """Persistence, journaling, and checkpointing methods for WeaveBrainCoordinator."""

    def _consolidate_universe(self) -> None:
        """Folds loose legacy state files into universe.loom container."""
        segments: Dict[str, bytes] = {"atlas": self.atlas.to_bytes()}

        snapshot_path = os.path.join(self.storage_dir, "latent_state.snapshot")
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, "rb") as f:
                    segments["snapshot"] = f.read()
            except Exception:
                pass

        for name in LEGACY_STATE_FILES:
            p = os.path.join(self.storage_dir, name)
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        segments[name] = f.read()
                except Exception:
                    pass

        # Convert the legacy 32-byte TXN journal into typed container records
        journal_blob = bytearray()
        legacy_journal = os.path.join(self.storage_dir, "cortex_journal.bin")
        if os.path.exists(legacy_journal):
            try:
                with open(legacy_journal, "rb") as f:
                    raw = f.read()
                for i in range(len(raw) // 32):
                    chunk = raw[i * 32:(i + 1) * 32]
                    magic, h, act, t, _ = struct.unpack("<4sQfd8s", chunk)
                    if magic == b"TXN\x00":
                        journal_blob.extend(pack_txn_record(h, act, t))
            except Exception:
                pass

        self.universe = UniverseContainer(
            self.universe_path, seed=self.seed, dimension=self.dimension, create=True
        )
        self.universe.rebuild(segments, journal=bytes(journal_blob))

        # Retire legacy files (moved, never deleted)
        backup_dir = os.path.join(self.storage_dir, "legacy_backup")
        legacy_files = [self.metric_path, self.atlas_path, snapshot_path, legacy_journal] + \
                       [os.path.join(self.storage_dir, n) for n in LEGACY_STATE_FILES]
        for p in legacy_files:
            if os.path.exists(p):
                try:
                    os.makedirs(backup_dir, exist_ok=True)
                    os.replace(p, os.path.join(backup_dir, os.path.basename(p)))
                except Exception:
                    pass
        self.atlas.atlas_path = None

    def _mark_atlas_dirty(self) -> None:
        """Centroid drift is folded into the container at checkpoint cadence."""
        if self.universe is not None:
            self._atlas_dirty = True
        else:
            self.atlas.save()

    def _journal_crystal_registration(self, crystal_path: str) -> None:
        """Crystal registrations survive a crash via immediate journal append."""
        if self.universe is None:
            self.atlas.save()
            return
        info = self.atlas.crystals.get(crystal_path)
        if info is None:
            return
        state_byte = STATE_MAP_TO_BYTE.get(info["state"], 2)
        centroid_bytes = np.asarray(info["centroid"], dtype=np.float32).tobytes()
        self.universe.append_journal(
            pack_crystal_record(state_byte, int(info["num_shards"]), crystal_path, centroid_bytes)
        )
        self._atlas_dirty = True

    def _carryover_segments(self) -> Dict[str, bytes]:
        """Current container segments for atomic rebuilds."""
        segments: Dict[str, bytes] = {}
        if self.universe is not None:
            for name in self.universe.segment_names():
                if name in DEAD_CONTAINER_SEGMENTS:
                    continue
                blob = self.universe.get_segment(name)
                if blob is not None:
                    segments[name] = blob
        return segments

    def _commit_state_segments(self, new_segments: Dict[str, bytes]) -> None:
        """Atomically folds new state segments into universe.loom."""
        if self.universe is not None:
            journal = self.universe.read_journal()
            segments = self._carryover_segments()
            segments.update(new_segments)
            segments["atlas"] = self.atlas.to_bytes()
            self.universe.rebuild(segments, journal=journal)
            self._atlas_dirty = False
        else:
            for name, blob in new_segments.items():
                with open(os.path.join(self.storage_dir, name), "wb") as f:
                    f.write(blob)

    def _state_read(self, name: str) -> Optional[bytes]:
        """Reads a state segment from universe.loom."""
        if self.universe is not None:
            blob = self.universe.get_segment(name)
            if blob:
                return blob
        p = os.path.join(self.storage_dir, name)
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return f.read()
            except Exception:
                return None
        return None

    def _load_persistence_layers(self) -> None:
        """Reconstructs RAM ledger from compressed snapshot and journal replay."""
        journal_blob = b""
        if self.universe is not None:
            try:
                journal_blob = self.universe.read_journal()
            except Exception:
                journal_blob = b""
            for rec_type, payload in iter_journal_records(journal_blob):
                if rec_type != JREC_CRYSTAL:
                    continue
                try:
                    state_byte, num_shards, path_len = CRYSTAL_PREFIX.unpack_from(payload, 0)
                    pos = CRYSTAL_PREFIX.size
                    cpath = payload[pos:pos + path_len].decode("utf-8")
                    pos += path_len
                    centroid = np.frombuffer(payload[pos:], dtype=np.float32).copy()
                    if cpath not in self.atlas.crystals:
                        self.atlas.crystals[cpath] = {
                            "state": STATE_MAP_FROM_BYTE.get(state_byte, "warm"),
                            "centroid": centroid,
                            "num_shards": num_shards
                        }
                except Exception:
                    pass

        self.hash_to_shard: Dict[int, Tuple[str, str]] = {}
        for crystal_path in list(self.atlas.crystals.keys()):
            if os.path.exists(crystal_path):
                try:
                    if detect_version(crystal_path) == 2:
                        with LoomStore(crystal_path, writable=False) as ro:
                            ids = ro.get_all_ids()
                    else:
                        ids = [e["shard_id"] for e in LoomSubstrate.get_journal(crystal_path)]
                    for sid in ids:
                        self.hash_to_shard[hash_shard_id(sid)] = (sid, crystal_path)
                except Exception:
                    pass

        snapshot_blob = None
        if self.universe is not None:
            snapshot_blob = self.universe.get_segment("snapshot")
        if snapshot_blob is None:
            snapshot_path = os.path.join(self.storage_dir, "latent_state.snapshot")
            if os.path.exists(snapshot_path):
                try:
                    with open(snapshot_path, "rb") as f:
                        snapshot_blob = f.read()
                except Exception:
                    snapshot_blob = None
        if snapshot_blob:
            try:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(snapshot_blob)
                num_records = len(decompressed) // 16
                for i in range(num_records):
                    chunk = decompressed[i*16 : (i+1)*16]
                    h, act, t = struct.unpack("<QfI", chunk)
                    if h in self.hash_to_shard:
                        sid, crystal_path = self.hash_to_shard[h]
                        self.ram_ledger[sid] = {
                            "activation": act,
                            "latent_position": np.zeros(self.dimension, dtype=np.float32),
                            "velocity": np.zeros(self.dimension, dtype=np.float32),
                            "phase_angle": 0.0,
                            "hits": 0,
                            "last_recalled": float(t),
                            "crystal_path": crystal_path,
                            **default_cognitive_fields()
                        }
            except Exception:
                pass

        def _apply_txn(h: int, act: float, t: float) -> None:
            if h in self.hash_to_shard:
                sid, crystal_path = self.hash_to_shard[h]
                if sid not in self.ram_ledger:
                    self.ram_ledger[sid] = {
                        "activation": act,
                        "latent_position": np.zeros(self.dimension, dtype=np.float32),
                        "velocity": np.zeros(self.dimension, dtype=np.float32),
                        "phase_angle": 0.0,
                        "hits": 0,
                        "last_recalled": t,
                        "crystal_path": crystal_path,
                        **default_cognitive_fields()
                    }
                else:
                    self.ram_ledger[sid]["activation"] = act
                    self.ram_ledger[sid]["last_recalled"] = t

        if self.universe is not None:
            for rec_type, payload in iter_journal_records(journal_blob):
                if rec_type == JREC_TXN and len(payload) >= TXN_PAYLOAD.size:
                    try:
                        h, act, t = TXN_PAYLOAD.unpack(payload[:TXN_PAYLOAD.size])
                        _apply_txn(h, act, t)
                    except Exception:
                        pass
        else:
            journal_path = os.path.join(self.storage_dir, "cortex_journal.bin")
            if os.path.exists(journal_path):
                try:
                    with open(journal_path, "rb") as f:
                        journal_bytes = f.read()
                    for i in range(len(journal_bytes) // 32):
                        chunk = journal_bytes[i*32 : (i+1)*32]
                        magic, h, act, t, _ = struct.unpack("<4sQfd8s", chunk)
                        if magic == b"TXN\x00":
                            _apply_txn(h, act, t)
                except Exception:
                    pass

    def _append_journal_entry(self, shard_id: str, activation: float, timestamp: float) -> None:
        """Appends a 32-byte transaction entry."""
        self._append_journal_entries([(shard_id, activation, timestamp)])

    def _append_journal_entries(self, entries: List[Tuple[str, float, float]]) -> None:
        """Appends many ledger transactions with an atomic write."""
        if not entries:
            return
        try:
            if self.universe is not None:
                blob = bytearray()
                for shard_id, activation, timestamp in entries:
                    blob.extend(pack_txn_record(hash_shard_id(shard_id), activation, timestamp))
                self.universe.append_journal(bytes(blob))
            else:
                blob = bytearray()
                for shard_id, activation, timestamp in entries:
                    h = hash_shard_id(shard_id)
                    blob.extend(struct.pack("<4sQfd8s", b"TXN\x00", h, activation, timestamp, b"\x00" * 8))
                journal_path = os.path.join(self.storage_dir, "cortex_journal.bin")
                with open(journal_path, "ab") as f:
                    f.write(blob)
        except Exception:
            pass

    def _ledger_put_many(self, entries: Dict[str, Dict[str, Any]]) -> List[str]:
        """Places entries into ram_ledger under RAM_LEDGER_MAX cap."""
        cap = tuning_manager.get_int("RAM_LEDGER_MAX", 50000)
        accepted: List[str] = []
        new_items: List[Tuple[str, Dict[str, Any]]] = []
        for sid, entry in entries.items():
            if sid in self.ram_ledger:
                self.ram_ledger[sid] = entry
                accepted.append(sid)
            else:
                new_items.append((sid, entry))
        if not new_items:
            return accepted

        space = cap - len(self.ram_ledger)
        if space >= len(new_items):
            for sid, entry in new_items:
                self.ram_ledger[sid] = entry
                accepted.append(sid)
            return accepted

        new_items.sort(key=lambda kv: kv[1].get("activation", 0.0), reverse=True)
        head = max(0, space)
        for sid, entry in new_items[:head]:
            self.ram_ledger[sid] = entry
            accepted.append(sid)
        remaining = new_items[head:]
        if remaining:
            weakest = heapq.nsmallest(
                len(remaining), self.ram_ledger.items(),
                key=lambda kv: kv[1].get("activation", 0.0)
            )
            for (sid, entry), (wsid, wstate) in zip(remaining, weakest):
                if entry.get("activation", 0.0) > wstate.get("activation", 0.0):
                    del self.ram_ledger[wsid]
                    self.ram_ledger[sid] = entry
                    accepted.append(sid)
                else:
                    break
        return accepted

    def checkpoint(self) -> None:
        """Folds Active RAM ledger snapshot + atlas into universe.loom."""
        try:
            records = bytearray()
            for sid, entry in list(self.ram_ledger.items()):
                act = entry.get("activation", 0.0)
                if act > 0.0:
                    h = hash_shard_id(sid)
                    t = int(entry.get("last_recalled", 0.0))
                    records.extend(struct.pack("<QfI", h, act, t))

            if self.universe is not None:
                cctx = zstd.ZstdCompressor()
                segments = self._carryover_segments()
                segments["atlas"] = self.atlas.to_bytes()
                segments["snapshot"] = cctx.compress(bytes(records)) if records else b""
                self.universe.rebuild(segments, journal=b"")
                self._atlas_dirty = False
            else:
                if records:
                    cctx = zstd.ZstdCompressor()
                    compressed = cctx.compress(bytes(records))
                    snapshot_path = os.path.join(self.storage_dir, "latent_state.snapshot")
                    with open(snapshot_path, "wb") as f:
                        f.write(compressed)
                journal_path = os.path.join(self.storage_dir, "cortex_journal.bin")
                with open(journal_path, "wb") as f:
                    f.truncate(0)
        except Exception:
            pass

    def _build_living_state_blob(self) -> bytes:
        """Full-fidelity msgpack serialization of living cognitive & shard states."""
        import msgpack

        living: Dict[str, Dict[str, Any]] = {}
        for sid, entry in self.ram_ledger.items():
            living[sid] = {
                "activation": float(entry.get("activation", 0.0)),
                "latent_position": np.asarray(entry["latent_position"], dtype=np.float32).tolist(),
                "velocity": np.asarray(
                    entry.get("velocity", np.zeros(self.dimension, dtype=np.float32)), dtype=np.float32
                ).tolist(),
                "phase_angle": float(entry.get("phase_angle", 0.0)),
                "hits": int(entry.get("hits", 0)),
                "last_recalled": float(entry.get("last_recalled", 0.0)),
                "crystal_path": entry.get("crystal_path", ""),
                "crystal_idx": entry.get("crystal_idx"),
                "energy": float(entry.get("energy", 1.0)),
                "entropy": float(entry.get("entropy", 0.0)),
                "stability": float(entry.get("stability", 1.0)),
                "resonance": float(entry.get("resonance", 0.0)),
                "momentum": float(entry.get("momentum", 0.0)),
                "decay": float(entry.get("decay", 0.05)),
                "attention": float(entry.get("attention", 0.0)),
            }

        assemblies = [{
            "id": a.assembly_id,
            "dominant": a.dominant_concept,
            "activations": a.node_activations,
            "confidence": a.confidence,
            "stability": a.stability,
            "phase_coherence": a.phase_coherence,
            "basin_strength": a.meta_data.get("basin_strength", 0.0),
        } for a in self.active_assemblies]

        causal_edges = [{
            "from": u, "to": v,
            "weight": float(d.get("weight", 0.0)),
            "frequency": int(d.get("frequency", 1)),
        } for u, v, d in self.causal_graph.edges(data=True)]

        state = {
            "tick": self.cognitive_tick,
            "entropy": self.cognitive_entropy,
            "coherence": self.cognitive_coherence,
            "energy_budget": self.cognitive_energy_budget,
            "latent_field": self.latent_field.tolist() if self.latent_field is not None else None,
            "working_latent": self.working_latent.tolist() if self.working_latent is not None else None,
            "latent_velocity": self.latent_velocity.tolist() if self.latent_velocity is not None else None,
            "assemblies": assemblies,
            "causal_edges": causal_edges,
            "living_ledger": living,
            "superseded": self.superseded_by_crystal,
        }
        return msgpack.packb(state, use_bin_type=True, default=np_msgpack_default)

    def save_cortex_state(self) -> None:
        """Writes living state to living_state.bin segment in universe.loom."""
        try:
            segment = self._build_living_state_blob()
            self._commit_state_segments({"living_state.bin": segment})
        except Exception as e:
            print("Error in Weaver save_cortex_state:", e)
            raise e

    def autosave(self, force_full: bool = False) -> None:
        """Periodic durability point folding live RAM ledger into universe.loom."""
        self.consolidate_traces()
        try:
            now = time.time()
            do_full = force_full or (
                now - self._last_full_save_ts >= tuning_manager.get_int("VIZ_FULL_SAVE_INTERVAL_SEC", 180)
            )

            if self.universe is None:
                self.checkpoint()
                if do_full:
                    self.save_cortex_state()
                    self._last_full_save_ts = now
                return

            records = bytearray()
            for sid, entry in list(self.ram_ledger.items()):
                act = entry.get("activation", 0.0)
                if act > 0.0:
                    h = hash_shard_id(sid)
                    t = int(entry.get("last_recalled", 0.0))
                    records.extend(struct.pack("<QfI", h, act, t))

            segments = self._carryover_segments()
            segments["atlas"] = self.atlas.to_bytes()
            cctx = zstd.ZstdCompressor()
            segments["snapshot"] = cctx.compress(bytes(records)) if records else b""

            if do_full:
                segments["living_state.bin"] = self._build_living_state_blob()
                self._last_full_save_ts = now

            self.universe.rebuild(segments, journal=b"")
            self._atlas_dirty = False
        except Exception:
            pass

    def load_cortex_state(self) -> bool:
        """Restores full-fidelity cognitive + per-shard living state."""
        blob = self._state_read("living_state.bin")
        if blob is None:
            return False

        try:
            import msgpack
            state = msgpack.unpackb(blob, raw=False)

            self.cognitive_tick = state.get("tick", 0)
            self.cognitive_entropy = state.get("entropy", 0.5)
            self.cognitive_coherence = state.get("coherence", 0.5)
            self.cognitive_energy_budget = state.get("energy_budget", 1.0)
            self._last_cognitive_coherence = self.cognitive_coherence

            lf = state.get("latent_field")
            self.latent_field = np.array(lf, dtype=np.float32) if lf is not None else None
            wl = state.get("working_latent")
            self.working_latent = np.array(wl, dtype=np.float32) if wl is not None else None
            lv = state.get("latent_velocity")
            self.latent_velocity = np.array(lv, dtype=np.float32) if lv is not None else None

            self.active_assemblies = [
                CognitiveAssembly(
                    assembly_id=a["id"],
                    dominant_concept=a["dominant"],
                    node_activations=a["activations"],
                    confidence=a["confidence"],
                    stability=a["stability"],
                    phase_coherence=a.get("phase_coherence", self.cognitive_coherence),
                    meta_data={"basin_strength": a.get("basin_strength", 0.0)},
                )
                for a in state.get("assemblies", [])
            ]

            self.causal_graph.clear()
            for e in state.get("causal_edges", []):
                self.causal_graph.add_edge(
                    e["from"], e["to"], weight=e["weight"], frequency=e["frequency"], last_seen=time.time()
                )

            self.superseded_by_crystal = {
                cpath: dict(sidmap) for cpath, sidmap in state.get("superseded", {}).items()
            }
            self._superseded_idx_cache.clear()

            for sid, rec in state.get("living_ledger", {}).items():
                entry = self.ram_ledger.get(sid, {})
                entry.update({
                    "activation": rec["activation"],
                    "latent_position": np.array(rec["latent_position"], dtype=np.float32),
                    "velocity": np.array(rec["velocity"], dtype=np.float32),
                    "phase_angle": rec["phase_angle"],
                    "hits": rec["hits"],
                    "last_recalled": rec["last_recalled"],
                    "crystal_path": rec.get("crystal_path") or entry.get("crystal_path", ""),
                    "crystal_idx": rec.get("crystal_idx", entry.get("crystal_idx")),
                    "energy": rec["energy"],
                    "entropy": rec["entropy"],
                    "stability": rec["stability"],
                    "resonance": rec["resonance"],
                    "momentum": rec["momentum"],
                    "decay": rec["decay"],
                    "attention": rec["attention"],
                })
                self.ram_ledger[sid] = entry

            return True
        except Exception as e:
            print("Error loading Cortex state from Weaver:", e)
            return False
