"""Stage 1: edm4hep ROOT files -> ragged HDF5 shards (requires the key4hep stack).

Hit-to-particle assignment is the truth one: digitised hit -> (link with the
largest weight) -> SimTrackerHit -> MCParticle. Hits are ordered by the
SimTrackerHit time. The KF baseline track is matched to a particle by hit
majority (> 50 % of its hits).
"""

from collections import defaultdict

import numpy as np
import torch

from fcctrackfit.convert.cellid import BitFieldCoder
from fcctrackfit.data.shards import COL, HIT_COLUMNS, KIND_SI_3D, KIND_SI_PLANE, derived_columns, write_shard
from fcctrackfit.geometry.perigee import from_lcio, perigee_from_state, perigee_to_state

KINDS = {"si_plane": KIND_SI_PLANE, "si_3d": KIND_SI_3D}


def _cellid_coders(reader, trackers):
    coders = {}
    try:
        meta = reader.get("metadata")[0]
    except Exception:
        meta = None
    for t in trackers:
        # digitised hits share the encoding of their sim-hit collection, which is
        # where the CLD reconstruction stores it
        names = [t["hits"]] + ([t["cellid_from"]] if "cellid_from" in t else [])
        enc = None
        for name in names:
            if meta is None or enc:
                break
            try:
                enc = meta.get_parameter(f"{name}__CellIDEncoding")
            except Exception:
                enc = None
        if not enc:
            raise RuntimeError(f"no CellIDEncoding metadata for any of {names}")
        coders[t["hits"]] = BitFieldCoder(str(enc))
    return coders


def _is_secondary(sim):
    if hasattr(sim, "isProducedBySecondary"):
        return bool(sim.isProducedBySecondary())
    return bool(sim.getQuality() & (1 << 30))


def _hit_record(hit, kind, coder):
    pos = hit.getPosition()
    row = np.zeros(len(HIT_COLUMNS), np.float32)
    cell = np.array([hit.getCellID()], dtype=np.uint64)
    for name in ("system", "layer", "side"):
        row[COL[name]] = coder.get(cell, name)[0]
    row[COL["t"]] = hit.getTime()
    row[COL["kind"]] = kind
    if kind == KIND_SI_PLANE:
        u, v = hit.getU(), hit.getV()
        row[COL["u_theta"]], row[COL["u_phi"]] = u[0], u[1]
        row[COL["v_theta"]], row[COL["v_phi"]] = v[0], v[1]
        row[COL["du"]], row[COL["dv"]] = hit.getDu(), hit.getDv()
    else:
        cov = hit.getCovMatrix()
        row[COL["du"]] = np.sqrt(max(cov[0] + cov[2], 0.0))
        row[COL["dv"]] = np.sqrt(max(cov[5], 0.0))
    return (pos.x, pos.y, pos.z), row


def convert_event(event, cfg, coders, available):
    sel = cfg["selection"]
    bz = cfg["bz"]
    mcs = event.get(cfg["mc_particles"])
    mc_id = mcs.getID()

    # digitised hit -> best link
    best = {}  # (collection id, index) -> (weight, mc index, sim time, secondary, payload)
    for t in cfg["trackers"]:
        if t["kind"] == "drift":
            if cfg.get("use_drift"):
                raise NotImplementedError("drift-chamber hits are not supported yet (plan milestone 6)")
            continue
        if t["hits"] not in available or t["links"] not in available:
            continue
        hits = event.get(t["hits"])
        hits_id = hits.getID()
        coder = coders[t["hits"]]
        for link in event.get(t["links"]):
            src = link.getFrom().getObjectID()
            if src.collectionID != hits_id:
                continue
            sim = link.getTo()
            mc = sim.getParticle()
            if not mc.isAvailable() or mc.getObjectID().collectionID != mc_id:
                continue
            key = (hits_id, src.index)
            w = link.getWeight()
            if key not in best or w > best[key][0]:
                best[key] = (w, mc.getObjectID().index, sim.getTime(), _is_secondary(sim), (hits[src.index], t["kind"], coder))

    per_mc = defaultdict(list)
    hit_to_mc = {}
    for key, (w, mc_index, sim_time, secondary, payload) in best.items():
        hit_to_mc[key] = mc_index
        if secondary and sel.get("drop_secondary_hits", True):
            continue
        per_mc[mc_index].append((sim_time, payload))

    # KF baseline: track -> majority particle, AtIP state
    kf = {}
    if cfg.get("kf_tracks") in available:
        for track in event.get(cfg["kf_tracks"]):
            ids = [hit_to_mc.get((h.getObjectID().collectionID, h.getObjectID().index)) for h in track.getTrackerHits()]
            counts = defaultdict(int)
            for i in ids:
                if i is not None:
                    counts[i] += 1
            if not counts:
                continue
            mc_index, n = max(counts.items(), key=lambda kv: kv[1])
            if n <= 0.5 * len(ids) or (mc_index in kf and kf[mc_index][0] >= n):
                continue
            for ts in track.getTrackStates():
                if ts.location == ts.AtIP:
                    lcio = [ts.D0, ts.phi, ts.omega, ts.Z0, ts.tanLambda]
                    ref = ts.referencePoint
                    # CovMatrix6f has an unchecked operator[] and no length: never iterate it
                    cov = ts.covMatrix
                    kf[mc_index] = (n, lcio, (ref.x, ref.y, ref.z), [cov[i] for i in range(21)])
                    break

    tracks = []
    for mc_index, entries in per_mc.items():
        mc = mcs[mc_index]
        if mc.getPDG() not in sel["pdg"] or mc.getGeneratorStatus() not in sel["generator_status"]:
            continue
        mom, vtx = mc.getMomentum(), mc.getVertex()
        p = np.array([mom.x, mom.y, mom.z])
        if abs(p[2]) / np.linalg.norm(p) > sel["max_abs_cos_theta"]:
            continue
        if not sel["min_hits"] <= len(entries) <= sel["max_hits"]:
            continue
        entries.sort(key=lambda e: e[0])
        pos, rows = zip(*(_hit_record(hit, KINDS[kind], coder) for _, (hit, kind, coder) in entries))
        rows = np.stack(rows)
        f64 = torch.float64
        truth = perigee_from_state(torch.tensor([vtx.x, vtx.y, vtx.z], dtype=f64), torch.tensor(p, dtype=f64),
                                   torch.tensor(float(mc.getCharge()), dtype=f64), bz)  # fmt: skip
        kf_par, kf_cov, ok = np.full(5, np.nan), np.zeros(21, np.float32), False
        if mc_index in kf:
            _, lcio, ref, cov = kf[mc_index]
            kf_par = kf_perigee(lcio, ref, bz)
            kf_cov[: len(cov)] = cov[:21]
            ok = True
        tracks.append((np.array(pos), rows, truth.numpy(), kf_par, ok, kf_cov))
    return tracks


def kf_perigee(lcio, ref, bz):
    """LCIO AtIP state with arbitrary reference point -> perigee w.r.t. the z-axis."""
    par = from_lcio(torch.tensor(lcio, dtype=torch.float64), bz)
    pca, mom, q = perigee_to_state(par, bz)
    return perigee_from_state(pca + torch.tensor(ref, dtype=torch.float64), mom, q, bz).numpy()


def convert_file(in_path, cfg, out_path, max_events=None):
    from podio import root_io

    reader = root_io.Reader(in_path)
    coders = _cellid_coders(reader, [t for t in cfg["trackers"] if t["kind"] != "drift"])
    out = defaultdict(list)
    for i_event, event in enumerate(reader.get("events")):
        if max_events is not None and i_event >= max_events:
            break
        available = set(event.getAvailableCollections())
        for pos, rows, truth, kf_par, ok, kf_cov in convert_event(event, cfg, coders, available):
            out["pos"].append(pos)
            out["rows"].append(rows)
            out["truth"].append(truth)
            out["kf"].append(kf_par)
            out["kf_ok"].append(ok)
            out["kf_cov"].append(kf_cov)
            out["event"].append(i_event)
    if not out["pos"]:
        raise RuntimeError(f"{in_path}: no tracks selected")
    pos = np.concatenate(out["pos"])
    feat = np.concatenate(out["rows"])
    feat[:, COL["x"]], feat[:, COL["y"]], feat[:, COL["z"]] = pos.T
    feat[:, COL["r"]], feat[:, COL["phi"]], feat[:, COL["theta"]] = derived_columns(pos)
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in out["pos"]])])
    write_shard(out_path, hit_pos=pos, hit_feat=feat, offsets=offsets, truth=np.stack(out["truth"]),
                kf=np.stack(out["kf"]), kf_ok=np.array(out["kf_ok"]), event=np.array(out["event"]),
                detector=cfg["name"], bz=cfg["bz"], kf_cov=np.stack(out["kf_cov"]))  # fmt: skip
    return len(out["truth"]), int(np.sum(out["kf_ok"]))
