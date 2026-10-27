"""Write toy tracks as an edm4hep file following the Stage 1 sample contract.

Used to test the converter without a full ddsim production; also a compact
reference for which collections and relations the converter reads.
"""

import torch

from fcctrackfit.data.shards import COL
from fcctrackfit.geometry.perigee import perigee_to_state, to_lcio

ENCODING = "system:5,side:-2,layer:6,module:11,sensor:8"


def encode_cell(system, layer, side=0):
    return int(system) | ((int(side) & 0x3) << 5) | (int(layer) << 7)


def write_toy_edm4hep(path, shard, kf_from_truth=True, tracks_per_event=1):
    import edm4hep
    import podio
    from podio import root_io

    bz = shard["bz"]
    off = shard["offsets"]
    n_tracks = len(off) - 1
    writer = root_io.Writer(str(path))
    for first in range(0, n_tracks, tracks_per_event):
        mcs = edm4hep.MCParticleCollection()
        sims = edm4hep.SimTrackerHitCollection()
        hits = edm4hep.TrackerHitPlaneCollection()
        links = edm4hep.TrackerHitSimTrackerHitLinkCollection()
        tracks = edm4hep.TrackCollection()
        for t in range(first, min(first + tracks_per_event, n_tracks)):
            truth = torch.from_numpy(shard["truth"][t])
            pca, mom, q = perigee_to_state(truth, bz)
            mc = mcs.create()
            mc.setPDG(13 if q < 0 else -13)
            mc.setGeneratorStatus(1)
            mc.setCharge(float(q))
            mc.setMass(0.1056584)
            mc.setVertex(edm4hep.Vector3d(*pca.tolist()))
            mc.setMomentum(edm4hep.Vector3d(*mom.tolist()))
            track = tracks.create() if kf_from_truth else None
            # store hits in reverse order so the converter must sort by time
            for i in reversed(range(off[t], off[t + 1])):
                x, y, z = shard["hit_pos"][i]
                f = shard["hit_feat"][i]
                cell = encode_cell(f[COL["system"]], f[COL["layer"]])
                sim = sims.create()
                sim.setCellID(cell)
                sim.setTime(float(f[COL["t"]]))
                sim.setPosition(edm4hep.Vector3d(x, y, z))
                sim.setParticle(mc)
                hit = hits.create()
                hit.setCellID(cell)
                hit.setTime(float(f[COL["t"]]))
                hit.setPosition(edm4hep.Vector3d(x, y, z))
                hit.setDu(float(f[COL["du"]]))
                hit.setDv(float(f[COL["dv"]]))
                link = links.create()
                link.setFrom(hit)
                link.setTo(sim)
                link.setWeight(1.0)
                if track is not None:
                    track.addToTrackerHits(hit)
            if track is not None:
                lc = to_lcio(truth, bz).tolist()
                ts = edm4hep.TrackState()
                ts.location = edm4hep.TrackState.AtIP
                ts.D0, ts.phi, ts.omega, ts.Z0, ts.tanLambda = lc
                track.addToTrackStates(ts)
        frame = podio.Frame()
        frame.put(mcs, "MCParticles")
        frame.put(sims, "TrackerSimHits")
        frame.put(hits, "TrackerHits")
        frame.put(links, "TrackerHitRelations")
        frame.put(tracks, "KFTracks")
        writer.write_frame(frame, "events")
    meta = podio.Frame()
    meta.put_parameter("TrackerHits__CellIDEncoding", ENCODING)
    writer.write_frame(meta, "metadata")
    writer.finish()


TOY_EDM4HEP_CONFIG = {
    "name": "toy-edm4hep",
    "bz": 2.0,
    "mc_particles": "MCParticles",
    "trackers": [{"hits": "TrackerHits", "links": "TrackerHitRelations", "kind": "si_plane"}],
    "kf_tracks": "KFTracks",
    "selection": {"pdg": [13, -13], "generator_status": [1], "max_abs_cos_theta": 0.99, "min_hits": 6,
                  "max_hits": 30, "drop_secondary_hits": True},  # fmt: skip
}
