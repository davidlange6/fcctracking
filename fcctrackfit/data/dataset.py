"""Streaming packed-batch loader over ragged HDF5 shards."""

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from fcctrackfit.data.shards import read_shard


def gather_tracks(shard, idx):
    """Pack the tracks idx (in that order) into a batch dict of torch tensors."""
    off = shard["offsets"]
    starts = off[idx]
    lengths = off[idx + 1] - starts
    cu = np.concatenate([[0], np.cumsum(lengths)])
    hit_idx = np.repeat(starts - cu[:-1], lengths) + np.arange(cu[-1])
    seed_idx = shard["seed_idx"][idx] - starts[:, None] + cu[:-1, None]
    return {
        "pos": torch.from_numpy(shard["hit_pos"][hit_idx]),
        "feat": torch.from_numpy(shard["hit_feat"][hit_idx]),
        "cu_seqlens": torch.from_numpy(cu),
        "seed_idx": torch.from_numpy(seed_idx),
        "truth": torch.from_numpy(shard["truth"][idx]),
        "kf": torch.from_numpy(shard["kf"][idx]),
        "kf_ok": torch.from_numpy(shard["kf_ok"][idx]),
    }


def usable_tracks(shard):
    return np.nonzero((shard["seed_idx"] >= 0).all(1))[0]


class PackedTrackStream(IterableDataset):
    """Yields packed batches; shards are split across DDP ranks and loader workers.

    Each epoch shuffles the shard order and the tracks within each shard. The
    last incomplete batch of a shard is dropped when shuffle is on.
    """

    def __init__(self, paths, batch_size, shuffle=True, rank=0, world_size=1, seed=0, max_tracks=None):
        self.paths = list(paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.max_tracks = max_tracks
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(len(self.paths)) if self.shuffle else np.arange(len(self.paths))
        info = get_worker_info()
        n_workers, worker = (info.num_workers, info.id) if info else (1, 0)
        mine = order[self.rank * n_workers + worker :: self.world_size * n_workers]
        if len(mine) == 0:
            raise RuntimeError("fewer shards than ranks x workers")
        seen = 0
        for i in mine:
            shard = read_shard(self.paths[i])
            idx = usable_tracks(shard)
            if self.shuffle:
                idx = rng.permutation(idx)
            stop = len(idx) - (len(idx) % self.batch_size if self.shuffle else 0)
            for b in range(0, stop, self.batch_size):
                batch_idx = idx[b : b + self.batch_size]
                if self.max_tracks is not None:
                    if seen >= self.max_tracks:
                        return
                    batch_idx = batch_idx[: self.max_tracks - seen]
                seen += len(batch_idx)
                yield gather_tracks(shard, batch_idx)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
