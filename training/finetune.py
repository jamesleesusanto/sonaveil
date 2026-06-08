"""
Fine-tunes htdemucs_6s into a 2-stem (piano / orchestra) separator.

Starts from the pretrained 6-source htdemucs_6s, keeps the entire network
except the two source-dependent output layers, and rebuilds those for 2
sources. The new head is warm-started so the model *begins* at a sensible
baseline:

    piano  output  <- pretrained "piano" output
    orch.  output  <- sum of the other 5 pretrained outputs

Because the final layers are linear, summing their weights is equivalent to
summing the stem predictions, i.e. orchestra == drums+bass+other+vocals+guitar
at init -- exactly the behaviour app.py currently fakes with ffmpeg. Training
then refines both stems jointly.

Resumable: re-running with the same --checkpoint-dir picks up the latest
checkpoint (model + optimizer + epoch). Use --init-from to warm-start a fresh
run from a previous run's weights (the "train again later on new data" path).

Usage (Colab T4 smoke test):
    python training/finetune.py --epochs 2 --segment 5 --batch-size 2

Usage (RunPod real run):
    python training/finetune.py --epochs 200 --segment 10 --batch-size 16 \
        --checkpoint-dir ckpt
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model

SOURCES = ["piano", "orchestra"]
BASE_MODEL = "htdemucs_6s"
CHANNELS = 2


# ---------------------------------------------------------------------------
# Audio I/O (ffmpeg, same as make_mixtures.py -- avoids torchaudio mp3 issues)
# ---------------------------------------------------------------------------
def load_audio(path: Path, sr: int) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(CHANNELS), "-ar", str(sr), "-"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, CHANNELS)


def ensure_cache(wav: Path, sr: int) -> Path:
    """Decode `wav` to a (T, C) float32 .npy once; reuse it forever after.

    Re-decoding every wav on every access (re-spawning ffmpeg each epoch)
    starves the GPU on large datasets. We decode once to a memory-mappable
    .npy so __getitem__ becomes a cheap slice. The cache sits next to the wav
    and is wiped whenever data/mixtures is regenerated.
    """
    npy = wav.with_suffix(".npy")
    if not npy.exists():
        np.save(npy, load_audio(wav, sr))
    return npy


# ---------------------------------------------------------------------------
# Dataset: each item is a random segment of one mixture track.
# ---------------------------------------------------------------------------
class MixtureDataset(Dataset):
    STEMS = ("mixture", "piano", "orchestra")

    def __init__(self, tracks: list[Path], segment: float, sr: int):
        self.tracks = tracks
        self.seg_len = int(segment * sr)
        self.sr = sr
        # Build the decode cache once, here in the main process (before workers
        # fork), so each .npy is written exactly once with no inter-worker race.
        for d in tracks:
            for stem in self.STEMS:
                ensure_cache(d / f"{stem}.wav", sr)

    def __len__(self) -> int:
        return len(self.tracks)

    def _load(self, wav: Path) -> np.ndarray:
        return np.load(wav.with_suffix(".npy"), mmap_mode="r")

    def _seg(self, audio: np.ndarray, start: int) -> torch.Tensor:
        # audio is (T, C); return (C, seg_len), padding if needed.
        chunk = audio[start:start + self.seg_len]
        if len(chunk) < self.seg_len:
            pad = np.zeros((self.seg_len - len(chunk), CHANNELS), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        return torch.from_numpy(chunk.T.copy())

    def __getitem__(self, idx: int):
        d = self.tracks[idx]
        mix = self._load(d / "mixture.wav")
        piano = self._load(d / "piano.wav")
        orch = self._load(d / "orchestra.wav")
        n = min(len(mix), len(piano), len(orch))
        start = 0 if n <= self.seg_len else np.random.randint(0, n - self.seg_len)
        mixture = self._seg(mix[:n], start)                 # (C, T)
        targets = torch.stack([self._seg(piano[:n], start),
                               self._seg(orch[:n], start)])  # (S, C, T)
        return mixture, targets


# ---------------------------------------------------------------------------
# Model: build the 2-source net from pretrained weights + warm-started head.
# ---------------------------------------------------------------------------
def build_model() -> HTDemucs:
    pre = get_model(BASE_MODEL).models[0]
    args, kwargs = pre._init_args_kwargs
    kwargs = dict(kwargs)
    kwargs["sources"] = SOURCES
    model = HTDemucs(*args, **kwargs)

    old = pre.state_dict()
    new = model.state_dict()
    for k in new:
        if k in old and old[k].shape == new[k].shape:
            new[k] = old[k].clone()

    old_sources = list(pre.sources)
    pi = old_sources.index("piano")
    non_piano = [i for i in range(len(old_sources)) if i != pi]

    # Output channels are grouped source-major; warm-start each branch's head.
    # spec branch: 4 ch/source (2 audio * 2 complex); time branch: 2 ch/source.
    for name, per in (("decoder.3.conv_tr", 4), ("tdecoder.3.conv_tr", 2)):
        w_old, b_old = old[name + ".weight"], old[name + ".bias"]
        w_new, b_new = new[name + ".weight"].clone(), new[name + ".bias"].clone()
        sl = lambda i: slice(i * per, (i + 1) * per)
        w_new[:, sl(0)] = w_old[:, sl(pi)]           # piano <- piano
        b_new[sl(0)] = b_old[sl(pi)]
        w_new[:, sl(1)] = sum(w_old[:, sl(i)] for i in non_piano)  # orch <- sum
        b_new[sl(1)] = sum(b_old[sl(i)] for i in non_piano)
        new[name + ".weight"], new[name + ".bias"] = w_new, b_new

    model.load_state_dict(new)
    return model


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for mixture, targets in loader:
            mixture, targets = mixture.to(device), targets.to(device)
            estimate = model(mixture)            # (B, S, C, T)
            loss = F.l1_loss(estimate, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * mixture.size(0)
            count += mixture.size(0)
    return total / max(count, 1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/mixtures"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("ckpt"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--segment", type=float, default=5.0, help="seconds per chunk")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--init-from", type=Path, default=None,
                   help="warm-start a fresh run from a previous checkpoint's weights")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sr = 44100
    tracks = sorted(d for d in args.data_root.glob("track_*")
                    if (d / "mixture.wav").exists())
    if not tracks:
        print(f"error: no mixtures in {args.data_root}; run make_mixtures.py first")
        return 1

    n_val = max(1, int(len(tracks) * args.val_fraction)) if len(tracks) > 1 else 0
    val_tracks, train_tracks = tracks[:n_val], tracks[n_val:]
    print(f"tracks: {len(tracks)} ({len(train_tracks)} train / {len(val_tracks)} val)"
          f"  device: {args.device}")

    train_dl = DataLoader(MixtureDataset(train_tracks, args.segment, sr),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=False)
    val_dl = (DataLoader(MixtureDataset(val_tracks, args.segment, sr),
                         batch_size=args.batch_size, num_workers=args.workers)
              if val_tracks else None)

    model = build_model().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = args.checkpoint_dir / "last.pt"
    best_ckpt = args.checkpoint_dir / "best.pt"
    start_epoch, best_val = 0, float("inf")

    if args.init_from and args.init_from.exists():
        # Fresh run, but start from prior weights (do NOT restore optimizer/epoch).
        ck = torch.load(args.init_from, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"warm-started weights from {args.init_from}")
    elif last_ckpt.exists():
        # Resume an interrupted run: restore weights, optimizer, and progress.
        ck = torch.load(last_ckpt, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from {last_ckpt} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_dl, args.device, optimizer)
        val_loss = run_epoch(model, val_dl, args.device) if val_dl else float("nan")
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  train L1 {train_loss:.4f}  "
              f"val L1 {val_loss:.4f}  ({dt:.0f}s)")

        ckpt = {"model": model.state_dict(), "optim": optimizer.state_dict(),
                "epoch": epoch, "sources": SOURCES, "best_val": best_val}
        torch.save(ckpt, last_ckpt)
        if val_dl and val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_ckpt)
            print(f"  -> new best (val L1 {best_val:.4f}) saved to {best_ckpt}")

    print(f"done. checkpoints in {args.checkpoint_dir}/  "
          f"(export with: python training/export.py --checkpoint {best_ckpt if val_dl else last_ckpt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
