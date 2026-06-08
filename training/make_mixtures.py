"""
Builds synthetic training mixtures from solo piano and orchestra clips.

Pairs a random piano clip with a random orchestra clip, applies a random
time offset and random gain to each, then sums them into a mixture. The two
sources are kept as ground-truth stems, giving the supervised triplet a
2-stem separator trains on:

    data/mixtures/track_0001/
    ├── mixture.wav     (piano + orchestra, summed)
    ├── piano.wav       (ground-truth piano)
    └── orchestra.wav   (ground-truth orchestra)

Audio I/O goes through ffmpeg (decode mp3 -> raw float32 PCM, encode raw ->
wav) rather than torchaudio/soundfile, so it runs anywhere ffmpeg exists
(local, Colab, RunPod) without the torchaudio mp3 backend issues.

Usage:
    python training/make_mixtures.py --num 20                 # smoke test
    python training/make_mixtures.py --num 2000 --sr 44100    # real run
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

CHANNELS = 2  # demucs trains on stereo


def decode(path: Path, sr: int) -> np.ndarray:
    """Decode an audio file to a (samples, CHANNELS) float32 array via ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(CHANNELS), "-ar", str(sr),
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio.reshape(-1, CHANNELS)


def encode(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write a (samples, CHANNELS) float32 array to a wav file via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(CHANNELS), "-ar", str(sr),
            "-i", "-",
            str(path),
        ],
        check=True,
        input=audio.astype(np.float32).tobytes(),
        stderr=subprocess.PIPE,
    )


def random_segment(audio: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    """Extract `length` samples starting at a random offset (the time offset)."""
    if len(audio) <= length:
        # Shorter than target: pad with silence at a random position.
        out = np.zeros((length, CHANNELS), dtype=np.float32)
        start = rng.randint(0, length - len(audio))
        out[start:start + len(audio)] = audio
        return out
    start = rng.randint(0, len(audio) - length)
    return audio[start:start + length]


def normalize_rms(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale so the signal's RMS sits at `target_dbfs` (dB relative to full scale).

    Equalizes the perceived loudness of sources that were mastered at different
    levels, so the later random gain creates balanced variation rather than
    stacking on top of arbitrary baseline differences. Silent clips are left
    untouched to avoid divide-by-zero.
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-9:
        return audio
    target_rms = 10.0 ** (target_dbfs / 20.0)
    return audio * (target_rms / rms)


def random_gain(audio: np.ndarray, db: float, rng: random.Random) -> np.ndarray:
    """Scale by a random gain uniformly within +/- `db` decibels."""
    gain_db = rng.uniform(-db, db)
    return audio * (10.0 ** (gain_db / 20.0))


def list_audio(directory: Path) -> list[Path]:
    files = sorted(
        p for p in directory.glob("*")
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}
    )
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piano-dir", type=Path, default=Path("data/piano"))
    parser.add_argument("--orchestra-dir", type=Path, default=Path("data/orchestra"))
    parser.add_argument("--out", type=Path, default=Path("data/mixtures"))
    parser.add_argument("--num", type=int, default=20,
                        help="number of mixtures to generate")
    parser.add_argument("--clip-seconds", type=float, default=30.0,
                        help="length of each mixture (clamped to source length)")
    parser.add_argument("--sr", type=int, default=44100, help="sample rate")
    parser.add_argument("--target-rms", type=float, default=-20.0,
                        help="dBFS RMS each source is normalized to before mixing")
    parser.add_argument("--gain-db", type=float, default=6.0,
                        help="max +/- random gain applied per source")
    parser.add_argument("--seed", type=int, default=0, help="for reproducibility")
    args = parser.parse_args()

    piano_files = list_audio(args.piano_dir)
    orchestra_files = list_audio(args.orchestra_dir)

    if not piano_files:
        print(f"error: no audio in {args.piano_dir}", file=sys.stderr)
        return 1
    if not orchestra_files:
        print(f"error: no audio in {args.orchestra_dir}", file=sys.stderr)
        return 1

    print(f"piano clips: {len(piano_files)}  orchestra clips: {len(orchestra_files)}")
    print(f"generating {args.num} mixtures -> {args.out}")

    rng = random.Random(args.seed)
    length = int(args.clip_seconds * args.sr)
    args.out.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.num + 1):
        piano_path = rng.choice(piano_files)
        orchestra_path = rng.choice(orchestra_files)

        piano = decode(piano_path, args.sr)
        orchestra = decode(orchestra_path, args.sr)

        # Clamp target length so all three stems share the same length even if
        # a picked source is shorter than --clip-seconds.
        seg_len = min(length, len(piano), len(orchestra))
        if seg_len < args.sr:  # less than 1s of usable audio, skip
            print(f"  track_{i:04d}: sources too short, skipping")
            continue

        # Equalize loudness to a common baseline, then add random balance variation.
        piano = normalize_rms(random_segment(piano, seg_len, rng), args.target_rms)
        orchestra = normalize_rms(random_segment(orchestra, seg_len, rng), args.target_rms)
        piano = random_gain(piano, args.gain_db, rng)
        orchestra = random_gain(orchestra, args.gain_db, rng)
        mixture = piano + orchestra

        # Guard against clipping: scale the whole triplet down if the mixture
        # peaks above 1.0, preserving the piano + orchestra == mixture identity.
        peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
        if peak > 1.0:
            scale = 0.99 / peak
            piano *= scale
            orchestra *= scale
            mixture *= scale

        track_dir = args.out / f"track_{i:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        encode(track_dir / "mixture.wav", mixture, args.sr)
        encode(track_dir / "piano.wav", piano, args.sr)
        encode(track_dir / "orchestra.wav", orchestra, args.sr)
        print(f"  track_{i:04d}: {seg_len / args.sr:.1f}s "
              f"({piano_path.name} + {orchestra_path.name})")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
