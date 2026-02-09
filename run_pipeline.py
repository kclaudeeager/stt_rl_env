import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_CACHE = DATA_DIR / "split_indices.json"


def create_local_stub_dataset():
    """Create a tiny local dataset (WAVs + transcripts) for offline testing."""
    import wave
    import struct
    import math

    print("No HF_TOKEN or download failed — creating local stub dataset...")
    audio_dir = DATA_DIR / "audio_stub"
    audio_dir.mkdir(parents=True, exist_ok=True)

    transcripts = {}

    samplerate = 16000
    duration_s = 0.6
    n_samples = int(samplerate * duration_s)

    def write_sine(path, freq=440.0, amp=0.3):
        with wave.open(str(path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(samplerate)
            for i in range(n_samples):
                t = i / samplerate
                val = int(amp * 32767.0 * math.sin(2 * math.pi * freq * t))
                w.writeframes(struct.pack('<h', val))

    samples = [
        ("stub_1.wav", "Habari, hii ni taarifa ya mtihani."),
        ("stub_2.wav", "Hali ya afya inahitaji tahadhari."),
        ("stub_3.wav", "Kilimo ni muhimu kwa uchumi wa jamii."),
    ]

    for i, (fname, text) in enumerate(samples, 1):
        p = audio_dir / fname
        write_sine(p, freq=300 + i * 50)
        transcripts[str(p.relative_to(PROJECT_ROOT))] = text

    # Save transcripts
    tr_path = DATA_DIR / "transcripts.json"
    with open(tr_path, "w") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)

    # Save minimal split metadata so other tools know counts
    cache = {
        "train_count": 2,
        "val_count": 1,
        "test_count": 0,
        "seed": 42,
        "note": "local stub dataset created by run_pipeline.py",
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_CACHE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Created {len(samples)} stub audio files under {audio_dir}")
    print(f"Wrote transcripts to {tr_path}")
    print(f"Wrote split metadata to {SPLIT_CACHE}")


if __name__ == "__main__":
    print("=== RUN PIPELINE ===")

    hf_token = os.environ.get("HF_TOKEN")

    loader_ok = False
    
    # Priority 1: Try FLEURS dataset (has benchmark results)
    if not loader_ok:
        try:
            from data.loader import FLEURSLoader
            loader = FLEURSLoader(repo_id="google/fleurs", config="sw_ke")
            print("Attempting to load FLEURS dataset (google/fleurs, sw_ke config)...")
            loader.prepare()
            loader_ok = True
            print("✓ FLEURS dataset downloaded and splits prepared.")
        except Exception as e:
            print(f"✗ FLEURS download failed: {e}")
            loader_ok = False
    
    # Priority 2: Try local dataset (fallback)
    if not loader_ok:
        try:
            from data.loader import AfrivoiceSwahiliLoader
            local_root = "data/local"
            if os.path.exists(local_root) and os.path.isdir(local_root):
                # Check if it has domain folders
                has_domains = any(os.path.isdir(os.path.join(local_root, d)) for d in ["health", "agriculture", "government"])
                if has_domains:
                    loader = AfrivoiceSwahiliLoader()
                    print(f"Local domain dataset detected at {local_root}...")
                    loader.prepare()
                    loader_ok = True
                    print("Local dataset splits prepared.")
        except Exception as e:
            print(f"Local dataset load failed: {e}")
            loader_ok = False
    
    # Priority 3: Try Afrivoice
    if not loader_ok:
        try:
            from data.loader import AfrivoiceSwahiliLoader
            if hf_token:
                loader = AfrivoiceSwahiliLoader()
                print("HF_TOKEN detected — attempting to download Afrivoice splits (this may take a while)...")
                loader.prepare()
                loader_ok = True
                print("Dataset downloaded and splits prepared.")
        except Exception as e:
            print(f"Dataset download failed: {e}")
            loader_ok = False

    # Priority 4: Create stub dataset for testing
    if not loader_ok:
        create_local_stub_dataset()

    # Run evaluation (uses fake eval if real not implemented)
    try:
        from training.eval import evaluate
        print("Running evaluation (smoke test)...")
        wer = evaluate(None, None)
        print(f"Evaluation WER (smoke): {wer}")
    except Exception as e:
        print(f"Evaluation step failed: {e}")

    # Run a real episode with Groq agent
    print("Launching episode (real Groq agent)...")
    proc = subprocess.run(["uv", "run", "python", "run_episode.py"], capture_output=True, text=True)
    print("--- run_episode output ---")
    print(proc.stdout)
    if proc.returncode != 0:
        print("run_episode failed:")
        print(proc.stderr)

    # Check trajectory file
    traj_path = PROJECT_ROOT / "trajectory.json"
    if traj_path.exists():
        print(f"Trajectory written to {traj_path} (size={traj_path.stat().st_size} bytes)")
    else:
        print("No trajectory.json found — something went wrong in the episode run.")

    print("=== PIPELINE COMPLETE ===")
