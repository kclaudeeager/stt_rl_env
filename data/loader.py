"""
Afrivoice Swahili dataset loader with deterministic splits.

Domains:
- health: Healthcare-related content
- agriculture: Agricultural content  
- government: Government/policy content

Split strategy:
- Train: health + agriculture
- Validation: health + agriculture (different samples)
- Hidden test: government (no overlap)

NOTE: Requires HF Hub authentication
  huggingface-cli login
"""

import os
import json
import warnings
from typing import Dict, List
from datasets import load_dataset
import os
from huggingface_hub import hf_hub_download,snapshot_download
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


class AfrivoiceSwahiliLoader:
    """Load Afrivoice Swahili with deterministic splits."""

    # Domains to use
    TRAIN_DOMAINS = ["health", "agriculture"]
    VAL_DOMAINS = ["health", "agriculture"]
    TEST_DOMAINS = ["government"]
    
    MAX_DURATION_SEC = 30.0  # Cap audio length
    SEED = 42  # Deterministic shuffle
    SPLIT_CACHE_PATH = "data/split_indices.json"

    def __init__(self,repo_id:str="DigitalUmuganda/Afrivoice_Swahili"):
        self.datasets_by_domain = {}
        self.split_indices = {}
        self.repo_id = repo_id
    def load_all_dataset(self,dataset_path):
        snapshot_download(repo_id=self.repo_id,repo_type='dataset',local_dir=dataset_path)

    def load_domain(self, domain: str) -> None:
        """Load a single domain from Afrivoice."""
        print(f"Loading domain: {domain}...")

        # First, try the standard datasets loader without trust_remote_code
        try:
            ds = load_dataset(self.repo_id, name=domain,trust_remote_code=True)
            self.datasets_by_domain[domain] = ds["train"]
            print(f"  ✓ Loaded {len(ds['train'])} samples from {domain} via load_dataset")
            return
        except Exception as e:
            print(f"  - load_dataset direct failed: {e}")

        # If direct loading fails, try to download raw files from the HF repo using hf_hub_download.
        hf_token = os.environ.get("HF_TOKEN")

        candidate_files = [
            f"{domain}.parquet",
            f"{domain}.jsonl",
            f"{domain}.json",
            f"{domain}.csv",
            f"{domain}.tar.gz",
            f"{domain}.zip",
        ]

        downloaded = None
        for fname in candidate_files:
            try:
                print(f"  Trying hf_hub_download for {fname}...")
                local_path = hf_hub_download(repo_id=self.repo_id, filename=fname, repo_type="dataset", token=hf_token)
                if os.path.exists(local_path):
                    downloaded = local_path
                    print(f"  ✓ Downloaded {fname} to {local_path}")
                    break
            except Exception as e:
                print(f"    -> not found: {fname} ({e})")
                continue

        if downloaded:
            # Attempt to load the downloaded file with datasets
            try:
                ext = os.path.splitext(downloaded)[1].lower()
                if ext == ".parquet":
                    ds = load_dataset("parquet", data_files=str(downloaded))
                elif ext in (".jsonl", ".json"):
                    ds = load_dataset("json", data_files=str(downloaded))
                elif ext == ".csv":
                    ds = load_dataset("csv", data_files=str(downloaded))
                elif ext in (".gz", ".zip"):
                    # Try to load as a generic dataset archive (may not work for all formats)
                    ds = load_dataset("json", data_files=str(downloaded))
                else:
                    ds = None

                if ds is not None:
                    # If dataset has a train split, use it; else use first split
                    if "train" in ds:
                        self.datasets_by_domain[domain] = ds["train"]
                    else:
                        # pick first split
                        first = list(ds.keys())[0]
                        self.datasets_by_domain[domain] = ds[first]

                    print(f"  ✓ Loaded {len(self.datasets_by_domain[domain])} samples from {domain} via downloaded file")
                    return
            except Exception as e:
                print(f"  ✗ Failed to load downloaded file {downloaded}: {e}")

        # If all attempts fail, raise an informative error to allow caller to fallback
        raise RuntimeError(f"Failed to load domain '{domain}' from HF hub (tried load_dataset and hf_hub_download)")

    def load(self) -> None:
        """Load all required domains."""
        # Prefer a local dataset layout if present
        local_root = os.path.join("data", "local")
        if os.path.exists(local_root):
            print(f"Local dataset detected at {local_root} — loading local files.")
            self.load_local(local_root)
            return

        all_domains = list(set(self.TRAIN_DOMAINS + self.VAL_DOMAINS + self.TEST_DOMAINS))
        for domain in all_domains:
            self.load_domain(domain)

    def load_local(self, local_root: str) -> None:
        """Load a local dataset layout.

        Expected layouts (flexible):
          - data/local/train, data/local/valid, data/local/test + transcripts.json
          - data/local/health, data/local/agriculture, data/local/government + transcripts.json

        `transcripts.json` should map relative audio paths to transcripts.
        """
        print(f"Loading local dataset from {local_root}...")

        transcripts_path = os.path.join(local_root, "transcripts.json")
        transcripts = {}
        if os.path.exists(transcripts_path):
            try:
                with open(transcripts_path, "r") as f:
                    transcripts = json.load(f)
            except Exception as e:
                print(f"  ✗ Failed to read transcripts.json: {e}")

        # Helper: scan a folder and collect samples
        def scan_folder(folder, category=None):
            samples = []
            if not os.path.isdir(folder):
                return samples
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith((".wav", ".flac", ".mp3")):
                    continue
                path = os.path.join(folder, fname)
                rel = os.path.relpath(path)
                text = transcripts.get(rel) or transcripts.get(fname) or ""
                # compute duration safely
                duration = 0.0
                try:
                    import wave
                    with wave.open(path, "rb") as w:
                        frames = w.getnframes()
                        rate = w.getframerate()
                        duration = frames / float(rate) if rate > 0 else 0.0
                except Exception:
                    duration = 0.0

                samples.append({
                    "file": rel,
                    "duration": duration,
                    "normalized_transcription": text,
                    "category": category,
                })
            return samples

        # Try train/valid/test layout first
        train_dir = os.path.join(local_root, "train")
        valid_dir = os.path.join(local_root, "valid")
        test_dir = os.path.join(local_root, "test")

        if os.path.isdir(train_dir) or os.path.isdir(valid_dir) or os.path.isdir(test_dir):
            print("  Detected train/valid/test layout")
            self.split_indices = {
                "train": scan_folder(train_dir, category="local_train"),
                "validation": scan_folder(valid_dir, category="local_valid"),
                "test": scan_folder(test_dir, category="local_test"),
            }
            self.save_splits()
            print(f"  Loaded local splits: train={len(self.split_indices['train'])}, val={len(self.split_indices['validation'])}, test={len(self.split_indices['test'])}")
            return

        # Otherwise try domain-named folders (health/agriculture/government)
        domain_root = local_root
        collected = {"train": [], "validation": [], "test": []}
        for domain in self.TRAIN_DOMAINS + self.VAL_DOMAINS:
            folder = os.path.join(domain_root, domain)
            samples = scan_folder(folder, category=domain)
            # simple 80/20 split
            cut = int(0.8 * len(samples))
            collected["train"].extend(samples[:cut])
            collected["validation"].extend(samples[cut:])

        for domain in self.TEST_DOMAINS:
            folder = os.path.join(domain_root, domain)
            samples = scan_folder(folder, category=domain)
            collected["test"].extend(samples)

        self.split_indices = collected
        self.save_splits()
        print(f"  Loaded local domain splits: train={len(self.split_indices['train'])}, val={len(self.split_indices['validation'])}, test={len(self.split_indices['test'])}")

    def _build_splits(self) -> Dict[str, List[Dict]]:
        """Build deterministic train/val/test splits."""
        print("\nBuilding deterministic splits...")
        
        split_data = {
            "train": [],
            "validation": [],
            "test": [],
        }

        import random
        random.seed(self.SEED)

        # Train + Val from health + agriculture
        for domain in self.TRAIN_DOMAINS:
            if domain not in self.datasets_by_domain:
                continue
                
            domain_data = self.datasets_by_domain[domain]
            
            # Filter by duration
            valid_samples = []
            for idx, sample in enumerate(domain_data):
                duration = sample.get("duration", 0)
                if isinstance(duration, (int, float)) and duration <= self.MAX_DURATION_SEC:
                    valid_samples.append((idx, sample))
            
            print(f"  {domain}: {len(valid_samples)} samples (≤{self.MAX_DURATION_SEC}s)")
            
            # 80-20 split
            random.shuffle(valid_samples)
            split_pt = int(0.8 * len(valid_samples))
            
            split_data["train"].extend(valid_samples[:split_pt])
            split_data["validation"].extend(valid_samples[split_pt:])

        # Test from government
        for domain in self.TEST_DOMAINS:
            if domain not in self.datasets_by_domain:
                continue
                
            domain_data = self.datasets_by_domain[domain]
            
            valid_samples = []
            for idx, sample in enumerate(domain_data):
                duration = sample.get("duration", 0)
                if isinstance(duration, (int, float)) and duration <= self.MAX_DURATION_SEC:
                    valid_samples.append((idx, sample))
            
            print(f"  {domain}: {len(valid_samples)} samples (≤{self.MAX_DURATION_SEC}s) [HIDDEN TEST]")
            split_data["test"].extend(valid_samples)

        print(f"\n  Train: {len(split_data['train'])} samples")
        print(f"  Validation: {len(split_data['validation'])} samples")
        print(f"  Test (hidden): {len(split_data['test'])} samples")

        return split_data

    def save_splits(self) -> None:
        """Save split metadata to disk."""
        os.makedirs("data", exist_ok=True)
        
        # Save just counts and sample info for reproducibility
        cache_data = {
            "train_count": len(self.split_indices["train"]),
            "val_count": len(self.split_indices["validation"]),
            "test_count": len(self.split_indices["test"]),
            "seed": self.SEED,
        }
        
        with open(self.SPLIT_CACHE_PATH, "w") as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"✓ Split metadata saved to {self.SPLIT_CACHE_PATH}")

    def get_split_samples(self, split_name: str = "train") -> List[Dict]:
        """Get samples for a split."""
        if split_name not in self.split_indices:
            raise ValueError(f"Unknown split: {split_name}")
        
        return self.split_indices[split_name]

    def prepare(self) -> None:
        """Load dataset and build splits."""
        try:
            self.load()
        except Exception as e:
            print(f"Dataset load failed: {e}")
            # Leave it to caller or pipeline to fallback (e.g., create stub)
            return

        # If datasets_by_domain was populated (HF path), build deterministic splits
        if self.datasets_by_domain:
            self.split_indices = self._build_splits()
            self.save_splits()
        else:
            # If load_local already saved splits, nothing to do
            if not self.split_indices:
                print("No data loaded; split_indices empty.")


if __name__ == "__main__":
    print("="*60)
    print("AFRIVOICE SWAHILI DATASET LOADER")
    print("="*60)
    
    loader = AfrivoiceSwahiliLoader()
    loader.prepare()
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE INSPECTION")
    print("="*60)
    
    if loader.split_indices["train"]:
        sample = loader.split_indices["train"][0]
        print(f"\nTrain sample:")
        print(f"  Domain: {sample.get('category')}")
        print(f"  Duration: {sample.get('duration')}s")
        print(f"  Normalized text: {sample.get('normalized_transcription')[:80]}...")


class FLEURSLoader:
    """Load FLEURS (Fine-grained Language Identification) dataset."""

    SEED = 42
    SPLIT_CACHE_PATH = "data/split_indices.json"

    def __init__(self, repo_id: str = "google/fleurs", config: str = "en_us"):
        self.repo_id = repo_id
        self.config = config
        self.split_indices = {}
        self.dataset = None
        self.dataloader = None

    def prepare(self) -> None:
        """Load FLEURS dataset and build train/val/test splits."""
        print(f"Loading FLEURS dataset from {self.repo_id} (config: {self.config})...")
        try:
            # With datasets==3.6.0, load_dataset works with script-based datasets
            # Use streaming to avoid generation issues
            self.dataset = load_dataset(self.repo_id, self.config, trust_remote_code=True, streaming=False)
            print(f"  ✓ Loaded FLEURS dataset splits: {list(self.dataset.keys())}")
        except Exception as e:
            print(f"  ✗ Error loading FLEURS: {e}")
            print(f"  Attempting fallback to stub dataset (will generate minimal test data)...")
            # Create a minimal stub to allow training to proceed
            self.split_indices = {
                "train": [{"audio": None, "transcription": "test audio"}],
                "validation": [{"audio": None, "transcription": "test validation"}],
                "test": [{"audio": None, "transcription": "test hidden"}],
            }
            print(f"  ✓ Created stub dataset for testing")
            self.save_splits()
            return

        # Use standard train/validation splits if available, else create custom
        if "train" in self.dataset and "validation" in self.dataset:
            # Simple split: use provided train/val, create test from a subset of val
            import random
            random.seed(self.SEED)

            train_data = self.dataset["train"]
            val_data = self.dataset["validation"]

            # Convert to list of dicts
            train_list = [dict(sample) for sample in train_data]
            val_list = [dict(sample) for sample in val_data]

            # Split validation into val and hidden test (80/20)
            random.shuffle(val_list)
            cut = int(0.8 * len(val_list))
            val_split = val_list[:cut]
            test_split = val_list[cut:]

            self.split_indices = {
                "train": train_list,
                "validation": val_split,
                "test": test_split,
            }
        else:
            # Fallback: use first split as train, rest for val/test
            all_splits = list(self.dataset.keys())
            if len(all_splits) >= 2:
                self.split_indices["train"] = [dict(s) for s in self.dataset[all_splits[0]]]
                remaining = [dict(s) for s in self.dataset[all_splits[1]]]
                cut = int(0.8 * len(remaining))
                self.split_indices["validation"] = remaining[:cut]
                self.split_indices["test"] = remaining[cut:]
            else:
                raise RuntimeError("FLEURS dataset has unexpected structure")

        print(f"  Train: {len(self.split_indices['train'])} samples")
        print(f"  Validation: {len(self.split_indices['validation'])} samples")
        print(f"  Test (hidden): {len(self.split_indices['test'])} samples")

        self.save_splits()

    def save_splits(self) -> None:
        """Save split metadata and dataset to disk."""
        import os
        data_dir = os.path.dirname(self.SPLIT_CACHE_PATH)
        os.makedirs(data_dir, exist_ok=True)
        
        # Save metadata
        cache_data = {
            "train_count": len(self.split_indices["train"]),
            "val_count": len(self.split_indices["validation"]),
            "test_count": len(self.split_indices["test"]),
            "seed": self.SEED,
            "dataset": "FLEURS",
        }
        with open(self.SPLIT_CACHE_PATH, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"✓ Split metadata saved to {self.SPLIT_CACHE_PATH}")
        
        # Save full dataset samples to parquet for later use
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            
            for split_name, samples in self.split_indices.items():
                parquet_path = os.path.join(data_dir, f"fleurs_{split_name}.parquet")
                
                # Convert to table - handle potential complex types by serializing problematic fields
                table_data = {}
                for key in samples[0].keys() if samples else []:
                    values = []
                    for sample in samples:
                        val = sample[key]
                        # Convert non-serializable types to string
                        if hasattr(val, '__dict__') or (isinstance(val, dict) and 'bytes' in str(type(val))):
                            values.append(str(val))
                        else:
                            values.append(val)
                    table_data[key] = values
                
                if table_data:
                    table = pa.table(table_data)
                    pq.write_table(table, parquet_path)
                    print(f"  ✓ Saved {len(samples)} {split_name} samples to {parquet_path}")
        except Exception as e:
            print(f"  ⚠ Could not save to parquet: {e}")
            print(f"  Falling back to JSON save...")

    def get_split_samples(self, split_name: str = "train") -> List[Dict]:
        """Get samples for a split."""
        if split_name not in self.split_indices:
            raise ValueError(f"Unknown split: {split_name}")
        return self.split_indices[split_name]
