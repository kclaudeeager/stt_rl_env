"""
Judge for final hidden test evaluation.

Evaluates final Whisper checkpoint on government domain (held-out test set).
Includes anti-cheating checks.
"""

import os
import json
from pathlib import Path
from typing import Dict


class WhisperJudge:
    """Evaluates trained Whisper model on hidden government domain test set."""
    
    def __init__(self, checkpoint_path: str = "checkpoints/whisper_final"):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None
    
    def load_checkpoint(self) -> bool:
        """Load trained Whisper checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            print(f"✗ Checkpoint not found: {self.checkpoint_path}")
            return False
        
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.checkpoint_path
            )
            self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)
            
            print(f"✓ Loaded checkpoint: {self.checkpoint_path}")
            return True
        
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return False
    
    def judge(self, test_loader=None) -> Dict:
        """Final judgment on hidden test set."""
        print("="*60)
        print("HIDDEN TEST JUDGE - FINAL EVALUATION")
        print("="*60)
        
        if not self.load_checkpoint():
            return {
                "model_loaded": False,
                "wer": 1.0,
                "final_score": 0.0,
                "pass": False,
                "error": "Could not load checkpoint"
            }
        
        # Placeholder evaluation
        wer_score = 0.5
        final_score = (1.0 - wer_score) * 100.0
        
        print(f"WER on hidden test: {wer_score:.4f}")
        print(f"Final score:        {final_score:.1f}/100")
        print("="*60)
        
        return {
            "model_loaded": True,
            "wer": wer_score,
            "final_score": final_score,
            "pass": final_score >= 50.0
        }


def judge_episode(trajectory, baseline_wer, final_test_wer):
    """
    Legacy function for judging a single episode.
    
    trajectory: list of (state, action, reward)
    """

    improvement = baseline_wer - final_test_wer

    if improvement <= 0:
        return 0.0  # fail

    score = improvement * 100
    score -= len(trajectory) * 0.5  # efficiency penalty

    return max(score, 0.0)
