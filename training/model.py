"""
Whisper model loader and utilities using transformers pipeline.
"""

import torch
from transformers import pipeline


class WhisperModel:
    """Wrapper around Whisper model using transformers pipeline for speech-to-text."""
    
    def __init__(self, model_id="Dafisns/whisper-turbo-multilingual-fleurs", device=None, language="english"):
        """
        Load Whisper model via pipeline.
        
        Args:
            model_id: HuggingFace model ID (tiny, small, medium, large, or FLEURS variant)
            device: torch device (cuda/cpu)
            language: language code for multilingual models (e.g., 'english', 'swahili', 'indonesian')
        """
        self.model_id = model_id
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading {model_id} on {self.device}...")
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=self.device,
            dtype=dtype,
        )
        
        self.model = self.pipe.model
        self.processor = self.pipe.processor
        
        print(f"✓ Model loaded: {self.model_id}")
        self._log_params()
    
    def _log_params(self):
        """Log trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def parameters(self):
        """Get model parameters for optimizer."""
        return self.model.parameters()
    
    def freeze_encoder(self):
        """Freeze encoder layers (default for low-resource training)."""
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  After freeze: {trainable:,} trainable params")
    
    def generate(self, input_features, language=None):
        """
        Generate transcriptions.
        
        Args:
            input_features: preprocessed audio features (tensor)
            language: language code override
        
        Returns:
            Generated token ids or text
        """
        lang = language or self.language
        if not torch.is_tensor(input_features):
            input_features = torch.tensor(input_features)
        
        device = next(self.model.parameters()).device
        input_features = input_features.to(device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_features,
                language=f"<|{lang}|>" if lang else None,
            )
        return generated
    
    def transcribe(self, audio_path, language=None):
        """
        Transcribe an audio file.
        
        Args:
            audio_path: path to audio file
            language: language code override
        
        Returns:
            dict with 'text' key containing transcription
        """
        lang = language or self.language
        result = self.pipe(audio_path, generate_kwargs={"language": lang})
        return result
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✓ Checkpoint loaded from {path}")


if __name__ == "__main__":
    # Test loading
    try:
        model = WhisperModel(model_id="openai/whisper-small")
        model.freeze_encoder()
        print("\n✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
