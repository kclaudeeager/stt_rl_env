#!/usr/bin/env python3
"""
Whisper baseline test - verify model loads and inference works.
"""

import torch
from training.model import WhisperModel


def test_whisper_load():
    """Test Whisper model loading."""
    print("="*60)
    print("WHISPER BASELINE TEST")
    print("="*60)
    
    try:
        device = "cpu"  # Use CPU for now, MPS not fully supported
        
        # Load tiny model (70M params)
        print("\n1. Loading whisper-tiny...")
        model = WhisperModel(model_id="openai/whisper-tiny", device=device)
        
        # Freeze encoder
        print("\n2. Freezing encoder for low-resource training...")
        model.freeze_encoder()
        
        # Test forward pass with dummy audio
        print("\n3. Testing forward pass with dummy audio...")
        dummy_audio = torch.randn(16000)  # 1 second at 16kHz
        
        features = model.processor(
            dummy_audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        input_features = features.input_features.to(device)
        print(f"  Input shape: {input_features.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model.model.encoder(input_features)
        
        print(f"  Output shape: {output.last_hidden_state.shape}")
        print("  ✓ Forward pass successful")
        
        # Test generation
        print("\n4. Testing generation...")
        predicted_ids = model.model.generate(input_features, max_new_tokens=50)
        transcription = model.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        print(f"  Generated: {transcription[0]}")
        
        print("\n" + "="*60)
        print("✓ WHISPER BASELINE TEST PASSED")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_whisper_load()
    exit(0 if success else 1)
