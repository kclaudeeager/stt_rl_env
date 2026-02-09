"""
Evaluation utilities - compute WER on validation set.
"""

import torch
from jiwer import wer
from typing import List


def evaluate_wer(model, processor, dataloader, device="cpu") -> float:
    """
    Compute WER on a validation set.
    
    Args:
        model: Whisper model
        processor: Feature processor
        dataloader: DataLoader with audio samples
        device: torch device
    
    Returns:
        WER score (0.0 to 1.0)
    """
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Process audio
            audio = batch.get("audio")
            transcriptions = batch.get("normalized_transcription", [])
            
            if audio is None or not transcriptions:
                continue
            
            # Extract features
            if isinstance(audio, dict):
                # From datasets library
                features = processor(
                    audio["array"],
                    sampling_rate=audio["sampling_rate"],
                    return_tensors="pt"
                )
            else:
                features = processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
            
            input_features = features.input_features.to(device)
            
            # Generate predictions
            predicted_ids = model.model.generate(input_features)
            pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            predictions.extend(pred_text)
            references.extend(transcriptions)
    
    if not references:
        return 1.0
    
    # Compute WER
    wer_score = wer(references, predictions)
    return wer_score
