from .nodes import (
    Qwen3ASRLoader,
    Qwen3ASRTranscribe,
    Qwen3ASRBatchTranscribe,
)

NODE_CLASS_MAPPINGS = {
    "Qwen3ASRLoader": Qwen3ASRLoader,
    "Qwen3ASRTranscribe": Qwen3ASRTranscribe,
    "Qwen3ASRBatchTranscribe": Qwen3ASRBatchTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRLoader": "Qwen3-ASR Loader",
    "Qwen3ASRTranscribe": "Qwen3-ASR Transcribe",
    "Qwen3ASRBatchTranscribe": "Qwen3-ASR Batch Transcribe",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
