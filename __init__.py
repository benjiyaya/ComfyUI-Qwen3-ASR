from .nodes import (
    Qwen3ASRLoader,
    Qwen3ASRTranscribe,
    Qwen3ASRBatchTranscribe,
    SubtitleBurnIn,
)

NODE_CLASS_MAPPINGS = {
    "Qwen3ASRLoader": Qwen3ASRLoader,
    "Qwen3ASRTranscribe": Qwen3ASRTranscribe,
    "Qwen3ASRBatchTranscribe": Qwen3ASRBatchTranscribe,
    "SubtitleBurnIn": SubtitleBurnIn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRLoader": "Qwen3-ASR Loader",
    "Qwen3ASRTranscribe": "Qwen3-ASR Transcribe",
    "Qwen3ASRBatchTranscribe": "Qwen3-ASR Batch Transcribe",
    "SubtitleBurnIn": "Qwen3-ASR Subtitle To Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]