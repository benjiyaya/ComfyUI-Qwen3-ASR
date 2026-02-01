from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


import os
import shutil
import torch
import numpy as np
import folder_paths
import comfy.model_management as mm
from qwen_asr import Qwen3ASRModel

# Register Qwen3-ASR models folder with ComfyUI
QWEN3_ASR_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-ASR")
os.makedirs(QWEN3_ASR_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-ASR", QWEN3_ASR_MODELS_DIR)



# -----------------------------
# Color presets (dropdown)
# -----------------------------

COLOR_PRESETS = {
    "White": (255, 255, 255),
    "Yellow": (255, 235, 59),
    "Cyan": (0, 255, 255),
    "Green": (76, 175, 80),
    "Red": (244, 67, 54),
    "Orange": (255, 152, 0),
    "Blue": (33, 150, 243),
    "Magenta": (233, 30, 99),
    "Light Gray": (220, 220, 220),
    "Black": (0, 0, 0),
}


# -----------------------------
# Parsing
# -----------------------------

LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+?)\s*$")


@dataclass
class WordStamp:
    start: float
    end: float
    word: str


@dataclass
class Caption:
    start: float
    end: float
    text: str


def parse_word_timestamps(text: str) -> List[WordStamp]:
    words: List[WordStamp] = []
    for raw_line in (text or "").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        m = LINE_RE.match(raw_line)
        if not m:
            continue
        start = float(m.group(1))
        end = float(m.group(2))
        word = m.group(3).strip()
        if not word:
            continue
        if end < start:
            start, end = end, start
        words.append(WordStamp(start=start, end=end, word=word))
    words.sort(key=lambda w: (w.start, w.end))
    return words


def build_captions(
    words: List[WordStamp],
    gap_threshold: float = 0.35,
    max_chars: int = 42,
    max_caption_duration: float = 3.5,
) -> List[Caption]:
    if not words:
        return []

    captions: List[Caption] = []

    cur_words: List[str] = []
    cur_start = words[0].start
    cur_end = words[0].end

    def flush():
        nonlocal cur_words, cur_start, cur_end
        txt = " ".join(cur_words).strip()
        if txt:
            captions.append(Caption(start=cur_start, end=cur_end, text=txt))
        cur_words = []

    cur_words.append(words[0].word)

    for i in range(1, len(words)):
        w = words[i]
        prev = words[i - 1]
        gap = w.start - prev.end

        next_text = (" ".join(cur_words + [w.word])).strip()
        next_duration = max(cur_end, w.end) - cur_start

        should_break = False
        if gap > gap_threshold:
            should_break = True
        elif len(next_text) > max_chars:
            should_break = True
        elif next_duration > max_caption_duration:
            should_break = True

        if should_break:
            flush()
            cur_start = w.start
            cur_end = w.end
            cur_words.append(w.word)
        else:
            cur_words.append(w.word)
            cur_end = max(cur_end, w.end)

    flush()
    return captions


def caption_for_time(captions: List[Caption], t: float) -> Optional[str]:
    for c in captions:
        if c.start <= t < c.end:
            return c.text
        if c.start > t:
            break
    return None


# -----------------------------
# Fonts
# -----------------------------

def get_default_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size=size)
        except Exception:
            return ImageFont.load_default()


# -----------------------------
# Overlay rendering (PIL) -> Torch RGBA
# -----------------------------

def render_subtitle_overlay_rgba(
    size_wh: Tuple[int, int],
    text: str,
    font_size: int,
    bottom_margin: int,
    max_width_ratio: float,
    outline: int,
    text_color: Tuple[int, int, int],
) -> Image.Image:
    """
    Returns a PIL RGBA image (same size as frame) with ONLY the subtitle drawn.
    Background is transparent (alpha=0).
    """
    W, H = size_wh
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_default_font(font_size)

    max_width = int(W * max_width_ratio)

    def text_width(s: str) -> int:
        bbox = draw.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0]

    # Wrap into multiple centered lines
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and text_width(cand) > max_width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)

    # Compute total height
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    line_gap = max(6, font_size // 6)
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)

    y = H - bottom_margin - total_h
    y = max(10, y)

    # Draw outline + fill (with alpha 255)
    fill_rgba = (text_color[0], text_color[1], text_color[2], 255)
    outline_rgba = (0, 0, 0, 255)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (W - w) // 2

        if outline > 0:
            for ox in range(-outline, outline + 1):
                for oy in range(-outline, outline + 1):
                    if ox == 0 and oy == 0:
                        continue
                    draw.text((x + ox, y + oy), line, font=font, fill=outline_rgba)

        draw.text((x, y), line, font=font, fill=fill_rgba)
        y += h + line_gap

    return overlay


def pil_rgba_to_torch(overlay_rgba: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Returns torch float tensor (H,W,4) in 0..1 on requested device.
    """
    arr = np.array(overlay_rgba).astype(np.float32) / 255.0  # (H,W,4)
    t = torch.from_numpy(arr).to(device=device)
    return t



# -----------------------------
# ComfyUI Node
# -----------------------------

class SubtitleBurnIn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text_timestamps": ("STRING", {"multiline": True, "default": ""}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.1}),
                "text_color": (list(COLOR_PRESETS.keys()),),
            },
            "optional": {
                "gap_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_chars": ("INT", {"default": 42, "min": 10, "max": 120, "step": 1}),
                "max_caption_duration": ("FLOAT", {"default": 3.5, "min": 0.5, "max": 10.0, "step": 0.1}),
                "font_size": ("INT", {"default": 42, "min": 10, "max": 140, "step": 1}),
                "bottom_margin": ("INT", {"default": 40, "min": 0, "max": 300, "step": 1}),
                "outline": ("INT", {"default": 3, "min": 0, "max": 12, "step": 1}),
                "max_width_ratio": ("FLOAT", {"default": 0.92, "min": 0.5, "max": 0.99, "step": 0.01}),
                "log_every_seconds": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 60.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "burn_in"
    CATEGORY = "video/subtitles"

    def burn_in(
        self,
        images: torch.Tensor,
        text_timestamps: str,
        fps: float,
        text_color: str,
        gap_threshold: float = 0.35,
        max_chars: int = 42,
        max_caption_duration: float = 3.5,
        font_size: int = 42,
        bottom_margin: int = 40,
        outline: int = 3,
        max_width_ratio: float = 0.92,
        log_every_seconds: float = 2.0,
    ):
        if images is None or not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Expected 'images' as a ComfyUI IMAGE batch tensor with shape (B,H,W,C).")

        words = parse_word_timestamps(text_timestamps)
        captions = build_captions(
            words,
            gap_threshold=float(gap_threshold),
            max_chars=int(max_chars),
            max_caption_duration=float(max_caption_duration),
        )

        if not captions:
            print("[SubtitleBurnIn] No captions to draw -> passthrough")
            return (images,)

        device = images.device
        color_rgb = COLOR_PRESETS.get(text_color, (255, 255, 255))

        B, H, W, C = images.shape
        if C != 3:
            raise ValueError(f"Expected RGB images (C=3). Got C={C}.")

        images_f = images.float()

        # ---- logging ----
        start_time = time.time()
        next_log_time = start_time + float(log_every_seconds)
        total_seconds = B / float(fps) if fps > 0 else 0.0
        print(
            f"[SubtitleBurnIn] Start: {B} frames ({total_seconds:.2f}s) @ {fps:.3f} fps | "
            f"captions={len(captions)} | color={text_color} | log_every={log_every_seconds:.1f}s | mode=cache+gpu_blend"
        )

        # ---- overlay cache ----
        # Keyed by caption text + relevant render params (so changing font/outline/color regenerates)
        overlay_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        # value = (overlay_rgb(H,W,3), alpha(H,W,1)) on GPU

        out: List[torch.Tensor] = []

        for i in range(B):
            t = float(i) / float(fps)
            cap = caption_for_time(captions, t)

            if not cap:
                out.append(images_f[i])
            else:
                key = (
                    cap,
                    W,
                    H,
                    int(font_size),
                    int(bottom_margin),
                    float(max_width_ratio),
                    int(outline),
                    color_rgb,
                )

                cached = overlay_cache.get(key)
                if cached is None:
                    # Render overlay in PIL (RGBA) once for this caption+params
                    overlay_pil = render_subtitle_overlay_rgba(
                        (W, H),
                        cap,
                        font_size=int(font_size),
                        bottom_margin=int(bottom_margin),
                        max_width_ratio=float(max_width_ratio),
                        outline=int(outline),
                        text_color=color_rgb,
                    )
                    overlay_t = pil_rgba_to_torch(overlay_pil, device=device)  # (H,W,4) float 0..1
                    overlay_rgb_t = overlay_t[..., :3]  # (H,W,3)
                    alpha_t = overlay_t[..., 3:4]       # (H,W,1)
                    overlay_cache[key] = (overlay_rgb_t, alpha_t)
                    overlay_rgb, alpha = overlay_rgb_t, alpha_t
                else:
                    overlay_rgb, alpha = cached

                # GPU blend: out = frame*(1-a) + overlay_rgb*a
                frame = images_f[i]
                a = alpha
                blended = frame * (1.0 - a) + overlay_rgb * a
                out.append(blended)

            # ---- timed progress log ----
            done = i + 1
            now = time.time()
            if done == 1 or done == B or (log_every_seconds > 0 and now >= next_log_time):
                elapsed = now - start_time
                fps_eff = (done / elapsed) if elapsed > 0 else 0.0
                remaining = B - done
                eta_s = (remaining / fps_eff) if fps_eff > 0 else 0.0
                pct = (done * 100.0) / B
                video_t = done / float(fps) if fps > 0 else 0.0
                print(
                    f"[SubtitleBurnIn] {done}/{B} ({pct:.1f}%) | "
                    f"{fps_eff:.2f} frames/s | ETA {eta_s:.1f}s | video {video_t:.2f}s/{total_seconds:.2f}s | "
                    f"cache={len(overlay_cache)}"
                )
                next_log_time = now + float(log_every_seconds)

        out_t = torch.stack(out, dim=0).to(device)

        total = time.time() - start_time
        print(f"[SubtitleBurnIn] Done: {B} frames in {total:.2f}s ({(B/total if total>0 else 0):.2f} frames/s) | cache={len(overlay_cache)}")

        return (out_t,)



# Model repo mappings
QWEN3_ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B": "Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR-0.6B",
}

QWEN3_FORCED_ALIGNERS = {
    "None": None,
    "Qwen/Qwen3-ForcedAligner-0.6B": "Qwen3-ForcedAligner-0.6B",
}

# Supported languages
SUPPORTED_LANGUAGES = [
    "auto",
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Hungarian", "Macedonian", "Romanian"
]


def get_local_model_path(repo_id: str) -> str:
    folder_name = QWEN3_ASR_MODELS.get(repo_id) or QWEN3_FORCED_ALIGNERS.get(repo_id) or repo_id.replace("/", "_")
    return os.path.join(QWEN3_ASR_MODELS_DIR, folder_name)


def migrate_cached_model(repo_id: str, target_path: str) -> bool:
    if os.path.exists(target_path) and os.listdir(target_path):
        return True
    
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hf_model_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    if os.path.exists(hf_model_dir):
        snapshots_dir = os.path.join(hf_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                source = os.path.join(snapshots_dir, snapshots[0])
                print(f"Migrating model from HuggingFace cache: {source} -> {target_path}")
                shutil.copytree(source, target_path, dirs_exist_ok=True)
                return True
    
    ms_cache = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    ms_model_dir = os.path.join(ms_cache, repo_id.replace("/", os.sep))
    if os.path.exists(ms_model_dir):
        print(f"Migrating model from ModelScope cache: {ms_model_dir} -> {target_path}")
        shutil.copytree(ms_model_dir, target_path, dirs_exist_ok=True)
        return True
    
    return False


def download_model_to_comfyui(repo_id: str, source: str) -> str:
    target_path = get_local_model_path(repo_id)
    
    if migrate_cached_model(repo_id, target_path):
        print(f"Model available at: {target_path}")
        return target_path
    
    os.makedirs(target_path, exist_ok=True)
    
    if source == "ModelScope":
        from modelscope import snapshot_download
        print(f"Downloading {repo_id} from ModelScope to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo_id} from HuggingFace to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    
    return target_path


def load_audio_input(audio_input):
    if audio_input is None:
        return None
        
    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]
    
    wav = waveform[0]
    
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0)
    else:
        wav = wav.squeeze(0)
        
    return (wav.numpy().astype(np.float32), sr)



def _chunk_audio_np(wav: np.ndarray, sr: int, chunk_s: float = 50.0, overlap_s: float = 1.0):
    """Split mono float32 waveform into (chunk_wav, sr, offset_seconds) tuples.
    Designed to avoid backend hard limits on long audio inputs.
    """
    chunk_n = int(max(1.0, float(chunk_s)) * sr)
    overlap_n = int(max(0.0, float(overlap_s)) * sr)
    step = max(1, chunk_n - overlap_n)

    chunks = []
    for start in range(0, len(wav), step):
        end = min(len(wav), start + chunk_n)
        chunk = wav[start:end].astype(np.float32, copy=False)
        chunks.append((chunk, sr, start / float(sr)))
        if end >= len(wav):
            break
    return chunks


def _merge_overlap_words(prev_text: str, new_text: str, max_words: int = 20) -> str:
    """Merge two texts by removing duplicated word overlap (best-effort)."""
    prev = (prev_text or "").strip()
    new = (new_text or "").strip()
    if not prev:
        return new
    if not new:
        return prev

    pw = prev.split()
    nw = new.split()
    max_k = min(max_words, len(pw), len(nw))
    best_k = 0
    for k in range(1, max_k + 1):
        if pw[-k:] == nw[:k]:
            best_k = k
    if best_k > 0:
        merged = pw + nw[best_k:]
        return " ".join(merged)
    return prev + " " + new


class Qwen3ASRLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": (list(QWEN3_ASR_MODELS.keys()), {"default": "Qwen/Qwen3-ASR-1.7B"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            },
            "optional": {
                "forced_aligner": (list(QWEN3_FORCED_ALIGNERS.keys()), {"default": "None"}),
                "local_model_path": ("STRING", {"default": "", "multiline": False}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 64}),
                "max_inference_batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("QWEN3_ASR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-ASR"

    def load_model(self, repo_id, source, precision, attention, forced_aligner="None", local_model_path="", max_new_tokens=2048, max_inference_batch_size=32):
        device = mm.get_torch_device()
        
        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
                print("Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        if local_model_path and local_model_path.strip() != "":
            model_path = local_model_path.strip()
            print(f"Loading from local path: {model_path}")
        else:
            local_path = get_local_model_path(repo_id)
            if os.path.exists(local_path) and os.listdir(local_path):
                model_path = local_path
                print(f"Loading from ComfyUI models folder: {model_path}")
            else:
                model_path = download_model_to_comfyui(repo_id, source)
        
        model_kwargs = dict(
            dtype=dtype,
            device_map=str(device),
            max_inference_batch_size=int(max_inference_batch_size),
            max_new_tokens=int(max_new_tokens),
        )
        if attention != "auto":
            model_kwargs["attn_implementation"] = attention
            
        if forced_aligner and forced_aligner != "None":
            aligner_local = get_local_model_path(forced_aligner)
            if not (os.path.exists(aligner_local) and os.listdir(aligner_local)):
                aligner_local = download_model_to_comfyui(forced_aligner, source)
            model_kwargs["forced_aligner"] = aligner_local
            model_kwargs["forced_aligner_kwargs"] = dict(
                dtype=dtype,
                device_map=str(device),
            )
            if attention != "auto":
                model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = attention
        
        print(f"Loading Qwen3-ASR model from {model_path}...")
        model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
        
        return (model,)


class Qwen3ASRTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_ASR_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "context": ("STRING", {"default": "", "multiline": True}),
                "return_timestamps": ("BOOLEAN", {"default": False}),
                "chunk_seconds": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 600.0, "step": 1.0}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "chunk_seconds": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 600.0, "step": 1.0}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "language", "timestamps")
    FUNCTION = "transcribe"
    CATEGORY = "Qwen3-ASR"

    def transcribe(self, model, audio, language="auto", context="", return_timestamps=False, chunk_seconds=50.0, overlap_seconds=1.0):
        audio_data = load_audio_input(audio)
        if audio_data is None:
            return ("", "", "")

        wav, sr = audio_data
        try:
            dur = len(wav) / float(sr)
        except Exception:
            dur = 0.0

        lang = None if language == "auto" else language
        ctx = context.strip() if isinstance(context, str) else ""
        ctx = ctx if ctx else None

        # For short audio, do a single pass.
        if dur <= float(chunk_seconds) + 2.0:
            results = model.transcribe(
                audio=audio_data,
                language=lang,
                context=ctx,
                return_time_stamps=return_timestamps,
            )

            result = results[0]
            text = result.text or ""
            detected_lang = result.language or ""

            timestamps_str = ""
            if return_timestamps and getattr(result, "time_stamps", None):
                ts_lines = []
                for ts in result.time_stamps:
                    ts_lines.append(f"{ts.start_time:.2f}-{ts.end_time:.2f}: {ts.text}")
                timestamps_str = "\n".join(ts_lines)

            return (text, detected_lang, timestamps_str)

        # Long audio: chunk to avoid backend hard limits (often ~60s).
        chunks = _chunk_audio_np(wav, sr, chunk_s=float(chunk_seconds), overlap_s=float(overlap_seconds))

        merged_text = ""
        detected_lang = ""
        ts_lines_all = []

        rolling_ctx = ctx
        for i, (chunk_wav, chunk_sr, offset_s) in enumerate(chunks):
            results = model.transcribe(
                audio=(chunk_wav, chunk_sr),
                language=lang,
                context=rolling_ctx,
                return_time_stamps=return_timestamps,
            )
            result = results[0]
            detected_lang = detected_lang or (result.language or "")

            chunk_text = (result.text or "").strip()
            if chunk_text:
                merged_text = _merge_overlap_words(merged_text, chunk_text, max_words=20)
                # Keep a tiny rolling context to help continuity without exploding prompt.
                rolling_ctx = (merged_text[-400:]) if merged_text else rolling_ctx

            if return_timestamps and getattr(result, "time_stamps", None):
                for ts in result.time_stamps:
                    ts_lines_all.append(f"{(ts.start_time + offset_s):.2f}-{(ts.end_time + offset_s):.2f}: {ts.text}")

        timestamps_str = "\n".join(ts_lines_all) if (return_timestamps and ts_lines_all) else ""
        return (merged_text.strip(), detected_lang, timestamps_str)


class Qwen3ASRBatchTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_ASR_MODEL",),
                "audio_list": ("AUDIO",),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "return_timestamps": ("BOOLEAN", {"default": False}),
                "chunk_seconds": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 600.0, "step": 1.0}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcriptions",)
    FUNCTION = "batch_transcribe"
    CATEGORY = "Qwen3-ASR"

    def batch_transcribe(self, model, audio_list, language="auto", return_timestamps=False, chunk_seconds=50.0, overlap_seconds=1.0):
        # Accept single AUDIO or a list of AUDIO
        if not isinstance(audio_list, list):
            audio_list = [audio_list]

        lang = None if language == "auto" else language

        output_lines = []
        for i, audio in enumerate(audio_list):
            audio_data = load_audio_input(audio)
            if audio_data is None:
                output_lines.append(f"[{i}] (?): <no audio>")
                continue

            wav, sr = audio_data
            dur = len(wav) / float(sr)
            text_all = ""
            ts_all = []

            if dur <= float(chunk_seconds) + 2.0:
                results = model.transcribe(
                    audio=audio_data,
                    language=lang,
                    return_time_stamps=return_timestamps,
                )
                r = results[0]
                text_all = (r.text or "").strip()
                det_lang = r.language or ""
                if return_timestamps and getattr(r, "time_stamps", None):
                    for ts in r.time_stamps:
                        ts_all.append(f"    {ts.start_time:.2f}-{ts.end_time:.2f}: {ts.text}")
            else:
                chunks = _chunk_audio_np(wav, sr, chunk_s=float(chunk_seconds), overlap_s=float(overlap_seconds))
                det_lang = ""
                for (chunk_wav, chunk_sr, offset_s) in chunks:
                    results = model.transcribe(
                        audio=(chunk_wav, chunk_sr),
                        language=lang,
                        return_time_stamps=return_timestamps,
                    )
                    r = results[0]
                    det_lang = det_lang or (r.language or "")
                    text_all = _merge_overlap_words(text_all, (r.text or "").strip(), max_words=20)

                    if return_timestamps and getattr(r, "time_stamps", None):
                        for ts in r.time_stamps:
                            ts_all.append(f"    {(ts.start_time + offset_s):.2f}-{(ts.end_time + offset_s):.2f}: {ts.text}")

            output_lines.append(f"[{i}] ({det_lang}): {text_all}")
            if ts_all:
                output_lines.extend(ts_all)

        return ("\n".join(output_lines),)