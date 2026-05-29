from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SpeechEnhancementDataset(Dataset):
    def __init__(
        self,
        data_dir: str | None = None,
        num_samples: int = 1000,
        input_size: tuple[int, int] = (128, 64),
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 128,
        split: str = "train",
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.input_size = input_size
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split = split

        self.clean_files: list[str] = []
        self.noise_files: list[str] = []
        self.noisy_files: list[str] = []
        self.data_mode = "synthetic"
        self.use_real_data = False

        if data_dir:
            data_root = Path(data_dir)
            if data_root.exists():
                clean_audio_files = self._scan_audio_files(data_root / "clean")
                noise_audio_files = self._scan_audio_files(data_root / "noise")
                noisy_audio_files = self._scan_audio_files(data_root / "noisy") if (data_root / "noisy").exists() else []

                clean_png_files = self._scan_image_files(data_root / "clean")
                noise_png_files = self._scan_image_files(data_root / "noise")
                noisy_png_files = self._scan_image_files(data_root / "noisy") if (data_root / "noisy").exists() else []

                if clean_png_files and noise_png_files and not clean_audio_files and not noise_audio_files:
                    self.data_mode = "png"
                    self.clean_files = self._split_single_list(clean_png_files, split_ratio, seed)
                    self.noise_files = self._split_single_list(noise_png_files, split_ratio, seed)
                    self.noisy_files = noisy_png_files[: len(self.clean_files)] if noisy_png_files else []
                    self.use_real_data = len(self.clean_files) > 0 and len(self.noise_files) > 0
                elif clean_audio_files and noise_audio_files and not clean_png_files and not noise_png_files:
                    self.data_mode = "audio"
                    self.clean_files = self._split_single_list(clean_audio_files, split_ratio, seed)
                    self.noise_files = self._split_single_list(noise_audio_files, split_ratio, seed)
                    self.noisy_files = noisy_audio_files[: len(self.clean_files)] if noisy_audio_files else []
                    self.use_real_data = len(self.clean_files) > 0 and len(self.noise_files) > 0
                elif any((clean_audio_files, noise_audio_files, clean_png_files, noise_png_files)):
                    print(f"[WARNING] Mixed or incomplete data types under {data_root}. Falling back to synthetic samples.")
                else:
                    print(f"[WARNING] Missing supported files under {data_root}. Falling back to synthetic samples.")
            else:
                print(f"[WARNING] Data directory does not exist: {data_dir}. Falling back to synthetic samples.")

        self.actual_samples = len(self.noisy_files) if self.use_real_data and self.noisy_files else (
            max(len(self.clean_files), len(self.noise_files)) * 5 if self.use_real_data else num_samples
        )
        if self.actual_samples <= 0:
            self.actual_samples = num_samples
            self.use_real_data = False

    def _split_single_list(self, file_list: list[str], split_ratio: tuple[float, float, float], seed: int) -> list[str]:
        rng = random.Random(seed)
        shuffled = file_list.copy()
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * split_ratio[0])
        val_end = int(total * (split_ratio[0] + split_ratio[1]))

        if self.split == "train":
            return shuffled[:train_end]
        if self.split == "val":
            return shuffled[train_end:val_end]
        if self.split == "test":
            return shuffled[val_end:]
        raise ValueError("split must be one of 'train', 'val', 'test'")

    def _scan_audio_files(self, directory: Path) -> list[str]:
        if not directory.exists():
            return []
        audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}
        return [str(directory / name) for name in sorted(os.listdir(directory)) if Path(name).suffix.lower() in audio_extensions]

    def _scan_image_files(self, directory: Path) -> list[str]:
        if not directory.exists():
            return []
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        return [str(directory / name) for name in sorted(os.listdir(directory)) if Path(name).suffix.lower() in image_extensions]

    def _load_audio(self, filepath: str, target_sr: Optional[int] = None) -> torch.Tensor:
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("Please install librosa: pip install librosa") from exc

        audio, _ = librosa.load(filepath, sr=target_sr, mono=True)
        return torch.as_tensor(audio, dtype=torch.float32)

    def _audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        try:
            import librosa
            from scipy.ndimage import zoom
        except ImportError as exc:
            raise ImportError("Please install librosa and scipy: pip install librosa scipy") from exc

        stft_matrix = librosa.stft(audio.cpu().numpy(), n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft_matrix)
        mel_spec = librosa.feature.melspectrogram(
            S=magnitude,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_min = float(mel_spec_db.min())
        mel_max = float(mel_spec_db.max())
        denom = max(mel_max - mel_min, 1e-8)
        mel_spec_db = (mel_spec_db - mel_min) / denom * 2 - 1

        target_f, target_t = self.input_size
        freq, time = mel_spec_db.shape

        if time < target_t:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_t - time)), mode="constant")
        elif time > target_t:
            mel_spec_db = mel_spec_db[:, :target_t]

        if freq != target_f:
            mel_spec_db = zoom(mel_spec_db, (target_f / freq, 1), order=1)

        delta = librosa.feature.delta(mel_spec_db)
        spectrogram = np.stack([mel_spec_db, delta], axis=0)
        return torch.as_tensor(spectrogram, dtype=torch.float32)

    def _load_spectrogram_image(self, filepath: str) -> torch.Tensor:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("Please install opencv-python: pip install opencv-python") from exc

        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image file: {filepath}")

        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image.astype(np.float32)
        image_min = float(image.min())
        image_max = float(image.max())
        denom = max(image_max - image_min, 1e-8)
        image = (image - image_min) / denom * 2 - 1

        target_f, target_t = self.input_size
        image = cv2.resize(image, (target_t, target_f), interpolation=cv2.INTER_LINEAR)
        spectrogram = np.stack([image, image], axis=0)
        return torch.as_tensor(spectrogram, dtype=torch.float32)

    def _generate_noisy(self, clean_audio: torch.Tensor, noise_audio: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        min_len = min(len(clean_audio), len(noise_audio))
        clean_audio = clean_audio[:min_len]
        noise_audio = noise_audio[:min_len]

        clean_power = torch.mean(clean_audio ** 2)
        noise_power = torch.mean(noise_audio ** 2)
        noise_power = noise_power if noise_power > 0 else torch.tensor(1e-10, device=noise_audio.device)

        snr_linear = 10 ** (snr_db / 10)
        noise_scale = torch.sqrt(clean_power / (noise_power * snr_linear + 1e-10))
        return clean_audio + noise_scale * noise_audio

    def __len__(self) -> int:
        return self.actual_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        if self.use_real_data and self.clean_files and self.noise_files:
            try:
                if self.data_mode == "png":
                    clean_path = self.clean_files[idx % len(self.clean_files)]
                    noise_path = self.noise_files[idx % len(self.noise_files)]
                    clean_spec = self._load_spectrogram_image(clean_path)
                    noise_spec = self._load_spectrogram_image(noise_path)

                    if self.noisy_files:
                        noisy_path = self.noisy_files[idx % len(self.noisy_files)]
                        noisy_spec = self._load_spectrogram_image(noisy_path)
                    else:
                        noisy_spec = torch.clamp(clean_spec + 0.5 * noise_spec, min=-1.0, max=1.0)

                    return {
                        "noisy": noisy_spec,
                        "noise": noise_spec,
                        "clean": clean_spec,
                        "idx": idx,
                        "split": self.split,
                    }

                if self.noisy_files:
                    noisy_path = self.noisy_files[idx % len(self.noisy_files)]
                    clean_path = self.clean_files[idx % len(self.clean_files)]
                    noise_path = self.noise_files[idx % len(self.noise_files)]
                    noisy_audio = self._load_audio(noisy_path, self.sample_rate)
                    clean_audio = self._load_audio(clean_path, self.sample_rate)
                    noise_audio = self._load_audio(noise_path, self.sample_rate)
                else:
                    clean_path = self.clean_files[idx % len(self.clean_files)]
                    noise_path = self.noise_files[(idx * 3 + 7) % len(self.noise_files)]
                    clean_audio = self._load_audio(clean_path, self.sample_rate)
                    noise_audio = self._load_audio(noise_path, self.sample_rate)
                    snr_db = float(np.random.uniform(0, 20))
                    noisy_audio = self._generate_noisy(clean_audio, noise_audio, snr_db)

                return {
                    "noisy": self._audio_to_spectrogram(noisy_audio),
                    "noise": self._audio_to_spectrogram(noise_audio),
                    "clean": self._audio_to_spectrogram(clean_audio),
                    "idx": idx,
                    "split": self.split,
                }
            except Exception as exc:
                print(f"[WARNING] Error loading sample {idx}: {exc}")

        x_clean = torch.randn(2, self.input_size[0], self.input_size[1])
        noise = torch.randn(2, self.input_size[0], self.input_size[1]) * 0.5
        x_noisy = x_clean + noise
        return {"noisy": x_noisy, "noise": noise, "clean": x_clean, "idx": idx, "split": self.split}


def prepare_dataset_structure(base_dir: str = "data/") -> None:
    base_path = Path(base_dir)
    for sub_dir in ("clean", "noise", "noisy"):
        (base_path / sub_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {base_path / sub_dir}")


def generate_and_save_noisy_audio(
    data_dir: str = "data/",
    output_dir: str | None = None,
    num_samples: int = 100,
    snr_range: tuple[float, float] = (0, 20),
    sample_rate: int = 16000,
) -> int:
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("Please install librosa and soundfile: pip install librosa soundfile") from exc

    data_root = Path(data_dir)
    output_path = Path(output_dir) if output_dir else data_root / "noisy"
    output_path.mkdir(parents=True, exist_ok=True)

    clean_dir = data_root / "clean"
    noise_dir = data_root / "noise"
    clean_files = [name for name in sorted(os.listdir(clean_dir)) if Path(name).suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}]
    noise_files = [name for name in sorted(os.listdir(noise_dir)) if Path(name).suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}]

    if not clean_files or not noise_files:
        raise ValueError("clean or noise directory is empty")

    generated_count = 0
    for index in range(num_samples):
        try:
            clean_file = clean_files[index % len(clean_files)]
            noise_file = noise_files[index % len(noise_files)]
            clean_audio, _ = librosa.load(str(clean_dir / clean_file), sr=sample_rate, mono=True)
            noise_audio, _ = librosa.load(str(noise_dir / noise_file), sr=sample_rate, mono=True)

            min_len = min(len(clean_audio), len(noise_audio))
            clean_audio = clean_audio[:min_len]
            noise_audio = noise_audio[:min_len]

            snr_db = float(np.random.uniform(snr_range[0], snr_range[1]))
            clean_power = np.mean(clean_audio**2)
            noise_power = np.mean(noise_audio**2)
            noise_power = noise_power if noise_power > 0 else 1e-10
            snr_linear = 10 ** (snr_db / 10)
            noise_scale = np.sqrt(clean_power / (noise_power * snr_linear + 1e-10))
            noisy_audio = clean_audio + noise_scale * noise_audio
            max_val = max(np.abs(noisy_audio).max(), 1e-8)
            noisy_audio = noisy_audio / max_val * 0.9

            output_file = output_path / f"noisy_{index + 1:04d}_snr{snr_db:.1f}dB.wav"
            sf.write(str(output_file), noisy_audio, sample_rate)
            generated_count += 1
        except Exception as exc:
            print(f"[WARNING] Error generating sample {index}: {exc}")

    print(f"Generated {generated_count} noisy samples in {output_path}")
    return generated_count
