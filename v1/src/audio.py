from __future__ import annotations
from pathlib import Path
import math
import shutil
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt

AUDIO_EXTS = {'.wav','.mp3','.flac','.ogg','.m4a','.aac','.wma'}
FAST_SF_EXTS = {'.wav','.flac','.ogg'}


def audio_info(path):
    path = Path(path)
    try:
        info = sf.info(str(path))
        return {
            'readable': True,
            'samplerate': int(info.samplerate),
            'channels': int(info.channels),
            'frames': int(info.frames),
            'duration_sec': float(info.duration),
            'format': info.format,
            'subtype': info.subtype,
        }
    except Exception as e:
        # librosa/audioread is kept only as an audit fallback. Training should use
        # the pre-transcoded PCM WAV cache created by 02b_cache_kaggle_audio.py.
        try:
            y, sr = librosa.load(str(path), sr=None, mono=False, duration=0.2)
            return {
                'readable': True,
                'samplerate': int(sr),
                'channels': 1 if y.ndim == 1 else int(y.shape[0]),
                'frames': None,
                'duration_sec': float(librosa.get_duration(path=str(path))),
                'format': path.suffix.lower(),
                'subtype': '',
                'fallback': True,
            }
        except Exception as e2:
            return {'readable': False, 'error': f'{e}; fallback={e2}'}


def _fix_length(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) < n:
        y = np.pad(y, (0, n-len(y)))
    elif len(y) > n:
        y = y[:n]
    return y.astype(np.float32, copy=False)


def _load_segment_soundfile(path: Path, start_sec: float, duration_sec: float, target_sr: int) -> np.ndarray:
    """Fast random-access loader for training-ready WAV/FLAC/OGG files."""
    with sf.SoundFile(str(path), mode='r') as f:
        src_sr = int(f.samplerate)
        start_frame = max(0, int(round(float(start_sec) * src_sr)))
        frames = max(1, int(round(float(duration_sec) * src_sr)))
        f.seek(min(start_frame, len(f)))
        y = f.read(frames=frames, dtype='float32', always_2d=True)
    if y.shape[1] > 1:
        y = y.mean(axis=1)
    else:
        y = y[:, 0]
    if src_sr != int(target_sr):
        y = librosa.resample(y, orig_sr=src_sr, target_sr=int(target_sr), res_type='soxr_hq')
    return _fix_length(y, int(round(target_sr * duration_sec)))


def _load_segment_ffmpeg(path: Path, start_sec: float, duration_sec: float, target_sr: int) -> np.ndarray:
    """Error-tolerant fallback decoder. Prefer the one-time WAV cache for training."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg was not found on PATH')
    cmd = [
        ffmpeg, '-hide_banner', '-loglevel', 'error',
        '-err_detect', 'ignore_err', '-fflags', '+discardcorrupt',
        '-ss', f'{max(0.0, float(start_sec)):.6f}', '-t', f'{float(duration_sec):.6f}',
        '-i', str(path), '-vn', '-ac', '1', '-ar', str(int(target_sr)),
        '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1'
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0 or len(p.stdout) == 0:
        msg = p.stderr.decode('utf-8', errors='replace')[-2000:]
        raise RuntimeError(f'ffmpeg decode failed (code={p.returncode}): {msg}')
    y = np.frombuffer(p.stdout, dtype='<f4').copy()
    return _fix_length(y, int(round(target_sr * duration_sec)))


def load_audio(path, target_sr=32000, mono=True):
    path = Path(path)
    if path.suffix.lower() in FAST_SF_EXTS:
        info = sf.info(str(path))
        duration = max(float(info.duration), 1.0 / max(1, int(info.samplerate)))
        y = _load_segment_soundfile(path, 0.0, duration, int(target_sr))
        return y.astype(np.float32), int(target_sr)
    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=mono)
        return y.astype(np.float32), int(target_sr)
    except Exception:
        info = audio_info(path)
        if not info.get('readable'):
            raise
        y = _load_segment_ffmpeg(path, 0.0, float(info['duration_sec']), int(target_sr))
        return y, int(target_sr)


def load_segment(path, start_sec, duration_sec, target_sr=32000):
    """Load exactly one fixed-duration mono segment.

    Training-ready WAV/FLAC/OGG uses SoundFile seek/read, avoiding repeated MP3
    decoding. Compressed/problem files fall back to ffmpeg. The returned array is
    always exactly target_sr * duration_sec samples.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in FAST_SF_EXTS:
            y = _load_segment_soundfile(path, start_sec, duration_sec, int(target_sr))
        else:
            y = _load_segment_ffmpeg(path, start_sec, duration_sec, int(target_sr))
    except Exception as e:
        # Last-resort librosa fallback for systems where ffmpeg is unavailable.
        try:
            y, _ = librosa.load(
                str(path), sr=int(target_sr), mono=True,
                offset=max(0, float(start_sec)), duration=float(duration_sec)
            )
            y = _fix_length(y, int(round(target_sr * duration_sec)))
        except Exception as e2:
            raise RuntimeError(
                f'Could not decode audio segment: path={path}, start={start_sec}, '
                f'duration={duration_sec}. Primary error: {e}. Fallback error: {e2}'
            ) from e2
    return y.astype(np.float32, copy=False), int(target_sr)


def bandpass(y, sr, low=800, high=15000, order=6):
    nyq = sr/2
    high = min(high, nyq*0.98)
    if low <= 0 or low >= high: return y
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sosfiltfilt(sos, y).astype(np.float32)


def spectral_gate(y, sr, noise_seconds=0.5, n_fft=1024, hop=256, strength=1.2):
    if len(y) < n_fft: return y
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    noise_frames = max(1, int(noise_seconds * sr / hop))
    noise = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
    mask = np.clip((mag - strength*noise) / (mag + 1e-8), 0.0, 1.0)
    out = librosa.istft(mag*mask*np.exp(1j*phase), hop_length=hop, length=len(y))
    return out.astype(np.float32)


def rms_normalize(y, target_dbfs=-20.0):
    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    if rms < 1e-8: return y
    target = 10**(target_dbfs/20.0)
    return np.clip(y * (target/rms), -1.0, 1.0).astype(np.float32)



def report_full_preprocess(y, sr, cfg):
    """Student-report-compatible source-level preprocessing.

    Order: Butterworth band-pass -> spectral gating -> -40 dB active-region
    concatenation -> RMS normalization. This is intended for the one-time public
    WAV cache before 5-s segmentation, avoiding silence-heavy training clips.
    """
    p = cfg['preprocessing']
    y = bandpass(y, sr, p['bandpass_low_hz'], p['bandpass_high_hz'], p['filter_order'])
    y = spectral_gate(y, sr, p['spectral_gate_noise_seconds'])
    before = len(y)
    y_active, intervals = active_concat(y, p.get('silence_top_db', 40))
    if len(y_active) == 0:
        return np.zeros(0, dtype=np.float32), {'samples_before': int(before), 'samples_after': 0, 'active_intervals': 0}
    y_active = rms_normalize(y_active, p['rms_target_dbfs'])
    return y_active.astype(np.float32), {
        'samples_before': int(before), 'samples_after': int(len(y_active)),
        'active_intervals': int(len(intervals)),
        'retained_fraction': float(len(y_active)/max(1,before)),
    }

def report_sp_pipeline(y, sr, cfg):
    p = cfg['preprocessing']
    y = bandpass(y, sr, p['bandpass_low_hz'], p['bandpass_high_hz'], p['filter_order'])
    y = spectral_gate(y, sr, p['spectral_gate_noise_seconds'])
    y = rms_normalize(y, p['rms_target_dbfs'])
    return y


def active_concat(y, top_db=40):
    if len(y) == 0: return y, []
    intervals = librosa.effects.split(y, top_db=float(top_db))
    if len(intervals) == 0: return np.zeros(0, dtype=np.float32), []
    pieces = [y[s:e] for s,e in intervals]
    return np.concatenate(pieces).astype(np.float32), [(int(s),int(e)) for s,e in intervals]


def condition_features(y, sr):
    if len(y) == 0:
        return np.zeros(7, dtype=np.float32)
    rms = np.sqrt(np.mean(y*y)+1e-12)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) + 1e-12
    p = S / S.sum(axis=0, keepdims=True)
    entropy = float(np.mean(-np.sum(p*np.log(p), axis=0) / np.log(S.shape[0])))
    noise = np.median(np.abs(y[:max(1, min(len(y), int(0.5*sr)))])) + 1e-8
    signal = np.sqrt(np.mean(y*y))+1e-8
    snr_proxy = 20*np.log10(signal/noise)
    vals = np.array([snr_proxy, rms, centroid/(sr/2), bandwidth/(sr/2), flatness, zcr, entropy], dtype=np.float32)
    return np.nan_to_num(vals, nan=0.0, posinf=20.0, neginf=-20.0)


def add_noise_snr(y, snr_db, rng):
    noise = rng.normal(0,1,size=len(y)).astype(np.float32)
    p_sig = np.mean(y*y)+1e-10
    p_noise = np.mean(noise*noise)+1e-10
    scale = math.sqrt(p_sig/(p_noise*(10**(snr_db/10))))
    return np.clip(y + scale*noise, -1,1).astype(np.float32)


def augment_waveform(y, sr, cfg, rng):
    a = cfg.get('augmentation', {})
    out = y.copy()
    if rng.random() < float(a.get('pitch_probability',0)):
        steps = rng.uniform(-float(a.get('pitch_steps',1)), float(a.get('pitch_steps',1)))
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=steps).astype(np.float32)
    if rng.random() < float(a.get('noise_probability',0)):
        snr = rng.uniform(float(a.get('snr_min_db',12)), float(a.get('snr_max_db',30)))
        out = add_noise_snr(out, snr, rng)
    return out


def write_wav(path, y, sr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype='PCM_16')
