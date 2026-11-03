import numpy as np
from scipy import signal
from typing import List, Dict, Tuple


class SignalProcessor:
    """
    Processes the raw optical signal to extract the heart rate (BPM).
    Applies detrending, bandpass filtering, and Fourier transform.
    """

    def __init__(self, fps: float, min_hz: float = 0.7, max_hz: float = 3.0):
        self.fps = fps
        self.min_hz = min_hz
        self.max_hz = max_hz

    def estimate_bpm(self, raw_signal: List[float]) -> Tuple[float, np.ndarray]:
        """
        Calculates the Beats Per Minute (BPM) from a raw color signal.

        Args:
            raw_signal: List of mean color values per frame.

        Returns:
            Tuple containing the estimated BPM and the filtered signal array.
        """
        signal_array = np.array(raw_signal)

        detrended = signal.detrend(signal_array)

        b, a = signal.butter(4, [self.min_hz, self.max_hz], btype='bandpass', fs=self.fps)
        filtered = signal.filtfilt(b, a, detrended)

        fft_values = np.fft.rfft(filtered)
        frequencies = np.fft.rfftfreq(len(filtered), d=1.0 / self.fps)

        fft_values[0] = 0

        peak_idx = np.argmax(np.abs(fft_values))
        peak_freq = frequencies[peak_idx]

        bpm = peak_freq * 60.0
        return bpm, filtered

    def estimate_bpm_windowed(self, raw_rgb_signal: List[Tuple[float, float, float]], window_sec: int = 10, step_sec: int = 2) -> float:
        """
        Calculates the BPM using sliding windows to prevent global noise (like motion artifacts)
        from overwhelming the cardiac signal.

        Args:
            raw_rgb_signal: List of RGB mean color values per frame (in tuples).
            window_sec: Size of the sliding window in seconds.
            step_sec: Step size between consecutive windows in seconds.

        Returns:
            The estimated heart rate (BPM) computed as the median of all windowed estimates.
        """
        signal_array = np.array(raw_rgb_signal)
        window_frames = int(window_sec * self.fps)
        step_frames = int(step_sec * self.fps)

        if len(signal_array) < window_frames:
            pos_signal = self.apply_pos_algorithm(signal_array)
            bpm, _ = self.estimate_bpm(pos_signal.tolist())
            return bpm

        bpms = []
        for i in range(0, len(signal_array) - window_frames + 1, step_frames):
            window_rgb = signal_array[i:i + window_frames]

            pos_signal = self.apply_pos_algorithm(window_rgb)

            bpm, _ = self.estimate_bpm(pos_signal.tolist())
            bpms.append(bpm)

        return float(np.median(bpms))

    def apply_pos_algorithm(self, rgb_signal: np.ndarray) -> np.ndarray:
        """
        Applies the Plane-Orthogonal-to-Skin (POS) algorithm (Wang et al., 2016).

        Args:
            rgb_signal: Numpy array of shape (N_frames, 3) containing R, G, B means.

        Returns:
            1D numpy array representing the robust pulse signal.
        """
        mean_rgb = np.mean(rgb_signal, axis=0)

        if np.any(mean_rgb == 0):
            return rgb_signal[:, 1]

        norm_rgb = rgb_signal / mean_rgb

        rn = norm_rgb[:, 0]
        gn = norm_rgb[:, 1]
        bn = norm_rgb[:, 2]

        x = 3 * rn - 2 * gn
        y = 1.5 * rn + gn - 1.5 * bn

        std_x = np.std(x)
        std_y = np.std(y)

        alpha = std_x / std_y if std_y > 0 else 0

        h_signal = x - alpha * y
        return h_signal

    def compute_coherence_features(self, roi_signals: Dict[str, np.ndarray]) -> List[float]:
        """
        Computes the average magnitude squared coherence between all pairs of facial regions
        within the heart rate frequency band (0.7 - 3.0 Hz).

        Args:
            roi_signals: Dictionary mapping region names to their filtered 1D POS signals.

        Returns:
            A list of 6 coherence score features.
        """
        regions = list(roi_signals.keys())
        features = []

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                sig_a = roi_signals[regions[i]]
                sig_b = roi_signals[regions[j]]

                f, coherence_values = signal.coherence(sig_a, sig_b, fs=self.fps, nperseg=int(self.fps * 4))

                mask = (f >= 0.7) & (f <= 3.0)
                if np.any(mask):
                    mean_coherence = float(np.mean(coherence_values[mask]))
                else:
                    mean_coherence = 0.0

                features.append(mean_coherence)

        return features

    def butter_bandpass_filter(self, data: np.ndarray, lowcut: float = 0.7, highcut: float = 3.0,
                               order: int = 4) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter to isolate the heart rate frequency band.
        """
        nyq = 0.5 * self.fps
        low = lowcut / nyq
        high = highcut / nyq
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))

        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data
