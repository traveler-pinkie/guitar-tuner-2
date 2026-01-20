import numpy as np
import sys
from audio_engine import start_stream, stop_stream, AUDIO_QUEUE, callback_count
import dsp_processor as  dsp
from config import SAMPLE_RATE

detected_data = {
    "frequency": 0.0,
    "cents_off": 0.0,
    "note": ""

}

print("Starting audio stream...", flush=True)
stream = start_stream()
print("Audio stream started. Make noise near the microphone...", flush=True)

try:
    count = 0
    while True:
        if not AUDIO_QUEUE.empty():
            data = AUDIO_QUEUE.get()
            rms = np.sqrt(np.mean(data**2))
            count += 1
            window_data = dsp.preprocess_buffer(data)
            print(f"[{count}] RMS Level: {rms:.6f}", flush=True)

            result = dsp.calculate_difference_function(window_data)
            cmnd = dsp.calculate_cumulative_mean_normalized_difference_function(result)
            min_index = dsp.peak_valley_detection(cmnd)
            if min_index is None:
                print(f"[{count}] Pitch detection failed (no peak found)")
            elif min_index <= 0:
                print(f"[{count}] Invalid min_index: {min_index}")
            else:
                detected_data["frequency"] = SAMPLE_RATE / min_index
                print(f"[{count}] Detected frequency: {detected_data['frequency']:.2f} Hz")
except KeyboardInterrupt:
    print(f"\nStopping audio stream...", flush=True)

finally:
    stop_stream(stream)
    print("Audio stream stopped.", flush=True)