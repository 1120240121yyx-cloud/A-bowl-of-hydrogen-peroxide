---
pipeline_tag: audio-classification
---
This model is an ONNX version of a YAMNet-based classifier trained to recognize environmental and human-made sounds (e.g., speech, music, silence, barking, etc.).

Format: ONNX (.onnx)

Input: float32 mono audio, sampled at 16,000 Hz

Output: Score matrix per class [num_frames, num_classes]

Classes: See mapping in yamnet_class_map.csv

Example for uses
```python

import sounddevice as sd
import numpy as np
import onnxruntime
import scipy.signal
import csv
import threading
import time
from collections import deque, Counter

# Path to the ONNX model file
MODEL_PATH = "./yamnet.onnx"
# Create an inference session with ONNX runtime using CPU provider
session = onnxruntime.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
# Get the name of the model's input node
input_name = session.get_inputs()[0].name

def load_class_map(csv_path="yamnet_class_map.csv"):
    """Load class names from a CSV file into a dictionary mapping class IDs to names."""
    class_map = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            class_id = int(row[0])
            class_name = row[2]  # Class name is in the third column
            class_map[class_id] = class_name
    return class_map

# Load the class mapping from CSV
class_map = load_class_map()

# Audio buffer settings (1.5 seconds at 16kHz sample rate)
BUFFER_SIZE = int(16000 * 1.5) 
# Circular buffer to store audio samples
audio_buffer = deque(maxlen=BUFFER_SIZE)

# Queues for storing and consolidating detection results
detections_queue = deque(maxlen=3)  # Stores recent detections
consolidated_results = deque(maxlen=3)  # Stores consolidated results
last_printed_result = None  # Last result printed to console
last_inference_time = 0  # Timestamp of last inference

# Configuration for sleep mode (temporarily pausing inference)
sleep_triggers = {"music", "speech", "silence"}  # Classes that trigger sleep mode
sleep_duration = 3.0  # How long to sleep after detecting trigger classes
same_class_count = 0  # Counter for consecutive same-class detections
last_detected_class = None  # Last class detected
is_sleeping_until = 0  # Timestamp until which we're in sleep mode

def resample_if_needed(audio, original_sr, target_sr=16000):
    """Resample audio to target sample rate if needed."""
    if original_sr != target_sr:
        audio = scipy.signal.resample_poly(audio, target_sr, int(original_sr))
    return audio

def run_inference(audio_chunk):
    """Run the ONNX model inference on an audio chunk and process results."""
    global detections_queue, consolidated_results, last_printed_result
    global is_sleeping_until, same_class_count, last_detected_class

    try:
        # Run the ONNX model
        outputs = session.run(None, {input_name: audio_chunk.astype(np.float32)})
        scores = outputs[0]
        mean_scores = np.mean(scores, axis=0)  # Average scores across frames

        # Get the class with highest score
        class_id = np.argmax(mean_scores)
        confidence = mean_scores[class_id]
        class_name = class_map.get(class_id, f"Class {class_id}")

        # Track consecutive detections of the same class
        if class_name == last_detected_class:
            same_class_count += 1
        else:
            same_class_count = 1
            last_detected_class = class_name

        # Enter sleep mode if we detect trigger classes consecutively
        if same_class_count >= 3 and class_name in sleep_triggers:
            is_sleeping_until = time.time() + sleep_duration
            print(f" Pausing inferences for {sleep_duration} seconds (detected: {class_name})")
            same_class_count = 0

        # Add detection to queue
        detections_queue.append((class_name, confidence))

        # When queue is full, consolidate results
        if len(detections_queue) == 1:
            most_common_name, _ = Counter([d[0] for d in detections_queue]).most_common(1)[0]
            relevant_conf = [conf for n, conf in detections_queue if n == most_common_name]
            avg_conf = np.mean(relevant_conf) * 100
            if avg_conf > 20:  # Only consider results with >20% confidence
                consolidated_results.append((most_common_name, avg_conf))
            detections_queue.clear()

        # When we have enough consolidated results, print the final result
        if len(consolidated_results) == 3:
            names = [r[0] for r in consolidated_results]
            confidences = [r[1] for r in consolidated_results]
            most_common_name, _ = Counter(names).most_common(1)[0]
            avg_conf = np.mean([c for n, c in consolidated_results if n == most_common_name])
            msg = f" Consolidated result: {most_common_name} with average confidence {avg_conf:.2f}%"
            if msg != last_printed_result:
                print(msg)
                last_printed_result = msg

    except Exception as e:
        print(f" ONNX Error: {e}")

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio input stream."""
    global last_inference_time, is_sleeping_until

    if status:
        print(f" Status: {status}")

    # Get mono audio and resample if needed
    audio = indata[:, 0]
    original_sr = sd.query_devices(sd.default.device[0], 'input')['default_samplerate']
    audio = resample_if_needed(audio, original_sr)

    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Add audio to buffer
    audio_buffer.extend(audio)

    # Run inference if buffer is full and not in sleep mode
    if len(audio_buffer) >= BUFFER_SIZE:
        now = time.time()

        if now < is_sleeping_until:
            return

        # Throttle inference to once per second
        if now - last_inference_time > 1.0:
            last_inference_time = now
            audio_chunk = np.array(audio_buffer)
            # Run inference in a separate thread
            threading.Thread(target=run_inference, args=(audio_chunk,), daemon=True).start()

def main():
    """Main function to start audio streaming and processing."""
    print(" Starting real-time listening (Ctrl+C to stop)...")

    # List available input devices
    devices = sd.query_devices()
    input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]

    print("\nAvailable input devices:")
    for i in input_devices:
        print(f"{i}: {devices[i]['name']}")

    # Let user select input device
    device_id = int(input("Select input device ID: "))
    sd.default.device = (device_id, None)

    # Configure audio settings
    samplerate = int(sd.query_devices(device_id, 'input')['default_samplerate'])
    sd.default.samplerate = samplerate
    sd.default.channels = 1  # Mono audio

    print("🎧 Listening...")

    # Start audio stream with callback
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate):
        try:
            while True:
                time.sleep(0.1)  # Keep main thread alive
        except KeyboardInterrupt:
            print("\n Stopping listener...")

if __name__ == "__main__":
    main()

```

