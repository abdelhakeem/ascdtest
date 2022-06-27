from collections import deque
import numpy as np
import pyaudio
import webrtcvad


class AudioMonitor:
    def __init__(self,
                 *,
                 callback,
                 duration=1.1,
                 vad_duration=0.03,
                 aggressiveness=3,
                 trigger_percent=0.8,
                 format=pyaudio.paInt16,
                 channels=1,
                 rate=16000):
        self.callback = callback
        self.vad = webrtcvad.Vad(aggressiveness)
        self.trigger_percent = trigger_percent
        self.rate = rate
        self.recent = deque(maxlen=int(duration / vad_duration))
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=format,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=int(vad_duration * rate),
                                  stream_callback=self.stream_callback)

    def stream_callback(self, in_data, frame_count, time_info, status):
        self.recent.append((in_data, self.vad.is_speech(in_data, self.rate)))

        if len(self.recent) == self.recent.maxlen:
            num_voiced = len([f for f, speech in self.recent if speech])

            if num_voiced >= self.trigger_percent * self.recent.maxlen:
                buffer = b''.join([f for f, speech in self.recent])
                self.recent.clear()
                int16_data = np.frombuffer(buffer, dtype=np.int16)
                float32_data = int16_data.astype(dtype=np.float32) / 32768
                self.callback(float32_data)

        return None, pyaudio.paContinue

    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
