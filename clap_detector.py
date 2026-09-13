import sounddevice as sd
import numpy as np

print("Make a clap...")

while True:

    audio = sd.rec(
        4410,
        samplerate=44100,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    volume = np.sqrt(np.mean(audio ** 2))

    print("Volume:", volume)