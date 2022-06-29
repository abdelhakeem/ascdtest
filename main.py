import audio
import model
import time


tf_model = model.load_model()


def callback(frames):
    predictions = model.predictions(tf_model, frames)
    print('\n'.join([str(p) for p in predictions]))
    print()


def main():
    audio_monitor = audio.AudioMonitor(callback=callback,
                                       aggressiveness=0,
                                       trigger_percent=0.9,
                                       vad_duration=0.03)

    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
