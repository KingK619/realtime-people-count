import simpleaudio


class Sound:

    def play(self):
        wave_obj = simpleaudio.WaveObject.from_wave_file("bell.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
