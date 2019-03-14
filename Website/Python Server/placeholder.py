from librosa import load, effects


def this_is_going_to_totally_work_right(signal, sample_rate):
    new_signal = effects.time_stretch(signal, 1.2)
    return new_signal
