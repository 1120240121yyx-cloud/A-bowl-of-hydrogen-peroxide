"""Audio sources and optional decoders.

decode.py    non-WAV -> PCM WAV (Media Foundation / ffmpeg / soundfile), cached
             under work/decoded/, with a guaranteed fallback path.
Future
------
stream.py    non-file sources (network / phone microphone) with the same
             AudioSource contract used by ``screenlight.audio``.
"""
