"""Whisper Base speech-to-text on the robot: "I finished" while a dose is watched, and words in check-ins.

The slot session turns listening on only while a dose is watched (robot_microphone consent) or during a check-in
conversation (all check-in consents), and mutes it while the robot speaks. Audio stays in memory: silero VAD cuts
speech segments and Whisper Base turns each into text. In "done" mode only whether a "finished" phrase was heard
leaves this module; in "chat" mode the transcript is handed to the session, which sends that text (never audio) to
the server. Nothing is stored or logged here.

In "chat" mode what was said is handed over as soon as its last segment is decoded, unless the patient has started
speaking again, and comes with timings of each stage (no audio, no words) for the conversation history page.
The session learns that the patient paused as soon as the VAD releases their words (`heard_pause()`), before they
are decoded, so Reachy can answer at once with a 「嗯」 (`note_ack()`). Once the words it answered are handed over,
Reachy says it is thinking about them (「我再想一下喔。」, `think_aloud`, `note_filler()`): only then, so a patient
going on after their pause is never talked over within one turn (their words and Reachy's echo mixed in one segment
came out of SenseVoice garbled, losing 「活」 of 「我不想…活了」), and a cough that decodes to nothing gets no promise of
an answer. The microphone stays on meanwhile, and nothing it hears is ever silenced: should the 「嗯」 or the phrase
come back through it, it is recognised by its words and dropped.
"""

import difflib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from medcare_reachy.bridge.media import resample

log = logging.getLogger(__name__)

STT_DIR = Path.home() / ".medcare_reachy" / "models" / "stt"
STT_FILES = ("silero_vad.onnx",)
SAMPLE_RATE = 16000
VAD_WINDOW = 512                 # silero VAD's window at 16 kHz
MIN_SILENCE_SECONDS = 0.5        # VAD: a pause this long ends a speech segment
MIN_SPEECH_SECONDS = 0.25        # VAD: speech this long starts a segment (is_speech_detected() says so only then)
MUTE_TAIL_SECONDS = 0.35         # chat: the speaker's echo dies down after Reachy speaks (the echo guard catches the rest)
# chat: Reachy's acknowledgement (note_ack) comes back through the microphone, if at all, about MUTE_TAIL_SECONDS
# after it is handed over: the SDK's one audio pipeline delays both the speaker and the microphone by its ~0.17 s
# latency. Speech that starts from the acknowledgement until this long after its echo can have ended may be that echo.
ACK_ECHO_SECONDS = 0.5
# chat: the thinking phrase follows the hand-over this long after, unless by then the patient is speaking again (the
# VAD says so after MIN_SPEECH_SECONDS of speech, and the microphone reaches the app ~0.17 s late): Reachy doesn't
# start talking over a patient who went on, and its echo doesn't land in the middle of their words. Speech heard
# while Reachy's 「嗯」 may still be echoing, or being decoded, is waited for (its words decide) up to THINK_WAIT_SECONDS.
THINK_AFTER_SECONDS = 0.5
THINK_WAIT_SECONDS = 2.0
# How Whisper writes a 「嗯」 or a filler like it, at the start of a transcript. Characters that also begin
# words (恩人, 额头, 唔係, 哼歌) count only on their own.
ACK_LEAD = re.compile(r"^(?:[\W_]*(?:[嗯呃]+|[恩额唔哼]+(?![^\W_])|m+-?h?m+(?![a-z])|hm+(?![a-z])))+[\W_]*",
                      re.IGNORECASE)
# Reachy's thinking phrases (clips/manifest.json "variants" > "thinking": 「我再想一下喔。」「讓我想一想喔。」
# 「我想想看喔。」) in the Simplified words Whisper writes back. Rendered with the robot's voice and played through
# a simulated speaker and room, it wrote them word for word, or without the closing 哦; with the echo's start or end
# lost, the rest of the phrase (THINKING_FORMS); also 在 for 再 and 喔 for 哦, and with the patient talking at the
# same time 叫 for 让 and 享/响 for 想 (_SOUND_ALIKES).
THINKING_PHRASES = ("我再想一下哦", "让我想一想哦", "我想想看哦")
MIN_FORM_CHARS = 3   # shorter scraps ("一下", "想看") are never taken for the phrase
_SOUND_ALIKES = {"再": "[再在]", "哦": "[哦喔噢]", "让": "[让讓叫]", "想": "[想享响]"}


def _leftovers(phrase: str) -> set[str]:
    """The phrase, and what is left of it when the VAD missed its start or its end, at least MIN_FORM_CHARS long,
    each also without the closing 哦."""
    heads = {phrase[:end] for end in range(MIN_FORM_CHARS, len(phrase) + 1)}
    tails = {phrase[start:] for start in range(len(phrase) - MIN_FORM_CHARS + 1)}
    forms = heads | tails
    return {form for form in forms | {form.rstrip("哦") for form in forms} if len(form) >= MIN_FORM_CHARS}


THINKING_FORMS = tuple(sorted(set().union(*map(_leftovers, THINKING_PHRASES)), key=len, reverse=True))


def _thinking_form(words: str) -> str:
    return r"[\W_]*".join(_SOUND_ALIKES.get(ch, re.escape(ch)) for ch in words)


# A phrase counts only when punctuation, a space or the end follows it, so a word the patient said right after it
# stays whole: 「我想想看我想死…」 keeps its 想死.
THINKING_LEAD = re.compile(r"^(?:%s)(?=[\W_]|$)[\W_]*" % "|".join(map(_thinking_form, THINKING_FORMS)))
# "done" has no echo guard (the patient's claim repeats the prompt, which ends with 「我吃完了」 itself), so it keeps
# the longer tail: Reachy's own "我吃完了" must never be heard as the patient saying it.
DONE_MUTE_TAIL_SECONDS = 0.8
MAX_QUEUED_SEGMENTS = 8
ECHO_MIN_CHARS = 3               # chat: shorter transcripts ("嗯", "好") are never taken for Reachy's own voice
ECHO_RATIO = 0.6                 # chat: this similar to what Reachy just said -> its own voice
# chat: only speech starting this soon after listening resumed can be an echo, so the patient repeating Reachy's
# words later on ("吃早餐了" after "您今天吃早餐了吗？") is kept. echo_dropped counts the echoes inside this window.
ECHO_WINDOW = 1.5
MODES = ("done", "chat")
IDLE_SLEEP = 0.02
LANGUAGE_CODES = {"zh-TW": "zh", "en": "en"}
# Nice value of the listener thread and the decoder threads it starts: when the Pi is busy, speech yields to the
# camera stream. On 2 Oct 2026, while a dose was watched, decoding the patient's "I finished" coincided with the
# frame stream falling from 10 to 6-7 fps for about 3 s (the server needs 12). The ~1 s stream gaps seen that day
# came in every slot state, mostly with the listener off, so they have another cause (runner.py logs them).
VOICE_NICE = 5

# Substrings of what Whisper writes (Simplified or Traditional Chinese, or English) when someone says they
# have taken it. Any negation voids the match: "還沒吃完" is not "吃完".
DONE_WORDS = ("吃完", "吃好", "吃了", "吃过", "吃過", "吞了", "吞下", "服用了", "好了", "完成",
              "finished", "done", "took it", "taken")
NEGATIONS = ("没", "沒", "不", "未", "not", "n't", "haven", "didn")


def done_phrase(text: str | None) -> str | None:
    """The "finished" word heard in a transcript, or None."""
    text = (text or "").strip().lower()
    if not text or any(word in text for word in NEGATIONS):
        return None
    return next((word for word in DONE_WORDS if word in text), None)


def _plain(text: str | None) -> str:
    """Text without punctuation or spaces, for comparing what was heard with what Reachy said."""
    return re.sub(r"[\W_]+", "", (text or "").lower())


def without_ack(text: str) -> str:
    """A transcript without the 「嗯」 it starts with: Reachy's own, heard back, or the patient's filler."""
    return ACK_LEAD.sub("", text, count=1)


def without_filler(text: str) -> str:
    """A transcript without Reachy's own words heard back at its start: a 「嗯」, then one of its thinking phrases (or
    what is left of one), then a 「嗯」 again. Only at the start, and only a whole form followed by punctuation, a
    space or the end: every other word is the patient's and stays."""
    rest = without_ack(text)
    stripped = THINKING_LEAD.sub("", rest, count=1)
    return without_ack(stripped) if stripped != rest else rest


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def _in_thread(task) -> None:
    threading.Thread(target=task, name="medcare-thinking", daemon=True).start()


class _Heard(NamedTuple):
    """One transcribed chat segment, with the monotonic times its hand-over metrics come from."""
    text: str
    speech: float      # seconds of voiced audio
    ended: float       # when the patient stopped speaking
    released: float    # when the VAD released the segment
    decode: float      # seconds Whisper took


class _MicClock:
    """When each microphone sample was heard, also for audio that piled up while a segment was being decoded.

    Samples count from the VAD's last reset, like its segments' `start`. No sample is read before it was heard, so
    each read shows that sample 0 was heard at the latest `read time - samples so far / rate`; the earliest such
    bound, from reads that found audio just arrived, is the estimate. A backlog read late only gives a later bound.
    After a read found nothing, the next one starts afresh (audio the SDK dropped from a very long backlog).
    """

    def __init__(self):
        self.count, self.zero = 0, None

    def reset(self) -> None:
        self.count, self.zero = 0, None

    def idle(self) -> None:
        self.zero = None

    def add(self, count: int, read_at: float) -> None:
        """Count `count` new samples read at `read_at`."""
        self.count += count
        zero = read_at - self.count / SAMPLE_RATE
        self.zero = zero if self.zero is None else min(self.zero, zero)

    def at(self, index: int) -> float:
        return self.zero + index / SAMPLE_RATE


def lower_thread_priority(nice: int = VOICE_NICE) -> bool:
    """Lower the calling thread's CPU priority; threads it starts afterwards inherit it.

    Linux keeps the nice value per thread, so this leaves the camera stream's threads alone. Elsewhere, or when
    the system refuses, it does nothing.
    """
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
        return True
    except (AttributeError, OSError):
        return False


def to_mono_16k(samples, rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1) if samples.shape[1] <= 8 else samples.mean(axis=0)
    if rate == SAMPLE_RATE:
        return samples
    if rate % SAMPLE_RATE == 0:
        factor = rate // SAMPLE_RATE   # 48 kHz: average each 3 samples (low-pass, then decimate)
        usable = len(samples) // factor * factor
        return samples[:usable].reshape(-1, factor).mean(axis=1)
    return resample(samples, rate, SAMPLE_RATE)


_whisper_models: dict = {}
_whisper_models_lock = threading.Lock()


def _speech_models(models_dir: Path, language: str):
    import sherpa_onnx
    from faster_whisper import WhisperModel

    key = str(models_dir)
    with _whisper_models_lock:
        if key not in _whisper_models:
            _whisper_models[key] = WhisperModel(
                "base", device="cpu", compute_type="int8", cpu_threads=2,
                download_root=str(models_dir / "whisper"))
        recognizer = _whisper_models[key]
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(models_dir / "silero_vad.onnx")
    config.silero_vad.min_silence_duration = MIN_SILENCE_SECONDS
    config.silero_vad.min_speech_duration = MIN_SPEECH_SECONDS
    config.silero_vad.max_speech_duration = 8.0
    config.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    def transcribe(segment: np.ndarray) -> str:
        chunks, _ = recognizer.transcribe(
            segment, language=LANGUAGE_CODES.get(language, "zh"), beam_size=1,
            condition_on_previous_text=False, vad_filter=False)
        return "".join(chunk.text for chunk in chunks).strip()

    return vad, transcribe


class DoneListener:
    """Background thread pulling the robot's microphone.

    "done" mode: `heard_since()` reports a "finished" phrase. "chat" mode: `take_utterance()` returns what the
    patient said once its last segment is decoded and they haven't started speaking again; segments that only repeat
    what Reachy just said (`note_spoken()`) are dropped as its own voice, and so is its acknowledgement heard right
    after it said it (`note_ack()`, `note_filler()`).

    `think_aloud()`, when given, says Reachy's thinking phrase without waiting for it and returns its length in
    seconds (None when none was said). It is called, in a thread of its own (`spawn`), THINK_AFTER_SECONDS after a
    turn that Reachy acknowledged with its 「嗯」 is handed over, unless the patient has started speaking again by then
    or Reachy is already answering.
    """

    def __init__(self, robot, language: str = "zh-TW", models_dir: Path = STT_DIR, *, clock=time.monotonic,
                 load=_speech_models, think_aloud=None, spawn=_in_thread):
        self.robot, self.language, self.models_dir = robot, language, Path(models_dir)
        self.clock, self._load = clock, load
        self.think_aloud, self._spawn = think_aloud, spawn
        missing = [name for name in STT_FILES if not (self.models_dir / name).is_file()]
        self.available = not missing
        if missing:
            log.warning("listening for 'I finished' is off: missing %s in %s", ", ".join(missing), self.models_dir)
        self._lock = threading.Lock()
        self._active = False
        self._holds = 0
        self._released_at = float("-inf")   # when the robot last stopped speaking (listening resumes after a tail)
        self._heard: tuple[float, str] | None = None
        self._mode = "done"
        self._segments: list[_Heard] = []
        self._decoding = False        # a segment is being decoded, or the audio that came meanwhile isn't checked yet
        self._speech = False          # the VAD hears speech that hasn't ended yet
        self._speech_began = float("-inf")   # about when it began
        self._spoken = ""             # what Reachy said last (chat echo guard)
        self._echo_dropped = 0
        self._pause: float | None = None   # chat: when the patient stopped, for words not yet handed over
        self._ack: tuple[float, float] | None = None   # chat: (when Reachy said 「嗯」, the pause it answered)
        # chat: speech starting in here may be Reachy's 「嗯」 or thinking phrase heard back
        self._ack_echo = (float("-inf"), float("-inf"))
        self._think_at: float | None = None   # chat: when the thinking phrase is due (think_aloud)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── control (called from the slot session) ──
    def start(self) -> None:
        if self.available and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="medcare-voice", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
            self._thread = None

    def set_active(self, active: bool, mode: str = "done") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown listening mode {mode}")
        with self._lock:
            if (active and not self._active) or mode != self._mode:
                self._heard = None
                self._segments = []
                self._echo_dropped = 0
                self._pause = self._ack = None
            if not active or mode != self._mode:
                self._think_at = None
            self._mode = mode
            self._active = bool(active) and self.available

    def hold(self) -> None:
        with self._lock:
            self._holds += 1

    def release(self) -> None:
        with self._lock:
            self._holds = max(0, self._holds - 1)
            self._released_at = self.clock()

    def heard_since(self, since: float) -> str | None:
        with self._lock:
            heard = self._heard
        return heard[1] if heard and heard[0] >= since else None

    def note_spoken(self, text: str | None) -> None:
        """Chat: what Reachy is saying, so that hearing it back through the microphone isn't taken for the patient."""
        with self._lock:
            self._spoken = _plain(text)

    def heard_pause(self) -> float | None:
        """Chat: when the patient stopped speaking, known as soon as the VAD releases their words (before they are
        decoded; only once decoded when they may be Reachy's 「嗯」 heard back) and until they are handed over. With
        several segments it stays the first one's: one utterance, one pause. None when nothing is waiting, also once
        all that was waiting decoded to nothing (a cough) or to Reachy's own voice."""
        with self._lock:
            return self._pause

    def hears_speech(self) -> bool:
        """Chat: the VAD hears someone speaking right now."""
        with self._lock:
            return self._speech

    def note_ack(self, seconds: float) -> None:
        """Chat: Reachy has just handed the speaker its 「嗯」, `seconds` long, the moment the patient paused.

        Unlike hold(), this mutes nothing: the segment being decoded is kept, and so is every word the patient says
        under or after the 「嗯」 (silencing that audio by time once cut 「活了」 from 「我不想…活了」). Instead, speech
        that starts while its echo could still come back is checked by its words: the 「嗯」 it starts with is cut off,
        and speech that was nothing else is dropped (`_patients_words`, `without_filler`). Such speech is reported as
        a pause only once it is decoded, so Reachy never answers its own voice.
        """
        with self._lock:
            now = self.clock()
            self._ack_echo = (now, now + seconds + MUTE_TAIL_SECONDS + ACK_ECHO_SECONDS)
            if self._pause is not None and self._ack is None:
                self._ack = (now, self._pause)

    def note_filler(self, seconds: float) -> None:
        """Chat: Reachy has just handed the speaker its thinking phrase, `seconds` long, after the turn it answers was
        handed over. Like the 「嗯」's (note_ack), its echo is recognised by its words, never by muting: speech starting
        while it could come back loses the phrase it starts with, nothing else. The 「嗯」's window, when still open,
        is extended rather than replaced."""
        with self._lock:
            now = self.clock()
            start, end = self._ack_echo
            self._ack_echo = (start if end > now else now,
                              max(end, now + seconds + MUTE_TAIL_SECONDS + ACK_ECHO_SECONDS))

    def _maybe_think(self) -> None:
        """Worker: say the thinking phrase once it is due (`_think_at`), unless the patient is speaking again or has
        said more meanwhile: Reachy doesn't talk over them, and their words get their own 「嗯」. Speech that may be
        Reachy's own 「嗯」 heard back (begun inside its echo window, or still being decoded) is waited for."""
        with self._lock:
            now, due = self.clock(), self._think_at
            if due is None or now < due:
                return
            if self._decoding or (self._speech and self._in_ack_echo(self._speech_began)):
                if now < due + THINK_WAIT_SECONDS:
                    return   # its words decide: dropped as Reachy's echo, or the patient's (_segments)
            self._think_at = None
            if self._speech or self._decoding or self._segments:
                return
        self._spawn(self._think_aloud)   # play_clip may wait for the 「嗯」 to end: never hold up listening

    def _think_aloud(self) -> None:
        try:
            seconds = self.think_aloud()
        except Exception:   # a courtesy: never worth more than a log line
            log.exception("the check-in's thinking phrase failed")
            return
        if seconds:
            self.note_filler(seconds)

    def take_utterance(self) -> str | None:
        heard = self.take_utterance_with_metrics()
        return heard[0] if heard else None

    def take_utterance_with_metrics(self) -> tuple[str, dict] | None:
        """Chat mode: everything said since the last call, and how long each stage took (milliseconds).

        Handed over as soon as the last segment is decoded, unless the VAD hears the patient speaking again: then
        that segment is waited for and joined to the rest.

        handover_ms runs from the end of the patient's speech to this call. Words said while Reachy was still
        answering (the server thinking, then Reachy speaking) can't be taken before it has finished speaking, so
        for those it runs from then instead: waiting behind Reachy's reply is not hand-over time.
        When Reachy acknowledged the pause, ack_ms runs from the end of the patient's speech to the 「嗯」 being
        handed to the speaker, and ack_resumed says whether the patient went on speaking after it within this turn.
        Like every time here, the end of speech is when the app received it: the SDK delays the microphone by
        ~0.17 s and the speaker by as much again, so the patient hears the 「嗯」 about 0.35 s later than ack_ms.
        echo_dropped counts each segment Reachy's own voice was taken out of (a reply's tail, a 「嗯」 or a thinking
        phrase).

        A turn Reachy acknowledged with its 「嗯」 is followed by its thinking phrase (`think_aloud`,
        THINK_AFTER_SECONDS later): the patient's words are in, Reachy works on its reply.
        """
        with self._lock:
            if not self._segments or self._decoding or self._speech:
                return None
            segments, self._segments = self._segments, []
            echoes, self._echo_dropped = self._echo_dropped, 0
            ack, self._ack, self._pause = self._ack, None, None
            now, released_at = self.clock(), self._released_at
            if ack is not None and self.think_aloud is not None:
                self._think_at = now + THINK_AFTER_SECONDS
        text = " ".join(segment.text for segment in segments).strip()
        if not text:
            return None
        last = segments[-1]
        metrics = {
            "speech_ms": _ms(sum(segment.speech for segment in segments)),
            "segments": len(segments),
            "vad_release_ms": _ms(last.released - last.ended),
            "stt_ms": _ms(sum(segment.decode for segment in segments)),
            "stt_last_ms": _ms(last.decode),
            "handover_ms": _ms(now - max(last.ended, released_at)),
            "echo_dropped": echoes,
            "listen_mode": "chat",
        }
        if ack is not None:
            acked, pause = ack
            metrics["ack_ms"] = _ms(acked - pause)
            metrics["ack_resumed"] = any(segment.ended - segment.speech > acked for segment in segments)
        return text, metrics

    # ── worker ──
    def _listening(self) -> bool:
        with self._lock:
            # The tail follows the current mode: the "say when done" prompt is released before watching starts.
            tail = MUTE_TAIL_SECONDS if self._mode == "chat" else DONE_MUTE_TAIL_SECONDS
            return self._active and self._holds == 0 and self.clock() >= self._released_at + tail

    def _run(self) -> None:
        lower_thread_priority()   # before loading: model worker threads inherit the lower priority
        try:
            vad, transcribe = self._load(self.models_dir, self.language)
        except Exception:
            log.exception("speech models failed to load; listening for 'I finished' is off")
            self.available = False
            return
        pending = np.zeros(0, dtype=np.float32)
        dirty = False
        fed = 0   # samples fed to the VAD since its last reset: what its segments' start counts in
        mic = _MicClock()   # when each of those samples was heard
        while not self._stop.is_set():
            chunk = self.robot.get_audio()
            if not self._listening():
                if dirty:   # forget half-heard speech from before a pause or from our own prompt
                    vad.reset()
                    pending, dirty, fed = np.zeros(0, dtype=np.float32), False, 0
                    mic.reset()
                with self._lock:
                    self._decoding = self._speech = False
                    self._think_at = None   # Reachy is speaking: its answer is here, no phrase is due any more
                if chunk is None:
                    time.sleep(IDLE_SLEEP)
                continue
            if chunk is None:
                mic.idle()
                with self._lock:
                    self._decoding = False   # nothing came in while the last segment was decoded
                self._maybe_think()
                time.sleep(IDLE_SLEEP)
                continue
            fed_at = self.clock()   # the newest sample of this chunk was heard about now (earlier, if backlogged)
            samples = to_mono_16k(*chunk)
            mic.add(len(samples), fed_at)
            pending = np.concatenate([pending, samples])
            dirty = True
            while len(pending) >= VAD_WINDOW:
                vad.accept_waveform(pending[:VAD_WINDOW])
                pending = pending[VAD_WINDOW:]
                fed += VAD_WINDOW
            with self._lock:
                # Sampled after every chunk, including the audio that came in while the last segment was decoded:
                # chat hands over only once that is checked and the patient hasn't started speaking again.
                speech = bool(vad.is_speech_detected())
                if speech and not self._speech:   # the VAD needs MIN_SPEECH_SECONDS of it to say so
                    self._speech_began = mic.at(fed) - MIN_SPEECH_SECONDS
                self._speech = speech
                self._decoding = not vad.empty()
            while not vad.empty():
                front = vad.front
                segment = np.asarray(front.samples, dtype=np.float32)
                start = getattr(front, "start", None)
                vad.pop()
                speech = len(segment) / SAMPLE_RATE
                ended = fed_at if start is None else mic.at(start + len(segment))
                began = ended - speech
                with self._lock:
                    # The session can answer the pause while this is decoded; not when it may be Reachy's own
                    # 「嗯」 or thinking phrase heard back, though, or Reachy would answer itself (its words decide,
                    # below).
                    if self._mode == "chat" and self._pause is None and not self._in_ack_echo(began):
                        self._pause = ended
                decode_from = self.clock()
                text = transcribe(segment)
                decode = self.clock() - decode_from
                if not self._listening():
                    with self._lock:
                        if not self._segments:
                            self._pause = self._ack = None   # muted meanwhile: these words are dropped
                    continue
                with self._lock:
                    if self._mode == "chat":
                        text = self._patients_words((text or "").strip(), began)
                        if text:
                            if self._pause is None:
                                self._pause = ended
                            heard = _Heard(text, speech, ended, fed_at, decode)
                            self._segments = (self._segments + [heard])[-MAX_QUEUED_SEGMENTS:]
                        elif not self._segments:   # nothing left to hand over: that pause led nowhere
                            self._pause = self._ack = None
                    elif (word := done_phrase(text)) is not None:
                        self._heard = (self.clock(), word)
            self._maybe_think()

    def _in_ack_echo(self, began: float) -> bool:
        """Chat (under the lock): speech starting at `began` may begin with Reachy's own 「嗯」 or thinking phrase
        heard back."""
        return self._ack_echo[0] <= began < self._ack_echo[1]

    def _patients_words(self, text: str, began: float) -> str:
        """Chat (under the lock): a transcript without Reachy's own voice heard back; echo_dropped counts each
        time some was taken out. Right after Reachy's 「嗯」 or thinking phrase, a 嗯 or thinking phrase the speech
        starts with is taken for its echo (a 嗯 of the patient's own says nothing either); outside that window
        nothing is."""
        if text and self._is_echo(text, began):
            words = ""
        elif text and self._in_ack_echo(began):
            words = without_filler(text)
        else:
            words = text
        if words != text:
            self._echo_dropped += 1
        return words

    def _is_echo(self, text: str, began: float) -> bool:
        """Chat: a segment that only repeats what Reachy just said is its speaker heard through the microphone.

        An echo is the tail of Reachy's own speech, so it starts before listening resumes or right after (`began`:
        when the segment's speech started); the patient repeating Reachy's words later on is kept.
        """
        heard = _plain(text)
        if len(heard) < ECHO_MIN_CHARS or not self._spoken:
            return False
        if began >= self._released_at + MUTE_TAIL_SECONDS + ECHO_WINDOW:
            return False
        return heard in self._spoken or difflib.SequenceMatcher(None, heard, self._spoken).ratio() >= ECHO_RATIO
