# Autocut — automatic silence and filler cutting (3 cameras, 1 audio)

Tool for the Punto de aleación workflow: you record with 3 cameras and one
good audio source (the "master"). The script detects silences and filler words
in that master audio and applies the exact same cuts to all 3 cameras, so they
stay in sync. Which camera is on screen at each moment remains your creative
call in the edit.

## What it does

1. **Syncs each camera to the master audio** by comparing waveforms (uses each
   camera's scratch audio, even a bad built-in mic). No clap needed if the
   cameras record their own audio.
2. **Detects long silences** in the master audio (default: over 0.7 s below
   -35 dB) and trims them, keeping 0.25 s of breathing room on each side so the
   pacing survives.
3. **Detects filler words** by transcribing the audio with Whisper (Spanish)
   and cutting "eh", "este", "em", "mmm", "ehm", "uh", "um". The list is
   configurable.
4. **Applies the same cut list to all 3 cameras** and produces:
   - `camX_cut.mp4` — each camera already cut (video + its scratch audio)
   - `master_audio_cut.wav` — the good audio, cut the same way
   - `camX_cut.fcpxml` — importable timeline for DaVinci Resolve and Premiere
     (references the original files, no re-encoding)
   - `camX_cut.edl` — same thing in EDL format (universal fallback)
   - `cuts.json` — the exact cut list, in case you want to adjust by hand

## Install (one time)

You need a computer with:

1. **Python 3.9+** — https://www.python.org/downloads/
2. **ffmpeg** —
   - Windows: download from https://www.gyan.dev/ffmpeg/builds/ and add `ffmpeg.exe` and `ffprobe.exe` to your PATH
   - Mac: `brew install ffmpeg`
3. **Python dependencies**: in a terminal, inside this folder:

```
pip install -r requirements.txt
```

The first run downloads the Whisper model (a few hundred MB, one time only).
Everything runs locally; your video never leaves your computer.

## Usage

```
python3 autocut.py --master master_audio.wav --cams cam1.mp4 cam2.mp4 cam3.mp4 --out output/
```

- `--master`: the good audio file (.wav, .mp3, or even a video whose audio is the good one).
- `--cams`: the 3 camera videos.
- `--out`: folder for the results.

For a ~20 minute video, expect several minutes of processing (transcription is
the slow part; normal on CPU).

### Useful options

| Option | What it does | Default |
|---|---|---|
| `--min-silence 0.9` | only cut silences longer than this | 0.7 s |
| `--silence-threshold -40` | silence threshold (dB). Lower it if there's background noise | -35 |
| `--keep-gap 0.4` | breathing room kept on each side of a cut | 0.25 s |
| `--fillers "eh,mmm"` | filler words to cut | eh,este,em,mmm,ehm,uh,um |
| `--whisper-model medium` | more accurate model (slower); `tiny`/`base` are faster | small |
| `--no-whisper` | cut silences only, no fillers | — |
| `--no-render` | skip the cut MP4s, timelines only | — |
| `--offsets cam2.mp4=1.5` | manual sync if auto-sync fails | auto |

### If auto-sync fails

The script reports a confidence for each sync. If you see "LOW CONFIDENCE" or a
camera recorded no audio of its own, do a clap or a visible/audible tap at the
start of the recording and pass the offset manually:
`--offsets cam2.mp4=2.35` (positive = the camera started recording 2.35 s
before the master; find it by comparing the clap instant in both).

## Working with the result

**Recommended flow (best quality):** import the `.fcpxml` into DaVinci Resolve
(File > Import > Timeline) or Premiere (File > Import). Each timeline
references the camera's original file with the cuts already placed, no quality
loss. Add `master_audio_cut.wav` as the main audio track, mute the scratch
audio, and build your multicam on top.

**Quick flow (preview):** the `camX_cut.mp4` files are already cut and ready to
review or publish directly. Note: video can only cut on frame boundaries
(~40 ms), so the MP4s can drift a fraction of a second from the audio over a
long video. For the final master, use the timeline flow.

## Tips

- If you cut "este", review the result: it's also a normal word ("este video").
  If it produces false cuts, remove it from `--fillers`.
- Don't cut every silence: a very low `--min-silence` makes the video feel
  rushed. 0.7–1.0 s usually gives a natural pace.
- `transcript.json` is saved in the output folder: re-runs with different
  settings skip the transcription.
