# Punto de aleación — automation tools

Repo for all Punto de aleación automation. Tools:
- `autocut.py` — cut silences and filler words across 3 cameras (below)

## Autocut — automatic silence and filler cutting (3 cameras, 1 audio)

Tool for the Punto de aleación workflow: you record with 3 cameras and one
good audio source (the "master"). The script detects silences and filler words
in that master audio and applies the exact same cuts to all 3 cameras, so they
stay in sync. Which camera is on screen at each moment remains your creative
call in the edit.

## What it does

1. **Syncs each camera to the master audio** by comparing waveforms (uses each
   camera's scratch audio, even a bad built-in mic). No clap needed if the
   cameras record their own audio.
2. **Detects long silences** in the master audio (default: over 0.5 s below
   -35 dB) and trims them, keeping 0.13 s of breathing room on each side (the
   editor leaves ~0.25 s per pause).
3. **Detects filler words** by transcribing the audio with Whisper (Spanish)
   and cutting "eh", "em", "mm". Stretched spellings match too ("eeeh",
   "mmmm"). The list is configurable.
4. **Builds one Final Cut Pro timeline** (`<master>_autocut.fcpxml`): the
   master camera on the main track, the other cameras stacked above it, all
   synced and cut at the same points, their scratch audio muted. It references
   the original files: no re-encoding, no quality loss.
5. Also writes `cuts.json` (the exact cut list) and `transcript.json`.

## Install (one time, Mac)

```
brew install ffmpeg python@3.12
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads the Whisper model (a few hundred MB, one time only).
Everything runs locally; your video never leaves your computer.

## Usage

The usual case: the mics are plugged into one of the cameras, so that camera
is the master.

```
.venv/bin/python autocut.py --master camA.mov --cams camA.mov camB.mov camC.mov --out output/
```

- `--master`: the file with the good audio. A camera video (recommended, it
  becomes the main track) or a separate audio file (it goes on its own track
  below and all cameras are muted).
- `--cams`: the camera videos.
- `--out`: folder for the results.

For a ~1 hour episode, expect several minutes of processing (transcription is
the slow part; normal on CPU).

### Useful options

| Option | What it does | Default |
|---|---|---|
| `--min-silence 0.8` | only cut silences longer than this | 0.5 s |
| `--silence-threshold -40` | silence threshold (dB). Lower it if there's background noise | -35 |
| `--keep-gap 0.3` | breathing room kept on each side of a cut | 0.13 s |
| `--timeline 2160p24` | Final Cut project format (`master` = master camera's) | 1080p60 |
| `--fillers "eh,mm"` | filler words to cut | eh,em,mm |
| `--whisper-model medium` | more accurate model (slower); `tiny`/`base` are faster | small |
| `--no-whisper` | cut silences only, no fillers | — |
| `--export edl` / `both` | per-camera EDLs for other editors | fcpxml |
| `--render` | also render cut MP4 previews and cut master audio (slow) | off |
| `--offsets camB.mov=1.5` | manual sync if auto-sync fails | auto |

### If auto-sync fails

The script reports a confidence for each sync. If you see "LOW CONFIDENCE" or a
camera recorded no audio of its own, do a clap at the start of the recording
and pass the offset manually: `--offsets camB.mov=2.35` (positive = the camera
started recording 2.35 s before the master).

## Working with the result in Final Cut Pro

1. File > Import > XML… and pick the `.fcpxml`. It creates an event
   "Punto de aleación" with the project.
2. If the files are on a different Mac than where autocut ran, Final Cut shows
   them as missing: select the clips, File > Relink Files…, and point to your
   copies (same file names).
3. Edit as usual: cameras are stacked, already synced and cut; disable the
   angles you don't want at each moment.

## Tips

- Don't cut every silence: a very low `--min-silence` makes the video feel
  rushed. 0.8–1.0 s usually gives a natural pace.
- Re-runs into the same output folder reuse `transcript.json` and skip the
  transcription.
- Phones record at a variable frame rate; the timeline snaps to the nearest
  standard rate (30, 29.97, 60…). Check sync at the end of long episodes.
