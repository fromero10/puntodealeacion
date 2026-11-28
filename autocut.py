#!/usr/bin/env python3
"""
autocut.py - Auto-cut silences and filler words from multi-camera podcast footage.

Setup: 3 cameras + 1 master audio source. The master audio decides every cut;
the same cut list is applied identically to all cameras so they stay in sync.

Usage:
  python3 autocut.py --master master_audio.wav --cams cam1.mp4 cam2.mp4 cam3.mp4 --out out/

Camera sync: automatic waveform cross-correlation of each camera's scratch
audio against the master audio. Fallback: manual offsets with --offsets.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

SR = 16000  # analysis sample rate


# ---------------------------------------------------------------- helpers

def run(cmd, capture=True, check=True):
    p = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None, text=True)
    if check and p.returncode != 0:
        raise RuntimeError("command failed: %s\n%s" % (" ".join(cmd), (p.stderr or "")[-2000:]))
    return p


def ffprobe_json(path):
    p = run(["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path])
    return json.loads(p.stdout)


def media_info(path):
    j = ffprobe_json(path)
    dur = float(j["format"]["duration"])
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps, w, h = 25.0, None, None
    if v:
        num, den = v.get("avg_frame_rate", "25/1").split("/")
        fps = float(num) / float(den) if float(den) else 25.0
        if fps <= 0:
            fps = 25.0
        w, h = v.get("width"), v.get("height")
        rot = next((abs(int(sd.get("rotation", 0))) for sd in v.get("side_data_list", [])
                    if "rotation" in sd), 0)
        if rot in (90, 270):
            w, h = h, w
    # prefer the nominal rate (r_frame_rate): a phone's VFR average of 59.968
    # would otherwise snap to 59.94 instead of 60
    nominal = v.get("r_frame_rate", "0/1").split("/") if v else ["0", "1"]
    nominal = float(nominal[0]) / float(nominal[1]) if float(nominal[1]) else 0
    rate = next((r for r in STANDARD_RATES if abs(r[0] / r[1] - nominal) < 0.001), None) \
        or snap_rate(fps)
    # camera timecode (e.g. Blackmagic Camera app); plain iPhone files have none
    tcs = [j["format"].get("tags", {}).get("timecode")] + \
          [s.get("tags", {}).get("timecode") for s in streams]
    tc_str = next((t for t in tcs if t), None)
    return {"duration": dur, "has_audio": a is not None, "has_video": v is not None,
            "fps": fps, "rate": rate, "width": w, "height": h,
            "tc_start": parse_timecode(tc_str, rate) if tc_str else Fraction(0),
            "audio_channels": int(a.get("channels", 2)) if a else 0,
            "audio_rate": int(a.get("sample_rate", 48000)) if a else 0}


# Standard frame rates as (num, den). Phones record variable frame rate; we
# snap the average to the nearest standard so FCPXML times land on frames.
STANDARD_RATES = [(24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1),
                  (50, 1), (60000, 1001), (60, 1)]


def snap_rate(fps):
    return min(STANDARD_RATES, key=lambda r: abs(r[0] / r[1] - fps))


def parse_timecode(tc_str, rate):
    h, m, s, f = [int(x) for x in re.split(r"[:;.]", tc_str)]
    nominal = int(round(rate[0] / rate[1]))
    return Fraction(((h * 60 + m) * 60 + s) * nominal + f) * Fraction(rate[1], rate[0])


def extract_wav(src, dst, seconds=None):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", dst]
    run(cmd)


def load_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        data = w.readframes(n)
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return a


def next_pow2(n):
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------- sync

def sync_offset(master, cam, sr=SR):
    """Return (offset_sec, confidence) such that cam_time = master_time + offset."""
    n = next_pow2(len(master) + len(cam))
    M = np.fft.rfft(master, n)
    C = np.fft.rfft(cam, n)
    corr = np.fft.irfft(C * np.conj(M), n)
    # circular correlation: index k means lag k for k <= n//2, else lag k - n
    idx = np.arange(n)
    lags = np.where(idx <= n // 2, idx, idx - n)
    peak_idx = int(np.argmax(np.abs(corr)))
    lag = int(lags[peak_idx])
    peak = abs(float(corr[peak_idx]))
    # confidence: peak vs best peak at least 0.5s away from it
    excl = int(0.5 * sr)
    circ_dist = np.minimum((idx - peak_idx) % n, (peak_idx - idx) % n)
    far = np.abs(corr)[circ_dist > excl]
    second = float(np.max(far)) if far.size else 1e-9
    conf = peak / max(second, 1e-9)
    return lag / sr, conf


# ---------------------------------------------------------------- silence

def detect_silences(master_file, threshold_db, min_dur):
    """ffmpeg silencedetect on the master file. Returns [(start, end), ...]."""
    p = run(["ffmpeg", "-hide_banner", "-nostats", "-i", master_file,
             "-af", "silencedetect=noise=%sdB:d=%s" % (threshold_db, min_dur),
             "-f", "null", "-"], check=False)
    txt = (p.stdout or "") + (p.stderr or "")
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", txt)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", txt)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        if e is not None:
            out.append((s, e))
    return out


# ---------------------------------------------------------------- whisper

def transcribe(master_file, model_size, cache_path):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    from faster_whisper import WhisperModel
    print("  loading whisper model '%s' (downloads on first run)..." % model_size, flush=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # decode with ffmpeg ourselves: faster-whisper's own decoder breaks on newer PyAV
    wav = (cache_path or master_file) + ".16k.wav"
    extract_wav(master_file, wav)
    audio = load_wav(wav)
    os.remove(wav)
    # Whisper tends to drop disfluencies; a prompt full of them makes it keep "eh"
    segs, _ = model.transcribe(audio, language="es", word_timestamps=True,
                               vad_filter=True,
                               initial_prompt="Eh, bueno, eh... mmm, o sea, eh, sí, ehh.")
    words = []
    for seg in segs:
        for w in (seg.words or []):
            words.append({"start": round(w.start, 3), "end": round(w.end, 3),
                          "word": w.word})
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(words, f, ensure_ascii=False)
    return words


FILLER_NORMALIZE = str.maketrans("", "", ".,;:!?\"'()[]{}…")


def collapse(word):
    """'Eeehhh,' -> 'eh', 'mmm' -> 'm': matches any stretched spelling."""
    return re.sub(r"(.)\1+", r"\1", word.strip().lower().translate(FILLER_NORMALIZE))


def find_fillers(words, fillers):
    fillers = {collapse(f) for f in fillers}
    hits = []
    for w in words:
        t = collapse(w["word"])
        if t in fillers:
            hits.append((w["start"], w["end"], t))
    return hits


# ---------------------------------------------------------------- cut math

def build_removals(silences, filler_hits, keep_gap, filler_pad, duration):
    rem = []
    for (s, e) in silences:
        a, b = s + keep_gap, e - keep_gap
        if b - a >= 0.10:
            rem.append((a, b, "silence"))
    for (s, e, t) in filler_hits:
        rem.append((max(0.0, s - filler_pad), e + filler_pad, "filler:%s" % t))
    rem.sort()
    merged = []
    for a, b, kind in rem:
        if merged and a <= merged[-1][1] + 0.02:
            pa, pb, pk = merged[-1]
            merged[-1] = (pa, max(pb, b), pk + "+" + kind if kind not in pk else pk)
        else:
            merged.append((a, b, kind))
    return [(max(0.0, a), min(duration, b), k) for a, b, k in merged if b - a >= 0.05]


def kept_segments(duration, removals):
    segs = []
    cur = 0.0
    for a, b, _ in sorted(removals):
        if a - cur >= 0.05:
            segs.append((cur, a))
        cur = max(cur, b)
    if duration - cur >= 0.05:
        segs.append((cur, duration))
    return segs


# ---------------------------------------------------------------- rendering

def render_camera(cam_path, cam_info, kept_master, offset, out_dir, name):
    """kept_master: kept segments in MASTER time. Render in camera time."""
    tmp_dir = os.path.join(out_dir, "_segments_" + name)
    os.makedirs(tmp_dir, exist_ok=True)
    segs = []
    for (ms, me) in kept_master:
        cs, ce = ms + offset, me + offset
        cs = max(0.0, cs)
        ce = min(cam_info["duration"], ce)
        if ce - cs >= 0.08:
            segs.append((cs, ce))
    files = []
    fps = cam_info["fps"]
    for i, (cs, ce) in enumerate(segs):
        seg = os.path.join(tmp_dir, "seg_%04d.mp4" % i)
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % cs, "-i", cam_path,
               "-t", "%.3f" % (ce - cs), "-map", "0:v:0",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-r", "%.3f" % fps]
        if cam_info["width"] and cam_info["height"]:
            cmd += ["-s", "%dx%d" % (cam_info["width"], cam_info["height"])]
        if cam_info["has_audio"]:
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-map", "1:a:0", "-c:a", "aac", "-shortest"]
        cmd += [seg]
        run(cmd)
        files.append(seg)
        if (i + 1) % 25 == 0:
            print("    %s: %d/%d segments" % (name, i + 1, len(segs)), flush=True)
    listfile = os.path.join(tmp_dir, "concat.txt")
    with open(listfile, "w") as f:
        for seg in files:
            f.write("file '%s'\n" % os.path.abspath(seg))
    out = os.path.join(out_dir, name + "_cut.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", out])
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out, segs


def render_master_audio(master_path, kept, out_path):
    if not kept:
        raise RuntimeError("nothing kept")
    parts = []
    tmp = out_path + ".parts"
    os.makedirs(tmp, exist_ok=True)
    for i, (s, e) in enumerate(kept):
        p = os.path.join(tmp, "p_%04d.wav" % i)
        run(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % s, "-i", master_path,
             "-t", "%.3f" % (e - s), "-ac", "1", "-ar", "48000", p])
        parts.append(p)
    listfile = os.path.join(tmp, "concat.txt")
    with open(listfile, "w") as f:
        for p in parts:
            f.write("file '%s'\n" % os.path.abspath(p))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", out_path])
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- exports

def tc(sec, fps):
    f = int(round(sec * fps))
    fr = f % int(round(fps))
    s = (f // int(round(fps))) % 60
    m = (f // (int(round(fps)) * 60)) % 60
    h = f // (int(round(fps)) * 3600)
    return "%02d:%02d:%02d:%02d" % (h, m, s, fr)


def write_edl(path, title, clip_name, segs_cam_time, fps):
    with open(path, "w") as f:
        f.write("TITLE: %s\nFCM: NON-DROP FRAME\n\n" % title)
        rec = 0.0
        for i, (cs, ce) in enumerate(segs_cam_time):
            d = ce - cs
            f.write("%03d  AX       V     C        %s %s %s %s\n"
                    % (i + 1, tc(cs, fps), tc(ce, fps), tc(rec, fps), tc(rec + d, fps)))
            f.write("* FROM CLIP NAME: %s\n\n" % clip_name)
            rec += d


def fcpxml_time(t):
    t = Fraction(t)
    return "0s" if t == 0 else "%d/%ds" % (t.numerator, t.denominator)


def to_frames(sec, frame):
    """Snap seconds to a whole number of frames; returns Fraction seconds."""
    return Fraction(int(round(Fraction(sec) / frame))) * frame


def write_fcpxml(path, project, tracks, kept, master_audio=None, timeline=None):
    """One Final Cut timeline with every camera stacked, all cut identically.

    tracks: list of dicts {file, info, offset, muted}; tracks[0] is the main
    (spine) camera, the rest are connected clips in lanes 1, 2, ... above it.
    kept: kept segments in master time.
    master_audio: optional {file, info} when the master is a separate audio
    file; it goes in lane -1 and every camera is muted.
    timeline: {width, height, rate} of the project; default: the main camera's.
    """
    main = timeline or tracks[0]["info"]
    frame = Fraction(main["rate"][1], main["rate"][0])  # timeline frame
    res, fmt_ids, lines = [], {}, []

    def src_frame(info):
        return Fraction(info["rate"][1], info["rate"][0])

    def fmt_id(info):
        key = (info["width"], info["height"], info["rate"])
        if key not in fmt_ids:
            fmt_ids[key] = "r%d" % (len(fmt_ids) + 1)
            res.append('  <format id="%s" frameDuration="%s" width="%d" height="%d"/>'
                       % (fmt_ids[key], fcpxml_time(Fraction(info["rate"][1], info["rate"][0])),
                          info["width"] or 1920, info["height"] or 1080))
        return fmt_ids[key]

    for t in tracks:
        fmt_id(t["info"])
    items = tracks + ([master_audio] if master_audio else [])
    for i, t in enumerate(items):
        info = t["info"]
        t["id"] = "a%d" % (i + 1)
        attrs = 'id="%s" name=%s start="%s" duration="%s"' % (
            t["id"], quoteattr(os.path.basename(t["file"])),
            fcpxml_time(info["tc_start"]), fcpxml_time(to_frames(info["duration"], src_frame(info))))
        if info["has_video"]:
            attrs += ' hasVideo="1" format="%s"' % fmt_id(info)
        if info["has_audio"]:
            attrs += ' hasAudio="1" audioSources="1" audioChannels="%d" audioRate="%d"' % (
                info["audio_channels"], info["audio_rate"])
        res.append('  <asset %s>' % attrs)
        res.append('   <media-rep kind="original-media" src=%s/>'
                   % quoteattr(Path(t["file"]).resolve().as_uri()))
        res.append('  </asset>')

    def clip_bounds(t, ms, me):
        """Clip [ms, me] (master time) to the file; returns (shift, start, dur) or None."""
        info = t["info"]
        s, e = ms + t.get("offset", 0.0), me + t.get("offset", 0.0)
        shift = max(0.0, -s)
        s, e = max(0.0, s), min(info["duration"], e)
        dur = to_frames(e - s, frame)
        if dur < 5 * frame:  # skip slivers where a camera starts/stops mid-segment
            return None
        return to_frames(shift, frame), info["tc_start"] + to_frames(s, src_frame(info)), dur

    def clip_xml(t, offset, start, dur, lane=None, children=()):
        open_tag = '<asset-clip ref="%s"%s offset="%s" start="%s" duration="%s" name=%s' % (
            t["id"], ' lane="%d"' % lane if lane is not None else "",
            fcpxml_time(offset), fcpxml_time(start), fcpxml_time(dur),
            quoteattr(os.path.splitext(os.path.basename(t["file"]))[0]))
        inner = (['<adjust-volume amount="-96dB"/>'] if t.get("muted") else []) + list(children)
        if not inner:
            return [open_tag + "/>"]
        return [open_tag + ">"] + ["  " + c for c in inner] + ["</asset-clip>"]

    cursor = Fraction(0)
    spine = []
    for (ms, me) in kept:
        b = clip_bounds(tracks[0], ms, me)
        if not b:
            continue
        _, pstart, pdur = b
        children = []
        for lane, t in enumerate(tracks[1:], start=1):
            cb = clip_bounds(t, ms, me)
            if cb:
                shift, cstart, cdur = cb
                children += clip_xml(t, pstart + shift, cstart, min(cdur, pdur - shift), lane)
        if master_audio:
            cb = clip_bounds(master_audio, ms, me)
            if cb:
                shift, cstart, cdur = cb
                children += clip_xml(master_audio, pstart + shift, cstart, min(cdur, pdur - shift), -1)
        spine += clip_xml(tracks[0], cursor, pstart, pdur, children=children)
        cursor += pdur

    lines += ['<?xml version="1.0" encoding="UTF-8"?>', '<!DOCTYPE fcpxml>',
              '<fcpxml version="1.9">', ' <resources>'] + res + [' </resources>']
    lines += [' <library>', '  <event name="Punto de aleación">',
              '   <project name=%s>' % quoteattr(project),
              '    <sequence format="%s" duration="%s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">'
              % (fmt_id(main), fcpxml_time(cursor)),
              '     <spine>'] + ["      " + l for l in spine]
    lines += ['     </spine>', '    </sequence>', '   </project>', '  </event>', ' </library>', '</fcpxml>']
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_timeline(spec):
    """'1080p60' -> {width, height, rate}; 'master' -> None (use main camera)."""
    if spec == "master":
        return None
    m = re.fullmatch(r"(\d+)p([\d.]+)", spec)
    if not m:
        raise SystemExit("--timeline must look like 1080p60 or be 'master'")
    h = int(m.group(1))
    return {"width": h * 16 // 9, "height": h, "rate": snap_rate(float(m.group(2)))}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Auto-cut silences and filler words "
                                             "across multi-camera footage using one master audio.")
    ap.add_argument("--master", required=True, help="master audio file (wav/mp3) or video with the good audio")
    ap.add_argument("--cams", nargs="+", required=True, help="camera video files")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--offsets", default="auto",
                    help="'auto' (default) or comma list like cam1.mp4=2.5,cam2.mp4=-1.0 (cam_time = master_time + offset)")
    ap.add_argument("--sync-window", type=float, default=300,
                    help="seconds from the start used for auto sync (default 300)")
    ap.add_argument("--min-silence", type=float, default=0.5,
                    help="minimum silence length to cut, seconds (default 0.5)")
    ap.add_argument("--silence-threshold", type=float, default=-35,
                    help="silence threshold in dB (default -35)")
    ap.add_argument("--keep-gap", type=float, default=0.13,
                    help="silence padding kept on each side so the video breathes, seconds (default 0.13; "
                         "the editor leaves ~0.25 s per pause)")
    ap.add_argument("--fillers", default="eh,em,mm",
                    help="comma-separated filler words to cut; stretched spellings match too "
                         "(eh = eeh = ehhh). Default: eh,em,mm")
    ap.add_argument("--filler-pad", type=float, default=0.06,
                    help="seconds of padding around a filler word (default 0.06)")
    ap.add_argument("--whisper-model", default="small",
                    help="whisper model size: tiny, base, small, medium (default small)")
    ap.add_argument("--no-whisper", action="store_true", help="skip filler-word detection")
    ap.add_argument("--export", default="fcpxml", choices=["fcpxml", "edl", "both", "none"],
                    help="timeline export: fcpxml (default, one Final Cut timeline with all cameras stacked), "
                         "edl (one per camera), both, none")
    ap.add_argument("--timeline", default="1080p60",
                    help="Final Cut project format: 1080p60 (default, the editor's setting), "
                         "e.g. 2160p24, 720p30, or 'master' to use the master camera's format")
    ap.add_argument("--render", action="store_true",
                    help="also render cut MP4 previews per camera and the cut master audio (slow)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "_analysis")
    os.makedirs(tmp, exist_ok=True)

    minfo = media_info(args.master)
    master_dur = minfo["duration"]
    print("master: %s (%.1fs)" % (args.master, master_dur), flush=True)

    master_wav = os.path.join(tmp, "master.wav")
    print("extracting master audio...", flush=True)
    extract_wav(args.master, master_wav, seconds=args.sync_window)

    # ---- sync offsets
    manual = {}
    if args.offsets != "auto":
        for part in args.offsets.split(","):
            k, v = part.split("=")
            manual[os.path.basename(k.strip())] = float(v)

    mw = load_wav(master_wav)
    cam_infos = {}
    offsets = {}
    for cam in args.cams:
        ci = media_info(cam)
        cam_infos[cam] = ci
        base = os.path.basename(cam)
        if base in manual:
            offsets[cam] = manual[base]
            print("%s: manual offset %.3fs" % (base, offsets[cam]), flush=True)
            continue
        if not ci["has_audio"]:
            print("%s: NO audio track - cannot auto-sync. Provide --offsets %s=SECONDS"
                  % (base, base), flush=True)
            offsets[cam] = 0.0
            continue
        cwav = os.path.join(tmp, base + ".wav")
        extract_wav(cam, cwav, seconds=args.sync_window)
        cw = load_wav(cwav)
        off, conf = sync_offset(mw, cw)
        offsets[cam] = off
        flag = "" if conf >= 2.0 else "  (LOW CONFIDENCE %.1f - verify with a clap or set --offsets)" % conf
        print("%s: offset %+.3fs (confidence %.1f)%s" % (base, off, conf, flag), flush=True)

    # ---- silence detection (full master)
    print("detecting silences...", flush=True)
    silences = detect_silences(args.master, args.silence_threshold, args.min_silence)
    print("  %d silences >= %.2fs" % (len(silences), args.min_silence), flush=True)

    # ---- filler detection
    filler_hits = []
    if not args.no_whisper:
        print("transcribing master audio (filler detection)...", flush=True)
        words = transcribe(args.master, args.whisper_model,
                           os.path.join(args.out, "transcript.json"))
        filler_list = [f.strip() for f in args.fillers.split(",") if f.strip()]
        filler_hits = find_fillers(words, filler_list)
        print("  %d words, %d filler hits (%s)" % (len(words), len(filler_hits), ",".join(filler_list)), flush=True)

    removals = build_removals(silences, filler_hits, args.keep_gap, args.filler_pad, master_dur)
    kept = kept_segments(master_dur, removals)
    removed_total = sum(b - a for a, b, _ in removals)
    print("cuts: %d removal ranges, %.1fs removed of %.1fs (%.0f%%), %d kept segments"
          % (len(removals), removed_total, master_dur,
             100.0 * removed_total / max(master_dur, 1e-9), len(kept)), flush=True)

    with open(os.path.join(args.out, "cuts.json"), "w") as f:
        json.dump({
            "master": os.path.abspath(args.master),
            "master_duration": master_dur,
            "removed": [{"start": round(a, 3), "end": round(b, 3), "kind": k} for a, b, k in removals],
            "kept": [{"start": round(a, 3), "end": round(b, 3)} for a, b in kept],
            "offsets": {os.path.basename(c): round(o, 4) for c, o in offsets.items()},
        }, f, indent=2, ensure_ascii=False)

    # ---- Final Cut timeline: master camera on the spine, others stacked above, muted
    if args.export in ("fcpxml", "both"):
        master_abs = os.path.abspath(args.master)
        main_cam = next((c for c in args.cams if os.path.abspath(c) == master_abs), args.cams[0])
        master_is_cam = os.path.abspath(main_cam) == master_abs
        order = [main_cam] + [c for c in args.cams if c != main_cam]
        tracks = [{"file": c, "info": cam_infos[c], "offset": offsets[c],
                   "muted": not (master_is_cam and c == main_cam)} for c in order]
        master_audio = None if master_is_cam else {"file": args.master, "info": minfo, "offset": 0.0}
        project = os.path.splitext(os.path.basename(args.master))[0] + "_autocut"
        out_xml = os.path.join(args.out, project + ".fcpxml")
        write_fcpxml(out_xml, project, tracks, kept, master_audio, parse_timeline(args.timeline))
        print("final cut timeline: %s" % out_xml, flush=True)

    if args.render:
        print("rendering cut master audio...", flush=True)
        render_master_audio(args.master, kept, os.path.join(args.out, "master_audio_cut.wav"))

    # ---- per-camera EDLs and previews
    for cam in args.cams:
        base = os.path.splitext(os.path.basename(cam))[0]
        ci = cam_infos[cam]
        off = offsets[cam]
        segs_cam = [(max(0.0, s + off), min(ci["duration"], e + off)) for s, e in kept]
        segs_cam = [(s, e) for s, e in segs_cam if e - s >= 0.08]
        if args.export in ("edl", "both"):
            write_edl(os.path.join(args.out, base + "_cut.edl"), base + "_cut",
                      os.path.basename(cam), segs_cam, ci["fps"])
        if args.render:
            print("rendering %s (%d segments)..." % (base, len(segs_cam)), flush=True)
            render_camera(cam, ci, kept, off, args.out, base)
    print("done. outputs in %s" % os.path.abspath(args.out), flush=True)


if __name__ == "__main__":
    main()
