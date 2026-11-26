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
    return {"duration": dur, "has_audio": a is not None, "has_video": v is not None,
            "fps": fps, "width": w, "height": h}


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
    segs, _ = model.transcribe(master_file, language="es", word_timestamps=True,
                               vad_filter=True)
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


def find_fillers(words, fillers):
    fillers = {f.lower() for f in fillers}
    hits = []
    for w in words:
        t = w["word"].strip().lower().translate(FILLER_NORMALIZE)
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

def tc(sec, fps=25):
    f = int(round(sec * fps))
    fr = f % int(round(fps))
    s = (f // int(round(fps))) % 60
    m = (f // (int(round(fps)) * 60)) % 60
    h = f // (int(round(fps)) * 3600)
    return "%02d:%02d:%02d:%02d" % (h, m, s, fr)


def write_edl(path, title, clip_name, segs_cam_time):
    with open(path, "w") as f:
        f.write("TITLE: %s\nFCM: NON-DROP FRAME\n\n" % title)
        rec = 0.0
        for i, (cs, ce) in enumerate(segs_cam_time):
            d = ce - cs
            f.write("%03d  AX       V     C        %s %s %s %s\n"
                    % (i + 1, tc(cs), tc(ce), tc(rec), tc(rec + d)))
            f.write("* FROM CLIP NAME: %s\n\n" % clip_name)
            rec += d


def fcpxml_time(sec):
    ms = int(round(sec * 1000))
    return "%d/1000s" % ms


def write_fcpxml(path, project, cam_file, segs_cam_time, total_dur):
    abs_src = os.path.abspath(cam_file)
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE fcpxml>')
    lines.append('<fcpxml version="1.9">')
    lines.append(' <resources>')
    lines.append('  <format id="r1" name="FFVideoFormat1080p25" frameDuration="1/25s" width="1920" height="1080"/>')
    lines.append('  <asset id="r2" name="%s" src="file://%s" start="0s" duration="%s" hasVideo="1" hasAudio="1" format="r1"/>'
                 % (os.path.basename(cam_file), abs_src, fcpxml_time(segs_cam_time[-1][1] if segs_cam_time else total_dur)))
    lines.append(' </resources>')
    lines.append(' <library>')
    lines.append('  <event name="autocut">')
    lines.append('   <project name="%s">' % project)
    lines.append('    <sequence format="r1" duration="%s">' % fcpxml_time(total_dur))
    lines.append('     <spine>')
    off = 0.0
    for (cs, ce) in segs_cam_time:
        d = ce - cs
        lines.append('      <asset-clip ref="r2" offset="%s" start="%s" duration="%s"/>'
                     % (fcpxml_time(off), fcpxml_time(cs), fcpxml_time(d)))
        off += d
    lines.append('     </spine>')
    lines.append('    </sequence>')
    lines.append('   </project>')
    lines.append('  </event>')
    lines.append(' </library>')
    lines.append('</fcpxml>')
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


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
    ap.add_argument("--min-silence", type=float, default=0.7,
                    help="minimum silence length to cut, seconds (default 0.7)")
    ap.add_argument("--silence-threshold", type=float, default=-35,
                    help="silence threshold in dB (default -35)")
    ap.add_argument("--keep-gap", type=float, default=0.25,
                    help="silence padding kept on each side, seconds (default 0.25)")
    ap.add_argument("--fillers", default="eh,este,em,mmm,ehm,uh,um",
                    help="comma-separated filler words to cut (Spanish default)")
    ap.add_argument("--filler-pad", type=float, default=0.06,
                    help="seconds of padding around a filler word (default 0.06)")
    ap.add_argument("--whisper-model", default="small",
                    help="whisper model size: tiny, base, small, medium (default small)")
    ap.add_argument("--no-whisper", action="store_true", help="skip filler-word detection")
    ap.add_argument("--export", default="both", choices=["fcpxml", "edl", "both", "none"],
                    help="timeline export format(s), default both. FCPXML imports into DaVinci Resolve and Premiere.")
    ap.add_argument("--no-render", action="store_true", help="only compute cuts and exports, skip video rendering")
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

    # ---- master audio cut
    print("rendering cut master audio...", flush=True)
    render_master_audio(args.master, kept, os.path.join(args.out, "master_audio_cut.wav"))

    # ---- cameras
    for cam in args.cams:
        base = os.path.splitext(os.path.basename(cam))[0]
        ci = cam_infos[cam]
        off = offsets[cam]
        segs_cam = [(max(0.0, s + off), min(ci["duration"], e + off)) for s, e in kept]
        segs_cam = [(s, e) for s, e in segs_cam if e - s >= 0.08]
        total = sum(e - s for s, e in segs_cam)
        if args.export in ("edl", "both"):
            write_edl(os.path.join(args.out, base + "_cut.edl"), base + "_cut",
                      os.path.basename(cam), segs_cam)
        if args.export in ("fcpxml", "both"):
            write_fcpxml(os.path.join(args.out, base + "_cut.fcpxml"), base + "_cut",
                         cam, segs_cam, total)
        if not args.no_render:
            print("rendering %s (%d segments)..." % (base, len(segs_cam)), flush=True)
            render_camera(cam, ci, kept, off, args.out, base)
    print("done. outputs in %s" % os.path.abspath(args.out), flush=True)


if __name__ == "__main__":
    main()
