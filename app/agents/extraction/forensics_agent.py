"""
agents/forensics_agent.py
Invoice image forensics:
  1. Metadata check   — EXIF editing-tool fingerprints
  2. ELA              — Error Level Analysis for tampered regions
  3. Copy-move check  — cloned regions via ORB keypoint matching
"""
import io
import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try loading exifread; gracefully degrade if missing
try:
    import exifread
    _EXIFREAD_OK = True
except ImportError:
    _EXIFREAD_OK = False
    logger.warning("exifread not installed — EXIF check disabled.")

# ── Known editing-tool signatures in EXIF ──────────────────────────────────────
_EDITOR_TAGS = {
    "photoshop", "gimp", "canva", "snapseed", "lightroom",
    "pixlr", "picsart", "facetune", "meitu", "adobe",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Metadata Check
# ══════════════════════════════════════════════════════════════════════════════

def check_metadata(image_path: str) -> dict:
    """
    Inspect EXIF metadata for editing-tool signatures.

    Returns
    -------
    dict: risk (0-20), flags (list[str]), details
    """
    risk = 0
    flags = []

    if not _EXIFREAD_OK:
        return {"risk": 0, "flags": ["exifread not installed"], "details": {}}

    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception as e:
        return {"risk": 5, "flags": [f"EXIF read error: {e}"], "details": {}}

    details = {str(k): str(v) for k, v in tags.items()}

    # Look for editing tools
    combined = " ".join(details.values()).lower()
    for editor in _EDITOR_TAGS:
        if editor in combined:
            risk = min(risk + 10, 20)
            flags.append(f"Editing tool detected: {editor}")

    # No EXIF at all (common in screenshots or clean PDFs) -> no risk/flag added
    if not tags:
        pass

    return {"risk": risk, "flags": flags, "details": details}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Error Level Analysis (ELA)
# ══════════════════════════════════════════════════════════════════════════════

def run_ela(image_path: str, quality: int = 90) -> dict:
    """
    Recompress the image at *quality* and compute pixel-wise difference.
    Heavily edited areas show higher error levels.

    Returns
    -------
    dict: risk (0-25), max_diff, mean_diff, flags
    """
    try:
        original = Image.open(image_path).convert("RGB")

        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig_arr  = np.array(original,     dtype=np.float32)
        recomp_arr= np.array(recompressed, dtype=np.float32)

        diff       = np.abs(orig_arr - recomp_arr)
        max_diff   = float(diff.max())
        mean_diff  = float(diff.mean())

        # Normalise to risk score 0-25
        risk = min(int((mean_diff / 50.0) * 25), 25)

        flags = []
        if mean_diff > 15:
            flags.append(f"High ELA mean diff ({mean_diff:.1f}) — possible tampering")
        if max_diff > 100:
            flags.append(f"ELA max diff {max_diff:.0f} — localised edit detected")

        return {
            "risk":      risk,
            "max_diff":  round(max_diff, 2),
            "mean_diff": round(mean_diff, 2),
            "flags":     flags,
        }

    except Exception as e:
        logger.error("ELA failed: %s", e)
        return {"risk": 0, "max_diff": 0, "mean_diff": 0, "flags": [f"ELA error: {e}"]}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Copy-Move Detection (ORB keypoint matching)
# ══════════════════════════════════════════════════════════════════════════════

def check_copy_move(image_path: str) -> dict:
    """
    Detect cloned/pasted regions using ORB feature matching.
    High intra-image matching → copy-move forgery.

    Returns
    -------
    dict: risk (0-25), match_count, flags
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"risk": 0, "match_count": 0, "flags": ["OpenCV could not read image"]}

        # Downsample for speed
        h, w = img.shape
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        orb  = cv2.ORB_create(nfeatures=500)
        kps, descs = orb.detectAndCompute(img, None)

        if descs is None or len(kps) < 10:
            return {"risk": 0, "match_count": 0, "flags": ["Too few keypoints for copy-move check"]}

        # Self-match with BruteForce-Hamming
        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descs, descs)

        # Exclude trivial self-matches (same keypoint → distance 0)
        non_trivial = [m for m in matches if m.distance > 0 and m.distance < 30]

        # Filter to pairs far apart spatially (copy from one region to another)
        suspicious = 0
        for m in non_trivial:
            pt1 = np.array(kps[m.queryIdx].pt)
            pt2 = np.array(kps[m.trainIdx].pt)
            if np.linalg.norm(pt1 - pt2) > 30:
                suspicious += 1

        risk  = min(int((suspicious / max(len(kps), 1)) * 100), 25)
        flags = []
        if suspicious > 20:
            flags.append(f"Copy-move: {suspicious} suspicious region matches detected")

        return {"risk": risk, "match_count": suspicious, "flags": flags}

    except Exception as e:
        logger.error("Copy-move check failed: %s", e)
        return {"risk": 0, "match_count": 0, "flags": [f"Copy-move error: {e}"]}


# ══════════════════════════════════════════════════════════════════════════════
# Public entry-point
# ══════════════════════════════════════════════════════════════════════════════

def run_forensics(image_path: str) -> dict:
    """
    Run all three forensics checks and return combined results.

    Returns
    -------
    dict: total_risk, metadata, ela, copy_move, all_flags
    """
    meta      = check_metadata(image_path)
    ela       = run_ela(image_path)
    copy_move = check_copy_move(image_path)

    total_risk = meta["risk"] + ela["risk"] + copy_move["risk"]
    all_flags  = meta["flags"] + ela["flags"] + copy_move["flags"]

    return {
        "total_risk": total_risk,
        "metadata":   meta,
        "ela":        ela,
        "copy_move":  copy_move,
        "all_flags":  all_flags,
    }