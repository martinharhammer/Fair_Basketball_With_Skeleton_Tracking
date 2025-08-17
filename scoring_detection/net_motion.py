import cv2
import json
import os
import numpy as np
import datetime as dt

# --- paths ---
frames_dir = "002/raw_frames"   # path to your extracted frames
output_dir = "output/frames_with_hoop"
stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dated_output_dir = os.path.join(output_dir, stamp)
os.makedirs(dated_output_dir, exist_ok=True)
print(f"[INFO] Saving ROI frames to: {dated_output_dir}")

# --- load frames ---
frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])

# --- load annotations ---
hoop_annos = {}
with open("002/hoop/detections.jsonl", "r") as f:
    for line in f:
        ann = json.loads(line)
        hoop_annos[ann["frame"]] = ann   # expects "bbox": [cx, cy, w, h]  <-- center-based

print(f"[INFO] Loaded {len(frame_files)} frames")
print(f"[INFO] Loaded {len(hoop_annos)} hoop annotations")

def to_topleft_xywh_from_center(cx, cy, w, h):
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    w = int(round(w))
    h = int(round(h))
    return x, y, w, h

def define_ROI(x, y, w, h, expand_ratio=0.30, img_shape=None):
    """Expand bbox by expand_ratio on all sides, clamp to image bounds."""
    W, H = img_shape[1], img_shape[0]
    x -= w * expand_ratio
    y -= h * expand_ratio
    w *= 1 + 2 * expand_ratio
    h *= 1 + 2 * expand_ratio
    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    w = int(round(min(W - x, w)))
    h = int(round(min(H - y, h)))
    return x, y, w, h

# ----- Net-mask parameters (relative to the TIGHT bbox, not ROI) -----
TOP_FRAC    = 0.35  # keep lower 65% of the bbox (exclude rim/backboard)
TAPER_FRAC  = 0.12  # widen toward the bottom (each side) to match net flare
BOTTOM_PAD  = 0.04  # trim very bottom few % (avoid floor/foreground)
# --------------------------------------------------------------------

def build_net_trapezoid_from_bbox(bx, by, bw, bh, img_shape):
    H, W = img_shape[0], img_shape[1]
    top_y  = int(round(by + TOP_FRAC * bh))
    bot_y  = int(round(by + (1.0 - BOTTOM_PAD) * bh))
    left_x = int(round(bx))
    right_x= int(round(bx + bw))
    taper  = int(round(TAPER_FRAC * bw))
    # clamp
    top_y   = max(0, min(H - 1, top_y))
    bot_y   = max(0, min(H - 1, bot_y))
    left_x  = max(0, min(W - 1, left_x))
    right_x = max(0, min(W - 1, right_x))
    # trapezoid (clockwise)
    tl = (left_x + taper,  top_y)
    tr = (right_x - taper, top_y)
    br = (right_x,         bot_y)
    bl = (left_x,          bot_y)
    poly = np.array([[tl], [tr], [br], [bl]], dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return poly, mask

def draw_mask_overlay(img_bgr, poly_pts, color=(0,255,0), alpha=0.25, outline_thickness=2):
    overlay = img_bgr.copy()
    cv2.fillPoly(overlay, [poly_pts], color)
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
    cv2.polylines(img_bgr, [poly_pts], True, color, outline_thickness)

def auto_canny(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    lo = max(0, int(0.66 * _))
    hi = min(255, int(1.33 * _))
    if hi <= lo:
        lo, hi = 50, 150
    return cv2.Canny(blur, lo, hi)

def compute_ecr(prev_edges, curr_edges):
    prev_b = (prev_edges > 0).astype(np.uint8)
    curr_b = (curr_edges > 0).astype(np.uint8)
    added   = np.sum((curr_b == 1) & (prev_b == 0))
    removed = np.sum((curr_b == 0) & (prev_b == 1))
    before  = max(1, int(np.sum(prev_b)))
    return (added + removed) / float(before)

def phase_shift_same_size(prev_gray_same, curr_gray_same):
    """prev_gray_same and curr_gray_same must be same shape."""
    h, w = prev_gray_same.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_64F)
    shift, _ = cv2.phaseCorrelate(prev_gray_same.astype(np.float64),
                                  curr_gray_same.astype(np.float64), win)
    dx, dy = shift  # (dx, dy)
    return float(dx), float(dy)

# --- state ---
prev_edges_full = None        # full-frame masked edges (after alignment)
prev_gray_full  = None        # previous full-frame grayscale
prev_exp_rect   = None        # (rx, ry, rw, rh)
baseline_edges_count = None

ECR_THR   = 0.80   # tweak as needed
DROP_THR  = 0.30

for fname in frame_files:
    frame_path = os.path.join(frames_dir, fname)
    img = cv2.imread(frame_path)
    if img is None:
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    verdict_text = "NO_MOVEMENT"
    color_verdict = (0, 200, 0)
    ecr = 0.0
    curr_edges_count = 0
    occluded = 0  # for HUD

    if fname in hoop_annos and "bbox" in hoop_annos[fname]:
        cx, cy, bw, bh = map(float, hoop_annos[fname]["bbox"])
        bx, by, bw, bh = map(int, to_topleft_xywh_from_center(cx, cy, bw, bh))

        # 1) expanded ROI for stabilization (we won't draw it)
        rx, ry, rw, rh = define_ROI(bx, by, bw, bh, expand_ratio=0.30, img_shape=img.shape)

        # 2) net mask from tight bbox (for detection) and draw overlay
        poly_pts, net_mask_full = build_net_trapezoid_from_bbox(bx, by, bw, bh, img.shape)
        draw_mask_overlay(img, poly_pts, color=(0,255,0), alpha=0.25)

        # 3) edges inside tight bbox (fast)
        roi = img[by:by+bh, bx:bx+bw]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_mask = net_mask_full[by:by+bh, bx:bx+bw]
        edges = auto_canny(roi_gray)
        edges_m_roi = cv2.bitwise_and(edges, edges, mask=roi_mask.astype(np.uint8))

        # 4) paste masked edges onto a full-frame canvas (stable size)
        H, W = img.shape[:2]
        edges_m_full = np.zeros((H, W), dtype=np.uint8)
        edges_m_full[by:by+bh, bx:bx+bw] = edges_m_roi

        # edge count for occlusion logic
        curr_edges_count = int(np.sum(edges_m_full > 0))
        if baseline_edges_count is None:
            baseline_edges_count = max(1, curr_edges_count)

        # 5) occlusion guard
        occluded = int(curr_edges_count < (1.0 - DROP_THR) * baseline_edges_count)
        if not occluded:
            baseline_edges_count = int(0.9 * baseline_edges_count + 0.1 * curr_edges_count)

        # 6) estimate camera shift on the INTERSECTION of expanded ROIs (same size)
        edges_m_full_aligned = edges_m_full
        if (prev_gray_full is not None) and (prev_exp_rect is not None) and (not occluded):
            prx, pry, prw, prh = prev_exp_rect

            # intersection rectangle between current (rx,ry,rw,rh) and previous (prx,pry,prw,prh)
            ix1 = max(rx, prx)
            iy1 = max(ry, pry)
            ix2 = min(rx + rw, prx + prw)
            iy2 = min(ry + rh, pry + prh)
            iw = ix2 - ix1
            ih = iy2 - iy1

            if iw >= 32 and ih >= 32:  # require some minimum area
                prev_patch = prev_gray_full[iy1:iy2, ix1:ix2]
                curr_patch = img_gray[iy1:iy2, ix1:ix2]
                dx, dy = phase_shift_same_size(prev_patch, curr_patch)

                # align current full-frame edges onto previous coordinate frame
                M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
                edges_m_full_aligned = cv2.warpAffine(
                    edges_m_full, M, (W, H),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            # else: skip stabilization for this frame

        # 7) decide: only when not occluded
        if (prev_edges_full is not None) and (not occluded):
            ecr = compute_ecr(prev_edges_full, edges_m_full_aligned)
            if ecr > ECR_THR:
                verdict_text = "NET_MOVED"
                color_verdict = (0, 0, 255)
            else:
                verdict_text = "NO_MOVEMENT"
                color_verdict = (0, 200, 0)
        elif occluded:
            verdict_text = "OCCLUDED"
            color_verdict = (0, 255, 255)

        # 8) update references (only if not occluded)
        if not occluded:
            prev_edges_full = edges_m_full_aligned.copy()
            prev_gray_full  = img_gray.copy()
            prev_exp_rect   = (rx, ry, rw, rh)

    # --- top-right overlay ---
    H, W = img.shape[:2]
    margin = 20
    debug_text = f"ECR={ecr:.3f} edges={curr_edges_count} base={baseline_edges_count} occ={occluded}"
    cv2.putText(img, verdict_text, (W - 200, margin+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_verdict, 2)
    cv2.putText(img, debug_text, (W - 200, margin+45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    out_path = os.path.join(dated_output_dir, fname)
    cv2.imwrite(out_path, img)

