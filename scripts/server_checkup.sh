#!/usr/bin/env bash
# Server pre-inference checkup (bash edition): env, data, model weights.
# Verifies everything a full hybrid_som / Qwen / GUI-Actor run needs WITHOUT
# loading multi-GB VLM weights. Exits 0 when READY, 1 otherwise.
#
# Usage (on the server):
#   bash scripts/server_checkup.sh [--quick]
#   bash scripts/server_checkup.sh --dataset-dir <ScreenSpot-v2> \
#       --assets-dir <pretrained> [--quick]
#
# --quick skips the slow probes (image sampling, YOLO smoke, browser launch).
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="$HOME"
DATASET_DIR="$HOME/workspace/TERM_PROJECTS/stu2/GUI-Web-Agent/dataset/ScreenSpot-v2"
ASSETS_DIR="$HOME/workspace/TERM_PROJECTS/stu2/GUI-Web-Agent/pretrained"
QWEN_DIR="Qwen2.5-VL-3B-Instruct"
GUIACTOR_DIR="GUI-Actor-3B-Qwen2.5-VL"
QUICK=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset-dir) DATASET_DIR="$2"; shift 2;;
        --assets-dir) ASSETS_DIR="$2"; shift 2;;
        --qwen-dir) QWEN_DIR="$2"; shift 2;;
        --guiactor-dir) GUIACTOR_DIR="$2"; shift 2;;
        --quick) QUICK=1; shift;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

PASS=0; FAIL=0; FAIL_LIST=""
pass() { PASS=$((PASS+1)); echo "[PASS] $1${2:+ -- $2}"; }
fail() { FAIL=$((FAIL+1)); FAIL_LIST="$FAIL_LIST
  - $1"; echo "[FAIL] $1${2:+ -- $2}"; }
warn() { echo "[WARN] $1${2:+ -- $2}"; }

# pip executable: prefer the one bound to python3 (bare `pip` often absent).
if python3 -m pip --version >/dev/null 2>&1; then PIP="python3 -m pip";
elif command -v pip3 >/dev/null 2>&1; then PIP="pip3";
elif command -v pip >/dev/null 2>&1; then PIP="pip";
else PIP="pip"; fi

echo "repo: $REPO"
echo "dataset: $DATASET_DIR"
echo "assets: $ASSETS_DIR"
echo ""

# ---------------- ENV ----------------
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "none")
if [ "$PYVER" = "none" ]; then
    fail "python3" "not found"
else
    PYMAJ=${PYVER%%.*}; PYMIN=${PYVER##*.}
    if [ "$PYMAJ" -ge 3 ] && [ "$PYMIN" -ge 10 ] 2>/dev/null; then pass "python>=3.10" "$PYVER"; else fail "python>=3.10" "$PYVER"; fi
fi

pkg_ver() { $PIP show "$1" 2>/dev/null | awk '/^Version:/{print $2}'; }
check_pkg() { v=$(pkg_ver "$1"); [ -n "$v" ] && pass "pkg:$2" "$v" || fail "pkg:$2" "MISSING"; }
warn_pkg() { v=$(pkg_ver "$1"); [ -n "$v" ] && warn "pkg:$2" "$v" || warn "pkg:$2" "MISSING"; }

for p in torch torchvision transformers accelerate ultralytics numpy playwright pytest pyyaml huggingface_hub safetensors qwen_vl_utils; do check_pkg "$p" "$p"; done
check_pkg pillow PIL
warn_pkg paddleocr "paddleocr (OCR)"; warn_pkg paddlepaddle "paddlepaddle (OCR)"
warn_pkg easyocr "easyocr (OCR)"; warn_pkg opencv-python-headless "cv2 (OCR)"

if command -v nvidia-smi >/dev/null 2>&1; then
    pass "nvidia-smi" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -2 | tr '\n' ';')"
    if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        pass "torch cuda" "$(python3 -c "import torch; print(torch.cuda.device_count(), 'GPU(s),', torch.cuda.get_device_properties(0).total_memory//2**30, 'GiB')" 2>/dev/null)"
    else
        fail "torch cuda" "nvidia-smi present but torch.cuda unavailable"
    fi
else
    warn "cuda" "nvidia-smi absent -- CPU inference only (slow)"
fi

if command -v stat >/dev/null 2>&1; then
    if [ "$(uname)" = "Darwin" ]; then FREE_MB=$(df -m "$REPO" | awk 'NR==2{print $4}'); else FREE_MB=$(df -m "$REPO" | awk 'NR==2{print $4}'); fi
    if [ "${FREE_MB:-0}" -ge 30720 ]; then pass "disk>=30G free" "$((FREE_MB/1024)) GiB free"; else fail "disk>=30G free" "${FREE_MB:-?} MB free"; fi
fi

if [ "$QUICK" -eq 1 ]; then
    echo "--quick: skipping image sampling, YOLO smoke, browser launch"
else
    if python3 -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={'width':800,'height':600})
    pg.goto('about:blank')
    print('v'+b.version)
    b.close()
" 2>/dev/null; then pass "playwright chromium launch" "see version above"; else fail "playwright chromium launch" "run: python -m playwright install chromium (+ --with-deps w/ sudo)"; fi
fi

# ---------------- REPO FILES ----------------
for rel in scripts/evaluate_screenspot.py scripts/run_agent.py scripts/convert_screenspotv2.py src/browser/proposers.py src/grounding/methods.py src/models/model_factory.py src/models/gui_actor_vlm.py src/models/vendor_gui_actor/modeling_qwen25vl.py; do
    [ -f "$REPO/$rel" ] && pass "repo:$rel" || fail "repo:$rel" "MISSING"
done

# ---------------- DATA ----------------
if [ -d "$DATASET_DIR" ]; then pass "dataset dir" "$DATASET_DIR"; else fail "dataset dir" "$DATASET_DIR"; fi
for plat in desktop mobile web; do
    jf="$DATASET_DIR/screenspot_${plat}_v2.json"
    if [ -f "$jf" ]; then
        n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$jf" 2>/dev/null || echo "?")
        pass "raw json $plat" "$n anns"
    else
        fail "raw json $plat" "MISSING $jf"
    fi
done
N_IMG=$(find "$DATASET_DIR/screenspotv2_image" -maxdepth 1 \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l | tr -d ' ')
if [ "${N_IMG:-0}" -ge 740 ]; then pass "raw images (~756)" "$N_IMG files"; else fail "raw images (~756)" "$N_IMG files"; fi

check_jsonl() {
    fname=$1; expect=$2
    jf="$REPO/data/screenspot/$fname"
    [ -f "$jf" ] || { fail "jsonl $fname" "MISSING"; return; }
    out=$(python3 - "$jf" "$expect" <<'EOF' 2>&1
import json, sys
jf, expect = sys.argv[1], int(sys.argv[2])
lines = open(jf).read().splitlines()
bad_json = bad_img = bad_mac = 0
for l in lines:
    try: s = json.loads(l)
    except Exception: bad_json += 1; continue
    ip = s.get("img_path", "")
    if "/Users/" in ip: bad_mac += 1
    from pathlib import Path
    if not Path(ip).is_file(): bad_img += 1
print(f"{len(lines)} {expect} {bad_json} {bad_img} {bad_mac}")
EOF
)
    # shellcheck disable=SC2181
    set -- $out
    if [ "$1" = "$expect" ] && [ "$3" = "0" ] && [ "$4" = "0" ] && [ "$5" = "0" ]; then
        pass "jsonl $fname" "$1/$expect lines, all img exist, no Mac paths"
    else
        fail "jsonl $fname" "lines=$1/$expect bad_json=$3 missing_img=$4 mac_paths=$5"
    fi
}
check_jsonl v2_full.jsonl 1272
check_jsonl v2_mini.jsonl 12

if [ "$QUICK" -eq 0 ]; then
    if python3 - "$DATASET_DIR" <<'EOF' 2>/dev/null; then
import random, sys
from pathlib import Path
from PIL import Image
pool = [f for ext in ("*.png", "*.jpg", "*.jpeg") for f in (Path(sys.argv[1]) / "screenspotv2_image").glob(ext)]
bad = 0
for f in random.Random(42).sample(pool, min(20, len(pool))):
    try: Image.open(f).verify()
    except Exception: bad += 1
sys.exit(1 if bad or not pool else 0)
EOF
        pass "sample 20 images open" "all readable"
    else
        fail "sample 20 images open" "corrupt/unreadable images found"
    fi
fi

# ---------------- MODEL WEIGHTS ----------------
need_file() { # name path min_mb
    if [ -f "$2" ]; then
        if [ "$(uname)" = "Darwin" ]; then sz=$(stat -f%z "$2"); else sz=$(stat -c%s "$2"); fi
        if [ "$sz" -ge $(($3 * 1048576)) ]; then pass "$1" "$2 [$((sz/1048576)) MB]"; else fail "$1" "too small: $sz bytes"; fi
    else
        fail "$1" "MISSING $2"
    fi
}
need_dir() { # name path (snapshot dir OR hf-cache layout)
    d="$2"
    if [ -d "$d" ] && { ls "$d"/*.safetensors.index.json >/dev/null 2>&1 || ls "$d"/model-*.safetensors >/dev/null 2>&1 || ls "$d"/*.safetensors >/dev/null 2>&1 || { [ -d "$d/blobs" ] && { [ -d "$d/snapshots" ] || [ -d "$d/refs" ]; }; }; } \
        && { [ -f "$d/config.json" ] || [ -d "$d/blobs" ]; }; then
        n=$(ls "$d"/model-*.safetensors "$d"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
        pass "$1" "$d [${n:-cache} shards]"
    else
        fail "$1" "MISSING index/shards/config under $d"
    fi
}
R="$REPO"
need_file "weights gpa" "$R/pretrained/detectors/gpa/model.pt" 10
need_file "weights omniparser_v3" "$R/pretrained/detectors/omniparser_v3/icon_detect_v3/model.pt" 100
need_file "weights uitag" "$R/pretrained/detectors/uitag/yolo-ui.pt" 5
need_file "weights omni icon_detect" "$R/pretrained/omniparser/weights/icon_detect/model.pt" 10
need_file "weights omni caption" "$R/pretrained/omniparser/weights/icon_caption_florence/model.safetensors" 100
need_dir "weights Qwen2.5-VL-3B" "$ASSETS_DIR/$QWEN_DIR"
need_dir "weights GUI-Actor-3B" "$ASSETS_DIR/$GUIACTOR_DIR"
if grep -q "pointer_pad_token_id" "$ASSETS_DIR/$GUIACTOR_DIR/config.json" 2>/dev/null; then
    pass "guiactor pointer ids" "present in config.json"
else
    fail "guiactor pointer ids" "not found"
fi

if [ "$QUICK" -eq 0 ]; then
    if python3 - "$R" <<'EOF' 2>/dev/null; then
import sys
from ultralytics import YOLO
from PIL import Image
import numpy as np
m = YOLO(sys.argv[1] + "/pretrained/detectors/gpa/model.pt")
img = Image.fromarray(np.full((720, 1280, 3), 200, dtype=np.uint8))
r = m.predict(img, conf=0.05, verbose=False, device="cpu")[0]
print(f"YOLO GPA smoke: {len(r.boxes) if r.boxes is not None else 0} boxes")
EOF
        pass "YOLO GPA smoke (cpu)" "see count above"
    else
        fail "YOLO GPA smoke (cpu)" "load/infer failed"
    fi
fi

# ---------------- VERDICT ----------------
TOTAL=$((PASS + FAIL))
echo ""
echo "$PASS/$TOTAL checks passed."
if [ "$FAIL" -gt 0 ]; then
    echo "BLOCKERS:$FAIL_LIST"
    echo "NOT READY."
    exit 1
fi
echo "READY for full inference."
