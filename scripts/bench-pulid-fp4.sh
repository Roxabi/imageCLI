#!/usr/bin/env bash
# bench-pulid-fp4.sh — run BF16 vs FP4 PuLID comparison, capture timing + VRAM
# Usage: bash scripts/bench-pulid-fp4.sh
# Output images → ~/.roxabi/forge/imagecli/pulid-fp4-bench/images/

set -euo pipefail

OUTDIR="$HOME/.roxabi/forge/imagecli/pulid-fp4-bench/images"
PROMPTS="images/prompts_in/pulid-fp4-bench"
mkdir -p "$OUTDIR"

run_bench() {
    local label="$1"
    local md="$2"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local start
    start=$(date +%s%N)
    uv run imagecli generate "$md" --output-dir "$OUTDIR" 2>&1 | \
        grep -E "Saved:|Peak VRAM|it/s|Engine:|Size:|Seed:" || true
    local end
    end=$(date +%s%N)
    local wall_ms=$(( (end - start) / 1000000 ))
    echo "  Wall-clock: ${wall_ms}ms  ($(( wall_ms / 1000 ))s)"
}

echo ""
echo "▶ PuLID BF16 vs FP4 benchmark — 1024×1024, seed 42 + 99"
echo "  Face ref: 006-frontal-dreamy-distant.png"
echo "  Output:   $OUTDIR"
echo ""

# ── Prompt 1: cinematic ──────────────────────────────────────────────────────
run_bench "cinematic — BF16 (pulid-flux2-klein)"     "$PROMPTS/cinematic-bf16.md"
run_bench "cinematic — FP4  (pulid-flux2-klein-fp4)" "$PROMPTS/cinematic-fp4.md"

# ── Prompt 2: portrait ───────────────────────────────────────────────────────
run_bench "portrait  — BF16 (pulid-flux2-klein)"     "$PROMPTS/portrait-bf16.md"
run_bench "portrait  — FP4  (pulid-flux2-klein-fp4)" "$PROMPTS/portrait-fp4.md"

echo ""
echo "✓ Done — images saved to $OUTDIR"
echo "  Run: open $OUTDIR  OR  imagecli (in session) → continue → gallery"
