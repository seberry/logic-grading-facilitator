# Fitch Debug Pipeline Trace — In-Progress Notes

**Date:** 2026-04-23  
**Status:** ~90% complete, one remaining issue

## What was done

1. **`app/routes/fitch_debug.py`** — Added `scan_pipeline_artifacts()` function that scans a scan's debug directory for VLM pipeline artifacts:
   - Finds `crop_*.jpg` / `crop_*.png` images and their paired `response_*.json` files
   - Finds `debug_bars.jpg` (bar detection output)
   - Returns a dict with `nodes` list and optional `bar_image` URL
   - Wired into the `fitch_debug_page` handler as `pipeline_artifacts` template variable

2. **`app/templates/debug_fitch.html`** — Added a collapsible `<details>` section at the bottom of the page:
   - Shows "Pipeline Trace — VLM crops & responses (N nodes)" summary
   - Displays bar detection image if `debug_bars.jpg` exists
   - Shows a responsive grid of crop images, each with a collapsible VLM response viewer
   - Only renders when a scan is selected AND pipeline artifacts exist

3. **`data/debug_fitch/tfl_handwritten_q6_S_iff_W/`** — Copied Q6 artifacts from `working/tfl_live_ocr_test/` for viewing in the debug page.

## Remaining work

### Static file serving for `/debug_fitch_assets/` (BLOCKING)

The template generates `<img src="/debug_fitch_assets/{scan_slug}/crop_ROOT.jpg">` URLs, but there is **no route or mount** in the FastAPI app that serves these files. The app imports `StaticFiles` in `server.py` but we couldn't find any existing `.mount()` calls to use as a pattern reference.

**Next step:** Find how the existing `/scans/` URL prefix serves images (the `scan_image_url()` function in `fitch_debug.py` returns `/scans/{name}` URLs). Likely either:
- A `StaticFiles` mount in `server.py` we missed (search for `mount` more broadly)
- A dynamic route that returns `FileResponse`  
- Or the `/scans/` prefix may be served by a reverse proxy / nginx in production

Then add a similar mount/route for `/debug_fitch_assets/` → `DEBUG_FITCH_DIR`.

### How to find the scan serving pattern

```bash
# Search for mount or file response patterns
grep -rn "mount\|FileResponse\|/scans" app/
grep -rn "StaticFiles" app/
```

Check `app/server.py` lines 80-200 for app setup, route registration, and static file mounts.