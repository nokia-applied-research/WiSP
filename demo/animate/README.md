# WiSP · MV-WSA live KV↔Expert split — animated demo

A self-contained, dependency-free animation of the **MV-WSA dynamic-resize
loop**: on one fixed GPU budget (iso-VRAM), the live controller observes that the
KV pool is mostly idle, **reclaims those bytes into resident experts** (cap 32→35,
KV 2.79→1.53 GiB), and serves the same agent trace **1.19× faster** — with zero
preemptions and byte-identical output.

Numbers are the measured `db` trace from the paper (real 24 GiB RTX 3090, vLLM
0.11.2, Qwen3-30B-A3B, AgentInstruct, concurrency 4, 36 timed turns): end-to-end
1355→1138 s (1.19×), TTFT 95→75 s (1.27×). The companion `os` trace gives 1.07×.

> Note: this animation illustrates the MV-WSA **online dual-resize controller**,
> which is reported in the paper. The controller code is **not** part of this v1
> release (it ships with the conference version — see the repo Roadmap); v1 ships
> the expert pager + the static iso-VRAM and byte-identity reproductions.

## Files
- **`mvwsa.svg`** — looping animated SVG (SMIL, no JS). Drop straight into a README.
- **`index.html`** — richer interactive version (open in any browser; good for
  GitHub Pages or for screen-recording a GIF).

## Put it on the repo homepage (README)

**1) Inline auto-playing animation (recommended).** Paste this into your repo's
root `README.md`. Use an `<img>` tag (not `![]()`), because GitHub only honors
`width` on HTML tags:

```html
<p align="center">
  <img src="demo/animate/mvwsa.svg" alt="WiSP MV-WSA live KV/Expert split" width="840">
</p>
```

The SVG **auto-plays and loops** on GitHub — its animation is pure SMIL (no
JavaScript), which GitHub renders even though it strips scripts. The path is
relative to wherever your `README.md` lives (this repo's root `README.md` uses
`demo/animate/mvwsa.svg`; if the README is *inside* `demo/animate/`, use
`src="mvwsa.svg"`).

**2) Make it clickable → opens the full interactive demo.** Wrap the image in a
link to the `index.html` served by GitHub Pages:

```html
<p align="center">
  <a href="https://USER.github.io/REPO/demo/animate/">
    <img src="demo/animate/mvwsa.svg" alt="WiSP MV-WSA live KV/Expert split" width="840">
  </a>
</p>
```

Enable Pages once: **repo Settings → Pages → Build and deployment → Deploy from a
branch → `main` / `(root)` → Save**. After it builds, the interactive
`index.html` is live at that URL (replace `USER`/`REPO`). Now the homepage shows
the looping animation, and clicking it opens the slower, interactive walkthrough.

> Note: GitHub caches images via its `camo` proxy, so a freshly pushed/updated
> SVG can take a minute (or a hard refresh) to show the latest version.

## View the interactive version locally
```bash
open demo/animate/index.html      # macOS
```

## GIF fallback (only if you want a guaranteed raster, e.g. for non-GitHub sites)
The animated SVG is preferred (crisp, tiny, scalable). If you specifically need a
GIF/MP4, open `index.html`, then screen-record the panel for one loop (~35 s),
or capture it headless with Puppeteer/Playwright. Ask and I can wire up a
one-command capture script.

## What the animation shows (left→right, top→bottom)
1. **Budget bar (iso-VRAM):** one fixed total, split into *Resident experts* (blue)
   and *KV cache pool* (green box). The green *used* fill is the **actual** KV
   working set.
2. **Static split:** the pool is sized for the worst case, so ~56% is idle while
   experts are capped at 32.
3. **Controller observes** the real KV peak (~44% of the pool).
4. **Live resize:** the boundary slides right — KV pool shrinks around its
   (unchanged) working set, the freed bytes become **+3 resident experts**.
5. **Race:** same trace, same budget — Live MV-WSA finishes the 36-turn agent
   session in 1138 s vs 1355 s for the static split (**1.19×**, TTFT 95→75 s).
