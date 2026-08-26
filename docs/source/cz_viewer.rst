Genome Browser
==============

``cz_viewer.html`` is a lightweight, IGV-style genome browser that visualizes
remote ``.cz`` methylation files **directly in the browser** using HTTP Range
requests — no server-side code required. It loads a shared *reference* ``.cz``
(genomic coordinates) plus one or more value ``.cz`` tracks (``mc``/``cov``),
joins them by row index, and renders per-cytosine methylation over any locus.
Multiple tracks can share a single reference.

.. raw:: html

   <p>
     <a class="reference external" href="_static/cz_viewer.html" target="_blank"
        style="display:inline-block;padding:8px 16px;margin:4px 0;background:#2980b9;
               color:#fff;border-radius:4px;text-decoration:none;font-weight:600;">
       &#8599; Open the genome browser full-screen
     </a>
   </p>
   <iframe src="_static/cz_viewer.html" title="cytozip genome browser"
           style="width:100%;height:80vh;border:1px solid #ccc;border-radius:6px;">
   </iframe>

Usage
-----

Quick start
~~~~~~~~~~~

1. Enter the **reference** ``.cz`` URL (the coordinate file with a ``pos``
   sort column) and click **Load reference** (only needed for ``.cz``
   methylation tracks).
2. Add tracks in any of these ways:

   - **+ Add track** — paste a single track URL (``.cz`` / ``.bw`` /
     ``.bedgraph.gz`` / ``.bed.gz`` / ``.bedpe.gz`` / ``.hic``); the format is
     detected from the extension. Give it an optional *modality* label.
   - **Track table** — load a CSV/TSV that lists many tracks at once (see
     `Track tables`_). This is the fastest way to build a multi-track,
     multi-modality view.

3. Type a locus (e.g. ``chr1:3,000,000-3,050,000``) and press **Go**, or pick a
   **Genome** and search a gene name. Drag to pan and scroll to zoom (in *Zoom*
   mode). All tracks zoom together.
4. Use the **cz** and **hic** sub-toolbars (next to *Save session*) to change
   the cytosine context (CG / CH / CHG / CHH / all), value (mC / 1 − mC),
   strand, and per-format resolution for every track at once.

Track tables
~~~~~~~~~~~~~

A track table loads many tracks at once from a CSV/TSV file or URL. The **first
row must be a header** naming the columns; a table without a recognizable
header is rejected.

.. list-table::
   :header-rows: 1
   :widths: 14 12 74

   * - Column
     - Required
     - Meaning
   * - ``name``
     - yes
     - Track label (shown in the header). Rows sharing a name are grouped
       together.
   * - ``path``
     - yes
     - Track URL (aka ``url``). The format is detected from the extension.
   * - ``group``
     - no
     - Middle grouping level; a group header is drawn above its tracks.
   * - ``color``
     - no
     - Track color (``#hex`` or CSS name). **Missing → a random color** is
       assigned.
   * - ``category``
     - no
     - Top grouping level. **Present → a three-level layout**
       (category → group → track); **absent → two levels** (group → track).
   * - ``modality``
     - no
     - Assay label (e.g. ``DNAm``, ``ATAC``, ``H3K27ac``, ``RNA``, ``HiC``),
       drawn as a vertical strip on the **right** of each track header.

After loading, rows are ordered **category → group → name → modality** (any
absent column is skipped). When a ``modality`` column is present, drag the
chips in the **Modality order** box (right of the *Load* button) to reorder the
modalities; the tracks re-sort instantly.

Tracks load **lazily**: the ones visible in the viewport open first and the
rest open as you scroll, so even a large table appears immediately. An example
multi-modality table is preloaded in the URL box.

Supported remote file formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All formats are read **remotely with HTTP Range requests only** — the whole
file is never downloaded. Add any of them via **+ Add track**, a **track
table**, or by restoring a saved session.

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Format
     - Extension
     - Notes
   * - cytozip methylation
     - ``.cz``
     - ``mc``/``cov`` value track joined to the shared **reference** ``.cz`` by
       row index. Requires a reference to be loaded first. Context
       (CG/CH/CHG/CHH/all, default **CG**), value (mC or 1 − mC) and strand are
       set globally in the **cz** sub-toolbar.
   * - BigWig
     - ``.bw`` / ``.bigwig``
     - Self-contained signal track (bar/wiggle). No reference needed; binned
       with the same resolution as ``.cz`` tracks.
   * - bedGraph
     - ``.bedgraph.gz`` / ``.bdg.gz`` / ``.bg.gz`` (+ ``.tbi``)
     - Quantitative signal (``chrom start end value``) drawn like BigWig. Must
       be **bgzip-compressed and tabix-indexed**.
   * - BED
     - ``.bed.gz`` (+ ``.tbi``)
     - Interval/annotation track drawn as row-packed boxes with labels. Must be
       **bgzip-compressed and tabix-indexed** so it can be read by range; the
       ``.tbi`` index is expected next to the file (``<url>.tbi``).
   * - BEDPE
     - ``.bedpe.gz`` (+ ``.tbi``)
     - Paired-end / loop track drawn as arcs connecting the two anchors. Also
       requires bgzip + tabix (``<url>.tbi``).
   * - Hi-C
     - ``.hic``
     - Juicebox contact matrix drawn as a pyramid heatmap. Resolution and
       **normalization** (NONE / KR / SCALE) are set in the **hic** sub-toolbar;
       the color scale follows the Hi-C track's Y-limits.

.. note::

   bedGraph / BED / BEDPE must be **bgzipped and tabix-indexed** (e.g.
   ``bgzip file.bed && tabix -p bed file.bed.gz``) — plain uncompressed files
   are not accepted because they cannot be read partially by range.

Toolbar tools and keyboard shortcuts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The toolbar above the tracks selects the active mouse tool. Each tool has a
one-letter shortcut (shown in parentheses in its tooltip):

.. list-table::
   :header-rows: 1
   :widths: 14 10 76

   * - Tool
     - Key
     - Action
   * - Pan
     - ``P``
     - Drag left/right on a track to move along the genome.
   * - Zoom
     - ``Z``
     - Drag to select a region and zoom into it; the mouse **wheel zooms only
       in this mode** (other modes let the page scroll). All tracks zoom
       together.
   * - Select
     - ``V``
     - Click a track (its left label **or** the plot) to select it; Shift adds a
       range, Ctrl/⌘ toggles one.
   * - Mark
     - ``M``
     - Drag to highlight a genomic region with the chosen color (swatch next to
       the tool).
   * - Save session
     - ``S``
     - Download the current view + tracks as a JSON session.
   * - Clear marks
     - ``C``
     - Remove all region marks.

Alongside the tools, the toolbar also has:

- **Select** ``All`` / ``None`` and a **by…** dropdown to batch-select every
  track sharing a given *modality*, *category*, *group*, or *name*. Any of these
  switches to the Select tool automatically.
- A **cz** sub-toolbar (methylation): global *context* (CG / CH / CHG / CHH /
  all), *value* (mC / 1 − mC), *strand*, and *resolution* for all ``.cz`` and
  BigWig tracks.
- A **hic** sub-toolbar: Hi-C bin *resolution* (Auto or a fixed bin size) and
  *normalization* (NONE / KR / SCALE). It appears only when a Hi-C track is
  loaded, and the two resolutions are independent of the cz resolution.
- A **Name pt** box to change the track-name font size (type a value or use the
  up/down arrows).
- **Saved tracks** — a manager listing every track loaded **this session**;
  re-add removed tracks, or toggle whole *modality / category / group* levels
  on/off.

A dashed vertical guide line follows the cursor across **all** tracks to make
comparing positions easy.

Working with tracks
~~~~~~~~~~~~~~~~~~~~~

- **Rename** a track by **double-clicking** its name; press Enter to confirm.
- **Reorder** tracks by dragging the grip handle (``⠿``) on the left up/down, or
  select some and use the **↑ Up / ↓ Down** buttons in the selection bar. With
  categories, selecting a whole *category* (or *group*) header moves it as a
  block.
- **Resize** a track's height: select it (or several) with the Select tool and
  type a pixel height into the selection bar's **Height** box.
- **Recolor** a track with its color swatch.
- **Per-track Y-limits**: click the ``y:auto`` button in a track header to set
  *Auto* or a fixed ``[min, max]`` range. For ``.cz``/BigWig this rescales the
  y-axis; for Hi-C it sets the contact **color scale**. The header meta shows
  the min/max (the fixed limits when set, otherwise the real data range) and the
  active resolution.
- **Global context / value / strand** are set once for all ``.cz`` tracks in the
  **cz** sub-toolbar; select a subset first to change only those (selection bar
  ``Ctx`` / ``Val``).
- **Resolution** bins signal for speed. In the **cz** sub-toolbar choose
  **Auto** to let the viewer pick a bin size from the current span (large views
  bin coarsely, zoomed-in views go per-base), or a fixed size. Hi-C resolution
  is set separately in the **hic** sub-toolbar. The active resolution is shown
  on the left of the ruler and in each track header.

Genome, gene search, and marks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Pick a **Genome** (hg38/hg19/mm10/mm39) to auto-add its reference sequence
  (``.2bit``) and gene-annotation tracks; then search a **gene name** (e.g.
  ``HOXA13``) or a coordinate in the locus box.
- Use the **Mark** tool to highlight regions of interest; marks persist across
  pan/zoom and are **included in exported images**.

Exporting
~~~~~~~~~

- **TSV** exports every loaded methylation track's per-site values in the
  current window as a matrix (``chrom``, ``pos``, ``context``, one column per
  track).
- **SVG / PNG / PDF** export the current view as a figure — including any region
  **marks** you have drawn.

Session and sharing
~~~~~~~~~~~~~~~~~~~~

The full view (genome, locus, tracks with their group / category / modality /
color / Y-limits, context, value, strand, cz + Hi-C resolution and Hi-C
normalization) is encoded in the page URL as you work, so copying the URL shares
the exact view. Use **Save** (shortcut ``S``) to download a JSON session and the
session file input to restore it later.

.. note::

   The **Saved tracks** manager is per-session only (kept in memory): it lists
   the tracks loaded in the current tab and is not persisted across reloads.
   Use **Save** / the session URL to keep a view.
   
Example session: https://dingwb.github.io/cytozip/build/html/_static/cz_viewer.html?session=https://neomorph.salk.edu/ftp/cz/SPN.json

Share a session via a remote JSON URL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of pasting a long encoded URL, you can host the session file anywhere
and hand out a short link. The viewer will fetch that JSON on load and restore
the whole session automatically — an easy way to publish a ready-made view
(like your own mini genome-browser site).

1. Build the view you want (reference, tracks, locus, colors, Y-limits, marks…).
2. Click **Save** (shortcut ``S``) to download ``cytozip_session.json``.
3. Upload that JSON to any web/FTP server that serves it over **HTTPS** with
   CORS enabled (the same requirement as the data files).
4. Share the viewer link with the file's URL appended as the ``session``
   parameter:

   .. code-block:: text

      https://your-host/path/cz_viewer.html?session=https://your-host/ftp/my_session.json

   ``?s=`` and ``#session=`` are accepted as aliases. Anyone opening that link
   gets the exact saved view.

.. note::

   The session JSON is fetched over HTTP(S) and is subject to the same
   CORS + HTTP Range requirements as the data files. If the JSON host has no
   CORS headers, the viewer falls back to the configured CORS proxy.

.. note::

   Remote reading works only when the data server sends CORS headers
   (``Access-Control-Allow-Origin``) over **HTTPS**, and supports HTTP Range
   requests. NCBI GEO (``ftp.ncbi.nlm.nih.gov``) is CORS-enabled and works out
   of the box. For servers **without** CORS, host this page on the same origin
   as the data files, or run the bundled ``docs/serve_viewer.py`` proxy and
   set the page's *CORS proxy prefix* field to ``/proxy?url=``.
