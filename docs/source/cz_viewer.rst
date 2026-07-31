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
   sort column) and click **Load reference**.
2. Click **+ Add cz track**, giving the URL of a value ``.cz`` file
   (``mc``/``cov``) that was built against the same reference. Repeat to stack
   multiple tracks.
3. Type a locus (e.g. ``chr1:3,000,000-3,050,000``) and press **Go**. Drag to
   pan, scroll to zoom (in *Zoom* mode), and switch the per-track cytosine
   context (CG / CH / CHG / CHH / all).

Supported remote file formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All formats are read **remotely with HTTP Range requests only** — the whole
file is never downloaded. Add any of them by pasting its URL into
**New track URL** (the format is detected from the extension), via the
**Track table**, or by restoring a saved session.

.. list-table::
   :header-rows: 1
   :widths: 16 20 64

   * - Format
     - Extension
     - Notes
   * - cytozip methylation
     - ``.cz``
     - ``mc``/``cov`` value track joined to the shared **reference** ``.cz`` by
       row index. Requires a reference to be loaded first. Per-track context
       (CG/CH/CHG/CHH/all) and value (mC or 1 − mC).
   * - BigWig
     - ``.bw`` / ``.bigwig``
     - Self-contained signal track (bar/wiggle). No reference needed.
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
     - Juicebox contact matrix drawn as a pyramid heatmap; the resolution is
       chosen automatically from the view span and KR normalization is used
       when available.

.. note::

   BED / BEDPE must be **bgzipped and tabix-indexed** (``bgzip file.bed &&
   tabix -p bed file.bed.gz``) — plain uncompressed ``.bed``/``.bedpe`` are not
   accepted because they cannot be read partially by range.

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
       in this mode** (other modes let the page scroll).
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

A dashed vertical guide line follows the cursor across **all** tracks to make
comparing positions easy.

Working with tracks
~~~~~~~~~~~~~~~~~~~~~

- **Reorder** tracks by dragging the grip handle (``⠿``) on the left up/down, or
  select some and use the **↑ Up / ↓ Down** buttons in the selection bar.
- **Resize** a track's height by dragging its bottom edge.
- **Recolor** a track with its color swatch.
- **Per-track Y-limits**: click the ``y:auto`` button to set *Auto* or a fixed
  ``[min, max]`` range.
- **Select multiple** tracks (Select tool, Shift/Ctrl) to batch-change context,
  value (mC / 1 − mC), color, height, Y-limits, or to delete/move them together.
- **Resolution** bins methylation for speed; choose **Auto** to let the viewer
  pick a bin size from the current span (large views bin coarsely, zoomed-in
  views go per-base).

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

The full view (genome, locus, tracks, colors, Y-limits, resolution) is encoded
in the page URL as you work, so copying the URL shares the exact view. Use
**Save** (shortcut ``S``) to download a JSON session and the session file input
to restore it later.

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
