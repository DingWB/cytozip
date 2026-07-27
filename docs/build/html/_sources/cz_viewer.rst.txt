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

1. Enter the **reference** ``.cz`` URL (the coordinate file with a ``pos``
   sort column) and click **Load reference**.
2. Click **+ Add track**, giving the URL of a value ``.cz`` file (``mc``/``cov``)
   that was built against the same reference. Repeat to stack multiple tracks.
3. Type a locus (e.g. ``chr1:3,000,000-3,050,000``) and press **Go**. Drag to
   pan, scroll to zoom, and switch the per-track cytosine context
   (CG / CH / CHG / CHH / all).

.. note::

   Remote reading works only when the data server sends CORS headers
   (``Access-Control-Allow-Origin``) over **HTTPS**, and supports HTTP Range
   requests. NCBI GEO (``ftp.ncbi.nlm.nih.gov``) is CORS-enabled and works out
   of the box. For servers **without** CORS, host this page on the same origin
   as the ``.cz`` files, or run the bundled ``docs/serve_viewer.py`` proxy and
   set the page's *CORS proxy prefix* field to ``/proxy?url=``.
