"""Reports: self-contained HTML and JSON renderings of analysis results, saved on disk.

* :mod:`reporting.html`   renders the JSON form of a result as one static HTML page
* :mod:`reporting.store`  saves, lists, serves and deletes report folders
"""

from reporting.html import render_html, render_result
from reporting.store import InvalidReportId, ReportNotFound, ReportRecord, ReportStore, summary

__all__ = [
    "InvalidReportId",
    "ReportNotFound",
    "ReportRecord",
    "ReportStore",
    "render_html",
    "render_result",
    "summary",
]
