"""Analysis: the saved model-analysis workflow reports.

The same browser as Reports, filtered to the workflow's ``assessment`` reports (the per-tool
reports stay under Reports). Run a new analysis from the Workflow section of the home menu.
"""

from __future__ import annotations

from cli.common import Session
from cli.flows.reports import ASSESSMENT, browse

TITLE = "Analysis"


def run_analysis(session: Session) -> None:
    """Browse, open and delete the saved model analyses."""
    browse(
        session,
        title=TITLE,
        keep=lambda record: record.analysis == ASSESSMENT,
        empty="No analysis saved yet in {root}: run one from the Workflow section of the home menu.",
    )
