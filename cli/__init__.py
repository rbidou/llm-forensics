"""The interactive terminal front-end.

Split so the parts stay testable in isolation:

* ``ui``         terminal input and output: the four-part layout and the progress bar
* ``completer``  path and cached-model completion as you type
* ``common``     the session, the step runner, prompts, report saving and formatting
* ``running``    analyses as background jobs, with progress and Ctrl+C cancellation
* ``selection``  the model selection step shared by every flow
* ``flows``      one module per menu entry
* ``app``        the home menu and the main loop

Everything here is presentation. The engine decides nothing about how it is
displayed, and this package decides nothing about how a model is analysed.
"""

from __future__ import annotations

from cli.app import run_cli

__all__ = ["run_cli"]
