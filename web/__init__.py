"""The browser front-end: a second renderer for the same engine the CLI drives.

Nothing in here measures anything. It starts analyses on a shared job runner,
serves their progress to a polling browser, and serves the saved reports.

* ``responses`` the API envelope and the engine result tuple
* ``engine``    the decisions, without Flask (params dict in, result tuple out)
* ``api``       JSON endpoints over the engine
* ``server``    the application factory, the pages and the request guards
"""

from __future__ import annotations

from web.server import DEFAULT_HOST, DEFAULT_PORT, create_app, serve

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "create_app", "serve"]
