from __future__ import annotations

from simajilord.diagnostics.secrets import findings


def test_tracked_project_files_contain_no_credential_shaped_values() -> None:
    assert findings() == ()
