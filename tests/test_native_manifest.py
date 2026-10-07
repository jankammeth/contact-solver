from contact_solver.cli.dimension_campaign import source_manifest
from contact_solver.solvers import collision_search


def test_dimension_manifest_tracks_native_source_and_build():
    sources = source_manifest()["sources"]
    assert "solvers/_cubic_minima.c" in sources
    assert "setup.py" in sources
    assert "pyproject.toml" in sources
    if collision_search._cubic_minima is not None:
        from pathlib import Path

        binary = Path(collision_search._cubic_minima.__file__).name
        assert "solvers/" + binary in sources
