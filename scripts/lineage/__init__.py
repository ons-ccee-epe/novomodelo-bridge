"""Data-map tooling: the authored lineage under ``docs/lineage/`` rendered to
``docs/<track>-data-map.md`` and cross-checked against the code.

``model`` loads and validates a track's TOML, ``trace`` derives file-level
lineage from the pipeline's AST, ``render`` writes the pt-BR page. The entry
point is ``scripts/gen-lineage-docs.py``; ``tests/test_lineage.py`` runs the
same checks plus the emission-coverage gate that needs a real conversion.
"""
