"""The novomodelo-compat policy: the minimum novomodelo-python version gate."""

from __future__ import annotations

#: The novomodelo / novomodelo-python release the bridge pairs with. ``pyproject.toml``
#: pins ``novomodelo-python`` to exactly this version, and the two change together.
#: The pin is exact because novomodelo loads a policy checkpoint only in the software
#: and version that wrote it, and the checkpoint writer stamps the installed
#: novomodelo-python's own version, so the DECOMP terminal boundary the bridge writes
#: loads only in the matching novomodelo release.
#:
#: The manifest records it (single source of truth), and the ``--validate``
#: gate treats it as the minimum an installed novomodelo-python must reach to
#: validate the output.
MIN_NOVOMODELO_VERSION = "0.18.0"


def _installed_novomodelo_python_version() -> str | None:
    """Return the installed ``novomodelo-python`` distribution version, or ``None``.

    The package imports as ``novomodelo`` but is distributed as ``novomodelo-python``;
    this reads the distribution metadata. Returns ``None`` when it is not
    installed, so the caller falls through to the generic "not installed" skip.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        return _dist_version("novomodelo-python")
    except PackageNotFoundError:
        return None


def _novomodelo_python_supports_output(installed: str) -> bool:
    """Whether an installed novomodelo-python *version* can load the bridge's output.

    ``True`` when *installed* is at least :data:`MIN_NOVOMODELO_VERSION` by a numeric
    release-segment comparison (so ``"0.10.0"`` and ``"0.11.2"`` qualify,
    ``"0.9.1"`` does not). A non-numeric pre-release suffix on a segment is
    ignored (``"0.10.0rc1"`` reads as ``0.10.0``); the gate only guards against an
    obviously-older install, so the leniency is deliberate.
    """

    def _release(value: str) -> tuple[int, ...]:
        parts: list[int] = []
        for segment in value.split("."):
            digits = ""
            for char in segment:
                if not char.isdigit():
                    break
                digits += char
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    return _release(installed) >= _release(MIN_NOVOMODELO_VERSION)
