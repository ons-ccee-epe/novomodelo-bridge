"""The ``dger.dat`` switches that gate optional inputs the converter reads.

NEWAVE consults a switch line before it reads several optional inputs, so a
file that is present but switched off plays no part in its run. The converter
mirrors that, so the converted case models what NEWAVE actually ran, and
reports every input it leaves out for that reason. A switch line absent from
``dger.dat`` counts as on, which keeps older decks converting as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

if TYPE_CHECKING:
    from inewave.newave import Dger

    from novomodelo_bridge.newave.files import NewaveFiles

SWITCH_OFF_CODE = "dger-switch-off"
_CATEGORY = "Input switches"


@dataclass(frozen=True)
class Switch:
    """One ``dger.dat`` switch and the input it gates.

    ``label`` is the line label as NEWAVE prints it, ``value`` the raw integer
    (``None`` when the line is absent), ``on`` whether ``ignored`` is
    considered, and ``ignored`` names the input in user-facing text.
    """

    field: str
    label: str
    value: int | None
    on: bool
    ignored: str


@dataclass(frozen=True)
class DgerSwitches:
    agrint: Switch
    c_adic: Switch
    ghmin: Switch
    re_dat: Switch
    restricao_eletrica: Switch
    min_outflow: Switch
    turbined_max: Switch
    turbined_min: Switch

    @classmethod
    def from_dger(cls, dger: Dger) -> DgerSwitches:
        def read(field: str) -> int | None:
            value = getattr(dger, field, None)
            # An absent line reads as None, and so does anything that is not
            # a number (a blank field, a test double): the input is considered.
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            return int(value)

        def plain(field: str, label: str, ignored: str) -> Switch:
            value = read(field)
            return Switch(field, label, value, value is None or value != 0, ignored)

        turbining = read("restricao_turbinamento")
        drop_min_outflow = read("desconsidera_vazao_minima")
        return cls(
            agrint=plain("agrupamento_livre", "AGRUPAMENTO LIVRE", "agrint.dat"),
            c_adic=plain(
                "considera_carga_adicional", "CONS. CARGA ADICIONAL", "c_adic.dat"
            ),
            ghmin=plain("considera_ghmin", "CONSIDERA GHMIN", "ghmin.dat"),
            re_dat=plain("restricoes_eletricas", "RESTRICOES ELETRICAS", "re.dat"),
            restricao_eletrica=plain(
                "restricoes_eletricas_especiais",
                "RESTRICOES ELETRICAS ESPECIAIS",
                "restricao-eletrica.csv",
            ),
            # Inverted: 1 tells NEWAVE to drop every minimum-outflow requirement.
            min_outflow=Switch(
                "desconsidera_vazao_minima",
                "DESCONSIDERA VAZMIN",
                drop_min_outflow,
                drop_min_outflow is None or drop_min_outflow == 0,
                "minimum outflow (VAZMIN, VAZMINT)",
            ),
            # 0 none, 1 maximum and minimum, 2 maximum only, 3 minimum only.
            turbined_max=Switch(
                "restricao_turbinamento",
                "REST. TURBINAMENTO",
                turbining,
                turbining is None or turbining in (1, 2),
                "TURBMAXT records of modif.dat",
            ),
            turbined_min=Switch(
                "restricao_turbinamento",
                "REST. TURBINAMENTO",
                turbining,
                turbining is None or turbining in (1, 3),
                "TURBMINT records of modif.dat",
            ),
        )


def switch_off_diagnostic(switch: Switch) -> Diagnostic:
    """The INFO diagnostic for an input the deck carries but *switch* turns off.

    INFO, not WARNING: NEWAVE's own configuration is being followed, so the
    verdict never turns on it. Emitted wherever the input would have been read,
    with one summary per switch so the sink de-duplicates repeats.
    """
    return Diagnostic(
        code=SWITCH_OFF_CODE,
        severity=Severity.INFO,
        category=_CATEGORY,
        title=f"{switch.ignored} ignored: {switch.label} = {switch.value} in dger.dat",
        summary=(
            f"NEWAVE does not consider {switch.ignored} with {switch.label} = "
            f"{switch.value}, so the converter leaves it out as well."
        ),
        notes=[f"switch: {switch.field}"],
        remediation=(
            f"Change {switch.label} in dger.dat if {switch.ignored} should be "
            "considered; otherwise this is informational."
        ),
    )


def switched_off_inputs(
    files: NewaveFiles,
    switches: DgerSwitches,
    *,
    restricao_eletrica_present: bool,
) -> list[Switch]:
    """The switches that are off while the input they gate is present.

    Minimum outflow needs no file (``hidr.dat`` carries it for every plant);
    the two turbining switches only matter when ``modif.dat`` exists, since
    they gate its dated records.
    """
    present = {
        "agrint": files.agrint is not None,
        "c_adic": files.c_adic is not None,
        "ghmin": files.ghmin is not None,
        "re_dat": files.re_dat is not None,
        "restricao_eletrica": restricao_eletrica_present,
        "min_outflow": True,
        "turbined_max": files.modif is not None,
        "turbined_min": files.modif is not None,
    }
    return [
        getattr(switches, name)
        for name, is_present in present.items()
        if is_present and not getattr(switches, name).on
    ]
