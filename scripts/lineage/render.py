"""Render a track's lineage to the pt-BR data-map page.

Layout, in order: a short introduction and legend; one dependency matrix per
output directory (deck files as rows, generated files as columns); one section
per generated file with its field table; an index by deck source that lists
what each file or register feeds and what in it is still unconverted; and a
closing backlog of everything not yet converted. Every part is derived from the
same :class:`~lineage.model.Lineage`, so the page cannot contradict itself.
"""

from __future__ import annotations

from collections import OrderedDict

from .model import (
    DIRECTORY_ORDER,
    Field,
    Lineage,
    Output,
    SourceRef,
    registry,
    used_items_by_token,
)

TRACK_NAME = {"newave": "NEWAVE", "decomp": "DECOMP"}
FIELD_STATUS_LABEL = {
    "derived": "derivado",
    "constant": "constante",
    "null": "sempre nulo",
}
SOURCE_STATUS_LABEL = {
    "deferred": "adiado",
    "unread": "não lido",
    "index": "arquivo de índice, lido para localizar os demais",
}
DIRECTORY_TITLE = {
    "": "Raiz do caso",
    "system": "`system/`",
    "scenarios": "`scenarios/`",
    "constraints": "`constraints/`",
    "boundary": "`boundary/`",
}

Token = tuple[str, str | None]

# Column and section order within a directory, shared by both tracks so the
# two pages line up; names not listed sort last, alphabetically.
CANONICAL_ORDER = (
    "config.json",
    "stages.json",
    "penalties.json",
    "initial_conditions.json",
    "post_study_stages.json",
    "buses.json",
    "lines.json",
    "thermals.json",
    "hydros.json",
    "hydro_production_models.json",
    "hydro_geometry.parquet",
    "hydro_energy_productivity.parquet",
    "tailrace_curves.parquet",
    "non_controllable_sources.json",
    "pumping_stations.json",
    "energy_contracts.json",
    "inflow_history.parquet",
    "inflow_seasonal_stats.parquet",
    "external_inflow_scenarios.parquet",
    "load_seasonal_stats.parquet",
    "load_factors.json",
    "external_load_scenarios.parquet",
    "non_controllable_stats.parquet",
    "non_controllable_factors.json",
    "external_ncs_scenarios.parquet",
    "hydro_bounds.parquet",
    "hydro_unit_group_bounds.parquet",
    "thermal_bounds.parquet",
    "line_bounds.parquet",
    "pumping_bounds.parquet",
    "contract_bounds.parquet",
    "penalty_overrides_bus.parquet",
    "penalty_overrides_hydro.parquet",
    "generic_constraints.json",
    "generic_constraint_bounds.parquet",
    "generic_parameters.json",
)


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _schema_url(path: str) -> str | None:
    from novomodelo_bridge.novomodelo.schemas import SCHEMA_URLS

    return SCHEMA_URLS.get(path)


class _Page:
    def __init__(self, lineage: Lineage) -> None:
        self.lineage = lineage
        self.track = lineage.track
        self.reg = registry(self.track)
        self.used = used_items_by_token(lineage)
        self.lines: list[str] = []

    # -- naming --------------------------------------------------------------

    def file_label(self, key: str) -> str:
        if key in self.reg:
            return self.reg[key].label
        for src in self.lineage.sources:
            if src.file == key:
                return src.label or key
        return key

    def token_label(self, token: Token, *, mark_optional: bool = False) -> str:
        file, register = token
        label = f"`{self.file_label(file)}`"
        if register:
            label += f" › `{register}`"
        if mark_optional and file in self.reg and self.reg[file].optional:
            label += " (opcional)"
        return label

    def ref_label(self, ref: SourceRef) -> str:
        label = self.token_label(ref.token)
        if ref.item:
            label += f" › `{ref.item}`"
        return label

    def reads_label(self, out: Output) -> str:
        groups: OrderedDict[str, list[str]] = OrderedDict()
        for ref in out.reads:
            groups.setdefault(ref.file, [])
            if ref.register:
                groups[ref.file].append(ref.register)
        parts = []
        for file, regs in groups.items():
            part = self.token_label((file, None), mark_optional=True)
            if regs:
                part += " (" + ", ".join(f"`{r}`" for r in regs) + ")"
            parts.append(part)
        return ", ".join(parts)

    # -- emit helpers --------------------------------------------------------

    def emit(self, *lines: str) -> None:
        self.lines.extend(lines)

    def table(self, header: list[str], rows: list[list[str]], center_from: int = 99):
        aligns = [":-:" if i >= center_from else "---" for i in range(len(header))]
        self.emit("| " + " | ".join(header) + " |", "| " + " | ".join(aligns) + " |")
        for row in rows:
            self.emit("| " + " | ".join(row) + " |")
        self.emit("")

    # -- sections ------------------------------------------------------------

    def intro(self) -> None:
        name = TRACK_NAME[self.track]
        command = f"convert {self.track}"
        self.emit(
            f"<!-- Gerado por scripts/gen-lineage-docs.py a partir de "
            f"docs/lineage/{self.track}.toml. Não edite à mão. -->",
            "",
            f"# Mapa de dados: {name} → Novomodelo",
            "",
            f"Esta página mostra de onde vem cada arquivo e cada campo do caso Novomodelo "
            f"que `{command}` escreve: qual arquivo do deck {name}, qual registro ou "
            f"coluna, e que transformação é aplicada no caminho. Ela também lista o "
            f"que o conversor ainda não converte, para que o trabalho pendente fique "
            f"visível no mesmo lugar.",
            "",
            f"O conteúdo é gerado a partir de `docs/lineage/{self.track}.toml` por "
            f"`scripts/gen-lineage-docs.py` e verificado pelos testes contra uma "
            f"conversão real do deck de exemplo do repositório, de modo que a página "
            f"não pode divergir do código sem quebrar a build.",
            "",
            "**Como ler.** Nas matrizes, ● indica que o arquivo gerado (coluna) "
            "depende do arquivo do deck (linha). *(opcional)* marca um arquivo que o "
            "deck pode não trazer; a conversão prossegue sem ele. Nas tabelas de "
            "campos, a coluna Origem cita arquivo › registro › coluna do deck e a "
            "coluna Transformação diz o que o conversor faz com o valor. *Derivado* "
            "marca um valor de escrituração (ids, datas, ordem) calculado a partir do "
            "deck; *constante* um valor fixo que o conversor sempre escreve; *sempre "
            "nulo* um campo do Novomodelo que ainda não recebe informação do deck.",
            "",
        )
        if self.track == "decomp":
            self.emit(
                "Salvo indicação, um registro de duas letras (`CT`, `UH`, `DP`, ...) "
                "é um registro do `dadger.rvN`.",
                "",
            )

    def outputs_in_order(self) -> list[Output]:
        order = {d: i for i, d in enumerate(DIRECTORY_ORDER)}
        names = {n: i for i, n in enumerate(CANONICAL_ORDER)}
        return sorted(
            self.lineage.outputs,
            key=lambda o: (order.get(o.directory, 99), names.get(o.name, 999), o.name),
        )

    def overview(self) -> None:
        self.emit("## Visão geral", "")
        self.emit(
            "Uma matriz por diretório do caso. As linhas são os arquivos do deck "
            "(no DECOMP, também os registros do `dadger.rvN`); as colunas, os "
            "arquivos gerados. As seções seguintes detalham cada coluna.",
            "",
        )
        outs = self.outputs_in_order()
        for directory in DIRECTORY_ORDER:
            cols = [o for o in outs if o.directory == directory]
            if not cols:
                continue
            self.emit(f"### {DIRECTORY_TITLE[directory]}", "")
            rows: list[list[str]] = []
            for token in self.row_tokens():
                marks = ["●" if token in self.read_tokens(o) else "" for o in cols]
                if any(marks):
                    rows.append([self.token_label(token, mark_optional=True), *marks])
            header = ["Arquivo do deck", *(f"`{o.name}`" for o in cols)]
            self.table(header, rows, center_from=1)

    def read_tokens(self, out: Output) -> set[Token]:
        return {ref.token for ref in out.reads}

    def row_tokens(self) -> list[Token]:
        """Row order: registry order for files; registers alphabetical within a file."""
        seen: OrderedDict[Token, None] = OrderedDict()
        for key in self.reg:
            regs = sorted(
                {
                    ref.register
                    for out in self.lineage.outputs
                    for ref in out.reads
                    if ref.file == key and ref.register
                }
            )
            if regs:
                for r in regs:
                    seen[(key, r)] = None
            else:
                seen[(key, None)] = None
        return list(seen)

    def outputs(self) -> None:
        self.emit("## Arquivos gerados", "")
        for out in self.outputs_in_order():
            self.emit(f"### `{out.path}`", "")
            self.emit(f"**Lê:** {self.reads_label(out)}  ")
            when = out.when or "sempre."
            self.emit(f"**Quando:** {when}  ")
            meta = []
            url = _schema_url(out.path)
            if url:
                meta.append(f"**Esquema:** [{url.rsplit('/', 1)[-1]}]({url})")
            meta.append(f"**Código:** `{out.module}`")
            self.emit(" · ".join(meta), "")
            if out.summary:
                self.emit(out.summary, "")
            if out.kind == "checkpoint":
                continue
            header = [
                "Coluna" if out.kind == "parquet" else "Campo",
                "Origem",
                "Transformação",
            ]
            self.table(header, [self.field_row(f) for f in out.fields])

    def field_row(self, f: Field) -> list[str]:
        origin = ", ".join(self.ref_label(r) for r in f.from_) or "—"
        tags = []
        if f.status in FIELD_STATUS_LABEL:
            tags.append(FIELD_STATUS_LABEL[f.status])
        if f.when:
            tags.append(f"condicional: {f.when}")
        if tags:
            origin += " *(" + "; ".join(tags) + ")*"
        return [f"`{f.path}`", _cell(origin), _cell(f.how)]

    def source_index(self) -> None:
        self.emit("## Índice por origem", "")
        self.emit(
            "Para cada arquivo do deck (e, no DECOMP, cada registro): o que ele "
            "alimenta e o que nele ainda não é convertido.",
            "",
        )
        sources = {s.file: s for s in self.lineage.sources}
        listed: set[str] = set()
        for token in self.index_tokens():
            file, register = token
            listed.add(file)
            src = sources.get(file)
            self.emit(f"### {self.token_label(token)}", "")
            used = self.used.get(token, [])
            open_items = [
                it for it in (src.items if src else ()) if it.register == register
            ]
            readers = sorted(
                {o.path for o in self.lineage.outputs if token in self.read_tokens(o)}
            )
            if src and src.status:
                state = SOURCE_STATUS_LABEL[src.status]
            elif used and open_items:
                state = "convertido em parte"
            elif used:
                state = "convertido"
            elif readers or any(it.status == "deferred" for it in open_items):
                state = "lido, não convertido"
            else:
                state = "não lido"
            line = f"**Estado:** {state}."
            if len(readers) == len(self.lineage.outputs):
                line += " **Lido por:** todos os arquivos gerados."
            elif len(readers) > 8:
                line += (
                    f" **Lido por:** {len(readers)} arquivos gerados; "
                    "ver a visão geral."
                )
            elif readers:
                line += " **Lido por:** " + ", ".join(f"`{p}`" for p in readers) + "."
            self.emit(line, "")
            if src and src.note and register is None:
                self.emit(src.note, "")
            for item, out, f in used:
                arrow = f"`{out.path}` › `{f.path}`"
                self.emit(f"- `{item}` → {arrow}" if item else f"- → {arrow}")
            for it in open_items:
                label = SOURCE_STATUS_LABEL[it.status]
                head = f"`{it.item}` — " if it.item else "todo o registro — "
                self.emit(f"- {head}*{label}.* {_cell(it.note)}")
            if used or open_items:
                self.emit("")
        for src in self.lineage.sources:
            if src.file in listed or src.file in self.reg:
                continue
            self.emit(f"### `{src.label}`", "")
            self.emit(f"**Estado:** {SOURCE_STATUS_LABEL[src.status]}.", "")
            if src.note:
                self.emit(src.note, "")

    def index_tokens(self) -> list[Token]:
        tokens = self.row_tokens()
        # Register-level entries also exist for registers only the inventory
        # mentions (deferred/unread), so they get their own heading.
        for src in self.lineage.sources:
            for it in src.items:
                if it.register and (src.file, it.register) not in tokens:
                    tokens.append((src.file, it.register))
        out: list[Token] = []
        for key in self.reg:
            out.extend(
                sorted((t for t in tokens if t[0] == key), key=lambda t: t[1] or "")
            )
        return out

    def backlog(self) -> None:
        self.emit("## Ainda não convertido", "")
        self.emit(
            "Tudo o que o conversor deixa de fora, reunido em um só lugar. Cada item "
            "aparece também no índice acima, junto do arquivo a que pertence.",
            "",
        )
        unread_files = [
            s for s in self.lineage.sources if s.status in ("unread", "deferred")
        ]
        if unread_files:
            self.emit("### Arquivos do deck não lidos", "")
            for src in unread_files:
                label = SOURCE_STATUS_LABEL[src.status]
                self.emit(
                    f"- `{self.file_label(src.file)}` — *{label}.* {_cell(src.note)}"
                )
            self.emit("")
        items = [
            (src, it)
            for src in self.lineage.sources
            for it in src.items
            if not src.status
        ]
        if items:
            self.emit("### Registros e campos do deck não convertidos", "")
            for src, it in items:
                token = self.token_label((src.file, it.register))
                if it.item:
                    token += f" › `{it.item}`"
                label = SOURCE_STATUS_LABEL[it.status]
                self.emit(f"- {token} — *{label}.* {_cell(it.note)}")
            self.emit("")
        nulls = [
            (out, f)
            for out in self.outputs_in_order()
            for f in out.fields
            if f.status == "null"
        ]
        if nulls:
            self.emit("### Campos do Novomodelo sem origem no deck", "")
            for out, f in nulls:
                self.emit(f"- `{out.path}` › `{f.path}` — {_cell(f.how)}")
            self.emit("")

    def render(self) -> str:
        self.intro()
        self.overview()
        self.outputs()
        self.source_index()
        self.backlog()
        text = "\n".join(self.lines).rstrip() + "\n"
        return text


def render(lineage: Lineage) -> str:
    """The complete Markdown page for *lineage*."""
    return _Page(lineage).render()


def page_path(track: str):
    from .model import REPO_ROOT

    return REPO_ROOT / "docs" / f"{track}-data-map.md"
