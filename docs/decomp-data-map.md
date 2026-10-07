<!-- Gerado por scripts/gen-lineage-docs.py a partir de docs/lineage/decomp.toml. Não edite à mão. -->

# Mapa de dados: DECOMP → Novomodelo

Esta página mostra de onde vem cada arquivo e cada campo do caso Novomodelo que `convert decomp` escreve: qual arquivo do deck DECOMP, qual registro ou coluna, e que transformação é aplicada no caminho. Ela também lista o que o conversor ainda não converte, para que o trabalho pendente fique visível no mesmo lugar.

O conteúdo é gerado a partir de `docs/lineage/decomp.toml` por `scripts/gen-lineage-docs.py` e verificado pelos testes contra uma conversão real do deck de exemplo do repositório, de modo que a página não pode divergir do código sem quebrar a build.

**Como ler.** Nas matrizes, ● indica que o arquivo gerado (coluna) depende do arquivo do deck (linha). *(opcional)* marca um arquivo que o deck pode não trazer; a conversão prossegue sem ele. Nas tabelas de campos, a coluna Origem cita arquivo › registro › coluna do deck e a coluna Transformação diz o que o conversor faz com o valor. *Derivado* marca um valor de escrituração (ids, datas, ordem) calculado a partir do deck; *constante* um valor fixo que o conversor sempre escreve; *sempre nulo* um campo do Novomodelo que ainda não recebe informação do deck.

Salvo indicação, um registro de duas letras (`CT`, `UH`, `DP`, ...) é um registro do `dadger.rvN`.

## Visão geral

Uma matriz por diretório do caso. As linhas são os arquivos do deck (no DECOMP, também os registros do `dadger.rvN`); as colunas, os arquivos gerados. As seções seguintes detalham cada coluna.

### Raiz do caso

| Arquivo do deck | `config.json` | `stages.json` | `penalties.json` | `initial_conditions.json` | `post_study_stages.json` |
| --- | :-: | :-: | :-: | :-: | :-: |
| `dadger.rvN` › `AC` |  |  | ● | ● |  |
| `dadger.rvN` › `AR` |  | ● |  |  | ● |
| `dadger.rvN` › `CD` |  |  | ● |  |  |
| `dadger.rvN` › `CT` |  |  |  | ● | ● |
| `dadger.rvN` › `DP` |  | ● | ● | ● | ● |
| `dadger.rvN` › `DT` |  | ● | ● | ● | ● |
| `dadger.rvN` › `FC` |  | ● |  |  |  |
| `dadger.rvN` › `GP` | ● |  |  |  |  |
| `dadger.rvN` › `NI` | ● |  |  |  |  |
| `dadger.rvN` › `SB` |  |  | ● | ● | ● |
| `dadger.rvN` › `TX` |  | ● |  |  |  |
| `dadger.rvN` › `UH` |  |  | ● | ● | ● |
| `vazoes.rvN` |  | ● |  |  |  |
| `hidr.dat` |  |  | ● | ● |  |
| `dadgnl.rvN` › `GL` (opcional) |  |  |  | ● | ● |
| `dadgnl.rvN` › `GS` (opcional) |  |  |  | ● | ● |
| `dadgnl.rvN` › `NL` (opcional) |  |  |  | ● | ● |
| `dadgnl.rvN` › `TG` (opcional) |  |  |  | ● | ● |
| `cortesh.dat` (opcional) |  | ● |  |  |  |

### `system/`

| Arquivo do deck | `buses.json` | `lines.json` | `thermals.json` | `hydros.json` | `hydro_production_models.json` | `hydro_geometry.parquet` | `hydro_energy_productivity.parquet` | `tailrace_curves.parquet` | `non_controllable_sources.json` | `pumping_stations.json` | `energy_contracts.json` |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `dadger.rvN` › `AC` |  |  |  | ● | ● | ● | ● |  |  |  |  |
| `dadger.rvN` › `CD` | ● |  |  |  |  |  |  |  |  |  |  |
| `dadger.rvN` › `CE` |  |  |  |  |  |  |  |  |  |  | ● |
| `dadger.rvN` › `CI` |  |  |  |  |  |  |  |  |  |  | ● |
| `dadger.rvN` › `CQ` |  |  |  | ● |  |  |  |  |  |  |  |
| `dadger.rvN` › `CT` | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `DP` | ● | ● | ● | ● | ● | ● | ● |  | ● | ● | ● |
| `dadger.rvN` › `DT` | ● | ● | ● | ● | ● | ● | ● |  | ● | ● | ● |
| `dadger.rvN` › `FD` |  |  |  | ● |  |  |  |  |  |  |  |
| `dadger.rvN` › `HQ` |  |  |  | ● |  |  |  |  |  |  |  |
| `dadger.rvN` › `IA` |  | ● |  |  |  |  |  |  |  |  |  |
| `dadger.rvN` › `LQ` |  |  |  | ● |  |  |  |  |  |  |  |
| `dadger.rvN` › `MP` |  |  |  | ● |  |  |  |  |  |  |  |
| `dadger.rvN` › `PQ` |  |  |  |  |  |  |  |  | ● |  |  |
| `dadger.rvN` › `RI` |  | ● |  |  |  |  |  |  |  |  |  |
| `dadger.rvN` › `SB` | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `UE` |  |  |  |  |  |  |  |  |  | ● |  |
| `dadger.rvN` › `UH` | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `hidr.dat` |  | ● |  | ● | ● | ● | ● |  |  |  |  |
| `dadgnl.rvN` › `GL` (opcional) |  |  | ● |  |  |  |  |  |  |  |  |
| `dadgnl.rvN` › `GS` (opcional) |  |  | ● |  |  |  |  |  |  |  |  |
| `dadgnl.rvN` › `NL` (opcional) |  |  | ● |  |  |  |  |  |  |  |  |
| `dadgnl.rvN` › `TG` (opcional) |  |  | ● |  |  |  |  |  |  |  |  |
| `renovaveis*` (opcional) |  |  |  |  |  |  |  |  | ● |  |  |
| `polinjus.csv` (opcional) |  |  |  |  |  |  |  | ● |  |  |  |

### `scenarios/`

| Arquivo do deck | `inflow_seasonal_stats.parquet` | `external_inflow_scenarios.parquet` | `load_seasonal_stats.parquet` | `load_factors.json` | `external_load_scenarios.parquet` | `non_controllable_stats.parquet` | `non_controllable_factors.json` | `external_ncs_scenarios.parquet` |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `dadger.rvN` › `AC` |  | ● |  |  |  |  |  |  |
| `dadger.rvN` › `CT` | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `DP` | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `DT` | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `PQ` |  |  |  |  |  | ● | ● | ● |
| `dadger.rvN` › `RI` |  |  | ● | ● | ● |  |  |  |
| `dadger.rvN` › `SB` | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `UH` | ● | ● | ● | ● | ● | ● | ● | ● |
| `vazoes.rvN` |  | ● |  |  | ● |  |  | ● |
| `hidr.dat` |  | ● | ● | ● | ● |  |  |  |
| `renovaveis*` (opcional) |  |  |  |  |  | ● | ● | ● |

### `constraints/`

| Arquivo do deck | `hydro_bounds.parquet` | `hydro_unit_group_bounds.parquet` | `thermal_bounds.parquet` | `line_bounds.parquet` | `pumping_bounds.parquet` | `contract_bounds.parquet` | `generic_constraints.json` | `generic_constraint_bounds.parquet` | `generic_parameters.json` |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `dadger.rvN` › `AC` | ● | ● |  |  |  |  | ● | ● | ● |
| `dadger.rvN` › `CD` |  |  |  |  |  |  | ● | ● |  |
| `dadger.rvN` › `CE` |  |  |  |  |  | ● |  |  |  |
| `dadger.rvN` › `CI` |  |  |  |  |  | ● |  |  |  |
| `dadger.rvN` › `CM` |  |  |  |  |  |  | ● | ● | ● |
| `dadger.rvN` › `CQ` | ● |  |  |  | ● |  | ● | ● |  |
| `dadger.rvN` › `CT` | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `CV` | ● |  |  |  |  |  | ● | ● |  |
| `dadger.rvN` › `DP` | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `DT` | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `FD` |  | ● |  |  |  |  | ● | ● |  |
| `dadger.rvN` › `FI` | ● |  | ● |  |  |  | ● | ● |  |
| `dadger.rvN` › `FT` | ● |  | ● |  |  |  | ● | ● |  |
| `dadger.rvN` › `FU` | ● |  | ● |  |  |  | ● | ● |  |
| `dadger.rvN` › `HE` |  |  |  |  |  |  | ● | ● | ● |
| `dadger.rvN` › `HQ` | ● |  |  |  | ● |  | ● | ● |  |
| `dadger.rvN` › `HV` | ● |  |  |  |  |  | ● | ● |  |
| `dadger.rvN` › `IA` |  |  |  | ● |  |  | ● | ● |  |
| `dadger.rvN` › `LQ` | ● |  |  |  | ● |  | ● | ● |  |
| `dadger.rvN` › `LU` | ● |  | ● |  |  |  | ● | ● |  |
| `dadger.rvN` › `LV` | ● |  |  |  |  |  | ● | ● |  |
| `dadger.rvN` › `MP` |  | ● |  |  |  |  | ● | ● |  |
| `dadger.rvN` › `PQ` |  |  |  |  |  |  | ● | ● |  |
| `dadger.rvN` › `RE` | ● |  | ● |  |  |  | ● | ● |  |
| `dadger.rvN` › `RI` |  | ● |  | ● |  |  | ● | ● |  |
| `dadger.rvN` › `RQ` | ● |  |  |  |  |  |  |  |  |
| `dadger.rvN` › `SB` | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `TI` | ● |  |  |  |  |  |  |  |  |
| `dadger.rvN` › `UE` |  |  |  |  | ● |  | ● | ● |  |
| `dadger.rvN` › `UH` | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| `dadger.rvN` › `VE` | ● |  |  |  |  |  |  |  |  |
| `hidr.dat` | ● | ● |  | ● |  |  | ● | ● | ● |
| `renovaveis*` (opcional) |  |  |  |  |  |  | ● | ● |  |
| `lib_restricao-eletrica-especial*.csv` (opcional) |  |  |  |  |  |  | ● | ● |  |
| `indices.csv` |  |  |  |  |  |  | ● | ● |  |

### `boundary/`

| Arquivo do deck | `boundary/` |
| --- | :-: |
| `dadger.rvN` › `AC` | ● |
| `dadger.rvN` › `CT` | ● |
| `dadger.rvN` › `CX` | ● |
| `dadger.rvN` › `DP` | ● |
| `dadger.rvN` › `DT` | ● |
| `dadger.rvN` › `FC` | ● |
| `dadger.rvN` › `SB` | ● |
| `dadger.rvN` › `UH` | ● |
| `vazoes.rvN` | ● |
| `hidr.dat` | ● |
| `dadgnl.rvN` › `GL` (opcional) | ● |
| `dadgnl.rvN` › `GS` (opcional) | ● |
| `dadgnl.rvN` › `NL` (opcional) | ● |
| `dadgnl.rvN` › `TG` (opcional) | ● |
| `cortesh.dat` (opcional) | ● |
| `cortes*.dat` (opcional) | ● |
| `mlt.dat` | ● |

## Arquivos gerados

### `config.json`

**Lê:** `dadger.rvN` (`GP`, `NI`)  
**Quando:** sempre.  
**Esquema:** [config.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/config.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/config.py`

Critérios de parada (`GP` como gap relativo e `NI` como limite de iterações) e fontes externas de cenários; todo o restante é fixo.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `simulation.enabled` | — *(constante)* | Sempre `true`: a simulação final é executada sobre a árvore explícita. |
| `simulation.selection.method` | — *(constante)* | Sempre `enumerated`: a simulação percorre cada caminho raiz-folha do grafo de nós com a probabilidade do próprio nó. |
| `training.scenario_source.inflow.scheme` | — *(constante)* | Sempre `external`: as vazões vêm da árvore explícita em `scenarios/external_inflow_scenarios.parquet`. |
| `training.scenario_source.load.scheme` | — *(constante)* | Sempre `external`: a carga determinística de `DP` é replicada em `scenarios/external_load_scenarios.parquet`. |
| `training.scenario_source.ncs.scheme` | — *(constante)* | Sempre `external`: a geração não controlável determinística é replicada em `scenarios/external_ncs_scenarios.parquet`. |
| `training.scenario_source.seed` | — *(constante)* | Sempre `0`. Com todas as classes externas e seleção enumerada nada é sorteado; o campo é exigido pelo esquema do Novomodelo e o valor é inerte. |
| `training.selection.method` | — *(constante)* | Sempre `enumerated`: o treinamento percorre o tronco determinístico e o leque terminal por completo. |
| `training.stopping_rules[].type` | — *(constante)* | Duas regras fixas, nesta ordem: `gap` (tolerância relativa) e `iteration_limit`. |
| `training.stopping_rules[].relative_tolerance` | `dadger.rvN` › `GP` › `gap` | O critério de convergência `GP` do deck, usado como gap relativo (`Zsup/Zinf - 1 <= GP`). O Novomodelo acrescenta por conta própria uma regra de estagnação de limites. |
| `training.stopping_rules[].limit` | `dadger.rvN` › `NI` › `iteracoes` | Número máximo de iterações de `NI`; `500` quando o campo está em branco. |

### `stages.json`

**Lê:** `dadger.rvN` (`DT`, `DP`, `TX`, `AR`, `FC`), `vazoes.rvN`, `cortesh.dat` (opcional)  
**Quando:** sempre.  
**Esquema:** [stages.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/stages.schema.json) · **Código:** `src/novomodelo_bridge/decomp/temporal.py`

Calendário operativo semanal a partir de `DT` e das durações de patamar de `DP`; grafo de nós com tronco determinístico e leque terminal ponderado pelas probabilidades do arquivo de vazões; medida de risco de `AR`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `policy_graph.type` | — *(constante)* | Sempre `finite_horizon`. |
| `policy_graph.annual_discount_rate` | `dadger.rvN` › `TX` › `taxa` | Taxa de desconto anual de `TX`, em percentual no deck, dividida por 100. |
| `policy_graph.nodes[].id` | `dadger.rvN` › `DP` › `estagio`, `vazoes.rvN` › `probabilidades · cenario` *(derivado)* | Id próprio do nó: os nós do tronco (estágios 0 a T-2) recebem `id` igual ao índice do estágio; os nós do leque terminal recebem `(T-1) + k`, com `k` o índice 0-based do cenário. |
| `policy_graph.nodes[].stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio a que o nó pertence; todos os nós do leque apontam para o último estágio. |
| `policy_graph.nodes[].scenario_id` | `vazoes.rvN` › `probabilidades · cenario` *(derivado)* | Coluna da biblioteca externa de vazões que o nó lê: `0` em todo o tronco; `cenario - 1` em cada nó do leque terminal. |
| `policy_graph.nodes[].label` | — *(derivado)* | Rótulo `trunk-{estágio}` nos nós do tronco e `fan-{cenário}` nos nós do leque. |
| `policy_graph.transitions[].source_id` | — *(derivado)* | Nó de origem: cada nó do tronco liga-se ao seguinte; o último nó do tronco liga-se a todos os nós do leque. |
| `policy_graph.transitions[].target_id` | — *(derivado)* | Nó de destino da aresta, na ordem canônica dos sucessores. |
| `policy_graph.transitions[].probability` | `vazoes.rvN` › `probabilidades · probabilidade (estagio terminal)` | `1.0` nas arestas do tronco; nas arestas para o leque, a probabilidade de cada cenário do estágio terminal na tabela de probabilidades do arquivo de vazões (validadas como somando 1). |
| `season_definitions.cycle_type` | — *(constante)* | Sempre `monthly`. |
| `season_definitions.seasons[].id` | — *(constante)* | Doze estações fixas, ids 0 a 11 (janeiro = 0). |
| `season_definitions.seasons[].month_start` | — *(constante)* | Mês de início de cada estação, 1 a 12. |
| `season_definitions.seasons[].label` | — *(constante)* | Nome do mês em inglês (`January` … `December`), convenção compartilhada com a conversão NEWAVE. |
| `stages[].id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | `estagio - 1`; os estágios de `DP` devem ser contíguos a partir de 1. |
| `stages[].start_date` | `dadger.rvN` › `DT` › `dia, mes, ano`, `dadger.rvN` › `DP` › `duracao` *(derivado)* | Data de `DT` no primeiro estágio (obrigatoriamente um sábado); nos demais, a data final do estágio anterior. Cada estágio dura a soma das durações de patamar de `DP` convertida em dias inteiros. |
| `stages[].end_date` | `dadger.rvN` › `DT` › `dia, mes, ano`, `dadger.rvN` › `DP` › `duracao` *(derivado)* | Data de início mais a duração do estágio (exclusiva). Todo estágio exceto o último deve somar 168 h; o último deve terminar em fronteira de mês civil e cobrir o segundo mês operativo. |
| `stages[].season_id` | `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado)* | Estações semanais recebem o mês da sexta-feira da primeira semana (data de `DT` + 6 dias) menos 1; o estágio mensal final recebe o mês do seu próprio início menos 1. |
| `stages[].blocks[].id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Índice 0-based do patamar dentro do estágio. |
| `stages[].blocks[].name` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Nome canônico pelo número de patamares: `HEAVY`/`MEDIUM`/`LIGHT` para três, `HEAVY`/`LIGHT` para dois, `SINGLE` para um, `BLOCK_i` para outros valores. |
| `stages[].blocks[].hours` | `dadger.rvN` › `DP` › `duracao` | Duração em horas de cada patamar (`duracao_k`). As durações devem coincidir entre todos os submercados do mesmo estágio; o deck define uma única grade temporal. |
| `stages[].risk_measure` | `dadger.rvN` › `AR` › `estagio, lamb, alfa`, `cortesh.dat` › `alfa_cvar, lambda_cvar` | `expectation` quando não há registro `AR` ou quando λ/α não se resolvem a valores positivos; caso contrário um objeto `cvar` com `alpha`/`lambda`. λ/α em branco em `AR` são lidos do cabeçalho `cortesh.dat` (localizado pelo registro `FC NEWV21`); valores > 1 são tratados como percentuais. A medida é emitida uniformemente em todos os estágios, não apenas a partir do estágio de `AR`, porque colapsa em esperança no tronco determinístico. |
| `stages[].risk_measure.cvar.alpha` | `dadger.rvN` › `AR` › `alfa`, `cortesh.dat` › `alfa_cvar` *(condicional: somente quando o deck resolve um CVaR ativo via `AR` (ou `AR` com λ/α em branco e `cortesh.dat` com CVaR habilitado))* | Fração da cauda (α) do CVaR, mesma convenção do DECOMP; dividida por 100 quando declarada em percentual. |
| `stages[].risk_measure.cvar.lambda` | `dadger.rvN` › `AR` › `lamb`, `cortesh.dat` › `lambda_cvar` *(condicional: somente quando o deck resolve um CVaR ativo via `AR` (ou `AR` com λ/α em branco e `cortesh.dat` com CVaR habilitado))* | Peso de aversão a risco (λ) do CVaR; dividido por 100 quando declarado em percentual. |
| `stages[].state_variables.storage` | — *(constante)* | Sempre `true`: o armazenamento é variável de estado em todos os estágios. |
| `stages[].state_variables.inflow_lags` | — *(constante)* | Sempre `false`: o modelo de vazões é de ordem 0 e as defasagens só são precificadas pela FCF de fronteira, quando importada. |

### `penalties.json`

**Lê:** `dadger.rvN` (`CD`, `SB`, `UH`, `AC`, `DP`, `DT`), `hidr.dat`  
**Quando:** sempre.  
**Esquema:** [penalties.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/penalties.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/config.py`

Custo de déficit de `CD` e micro-penalidades fixas do modelo de origem; o bloco hidráulico é escalado pela produtibilidade equivalente média (ρ_avg) e máxima (ρ_max) das usinas operadas, calculadas do cadastro `hidr.dat` ancorado no volume inicial de `UH`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `bus.deficit_segments[].cost` | `dadger.rvN` › `CD` › `custo` | O maior custo de déficit entre os submercados de `CD` (cada submercado deve ter um único custo, uniforme entre patamares e estágios). É o `MAX_CUSTO_DEFICIT` que ancora as penalidades hidráulicas abaixo. |
| `bus.deficit_segments[].depth_mw` | — *(sempre nulo)* | Sempre nulo: segmento único de profundidade total (`CD` deve declarar `limite_superior` = 100 %). |
| `bus.excess_cost` | — *(constante)* | Micro-penalidade fixa `0.000355` R$/MWh (excesso de energia, manual do modelo de origem). |
| `hydro.spillage_cost` | `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `0.000300 × ρ_avg`, com ρ_avg a média (zeros incluídos) da produtibilidade equivalente `ρ_esp · h_líquida` de cada usina operada, altura calculada com o polinômio cota-volume no volume inicial, descontados canal de fuga e perdas. |
| `hydro.turbined_cost` | `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `0.000333 × ρ_avg` (micro-penalidade de turbinamento escalada pela produtibilidade média do sistema). |
| `hydro.diversion_cost` | `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `0.000300 × ρ_avg` (micro-penalidade de volume desviado). |
| `hydro.storage_violation_below_cost` | `dadger.rvN` › `CD` › `custo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `10 × custo_déficit × ρ_max × (1e6/3600)`, em R$/hm³: patamar de evaporação convertido a energia equivalente, com ρ_max a maior produtibilidade equivalente entre as usinas operadas. Não há `PENALID` no DECOMP; o valor é sempre derivado. |
| `hydro.filling_target_violation_cost` | `dadger.rvN` › `CD` › `custo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `0.9 × custo_déficit × ρ_max × (1e6/3600)`, em R$/hm³. |
| `hydro.turbined_violation_below_cost` | `dadger.rvN` › `CD` › `custo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `10 × custo_déficit × ρ_avg` (folga de turbinamento mínimo, sem `PENALID` no DECOMP). |
| `hydro.outflow_violation_below_cost` | `dadger.rvN` › `CD` › `custo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `10 × custo_déficit × ρ_avg` (folga de vazão defluente mínima). |
| `hydro.outflow_violation_above_cost` | `dadger.rvN` › `CD` › `custo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `10 × custo_déficit × ρ_avg` (folga de vazão defluente máxima). |
| `hydro.generation_violation_below_cost` | `dadger.rvN` › `CD` › `custo` *(derivado)* | `10 × custo_déficit`, em R$/MWh (folga de geração mínima; domínio de energia, sem fator de produtibilidade). |
| `hydro.evaporation_violation_cost` | `dadger.rvN` › `CD` › `custo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `10 × custo_déficit × ρ_max` (folga de evaporação, patamar mais alto da ordem de mérito). |
| `hydro.water_withdrawal_violation_cost` | `dadger.rvN` › `CD` › `custo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `10 × custo_déficit × ρ_max` (folga de retirada de água para irrigação `TI`). |
| `hydro.inflow_nonnegativity_cost` | `dadger.rvN` › `CD` › `custo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas, tipo_perda`, `dadger.rvN` › `UH` › `volume_inicial` *(derivado)* | `water_withdrawal_violation_cost + 1` R$/(m³/s): a folga de vazão natural negativa nunca pode ser mais barata que a folga de retirada de água. |
| `line.exchange_cost` | — *(constante)* | Micro-penalidade fixa `0.000273` R$/MWh (intercâmbio). |
| `non_controllable_source.curtailment_cost` | — *(constante)* | Micro-penalidade fixa `0.000344` R$/MWh (corte de geração não controlável). |

### `initial_conditions.json`

**Lê:** `dadger.rvN` (`UH`, `SB`, `CT`, `AC`, `DP`, `DT`), `hidr.dat`, `dadgnl.rvN` (opcional) (`TG`, `GL`, `GS`, `NL`)  
**Quando:** sempre.  
**Esquema:** [initial_conditions.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/initial_conditions.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/hydro/entity.py`

Volume inicial de cada usina operada (`UH` com `volume_inicial`), em hm³; quando `dadgnl` declara despacho GNL comprometido, o arquivo também recebe as janelas `past_anticipated_commitments`. As observações recentes de vazão são acrescentadas apenas pelo importador da FCF de fronteira, fora desta conversão.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `storage[].hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id 0-based denso das usinas operadas (linhas `UH` com `volume_inicial`), atribuído na ordem crescente de `codigo_usina`. |
| `storage[].value_hm3` | `dadger.rvN` › `UH` › `volume_inicial`, `hidr.dat` › `volume_minimo, volume_maximo`, `hidr.dat` › `tipo_regulacao, volume_referencia`, `dadger.rvN` › `AC` › `VOLMIN, VOLMAX` | `volume_inicial` (% do volume útil) resolvido sobre a faixa efetiva do estágio inicial: `vmin + pct/100 · (vmax − vmin)`, com `vmin`/`vmax` do cadastro após os `AC VOLMIN`/`VOLMAX` vigentes no estágio 0, limitado à faixa. Usinas a fio d'água (`tipo_regulacao` = `D`) colapsam ao `volume_referencia` independentemente do percentual. Um `volume_morto_inicial` declarado é apenas avisado, não convertido. |
| `filling_storage` | — *(sempre nulo)* | Sempre lista vazia: o enchimento de volume morto não é convertido. |
| `past_anticipated_commitments[].thermal_id` | `dadgnl.rvN` › `TG` › `codigo_usina` *(derivado; condicional: somente quando `dadgnl` declara despacho GNL comprometido (algum `GL` com `geracao` não nula))* | Id da térmica GNL criada em `system/thermals.json` (ids densos após a última térmica de `CT`, em ordem crescente de `codigo_usina` de `TG`). |
| `past_anticipated_commitments[].start_date` | `dadgnl.rvN` › `GL` › `data_inicio`, `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado; condicional: somente quando `dadgnl` declara despacho GNL comprometido (algum `GL` com `geracao` não nula))* | Janelas de classe 2 (dentro do estudo): a data de início de cada estágio do estudo coberto pelo lead global (soma das semanas `GS` dos meses do estudo × 168 h). Janelas de classe 4 (já comandadas): a `data_inicio` de cada semana `GL` a partir do fim do horizonte, com um stub a 0 MW quando a primeira começa depois dele. |
| `past_anticipated_commitments[].end_date` | `dadgnl.rvN` › `GL` › `data_inicio`, `dadger.rvN` › `DP` › `duracao` *(derivado; condicional: somente quando `dadgnl` declara despacho GNL comprometido (algum `GL` com `geracao` não nula))* | Classe 2: a data final do estágio do estudo. Classe 4: a `data_inicio` da semana `GL` seguinte, ou uma semana operativa (168 h) após o início na última janela. Nenhuma janela pode cruzar o fim do horizonte. |
| `past_anticipated_commitments[].value_mw` | `dadgnl.rvN` › `GL` › `geracao`, `dadgnl.rvN` › `GL` › `duracao`, `dadgnl.rvN` › `TG` › `inflexibilidade, disponibilidade` *(condicional: somente quando `dadgnl` declara despacho GNL comprometido (algum `GL` com `geracao` não nula))* | MW comprometido: média de `geracao` ponderada pelas `duracao` de patamar de cada registro `GL`; nas janelas de classe 2, as semanas `GL` que caem no estágio são refundidas por horas sobre o estágio (0 onde não há nenhuma). O valor é limitado, com aviso, à faixa de `TG` do seu estágio (inflexibilidade e disponibilidade ponderadas pelas horas de patamar, herdando o último `TG` declarado); as janelas de classe 4 usam a do último estágio do estudo. |

### `post_study_stages.json`

**Lê:** `dadgnl.rvN` (opcional) (`TG`, `GL`, `GS`, `NL`), `dadger.rvN` (`DT`, `DP`, `AR`, `CT`, `SB`, `UH`)  
**Quando:** somente quando `dadgnl` declara despacho GNL comprometido (algum `GL` com `geracao` não nula) e um calendário `GS`  
**Código:** `src/novomodelo_bridge/decomp/converters/anticipated.py`

Calendário pós-estudo na grade semanal (sábados): preenche as semanas já comandadas de `GL` após o fim do horizonte e depois espelha os estágios do estudo; cada térmica GNL recebe limites de preço apenas nos estágios sinalizados (classe 3), nunca nos já comandados.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `stages[].start_date` | `dadger.rvN` › `DT` › `dia, mes, ano`, `dadger.rvN` › `DP` › `duracao`, `dadgnl.rvN` › `GL` › `data_inicio` *(derivado)* | Começa no fim do horizonte do estudo: um stub até o sábado seguinte (se necessário), semanas de 168 h até o corte já comandado comum a todas as usinas (`data_inicio` da última semana `GL` mais uma semana) e, a partir dele, um estágio espelhando cada estágio do estudo. |
| `stages[].duration_hours` | `dadger.rvN` › `DP` › `duracao` *(derivado)* | Horas até o próximo sábado no stub; `168` nas semanas; no espelho do estágio mensal do estudo, as horas até o último dia do mês civil do próprio estágio espelhado. |
| `thermal_bounds[].thermal_id` | `dadgnl.rvN` › `TG` › `codigo_usina` *(derivado)* | Id da térmica GNL em `system/thermals.json`. |
| `thermal_bounds[].post_study_stage_index` | `dadgnl.rvN` › `GL` › `data_inicio` *(derivado)* | Índice 0-based do estágio pós-estudo; apenas os estágios cujo início é igual ou posterior ao corte já comandado da usina recebem uma linha. |
| `thermal_bounds[].cost_per_mwh` | `dadgnl.rvN` › `TG` › `cvu`, `dadger.rvN` › `DP` › `duracao` | `cvu_1..3` do `TG` vigente no último estágio do estudo (o último declarado), ponderados pelas horas de patamar desse estágio. |
| `thermal_bounds[].min_mw` | `dadgnl.rvN` › `TG` › `inflexibilidade`, `dadger.rvN` › `DP` › `duracao` | `inflexibilidade_1..3` do mesmo `TG`, com a mesma ponderação. |
| `thermal_bounds[].max_mw` | `dadgnl.rvN` › `TG` › `disponibilidade`, `dadger.rvN` › `DP` › `duracao` | `disponibilidade_1..3` do mesmo `TG`, com a mesma ponderação. |

### `system/buses.json`

**Lê:** `dadger.rvN` (`SB`, `CD`, `UH`, `CT`, `DP`, `DT`)  
**Quando:** sempre.  
**Esquema:** [buses.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/buses.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/network.py`

Uma barra por submercado de `SB` (inclusive o fictício), mais uma barra de transbordo `IV` criada pela conversão; a curva de déficit vem de `CD`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `buses[].id` | `dadger.rvN` › `SB` › `codigo_submercado` *(derivado)* | Id 0-based denso na ordem crescente de `codigo_submercado`; a barra de transbordo `IV` recebe o último id. |
| `buses[].name` | `dadger.rvN` › `SB` › `nome_submercado` | Nome do submercado sem espaços laterais; `IV` para a barra de transbordo (o nome é reservado e não pode aparecer em `SB`). |
| `buses[].operational_start_date` | `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado)* | Data de início do estudo; submercados não têm data de entrada, o valor serve apenas à ordenação canônica. |
| `buses[].deficit_segments[].cost` | `dadger.rvN` › `CD` › `custo` | Custo de déficit do submercado em `CD`, exigido único (uma curva, um valor uniforme entre patamares e estágios). Submercados sem `CD` (o fictício, a barra `IV`) omitem o bloco e herdam o padrão global de `penalties.json`. |
| `buses[].deficit_segments[].depth_mw` | — *(sempre nulo)* | Sempre nulo: segmento único de profundidade total; um `limite_superior` diferente de 100 % em `CD` é rejeitado. |

### `system/lines.json`

**Lê:** `dadger.rvN` (`IA`, `RI`, `SB`, `UH`, `CT`, `DP`, `DT`), `hidr.dat`  
**Quando:** sempre.  
**Esquema:** [lines.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/lines.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/network.py`

Uma linha por par de submercados de `IA`. Quando Itaipu (código 66) é operada e `IA` não liga `IV` ao submercado dela, a conversão acrescenta a linha `IV-SE` com a capacidade de 60 Hz de `RI`. Um deck sem `IA` ainda gera o arquivo com lista vazia (TRACKED NOVOMODELO-GAP C4).

| Campo | Origem | Transformação |
| --- | --- | --- |
| `lines[].id` | `dadger.rvN` › `IA` › `nome_submercado_de, nome_submercado_para` *(derivado)* | Pares ordenados por (id da barra de origem, id da barra de destino); a linha `IV-SE` criada pela conversão recebe o id seguinte ao último par de `IA`. |
| `lines[].name` | `dadger.rvN` › `IA` › `nome_submercado_de, nome_submercado_para` *(derivado)* | `{de}-{para}` com os nomes de `IA`; `IV-SE` para a linha criada pela conversão. |
| `lines[].operational_start_date` | `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado)* | Data de início do estudo. |
| `lines[].source_bus_id` | `dadger.rvN` › `IA` › `nome_submercado_de` | Barra resolvida pelo nome de origem (`IV` resolve para a barra de transbordo). Na linha `IV-SE` criada, a barra de transbordo. |
| `lines[].target_bus_id` | `dadger.rvN` › `IA` › `nome_submercado_para`, `hidr.dat` › `submercado (usina 66)` | Barra resolvida pelo nome de destino. Na linha `IV-SE` criada, a barra do `submercado` de Itaipu no cadastro. |
| `lines[].capacity.direct_mw` | `dadger.rvN` › `IA` › `limite_de_para`, `dadger.rvN` › `RI` › `geracao_maxima_60_hz` | Maior limite de→para entre os patamares do estágio 1 (`limite_de_para_k`); o sentinela 99999 passa como capacidade grande. Na linha `IV-SE` criada, o maior `geracao_maxima_60_hz` de `RI` em todos os estágios e patamares (99999 sem `RI`). |
| `lines[].capacity.reverse_mw` | `dadger.rvN` › `IA` › `limite_para_de`, `dadger.rvN` › `RI` › `geracao_maxima_60_hz` | Maior limite para→de entre os patamares do estágio 1 (`limite_para_de_k`). Na linha `IV-SE` criada, o mesmo valor do sentido direto. |

### `system/thermals.json`

**Lê:** `dadger.rvN` (`CT`, `SB`, `UH`, `DP`, `DT`), `dadgnl.rvN` (opcional) (`TG`, `GL`, `GS`, `NL`)  
**Quando:** sempre.  
**Esquema:** [thermals.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/thermals.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/thermal.py`

Uma térmica por `codigo_usina` de `CT`, com os valores do estágio 1 ponderados pelas horas de patamar; as térmicas GNL de `dadgnl` (ausentes de `CT`) são acrescentadas ao final com ids seguintes, marcadas como despacho antecipado.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `thermals[].id` | `dadger.rvN` › `CT` › `codigo_usina`, `dadgnl.rvN` › `TG` › `codigo_usina` *(derivado)* | Id 0-based denso na ordem crescente de `codigo_usina` de `CT`; as térmicas GNL de `TG` continuam a numeração, também em ordem crescente de código. |
| `thermals[].name` | `dadger.rvN` › `CT` › `nome_usina`, `dadgnl.rvN` › `TG` › `nome` | Nome da usina sem espaços laterais. |
| `thermals[].operational_start_date` | `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado)* | Data de início do estudo. |
| `thermals[].bus_id` | `dadger.rvN` › `CT` › `codigo_submercado`, `dadgnl.rvN` › `TG` › `codigo_submercado` | Barra do submercado declarado. |
| `thermals[].cost_per_mwh` | `dadger.rvN` › `CT` › `cvu`, `dadger.rvN` › `DP` › `duracao`, `dadgnl.rvN` › `TG` › `cvu` | CVU do estágio 1 por patamar (`cvu_k`, em branco lido como 0), ponderado pelas horas de patamar de `DP`; estágios seguintes vão para `constraints/thermal_bounds.parquet`. GNL: o mesmo com o `TG` do estágio 1. |
| `thermals[].generation.min_mw` | `dadger.rvN` › `CT` › `inflexibilidade`, `dadger.rvN` › `DP` › `duracao`, `dadgnl.rvN` › `TG` › `inflexibilidade` | Inflexibilidade do estágio 1 por patamar, ponderada pelas horas de patamar. GNL: a menor inflexibilidade de `TG` entre os estágios, cada um ponderado pelas suas horas de patamar (um estágio sem registro `TG` herda o último declarado, como em `CT`), para que todo compromisso dentro da capacidade do seu estágio seja aceito. |
| `thermals[].generation.max_mw` | `dadger.rvN` › `CT` › `disponibilidade`, `dadger.rvN` › `DP` › `duracao`, `dadgnl.rvN` › `TG` › `disponibilidade` | Disponibilidade do estágio 1 por patamar, ponderada pelas horas de patamar. GNL: a maior disponibilidade de `TG` entre os estágios, com a mesma ponderação e herança da inflexibilidade. |
| `thermals[].anticipated_config.lead_time_hours` | `dadgnl.rvN` › `GL` › `data_inicio`, `dadgnl.rvN` › `GS` › `mes, semanas`, `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado; condicional: somente nas térmicas GNL, quando `dadgnl` declara despacho comprometido (algum `GL` com `geracao` não nula))* | Lead físico por usina, em horas: da data de início do estudo até o fim da última semana já comandada da usina em `GL` (o corte de classe 4). Sem calendário `GS`, a duração do primeiro estágio. |
| `thermals[].entry_stage_id` | — *(sempre nulo; condicional: somente nas térmicas GNL, quando `dadgnl` declara despacho comprometido)* | Sempre nulo: térmicas GNL estão ativas em todo o horizonte. Térmicas de `CT` omitem o campo. |
| `thermals[].exit_stage_id` | — *(sempre nulo; condicional: somente nas térmicas GNL, quando `dadgnl` declara despacho comprometido)* | Sempre nulo: térmicas GNL não saem dentro do horizonte. Térmicas de `CT` omitem o campo. |

### `system/hydros.json`

**Lê:** `dadger.rvN` (`UH`, `SB`, `CT`, `DP`, `DT`, `AC`, `MP`, `FD`, `HQ`, `LQ`, `CQ`), `hidr.dat`  
**Quando:** sempre.  
**Esquema:** [hydros.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/hydros.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/hydro/entity.py`

Uma entrada por usina hidráulica operada (registro `UH` com `volume_inicial` preenchido), ordenada pelo id Novomodelo; o cadastro vem do `hidr.dat` com as modificações `AC` já resolvidas por estágio, e o bloco `diversion` é acoplado só depois da resolução dos limites.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `hydros[].id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id 0-based denso, atribuído na ordem crescente do `codigo_usina` dos registros `UH` operados (com `volume_inicial`). Registros `UH` sem volume inicial são só de acoplamento e ficam fora do conjunto operado, com aviso. |
| `hydros[].name` | `hidr.dat` › `nome_usina` | Nome cadastral da usina no `hidr.dat`, sem espaços nas pontas. |
| `hydros[].operational_start_date` | `dadger.rvN` › `DT` › `dia`, `dadger.rvN` › `DT` › `mes`, `dadger.rvN` › `DT` › `ano` *(derivado)* | Data de início do estudo (`DT`), em ISO 8601, igual para todas as usinas: nenhuma usina entra em operação dentro do horizonte. |
| `hydros[].downstream_id` | `hidr.dat` › `codigo_usina_jusante`, `dadger.rvN` › `AC` › `NUMJUS`, `dadger.rvN` › `UH` › `codigo_usina` | Id Novomodelo da primeira usina operada encontrada descendo a cascata pela jusante efetiva (`codigo_usina_jusante`, sobrescrito por `AC NUMJUS`) do estágio inicial; usinas não operadas no caminho são atravessadas e o código 0 (foz) resulta em nulo. Um `AC NUMJUS` que varia ao longo do horizonte gera aviso e só o vínculo do estágio inicial é usado. |
| `hydros[].reservoir.min_storage_hm3` | `hidr.dat` › `volume_minimo`, `dadger.rvN` › `AC` › `VOLMIN`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia` | Menor valor, ao longo dos estágios, do `volume_minimo` efetivo (`AC VOLMIN` em vigor a partir do estágio resolvido pelo trio mês/semana/ano). Usina a fio d'água (`tipo_regulacao` = D) colapsa para `volume_referencia` (ou `volume_minimo` quando ausente ou não positivo). |
| `hydros[].reservoir.max_storage_hm3` | `hidr.dat` › `volume_maximo`, `dadger.rvN` › `AC` › `VOLMAX`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia` | Maior valor, ao longo dos estágios, do `volume_maximo` efetivo (`AC VOLMAX` em vigor a partir do estágio resolvido). Usina a fio d'água (`tipo_regulacao` = D) colapsa para `volume_referencia`, igual ao mínimo; estágios com faixa mais estreita que o envelope vão para `constraints/hydro_bounds.parquet`. |
| `hydros[].outflow.min_outflow_m3s` | `dadger.rvN` › `UH` › `vazao_defluente_minima` | Vazão defluente mínima declarada no `UH` da usina; 0.0 quando o campo está em branco. O mínimo por REE do `RQ` não entra aqui: vai para `constraints/hydro_bounds.parquet`. |
| `hydros[].outflow.max_outflow_m3s` | — *(sempre nulo)* | Sempre nulo: o DECOMP não declara defluência máxima cadastral. Um teto de `QDEF` (RHQ de termo único) vai para `constraints/hydro_bounds.parquet`. |
| `hydros[].generation.model` | `hidr.dat` › `a0_volume_cota..a4_volume_cota`, `dadger.rvN` › `AC` › `COTVOL`, `hidr.dat` › `produtibilidade_especifica`, `dadger.rvN` › `AC` › `PROESP` *(derivado)* | `fpha` quando o polinômio cota-volume efetivo (pós `AC COTVOL`) do estágio inicial não é identicamente nulo, a produtibilidade específica efetiva (pós `AC PROESP`) é positiva e o engolimento e a potência nominais efetivos (maior valor ao longo dos estágios, pós `AC NUMCON`/`NUMMAQ`/`POTEFE`/`VAZEFE`) são positivos; caso contrário `constant_productivity`, e a usina excluída só pela capacidade nula é listada em diagnóstico informativo, com o ρ_eq em `system/hydro_energy_productivity.parquet`. |
| `hydros[].generation.min_turbined_m3s` | — *(constante)* | Sempre 0.0: o cadastro não traz engolimento mínimo; pisos de `QTUR` (RHQ) vão para `constraints/hydro_bounds.parquet`. |
| `hydros[].generation.max_turbined_m3s` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1..5`, `hidr.dat` › `vazao_nominal_conjunto_1..5`, `hidr.dat` › `potencia_nominal_conjunto_1..5`, `hidr.dat` › `queda_nominal_conjunto_1..5`, `hidr.dat` › `tipo_turbina`, `hidr.dat` › `a0_volume_cota..a4_volume_cota`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas`, `hidr.dat` › `tipo_perda`, `dadger.rvN` › `AC` › `NUMCON`, `dadger.rvN` › `AC` › `NUMMAQ`, `dadger.rvN` › `AC` › `VAZEFE`, `dadger.rvN` › `AC` › `POTEFE`, `dadger.rvN` › `AC` › `COTVOL`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `dadger.rvN` › `AC` › `PROESP`, `dadger.rvN` › `AC` › `JUSMED`, `dadger.rvN` › `AC` › `PERHID` | Engolimento máximo corrigido pela queda: por estágio, min(Σ máquinas × vazão nominal × (h_op/h_nom)^k, potência instalada / ρ_eq), com h_op = ρ_eq/ρ_esp, ρ_eq = ρ_esp × (cota média sobre [`volume_minimo`, `volume_maximo`] − `canal_fuga_medio`, descontadas as perdas), h_nom = `queda_nominal_conjunto_k` e k = 0,5 (Francis/Pelton) ou 0,2 (Kaplan, `tipo_turbina` = 2); máximo ao longo do horizonte, sem desconto de TEIF/IP. Para Itaipu (código 66) é a soma dos envelopes dos dois grupos. `AC ALTEFE` não é consumido (o idecomp não expõe seu valor) e só gera aviso. |
| `hydros[].generation.min_generation_mw` | — *(constante)* | Sempre 0.0: o cadastro não traz geração mínima; pisos de `RE`/`FU` de termo único vão para `constraints/hydro_bounds.parquet` e os pisos `RI` de Itaipu para `constraints/hydro_unit_group_bounds.parquet`. |
| `hydros[].generation.max_generation_mw` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1..5`, `hidr.dat` › `potencia_nominal_conjunto_1..5`, `dadger.rvN` › `AC` › `NUMCON`, `dadger.rvN` › `AC` › `NUMMAQ`, `dadger.rvN` › `AC` › `POTEFE` | Potência instalada nominal: soma, sobre os conjuntos efetivos, de máquinas × potência unitária (com `AC NUMCON`/`NUMMAQ`/`POTEFE` por estágio), tomando o máximo ao longo do horizonte; sem desconto de TEIF/IP nem de manutenção (`MP`/`FD` vão para `constraints/hydro_unit_group_bounds.parquet`). Para Itaipu (código 66) é a soma dos envelopes dos dois grupos por frequência. |
| `hydros[].unit_groups[].id` | `dadger.rvN` › `MP` › `frequencia`, `dadger.rvN` › `FD` › `frequencia` *(derivado)* | Um único grupo espelho com id 0 por usina. Itaipu (código 66) declara um grupo por frequência dos registros `MP`/`FD`, ids 0 e 1 em ordem crescente de frequência (0 = 50 Hz, conjunto 1 do cadastro; 1 = 60 Hz, conjunto 2); `MP` e `FD` precisam concordar no conjunto de frequências. |
| `hydros[].unit_groups[].name` | `hidr.dat` › `nome_usina` | Nome da usina, copiado para cada grupo. |
| `hydros[].unit_groups[].bus_id` | `hidr.dat` › `submercado`, `dadger.rvN` › `SB` › `codigo_submercado` *(derivado)* | Id da barra do submercado da usina (`submercado` do `hidr.dat`, mapeado pela ordem crescente dos `SB`). O grupo de 60 Hz de Itaipu (id 1) é colocado na barra de transbordo `IV` criada pelo conversor (corredor para Ivaiporã); o de 50 Hz fica na barra do submercado da usina. |
| `hydros[].unit_groups[].min_generation_mw` | — *(constante)* | Sempre 0.0, espelhando o envelope da usina. |
| `hydros[].unit_groups[].max_generation_mw` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1..5`, `hidr.dat` › `potencia_nominal_conjunto_1..5`, `dadger.rvN` › `AC` › `NUMCON`, `dadger.rvN` › `AC` › `NUMMAQ`, `dadger.rvN` › `AC` › `POTEFE` | Espelha `generation.max_generation_mw` da usina no grupo único. Para Itaipu, cada grupo recebe o envelope nominal (máximo ao longo dos estágios) do seu próprio conjunto; a soma dos dois é o envelope da usina por construção. |
| `hydros[].unit_groups[].min_turbined_m3s` | — *(constante)* | Sempre 0.0, espelhando o envelope da usina. |
| `hydros[].unit_groups[].max_turbined_m3s` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1..5`, `hidr.dat` › `vazao_nominal_conjunto_1..5`, `hidr.dat` › `potencia_nominal_conjunto_1..5`, `hidr.dat` › `queda_nominal_conjunto_1..5`, `hidr.dat` › `tipo_turbina`, `hidr.dat` › `a0_volume_cota..a4_volume_cota`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas`, `hidr.dat` › `tipo_perda`, `dadger.rvN` › `AC` › `NUMCON`, `dadger.rvN` › `AC` › `NUMMAQ`, `dadger.rvN` › `AC` › `VAZEFE`, `dadger.rvN` › `AC` › `POTEFE`, `dadger.rvN` › `AC` › `COTVOL`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `dadger.rvN` › `AC` › `PROESP`, `dadger.rvN` › `AC` › `JUSMED`, `dadger.rvN` › `AC` › `PERHID` | Espelha `generation.max_turbined_m3s` da usina no grupo único. Para Itaipu, cada grupo recebe o engolimento corrigido pela queda do seu próprio conjunto, com o teto de potência aplicado por conjunto (não por usina), máximo ao longo dos estágios. |
| `hydros[].specific_productivity_mw_per_m3s_per_m` | `hidr.dat` › `produtibilidade_especifica`, `dadger.rvN` › `AC` › `PROESP` *(condicional: somente para usinas com `generation.model` = `fpha`)* | Produtibilidade específica efetiva do estágio inicial (`AC PROESP` aplicado), em MW/(m³/s)/m, copiada sem conversão; o Novomodelo deriva ρ_eq dela na FPHA computada. |
| `hydros[].efficiency.type` | — *(constante; condicional: somente para usinas com `generation.model` = `fpha`)* | Sempre `constant`: o DECOMP não traz curva de rendimento por vazão ou queda. |
| `hydros[].efficiency.value` | `hidr.dat` › `produtibilidade_especifica`, `dadger.rvN` › `AC` › `PROESP` *(condicional: somente para usinas com `generation.model` = `fpha`)* | Rendimento adimensional da turbina η = ρ_esp / 0,00981 (a produtibilidade específica já embute g/1000 × η); um valor acima de 1,0 é truncado para 1,0 com aviso. |
| `hydros[].tailrace.type` | — *(constante; condicional: somente para usinas FPHA cujo `canal_fuga_medio` efetivo é positivo)* | Sempre `polynomial`: o nível de jusante de reserva é um polinômio de grau zero. |
| `hydros[].tailrace.coefficients[]` | `hidr.dat` › `canal_fuga_medio`, `dadger.rvN` › `AC` › `JUSMED` *(condicional: somente para usinas FPHA cujo `canal_fuga_medio` efetivo é positivo)* | Lista com um único coeficiente: o nível médio do canal de fuga efetivo (`AC JUSMED` aplicado) do estágio inicial, em metros. É o nível de jusante de reserva que o Novomodelo usa quando a usina não tem família em `system/tailrace_curves.parquet`. |
| `hydros[].hydraulic_losses.type` | `hidr.dat` › `tipo_perda`, `hidr.dat` › `perdas`, `dadger.rvN` › `AC` › `PERHID` *(condicional: somente para usinas com `generation.model` = `fpha`)* | `factor` quando `tipo_perda` = 1 e a perda efetiva (`AC PERHID` aplicado) é positiva; `constant` quando `tipo_perda` = 2 e a perda é positiva; nos demais casos `factor` com valor 0.0, porque o Novomodelo exige o bloco em toda usina FPHA. |
| `hydros[].hydraulic_losses.value` | `hidr.dat` › `perdas`, `hidr.dat` › `tipo_perda`, `dadger.rvN` › `AC` › `PERHID` *(condicional: somente para usinas FPHA com `hydraulic_losses.type` = `factor`)* | Fração da queda bruta perdida: `perdas`/100 quando `tipo_perda` = 1; 0.0 quando a usina não tem perda modelada. |
| `hydros[].hydraulic_losses.value_m` | `hidr.dat` › `perdas`, `hidr.dat` › `tipo_perda`, `dadger.rvN` › `AC` › `PERHID` *(condicional: somente para usinas FPHA com `tipo_perda` = 2 e perda positiva)* | Perda de carga constante em metros, igual ao `perdas` efetivo (`AC PERHID` aplicado). |
| `hydros[].evaporation.coefficients_mm[]` | `dadger.rvN` › `UH` › `evaporacao`, `hidr.dat` › `evaporacao_JAN..evaporacao_DEZ` *(condicional: somente quando o `UH` da usina liga o flag `evaporacao` e o `hidr.dat` traz ao menos um coeficiente mensal não nulo)* | Os 12 coeficientes mensais de evaporação (mm/mês) do `hidr.dat`, de janeiro (índice 0) a dezembro, sem conversão; o Novomodelo os combina com a área de `system/hydro_geometry.parquet` e rateia cada estágio pela sua fração de horas do mês. `AC COFEVA` não é consumido. |
| `hydros[].diversion.downstream_id` | `hidr.dat` › `desvio`, `dadger.rvN` › `AC` › `DESVIO`, `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `LQ` › `limite_inferior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo` *(condicional: somente quando a usina tem piso positivo em `min_diversion_m3s` (RHQ `QDES` de termo único) com canal resolúvel, ou um `desvio` de base sem limite explícito cuja usina receptora é operada)* | Id Novomodelo da usina que recebe a água desviada: o `codigo_usina_jusante` do primeiro `AC DESVIO` em vigor ou, na falta dele, o `desvio` do `hidr.dat`. Um piso de desvio sem canal resolúvel (limite ausente ou receptora não operada) gera aviso e o bloco fica nulo; uma usina com desvio mas sem piso nem canal de base fica com `diversion` nulo. |
| `hydros[].diversion.max_flow_m3s` | `dadger.rvN` › `AC` › `DESVIO`, `hidr.dat` › `desvio` *(condicional: somente quando a usina tem piso positivo em `min_diversion_m3s` (RHQ `QDES` de termo único) com canal resolúvel, ou um `desvio` de base sem limite explícito cuja usina receptora é operada)* | `limite_vazao` do `AC DESVIO` em vigor. Para um `desvio` de base sem limite explícito, o canal é limitado pelo `generation.max_turbined_m3s` da usina receptora, para que a água desviada chegue a ela (caso MOXOTÓ → P.AFONSO 4); esse teto também vira `max_diversion_m3s` em `constraints/hydro_bounds.parquet`. |

### `system/hydro_production_models.json`

**Lê:** `dadger.rvN` (`UH`, `SB`, `CT`, `DP`, `DT`, `AC`), `hidr.dat`  
**Quando:** sempre.  
**Esquema:** [production_models.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/production_models.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/hydro/productivity.py`

Uma entrada por usina operada com um único intervalo de estágios cobrindo todo o horizonte; as usinas elegíveis recebem a FPHA computada pelo Novomodelo ajustada em torno do volume inicial, as demais produtibilidade constante.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `production_models[].hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id Novomodelo da usina, na mesma ordem de `system/hydros.json`. |
| `production_models[].selection_mode` | — *(constante)* | Sempre `stage_ranges`: o modelo é escolhido por intervalo de estágios. |
| `production_models[].stage_ranges[].start_stage_id` | — *(constante)* | Sempre 0: o intervalo único começa no primeiro estágio. |
| `production_models[].stage_ranges[].end_stage_id` | — *(constante)* | Sempre nulo: o intervalo único é aberto e vale até o fim do horizonte. |
| `production_models[].stage_ranges[].model` | `hidr.dat` › `a0_volume_cota..a4_volume_cota`, `dadger.rvN` › `AC` › `COTVOL`, `hidr.dat` › `produtibilidade_especifica`, `dadger.rvN` › `AC` › `PROESP` *(derivado)* | `fpha` para as usinas com polinômio cota-volume efetivo não nulo e produtibilidade específica positiva no estágio inicial e engolimento e potência nominais efetivos positivos; `constant_productivity` para as demais. É o mesmo critério de `generation.model` em `system/hydros.json`. |
| `production_models[].stage_ranges[].fpha_config.source` | — *(constante; condicional: somente para usinas com modelo `fpha`)* | Sempre `computed`: o Novomodelo ajusta os hiperplanos da FPHA a partir de `system/hydro_geometry.parquet` e `system/tailrace_curves.parquet`; os registros `FP` do DECOMP não são lidos. |
| `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3` | `dadger.rvN` › `UH` › `volume_inicial`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia` *(condicional: somente para usinas com modelo `fpha`)* | max(V_min, V_ini − 0,10 × (V_max − V_min)), com V_ini o volume inicial da usina em hm³ e [V_min, V_max] a faixa efetiva do estágio inicial. A meia-largura de 10 % do volume útil é um parâmetro de modelagem do novomodelo-bridge; fio d'água colapsa para o volume de referência. |
| `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3` | `dadger.rvN` › `UH` › `volume_inicial`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia` *(condicional: somente para usinas com modelo `fpha`)* | min(V_max, V_ini + 0,10 × (V_max − V_min)), com V_ini o volume inicial da usina em hm³ e [V_min, V_max] a faixa efetiva do estágio inicial (mesma construção de `volume_min_hm3`). |
| `production_models[].stage_ranges[].reference_volume.volume_hm3` | `dadger.rvN` › `UH` › `volume_inicial`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia` *(condicional: somente para usinas com modelo `fpha`)* | Volume inicial em hm³: V_min + `volume_inicial`/100 × (V_max − V_min) na faixa efetiva do estágio inicial; fio d'água colapsa para `volume_referencia`. É o mesmo valor de `storage` em `initial_conditions.json`, o ponto em que o DECOMP ancora a FPHA. |

### `system/hydro_geometry.parquet`

**Lê:** `dadger.rvN` (`UH`, `SB`, `CT`, `DP`, `DT`, `AC`), `hidr.dat`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/converters/fpha.py`

Curva volume → cota → área amostrada em 100 pontos por usina operada com polinômio cota-volume não nulo (também para as de produtibilidade constante, que servem de referência de remanso); usina a fio d'água contribui um único ponto.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id Novomodelo da usina; usinas com polinômio cota-volume identicamente nulo não têm linhas. |
| `volume_hm3` | `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia` *(derivado)* | Malha uniforme de 100 volumes sobre a faixa efetiva [`volume_minimo`, `volume_maximo`] do estágio inicial; faixa degenerada (fio d'água) produz um único ponto. |
| `height_m` | `hidr.dat` › `a0_volume_cota..a4_volume_cota`, `dadger.rvN` › `AC` › `COTVOL` | Cota de montante do polinômio cota-volume efetivo (pós `AC COTVOL`, ordem 1..5 do registro mapeada em a0..a4) avaliado em cada volume da malha, truncada em 0. |
| `area_km2` | `hidr.dat` › `a0_cota_area..a4_cota_area` | Área do reservatório pelo polinômio cota-área do `hidr.dat` avaliado na cota de cada ponto, truncada em 0. `AC COTARE` não é consumido. |

### `system/hydro_energy_productivity.parquet`

**Lê:** `dadger.rvN` (`UH`, `SB`, `CT`, `DP`, `DT`, `AC`), `hidr.dat`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/converters/hydro/productivity.py`

Uma linha por usina operada com produtibilidade constante (as usinas FPHA são omitidas, pois o Novomodelo calcula o ρ_eq delas); valor único para todo o horizonte, ancorado no volume inicial.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id Novomodelo da usina, só para as de `generation.model` = `constant_productivity`. |
| `stage_id` | — *(sempre nulo)* | Sempre nulo: o valor vale para todos os estágios; a variação de queda por estágio entra pelo teto de engolimento em `constraints/hydro_unit_group_bounds.parquet`. |
| `equivalent_productivity_mw_per_m3s` | `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota..a4_volume_cota`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas`, `hidr.dat` › `tipo_perda`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia`, `dadger.rvN` › `UH` › `volume_inicial`, `dadger.rvN` › `AC` › `PROESP`, `dadger.rvN` › `AC` › `COTVOL`, `dadger.rvN` › `AC` › `JUSMED`, `dadger.rvN` › `AC` › `PERHID`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX` | ρ_eq = ρ_esp × h_líq, com a cota de montante avaliada no volume inicial da usina (o ponto em que o DECOMP ancora a FPHA), menos `canal_fuga_medio` e a perda hidráulica (`tipo_perda` 1 = percentual, 2 = metros), queda líquida truncada em 0; todos os cadastrais são os efetivos do estágio inicial (`AC` aplicados). Polinômio cota-volume nulo gera aviso e ρ_eq = 0. |
| `reference_outflow_m3s` | — *(sempre nulo)* | Sempre nulo: o ρ_eq é uma constante, sem vazão de referência; a coluna existe porque o leitor exige o conjunto completo de colunas. |
| `specific_productivity_mw_per_m3s_per_m` | — *(sempre nulo)* | Sempre nulo: a produtibilidade específica só é emitida, em `system/hydros.json`, para as usinas FPHA, que não têm linha nesta tabela. |

### `system/tailrace_curves.parquet`

**Lê:** `polinjus.csv` (opcional), `dadger.rvN` (`UH`, `SB`, `CT`)  
**Quando:** somente quando o deck traz `polinjus.csv` e ao menos um segmento de curva de jusante pertence a uma usina operada  
**Código:** `src/novomodelo_bridge/decomp/converters/fpha.py`

Famílias de curvas de nível de jusante por usina (uma linha por segmento polinomial), copiadas do `polinjus.csv` para as usinas operadas e ordenadas por id Novomodelo, família e segmento; segmentos de usinas não operadas são descartados.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · codigo_usina`, `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id Novomodelo da usina do segmento; códigos ausentes do conjunto operado são descartados. |
| `family_id` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · indice_familia` *(derivado)* | Índice da família de curvas, copiado do `polinjus.csv`. |
| `downstream_reference_level_m` | `polinjus.csv` › `hidreletrica_curvajusante · nivel_montante_referencia` | Nível de montante de referência da família (metros), trazido para cada segmento da família; em branco vira nulo. |
| `segment_id` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · indice_polinomio` *(derivado)* | Índice do polinômio dentro da família, copiado do `polinjus.csv`. |
| `outflow_min_m3s` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · limite_inferior_vazao_jusante` | Limite inferior de vazão de jusante do segmento, sem conversão. |
| `outflow_max_m3s` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · limite_superior_vazao_jusante` | Limite superior de vazão de jusante do segmento, sem conversão. |
| `coefficient_0` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a0` | Coeficiente a0 do polinômio vazão → nível de jusante, sem conversão. |
| `coefficient_1` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a1` | Coeficiente a1 do polinômio vazão → nível de jusante, sem conversão. |
| `coefficient_2` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a2` | Coeficiente a2 do polinômio vazão → nível de jusante, sem conversão. |
| `coefficient_3` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a3` | Coeficiente a3 do polinômio vazão → nível de jusante, sem conversão. |
| `coefficient_4` | `polinjus.csv` › `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a4` | Coeficiente a4 do polinômio vazão → nível de jusante, sem conversão. |

### `system/non_controllable_sources.json`

**Lê:** `dadger.rvN` (`PQ`, `SB`, `UH`, `CT`, `DP`, `DT`), `renovaveis*` (opcional)  
**Quando:** sempre.  
**Esquema:** [non_controllable_sources.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/non_controllable_sources.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/ncs.py`

Uma fonte não controlável por série `PQ` (par submercado + nome), mais um parque por `codigo_pee` do arquivo de renováveis quando presente; todas must-run, pois a geração é abatida da carga na origem.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `non_controllable_sources[].id` | `dadger.rvN` › `PQ` › `codigo_submercado, nome`, `renovaveis*` › `PEE-GER-PER-PAT-CEN · codigo_pee` *(derivado)* | Séries `PQ` ordenadas por (`codigo_submercado`, `nome`) recebem ids 0-based densos; os parques `PEE` continuam a numeração em ordem crescente de `codigo_pee`. |
| `non_controllable_sources[].name` | `dadger.rvN` › `PQ` › `nome, codigo_submercado`, `renovaveis*` › `PEE-CAD · nome_pee`, `renovaveis*` › `PEE-SUBM · codigo_submercado` *(derivado)* | `{nome}_{codigo_submercado}` para `PQ`; `{nome_pee}_{codigo_submercado}` para os parques (`PEE_{codigo}` sem cadastro de nome). |
| `non_controllable_sources[].operational_start_date` | `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado)* | Data de início do estudo. |
| `non_controllable_sources[].bus_id` | `dadger.rvN` › `PQ` › `codigo_submercado`, `renovaveis*` › `PEE-SUBM · codigo_submercado` | Barra do submercado da série; parques resolvem pelo cartão `PEE-SUBM`. |
| `non_controllable_sources[].max_generation_mw` | `dadger.rvN` › `PQ` › `geracao`, `renovaveis*` › `PEE-GER-PER-PAT-CEN · geracao` | Maior geração por patamar (`geracao_k`, em branco lido como 0) em todos os estágios, após preencher estágios não declarados com o último declarado (estágio 1 obrigatório). Parques: o valor modal entre cenários de cada (estágio, patamar), com aviso se os cenários divergirem. O perfil por estágio e patamar vai para as estatísticas e fatores de NCS. |
| `non_controllable_sources[].allow_curtailment` | — *(constante)* | Sempre `false`: a geração é abatida da carga na origem, logo é must-run e fixada na disponibilidade. |

### `system/pumping_stations.json`

**Lê:** `dadger.rvN` (`UE`, `SB`, `UH`, `CT`, `DP`, `DT`)  
**Quando:** sempre.  
**Esquema:** [pumping_stations.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/pumping_stations.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/network.py`

Uma estação de bombeamento por registro `UE` (1:1); a água é elevada da usina de jusante para a de montante, ambas obrigatoriamente operadas. Sem `UE`, lista vazia.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `pumping_stations[].id` | `dadger.rvN` › `UE` › `codigo_usina` *(derivado)* | Id 0-based denso na ordem crescente de `codigo_usina` de `UE`; o mesmo mapa resolve os termos `QBOM` de `CQ`. |
| `pumping_stations[].name` | `dadger.rvN` › `UE` › `nome_usina` | Nome da estação sem espaços laterais. |
| `pumping_stations[].operational_start_date` | `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado)* | Data de início do estudo. |
| `pumping_stations[].bus_id` | `dadger.rvN` › `UE` › `codigo_submercado` | Barra do submercado declarado, que paga o consumo do bombeamento. |
| `pumping_stations[].source_hydro_id` | `dadger.rvN` › `UE` › `codigo_usina_jusante` | Id da usina de jusante (de onde a água é retirada), pelo mapa de ids de `UH`. |
| `pumping_stations[].destination_hydro_id` | `dadger.rvN` › `UE` › `codigo_usina_montante` | Id da usina de montante (que recebe a água bombeada), pelo mapa de ids de `UH`. |
| `pumping_stations[].consumption_mw_per_m3s` | `dadger.rvN` › `UE` › `taxa_consumo` | Taxa de consumo de energia por vazão bombeada, sem conversão. |
| `pumping_stations[].flow.min_m3s` | `dadger.rvN` › `UE` › `vazao_minima_bombeavel` | Vazão mínima bombeável, sem conversão. |
| `pumping_stations[].flow.max_m3s` | `dadger.rvN` › `UE` › `vazao_maxima_bombeavel` | Vazão máxima bombeável, sem conversão. |

### `system/energy_contracts.json`

**Lê:** `dadger.rvN` (`CI`, `CE`, `SB`, `UH`, `CT`, `DP`, `DT`)  
**Quando:** somente quando `dadger` declara ao menos um contrato `CI`/`CE` real (linha com nome ou com algum limite não nulo)  
**Esquema:** [energy_contracts.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/energy_contracts.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/contracts.py`

Contratos de importação (`CI`) e exportação (`CE`) de energia com os valores do estágio 1; uma linha sem nome e com todos os limites nulos é um placeholder e é ignorada. O `fator_perdas` não tem contraparte no Novomodelo e gera apenas um aviso.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `contracts[].id` | `dadger.rvN` › `CI` › `numero_contrato`, `dadger.rvN` › `CE` › `numero_contrato` *(derivado)* | Id 0-based denso: importações antes de exportações, cada grupo em ordem crescente de `numero_contrato` (os dois espaços de numeração colidem). |
| `contracts[].name` | `dadger.rvN` › `CI` › `nome_contrato`, `dadger.rvN` › `CE` › `nome_contrato` | Nome do contrato sem espaços laterais; `CI {numero}` ou `CE {numero}` quando em branco. |
| `contracts[].operational_start_date` | `dadger.rvN` › `DT` › `dia, mes, ano` *(derivado)* | Data de início do estudo; todo contrato é sempre ativo (`entry_stage_id`/`exit_stage_id` omitidos). |
| `contracts[].bus_id` | `dadger.rvN` › `CI` › `codigo_submercado`, `dadger.rvN` › `CE` › `codigo_submercado` | Barra do submercado declarado. |
| `contracts[].type` | — *(derivado)* | `import` para registros `CI`, `export` para registros `CE`. |
| `contracts[].price_per_mwh` | `dadger.rvN` › `CI` › `custo`, `dadger.rvN` › `CE` › `custo`, `dadger.rvN` › `DP` › `duracao` | `custo_k` do estágio 1 ponderado pelas horas de patamar; negativo nas exportações (receita), positivo nas importações. |
| `contracts[].limits.min_mw` | `dadger.rvN` › `CI` › `limite_inferior`, `dadger.rvN` › `CE` › `limite_inferior`, `dadger.rvN` › `DP` › `duracao` | `limite_inferior_k` do estágio 1 ponderado pelas horas de patamar (em branco lido como 0). |
| `contracts[].limits.max_mw` | `dadger.rvN` › `CI` › `limite_superior`, `dadger.rvN` › `CE` › `limite_superior`, `dadger.rvN` › `DP` › `duracao` | `limite_superior_k` do estágio 1 ponderado pelas horas de patamar (em branco lido como 0). |

### `scenarios/inflow_seasonal_stats.parquet`

**Lê:** `dadger.rvN` (`SB`, `CT`, `UH`, `DP`, `DT`)  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/scenarios.py`

Tabela identidade (média 0, desvio 1) por usina hidráulica e estágio: sob a convenção de vazões explícitas, o ruído padronizado é a própria vazão incremental de `external_inflow_scenarios.parquet`.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id 0-based denso das usinas operadas (registros `UH` com `volume_inicial` preenchido), em ordem crescente de `codigo_usina`. |
| `stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio do calendário operativo (`estagio` − 1); uma linha por usina e estágio. |
| `mean_m3s` | — *(constante)* | Sempre 0.0: média nula da convenção identidade, para que a vazão externa entre sem deslocamento. |
| `std_m3s` | — *(constante)* | Sempre 1.0: desvio unitário da convenção identidade, evitando a patologia de desvio zero no leque terminal. |

### `scenarios/external_inflow_scenarios.parquet`

**Lê:** `vazoes.rvN`, `hidr.dat`, `dadger.rvN` (`SB`, `CT`, `UH`, `DP`, `DT`, `AC`)  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/scenarios.py`

Árvore explícita de vazões incrementais por usina: o tronco determinístico (`previsoes`, uma coluna por estágio semanal) e o leque terminal (`cenarios_gerados`, uma coluna por cenário no último estágio).

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `stage_id` | `vazoes.rvN` › `previsoes · estagio`, `vazoes.rvN` › `cenarios_gerados · estagio` *(derivado)* | `estagio` − 1. As previsões devem cobrir exatamente os estágios semanais e os cenários gerados devem estar todos no estágio terminal; caso contrário a conversão falha. |
| `scenario_id` | `vazoes.rvN` › `cenarios_gerados · cenario` *(derivado)* | 0 em todo estágio do tronco; `cenario` − 1 no estágio terminal. |
| `hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id 0-based denso das usinas operadas, em ordem crescente de `codigo_usina`. |
| `value_m3s` | `vazoes.rvN` › `previsoes · coluna do posto`, `vazoes.rvN` › `cenarios_gerados · coluna do posto`, `hidr.dat` › `posto`, `dadger.rvN` › `AC` › `NUMPOS · codigo_posto` | Vazão incremental lida diretamente da coluna do posto da usina (`posto` do `hidr`, sobreposto por `AC NUMPOS` quando presente, avaliado no estágio inicial), sem subtração de montante: o arquivo de vazões do DECOMP já é incremental. Um `AC NUMPOS` que varia ao longo do horizonte gera aviso e usa o posto do estágio inicial. |

### `scenarios/load_seasonal_stats.parquet`

**Lê:** `dadger.rvN` (`DP`, `SB`, `CT`, `UH`, `DT`, `RI`), `hidr.dat`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/load.py`

Uma linha por barra (inclusive a barra de transbordo `IV`) e estágio, com a carga média do estágio e desvio zero: a carga do DECOMP é determinística.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `bus_id` | `dadger.rvN` › `SB` › `codigo_submercado` *(derivado)* | Id 0-based dos submercados `SB` em ordem crescente de código, seguido da barra de transbordo `IV` (última), criada pelo conversor e ausente do `DP`. |
| `stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio (`estagio` − 1); o número de patamares do `DP` deve coincidir com o calendário. |
| `mean_mw` | `dadger.rvN` › `DP` › `carga`, `dadger.rvN` › `DP` › `duracao`, `dadger.rvN` › `RI` › `carga_ande`, `hidr.dat` › `submercado` | Média do estágio ponderada pelas horas dos patamares, `Σ_b carga_b·duracao_b / Σ_b duracao_b`; campo de carga em branco lê 0. Quando Itaipu é operada e o `RI` declara `carga_ande`, esse valor por patamar (herdado adiante entre estágios) soma-se à carga `DP` do submercado de Itaipu (`submercado` do `hidr`); uma barra sem linha no `DP` recebe 0. |
| `std_mw` | — *(constante)* | Sempre 0.0: a carga do DECOMP é determinística, sem espalhamento estocástico. |

### `scenarios/load_factors.json`

**Lê:** `dadger.rvN` (`DP`, `SB`, `CT`, `UH`, `DT`, `RI`), `hidr.dat`  
**Quando:** sempre.  
**Esquema:** [load_factors.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/load_factors.schema.json) · **Código:** `src/novomodelo_bridge/decomp/load.py`

Perfil por patamar da carga de cada (barra, estágio) com carga positiva; um par de média zero (submercado fictício, barra `IV`) não recebe entrada.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `load_factors[].bus_id` | `dadger.rvN` › `SB` › `codigo_submercado` *(derivado)* | Id 0-based do submercado, o mesmo de `load_seasonal_stats.parquet`; entradas ordenadas por (barra, estágio). |
| `load_factors[].stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio (`estagio` − 1). |
| `load_factors[].block_factors[].block_id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Índice 0-based do patamar, de 0 a `numero_patamares` − 1, na ordem dos campos `carga`. |
| `load_factors[].block_factors[].factor` | `dadger.rvN` › `DP` › `carga`, `dadger.rvN` › `DP` › `duracao`, `dadger.rvN` › `RI` › `carga_ande` | `carga_b / média do estágio`, com a mesma carga por patamar (inclusive `carga_ande` somada à barra de Itaipu) usada em `mean_mw`; a invariante `Σ_b fator_b·duracao_b = horas do estágio` é verificada e uma violação aborta a conversão. |

### `scenarios/external_load_scenarios.parquet`

**Lê:** `dadger.rvN` (`DP`, `SB`, `CT`, `UH`, `DT`, `RI`), `hidr.dat`, `vazoes.rvN`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/scenarios.py`

Carga determinística replicada nas colunas de cenário da biblioteca de vazões (uma no tronco, a largura do leque no estágio terminal), para que toda classe estocástica seja externa com `scenario_id` coerente.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio, herdado de `load_seasonal_stats.parquet`; linhas ordenadas por (estágio, cenário, barra). |
| `scenario_id` | `vazoes.rvN` › `probabilidades · cenario` *(derivado)* | 0 em todo estágio do tronco; 0..N−1 no estágio terminal, com N igual ao número de cenários com probabilidade declarada para o último estágio em `vazoes`. |
| `bus_id` | `dadger.rvN` › `SB` › `codigo_submercado` *(derivado)* | O mesmo id de `load_seasonal_stats.parquet`, incluindo a barra de transbordo `IV` (com carga zero). |
| `value_mw` | `dadger.rvN` › `DP` › `carga`, `dadger.rvN` › `DP` › `duracao`, `dadger.rvN` › `RI` › `carga_ande`, `hidr.dat` › `submercado` | O valor de `mean_mw` de `load_seasonal_stats.parquet` (média do estágio ponderada pelas horas dos patamares, com `carga_ande` somada à barra de Itaipu), repetido sem alteração em cada coluna de cenário do estágio. |

### `scenarios/non_controllable_stats.parquet`

**Lê:** `dadger.rvN` (`PQ`, `SB`, `CT`, `UH`, `DP`, `DT`), `renovaveis*` (opcional)  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/converters/ncs.py`

Fração de disponibilidade por fonte não controlável (séries `PQ` e parques `PEE` de `renovaveis`) e estágio, com desvio zero.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `ncs_id` | `dadger.rvN` › `PQ` › `codigo_submercado`, `dadger.rvN` › `PQ` › `nome`, `renovaveis*` › `PEE-GER-PER-PAT-CEN · codigo_pee` *(derivado)* | Id 0-based denso: séries `PQ` ordenadas por (`codigo_submercado`, `nome`), seguidas dos parques `PEE` com geração declarada, em ordem crescente de `codigo_pee`. |
| `stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio; uma linha por fonte e estágio do calendário. |
| `mean` | `dadger.rvN` › `PQ` › `geracao`, `dadger.rvN` › `PQ` › `estagio`, `dadger.rvN` › `DP` › `duracao`, `renovaveis*` › `PEE-GER-PER-PAT-CEN · geracao` | Média do estágio ponderada pelas horas dos patamares dividida pela maior geração da série em todo o horizonte (o `max_generation_mw` da fonte); estágios não declarados herdam o anterior e o estágio 1 é obrigatório. Limitada a 1.0 pela tolerância do novomodelo; série de máximo zero lê 0. A geração `PEE` é lida do valor modal entre cenários, com aviso se divergirem. |
| `std` | — *(constante)* | Sempre 0.0: a geração não simulada do DECOMP é determinística. |

### `scenarios/non_controllable_factors.json`

**Lê:** `dadger.rvN` (`PQ`, `SB`, `CT`, `UH`, `DP`, `DT`), `renovaveis*` (opcional)  
**Quando:** sempre.  
**Esquema:** [non_controllable_factors.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/non_controllable_factors.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/ncs.py`

Perfil por patamar de cada fonte não controlável nos estágios de média positiva; um estágio de média zero não recebe entrada.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `non_controllable_factors[].ncs_id` | `dadger.rvN` › `PQ` › `codigo_submercado`, `dadger.rvN` › `PQ` › `nome`, `renovaveis*` › `PEE-GER-PER-PAT-CEN · codigo_pee` *(derivado)* | O mesmo id de `non_controllable_stats.parquet`: séries `PQ` ordenadas por (`codigo_submercado`, `nome`) e depois os parques `PEE` por `codigo_pee`. |
| `non_controllable_factors[].stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio. |
| `non_controllable_factors[].block_factors[].block_id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Índice 0-based do patamar, na ordem dos campos `geracao` do `PQ` ou do `patamar` do `PEE-GER-PER-PAT-CEN`. |
| `non_controllable_factors[].block_factors[].factor` | `dadger.rvN` › `PQ` › `geracao`, `dadger.rvN` › `DP` › `duracao`, `renovaveis*` › `PEE-GER-PER-PAT-CEN · geracao` | `geracao_b / média do estágio` por patamar; a invariante `Σ_b fator_b·duracao_b = horas do estágio` é verificada. Um patamar de geração zero sob média positiva recebe 1e-9, porque o esquema do novomodelo exige fator > 0 (aviso com a contagem). |

### `scenarios/external_ncs_scenarios.parquet`

**Lê:** `dadger.rvN` (`PQ`, `SB`, `CT`, `UH`, `DP`, `DT`), `renovaveis*` (opcional), `vazoes.rvN`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/scenarios.py`

Disponibilidade determinística das fontes não controláveis replicada nas colunas de cenário da biblioteca de vazões (uma no tronco, a largura do leque no estágio terminal), para que toda classe estocástica seja externa com `scenario_id` coerente.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio, herdado de `non_controllable_stats.parquet`; linhas ordenadas por (estágio, cenário, fonte). |
| `scenario_id` | `vazoes.rvN` › `probabilidades · cenario` *(derivado)* | 0 em todo estágio do tronco; 0..N−1 no estágio terminal, com N igual ao número de cenários com probabilidade declarada para o último estágio em `vazoes`. |
| `ncs_id` | `dadger.rvN` › `PQ` › `codigo_submercado`, `dadger.rvN` › `PQ` › `nome`, `renovaveis*` › `PEE-GER-PER-PAT-CEN · codigo_pee` *(derivado)* | O mesmo id de `non_controllable_stats.parquet`. |
| `availability_factor` | `dadger.rvN` › `PQ` › `geracao`, `dadger.rvN` › `DP` › `duracao`, `renovaveis*` › `PEE-GER-PER-PAT-CEN · geracao` | O valor de `mean` de `non_controllable_stats.parquet` (fração de disponibilidade do estágio), repetido sem alteração em cada coluna de cenário do estágio. |

### `constraints/hydro_bounds.parquet`

**Lê:** `dadger.rvN` (`UH`, `SB`, `CT`, `DP`, `DT`, `AC`, `RQ`, `VE`, `TI`, `RE`, `LU`, `FU`, `FT`, `FI`, `HQ`, `LQ`, `CQ`, `HV`, `LV`, `CV`), `hidr.dat`  
**Quando:** somente quando ao menos uma contribuição de limite é produzida (RQ/UH, VE, TI, faixa de armazenamento por estágio, restrição RE/RHQ/RHV de termo único ou canal de desvio de base)  
**Código:** `src/novomodelo_bridge/decomp/converters/bounds.py`

Sobrescritas por (usina, estágio, patamar) dos limites declarados em `system/hydros.json`. Toda contribuição (defluência mínima RQ/UH, faixa de armazenamento por estágio, volume de espera VE, irrigação TI, restrições RE/RHQ/RHV de termo único e coeficiente ±1, canal de desvio de base) é interseccionada por eixo (máximo dos pisos, mínimo dos tetos) e cada célula vira uma única linha; um estágio com contribuição por patamar é materializado em uma linha por patamar, sem linha-base. `FT`/`FI` entram só para decidir se uma `RE` tem termo único.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id Novomodelo da usina; linhas ordenadas por usina, estágio e patamar. |
| `stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio no calendário operativo (`estagio` − 1); os registros com trio mês/semana/ano (`AC`) e os limites por `estagio` (`LU`/`LQ`/`LV`, `VE`, `TI`) são resolvidos para esse índice. |
| `block_id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Nulo na linha-base (vale para todos os patamares); 0..n−1 quando algum contribuinte do eixo naquele estágio é por patamar (`LU`/`LQ` por patamar): então todos os patamares do estágio são materializados. O piso `RQ`/`UH` de defluência é por estágio (linha-base). Os eixos de armazenamento e retirada de água são sempre de estágio. |
| `min_outflow_m3s` | `dadger.rvN` › `UH` › `vazao_defluente_minima`, `dadger.rvN` › `UH` › `codigo_ree`, `dadger.rvN` › `RQ` › `codigo_ree`, `dadger.rvN` › `RQ` › `vazao`, `hidr.dat` › `vazao_minima_historica`, `dadger.rvN` › `AC` › `VAZMIN`, `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `LQ` › `limite_inferior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `codigo_usina_jusante`, `dadger.rvN` › `AC` › `NUMJUS` | Piso de defluência: a `vazao_defluente_minima` do `UH` (fixa em todos os estágios) ou, na sua ausência, o percentual por estágio do `RQ` da REE da usina (`vazao_1` → estágio 0, `vazao_2` → estágio 1, …, repetindo o último declarado além das colunas do registro) aplicado à `vazao_minima_historica` efetiva (`AC VAZMIN`), um piso por (usina, estágio); valores não positivos não geram linha. Um piso `QDEF` de RHQ de termo único (`LQ` limite inferior por patamar, herdado entre estágios) coexiste e o maior piso prevalece. O piso padrão `RQ`/`UH` é liberado para as usinas a fio d'água desde a cabeceira até (exclusive) o primeiro reservatório da cascata (`tipo_regulacao` M ou S; jusante efetiva `codigo_usina_jusante` com `AC NUMJUS`): essas usinas não recebem piso `RQ`/`UH`. Uma janela `QDEF` de RHQ é uma restrição declarada pelo usuário e não é liberada — continua valendo. |
| `max_outflow_m3s` | `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `HQ` › `estagio_inicial`, `dadger.rvN` › `HQ` › `estagio_final`, `dadger.rvN` › `LQ` › `limite_superior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente` | Teto de defluência de uma RHQ de termo único sobre `QDEF` com coeficiente ±1: `limite_superior` do `LQ` por patamar, herdado entre estágios dentro de [`estagio_inicial`, `estagio_final`]; coeficiente −1 troca e inverte os lados. RHQ com vários termos ou coeficiente diferente de ±1 vira restrição genérica. |
| `min_turbined_m3s` | `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `LQ` › `limite_inferior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente` | Piso de turbinamento de uma RHQ de termo único sobre `QTUR`: `limite_inferior` do `LQ` por patamar, herdado entre estágios. |
| `max_turbined_m3s` | `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `LQ` › `limite_superior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente` | Teto de turbinamento de uma RHQ de termo único sobre `QTUR`: `limite_superior` do `LQ` por patamar, truncado ao `generation.max_turbined_m3s` declarado da usina (o Novomodelo rejeita um teto acima da capacidade); um truncamento material gera aviso. |
| `min_generation_mw` | `dadger.rvN` › `RE` › `codigo_restricao`, `dadger.rvN` › `RE` › `estagio_inicial`, `dadger.rvN` › `RE` › `estagio_final`, `dadger.rvN` › `LU` › `limite_inferior`, `dadger.rvN` › `FU` › `codigo_usina`, `dadger.rvN` › `FU` › `coeficiente` | Piso de geração de uma RE de termo único com `FU` de coeficiente ±1 (sem termos `FT`/`FI`): `limite_inferior` do `LU` por patamar, herdado entre estágios dentro de [`estagio_inicial`, `estagio_final`]. |
| `max_generation_mw` | `dadger.rvN` › `RE` › `codigo_restricao`, `dadger.rvN` › `RE` › `estagio_inicial`, `dadger.rvN` › `RE` › `estagio_final`, `dadger.rvN` › `LU` › `limite_superior`, `dadger.rvN` › `FU` › `codigo_usina`, `dadger.rvN` › `FU` › `coeficiente` | Teto de geração de uma RE de termo único com `FU` de coeficiente ±1: `limite_superior` do `LU` por patamar, truncado ao `generation.max_generation_mw` declarado da usina (caso BELO MONTE, RE de 11000 MW acima da capacidade); um truncamento material gera aviso. |
| `min_storage_hm3` | `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia`, `dadger.rvN` › `HV` › `codigo_restricao`, `dadger.rvN` › `HV` › `estagio_inicial`, `dadger.rvN` › `HV` › `estagio_final`, `dadger.rvN` › `LV` › `limite_inferior`, `dadger.rvN` › `CV` › `codigo_usina`, `dadger.rvN` › `CV` › `tipo`, `dadger.rvN` › `CV` › `coeficiente` | Piso de armazenamento por estágio, sem patamar: o `volume_minimo` efetivo do estágio quando a faixa efetiva difere do envelope declarado em `system/hydros.json` (um `AC VOLMIN`/`VOLMAX` temporal), e/ou o `limite_inferior` do `LV` de uma RHV de termo único sobre `VARM`, que é relativo ao volume útil e é somado ao piso efetivo do estágio para virar hm³ absoluto. |
| `max_storage_hm3` | `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia`, `dadger.rvN` › `VE` › `codigo_usina`, `dadger.rvN` › `VE` › `volume`, `dadger.rvN` › `HV` › `codigo_restricao`, `dadger.rvN` › `LV` › `limite_superior`, `dadger.rvN` › `CV` › `codigo_usina`, `dadger.rvN` › `CV` › `tipo`, `dadger.rvN` › `CV` › `coeficiente` | Teto de armazenamento por estágio: o `volume_maximo` efetivo do estágio quando difere do envelope; o volume de espera `VE` (percentual do volume útil por estágio, `volume_k` → estágio k−1) convertido em hm³ absoluto e emitido só quando fica abaixo do envelope; e o `limite_superior` do `LV` de uma RHV de termo único sobre `VARM`, somado ao piso efetivo do estágio. O menor teto prevalece. |
| `min_diversion_m3s` | `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `LQ` › `limite_inferior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente`, `hidr.dat` › `desvio`, `dadger.rvN` › `AC` › `DESVIO` | Piso de vazão desviada de uma RHQ de termo único sobre `QDES`: `limite_inferior` do `LQ` por patamar. Um `desvio` de base sem limite RHQ recebe piso 0.0 em todos os estágios junto com o teto de canal. Um piso positivo obriga o bloco `diversion` em `system/hydros.json`. |
| `max_diversion_m3s` | `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `LQ` › `limite_superior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente`, `hidr.dat` › `desvio`, `dadger.rvN` › `AC` › `DESVIO` | Teto de vazão desviada de uma RHQ de termo único sobre `QDES`: `limite_superior` do `LQ` por patamar. Para um `desvio` de base (`hidr.dat` ou `AC DESVIO`) sem limite RHQ, o teto é o `limite_vazao` do `AC DESVIO` ou, na falta dele, o `generation.max_turbined_m3s` da usina receptora, em todos os estágios, para que a água desviada não fique presa em [0, 0]. |
| `min_spillage_m3s` | `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `LQ` › `limite_inferior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente` | Piso de vertimento de uma RHQ de termo único sobre `QVER`: `limite_inferior` do `LQ` por patamar, herdado entre estágios. |
| `max_spillage_m3s` | `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `LQ` › `limite_superior`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente` | Teto de vertimento de uma RHQ de termo único sobre `QVER`: `limite_superior` do `LQ` por patamar, herdado entre estágios. |
| `water_withdrawal_m3s` | `dadger.rvN` › `TI` › `codigo_usina`, `dadger.rvN` › `TI` › `taxa` | Retirada consuntiva de irrigação (m³/s, positiva = água removida) do registro `TI`, `taxa_k` → estágio k−1, sempre de estágio (sem patamar); estágios além das colunas declaradas repetem a última taxa e uma taxa nula não gera valor. Resolvida em tabela própria e anexada à linha da célula, pois não é um eixo de faixa (só piso). |

### `constraints/hydro_unit_group_bounds.parquet`

**Lê:** `dadger.rvN` (`UH`, `SB`, `CT`, `DP`, `DT`, `AC`, `MP`, `FD`, `RI`), `hidr.dat`  
**Quando:** somente quando algum grupo tem, em algum estágio, disponibilidade (MP × FD) ou engolimento corrigido pela queda abaixo do envelope declarado, ou quando Itaipu é operada com pisos `RI`  
**Código:** `src/novomodelo_bridge/decomp/group_bounds.py`

Sobrescritas por (usina, grupo, estágio, patamar) dos limites dos `unit_groups` de `system/hydros.json`, esparsas: só onde a disponibilidade por estágio (instalada × `MP` × `FD`) ou o engolimento corrigido pela queda do estágio fica abaixo do envelope do grupo, mais os pisos por frequência `RI` de Itaipu. Os valores são calculados em `converters/hydro/bounds.py`; este módulo só os dispõe em linhas.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id Novomodelo da usina. |
| `hydro_unit_group_id` | `dadger.rvN` › `MP` › `frequencia`, `dadger.rvN` › `FD` › `frequencia` *(derivado)* | Id do grupo dentro da usina: 0 para o grupo único; 0 (50 Hz) e 1 (60 Hz) para Itaipu, na ordem crescente das frequências de `MP`/`FD`. O Novomodelo casa a linha pelo id declarado, não pela posição. |
| `stage_id` | `dadger.rvN` › `DP` › `estagio` *(derivado)* | Índice 0-based do estágio no calendário operativo; as colunas `manutencao_k`/`fator_k` de `MP`/`FD` e o `estagio` do `RI` são resolvidos para esse índice. |
| `block_id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Nulo na linha-base; 0..n−1 só quando o piso `RI` de Itaipu difere entre patamares do estágio (a linha-base recebe então a média ponderada pelas horas). Disponibilidade e engolimento são valores de estágio. |
| `min_turbined_m3s` | — *(sempre nulo)* | Sempre nulo: nenhum registro do DECOMP declara turbinamento mínimo por grupo. |
| `max_turbined_m3s` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1..5`, `hidr.dat` › `vazao_nominal_conjunto_1..5`, `hidr.dat` › `potencia_nominal_conjunto_1..5`, `hidr.dat` › `queda_nominal_conjunto_1..5`, `hidr.dat` › `tipo_turbina`, `hidr.dat` › `a0_volume_cota..a4_volume_cota`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `perdas`, `hidr.dat` › `tipo_perda`, `dadger.rvN` › `AC` › `NUMCON`, `dadger.rvN` › `AC` › `NUMMAQ`, `dadger.rvN` › `AC` › `VAZEFE`, `dadger.rvN` › `AC` › `POTEFE`, `dadger.rvN` › `AC` › `COTVOL`, `dadger.rvN` › `AC` › `VOLMIN`, `dadger.rvN` › `AC` › `VOLMAX`, `dadger.rvN` › `AC` › `PROESP`, `dadger.rvN` › `AC` › `JUSMED`, `dadger.rvN` › `AC` › `PERHID` | Engolimento corrigido pela queda do estágio (mesma fórmula de `generation.max_turbined_m3s` em `system/hydros.json`, avaliada no estágio em vez do máximo), emitido só quando fica abaixo do envelope do grupo além da tolerância numérica: um `AC` de máquinas ou de cadastro que reduz o engolimento no meio do horizonte aparece aqui. Para Itaipu, por conjunto. |
| `min_generation_mw` | `dadger.rvN` › `RI` › `estagio`, `dadger.rvN` › `RI` › `geracao_minima_50_hz`, `dadger.rvN` › `RI` › `geracao_minima_60_hz` *(condicional: somente para Itaipu (código 66) operada com registro `RI`)* | Piso de geração por frequência de Itaipu: `geracao_minima_50_hz` no grupo 0 e `geracao_minima_60_hz` no grupo 1, por patamar (colunas `_1`..`_5`), com estágios não declarados herdando o anterior (o `RI` precisa declarar o estágio 1). Os campos `geracao_maxima_*` do `RI` não são consumidos. |
| `max_generation_mw` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1..5`, `hidr.dat` › `potencia_nominal_conjunto_1..5`, `dadger.rvN` › `AC` › `NUMCON`, `dadger.rvN` › `AC` › `NUMMAQ`, `dadger.rvN` › `AC` › `POTEFE`, `dadger.rvN` › `MP` › `codigo_usina`, `dadger.rvN` › `MP` › `manutencao`, `dadger.rvN` › `FD` › `codigo_usina`, `dadger.rvN` › `FD` › `fator` | Potência disponível do estágio: instalada efetiva do estágio × `manutencao_k` (`MP`) × `fator_k` (`FD`), fator 1.0 quando a usina não tem o registro, emitida só quando fica abaixo do envelope do grupo. Para Itaipu o cálculo é por grupo, com a linha `MP`/`FD` da frequência correspondente (obrigatória). O teto hidráulico ρ_eq × engolimento não é duplicado aqui: age pelo `max_turbined_m3s`. |

### `constraints/thermal_bounds.parquet`

**Lê:** `dadger.rvN` (`CT`, `RE`, `LU`, `FU`, `FT`, `FI`, `SB`, `UH`, `DP`, `DT`)  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/converters/thermal.py`

Limites de geração e CVU de cada térmica de `CT` em cada estágio (estágios não declarados herdam o último; estágio 1 obrigatório), intersectados com as restrições elétricas `RE` de termo único sobre uma térmica (`FT` com coeficiente ±1 e limites `LU`). Térmicas GNL não recebem linhas.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `thermal_id` | `dadger.rvN` › `CT` › `codigo_usina` *(derivado)* | Id da térmica em `system/thermals.json`. |
| `stage_id` | `dadger.rvN` › `CT` › `estagio`, `dadger.rvN` › `RE` › `estagio_inicial, estagio_final` *(derivado)* | Índice 0-based do estágio; uma linha de custo para cada estágio do calendário. |
| `block_id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Nulo quando `disponibilidade` e `inflexibilidade` são uniformes entre patamares e nenhuma `RE` por patamar atinge a térmica; senão `0..n-1`, uma linha por patamar com os limites exatos do patamar, mais uma linha de `block_id` nulo carregando apenas o custo do estágio. |
| `min_generation_mw` | `dadger.rvN` › `CT` › `inflexibilidade`, `dadger.rvN` › `DP` › `duracao`, `dadger.rvN` › `LU` › `limite_inferior`, `dadger.rvN` › `FT` › `coeficiente` | Inflexibilidade por patamar de `CT` (ponderada por horas na linha de base, exata por patamar); quando uma `RE` de termo único `FT` limita a térmica, o maior entre os limites inferiores (coeficiente −1 troca e inverte os lados). Nulo na linha que só carrega custo. |
| `max_generation_mw` | `dadger.rvN` › `CT` › `disponibilidade`, `dadger.rvN` › `DP` › `duracao`, `dadger.rvN` › `LU` › `limite_superior`, `dadger.rvN` › `FT` › `coeficiente` | Disponibilidade por patamar de `CT` (ponderada por horas na linha de base, exata por patamar); com `RE`/`FT`, o menor entre os limites superiores. Nulo na linha que só carrega custo. |
| `cost_per_mwh` | `dadger.rvN` › `CT` › `cvu`, `dadger.rvN` › `DP` › `duracao` | CVU do estágio ponderado pelas horas de patamar (`cvu_k`, em branco lido como 0), sempre em uma linha de `block_id` nulo; nulo nas linhas por patamar, pois o custo não é elegível a patamar no Novomodelo. |

### `constraints/line_bounds.parquet`

**Lê:** `dadger.rvN` (`IA`, `RI`, `SB`, `UH`, `CT`, `DP`, `DT`), `hidr.dat`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/decomp/converters/network.py`

Limites de intercâmbio por linha e estágio a partir de `IA` (estágios não declarados herdam o último declarado; estágio 1 obrigatório). Por (linha, estágio) há uma única linha de base quando os patamares são uniformes, ou uma linha por patamar quando não são, nunca ambas.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `line_id` | `dadger.rvN` › `IA` › `nome_submercado_de, nome_submercado_para` *(derivado)* | Id da linha em `system/lines.json`. |
| `stage_id` | `dadger.rvN` › `IA` › `estagio` *(derivado)* | Índice 0-based do estágio; uma linha para cada estágio do calendário, com herança dos estágios `IA` não declarados. |
| `block_id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Nulo na linha de base (todos os patamares); `0..n-1` quando os limites de `IA` variam entre patamares naquele estágio, caso em que cada patamar recebe a sua própria linha e a base é descartada. |
| `direct_mw` | `dadger.rvN` › `IA` › `limite_de_para`, `dadger.rvN` › `RI` › `geracao_maxima_60_hz` | Linha de base: o maior `limite_de_para_k` do estágio; linhas por patamar: o `limite_de_para_k` do próprio patamar, em MW absolutos. O sentinela 99999 passa como capacidade grande. A linha `IV-SE` criada pela conversão recebe o maior `geracao_maxima_60_hz` de `RI` em todos os estágios (99999 sem `RI`). |
| `reverse_mw` | `dadger.rvN` › `IA` › `limite_para_de`, `dadger.rvN` › `RI` › `geracao_maxima_60_hz` | Mesma regra de `direct_mw` com `limite_para_de_k`; a linha `IV-SE` criada usa o mesmo valor do sentido direto. |

### `constraints/pumping_bounds.parquet`

**Lê:** `dadger.rvN` (`HQ`, `LQ`, `CQ`, `UE`, `SB`, `UH`, `CT`, `DP`, `DT`)  
**Quando:** somente quando alguma restrição hidráulica `HQ` de termo único `QBOM` (coeficiente ±1 em `CQ`) limita uma estação de bombeamento declarada em `UE`  
**Código:** `src/novomodelo_bridge/decomp/converters/single_term_bounds.py`

Limites de vazão bombeada por estação, estágio e patamar, oriundos exclusivamente das restrições `HQ`/`LQ`/`CQ` cujo único termo é `QBOM`. Os limites de `UE` ficam em `system/pumping_stations.json`. Um `QBOM` sem estação correspondente é avisado e ignorado.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `pumping_station_id` | `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `UE` › `codigo_usina` *(derivado)* | Id da estação em `system/pumping_stations.json`, resolvido pelo `codigo_usina` do termo `QBOM`. |
| `stage_id` | `dadger.rvN` › `HQ` › `estagio_inicial, estagio_final`, `dadger.rvN` › `LQ` › `estagio` *(derivado)* | Índice 0-based de cada estágio entre `estagio_inicial` e `estagio_final` da `HQ`; estágios sem `LQ` herdam o limite declarado mais recente (os anteriores à primeira declaração herdam a primeira). |
| `block_id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Sempre `0..n-1`: os limites `LQ` são por patamar, logo cada estágio se materializa em uma linha por patamar do calendário. |
| `min_m3s` | `dadger.rvN` › `LQ` › `limite_inferior`, `dadger.rvN` › `CQ` › `coeficiente` | `limite_inferior_k` da `LQ` (coeficiente −1 troca e inverte os lados); nulo quando em branco ou no sentinela ilimitado. Com mais de uma restrição sobre a mesma estação, o maior dos limites inferiores. |
| `max_m3s` | `dadger.rvN` › `LQ` › `limite_superior`, `dadger.rvN` › `CQ` › `coeficiente` | `limite_superior_k` da `LQ`; nulo quando em branco ou no sentinela ilimitado. Com mais de uma restrição sobre a mesma estação, o menor dos limites superiores. |

### `constraints/contract_bounds.parquet`

**Lê:** `dadger.rvN` (`CI`, `CE`, `SB`, `UH`, `CT`, `DP`, `DT`)  
**Quando:** somente quando `dadger` declara ao menos um contrato `CI`/`CE` real (tabela não vazia)  
**Código:** `src/novomodelo_bridge/decomp/converters/contracts.py`

Limites e preço de cada contrato por estágio (estágios não declarados herdam o último; estágio 1 obrigatório). Por (contrato, estágio) há uma única linha de base quando `limite_inferior`, `limite_superior` e `custo` são uniformes entre patamares, ou uma linha por patamar quando não são, nunca ambas.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `contract_id` | `dadger.rvN` › `CI` › `numero_contrato`, `dadger.rvN` › `CE` › `numero_contrato` *(derivado)* | Id do contrato em `system/energy_contracts.json`. |
| `stage_id` | `dadger.rvN` › `CI` › `estagio`, `dadger.rvN` › `CE` › `estagio` *(derivado)* | Índice 0-based do estágio; uma entrada para cada estágio do calendário. |
| `block_id` | `dadger.rvN` › `DP` › `numero_patamares` *(derivado)* | Nulo na linha de base (todos os patamares); `0..n-1` quando algum dos três valores varia entre patamares naquele estágio, caso em que cada patamar recebe a sua própria linha e a base é descartada. |
| `min_mw` | `dadger.rvN` › `CI` › `limite_inferior`, `dadger.rvN` › `CE` › `limite_inferior`, `dadger.rvN` › `DP` › `duracao` | Linha de base: `limite_inferior_k` ponderado pelas horas de patamar; linhas por patamar: o valor exato do patamar. |
| `max_mw` | `dadger.rvN` › `CI` › `limite_superior`, `dadger.rvN` › `CE` › `limite_superior`, `dadger.rvN` › `DP` › `duracao` | Linha de base: `limite_superior_k` ponderado pelas horas de patamar; linhas por patamar: o valor exato do patamar. |
| `price_per_mwh` | `dadger.rvN` › `CI` › `custo`, `dadger.rvN` › `CE` › `custo`, `dadger.rvN` › `DP` › `duracao` | Linha de base: `custo_k` ponderado pelas horas de patamar; linhas por patamar: o custo exato do patamar. Negativo nas exportações, positivo nas importações. |

### `constraints/generic_constraints.json`

**Lê:** `dadger.rvN` (`SB`, `CT`, `UH`, `DP`, `DT`, `RE`, `LU`, `FU`, `FT`, `FI`, `HQ`, `LQ`, `CQ`, `HV`, `LV`, `CV`, `HE`, `CM`, `CD`, `IA`, `UE`, `RI`, `MP`, `FD`, `PQ`, `AC`), `hidr.dat`, `indices.csv`, `lib_restricao-eletrica-especial*.csv` (opcional), `renovaveis*` (opcional)  
**Quando:** somente quando ao menos uma restrição especial (RE com vários termos, RHQ/RHV, RHE ou restrição elétrica do arquivo LIBs) não se reduz a um limite de entidade e sobrevive como restrição genérica  
**Esquema:** [generic_constraints.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/generic_constraints.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/constraints.py`

Uma entrada por restrição especial que não se reduz a limite de entidade: `RE` com vários termos (`FU`/`FT`/`FI`), `RHQ`/`RHV` com vários termos ou coeficiente não unitário, `RHE` (energia armazenada por REE) e as restrições elétricas de forma longa do arquivo LIBs, emitidas nessa ordem sobre um único espaço de ids. Uma restrição com termo não resolvível (`FU` por frequência, térmica GNL, `FI` sem linha `IA`, `QBOM` sem estação `UE`, `VDEF`/`VDES`/`VBOM`, REE sem reservatório de regularização mensal) é pulada inteira, com diagnóstico.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `constraints[].id` | `dadger.rvN` › `RE` › `codigo_restricao`, `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `HV` › `codigo_restricao`, `dadger.rvN` › `HE` › `codigo_restricao`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-FORMULA · codigo_restricao`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO · codigo_restricao` *(derivado)* | Id 0-based sequencial, atribuído na ordem de emissão `RE` → `HQ`/`HV` → `HE` → LIBs (na ordem das linhas de declaração; restrições LIBs em ordem crescente de `codigo_restricao`), apenas às restrições que sobrevivem. O código original fica em `name`. |
| `constraints[].name` | `dadger.rvN` › `RE` › `codigo_restricao`, `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `HV` › `codigo_restricao`, `dadger.rvN` › `HE` › `codigo_restricao`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-FORMULA · codigo_restricao`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO · codigo_restricao` *(derivado)* | `RE_<codigo>`, `HQ_<codigo>`, `HV_<codigo>`, `RHE_<codigo>` ou `LIBS_ELEC_<codigo>`, conforme a família de origem. |
| `constraints[].description` | `dadger.rvN` › `RE` › `codigo_restricao`, `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `HV` › `codigo_restricao`, `dadger.rvN` › `HE` › `codigo_restricao`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-FORMULA · codigo_restricao` *(derivado)* | Texto fixo por família com o código de origem: `RE generic constraint N`, `RHQ generic constraint N`, `RHV generic constraint N`, `RHE stored-energy constraint N`, `LIBs electrical special constraint N`. |
| `constraints[].expression` | `dadger.rvN` › `FU` › `codigo_usina`, `dadger.rvN` › `FU` › `coeficiente`, `dadger.rvN` › `FT` › `codigo_usina`, `dadger.rvN` › `FT` › `coeficiente`, `dadger.rvN` › `FI` › `codigo_submercado_de`, `dadger.rvN` › `FI` › `codigo_submercado_para`, `dadger.rvN` › `FI` › `coeficiente`, `dadger.rvN` › `CQ` › `codigo_usina`, `dadger.rvN` › `CQ` › `tipo`, `dadger.rvN` › `CQ` › `coeficiente`, `dadger.rvN` › `CV` › `codigo_usina`, `dadger.rvN` › `CV` › `tipo`, `dadger.rvN` › `CV` › `coeficiente`, `dadger.rvN` › `CM` › `codigo_ree`, `dadger.rvN` › `CM` › `coeficiente`, `dadger.rvN` › `UH` › `codigo_ree`, `dadger.rvN` › `IA` › `nome_submercado_de`, `dadger.rvN` › `IA` › `nome_submercado_para`, `dadger.rvN` › `UE` › `codigo_usina`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-FORMULA · formula`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO · formula`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO · formula_limite`, `lib_restricao-eletrica-especial*.csv` › `EXPRESSAO-ELETRICA · formula`, `renovaveis*` › `PEE-CAD · codigo_pee` | Soma linear de termos `coeficiente * token`, com coeficiente unitário omitido: `FU` → `hydro_generation(id)`, `FT` → `thermal_generation(id)`, `FI` → `line_direct(id)`/`line_reverse(id)` pela orientação do par nas linhas `IA`, `CQ.tipo` `QDEF`/`QTUR`/`QVER`/`QDES` → `hydro_outflow`/`hydro_turbined`/`hydro_spillage`/`hydro_diversion(id)`, `QBOM` → `pumping_flow(id)` da estação `UE`, `CV.tipo` `VARM` → `hydro_storage(id)`; `HE` expande cada REE (`CM`) em `@rho_acum_h<id> * hydro_storage(id)` sobre os reservatórios (`M` ou `S`) com volume útil positivo ligados ao REE por `UH.codigo_ree`, com o sinal do `CM`. Nas restrições LIBs a fórmula (com `EXPRESSAO-ELETRICA` expandida) mantém apenas os termos de decisão `ger_usih`/`ger_usit`/`ger_pee`/`ger_conjh`/`ener_interc`, mapeados para os mesmos tokens (`ger_pee` → `non_controllable_generation`, `ger_conjh` → `hydro_generation(id, bus=…)`); os termos de dados (`demanda`, `carga_ande`, aliases, `disp_usih`) são dobrados no limite. |
| `constraints[].slack.enabled` | — *(constante)* | Sempre `true`: toda restrição genérica é flexível, penalizada em vez de infactível. |
| `constraints[].slack.penalty` | `dadger.rvN` › `CD` › `custo`, `dadger.rvN` › `HE` › `valor_penalidade`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO · custo_violacao` | `RE`/`HQ`/`HV`: 10 × o maior custo de déficit do `CD` entre os submercados. `RHE`: `valor_penalidade` do `HE`, ou 1000.0 (com aviso) quando ausente ou não positivo. LIBs: `custo_violacao` do `TRATAMENTO-VIOLACAO` quando positivo, senão o mesmo 10 × déficit. |

### `constraints/generic_constraint_bounds.parquet`

**Lê:** `dadger.rvN` (`SB`, `CT`, `UH`, `DP`, `DT`, `RE`, `LU`, `FU`, `FT`, `FI`, `HQ`, `LQ`, `CQ`, `HV`, `LV`, `CV`, `HE`, `CM`, `CD`, `IA`, `UE`, `RI`, `MP`, `FD`, `PQ`, `AC`), `hidr.dat`, `indices.csv`, `lib_restricao-eletrica-especial*.csv` (opcional), `renovaveis*` (opcional)  
**Quando:** somente quando ao menos uma restrição especial (RE com vários termos, RHQ/RHV, RHE ou restrição elétrica do arquivo LIBs) não se reduz a um limite de entidade e sobrevive como restrição genérica  
**Código:** `src/novomodelo_bridge/decomp/converters/constraints.py`

Uma linha por (restrição genérica, estágio, patamar) com limite em pelo menos um lado; a direção é codificada por qual extremo está preenchido (`>=` só inferior, `<=` só superior, `==` ambos iguais).

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `constraint_id` | `dadger.rvN` › `RE` › `codigo_restricao`, `dadger.rvN` › `HQ` › `codigo_restricao`, `dadger.rvN` › `HV` › `codigo_restricao`, `dadger.rvN` › `HE` › `codigo_restricao`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-FORMULA · codigo_restricao` *(derivado)* | O `constraints[].id` de `generic_constraints.json`; linhas na ordem de emissão das restrições. |
| `stage_id` | `dadger.rvN` › `RE` › `estagio_inicial`, `dadger.rvN` › `RE` › `estagio_final`, `dadger.rvN` › `LU` › `estagio`, `dadger.rvN` › `HQ` › `estagio_inicial`, `dadger.rvN` › `HQ` › `estagio_final`, `dadger.rvN` › `LQ` › `estagio`, `dadger.rvN` › `HV` › `estagio_inicial`, `dadger.rvN` › `HV` › `estagio_final`, `dadger.rvN` › `LV` › `estagio`, `dadger.rvN` › `HE` › `estagio`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-HORIZONTE-PERIODO · estagio_inicio`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-HORIZONTE-PERIODO · estagio_fim`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-HABILITA · codigo_regra_ativacao`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-REGRA-ATIVACAO · regra_ativacao` *(derivado)* | Índice 0-based (`estagio` − 1). `RE`/`HQ`/`HV`: cada estágio de `estagio_inicial` a `estagio_final`, com o limite `LU`/`LQ`/`LV` esparso herdado adiante (estágios antes da primeira declaração herdam a primeira). `HE`: os estágios com linha `HE`. LIBs: as células ativas dentro do horizonte, após avaliar a regra de ativação (`HABILITA`/`REGRA-ATIVACAO`) sobre a carga `DP` e `carga_ande`. |
| `block_id` | `dadger.rvN` › `LU` › `limite_inferior`, `dadger.rvN` › `LQ` › `limite_inferior`, `dadger.rvN` › `DP` › `numero_patamares`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR · patamar`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR · patamar` *(derivado)* | Famílias por patamar (`RE`, `HQ`, LIBs): índice 0-based do patamar, limitado ao número de patamares do estágio no calendário (um slot além disso é descartado). Famílias por estágio (`HV`, `HE`): nulo. |
| `bound_lower` | `dadger.rvN` › `LU` › `limite_inferior`, `dadger.rvN` › `LQ` › `limite_inferior`, `dadger.rvN` › `LV` › `limite_inferior`, `dadger.rvN` › `CV` › `coeficiente`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `dadger.rvN` › `AC` › `VOLMIN · volume`, `dadger.rvN` › `AC` › `VOLMAX · volume`, `dadger.rvN` › `HE` › `limite`, `dadger.rvN` › `HE` › `tipo_limite`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR · limite_inferior`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO · operador`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO · formula_limite`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR · formula_limite`, `lib_restricao-eletrica-especial*.csv` › `ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR · valor`, `dadger.rvN` › `DP` › `carga`, `dadger.rvN` › `RI` › `carga_ande`, `dadger.rvN` › `MP` › `manutencao`, `dadger.rvN` › `FD` › `fator` | `RE`/`HQ`: `limite_inferior` do patamar (nulo quando em branco). `HV`: `limite_inferior` do `LV` mais `Σ cᵢ·volume_minimo` efetivo das usinas do estágio, pois o `LV` é relativo ao volume útil e `hydro_storage` é absoluto. `HE`: `limite` em MWmês quando `tipo_limite` = 1; com `tipo_limite` = 2, percentual de `Σ ρ_acum·volume_maximo` dos reservatórios participantes (outro valor trata como absoluto, com aviso). LIBs: `limite_inferior` da fórmula (sentinela ±1E+31 vira nulo) ou o lado constante da inequação com os termos de dados avaliados na célula (`demanda`, `carga_ande`, aliases, `disp_usih` = potência disponível `MP`×`FD`). |
| `bound_upper` | `dadger.rvN` › `LU` › `limite_superior`, `dadger.rvN` › `LQ` › `limite_superior`, `dadger.rvN` › `LV` › `limite_superior`, `dadger.rvN` › `CV` › `coeficiente`, `hidr.dat` › `volume_minimo`, `dadger.rvN` › `AC` › `VOLMIN · volume`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR · limite_superior`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO · operador`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO · formula_limite`, `lib_restricao-eletrica-especial*.csv` › `RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR · formula_limite`, `lib_restricao-eletrica-especial*.csv` › `ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR · valor`, `dadger.rvN` › `DP` › `carga`, `dadger.rvN` › `RI` › `carga_ande`, `dadger.rvN` › `MP` › `manutencao`, `dadger.rvN` › `FD` › `fator` | `RE`/`HQ`: `limite_superior` do patamar (nulo quando em branco). `HV`: `limite_superior` do `LV` mais o mesmo deslocamento `Σ cᵢ·volume_minimo` efetivo. `HE`: sempre nulo, o `limite` é só inferior. LIBs: `limite_superior` da fórmula ou o lado constante da inequação, como em `bound_lower`; um lado com magnitude ≥ 1e21 também lê nulo. |

### `constraints/generic_parameters.json`

**Lê:** `dadger.rvN` (`HE`, `CM`, `UH`, `SB`, `CT`, `AC`, `DP`, `DT`), `hidr.dat`  
**Quando:** somente quando alguma restrição de energia armazenada `HE`/`CM` (RHE) sobrevive como restrição genérica e referencia ao menos um reservatório  
**Esquema:** [generic_parameters.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/generic_parameters.schema.json) · **Código:** `src/novomodelo_bridge/decomp/converters/scalar_parameters.py`

Declara, para cada usina operada, os parâmetros `@rho_eq_h{id}` e `@rho_acum_h{id}` usados em expressões de restrições genéricas; o `rho_acum` das usinas referenciadas por uma RHE recebe valores por estágio (produtibilidade acumulada integrada no volume, convenção de energia armazenada do DECOMP), os demais ficam calculados pelo Novomodelo.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `scalar_parameters[].id` | — *(derivado)* | Sequencial 0-based: para cada usina, em ordem crescente de id hidráulico, primeiro o `rho_eq` e depois o `rho_acum`. |
| `scalar_parameters[].name` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | `rho_eq_h{id}` ou `rho_acum_h{id}`, com o id 0-based da usina operada. |
| `scalar_parameters[].kind` | `dadger.rvN` › `HE` › `codigo_restricao`, `dadger.rvN` › `CM` › `codigo_ree`, `dadger.rvN` › `UH` › `codigo_ree` *(derivado)* | `computed` por padrão. `per_stage` no `rho_acum` de cada usina que participa da expressão de alguma RHE emitida: membros da REE (`codigo_ree` de `UH`) nomeada por `CM`, restritos a reservatórios (`tipo_regulacao` = `M` ou `S`) com volume útil positivo. |
| `scalar_parameters[].computed_spec.tag` | — *(constante)* | `equivalent_productivity` nas entradas `rho_eq`; `accumulated_productivity` nas entradas `rho_acum` sem valores por estágio. Ausente nas entradas `per_stage`. |
| `scalar_parameters[].computed_spec.hydro_id` | `dadger.rvN` › `UH` › `codigo_usina` *(derivado)* | Id 0-based da usina a que o parâmetro calculado se refere. |
| `scalar_parameters[].values[][]` | `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `hidr.dat` › `volume_minimo, volume_maximo, volume_referencia`, `hidr.dat` › `produtibilidade_especifica, canal_fuga_medio, tipo_regulacao`, `hidr.dat` › `codigo_usina_jusante`, `dadger.rvN` › `AC` › `COTVOL, JUSMED, VOLMIN, VOLMAX, NUMJUS`, `dadger.rvN` › `DP` › `duracao` | Pares `[estágio, ρ_acum]` em MWmês/hm³. Por estágio, a produtibilidade de energia armazenada própria de cada usina operada (integral da altura em `[vmin, vmax]` para `tipo_regulacao` = `M` ou `S`; ponto em `volume_referencia` para `D`) é somada ao longo da cascata operada (`codigo_usina_jusante`, com `AC NUMJUS`), e o total é dividido por `3600 · H / 1e6`, com `H` as horas do estágio. Os `AC COTVOL`/`JUSMED`/`VOLMIN`/`VOLMAX` vigentes deslocam o valor do estágio. Emitido apenas para as usinas referenciadas por uma RHE. |

### `boundary/`

**Lê:** `cortesh.dat` (opcional), `cortes*.dat` (opcional), `dadger.rvN` (`FC`, `CX`, `SB`, `CT`, `UH`, `DP`, `DT`, `AC`), `dadgnl.rvN` (opcional) (`TG`, `GL`, `GS`, `NL`), `vazoes.rvN`, `hidr.dat`, `mlt.dat`  
**Quando:** escrito por padrão pela CLI após a conversão, quando o deck declara `cortesh.dat` e `cortes*.dat` (registro `FC` ou busca por nome) e o novomodelo-python instalado passa na sonda de escrita e releitura do checkpoint; `--no-fcf` e `--dry-run` pulam a importação, e um deck sem arquivos de cortes converte sem fronteira (nota informativa)  
**Código:** `src/novomodelo_bridge/decomp/fcf/importer.py`

Importa a função de custo futuro de fronteira do modelo de origem: lê o cabeçalho `cortesh.dat` e os cortes do estágio de acoplamento em `cortes*.dat`, executa uma iteração do novomodelo sobre uma cópia do caso para ler o manifesto de estados terminal, mapeia os termos de armazenamento por código de usina (replicando complexos `CX`), os termos de vazão defasada por profundidade de mês (com a média de longo termo de `mlt.dat`, incrementalizada pela cascata efetiva, dobrada no RHS) e os termos GNL do `dadgnl` no anel antecipado, e grava `boundary/{manifest.bin, cuts/, basis/}` via novomodelo-python. Reescreve `config.json` apontando `policy.boundary` (`path = "boundary"`, `source_stage`) para o checkpoint e, quando há `mlt.dat`, `initial_conditions.json` com as janelas `recent_observations` das observações mensais e semanais de `vazoes`; o caso deve ser executado com `novomodelo run <caso> --output <caso>`.

## Índice por origem

Para cada arquivo do deck (e, no DECOMP, cada registro): o que ele alimenta e o que nele ainda não é convertido.

### `dadger.rvN` › `AC`

**Estado:** convertido em parte. **Lido por:** 13 arquivos gerados; ver a visão geral.

- `VOLMIN, VOLMAX` → `initial_conditions.json` › `storage[].value_hm3`
- `COTVOL, JUSMED, VOLMIN, VOLMAX, NUMJUS` → `constraints/generic_parameters.json` › `scalar_parameters[].values[][]`
- `NUMJUS` → `system/hydros.json` › `hydros[].downstream_id`
- `VOLMIN` → `system/hydros.json` › `hydros[].reservoir.min_storage_hm3`
- `VOLMAX` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `COTVOL` → `system/hydros.json` › `hydros[].generation.model`
- `PROESP` → `system/hydros.json` › `hydros[].generation.model`
- `NUMCON` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `NUMMAQ` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `VAZEFE` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `POTEFE` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `COTVOL` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `VOLMIN` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `VOLMAX` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `PROESP` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `JUSMED` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `PERHID` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `NUMCON` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `NUMMAQ` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `POTEFE` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `NUMCON` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `NUMMAQ` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `POTEFE` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `NUMCON` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `NUMMAQ` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `VAZEFE` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `POTEFE` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `COTVOL` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `VOLMIN` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `VOLMAX` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `PROESP` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `JUSMED` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `PERHID` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `PROESP` → `system/hydros.json` › `hydros[].specific_productivity_mw_per_m3s_per_m`
- `PROESP` → `system/hydros.json` › `hydros[].efficiency.value`
- `JUSMED` → `system/hydros.json` › `hydros[].tailrace.coefficients[]`
- `PERHID` → `system/hydros.json` › `hydros[].hydraulic_losses.type`
- `PERHID` → `system/hydros.json` › `hydros[].hydraulic_losses.value`
- `PERHID` → `system/hydros.json` › `hydros[].hydraulic_losses.value_m`
- `DESVIO` → `system/hydros.json` › `hydros[].diversion.downstream_id`
- `DESVIO` → `system/hydros.json` › `hydros[].diversion.max_flow_m3s`
- `COTVOL` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].model`
- `PROESP` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].model`
- `VOLMIN` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `VOLMAX` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `VOLMIN` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `VOLMAX` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `VOLMIN` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].reference_volume.volume_hm3`
- `VOLMAX` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].reference_volume.volume_hm3`
- `VOLMIN` → `system/hydro_geometry.parquet` › `volume_hm3`
- `VOLMAX` → `system/hydro_geometry.parquet` › `volume_hm3`
- `COTVOL` → `system/hydro_geometry.parquet` › `height_m`
- `PROESP` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `COTVOL` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `JUSMED` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `PERHID` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `VOLMIN` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `VOLMAX` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `VAZMIN` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `NUMJUS` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `VOLMIN` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `VOLMAX` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `VOLMIN` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `VOLMAX` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `DESVIO` → `constraints/hydro_bounds.parquet` › `min_diversion_m3s`
- `DESVIO` → `constraints/hydro_bounds.parquet` › `max_diversion_m3s`
- `NUMCON` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `NUMMAQ` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `VAZEFE` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `POTEFE` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `COTVOL` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `VOLMIN` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `VOLMAX` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `PROESP` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `JUSMED` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `PERHID` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `NUMCON` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `NUMMAQ` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `POTEFE` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `NUMPOS · codigo_posto` → `scenarios/external_inflow_scenarios.parquet` › `value_m3s`
- `VOLMIN · volume` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `VOLMAX · volume` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `VOLMIN · volume` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `ALTEFE` — *adiado.* Alteração da altura efetiva de queda de um conjunto de máquinas. O leitor idecomp expõe só as colunas de identificação e data, sem a coluna de valor, logo o conversor não consegue ingerir a alteração: o `check decomp` emite `decomp-ac-altefe-uningestable` e a queda nominal de `hidr.dat` é mantida. Depende de o idecomp expor o valor.
- `COFEVA` — *adiado.* Alteração do coeficiente de evaporação mensal. Lido pelo `check decomp` (diagnóstico `decomp-ac-overrides-deferred`), sem consumidor no conversor: a evaporação usa os coeficientes mensais de `hidr.dat`. Converter exige sobrepor o coeficiente do mês na tabela efetiva antes de emitir os coeficientes de evaporação.
- `COTARE` — *adiado.* Alteração de um coeficiente do polinômio cota-área. Lido pelo `check decomp`, sem consumidor: a FPHA usa o polinômio de `hidr.dat`. Converter exige sobrepor o coeficiente na tabela efetiva usada pelo ajuste da FPHA.
- `COTVAZ` — *adiado.* Alteração de um coeficiente do polinômio cota-vazão (curva de jusante). Lido pelo `check decomp`, sem consumidor: as curvas de jusante vêm de `polinjus`. Converter exige aplicar a alteração sobre a família de `tailrace_curves.parquet` da usina.
- `JUSENA` — *adiado.* Alteração do índice de aproveitamento de jusante para o cálculo de energia armazenada e afluente. Lido pelo `check decomp`, sem consumidor: o Novomodelo não agrega energia por REE.
- `NCHAVE` — *adiado.* Alteração do número da curva-chave (cota-vazão) e do nível de jusante da faixa. Lido pelo `check decomp`, sem consumidor: a seleção de família de jusante segue `polinjus`.
- `NPOSNW` — *adiado.* Alteração do posto de acoplamento com o NEWAVE. Lido pelo `check decomp`, sem consumidor: o acoplamento da FCF de fronteira usa o `posto` de `hidr.dat` (com `NUMPOS`) e o registro `CX`.
- `TIPERH` — *adiado.* Alteração do tipo de perdas hidráulicas. Lido pelo `check decomp`, sem consumidor: o conversor usa o `tipo_perda` de `hidr.dat`. Converter exige sobrepor o tipo na tabela efetiva antes de calcular perdas e produtibilidade.
- `VERTJU` — *adiado.* Consideração da influência do vertimento no canal de fuga. Lido pelo `check decomp`, sem consumidor: o Novomodelo não modela essa influência na cota de jusante.

### `dadger.rvN` › `AR`

**Estado:** convertido. **Lido por:** `post_study_stages.json`, `stages.json`.

- `estagio, lamb, alfa` → `stages.json` › `stages[].risk_measure`
- `alfa` → `stages.json` › `stages[].risk_measure.cvar.alpha`
- `lamb` → `stages.json` › `stages[].risk_measure.cvar.lambda`

### `dadger.rvN` › `CA`

**Estado:** lido, não convertido.

- todo o registro — *adiado.* Coeficientes da família `HA`/`LA`/`CA`; ver `HA`.

### `dadger.rvN` › `CD`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `penalties.json`, `system/buses.json`.

- `custo` → `penalties.json` › `bus.deficit_segments[].cost`
- `custo` → `penalties.json` › `hydro.storage_violation_below_cost`
- `custo` → `penalties.json` › `hydro.filling_target_violation_cost`
- `custo` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `custo` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `custo` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `custo` → `penalties.json` › `hydro.generation_violation_below_cost`
- `custo` → `penalties.json` › `hydro.evaporation_violation_cost`
- `custo` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `custo` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `custo` → `system/buses.json` › `buses[].deficit_segments[].cost`
- `custo` → `constraints/generic_constraints.json` › `constraints[].slack.penalty`
- `nome_curva` — *não lido.* Nome da curva de déficit. Não lido; apenas identificação.

### `dadger.rvN` › `CE`

**Estado:** convertido em parte. **Lido por:** `constraints/contract_bounds.parquet`, `system/energy_contracts.json`.

- `numero_contrato` → `system/energy_contracts.json` › `contracts[].id`
- `nome_contrato` → `system/energy_contracts.json` › `contracts[].name`
- `codigo_submercado` → `system/energy_contracts.json` › `contracts[].bus_id`
- `custo` → `system/energy_contracts.json` › `contracts[].price_per_mwh`
- `limite_inferior` → `system/energy_contracts.json` › `contracts[].limits.min_mw`
- `limite_superior` → `system/energy_contracts.json` › `contracts[].limits.max_mw`
- `numero_contrato` → `constraints/contract_bounds.parquet` › `contract_id`
- `estagio` → `constraints/contract_bounds.parquet` › `stage_id`
- `limite_inferior` → `constraints/contract_bounds.parquet` › `min_mw`
- `limite_superior` → `constraints/contract_bounds.parquet` › `max_mw`
- `custo` → `constraints/contract_bounds.parquet` › `price_per_mwh`
- `fator_perdas` — *adiado.* Fator de perdas do contrato de exportação. Lido e ignorado com o diagnóstico `contract-loss-factor-unmapped`: `energy_contracts` do Novomodelo não tem campo equivalente e o fator não é dobrado no preço nem nos limites.

### `dadger.rvN` › `CI`

**Estado:** convertido em parte. **Lido por:** `constraints/contract_bounds.parquet`, `system/energy_contracts.json`.

- `numero_contrato` → `system/energy_contracts.json` › `contracts[].id`
- `nome_contrato` → `system/energy_contracts.json` › `contracts[].name`
- `codigo_submercado` → `system/energy_contracts.json` › `contracts[].bus_id`
- `custo` → `system/energy_contracts.json` › `contracts[].price_per_mwh`
- `limite_inferior` → `system/energy_contracts.json` › `contracts[].limits.min_mw`
- `limite_superior` → `system/energy_contracts.json` › `contracts[].limits.max_mw`
- `numero_contrato` → `constraints/contract_bounds.parquet` › `contract_id`
- `estagio` → `constraints/contract_bounds.parquet` › `stage_id`
- `limite_inferior` → `constraints/contract_bounds.parquet` › `min_mw`
- `limite_superior` → `constraints/contract_bounds.parquet` › `max_mw`
- `custo` → `constraints/contract_bounds.parquet` › `price_per_mwh`
- `fator_perdas` — *adiado.* Fator de perdas do contrato de importação; mesmo tratamento do `CE`.

### `dadger.rvN` › `CM`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/generic_parameters.json`.

- `codigo_ree` → `constraints/generic_parameters.json` › `scalar_parameters[].kind`
- `codigo_ree` → `constraints/generic_constraints.json` › `constraints[].expression`
- `coeficiente` → `constraints/generic_constraints.json` › `constraints[].expression`

### `dadger.rvN` › `CQ`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`, `constraints/pumping_bounds.parquet`, `system/hydros.json`.

- `codigo_usina` → `constraints/pumping_bounds.parquet` › `pumping_station_id`
- `coeficiente` → `constraints/pumping_bounds.parquet` › `min_m3s`
- `coeficiente` → `constraints/pumping_bounds.parquet` › `max_m3s`
- `codigo_usina` → `system/hydros.json` › `hydros[].diversion.downstream_id`
- `tipo` → `system/hydros.json` › `hydros[].diversion.downstream_id`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `tipo` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `max_outflow_m3s`
- `tipo` → `constraints/hydro_bounds.parquet` › `max_outflow_m3s`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `max_outflow_m3s`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `min_turbined_m3s`
- `tipo` → `constraints/hydro_bounds.parquet` › `min_turbined_m3s`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `min_turbined_m3s`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `tipo` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `min_diversion_m3s`
- `tipo` → `constraints/hydro_bounds.parquet` › `min_diversion_m3s`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `min_diversion_m3s`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `max_diversion_m3s`
- `tipo` → `constraints/hydro_bounds.parquet` › `max_diversion_m3s`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `max_diversion_m3s`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `min_spillage_m3s`
- `tipo` → `constraints/hydro_bounds.parquet` › `min_spillage_m3s`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `min_spillage_m3s`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `max_spillage_m3s`
- `tipo` → `constraints/hydro_bounds.parquet` › `max_spillage_m3s`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `max_spillage_m3s`
- `codigo_usina` → `constraints/generic_constraints.json` › `constraints[].expression`
- `tipo` → `constraints/generic_constraints.json` › `constraints[].expression`
- `coeficiente` → `constraints/generic_constraints.json` › `constraints[].expression`
- `estagio` — *não lido.* Estágio da participação de vazão; ignorado como em `FU`.

### `dadger.rvN` › `CS`

**Estado:** não lido.

- todo o registro — *não lido.* Habilita a consistência de dados do DECOMP. Sem contrapartida.

### `dadger.rvN` › `CT`

**Estado:** convertido. **Lido por:** 31 arquivos gerados; ver a visão geral.

- `codigo_usina` → `system/thermals.json` › `thermals[].id`
- `nome_usina` → `system/thermals.json` › `thermals[].name`
- `codigo_submercado` → `system/thermals.json` › `thermals[].bus_id`
- `cvu` → `system/thermals.json` › `thermals[].cost_per_mwh`
- `inflexibilidade` → `system/thermals.json` › `thermals[].generation.min_mw`
- `disponibilidade` → `system/thermals.json` › `thermals[].generation.max_mw`
- `codigo_usina` → `constraints/thermal_bounds.parquet` › `thermal_id`
- `estagio` → `constraints/thermal_bounds.parquet` › `stage_id`
- `inflexibilidade` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `disponibilidade` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `cvu` → `constraints/thermal_bounds.parquet` › `cost_per_mwh`

### `dadger.rvN` › `CV`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`.

- `codigo_usina` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `tipo` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `tipo` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `codigo_usina` → `constraints/generic_constraints.json` › `constraints[].expression`
- `tipo` → `constraints/generic_constraints.json` › `constraints[].expression`
- `coeficiente` → `constraints/generic_constraints.json` › `constraints[].expression`
- `coeficiente` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `coeficiente` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `tipo (VDEF, VDES, VBOM)` — *adiado.* Tipos de volume de uma `HV` além de `VARM`. A forma hm³-para-vazão exige coeficiente por estágio (horas de patamar variam), que a expressão de restrição genérica do Novomodelo não carrega; a restrição é descartada com `decomp-rhv-volume-tipo-deferred`.
- `estagio` — *não lido.* Estágio da participação de volume; ignorado como em `FU`.

### `dadger.rvN` › `CX`

**Estado:** lido, não convertido. **Lido por:** `boundary/`.

### `dadger.rvN` › `DA`

**Estado:** não lido.

- todo o registro — *não lido.* Retiradas de água para outros usos (desvios) por UHE: usina de retirada, usina de retorno, vazão desviada, percentual de retorno e custo, por estágio. Não lido. Converter exige mapear a retirada para o canal de desvio da usina no Novomodelo e o retorno como afluência à usina de destino.

### `dadger.rvN` › `DP`

**Estado:** convertido. **Lido por:** 32 arquivos gerados; ver a visão geral.

- `estagio` → `stages.json` › `policy_graph.nodes[].id`
- `estagio` → `stages.json` › `policy_graph.nodes[].stage_id`
- `estagio` → `stages.json` › `stages[].id`
- `duracao` → `stages.json` › `stages[].start_date`
- `duracao` → `stages.json` › `stages[].end_date`
- `numero_patamares` → `stages.json` › `stages[].blocks[].id`
- `numero_patamares` → `stages.json` › `stages[].blocks[].name`
- `duracao` → `stages.json` › `stages[].blocks[].hours`
- `duracao` → `initial_conditions.json` › `past_anticipated_commitments[].end_date`
- `duracao` → `system/thermals.json` › `thermals[].cost_per_mwh`
- `duracao` → `system/thermals.json` › `thermals[].generation.min_mw`
- `duracao` → `system/thermals.json` › `thermals[].generation.max_mw`
- `duracao` → `constraints/generic_parameters.json` › `scalar_parameters[].values[][]`
- `numero_patamares` → `constraints/line_bounds.parquet` › `block_id`
- `numero_patamares` → `constraints/thermal_bounds.parquet` › `block_id`
- `duracao` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `duracao` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `duracao` → `constraints/thermal_bounds.parquet` › `cost_per_mwh`
- `numero_patamares` → `constraints/pumping_bounds.parquet` › `block_id`
- `duracao` → `post_study_stages.json` › `stages[].start_date`
- `duracao` → `post_study_stages.json` › `stages[].duration_hours`
- `duracao` → `post_study_stages.json` › `thermal_bounds[].cost_per_mwh`
- `duracao` → `post_study_stages.json` › `thermal_bounds[].min_mw`
- `duracao` → `post_study_stages.json` › `thermal_bounds[].max_mw`
- `duracao` → `system/energy_contracts.json` › `contracts[].price_per_mwh`
- `duracao` → `system/energy_contracts.json` › `contracts[].limits.min_mw`
- `duracao` → `system/energy_contracts.json` › `contracts[].limits.max_mw`
- `numero_patamares` → `constraints/contract_bounds.parquet` › `block_id`
- `duracao` → `constraints/contract_bounds.parquet` › `min_mw`
- `duracao` → `constraints/contract_bounds.parquet` › `max_mw`
- `duracao` → `constraints/contract_bounds.parquet` › `price_per_mwh`
- `estagio` → `constraints/hydro_bounds.parquet` › `stage_id`
- `numero_patamares` → `constraints/hydro_bounds.parquet` › `block_id`
- `estagio` → `constraints/hydro_unit_group_bounds.parquet` › `stage_id`
- `numero_patamares` → `constraints/hydro_unit_group_bounds.parquet` › `block_id`
- `estagio` → `scenarios/inflow_seasonal_stats.parquet` › `stage_id`
- `estagio` → `scenarios/load_seasonal_stats.parquet` › `stage_id`
- `carga` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `duracao` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `estagio` → `scenarios/load_factors.json` › `load_factors[].stage_id`
- `numero_patamares` → `scenarios/load_factors.json` › `load_factors[].block_factors[].block_id`
- `carga` → `scenarios/load_factors.json` › `load_factors[].block_factors[].factor`
- `duracao` → `scenarios/load_factors.json` › `load_factors[].block_factors[].factor`
- `estagio` → `scenarios/non_controllable_stats.parquet` › `stage_id`
- `duracao` → `scenarios/non_controllable_stats.parquet` › `mean`
- `estagio` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].stage_id`
- `numero_patamares` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].block_id`
- `duracao` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].factor`
- `estagio` → `scenarios/external_ncs_scenarios.parquet` › `stage_id`
- `duracao` → `scenarios/external_ncs_scenarios.parquet` › `availability_factor`
- `estagio` → `scenarios/external_load_scenarios.parquet` › `stage_id`
- `carga` → `scenarios/external_load_scenarios.parquet` › `value_mw`
- `duracao` → `scenarios/external_load_scenarios.parquet` › `value_mw`
- `numero_patamares` → `constraints/generic_constraint_bounds.parquet` › `block_id`
- `carga` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `carga` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`

### `dadger.rvN` › `DT`

**Estado:** convertido. **Lido por:** 32 arquivos gerados; ver a visão geral.

- `dia, mes, ano` → `stages.json` › `stages[].start_date`
- `dia, mes, ano` → `stages.json` › `stages[].end_date`
- `dia, mes, ano` → `stages.json` › `stages[].season_id`
- `dia, mes, ano` → `initial_conditions.json` › `past_anticipated_commitments[].start_date`
- `dia, mes, ano` → `system/buses.json` › `buses[].operational_start_date`
- `dia, mes, ano` → `system/lines.json` › `lines[].operational_start_date`
- `dia, mes, ano` → `system/thermals.json` › `thermals[].operational_start_date`
- `dia, mes, ano` → `system/thermals.json` › `thermals[].anticipated_config.lead_time_hours`
- `dia, mes, ano` → `system/pumping_stations.json` › `pumping_stations[].operational_start_date`
- `dia, mes, ano` → `system/non_controllable_sources.json` › `non_controllable_sources[].operational_start_date`
- `dia, mes, ano` → `post_study_stages.json` › `stages[].start_date`
- `dia, mes, ano` → `system/energy_contracts.json` › `contracts[].operational_start_date`
- `dia` → `system/hydros.json` › `hydros[].operational_start_date`
- `mes` → `system/hydros.json` › `hydros[].operational_start_date`
- `ano` → `system/hydros.json` › `hydros[].operational_start_date`

### `dadger.rvN` › `EA`

**Estado:** não lido.

- todo o registro — *não lido.* ENA dos meses anteriores ao estudo, por REE, para a tendência hidrológica do DECOMP. Não lido: a árvore de cenários vem pronta de `vazoes.rvN` e o estado de afluências passadas da FCF de fronteira é derivado das observações de `vazoes.rvN` e de `mlt.dat`.

### `dadger.rvN` › `ES`

**Estado:** não lido.

- todo o registro — *não lido.* ENA das semanas anteriores ao estudo, por REE. Não lido, pelo mesmo motivo de `EA`.

### `dadger.rvN` › `EV`

**Estado:** não lido.

- todo o registro — *não lido.* Configuração da evaporação (modelo e volume de referência). Não lido: a evaporação é convertida a partir do sinalizador `evaporacao` do `UH` e dos coeficientes mensais de `hidr.dat`.

### `dadger.rvN` › `EZ`

**Estado:** não lido.

- todo o registro — *não lido.* Percentual máximo do volume útil para acoplamento com o NEWAVE. O `check decomp` tenta inventariá-lo, mas o leitor idecomp não expõe acessor `ez`, então o registro nunca é lido.

### `dadger.rvN` › `FA`

**Estado:** não lido.

- todo o registro — *não lido.* Nome do arquivo índice CSV das LIBs. Não lido: o bridge procura `indices.csv` diretamente no diretório do deck.

### `dadger.rvN` › `FC`

**Estado:** lido, não convertido. **Lido por:** `boundary/`, `stages.json`.

### `dadger.rvN` › `FD`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_unit_group_bounds.parquet`, `system/hydros.json`.

- `frequencia` → `system/hydros.json` › `hydros[].unit_groups[].id`
- `frequencia` → `constraints/hydro_unit_group_bounds.parquet` › `hydro_unit_group_id`
- `codigo_usina` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `fator` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `fator` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `fator` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`

### `dadger.rvN` › `FE`

**Estado:** lido, não convertido.

- todo o registro — *adiado.* Coeficientes de participação elétrica de uma restrição `RE`. O idecomp não expõe leitor para este registro; o bridge detecta sua presença por varredura textual e emite `decomp-fe-participation-unreadable`, pois uma `RE` com termo `FE` fica sub-lida. Converter exige um leitor próprio e o descarte, por restrição, das `RE` que carregam `FE`.

### `dadger.rvN` › `FI`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`, `constraints/thermal_bounds.parquet`.

- `codigo_submercado_de` → `constraints/generic_constraints.json` › `constraints[].expression`
- `codigo_submercado_para` → `constraints/generic_constraints.json` › `constraints[].expression`
- `coeficiente` → `constraints/generic_constraints.json` › `constraints[].expression`
- `estagio` — *não lido.* Estágio da participação de intercâmbio; ignorado como em `FU`.

### `dadger.rvN` › `FJ`

**Estado:** não lido.

- todo o registro — *não lido.* Nome do arquivo de polinômios de jusante. Não lido: o bridge resolve `polinjus*` por prefixo no diretório do deck.

### `dadger.rvN` › `FP`

**Estado:** não lido.

- todo o registro — *não lido.* Parâmetros da função de produção por usina e estágio: janelas de volume e turbinamento e número de pontos de discretização. Não lido: o Novomodelo ajusta a FPHA internamente (`fpha_configs` com `source = computed`), com janela derivada do volume inicial e dos limites de armazenamento.

### `dadger.rvN` › `FT`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`, `constraints/thermal_bounds.parquet`.

- `coeficiente` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `coeficiente` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `codigo_usina` → `constraints/generic_constraints.json` › `constraints[].expression`
- `coeficiente` → `constraints/generic_constraints.json` › `constraints[].expression`
- `estagio` — *não lido.* Estágio da participação térmica; ignorado como em `FU`.

### `dadger.rvN` › `FU`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`, `constraints/thermal_bounds.parquet`.

- `codigo_usina` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `coeficiente` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `codigo_usina` → `constraints/generic_constraints.json` › `constraints[].expression`
- `coeficiente` → `constraints/generic_constraints.json` › `constraints[].expression`
- `frequencia` — *adiado.* Frequência (50/60 Hz) do termo de geração hidráulica de uma `RE`. Uma restrição genérica com termo dividido por frequência é descartada com `decomp-re-frequency-split-deferred`, pois o conversor não constrói o mapa frequência-barra; no caminho de limite simples a frequência é ignorada e o termo vale para a usina inteira.
- `estagio` — *não lido.* Estágio em que a participação é declarada. Ignorado: o coeficiente é herdado por toda a faixa `[estagio_inicial, estagio_final]` da restrição.

### `dadger.rvN` › `GP`

**Estado:** convertido. **Lido por:** `config.json`.

- `gap` → `config.json` › `training.stopping_rules[].relative_tolerance`

### `dadger.rvN` › `HA`

**Estado:** lido, não convertido.

- todo o registro — *adiado.* Cadastro da família de restrições `HA`/`LA`/`CA` (declaração, limites e coeficientes), sem leitor no idecomp. Detectada por varredura textual; diagnóstico `decomp-rha-family-unconverted`. Converter exige leitor próprio e um emissor equivalente ao das famílias `RE`/`HQ`/`HV`.

### `dadger.rvN` › `HE`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/generic_parameters.json`.

- `codigo_restricao` → `constraints/generic_parameters.json` › `scalar_parameters[].kind`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].id`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].name`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].description`
- `valor_penalidade` → `constraints/generic_constraints.json` › `constraints[].slack.penalty`
- `codigo_restricao` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `estagio` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `limite` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `tipo_limite` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `forma_calculo_produtibilidades` — *não lido.* Forma de cálculo das produtibilidades da restrição de energia armazenada. Capturada pelo censo de restrições, mas nenhum emissor a consome: a energia é expandida com a produtibilidade acumulada calculada pelo conversor.
- `tipo_valores_produtibilidades` — *não lido.* Tipo dos valores de produtibilidade; capturado e não consumido, como `forma_calculo_produtibilidades`.
- `arquivo_produtibilidades` — *não lido.* Arquivo externo de produtibilidades da restrição; capturado e não consumido, e o arquivo nunca é aberto.
- `tipo_penalidade` — *não lido.* Tipo da penalidade da restrição de energia; só `valor_penalidade` é usado como penalidade da folga.

### `dadger.rvN` › `HQ`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`, `constraints/pumping_bounds.parquet`, `system/hydros.json`.

- `estagio_inicial, estagio_final` → `constraints/pumping_bounds.parquet` › `stage_id`
- `codigo_restricao` → `system/hydros.json` › `hydros[].diversion.downstream_id`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `max_outflow_m3s`
- `estagio_inicial` → `constraints/hydro_bounds.parquet` › `max_outflow_m3s`
- `estagio_final` → `constraints/hydro_bounds.parquet` › `max_outflow_m3s`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `min_turbined_m3s`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `min_diversion_m3s`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `max_diversion_m3s`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `min_spillage_m3s`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `max_spillage_m3s`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].id`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].name`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].description`
- `codigo_restricao` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `estagio_inicial` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `estagio_final` → `constraints/generic_constraint_bounds.parquet` › `stage_id`

### `dadger.rvN` › `HV`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`.

- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `estagio_inicial` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `estagio_final` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].id`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].name`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].description`
- `codigo_restricao` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `estagio_inicial` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `estagio_final` → `constraints/generic_constraint_bounds.parquet` › `stage_id`

### `dadger.rvN` › `IA`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/line_bounds.parquet`, `system/lines.json`.

- `nome_submercado_de, nome_submercado_para` → `system/lines.json` › `lines[].id`
- `nome_submercado_de, nome_submercado_para` → `system/lines.json` › `lines[].name`
- `nome_submercado_de` → `system/lines.json` › `lines[].source_bus_id`
- `nome_submercado_para` → `system/lines.json` › `lines[].target_bus_id`
- `limite_de_para` → `system/lines.json` › `lines[].capacity.direct_mw`
- `limite_para_de` → `system/lines.json` › `lines[].capacity.reverse_mw`
- `nome_submercado_de, nome_submercado_para` → `constraints/line_bounds.parquet` › `line_id`
- `estagio` → `constraints/line_bounds.parquet` › `stage_id`
- `limite_de_para` → `constraints/line_bounds.parquet` › `direct_mw`
- `limite_para_de` → `constraints/line_bounds.parquet` › `reverse_mw`
- `nome_submercado_de` → `constraints/generic_constraints.json` › `constraints[].expression`
- `nome_submercado_para` → `constraints/generic_constraints.json` › `constraints[].expression`

### `dadger.rvN` › `IR`

**Estado:** não lido.

- todo o registro — *não lido.* Opções de geração de relatórios de saída do DECOMP. Sem contrapartida no caso Novomodelo.

### `dadger.rvN` › `LA`

**Estado:** lido, não convertido.

- todo o registro — *adiado.* Limites por estágio da família `HA`/`LA`/`CA`; ver `HA`.

### `dadger.rvN` › `LQ`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`, `constraints/pumping_bounds.parquet`, `system/hydros.json`.

- `estagio` → `constraints/pumping_bounds.parquet` › `stage_id`
- `limite_inferior` → `constraints/pumping_bounds.parquet` › `min_m3s`
- `limite_superior` → `constraints/pumping_bounds.parquet` › `max_m3s`
- `limite_inferior` → `system/hydros.json` › `hydros[].diversion.downstream_id`
- `limite_inferior` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `limite_superior` → `constraints/hydro_bounds.parquet` › `max_outflow_m3s`
- `limite_inferior` → `constraints/hydro_bounds.parquet` › `min_turbined_m3s`
- `limite_superior` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `limite_inferior` → `constraints/hydro_bounds.parquet` › `min_diversion_m3s`
- `limite_superior` → `constraints/hydro_bounds.parquet` › `max_diversion_m3s`
- `limite_inferior` → `constraints/hydro_bounds.parquet` › `min_spillage_m3s`
- `limite_superior` → `constraints/hydro_bounds.parquet` › `max_spillage_m3s`
- `estagio` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `limite_inferior` → `constraints/generic_constraint_bounds.parquet` › `block_id`
- `limite_inferior` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `limite_superior` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`

### `dadger.rvN` › `LU`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`, `constraints/thermal_bounds.parquet`.

- `limite_inferior` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `limite_superior` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `limite_inferior` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `limite_superior` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `estagio` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `limite_inferior` → `constraints/generic_constraint_bounds.parquet` › `block_id`
- `limite_inferior` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `limite_superior` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`

### `dadger.rvN` › `LV`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`.

- `limite_inferior` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `limite_superior` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `estagio` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `limite_inferior` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `limite_superior` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`

### `dadger.rvN` › `MP`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_unit_group_bounds.parquet`, `system/hydros.json`.

- `frequencia` → `system/hydros.json` › `hydros[].unit_groups[].id`
- `frequencia` → `constraints/hydro_unit_group_bounds.parquet` › `hydro_unit_group_id`
- `codigo_usina` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `manutencao` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `manutencao` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `manutencao` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`

### `dadger.rvN` › `MT`

**Estado:** não lido.

- todo o registro — *não lido.* Manutenções programadas das UTEs (fator por estágio). Não lido: a disponibilidade térmica vem só de `CT`. Converter exige multiplicar `disponibilidade` pelo fator ao emitir `thermal_bounds.parquet`.

### `dadger.rvN` › `NI`

**Estado:** convertido em parte. **Lido por:** `config.json`.

- `iteracoes` → `config.json` › `training.stopping_rules[].limit`
- `tipo_limite` — *não lido.* Tipo do limite de iterações. Não lido: só `iteracoes` é usado no critério `iteration_limit`.

### `dadger.rvN` › `PD`

**Estado:** não lido.

- todo o registro — *não lido.* Escolha do algoritmo de solução do PL. Parâmetro do solver do DECOMP; sem contrapartida.

### `dadger.rvN` › `PE`

**Estado:** não lido.

- todo o registro — *não lido.* Alteração das penalidades de vertimento, intercâmbio e desvio por submercado. Não lido: `penalties.json` usa os valores padrão do conversor. Converter exige mapear cada `tipo` para o campo de penalidade correspondente.

### `dadger.rvN` › `PQ`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `scenarios/external_ncs_scenarios.parquet`, `scenarios/non_controllable_factors.json`, `scenarios/non_controllable_stats.parquet`, `system/non_controllable_sources.json`.

- `codigo_submercado, nome` → `system/non_controllable_sources.json` › `non_controllable_sources[].id`
- `nome, codigo_submercado` → `system/non_controllable_sources.json` › `non_controllable_sources[].name`
- `codigo_submercado` → `system/non_controllable_sources.json` › `non_controllable_sources[].bus_id`
- `geracao` → `system/non_controllable_sources.json` › `non_controllable_sources[].max_generation_mw`
- `codigo_submercado` → `scenarios/non_controllable_stats.parquet` › `ncs_id`
- `nome` → `scenarios/non_controllable_stats.parquet` › `ncs_id`
- `geracao` → `scenarios/non_controllable_stats.parquet` › `mean`
- `estagio` → `scenarios/non_controllable_stats.parquet` › `mean`
- `codigo_submercado` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].ncs_id`
- `nome` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].ncs_id`
- `geracao` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].factor`
- `codigo_submercado` → `scenarios/external_ncs_scenarios.parquet` › `ncs_id`
- `nome` → `scenarios/external_ncs_scenarios.parquet` › `ncs_id`
- `geracao` → `scenarios/external_ncs_scenarios.parquet` › `availability_factor`

### `dadger.rvN` › `PU`

**Estado:** não lido.

- todo o registro — *não lido.* Habilita a solução do problema via PL único. Sem contrapartida.

### `dadger.rvN` › `PV`

**Estado:** não lido.

- todo o registro — *não lido.* Penalidades das variáveis de folga e tolerâncias de viabilidade das restrições. Parâmetros internos do solver do DECOMP; sem contrapartida.

### `dadger.rvN` › `QI`

**Estado:** não lido.

- todo o registro — *não lido.* Tempo de viagem por usina usado no cálculo da ENA. Não lido: só afeta a agregação em energia, que o Novomodelo não faz.

### `dadger.rvN` › `RC`

**Estado:** não lido.

- todo o registro — *não lido.* Inclusão de restrições do tipo escada por mnemônico. Não lido.

### `dadger.rvN` › `RE`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_bounds.parquet`, `constraints/thermal_bounds.parquet`.

- `estagio_inicial, estagio_final` → `constraints/thermal_bounds.parquet` › `stage_id`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `estagio_inicial` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `estagio_final` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `codigo_restricao` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `estagio_inicial` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `estagio_final` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].id`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].name`
- `codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].description`
- `codigo_restricao` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `estagio_inicial` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `estagio_final` → `constraints/generic_constraint_bounds.parquet` › `stage_id`

### `dadger.rvN` › `RI`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/hydro_unit_group_bounds.parquet`, `constraints/line_bounds.parquet`, `scenarios/external_load_scenarios.parquet`, `scenarios/load_factors.json`, `scenarios/load_seasonal_stats.parquet`, `system/lines.json`.

- `geracao_maxima_60_hz` → `system/lines.json` › `lines[].capacity.direct_mw`
- `geracao_maxima_60_hz` → `system/lines.json` › `lines[].capacity.reverse_mw`
- `geracao_maxima_60_hz` → `constraints/line_bounds.parquet` › `direct_mw`
- `geracao_maxima_60_hz` → `constraints/line_bounds.parquet` › `reverse_mw`
- `estagio` → `constraints/hydro_unit_group_bounds.parquet` › `min_generation_mw`
- `geracao_minima_50_hz` → `constraints/hydro_unit_group_bounds.parquet` › `min_generation_mw`
- `geracao_minima_60_hz` → `constraints/hydro_unit_group_bounds.parquet` › `min_generation_mw`
- `carga_ande` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `carga_ande` → `scenarios/load_factors.json` › `load_factors[].block_factors[].factor`
- `carga_ande` → `scenarios/external_load_scenarios.parquet` › `value_mw`
- `carga_ande` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `carga_ande` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `geracao_maxima_50_hz_1..5` — *não lido.* Geração máxima do setor de 50 Hz de Itaipu por patamar. Não lida: o conversor usa a geração máxima de 60 Hz e as mínimas de 50 e 60 Hz. Converter exige emiti-la como teto do grupo de unidades de 50 Hz.

### `dadger.rvN` › `RQ`

**Estado:** convertido. **Lido por:** `constraints/hydro_bounds.parquet`.

- `codigo_ree` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `vazao` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`

### `dadger.rvN` › `RT`

**Estado:** não lido.

- todo o registro — *não lido.* Retirada das restrições de soleira de vertedouro e de canal de desvio. Não lido; converter exige suprimir o limite correspondente em `hydro_bounds.parquet`.

### `dadger.rvN` › `SB`

**Estado:** convertido. **Lido por:** 32 arquivos gerados; ver a visão geral.

- `codigo_submercado` → `system/buses.json` › `buses[].id`
- `nome_submercado` → `system/buses.json` › `buses[].name`
- `codigo_submercado` → `system/hydros.json` › `hydros[].unit_groups[].bus_id`
- `codigo_submercado` → `scenarios/load_seasonal_stats.parquet` › `bus_id`
- `codigo_submercado` → `scenarios/load_factors.json` › `load_factors[].bus_id`
- `codigo_submercado` → `scenarios/external_load_scenarios.parquet` › `bus_id`

### `dadger.rvN` › `TE`

**Estado:** não lido.

- todo o registro — *não lido.* Título do estudo. Não lido.

### `dadger.rvN` › `TI`

**Estado:** convertido. **Lido por:** `constraints/hydro_bounds.parquet`.

- `codigo_usina` → `constraints/hydro_bounds.parquet` › `water_withdrawal_m3s`
- `taxa` → `constraints/hydro_bounds.parquet` › `water_withdrawal_m3s`

### `dadger.rvN` › `TS`

**Estado:** não lido.

- todo o registro — *não lido.* Tolerâncias do solver do DECOMP. Sem contrapartida.

### `dadger.rvN` › `TX`

**Estado:** convertido. **Lido por:** `stages.json`.

- `taxa` → `stages.json` › `policy_graph.annual_discount_rate`

### `dadger.rvN` › `UE`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/pumping_bounds.parquet`, `system/pumping_stations.json`.

- `codigo_usina` → `system/pumping_stations.json` › `pumping_stations[].id`
- `nome_usina` → `system/pumping_stations.json` › `pumping_stations[].name`
- `codigo_submercado` → `system/pumping_stations.json` › `pumping_stations[].bus_id`
- `codigo_usina_jusante` → `system/pumping_stations.json` › `pumping_stations[].source_hydro_id`
- `codigo_usina_montante` → `system/pumping_stations.json` › `pumping_stations[].destination_hydro_id`
- `taxa_consumo` → `system/pumping_stations.json` › `pumping_stations[].consumption_mw_per_m3s`
- `vazao_minima_bombeavel` → `system/pumping_stations.json` › `pumping_stations[].flow.min_m3s`
- `vazao_maxima_bombeavel` → `system/pumping_stations.json` › `pumping_stations[].flow.max_m3s`
- `codigo_usina` → `constraints/pumping_bounds.parquet` › `pumping_station_id`
- `codigo_usina` → `constraints/generic_constraints.json` › `constraints[].expression`

### `dadger.rvN` › `UH`

**Estado:** convertido em parte. **Lido por:** 32 arquivos gerados; ver a visão geral.

- `volume_inicial` → `penalties.json` › `hydro.spillage_cost`
- `volume_inicial` → `penalties.json` › `hydro.turbined_cost`
- `volume_inicial` → `penalties.json` › `hydro.diversion_cost`
- `volume_inicial` → `penalties.json` › `hydro.storage_violation_below_cost`
- `volume_inicial` → `penalties.json` › `hydro.filling_target_violation_cost`
- `volume_inicial` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `volume_inicial` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `volume_inicial` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `volume_inicial` → `penalties.json` › `hydro.evaporation_violation_cost`
- `volume_inicial` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `volume_inicial` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `codigo_usina` → `initial_conditions.json` › `storage[].hydro_id`
- `volume_inicial` → `initial_conditions.json` › `storage[].value_hm3`
- `codigo_usina` → `constraints/generic_parameters.json` › `scalar_parameters[].name`
- `codigo_ree` → `constraints/generic_parameters.json` › `scalar_parameters[].kind`
- `codigo_usina` → `constraints/generic_parameters.json` › `scalar_parameters[].computed_spec.hydro_id`
- `codigo_usina` → `system/hydros.json` › `hydros[].id`
- `codigo_usina` → `system/hydros.json` › `hydros[].downstream_id`
- `vazao_defluente_minima` → `system/hydros.json` › `hydros[].outflow.min_outflow_m3s`
- `evaporacao` → `system/hydros.json` › `hydros[].evaporation.coefficients_mm[]`
- `codigo_usina` → `system/hydro_production_models.json` › `production_models[].hydro_id`
- `volume_inicial` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `volume_inicial` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `volume_inicial` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].reference_volume.volume_hm3`
- `codigo_usina` → `system/hydro_geometry.parquet` › `hydro_id`
- `codigo_usina` → `system/hydro_energy_productivity.parquet` › `hydro_id`
- `volume_inicial` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `codigo_usina` → `system/tailrace_curves.parquet` › `hydro_id`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `hydro_id`
- `vazao_defluente_minima` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `codigo_ree` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `codigo_usina` → `constraints/hydro_unit_group_bounds.parquet` › `hydro_id`
- `codigo_usina` → `scenarios/inflow_seasonal_stats.parquet` › `hydro_id`
- `codigo_usina` → `scenarios/external_inflow_scenarios.parquet` › `hydro_id`
- `codigo_ree` → `constraints/generic_constraints.json` › `constraints[].expression`
- `balanco_hidrico_patamar` — *não lido.* Sinalizador de balanço hídrico por patamar. Não lido: o Novomodelo faz o balanço por estágio.
- `configuracao_newave` — *não lido.* Índice da configuração do NEWAVE para acoplamento. Não lido: a FCF de fronteira é acoplada por código de usina e por `CX`.
- `estagio_inicio_producao` — *não lido.* Estágio em que a usina começa a produzir. Não lido: o conversor não emite `entry_stage_id` para hidrelétricas (fica no padrão nulo do esquema). Converter exige mapear este estágio para `entry_stage_id`.
- `limite_superior_vertimento` — *não lido.* Limite superior de vertimento. Não lido; converter exige emiti-lo como `max_spillage_m3s` em `hydro_bounds.parquet`.

### `dadger.rvN` › `VA`

**Estado:** não lido.

- todo o registro — *não lido.* Posto cuja vazão influencia a cota de jusante de outra usina (fator de impacto incremental). Não lido: o Novomodelo não modela influência de vazão lateral na cota de jusante.

### `dadger.rvN` › `VE`

**Estado:** convertido. **Lido por:** `constraints/hydro_bounds.parquet`.

- `codigo_usina` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `volume` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`

### `dadger.rvN` › `VI`

**Estado:** lido, não convertido.

- todo o registro — *adiado.* Tempo de viagem da água até a usina de jusante e defluências das semanas anteriores ao estudo. O conversor lê o registro e monta `travel_time_hours` e `initial_conditions.past_defluences`, mas a emissão está desligada por um defeito na geração de cortes com água em trânsito no novomodelo (gap rastreado C-travel-time): o `convert decomp` apenas avisa `deferred at this milestone: water travel time (VI present)`.

### `dadger.rvN` › `VL`

**Estado:** não lido.

- todo o registro — *não lido.* Usina que sofre influência de vazão lateral na cota de jusante (polinômio e fator de impacto). Não lido; ver `VA`.

### `dadger.rvN` › `VT`

**Estado:** não lido.

- todo o registro — *não lido.* Nome do arquivo de cenários de vento. Não lido: a geração eólica vem de `renovaveis`.

### `dadger.rvN` › `VU`

**Estado:** não lido.

- todo o registro — *não lido.* Usina cuja defluência influencia a cota de jusante de outra. Não lido; ver `VA`.

### `vazoes.rvN`

**Estado:** convertido em parte. **Lido por:** `boundary/`, `scenarios/external_inflow_scenarios.parquet`, `scenarios/external_load_scenarios.parquet`, `scenarios/external_ncs_scenarios.parquet`, `stages.json`.

Árvore de cenários de vazão gerada pelo GEVAZP. O conversor lê probabilidades, previsões, cenários gerados e observações; as seções abaixo ficam de fora.

- `probabilidades · cenario` → `stages.json` › `policy_graph.nodes[].id`
- `probabilidades · cenario` → `stages.json` › `policy_graph.nodes[].scenario_id`
- `probabilidades · probabilidade (estagio terminal)` → `stages.json` › `policy_graph.transitions[].probability`
- `previsoes · estagio` → `scenarios/external_inflow_scenarios.parquet` › `stage_id`
- `cenarios_gerados · estagio` → `scenarios/external_inflow_scenarios.parquet` › `stage_id`
- `cenarios_gerados · cenario` → `scenarios/external_inflow_scenarios.parquet` › `scenario_id`
- `previsoes · coluna do posto` → `scenarios/external_inflow_scenarios.parquet` › `value_m3s`
- `cenarios_gerados · coluna do posto` → `scenarios/external_inflow_scenarios.parquet` › `value_m3s`
- `probabilidades · cenario` → `scenarios/external_ncs_scenarios.parquet` › `scenario_id`
- `probabilidades · cenario` → `scenarios/external_load_scenarios.parquet` › `scenario_id`
- `previsoes_com_postos_artificiais` — *não lido.* Previsões incluindo os postos artificiais de acoplamento. Não lidas: só os postos das usinas operadas interessam.
- `cenarios_calculados_com_postos_artificiais` — *não lido.* Cenários gerados incluindo postos artificiais. Não lidos, pelo mesmo motivo.
- `numero_aberturas_estagios` — *não lido.* Número de aberturas por estágio. Não lido: a forma da árvore é deduzida dos próprios cenários e das probabilidades.

### `hidr.dat`

**Estado:** convertido em parte. **Lido por:** 18 arquivos gerados; ver a visão geral.

Cadastro binário das hidrelétricas, compartilhado com o NEWAVE. As colunas abaixo não são lidas pela via DECOMP.

- `produtibilidade_especifica` → `penalties.json` › `hydro.spillage_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.spillage_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.spillage_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.spillage_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.turbined_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.turbined_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.turbined_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.turbined_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.diversion_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.diversion_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.diversion_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.diversion_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.storage_violation_below_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.storage_violation_below_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.storage_violation_below_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.storage_violation_below_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.filling_target_violation_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.filling_target_violation_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.filling_target_violation_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.filling_target_violation_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.evaporation_violation_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.evaporation_violation_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.evaporation_violation_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.evaporation_violation_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `produtibilidade_especifica` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `a0_volume_cota … a4_volume_cota` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `canal_fuga_medio` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `perdas, tipo_perda` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `volume_minimo, volume_maximo` → `initial_conditions.json` › `storage[].value_hm3`
- `tipo_regulacao, volume_referencia` → `initial_conditions.json` › `storage[].value_hm3`
- `submercado (usina 66)` → `system/lines.json` › `lines[].target_bus_id`
- `a0_volume_cota … a4_volume_cota` → `constraints/generic_parameters.json` › `scalar_parameters[].values[][]`
- `volume_minimo, volume_maximo, volume_referencia` → `constraints/generic_parameters.json` › `scalar_parameters[].values[][]`
- `produtibilidade_especifica, canal_fuga_medio, tipo_regulacao` → `constraints/generic_parameters.json` › `scalar_parameters[].values[][]`
- `codigo_usina_jusante` → `constraints/generic_parameters.json` › `scalar_parameters[].values[][]`
- `nome_usina` → `system/hydros.json` › `hydros[].name`
- `codigo_usina_jusante` → `system/hydros.json` › `hydros[].downstream_id`
- `volume_minimo` → `system/hydros.json` › `hydros[].reservoir.min_storage_hm3`
- `tipo_regulacao` → `system/hydros.json` › `hydros[].reservoir.min_storage_hm3`
- `volume_referencia` → `system/hydros.json` › `hydros[].reservoir.min_storage_hm3`
- `volume_maximo` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `tipo_regulacao` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `volume_referencia` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `a0_volume_cota..a4_volume_cota` → `system/hydros.json` › `hydros[].generation.model`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].generation.model`
- `numero_conjuntos_maquinas` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `maquinas_conjunto_1..5` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `vazao_nominal_conjunto_1..5` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `potencia_nominal_conjunto_1..5` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `queda_nominal_conjunto_1..5` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `tipo_turbina` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `a0_volume_cota..a4_volume_cota` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `volume_minimo` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `volume_maximo` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `canal_fuga_medio` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `perdas` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `tipo_perda` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `numero_conjuntos_maquinas` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `maquinas_conjunto_1..5` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `potencia_nominal_conjunto_1..5` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `nome_usina` → `system/hydros.json` › `hydros[].unit_groups[].name`
- `submercado` → `system/hydros.json` › `hydros[].unit_groups[].bus_id`
- `numero_conjuntos_maquinas` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `maquinas_conjunto_1..5` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `potencia_nominal_conjunto_1..5` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `numero_conjuntos_maquinas` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `maquinas_conjunto_1..5` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `vazao_nominal_conjunto_1..5` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `potencia_nominal_conjunto_1..5` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `queda_nominal_conjunto_1..5` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `tipo_turbina` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `a0_volume_cota..a4_volume_cota` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `volume_minimo` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `volume_maximo` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `canal_fuga_medio` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `perdas` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `tipo_perda` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].specific_productivity_mw_per_m3s_per_m`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].efficiency.value`
- `canal_fuga_medio` → `system/hydros.json` › `hydros[].tailrace.coefficients[]`
- `tipo_perda` → `system/hydros.json` › `hydros[].hydraulic_losses.type`
- `perdas` → `system/hydros.json` › `hydros[].hydraulic_losses.type`
- `perdas` → `system/hydros.json` › `hydros[].hydraulic_losses.value`
- `tipo_perda` → `system/hydros.json` › `hydros[].hydraulic_losses.value`
- `perdas` → `system/hydros.json` › `hydros[].hydraulic_losses.value_m`
- `tipo_perda` → `system/hydros.json` › `hydros[].hydraulic_losses.value_m`
- `evaporacao_JAN..evaporacao_DEZ` → `system/hydros.json` › `hydros[].evaporation.coefficients_mm[]`
- `desvio` → `system/hydros.json` › `hydros[].diversion.downstream_id`
- `desvio` → `system/hydros.json` › `hydros[].diversion.max_flow_m3s`
- `a0_volume_cota..a4_volume_cota` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].model`
- `produtibilidade_especifica` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].model`
- `volume_minimo` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `volume_maximo` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `tipo_regulacao` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `volume_referencia` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `volume_minimo` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `volume_maximo` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `tipo_regulacao` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `volume_referencia` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `volume_minimo` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].reference_volume.volume_hm3`
- `volume_maximo` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].reference_volume.volume_hm3`
- `tipo_regulacao` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].reference_volume.volume_hm3`
- `volume_referencia` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].reference_volume.volume_hm3`
- `volume_minimo` → `system/hydro_geometry.parquet` › `volume_hm3`
- `volume_maximo` → `system/hydro_geometry.parquet` › `volume_hm3`
- `tipo_regulacao` → `system/hydro_geometry.parquet` › `volume_hm3`
- `volume_referencia` → `system/hydro_geometry.parquet` › `volume_hm3`
- `a0_volume_cota..a4_volume_cota` → `system/hydro_geometry.parquet` › `height_m`
- `a0_cota_area..a4_cota_area` → `system/hydro_geometry.parquet` › `area_km2`
- `produtibilidade_especifica` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `a0_volume_cota..a4_volume_cota` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `canal_fuga_medio` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `perdas` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `tipo_perda` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `volume_minimo` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `volume_maximo` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `tipo_regulacao` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `volume_referencia` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `vazao_minima_historica` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `tipo_regulacao` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `codigo_usina_jusante` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `volume_minimo` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `volume_maximo` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `tipo_regulacao` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `volume_referencia` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `volume_minimo` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `volume_maximo` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `tipo_regulacao` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `volume_referencia` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `desvio` → `constraints/hydro_bounds.parquet` › `min_diversion_m3s`
- `desvio` → `constraints/hydro_bounds.parquet` › `max_diversion_m3s`
- `numero_conjuntos_maquinas` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `maquinas_conjunto_1..5` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `vazao_nominal_conjunto_1..5` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `potencia_nominal_conjunto_1..5` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `queda_nominal_conjunto_1..5` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `tipo_turbina` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `a0_volume_cota..a4_volume_cota` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `volume_minimo` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `volume_maximo` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `produtibilidade_especifica` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `canal_fuga_medio` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `perdas` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `tipo_perda` → `constraints/hydro_unit_group_bounds.parquet` › `max_turbined_m3s`
- `numero_conjuntos_maquinas` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `maquinas_conjunto_1..5` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `potencia_nominal_conjunto_1..5` → `constraints/hydro_unit_group_bounds.parquet` › `max_generation_mw`
- `posto` → `scenarios/external_inflow_scenarios.parquet` › `value_m3s`
- `submercado` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `submercado` → `scenarios/external_load_scenarios.parquet` › `value_mw`
- `volume_minimo` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `volume_maximo` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `volume_minimo` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `a0..a4_jusante_1..6, referencia_jusante_1..6, numero_polinomios_jusante` — *não lido.* Polinômios cota-vazão de jusante do cadastro. Não lidos: as curvas de jusante de `tailrace_curves.parquet` vêm de `polinjus`; sem `polinjus`, a usina fica com o `canal_fuga_medio` constante.
- `cota_minima, cota_maxima` — *não lido.* Cotas mínima e máxima do reservatório. Não lidas: a geometria usa o polinômio cota-volume avaliado nos volumes limites.
- `volume_vertedouro, volume_desvio` — *não lido.* Volumes de soleira do vertedouro e do canal de desvio. A tabela efetiva os carrega (com as alterações `AC VSVERT`/`VMDESV`), mas nenhum emissor os consome. Converter exige derivar deles um limite de vertimento/desvio por faixa de armazenamento, que o Novomodelo não modela.
- `fator_carga_maximo, fator_carga_minimo` — *não lido.* Fatores de carga máximo e mínimo. Não lidos; o Novomodelo não limita o fator de carga.
- `teif, ip` — *não lido.* Taxas de indisponibilidade forçada e programada. Não lidas na via DECOMP: a disponibilidade vem dos registros `MP` e `FD` do `dadger`.
- `numero_unidades_base, representacao_conjunto` — *não lido.* Número de unidades de base e forma de representação dos conjuntos. Não lidos: os grupos de unidades são derivados de `numero_conjuntos_maquinas` e das colunas por conjunto.
- `influencia_vertimento_canal_fuga` — *não lido.* Sinalizador de influência do vertimento no canal de fuga. Não lido: o Novomodelo não modela essa influência.
- `empresa, observacao, data` — *não lido.* Metadados do cadastro (empresa, observação, data). Não lidos.

### `dadgnl.rvN` › `GL`

**Estado:** convertido em parte. **Lido por:** `boundary/`, `initial_conditions.json`, `post_study_stages.json`, `system/thermals.json`.

- `data_inicio` → `initial_conditions.json` › `past_anticipated_commitments[].start_date`
- `data_inicio` → `initial_conditions.json` › `past_anticipated_commitments[].end_date`
- `geracao` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `duracao` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `data_inicio` → `system/thermals.json` › `thermals[].anticipated_config.lead_time_hours`
- `data_inicio` → `post_study_stages.json` › `stages[].start_date`
- `data_inicio` → `post_study_stages.json` › `thermal_bounds[].post_study_stage_index`
- `codigo_submercado` — *não lido.* Submercado repetido no registro de despacho comandado. Não lido: o submercado da térmica vem de `TG`.

### `dadgnl.rvN` › `GS`

**Estado:** convertido. **Lido por:** `boundary/`, `initial_conditions.json`, `post_study_stages.json`, `system/thermals.json`.

- `mes, semanas` → `system/thermals.json` › `thermals[].anticipated_config.lead_time_hours`

### `dadgnl.rvN` › `NL`

**Estado:** lido, não convertido. **Lido por:** `boundary/`, `initial_conditions.json`, `post_study_stages.json`, `system/thermals.json`.

- `codigo_submercado` — *não lido.* Submercado repetido no registro de lag. Não lido: só `codigo_usina` e `lag` são usados.

### `dadgnl.rvN` › `TG`

**Estado:** convertido. **Lido por:** `boundary/`, `initial_conditions.json`, `post_study_stages.json`, `system/thermals.json`.

- `codigo_usina` → `initial_conditions.json` › `past_anticipated_commitments[].thermal_id`
- `inflexibilidade, disponibilidade` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `codigo_usina` → `system/thermals.json` › `thermals[].id`
- `nome` → `system/thermals.json` › `thermals[].name`
- `codigo_submercado` → `system/thermals.json` › `thermals[].bus_id`
- `cvu` → `system/thermals.json` › `thermals[].cost_per_mwh`
- `inflexibilidade` → `system/thermals.json` › `thermals[].generation.min_mw`
- `disponibilidade` → `system/thermals.json` › `thermals[].generation.max_mw`
- `codigo_usina` → `post_study_stages.json` › `thermal_bounds[].thermal_id`
- `cvu` → `post_study_stages.json` › `thermal_bounds[].cost_per_mwh`
- `inflexibilidade` → `post_study_stages.json` › `thermal_bounds[].min_mw`
- `disponibilidade` → `post_study_stages.json` › `thermal_bounds[].max_mw`

### `renovaveis*`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `scenarios/external_ncs_scenarios.parquet`, `scenarios/non_controllable_factors.json`, `scenarios/non_controllable_stats.parquet`, `system/non_controllable_sources.json`.

Parques eólicos equivalentes (PEE) das LIBs. O conversor lê o cadastro, o submercado e a geração por período, patamar e cenário.

- `PEE-GER-PER-PAT-CEN · codigo_pee` → `system/non_controllable_sources.json` › `non_controllable_sources[].id`
- `PEE-CAD · nome_pee` → `system/non_controllable_sources.json` › `non_controllable_sources[].name`
- `PEE-SUBM · codigo_submercado` → `system/non_controllable_sources.json` › `non_controllable_sources[].name`
- `PEE-SUBM · codigo_submercado` → `system/non_controllable_sources.json` › `non_controllable_sources[].bus_id`
- `PEE-GER-PER-PAT-CEN · geracao` → `system/non_controllable_sources.json` › `non_controllable_sources[].max_generation_mw`
- `PEE-GER-PER-PAT-CEN · codigo_pee` → `scenarios/non_controllable_stats.parquet` › `ncs_id`
- `PEE-GER-PER-PAT-CEN · geracao` → `scenarios/non_controllable_stats.parquet` › `mean`
- `PEE-GER-PER-PAT-CEN · codigo_pee` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].ncs_id`
- `PEE-GER-PER-PAT-CEN · geracao` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].factor`
- `PEE-GER-PER-PAT-CEN · codigo_pee` → `scenarios/external_ncs_scenarios.parquet` › `ncs_id`
- `PEE-GER-PER-PAT-CEN · geracao` → `scenarios/external_ncs_scenarios.parquet` › `availability_factor`
- `PEE-CAD · codigo_pee` → `constraints/generic_constraints.json` › `constraints[].expression`
- `pee_config_per` — *não lido.* Configuração de operação do parque por período. Não lida: a geração por cenário já é o valor emitido como fonte não controlável.
- `pee_pot_inst_per` — *não lido.* Potência instalada do parque por período. Não lida: a fonte não controlável do Novomodelo não carrega capacidade instalada.

### `polinjus.csv`

**Estado:** convertido em parte. **Lido por:** `system/tailrace_curves.parquet`.

Famílias de curvas de jusante das LIBs. O conversor lê o cadastro das famílias e os segmentos polinomiais; os cards abaixo ficam de fora.

- `hidreletrica_curvajusante_polinomio_segmento · codigo_usina` → `system/tailrace_curves.parquet` › `hydro_id`
- `hidreletrica_curvajusante_polinomio_segmento · indice_familia` → `system/tailrace_curves.parquet` › `family_id`
- `hidreletrica_curvajusante · nivel_montante_referencia` → `system/tailrace_curves.parquet` › `downstream_reference_level_m`
- `hidreletrica_curvajusante_polinomio_segmento · indice_polinomio` → `system/tailrace_curves.parquet` › `segment_id`
- `hidreletrica_curvajusante_polinomio_segmento · limite_inferior_vazao_jusante` → `system/tailrace_curves.parquet` › `outflow_min_m3s`
- `hidreletrica_curvajusante_polinomio_segmento · limite_superior_vazao_jusante` → `system/tailrace_curves.parquet` › `outflow_max_m3s`
- `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a0` → `system/tailrace_curves.parquet` › `coefficient_0`
- `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a1` → `system/tailrace_curves.parquet` › `coefficient_1`
- `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a2` → `system/tailrace_curves.parquet` › `coefficient_2`
- `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a3` → `system/tailrace_curves.parquet` › `coefficient_3`
- `hidreletrica_curvajusante_polinomio_segmento · coeficiente_a4` → `system/tailrace_curves.parquet` › `coefficient_4`
- `hidreletrica_curvajusante_polinomio` — *não lido.* Cabeçalho de polinômio por família. Não lido: os segmentos trazem os coeficientes diretamente.
- `hidreletrica_curvajusante_afogamentoexplicito_padrao` — *não lido.* Habilitação padrão do tratamento de afogamento explícito. Não lida: o Novomodelo escolhe a família pela defluência de jusante sem afogamento explícito.
- `hidreletrica_curvajusante_afogamentoexplicito_usina` — *não lido.* Habilitação do afogamento explícito por usina. Não lida; ver o card padrão.

### `lib_restricao-eletrica-especial*.csv`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`.

Restrições elétricas especiais das LIBs. O conversor lê apenas os cards de forma longa indexados por período; as variantes abaixo ficam de fora.

- `RESTRICAO-ELETRICA-FORMULA · codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].id`
- `RESTRICAO-ELETRICA-INEQUACAO · codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].id`
- `RESTRICAO-ELETRICA-FORMULA · codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].name`
- `RESTRICAO-ELETRICA-INEQUACAO · codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].name`
- `RESTRICAO-ELETRICA-FORMULA · codigo_restricao` → `constraints/generic_constraints.json` › `constraints[].description`
- `RESTRICAO-ELETRICA-FORMULA · formula` → `constraints/generic_constraints.json` › `constraints[].expression`
- `RESTRICAO-ELETRICA-INEQUACAO · formula` → `constraints/generic_constraints.json` › `constraints[].expression`
- `RESTRICAO-ELETRICA-INEQUACAO · formula_limite` → `constraints/generic_constraints.json` › `constraints[].expression`
- `EXPRESSAO-ELETRICA · formula` → `constraints/generic_constraints.json` › `constraints[].expression`
- `RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO · custo_violacao` → `constraints/generic_constraints.json` › `constraints[].slack.penalty`
- `RESTRICAO-ELETRICA-FORMULA · codigo_restricao` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `RESTRICAO-ELETRICA-HORIZONTE-PERIODO · estagio_inicio` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `RESTRICAO-ELETRICA-HORIZONTE-PERIODO · estagio_fim` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `RESTRICAO-ELETRICA-HABILITA · codigo_regra_ativacao` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `RESTRICAO-ELETRICA-REGRA-ATIVACAO · regra_ativacao` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR · patamar` → `constraints/generic_constraint_bounds.parquet` › `block_id`
- `RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR · patamar` → `constraints/generic_constraint_bounds.parquet` › `block_id`
- `RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR · limite_inferior` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `RESTRICAO-ELETRICA-INEQUACAO · operador` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `RESTRICAO-ELETRICA-INEQUACAO · formula_limite` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR · formula_limite` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR · valor` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR · limite_superior` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `RESTRICAO-ELETRICA-INEQUACAO · operador` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `RESTRICAO-ELETRICA-INEQUACAO · formula_limite` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR · formula_limite` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR · valor` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `cards de forma curta (RE, RE-*, ALIAS-ELET*)` — *adiado.* Apelidos curtos dos mesmos cards. Não lidos: um arquivo só com forma curta cai no aviso `decomp-libs-electrical-present`. Converter exige ler os apelidos pelo acessor idecomp correspondente.
- `cards indexados por data (RESTRICAO-ELETRICA-HORIZONTE-DATA, -FORMULA-DATA-PATAMAR, -LIMITES-FORMULA-DATA-PATAMAR)` — *adiado.* Variantes em que o horizonte e os limites são dados por intervalo de datas em vez de estágios. Fora do escopo do leitor; ficam no aviso `decomp-libs-electrical-present`. Converter exige mapear as datas para índices de estágio do calendário operativo.
- `restricao_eletrica_formula_periodo_patamar` — *não lido.* Fórmula da restrição variando por intervalo de estágios e patamar. Não lida: só a fórmula fixa e os limites por período e patamar são lidos.
- `restricao_eletrica_tratamento_violacao_periodo` — *não lido.* Tratamento de violação por intervalo de estágios. Não lido: só o tratamento fixo da restrição é usado.

### `cortesh.dat`

**Estado:** convertido. **Lido por:** `boundary/`, `stages.json`.

- `alfa_cvar, lambda_cvar` → `stages.json` › `stages[].risk_measure`
- `alfa_cvar` → `stages.json` › `stages[].risk_measure.cvar.alpha`
- `lambda_cvar` → `stages.json` › `stages[].risk_measure.cvar.lambda`

### `cortes*.dat`

**Estado:** lido, não convertido. **Lido por:** `boundary/`.

### `caso.dat`

**Estado:** arquivo de índice, lido para localizar os demais.

Nomeia a revisão (`rv0`, `rv1`, ...) cujo índice lista os arquivos de dados. Lido apenas para localizar os demais arquivos do deck.

### `indices.csv`

**Estado:** arquivo de índice, lido para localizar os demais. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`.

Índice das LIBs do deck. O bridge lê apenas a entrada `RESTRICAO-ELETRICA-ESPECIAL` (caminho do arquivo de restrições elétricas).

- `demais entradas (inclusive as irmãs -ATIVACAO e -VIOLACAO)` — *não lido.* Os demais arquivos listados no índice não são abertos; `renovaveis*` e `polinjus*` são resolvidos por prefixo de nome, não pelo índice.

### `mlt.dat`

**Estado:** lido, não convertido. **Lido por:** `boundary/`.

### `prevs.rvN`

**Estado:** não lido.

Previsões de vazão semanais por posto, entrada do GEVAZP. Não lido: o conversor usa as previsões já consolidadas na seção de previsões de `vazoes.rvN`.

### `postos.dat`

**Estado:** não lido.

Cadastro de postos fluviométricos. Não lido: a ligação usina-posto vem da coluna `posto` de `hidr.dat` (com a alteração `AC NUMPOS`).

### `cortdeco.rvN / mapcut.rvN`

**Estado:** não lido.

Cortes e mapa de cortes da própria FCF do DECOMP, gravados por uma execução anterior. Não lidos: o bridge importa apenas a FCF de fronteira do NEWAVE (`cortesh.dat` e `cortes*.dat`) para o checkpoint `boundary/`.

### `arquivos de saída do DECOMP`

**Estado:** não lido.

Resultados de uma execução do DECOMP (`relato.rvN`, `dec_oper_*.csv`, `dec_estatfpha.csv`, `sumario.rvN`, `inviab_unic.rvN`, entre outros). Não são entrada da conversão; `compare decomp` lê `relato.rvN`, as tabelas `dec_oper_*.csv` e `dec_estatfpha.csv` para confrontar com a saída do Novomodelo.

## Ainda não convertido

Tudo o que o conversor deixa de fora, reunido em um só lugar. Cada item aparece também no índice acima, junto do arquivo a que pertence.

### Arquivos do deck não lidos

- `prevs.rvN` — *não lido.* Previsões de vazão semanais por posto, entrada do GEVAZP. Não lido: o conversor usa as previsões já consolidadas na seção de previsões de `vazoes.rvN`.
- `postos.dat` — *não lido.* Cadastro de postos fluviométricos. Não lido: a ligação usina-posto vem da coluna `posto` de `hidr.dat` (com a alteração `AC NUMPOS`).
- `cortdeco.rvN / mapcut.rvN` — *não lido.* Cortes e mapa de cortes da própria FCF do DECOMP, gravados por uma execução anterior. Não lidos: o bridge importa apenas a FCF de fronteira do NEWAVE (`cortesh.dat` e `cortes*.dat`) para o checkpoint `boundary/`.
- `arquivos de saída do DECOMP` — *não lido.* Resultados de uma execução do DECOMP (`relato.rvN`, `dec_oper_*.csv`, `dec_estatfpha.csv`, `sumario.rvN`, `inviab_unic.rvN`, entre outros). Não são entrada da conversão; `compare decomp` lê `relato.rvN`, as tabelas `dec_oper_*.csv` e `dec_estatfpha.csv` para confrontar com a saída do Novomodelo.

### Registros e campos do deck não convertidos

- `dadger.rvN` › `VI` — *adiado.* Tempo de viagem da água até a usina de jusante e defluências das semanas anteriores ao estudo. O conversor lê o registro e monta `travel_time_hours` e `initial_conditions.past_defluences`, mas a emissão está desligada por um defeito na geração de cortes com água em trânsito no novomodelo (gap rastreado C-travel-time): o `convert decomp` apenas avisa `deferred at this milestone: water travel time (VI present)`.
- `dadger.rvN` › `FE` — *adiado.* Coeficientes de participação elétrica de uma restrição `RE`. O idecomp não expõe leitor para este registro; o bridge detecta sua presença por varredura textual e emite `decomp-fe-participation-unreadable`, pois uma `RE` com termo `FE` fica sub-lida. Converter exige um leitor próprio e o descarte, por restrição, das `RE` que carregam `FE`.
- `dadger.rvN` › `HA` — *adiado.* Cadastro da família de restrições `HA`/`LA`/`CA` (declaração, limites e coeficientes), sem leitor no idecomp. Detectada por varredura textual; diagnóstico `decomp-rha-family-unconverted`. Converter exige leitor próprio e um emissor equivalente ao das famílias `RE`/`HQ`/`HV`.
- `dadger.rvN` › `LA` — *adiado.* Limites por estágio da família `HA`/`LA`/`CA`; ver `HA`.
- `dadger.rvN` › `CA` — *adiado.* Coeficientes da família `HA`/`LA`/`CA`; ver `HA`.
- `dadger.rvN` › `AC` › `ALTEFE` — *adiado.* Alteração da altura efetiva de queda de um conjunto de máquinas. O leitor idecomp expõe só as colunas de identificação e data, sem a coluna de valor, logo o conversor não consegue ingerir a alteração: o `check decomp` emite `decomp-ac-altefe-uningestable` e a queda nominal de `hidr.dat` é mantida. Depende de o idecomp expor o valor.
- `dadger.rvN` › `AC` › `COFEVA` — *adiado.* Alteração do coeficiente de evaporação mensal. Lido pelo `check decomp` (diagnóstico `decomp-ac-overrides-deferred`), sem consumidor no conversor: a evaporação usa os coeficientes mensais de `hidr.dat`. Converter exige sobrepor o coeficiente do mês na tabela efetiva antes de emitir os coeficientes de evaporação.
- `dadger.rvN` › `AC` › `COTARE` — *adiado.* Alteração de um coeficiente do polinômio cota-área. Lido pelo `check decomp`, sem consumidor: a FPHA usa o polinômio de `hidr.dat`. Converter exige sobrepor o coeficiente na tabela efetiva usada pelo ajuste da FPHA.
- `dadger.rvN` › `AC` › `COTVAZ` — *adiado.* Alteração de um coeficiente do polinômio cota-vazão (curva de jusante). Lido pelo `check decomp`, sem consumidor: as curvas de jusante vêm de `polinjus`. Converter exige aplicar a alteração sobre a família de `tailrace_curves.parquet` da usina.
- `dadger.rvN` › `AC` › `JUSENA` — *adiado.* Alteração do índice de aproveitamento de jusante para o cálculo de energia armazenada e afluente. Lido pelo `check decomp`, sem consumidor: o Novomodelo não agrega energia por REE.
- `dadger.rvN` › `AC` › `NCHAVE` — *adiado.* Alteração do número da curva-chave (cota-vazão) e do nível de jusante da faixa. Lido pelo `check decomp`, sem consumidor: a seleção de família de jusante segue `polinjus`.
- `dadger.rvN` › `AC` › `NPOSNW` — *adiado.* Alteração do posto de acoplamento com o NEWAVE. Lido pelo `check decomp`, sem consumidor: o acoplamento da FCF de fronteira usa o `posto` de `hidr.dat` (com `NUMPOS`) e o registro `CX`.
- `dadger.rvN` › `AC` › `TIPERH` — *adiado.* Alteração do tipo de perdas hidráulicas. Lido pelo `check decomp`, sem consumidor: o conversor usa o `tipo_perda` de `hidr.dat`. Converter exige sobrepor o tipo na tabela efetiva antes de calcular perdas e produtibilidade.
- `dadger.rvN` › `AC` › `VERTJU` — *adiado.* Consideração da influência do vertimento no canal de fuga. Lido pelo `check decomp`, sem consumidor: o Novomodelo não modela essa influência na cota de jusante.
- `dadger.rvN` › `CE` › `fator_perdas` — *adiado.* Fator de perdas do contrato de exportação. Lido e ignorado com o diagnóstico `contract-loss-factor-unmapped`: `energy_contracts` do Novomodelo não tem campo equivalente e o fator não é dobrado no preço nem nos limites.
- `dadger.rvN` › `CI` › `fator_perdas` — *adiado.* Fator de perdas do contrato de importação; mesmo tratamento do `CE`.
- `dadger.rvN` › `FU` › `frequencia` — *adiado.* Frequência (50/60 Hz) do termo de geração hidráulica de uma `RE`. Uma restrição genérica com termo dividido por frequência é descartada com `decomp-re-frequency-split-deferred`, pois o conversor não constrói o mapa frequência-barra; no caminho de limite simples a frequência é ignorada e o termo vale para a usina inteira.
- `dadger.rvN` › `CV` › `tipo (VDEF, VDES, VBOM)` — *adiado.* Tipos de volume de uma `HV` além de `VARM`. A forma hm³-para-vazão exige coeficiente por estágio (horas de patamar variam), que a expressão de restrição genérica do Novomodelo não carrega; a restrição é descartada com `decomp-rhv-volume-tipo-deferred`.
- `dadger.rvN` › `UH` › `balanco_hidrico_patamar` — *não lido.* Sinalizador de balanço hídrico por patamar. Não lido: o Novomodelo faz o balanço por estágio.
- `dadger.rvN` › `UH` › `configuracao_newave` — *não lido.* Índice da configuração do NEWAVE para acoplamento. Não lido: a FCF de fronteira é acoplada por código de usina e por `CX`.
- `dadger.rvN` › `UH` › `estagio_inicio_producao` — *não lido.* Estágio em que a usina começa a produzir. Não lido: o conversor não emite `entry_stage_id` para hidrelétricas (fica no padrão nulo do esquema). Converter exige mapear este estágio para `entry_stage_id`.
- `dadger.rvN` › `UH` › `limite_superior_vertimento` — *não lido.* Limite superior de vertimento. Não lido; converter exige emiti-lo como `max_spillage_m3s` em `hydro_bounds.parquet`.
- `dadger.rvN` › `RI` › `geracao_maxima_50_hz_1..5` — *não lido.* Geração máxima do setor de 50 Hz de Itaipu por patamar. Não lida: o conversor usa a geração máxima de 60 Hz e as mínimas de 50 e 60 Hz. Converter exige emiti-la como teto do grupo de unidades de 50 Hz.
- `dadger.rvN` › `CD` › `nome_curva` — *não lido.* Nome da curva de déficit. Não lido; apenas identificação.
- `dadger.rvN` › `NI` › `tipo_limite` — *não lido.* Tipo do limite de iterações. Não lido: só `iteracoes` é usado no critério `iteration_limit`.
- `dadger.rvN` › `HE` › `forma_calculo_produtibilidades` — *não lido.* Forma de cálculo das produtibilidades da restrição de energia armazenada. Capturada pelo censo de restrições, mas nenhum emissor a consome: a energia é expandida com a produtibilidade acumulada calculada pelo conversor.
- `dadger.rvN` › `HE` › `tipo_valores_produtibilidades` — *não lido.* Tipo dos valores de produtibilidade; capturado e não consumido, como `forma_calculo_produtibilidades`.
- `dadger.rvN` › `HE` › `arquivo_produtibilidades` — *não lido.* Arquivo externo de produtibilidades da restrição; capturado e não consumido, e o arquivo nunca é aberto.
- `dadger.rvN` › `HE` › `tipo_penalidade` — *não lido.* Tipo da penalidade da restrição de energia; só `valor_penalidade` é usado como penalidade da folga.
- `dadger.rvN` › `FU` › `estagio` — *não lido.* Estágio em que a participação é declarada. Ignorado: o coeficiente é herdado por toda a faixa `[estagio_inicial, estagio_final]` da restrição.
- `dadger.rvN` › `FT` › `estagio` — *não lido.* Estágio da participação térmica; ignorado como em `FU`.
- `dadger.rvN` › `FI` › `estagio` — *não lido.* Estágio da participação de intercâmbio; ignorado como em `FU`.
- `dadger.rvN` › `CQ` › `estagio` — *não lido.* Estágio da participação de vazão; ignorado como em `FU`.
- `dadger.rvN` › `CV` › `estagio` — *não lido.* Estágio da participação de volume; ignorado como em `FU`.
- `dadger.rvN` › `DA` — *não lido.* Retiradas de água para outros usos (desvios) por UHE: usina de retirada, usina de retorno, vazão desviada, percentual de retorno e custo, por estágio. Não lido. Converter exige mapear a retirada para o canal de desvio da usina no Novomodelo e o retorno como afluência à usina de destino.
- `dadger.rvN` › `EA` — *não lido.* ENA dos meses anteriores ao estudo, por REE, para a tendência hidrológica do DECOMP. Não lido: a árvore de cenários vem pronta de `vazoes.rvN` e o estado de afluências passadas da FCF de fronteira é derivado das observações de `vazoes.rvN` e de `mlt.dat`.
- `dadger.rvN` › `ES` — *não lido.* ENA das semanas anteriores ao estudo, por REE. Não lido, pelo mesmo motivo de `EA`.
- `dadger.rvN` › `EZ` — *não lido.* Percentual máximo do volume útil para acoplamento com o NEWAVE. O `check decomp` tenta inventariá-lo, mas o leitor idecomp não expõe acessor `ez`, então o registro nunca é lido.
- `dadger.rvN` › `FP` — *não lido.* Parâmetros da função de produção por usina e estágio: janelas de volume e turbinamento e número de pontos de discretização. Não lido: o Novomodelo ajusta a FPHA internamente (`fpha_configs` com `source = computed`), com janela derivada do volume inicial e dos limites de armazenamento.
- `dadger.rvN` › `IR` — *não lido.* Opções de geração de relatórios de saída do DECOMP. Sem contrapartida no caso Novomodelo.
- `dadger.rvN` › `MT` — *não lido.* Manutenções programadas das UTEs (fator por estágio). Não lido: a disponibilidade térmica vem só de `CT`. Converter exige multiplicar `disponibilidade` pelo fator ao emitir `thermal_bounds.parquet`.
- `dadger.rvN` › `PD` — *não lido.* Escolha do algoritmo de solução do PL. Parâmetro do solver do DECOMP; sem contrapartida.
- `dadger.rvN` › `PE` — *não lido.* Alteração das penalidades de vertimento, intercâmbio e desvio por submercado. Não lido: `penalties.json` usa os valores padrão do conversor. Converter exige mapear cada `tipo` para o campo de penalidade correspondente.
- `dadger.rvN` › `PV` — *não lido.* Penalidades das variáveis de folga e tolerâncias de viabilidade das restrições. Parâmetros internos do solver do DECOMP; sem contrapartida.
- `dadger.rvN` › `QI` — *não lido.* Tempo de viagem por usina usado no cálculo da ENA. Não lido: só afeta a agregação em energia, que o Novomodelo não faz.
- `dadger.rvN` › `RT` — *não lido.* Retirada das restrições de soleira de vertedouro e de canal de desvio. Não lido; converter exige suprimir o limite correspondente em `hydro_bounds.parquet`.
- `dadger.rvN` › `TS` — *não lido.* Tolerâncias do solver do DECOMP. Sem contrapartida.
- `dadger.rvN` › `VA` — *não lido.* Posto cuja vazão influencia a cota de jusante de outra usina (fator de impacto incremental). Não lido: o Novomodelo não modela influência de vazão lateral na cota de jusante.
- `dadger.rvN` › `VL` — *não lido.* Usina que sofre influência de vazão lateral na cota de jusante (polinômio e fator de impacto). Não lido; ver `VA`.
- `dadger.rvN` › `VU` — *não lido.* Usina cuja defluência influencia a cota de jusante de outra. Não lido; ver `VA`.
- `dadger.rvN` › `CS` — *não lido.* Habilita a consistência de dados do DECOMP. Sem contrapartida.
- `dadger.rvN` › `EV` — *não lido.* Configuração da evaporação (modelo e volume de referência). Não lido: a evaporação é convertida a partir do sinalizador `evaporacao` do `UH` e dos coeficientes mensais de `hidr.dat`.
- `dadger.rvN` › `FA` — *não lido.* Nome do arquivo índice CSV das LIBs. Não lido: o bridge procura `indices.csv` diretamente no diretório do deck.
- `dadger.rvN` › `FJ` — *não lido.* Nome do arquivo de polinômios de jusante. Não lido: o bridge resolve `polinjus*` por prefixo no diretório do deck.
- `dadger.rvN` › `PU` — *não lido.* Habilita a solução do problema via PL único. Sem contrapartida.
- `dadger.rvN` › `RC` — *não lido.* Inclusão de restrições do tipo escada por mnemônico. Não lido.
- `dadger.rvN` › `TE` — *não lido.* Título do estudo. Não lido.
- `dadger.rvN` › `VT` — *não lido.* Nome do arquivo de cenários de vento. Não lido: a geração eólica vem de `renovaveis`.
- `dadgnl.rvN` › `GL` › `codigo_submercado` — *não lido.* Submercado repetido no registro de despacho comandado. Não lido: o submercado da térmica vem de `TG`.
- `dadgnl.rvN` › `NL` › `codigo_submercado` — *não lido.* Submercado repetido no registro de lag. Não lido: só `codigo_usina` e `lag` são usados.
- `hidr.dat` › `a0..a4_jusante_1..6, referencia_jusante_1..6, numero_polinomios_jusante` — *não lido.* Polinômios cota-vazão de jusante do cadastro. Não lidos: as curvas de jusante de `tailrace_curves.parquet` vêm de `polinjus`; sem `polinjus`, a usina fica com o `canal_fuga_medio` constante.
- `hidr.dat` › `cota_minima, cota_maxima` — *não lido.* Cotas mínima e máxima do reservatório. Não lidas: a geometria usa o polinômio cota-volume avaliado nos volumes limites.
- `hidr.dat` › `volume_vertedouro, volume_desvio` — *não lido.* Volumes de soleira do vertedouro e do canal de desvio. A tabela efetiva os carrega (com as alterações `AC VSVERT`/`VMDESV`), mas nenhum emissor os consome. Converter exige derivar deles um limite de vertimento/desvio por faixa de armazenamento, que o Novomodelo não modela.
- `hidr.dat` › `fator_carga_maximo, fator_carga_minimo` — *não lido.* Fatores de carga máximo e mínimo. Não lidos; o Novomodelo não limita o fator de carga.
- `hidr.dat` › `teif, ip` — *não lido.* Taxas de indisponibilidade forçada e programada. Não lidas na via DECOMP: a disponibilidade vem dos registros `MP` e `FD` do `dadger`.
- `hidr.dat` › `numero_unidades_base, representacao_conjunto` — *não lido.* Número de unidades de base e forma de representação dos conjuntos. Não lidos: os grupos de unidades são derivados de `numero_conjuntos_maquinas` e das colunas por conjunto.
- `hidr.dat` › `influencia_vertimento_canal_fuga` — *não lido.* Sinalizador de influência do vertimento no canal de fuga. Não lido: o Novomodelo não modela essa influência.
- `hidr.dat` › `empresa, observacao, data` — *não lido.* Metadados do cadastro (empresa, observação, data). Não lidos.
- `vazoes.rvN` › `previsoes_com_postos_artificiais` — *não lido.* Previsões incluindo os postos artificiais de acoplamento. Não lidas: só os postos das usinas operadas interessam.
- `vazoes.rvN` › `cenarios_calculados_com_postos_artificiais` — *não lido.* Cenários gerados incluindo postos artificiais. Não lidos, pelo mesmo motivo.
- `vazoes.rvN` › `numero_aberturas_estagios` — *não lido.* Número de aberturas por estágio. Não lido: a forma da árvore é deduzida dos próprios cenários e das probabilidades.
- `polinjus.csv` › `hidreletrica_curvajusante_polinomio` — *não lido.* Cabeçalho de polinômio por família. Não lido: os segmentos trazem os coeficientes diretamente.
- `polinjus.csv` › `hidreletrica_curvajusante_afogamentoexplicito_padrao` — *não lido.* Habilitação padrão do tratamento de afogamento explícito. Não lida: o Novomodelo escolhe a família pela defluência de jusante sem afogamento explícito.
- `polinjus.csv` › `hidreletrica_curvajusante_afogamentoexplicito_usina` — *não lido.* Habilitação do afogamento explícito por usina. Não lida; ver o card padrão.
- `renovaveis*` › `pee_config_per` — *não lido.* Configuração de operação do parque por período. Não lida: a geração por cenário já é o valor emitido como fonte não controlável.
- `renovaveis*` › `pee_pot_inst_per` — *não lido.* Potência instalada do parque por período. Não lida: a fonte não controlável do Novomodelo não carrega capacidade instalada.
- `lib_restricao-eletrica-especial*.csv` › `cards de forma curta (RE, RE-*, ALIAS-ELET*)` — *adiado.* Apelidos curtos dos mesmos cards. Não lidos: um arquivo só com forma curta cai no aviso `decomp-libs-electrical-present`. Converter exige ler os apelidos pelo acessor idecomp correspondente.
- `lib_restricao-eletrica-especial*.csv` › `cards indexados por data (RESTRICAO-ELETRICA-HORIZONTE-DATA, -FORMULA-DATA-PATAMAR, -LIMITES-FORMULA-DATA-PATAMAR)` — *adiado.* Variantes em que o horizonte e os limites são dados por intervalo de datas em vez de estágios. Fora do escopo do leitor; ficam no aviso `decomp-libs-electrical-present`. Converter exige mapear as datas para índices de estágio do calendário operativo.
- `lib_restricao-eletrica-especial*.csv` › `restricao_eletrica_formula_periodo_patamar` — *não lido.* Fórmula da restrição variando por intervalo de estágios e patamar. Não lida: só a fórmula fixa e os limites por período e patamar são lidos.
- `lib_restricao-eletrica-especial*.csv` › `restricao_eletrica_tratamento_violacao_periodo` — *não lido.* Tratamento de violação por intervalo de estágios. Não lido: só o tratamento fixo da restrição é usado.

### Campos do Novomodelo sem origem no deck

- `penalties.json` › `bus.deficit_segments[].depth_mw` — Sempre nulo: segmento único de profundidade total (`CD` deve declarar `limite_superior` = 100 %).
- `initial_conditions.json` › `filling_storage` — Sempre lista vazia: o enchimento de volume morto não é convertido.
- `system/buses.json` › `buses[].deficit_segments[].depth_mw` — Sempre nulo: segmento único de profundidade total; um `limite_superior` diferente de 100 % em `CD` é rejeitado.
- `system/thermals.json` › `thermals[].entry_stage_id` — Sempre nulo: térmicas GNL estão ativas em todo o horizonte. Térmicas de `CT` omitem o campo.
- `system/thermals.json` › `thermals[].exit_stage_id` — Sempre nulo: térmicas GNL não saem dentro do horizonte. Térmicas de `CT` omitem o campo.
- `system/hydros.json` › `hydros[].outflow.max_outflow_m3s` — Sempre nulo: o DECOMP não declara defluência máxima cadastral. Um teto de `QDEF` (RHQ de termo único) vai para `constraints/hydro_bounds.parquet`.
- `system/hydro_energy_productivity.parquet` › `stage_id` — Sempre nulo: o valor vale para todos os estágios; a variação de queda por estágio entra pelo teto de engolimento em `constraints/hydro_unit_group_bounds.parquet`.
- `system/hydro_energy_productivity.parquet` › `reference_outflow_m3s` — Sempre nulo: o ρ_eq é uma constante, sem vazão de referência; a coluna existe porque o leitor exige o conjunto completo de colunas.
- `system/hydro_energy_productivity.parquet` › `specific_productivity_mw_per_m3s_per_m` — Sempre nulo: a produtibilidade específica só é emitida, em `system/hydros.json`, para as usinas FPHA, que não têm linha nesta tabela.
- `constraints/hydro_unit_group_bounds.parquet` › `min_turbined_m3s` — Sempre nulo: nenhum registro do DECOMP declara turbinamento mínimo por grupo.
