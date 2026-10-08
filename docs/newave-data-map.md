<!-- Gerado por scripts/gen-lineage-docs.py a partir de docs/lineage/newave.toml. Não edite à mão. -->

# Mapa de dados: NEWAVE → Novomodelo

Esta página mostra de onde vem cada arquivo e cada campo do caso Novomodelo que `convert newave` escreve: qual arquivo do deck NEWAVE, qual registro ou coluna, e que transformação é aplicada no caminho. Ela também lista o que o conversor ainda não converte, para que o trabalho pendente fique visível no mesmo lugar.

O conteúdo é gerado a partir de `docs/lineage/newave.toml` por `scripts/gen-lineage-docs.py` e verificado pelos testes contra uma conversão real do deck de exemplo do repositório, de modo que a página não pode divergir do código sem quebrar a build.

**Como ler.** Nas matrizes, ● indica que o arquivo gerado (coluna) depende do arquivo do deck (linha). *(opcional)* marca um arquivo que o deck pode não trazer; a conversão prossegue sem ele. Nas tabelas de campos, a coluna Origem cita arquivo › registro › coluna do deck e a coluna Transformação diz o que o conversor faz com o valor. *Derivado* marca um valor de escrituração (ids, datas, ordem) calculado a partir do deck; *constante* um valor fixo que o conversor sempre escreve; *sempre nulo* um campo do Novomodelo que ainda não recebe informação do deck.

## Visão geral

Uma matriz por diretório do caso. As linhas são os arquivos do deck (no DECOMP, também os registros do `dadger.rvN`); as colunas, os arquivos gerados. As seções seguintes detalham cada coluna.

### Raiz do caso

| Arquivo do deck | `config.json` | `stages.json` | `penalties.json` | `initial_conditions.json` |
| --- | :-: | :-: | :-: | :-: |
| `dger.dat` | ● | ● |  | ● |
| `confhd.dat` |  |  | ● | ● |
| `conft.dat` |  |  |  | ● |
| `sistema.dat` |  |  | ● |  |
| `term.dat` |  |  |  | ● |
| `patamar.dat` |  | ● |  | ● |
| `hidr.dat` |  |  | ● | ● |
| `modif.dat` (opcional) |  |  | ● | ● |
| `penalid.dat` (opcional) |  |  | ● |  |
| `vazpast.dat` (opcional) |  |  |  | ● |
| `exph.dat` (opcional) |  |  | ● | ● |
| `cvar.dat` (opcional) |  | ● |  |  |
| `shist.dat` (opcional) | ● | ● |  |  |
| `adterm.dat` (opcional) |  |  |  | ● |

### `system/`

| Arquivo do deck | `buses.json` | `lines.json` | `thermals.json` | `hydros.json` | `hydro_production_models.json` | `hydro_geometry.parquet` | `hydro_energy_productivity.parquet` | `tailrace_curves.parquet` | `non_controllable_sources.json` |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `dger.dat` | ● | ● | ● | ● | ● |  | ● |  | ● |
| `confhd.dat` |  |  |  | ● | ● | ● | ● | ● |  |
| `conft.dat` |  |  | ● |  |  |  |  |  |  |
| `sistema.dat` | ● | ● | ● | ● |  |  |  |  | ● |
| `clast.dat` |  |  | ● |  |  |  |  |  |  |
| `term.dat` |  |  | ● |  |  |  |  |  |  |
| `ree.dat` | ● | ● | ● | ● |  |  |  |  | ● |
| `patamar.dat` |  |  | ● |  |  |  |  |  |  |
| `hidr.dat` |  |  |  | ● | ● | ● | ● | ● |  |
| `modif.dat` (opcional) |  |  |  | ● | ● | ● | ● |  |  |
| `expt.dat` (opcional) |  |  | ● |  |  |  |  |  |  |
| `exph.dat` (opcional) |  |  |  | ● | ● | ● | ● | ● |  |
| `manutt.dat` (opcional) |  |  | ● |  |  |  |  |  |  |
| `volref_saz.dat` (opcional) |  |  |  | ● | ● |  | ● |  |  |
| `adterm.dat` (opcional) |  |  | ● |  |  |  |  |  |  |
| `polinjus.csv` (opcional) |  |  |  |  |  |  |  | ● |  |
| `tratamento-fpha.csv` (opcional) |  |  |  |  | ● |  |  |  |  |

### `scenarios/`

| Arquivo do deck | `inflow_history.parquet` | `inflow_seasonal_stats.parquet` | `load_seasonal_stats.parquet` | `load_factors.json` | `non_controllable_stats.parquet` | `non_controllable_factors.json` |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| `dger.dat` | ● | ● | ● | ● | ● | ● |
| `confhd.dat` | ● | ● |  |  |  |  |
| `sistema.dat` |  |  | ● | ● | ● | ● |
| `ree.dat` |  |  | ● | ● | ● | ● |
| `patamar.dat` |  |  |  | ● |  | ● |
| `hidr.dat` | ● | ● |  |  |  |  |
| `vazoes.dat` | ● | ● |  |  |  |  |
| `exph.dat` (opcional) | ● | ● |  |  |  |  |
| `c_adic.dat` (opcional) |  |  | ● |  |  |  |

### `constraints/`

| Arquivo do deck | `hydro_bounds.parquet` | `thermal_bounds.parquet` | `line_bounds.parquet` | `penalty_overrides_bus.parquet` | `penalty_overrides_hydro.parquet` | `generic_constraints.json` | `generic_constraint_bounds.parquet` | `generic_parameters.json` |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `dger.dat` | ● | ● | ● | ● | ● | ● | ● | ● |
| `confhd.dat` | ● |  |  |  | ● | ● | ● | ● |
| `conft.dat` |  | ● |  |  |  |  |  |  |
| `sistema.dat` |  |  | ● | ● | ● | ● | ● |  |
| `clast.dat` |  | ● |  |  |  |  |  |  |
| `term.dat` |  | ● |  |  |  |  |  |  |
| `ree.dat` |  |  |  | ● |  | ● | ● | ● |
| `patamar.dat` |  |  | ● |  |  | ● | ● |  |
| `hidr.dat` | ● |  |  |  | ● | ● | ● | ● |
| `modif.dat` (opcional) | ● |  |  |  | ● | ● | ● | ● |
| `ghmin.dat` (opcional) | ● |  |  |  |  |  |  |  |
| `penalid.dat` (opcional) |  |  |  |  | ● | ● | ● |  |
| `dsvagua.dat` (opcional) | ● |  |  |  |  |  |  |  |
| `curva.dat` (opcional) |  |  |  |  |  | ● | ● | ● |
| `expt.dat` (opcional) |  | ● |  |  |  |  |  |  |
| `exph.dat` (opcional) | ● |  |  |  | ● | ● | ● | ● |
| `manutt.dat` (opcional) |  | ● |  |  |  |  |  |  |
| `agrint.dat` (opcional) |  |  |  |  |  | ● | ● |  |
| `re.dat` (opcional) |  |  |  |  |  | ● | ● |  |
| `volref_saz.dat` (opcional) | ● |  |  |  |  |  |  |  |
| `restricao-eletrica.csv` |  |  |  |  |  | ● | ● |  |

## Arquivos gerados

### `config.json`

**Lê:** `dger.dat`, `shist.dat` (opcional)  
**Quando:** sempre.  
**Esquema:** [config.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/config.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/temporal.py`

Parâmetros de treinamento e simulação do Novomodelo derivados dos flags de `dger.dat`; `shist.dat` entra somente no modo determinístico e na simulação histórica.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `estimation.max_order` | `dger.dat` › `ordem_maxima_parp` | Valor direto (6 quando ausente). No modo determinístico (`num_forwards` = 1, `num_aberturas` = 1, `tipo_simulacao_final` = 2 e `shist.dat` com `varredura` = 0 e um único ano) é forçado a 0 para não carregar estado de defasagem de afluência. |
| `estimation.order_selection` | `dger.dat` › `consideracao_media_anual_afluencias` | 0 → `pacf`; 1, 2 ou 3 → `pacf_annual` (o Novomodelo implementa a variante exata de 12 eixos, opção 3). Campo omitido quando o flag está ausente; no modo determinístico é `pacf`. |
| `exports.states` | `dger.dat` › `impressao_estados_geracao_cortes` | `true` quando o flag vale 0 (NEWAVE grava os estados visitados na geração de cortes); qualquer outro valor mantém `false`. |
| `exports.stochastic` | — *(constante)* | Sempre `true`: a exportação do modelo estocástico é ligada para permitir a comparação das estatísticas de afluência. |
| `modeling.inflow_non_negativity.method` | — *(constante)* | Sempre `truncation_with_penalty`: a afluência incremental negativa é truncada e a folga penalizada por `hydro.inflow_nonnegativity_cost`. |
| `simulation.enabled` | `dger.dat` › `tipo_execucao`, `dger.dat` › `tipo_simulacao_final` | `true` quando `tipo_execucao` = 0 (somente simulação); caso contrário `true` se `tipo_simulacao_final` ≠ 0. |
| `simulation.selection.method` | — *(constante)* | Sempre `sampled`. |
| `simulation.selection.num_scenarios` | `dger.dat` › `num_series_sinteticas`, `shist.dat` › `anos_inicio_simulacoes`, `shist.dat` › `ano_inicio_varredura` | `num_series_sinteticas` (200 quando ausente). Na simulação histórica (`tipo_simulacao_final` = 2) é substituído pelo número de anos iniciais do `shist.dat`, pois cada cenário histórico é um ano de partida. |
| `simulation.scenario_source.seed` | — *(constante; condicional: somente quando a simulação está habilitada)* | Sempre 42; o NEWAVE não expõe semente equivalente. |
| `simulation.scenario_source.inflow.scheme` | `dger.dat` › `tipo_simulacao_final`, `dger.dat` › `considera_reamostragem_cenarios` *(condicional: somente quando a simulação está habilitada)* | `tipo_simulacao_final` = 2 → `historical`; senão `considera_reamostragem_cenarios` = 1 → `out_of_sample`, caso contrário `in_sample`. |
| `simulation.scenario_source.historical_years` | `shist.dat` › `varredura`, `shist.dat` › `anos_inicio_simulacoes`, `shist.dat` › `ano_inicio_varredura`, `dger.dat` › `ano_inicial_historico`, `dger.dat` › `ano_inicio_estudo` *(condicional: somente quando `tipo_simulacao_final` = 2 (simulação histórica))* | `varredura` = 0 → lista explícita de `anos_inicio_simulacoes`; `varredura` = 1 → intervalo de `ano_inicio_varredura` até `ano_inicio_estudo` − (anos de estudo + pós-estudo). Sem `shist.dat`: intervalo de `ano_inicial_historico` + 1 até `ano_inicio_estudo` − 1. |
| `training.selection.method` | — *(constante)* | Sempre `sampled`. |
| `training.selection.forward_passes` | `dger.dat` › `num_forwards` | Valor direto (1 quando ausente). |
| `training.stopping_rules[].type` | — *(constante)* | Sempre `iteration_limit`, única regra de parada emitida; o NEWAVE não tem critério de gap equivalente exposto aqui. |
| `training.stopping_rules[].limit` | `dger.dat` › `num_max_iteracoes` | Valor direto (200 quando ausente). |
| `training.cut_selection.row_activity_tolerance` | — *(constante)* | Sempre 1e-6. |
| `training.cut_selection.selection.method` | — *(constante; condicional: somente quando `selecao_de_cortes_forward` = 1 ou `selecao_de_cortes_backward` = 1)* | Sempre `lml1` (seleção Level-1 de memória limitada). O Novomodelo tem um único interruptor para os dois passos; a união dos flags do NEWAVE o liga. |
| `training.cut_selection.selection.check_frequency` | — *(constante; condicional: somente quando `selecao_de_cortes_forward` = 1 ou `selecao_de_cortes_backward` = 1)* | Sempre 1. A janela de memória do NEWAVE não tem equivalente no `lml1` baseado em valor e é descartada. |
| `training.parallelism.backward_scheduler.method` | — *(constante)* | Sempre `by_node`: cada unidade de trabalho do backward é um par (ponto de tentativa, bloco de aberturas). |
| `training.parallelism.backward_scheduler.block_size` | `dger.dat` › `num_aberturas` *(derivado)* | `ceil(num_aberturas / 2)`; coincide com o padrão do Novomodelo, mas é fixado a partir do deck. |
| `training.scenario_source.seed` | — *(constante; condicional: somente quando `tipo_execucao` = 1 e (`considera_reamostragem_cenarios` = 1 ou modo determinístico))* | Sempre 42. |
| `training.scenario_source.inflow.scheme` | `dger.dat` › `considera_reamostragem_cenarios`, `dger.dat` › `num_forwards`, `dger.dat` › `num_aberturas`, `dger.dat` › `tipo_simulacao_final`, `shist.dat` › `varredura`, `shist.dat` › `anos_inicio_simulacoes` *(condicional: somente quando `tipo_execucao` = 1 e (`considera_reamostragem_cenarios` = 1 ou modo determinístico))* | `out_of_sample` quando `considera_reamostragem_cenarios` = 1; `historical` no modo determinístico (treinamento reutiliza o único ano histórico da simulação). |
| `training.scenario_source.historical_years` | `shist.dat` › `varredura`, `shist.dat` › `anos_inicio_simulacoes`, `shist.dat` › `ano_inicio_varredura`, `dger.dat` › `ano_inicial_historico` *(condicional: somente no modo determinístico)* | Mesma resolução de `simulation.scenario_source.historical_years`; no modo determinístico resulta na lista de um único ano de `anos_inicio_simulacoes`. |
| `training.enabled` | `dger.dat` › `tipo_execucao` *(condicional: somente quando `tipo_execucao` = 0 (somente simulação))* | Escrito como `false` apenas nesse caso; omitido (padrão `true`) quando `tipo_execucao` = 1. |

### `stages.json`

**Lê:** `dger.dat`, `patamar.dat`, `cvar.dat` (opcional), `shist.dat` (opcional)  
**Quando:** sempre.  
**Esquema:** [stages.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/stages.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/temporal.py`

Um estágio mensal por mês de estudo e de pós-estudo, com blocos (patamares) de `patamar.dat` e medida de risco de `dger.dat`/`cvar.dat`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `season_definitions.cycle_type` | — *(constante)* | Sempre `monthly`: doze estações calendário. |
| `season_definitions.seasons[].id` | — *(constante)* | 0 a 11 (janeiro = 0). |
| `season_definitions.seasons[].month_start` | — *(constante)* | 1 a 12, o mês calendário de cada estação. |
| `season_definitions.seasons[].label` | — *(constante)* | Nome do mês em inglês (`January` … `December`). |
| `policy_graph.type` | — *(constante)* | Sempre `finite_horizon`. |
| `policy_graph.annual_discount_rate` | `dger.dat` › `taxa_de_desconto` | Taxa anual em percentual dividida por 100 (0.0 quando ausente). |
| `policy_graph.transitions[].source_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Uma transição por par de estágios consecutivos (0→1, 1→2, …) até o último estágio do pós-estudo. |
| `policy_graph.transitions[].target_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | `source_id` + 1. |
| `policy_graph.transitions[].probability` | — *(constante)* | Sempre 1.0: cadeia linear de estágios. |
| `stages[].id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | 0-based, um por mês: o estudo vai de `mes_inicio_estudo` a dezembro de `ano_inicio_estudo` + `num_anos_estudo` − 1, seguido de `num_anos_pos_estudo` anos completos. |
| `stages[].start_date` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` *(derivado)* | Primeiro dia do mês do estágio (ISO). |
| `stages[].end_date` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` *(derivado)* | Primeiro dia do mês seguinte (fim exclusivo). |
| `stages[].season_id` | `dger.dat` › `mes_inicio_estudo` *(derivado)* | Mês calendário do estágio − 1. |
| `stages[].blocks[].id` | `patamar.dat` › `numero_patamares` *(derivado)* | Índice do patamar − 1, para cada um dos `numero_patamares` patamares. |
| `stages[].blocks[].name` | `patamar.dat` › `numero_patamares` *(derivado)* | Nome canônico por quantidade de patamares: 1 → `SINGLE`; 2 → `HEAVY`, `LIGHT`; 3 → `HEAVY`, `MEDIUM`, `LIGHT`; outras → `BLOCK_i`. |
| `stages[].blocks[].hours` | `patamar.dat` › `duracao_mensal_patamares · valor` | Fração de duração do patamar no (ano, mês) do estágio multiplicada pelas horas do mês calendário. Ano ausente usa o último ano do arquivo no mesmo mês; sem registro, fração igual 1/N com aviso. |
| `stages[].num_openings` | `dger.dat` › `num_aberturas` | Valor direto (1 quando ausente), igual em todos os estágios. |
| `stages[].risk_measure` | `dger.dat` › `cvar`, `cvar.dat` › `valores_constantes` | `expectation` quando `cvar` = 0, quando `cvar.dat` está ausente ou no modo determinístico. Com `cvar` = 1 ou 2 e `cvar.dat` presente, vira o objeto `cvar` documentado nos campos `risk_measure.cvar.*`. |
| `stages[].risk_measure.cvar.alpha` | `dger.dat` › `cvar`, `cvar.dat` › `valores_constantes`, `cvar.dat` › `alfa_variavel · valor` *(condicional: somente quando `cvar` = 1 ou 2 em `dger.dat` e `cvar.dat` está presente, fora do modo determinístico)* | `cvar` = 1: primeiro valor de `valores_constantes` / 100 em todos os estágios. `cvar` = 2: `alfa_variavel` do (ano, mês) do estágio / 100; valor 0 ou mês ausente recai no constante. |
| `stages[].risk_measure.cvar.lambda` | `dger.dat` › `cvar`, `cvar.dat` › `valores_constantes`, `cvar.dat` › `lambda_variavel · valor` *(condicional: somente quando `cvar` = 1 ou 2 em `dger.dat` e `cvar.dat` está presente, fora do modo determinístico)* | `cvar` = 1: segundo valor de `valores_constantes` / 100. `cvar` = 2: `lambda_variavel` do (ano, mês) / 100, com o mesmo recuo ao constante. |
| `stages[].state_variables.storage` | — *(constante)* | Sempre `true`: o armazenamento é variável de estado em todo estágio. |
| `stages[].state_variables.inflow_lags` | `dger.dat` › `num_forwards`, `dger.dat` › `num_aberturas`, `dger.dat` › `tipo_simulacao_final`, `shist.dat` › `varredura`, `shist.dat` › `anos_inicio_simulacoes` | `true`, exceto no modo determinístico, em que as defasagens de afluência não acrescentam informação e são desligadas. |
| `stages[].sampling_method` | `dger.dat` › `num_forwards`, `dger.dat` › `num_aberturas`, `dger.dat` › `tipo_simulacao_final`, `shist.dat` › `varredura`, `shist.dat` › `anos_inicio_simulacoes` *(condicional: somente no modo determinístico)* | `historical_residuals`: cada estágio amostra o resíduo diretamente do único cenário histórico. |
| `pre_study_stages[].id` | `dger.dat` › `num_anos_pre_estudo` *(derivado; condicional: somente quando `num_anos_pre_estudo` > 0)* | Ids negativos −N … −1 para os N = 12 × `num_anos_pre_estudo` meses anteriores ao início do estudo. |
| `pre_study_stages[].start_date` | `dger.dat` › `num_anos_pre_estudo`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` *(derivado; condicional: somente quando `num_anos_pre_estudo` > 0)* | Primeiro dia de cada mês pré-estudo, andando para trás a partir do início do estudo. |
| `pre_study_stages[].end_date` | `dger.dat` › `num_anos_pre_estudo`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` *(derivado; condicional: somente quando `num_anos_pre_estudo` > 0)* | Primeiro dia do mês seguinte (fim exclusivo). |
| `pre_study_stages[].season_id` | `dger.dat` › `num_anos_pre_estudo` *(derivado; condicional: somente quando `num_anos_pre_estudo` > 0)* | Mês calendário − 1. |

### `penalties.json`

**Lê:** `sistema.dat`, `penalid.dat` (opcional), `hidr.dat`, `confhd.dat`, `modif.dat` (opcional), `exph.dat` (opcional)  
**Quando:** sempre.  
**Esquema:** [penalties.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/penalties.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/network.py`

Penalidades globais: custo de déficit de `sistema.dat`, micropenalidades internas do NEWAVE (manual v30, §3.24) e penalidades de `penalid.dat` convertidas de R$/MWh para o domínio de vazão com as produtibilidades do SIN.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `bus.deficit_segments[].cost` | `sistema.dat` › `custo_deficit · custo` | Custo do primeiro patamar de déficit do submercado de menor código, em R$/MWh, sem conversão. Um único segmento sem profundidade. |
| `bus.deficit_segments[].depth_mw` | — *(sempre nulo)* | Sempre nulo: o segmento global é ilimitado; as profundidades por patamar ficam em `system/buses.json`. |
| `bus.excess_cost` | — *(constante)* | Sempre 0.000355 R$/MWh (pEXC, excesso de energia). Domínio de energia, sem fator de produtibilidade. |
| `line.exchange_cost` | — *(constante)* | Sempre 0.000273 R$/MWh (pINT, intercâmbio). Sem fator de produtibilidade. |
| `non_controllable_source.curtailment_cost` | — *(constante)* | Sempre 0.000344 R$/MWh (pCORTEOL, corte de geração não simulada). Sem fator de produtibilidade. |
| `hydro.spillage_cost` | `hidr.dat` › `cadastro · produtibilidade_especifica`, `hidr.dat` › `cadastro · volume_minimo`, `hidr.dat` › `cadastro · volume_maximo`, `hidr.dat` › `cadastro · canal_fuga_medio` | 0.000300 (pEVERT, coluna NEWAVE individualizado) × PROD_MEDIA_SIN, a média da produtibilidade equivalente volume mínimo→máximo (PRODT) sobre as usinas ativas, zeros incluídos. Resultado em R$/(m³/s·h). |
| `hydro.turbined_cost` | `hidr.dat` › `cadastro · produtibilidade_especifica`, `hidr.dat` › `cadastro · volume_minimo`, `hidr.dat` › `cadastro · volume_maximo`, `hidr.dat` › `cadastro · canal_fuga_medio` | 0.000333 (pTURB) × PROD_MEDIA_SIN. |
| `hydro.diversion_cost` | `hidr.dat` › `cadastro · produtibilidade_especifica`, `hidr.dat` › `cadastro · volume_minimo`, `hidr.dat` › `cadastro · volume_maximo`, `hidr.dat` › `cadastro · canal_fuga_medio` | 0.000300 (pCDESV, volume desviado) × PROD_MEDIA_SIN. |
| `hydro.turbined_violation_below_cost` | `penalid.dat` › `TURBMN · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica` | TURBMN (primeiro valor não nulo de `penalid.dat`; sem registro, 10 × maior custo de déficit) × PROD_MEDIA_SIN. |
| `hydro.outflow_violation_below_cost` | `penalid.dat` › `VAZMIN · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica` | VAZMIN (sem registro, 10 × maior custo de déficit) × PROD_MEDIA_SIN. |
| `hydro.outflow_violation_above_cost` | `penalid.dat` › `TURBMX · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica` | TURBMX (sem registro, 10 × maior custo de déficit) × PROD_MEDIA_SIN. |
| `hydro.generation_violation_below_cost` | `penalid.dat` › `GHMIN · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo` | GHMIN (sem registro, 10 × maior custo de déficit), em R$/MWh sem fator de produtibilidade: a folga é no domínio de energia. |
| `hydro.water_withdrawal_violation_cost` | `penalid.dat` › `DESVIO · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica`, `confhd.dat` › `usinas · codigo_usina_jusante` | DESVIO (sem registro, 10 × maior custo de déficit) × MAX_PRODTACUM_SIN, a maior produtibilidade acumulada de cascata avaliada na altura máxima (volume útil integral), com a topologia de `codigo_usina_jusante`. |
| `hydro.evaporation_violation_cost` | `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica`, `confhd.dat` › `usinas · codigo_usina_jusante` | Sem variável em `penalid.dat`: 10 × maior custo de déficit × MAX_PRODTACUM_SIN (manual p.87, sem a divisão por C_M3S2HM3 porque a folga do Novomodelo é em m³/s). |
| `hydro.inflow_nonnegativity_cost` | `penalid.dat` › `DESVIO · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica` | `water_withdrawal_violation_cost` + 1 R$/(m³/s), para que a folga de afluência nunca seja mais barata que a violação de desvio de água. Sem contrapartida no NEWAVE. |
| `hydro.storage_violation_below_cost` | `penalid.dat` › `VOLMIN · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica`, `confhd.dat` › `usinas · codigo_usina_jusante` | VOLMIN (sem registro, 10 × maior custo de déficit) × MAX_PRODTACUM_SIN × 1e6/3600 (≈ 277.78), convertendo para R$/hm³ pela equivalência volumétrica. Slot ainda não usado pelo LP do Novomodelo. |
| `hydro.filling_target_violation_cost` | `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica`, `confhd.dat` › `usinas · codigo_usina_jusante` | Sem variável em `penalid.dat`: 0.9 × maior custo de déficit × MAX_PRODTACUM_SIN × 1e6/3600, em R$/hm³. Slot ainda não usado pelo LP do Novomodelo. |

### `initial_conditions.json`

**Lê:** `confhd.dat`, `hidr.dat`, `modif.dat` (opcional), `exph.dat` (opcional), `dger.dat`, `adterm.dat` (opcional), `patamar.dat`, `term.dat`, `conft.dat`, `vazpast.dat` (opcional)  
**Quando:** sempre.  
**Esquema:** [initial_conditions.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/initial_conditions.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/initial_conditions.py`

Armazenamento inicial por usina ativa; opcionalmente o volume morto já enchido das usinas em enchimento, os despachos antecipados de `adterm.dat` e a tendência hidrológica de `vazpast.dat`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `storage[].hydro_id` | `confhd.dat` › `usinas · codigo_usina` *(derivado)* | Id Novomodelo 0-based da usina ativa (`usina_existente` = EX ou EE, não fictícia), atribuído em ordem crescente de `codigo_usina`. Ordenado pelo id. |
| `storage[].value_hm3` | `confhd.dat` › `usinas · volume_inicial_percentual`, `hidr.dat` › `cadastro · volume_minimo`, `hidr.dat` › `cadastro · volume_maximo`, `hidr.dat` › `cadastro · tipo_regulacao`, `hidr.dat` › `cadastro · volume_referencia`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `VOLMAX · volume` | Percentual (limitado a [0, 100]) do volume útil `volume_maximo` − `volume_minimo`, somado ao mínimo, após VOLMIN/VOLMAX permanentes de `modif.dat`. Usinas `tipo_regulacao` D e S ancoram em `volume_referencia`, como o colapso de fio d'água dos limites; sem `volume_referencia`, seguem o percentual. |
| `filling_storage` | `confhd.dat` › `usinas · usina_existente`, `exph.dat` › `expansoes · data_inicio_enchimento` | Lista vazia quando nenhuma usina NE de `confhd.dat` tem linha de enchimento de volume morto em `exph.dat`; caso contrário, ver os campos `filling_storage[].*`. |
| `filling_storage[].hydro_id` | `confhd.dat` › `usinas · codigo_usina`, `confhd.dat` › `usinas · usina_existente`, `exph.dat` › `expansoes · data_inicio_enchimento` *(derivado; condicional: somente para usinas NE com linha de enchimento (`data_inicio_enchimento` não nula) em `exph.dat`)* | Id Novomodelo da usina em enchimento; ela nunca aparece também em `storage`. |
| `filling_storage[].value_hm3` | `exph.dat` › `expansoes · volume_morto`, `hidr.dat` › `cadastro · volume_minimo` *(condicional: somente para usinas NE com linha de enchimento em `exph.dat`)* | `volume_morto` (percentual, limitado a [0, 100]) / 100 × `volume_minimo`: a fração do volume morto já represada no início do enchimento. |
| `past_anticipated_commitments[].thermal_id` | `dger.dat` › `despacho_antecipado_gnl`, `adterm.dat` › `despachos · codigo_usina` *(derivado; condicional: somente quando `despacho_antecipado_gnl` ≠ 0 e `adterm.dat` tem despachos)* | Id Novomodelo da térmica de `conft.dat`; códigos ausentes do mapa são ignorados com aviso. |
| `past_anticipated_commitments[].start_date` | `adterm.dat` › `despachos · lag`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` *(derivado; condicional: somente quando `despacho_antecipado_gnl` ≠ 0 e `adterm.dat` tem despachos)* | Primeiro dia do mês de entrega: `lag` = 1 é o primeiro mês do estudo, `lag` = 2 o segundo, e assim por diante; uma janela por `lag` de 1 ao máximo da usina. |
| `past_anticipated_commitments[].end_date` | `adterm.dat` › `despachos · lag`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` *(derivado; condicional: somente quando `despacho_antecipado_gnl` ≠ 0 e `adterm.dat` tem despachos)* | Primeiro dia do mês seguinte à entrega (fim exclusivo). |
| `past_anticipated_commitments[].value_mw` | `adterm.dat` › `despachos · valor`, `adterm.dat` › `despachos · patamar`, `patamar.dat` › `duracao_mensal_patamares · valor`, `term.dat` › `usinas · potencia_instalada`, `term.dat` › `usinas · fator_capacidade_maximo`, `term.dat` › `usinas · geracao_minima` *(condicional: somente quando `despacho_antecipado_gnl` ≠ 0 e `adterm.dat` tem despachos)* | Média dos MW por patamar ponderada pela fração de duração do patamar no mês de entrega (preserva o MWh comprometido). Limitado ao par estático [`min_mw`, `max_mw`] da térmica — o envelope dos limites por estágio —, com aviso. |
| `recent_observations[].hydro_id` | `vazpast.dat` › `tendencia · codigo_usina`, `confhd.dat` › `usinas · posto` *(derivado; condicional: somente quando `vazpast.dat` está presente com dados de tendência)* | `codigo_usina` de `vazpast.dat` é o posto; mapeado ao id Novomodelo pela coluna `posto` de `confhd.dat`. Ordenado pelo id. |
| `recent_observations[].start_date` | `vazpast.dat` › `tendencia · mes`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` *(derivado; condicional: somente quando `vazpast.dat` está presente com dados de tendência)* | Primeiro dia de cada um dos 12 meses calendário anteriores ao início do estudo, do mais antigo ao mais recente; mês sem valor na tendência é omitido. |
| `recent_observations[].end_date` | `vazpast.dat` › `tendencia · mes`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` *(derivado; condicional: somente quando `vazpast.dat` está presente com dados de tendência)* | Primeiro dia do mês seguinte (fim exclusivo). |
| `recent_observations[].value_m3s` | `vazpast.dat` › `tendencia · valor`, `confhd.dat` › `usinas · posto`, `confhd.dat` › `usinas · codigo_usina_jusante` *(condicional: somente quando `vazpast.dat` está presente com dados de tendência)* | Vazão natural do posto convertida em incremental subtraindo, mês a mês, a vazão natural dos postos imediatamente a montante na cascata de `codigo_usina_jusante`. |

### `system/buses.json`

**Lê:** `sistema.dat`, `dger.dat`, `ree.dat`  
**Quando:** sempre.  
**Esquema:** [buses.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/buses.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/network.py`

Uma barra por submercado de `sistema.dat` (fictícios incluídos), com os patamares de déficit.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `buses[].id` | `sistema.dat` › `custo_deficit · codigo_submercado`, `ree.dat` › `rees · submercado` *(derivado)* | Id 0-based em ordem crescente de `codigo_submercado`; códigos que só aparecem em `ree.dat` são acrescentados ao mapa. Lista ordenada pelo id. |
| `buses[].name` | `sistema.dat` › `custo_deficit · nome_submercado` | Nome do submercado, sem espaços nas pontas. |
| `buses[].operational_start_date` | `dger.dat` › `ano_inicial_historico` *(derivado)* | 1º de janeiro de `ano_inicial_historico` (1931 quando ausente): submercados não têm data de entrada; o Novomodelo usa a data só como chave de ordenação. |
| `buses[].deficit_segments[].cost` | `sistema.dat` › `custo_deficit · custo`, `sistema.dat` › `custo_deficit · patamar_deficit` | Um segmento por `patamar_deficit` com custo positivo, em ordem de patamar, R$/MWh direto. Submercado sem custo positivo (fictício) recebe um único segmento com o primeiro custo positivo do arquivo. |
| `buses[].deficit_segments[].depth_mw` | `sistema.dat` › `custo_deficit · corte` | Profundidade `corte` do patamar; o último segmento de cada barra é sempre nulo (ilimitado). |

### `system/lines.json`

**Lê:** `sistema.dat`, `dger.dat`, `ree.dat`  
**Quando:** sempre.  
**Esquema:** [lines.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/lines.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/network.py`

Uma linha por par não ordenado de submercados presente em `limites_intercambio`, com as capacidades do primeiro mês do estudo.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `lines[].id` | `sistema.dat` › `limites_intercambio · submercado_de`, `sistema.dat` › `limites_intercambio · submercado_para` *(derivado)* | Pares canônicos (menor código, maior código) coletados de todas as datas do arquivo, ordenados e numerados de 0; o mesmo mapa é usado em `constraints/line_bounds.parquet`. |
| `lines[].name` | `sistema.dat` › `limites_intercambio · submercado_de`, `sistema.dat` › `limites_intercambio · submercado_para` *(derivado)* | `{código origem}_{código destino}` com os códigos numéricos do par canônico. |
| `lines[].operational_start_date` | `dger.dat` › `ano_inicial_historico` *(derivado)* | 1º de janeiro de `ano_inicial_historico`, como nas barras. |
| `lines[].source_bus_id` | `sistema.dat` › `limites_intercambio · submercado_de` *(derivado)* | Id Novomodelo do menor código do par. |
| `lines[].target_bus_id` | `sistema.dat` › `limites_intercambio · submercado_para` *(derivado)* | Id Novomodelo do maior código do par. |
| `lines[].capacity.direct_mw` | `sistema.dat` › `limites_intercambio · valor`, `sistema.dat` › `limites_intercambio · sentido`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` | Limite (MW) no sentido menor→maior código no mês de início do estudo (`sentido` 0 é de→para, 1 é para→de). Sem linha nessa data usa a primeira data com valor; par sem linha recebe 0. |
| `lines[].capacity.reverse_mw` | `sistema.dat` › `limites_intercambio · valor`, `sistema.dat` › `limites_intercambio · sentido`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo` | Limite (MW) no sentido maior→menor código, mesma regra de data. |
| `lines[].exchange_cost` | `sistema.dat` › `custo_deficit · ficticio` | Presente só em linhas que tocam um submercado fictício: 0.000273 × 0.5, meia pINT, para que uma rota real→fictício→real custe o mesmo que o intercâmbio direto. Demais linhas usam o valor global de `penalties.json`. |

### `system/thermals.json`

**Lê:** `conft.dat`, `clast.dat`, `term.dat`, `expt.dat` (opcional), `manutt.dat` (opcional), `dger.dat`, `adterm.dat` (opcional), `patamar.dat`, `sistema.dat`, `ree.dat`  
**Quando:** sempre.  
**Esquema:** [thermals.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/thermals.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/thermal.py`

Uma entrada por usina térmica de `conft.dat`, ordenada pelo id Novomodelo, com o par estático de geração — o envelope dos limites por estágio — e o custo do primeiro ano de `clast.dat`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `thermals[].id` | `conft.dat` › `usinas · codigo_usina` *(derivado)* | Id 0-based em ordem crescente de `codigo_usina`. |
| `thermals[].name` | `conft.dat` › `usinas · nome_usina` | Nome sem espaços nas pontas. |
| `thermals[].operational_start_date` | `dger.dat` › `ano_inicial_historico` *(derivado)* | 1º de janeiro de `ano_inicial_historico`: o NEWAVE não tem data de entrada por térmica. |
| `thermals[].bus_id` | `conft.dat` › `usinas · submercado` *(derivado)* | Id Novomodelo do submercado da usina. |
| `thermals[].cost_per_mwh` | `clast.dat` › `usinas · valor (indice_ano_estudo = 1)` | Custo do primeiro ano de estudo; 0.0 para usina sem linha em `clast.dat`. Anos com custo diferente e as modificações datadas vão para `constraints/thermal_bounds.parquet`. |
| `thermals[].generation.min_mw` | `term.dat` › `usinas · geracao_minima (mes 1 a 12)`, `term.dat` › `usinas · geracao_minima (mes = 13)`, `expt.dat` › `expansoes · tipo = GTMIN · modificacao`, `expt.dat` › `expansoes · data_inicio`, `expt.dat` › `expansoes · data_fim`, `conft.dat` › `usinas · usina_existente`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo` | Menor mínimo do horizonte: o envelope de `min_generation_mw` de `constraints/thermal_bounds.parquet`, que detalha as fontes. Ler só a linha de cadastro do `term.dat` publicaria um par que contradiz a tabela por estágio, porque as janelas de `expt.dat`, o valor dos demais anos de `term.dat`, as manutenções e a regra de IP dos anos de manutenção entram apenas por ela. Usina que nenhuma fonte de capacidade descreve recebe 0. |
| `thermals[].generation.max_mw` | `term.dat` › `usinas · potencia_instalada`, `term.dat` › `usinas · fator_capacidade_maximo`, `term.dat` › `usinas · teif`, `term.dat` › `usinas · indisponibilidade_programada`, `expt.dat` › `expansoes · tipo = POTEF · modificacao`, `expt.dat` › `expansoes · tipo = FCMAX · modificacao`, `expt.dat` › `expansoes · tipo = TEIFT · modificacao`, `expt.dat` › `expansoes · tipo = IPTER · modificacao`, `manutt.dat` › `manutencoes · potencia`, `conft.dat` › `usinas · usina_existente`, `dger.dat` › `num_anos_manutencao_utes` | Maior máximo do horizonte: o envelope de `max_generation_mw` de `constraints/thermal_bounds.parquet`, que detalha as fontes — logo já com TEIF, IP e manutenção. Quando a geração mínima de cadastro supera o produto `potencia_instalada` × `fator_capacidade_maximo` / 100, o teto por estágio é elevado ao mínimo inflexível e o envelope herda essa ordenação, em vez de publicar o intervalo vazio que o Novomodelo rejeita (`max_mw` tem de ser ≥ `min_mw`). Usina que nenhuma fonte de capacidade descreve recebe 0. |
| `thermals[].anticipated_config` | `dger.dat` › `despacho_antecipado_gnl`, `adterm.dat` › `despachos · codigo_usina` | Nulo quando `despacho_antecipado_gnl` = 0, quando `adterm.dat` está ausente ou quando a usina não tem despacho nele; caso contrário o objeto `anticipated_config.lead_stages`. |
| `thermals[].anticipated_config.lead_stages` | `adterm.dat` › `despachos · lag` *(condicional: somente para térmicas com despacho em `adterm.dat` quando `despacho_antecipado_gnl` ≠ 0)* | Maior `lag` da usina, limitado ao número total de estágios (aviso quando truncado). |
| `thermals[].entry_stage_id` | — *(sempre nulo)* | Sempre nulo: térmicas do NEWAVE não entram dentro do horizonte; a expansão de `expt.dat` é expressa por limites por estágio. |
| `thermals[].exit_stage_id` | — *(sempre nulo)* | Sempre nulo: térmicas do NEWAVE não saem dentro do horizonte. |

### `system/hydros.json`

**Lê:** `hidr.dat`, `confhd.dat`, `ree.dat`, `sistema.dat`, `modif.dat` (opcional), `exph.dat` (opcional), `dger.dat`, `volref_saz.dat` (opcional)  
**Quando:** sempre.  
**Esquema:** [hydros.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/hydros.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/hydro/entity.py`

Uma entrada por usina hidrelétrica ativa de `confhd.dat` (em operação — `EX` ou `EE` — e não fictícias, mais usinas NE com enchimento de volume morto em `exph.dat`), com o cadastro de `hidr.dat` já corrigido pelos registros permanentes de `modif.dat`, ordenada pelo id Novomodelo.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `hydros[].id` | `confhd.dat` › `codigo_usina`, `confhd.dat` › `usina_existente`, `confhd.dat` › `posto`, `hidr.dat` › `produtibilidade_especifica`, `exph.dat` › `data_inicio_enchimento` *(derivado)* | Id 0-based denso atribuído em ordem crescente de `codigo_usina` entre as usinas ativas: `usina_existente` em `EX`/`EE` menos as fictícias (produtibilidade específica zero compartilhando o `posto` de uma usina geradora), mais as usinas NE com registro de enchimento em `exph.dat`. |
| `hydros[].name` | `confhd.dat` › `nome_usina` | Nome da usina em `confhd.dat`, sem espaços nas extremidades. |
| `hydros[].operational_start_date` | `dger.dat` › `ano_inicial_historico`, `exph.dat` › `data_inicio_enchimento`, `exph.dat` › `duracao_enchimento` *(derivado)* | Usinas existentes: 1º de janeiro de `ano_inicial_historico` (1931 se ausente). Usina NE em enchimento: primeiro dia do mês em que o enchimento termina (`data_inicio_enchimento` mais `duracao_enchimento` meses). No Novomodelo é chave de ordenação, não porta de entrada em operação. |
| `hydros[].downstream_id` | `confhd.dat` › `codigo_usina_jusante`, `confhd.dat` › `usina_existente`, `confhd.dat` › `posto`, `hidr.dat` › `produtibilidade_especifica`, `exph.dat` › `data_inicio_enchimento` *(derivado)* | Id Novomodelo da próxima usina real a jusante. A cadeia `codigo_usina_jusante` é percorrida de forma transparente através de usinas fictícias e NE/NC; quando o jusante é 0 mas uma usina fictícia compartilha o `posto` da usina, ela é tomada como elo implícito da cascata. Nulo quando a cascata termina no mar. |
| `hydros[].reservoir.min_storage_hm3` | `hidr.dat` › `volume_minimo`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia`, `modif.dat` › `VOLMIN · volume, unidade` | `volume_minimo` do cadastro (sobrescrito pelo registro permanente `VOLMIN` de `modif.dat`, em hm³ com `unidade` = h ou, com `%`, como percentual do volume útil do cadastro). Usinas `tipo_regulacao` D e S recebem `volume_referencia`, colapsando a faixa útil a um ponto (faixa completa se ausente). |
| `hydros[].reservoir.max_storage_hm3` | `hidr.dat` › `volume_maximo`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_referencia`, `modif.dat` › `VOLMAX · volume, unidade`, `modif.dat` › `VOLMIN · volume` | `volume_maximo` do cadastro (sobrescrito por `VOLMAX` de `modif.dat`, em hm³ com `unidade` = h ou, com `%`, como percentual do volume útil do cadastro). Usinas `D` e `S` recebem `volume_referencia` (faixa completa se ausente), pois o NEWAVE não acumula água entre estágios nessas usinas. |
| `hydros[].outflow.min_outflow_m3s` | `dger.dat` › `desconsidera_vazao_minima`, `hidr.dat` › `vazao_minima_historica`, `modif.dat` › `VAZMIN · vazao` | `vazao_minima_historica` (sobrescrita pelo registro permanente `VAZMIN` de `modif.dat`), ou 0 quando não positiva. Os registros temporais `VAZMINT` vão para `constraints/hydro_bounds.parquet`. |
| `hydros[].outflow.max_outflow_m3s` | — *(sempre nulo)* | Sempre nulo: o NEWAVE não impõe vazão defluente máxima estática; o registro `VAZMAXT` de `modif.dat` não é convertido. |
| `hydros[].generation.model` | `dger.dat` › `funcao_producao_uhe`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4`, `hidr.dat` › `produtibilidade_especifica`, `exph.dat` › `data_entrada_operacao` | `fpha` quando `funcao_producao_uhe = 0` e a usina tem polinômio cota-volume não nulo, produtibilidade específica positiva e engolimento e potência nominais (Σ máquinas × nominal por conjunto, após `MODIF NUMCNJ`/`NUMMAQ`, na configuração declarada — a final, para usina `EE` com entrada de máquinas em `exph.dat`) positivos; `constant_productivity` nos demais casos. Uma usina excluída só pela capacidade nula é listada em diagnóstico informativo. |
| `hydros[].generation.min_turbined_m3s` | — *(constante)* | Sempre 0.0: o NEWAVE não impõe turbinamento mínimo estático; `TURBMINT` por estágio vai para `constraints/hydro_bounds.parquet`. |
| `hydros[].generation.max_turbined_m3s` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1 … maquinas_conjunto_5`, `hidr.dat` › `vazao_nominal_conjunto_1 … vazao_nominal_conjunto_5`, `hidr.dat` › `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5`, `modif.dat` › `POTEFE · potencia, conjunto`, `hidr.dat` › `queda_nominal_conjunto_1 … queda_nominal_conjunto_5`, `hidr.dat` › `teif`, `hidr.dat` › `ip`, `hidr.dat` › `tipo_turbina`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `volume_referencia`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `tipo_perda`, `hidr.dat` › `perdas`, `modif.dat` › `NUMCNJ · numero`, `modif.dat` › `NUMMAQ · numero_maquinas`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `VOLMAX · volume`, `modif.dat` › `CFUGA · nivel`, `modif.dat` › `CMONT · nivel`, `volref_saz.dat` › `valor`, `dger.dat` › `sazonaliza_cfuga_cmont`, `confhd.dat` › `usina_existente`, `exph.dat` › `data_entrada_operacao`, `exph.dat` › `conjunto_maquina_entrada` | Engolimento máximo na queda de operação: soma por conjunto de `n · vazao_nominal · (h_op / queda_nominal)^k` (k = 0,5 para Francis/Pelton, 0,2 para Kaplan), limitada por `potência instalada / (ρ_esp · h_op)` e multiplicada pela disponibilidade `(1 − teif/100) · (1 − ip/100)`. `h_op` é a queda líquida média entre `volume_minimo` e 65% do volume útil (regulação `M`) ou na cota de `volume_referencia` (`D`/`S`), descontados canal de fuga e perdas. Usina `EE` com entrada de máquinas em `exph.dat` declara a configuração final, com todas as máquinas em operação; a rampa até ela vai para `hydro_bounds.parquet`. Para usina com `CFUGA`/`CMONT` ou linha em `volref_saz.dat`, o valor é elevado ao máximo dos limites por estágio emitidos em `hydro_bounds.parquet`, pois o Novomodelo não admite limite por estágio acima do declarado. |
| `hydros[].generation.min_generation_mw` | — *(constante)* | Sempre 0.0: a geração mínima por estágio de `ghmin.dat` é emitida em `constraints/hydro_bounds.parquet`. |
| `hydros[].generation.max_generation_mw` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1 … maquinas_conjunto_5`, `hidr.dat` › `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5`, `modif.dat` › `POTEFE · potencia, conjunto`, `modif.dat` › `NUMCNJ · numero`, `modif.dat` › `NUMMAQ · numero_maquinas`, `confhd.dat` › `usina_existente`, `exph.dat` › `data_entrada_operacao`, `exph.dat` › `conjunto_maquina_entrada` | Potência instalada nominal: soma por conjunto de `maquinas_conjunto · potencia_nominal_conjunto` (potência do conjunto sobrescrita pelo registro permanente `POTEFE` de `modif.dat`), sem desconto de `teif`/`ip` (o teto GHmax da FPHA do NEWAVE). Usina `EE` com entrada de máquinas em `exph.dat` declara a configuração final: a de `modif.dat` no início do estudo mais as máquinas que entram, até o total de `hidr.dat`. |
| `hydros[].unit_groups[].id` | — *(constante)* | Sempre 0: cada usina recebe um único grupo de unidades espelho. |
| `hydros[].unit_groups[].name` | `confhd.dat` › `nome_usina` | Igual ao `name` da usina. |
| `hydros[].unit_groups[].bus_id` | `confhd.dat` › `ree`, `ree.dat` › `codigo`, `ree.dat` › `submercado`, `sistema.dat` › `custo_deficit · codigo_submercado` *(derivado)* | Id Novomodelo do submercado ao qual pertence o REE da usina (`ree` em `confhd.dat` → `submercado` em `ree.dat`). Os ids de barra seguem a ordem crescente dos códigos de submercado de `sistema.dat` e `ree.dat`. |
| `hydros[].unit_groups[].min_generation_mw` | — *(constante)* | Sempre 0.0: espelha `generation.min_generation_mw`. |
| `hydros[].unit_groups[].max_generation_mw` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1 … maquinas_conjunto_5`, `hidr.dat` › `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5`, `modif.dat` › `POTEFE · potencia, conjunto`, `modif.dat` › `NUMCNJ · numero`, `modif.dat` › `NUMMAQ · numero_maquinas` | Espelha `generation.max_generation_mw` (grupo único, soma dos grupos igual ao envelope da usina). |
| `hydros[].unit_groups[].min_turbined_m3s` | — *(constante)* | Sempre 0.0: espelha `generation.min_turbined_m3s`. |
| `hydros[].unit_groups[].max_turbined_m3s` | `hidr.dat` › `numero_conjuntos_maquinas`, `hidr.dat` › `maquinas_conjunto_1 … maquinas_conjunto_5`, `hidr.dat` › `vazao_nominal_conjunto_1 … vazao_nominal_conjunto_5`, `hidr.dat` › `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5`, `modif.dat` › `POTEFE · potencia, conjunto`, `hidr.dat` › `queda_nominal_conjunto_1 … queda_nominal_conjunto_5`, `hidr.dat` › `teif`, `hidr.dat` › `ip`, `hidr.dat` › `tipo_turbina`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `produtibilidade_especifica`, `modif.dat` › `CFUGA · nivel`, `modif.dat` › `CMONT · nivel`, `volref_saz.dat` › `valor` | Espelha `generation.max_turbined_m3s`. |
| `hydros[].specific_productivity_mw_per_m3s_per_m` | `hidr.dat` › `produtibilidade_especifica` | `produtibilidade_especifica` do cadastro, em MW/((m³/s)·m); nulo quando zero ou ausente. |
| `hydros[].evaporation` | `hidr.dat` › `evaporacao_JAN … evaporacao_DEZ` | Nulo quando os doze coeficientes `evaporacao_*` do cadastro são zero; caso contrário, o bloco com `coefficients_mm` (e `reference_volumes_hm3` quando há referência sazonal). |
| `hydros[].evaporation.coefficients_mm[]` | `hidr.dat` › `evaporacao_JAN … evaporacao_DEZ` | Os doze coeficientes mensais de evaporação (mm), de janeiro a dezembro, tal como no cadastro. |
| `hydros[].evaporation.reference_volumes_hm3[]` | `volref_saz.dat` › `mes`, `volref_saz.dat` › `valor`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo` *(condicional: somente quando a usina tem evaporação e uma linha não nula em `volref_saz.dat`)* | Doze volumes absolutos (hm³), um por mês civil: `volume_minimo + valor` (volume útil sazonal de `volref_saz.dat`; 0 para mês ausente, isto é, operar em `volume_minimo`), limitado ao intervalo [`min_storage_hm3`, `max_storage_hm3`]. |
| `hydros[].tailrace.type` | — *(constante)* | Sempre `polynomial`. O bloco `tailrace` só existe quando `canal_fuga_medio` é positivo; caso contrário é nulo. |
| `hydros[].tailrace.coefficients[]` | `hidr.dat` › `canal_fuga_medio` | Polinômio de grau zero: o único coeficiente é `canal_fuga_medio` (m), o nível médio do canal de fuga que o Novomodelo subtrai da cota de montante ao derivar ρ_eq. |
| `hydros[].diversion` | — *(sempre nulo)* | Sempre nulo: os desvios de água de `dsvagua.dat` são convertidos como retirada por estágio (`water_withdrawal_m3s` em `constraints/hydro_bounds.parquet`), não como canal de desvio. |
| `hydros[].filling` | `exph.dat` › `data_inicio_enchimento`, `exph.dat` › `duracao_enchimento` | Nulo para usinas existentes e para usina NE cujo `duracao_enchimento` é zero; usina NE admitida com enchimento de volume morto recebe o bloco com `start_stage_id` e `filling_min_rate_m3s`. |
| `hydros[].filling.start_stage_id` | `exph.dat` › `data_inicio_enchimento`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo` *(derivado; condicional: somente para usina NE com registro de enchimento em `exph.dat` e `duracao_enchimento` maior que zero)* | Estágio 0-based do mês de `data_inicio_enchimento`, limitado a 0 quando anterior ao início do estudo. |
| `hydros[].filling.filling_min_rate_m3s` | `exph.dat` › `volume_morto`, `exph.dat` › `data_inicio_enchimento`, `exph.dat` › `duracao_enchimento`, `hidr.dat` › `volume_minimo`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo` *(condicional: somente para usina NE com registro de enchimento em `exph.dat` e `duracao_enchimento` maior que zero)* | Taxa constante (m³/s) que leva o armazenamento de `volume_morto` % de `min_storage_hm3` até `min_storage_hm3` ao longo da janela de enchimento: volume restante dividido pela soma, nos estágios da janela dentro do horizonte, do fator hm³ por (m³/s) de cada mês civil. |
| `hydros[].efficiency` | `dger.dat` › `funcao_producao_uhe`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4` | Nulo para usinas fora da FPHA (caso linear ou usina sem polinômio cota-volume / produtibilidade específica / capacidade nominal); para usinas `fpha`, o bloco com `type` e `value`. |
| `hydros[].efficiency.type` | — *(constante)* | Sempre `constant`. |
| `hydros[].efficiency.value` | `hidr.dat` › `produtibilidade_especifica` | Rendimento adimensional da turbina `produtibilidade_especifica / 0,00981` (ρ_esp do NEWAVE já embute g/1000), limitado a 1,0 com aviso quando o cadastro implica valor maior. |
| `hydros[].hydraulic_losses` | `hidr.dat` › `tipo_perda`, `hidr.dat` › `perdas` | Nulo quando `tipo_perda` não é 1 nem 2 ou `perdas` não é positivo; caso contrário, o bloco com `type` e `value` ou `value_m`. |
| `hydros[].hydraulic_losses.type` | `hidr.dat` › `tipo_perda` | `factor` para `tipo_perda = 1` (perda percentual da queda bruta); `constant` para `tipo_perda = 2` (perda em metros). |
| `hydros[].hydraulic_losses.value` | `hidr.dat` › `perdas` *(condicional: somente quando `tipo_perda = 1`)* | `perdas / 100`, fração da queda bruta perdida. |
| `hydros[].hydraulic_losses.value_m` | `hidr.dat` › `perdas` | `perdas` em metros, quando `tipo_perda = 2`. |
| `hydros[].penalties` | — *(sempre nulo)* | Sempre nulo: as penalidades de `penalid.dat` são convertidas uma única vez, com a produtibilidade média do SIN, para os valores globais de `penalties.json`, sem sobrescrita por usina. |
| `hydros[].entry_stage_id` | `exph.dat` › `data_inicio_enchimento`, `exph.dat` › `duracao_enchimento`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo` *(derivado)* | Nulo para usinas existentes. Usina NE em enchimento: estágio 0-based em que o enchimento termina (`data_inicio_enchimento` mais `duracao_enchimento` meses, nunca antes de `start_stage_id`), podendo cair além do horizonte. |
| `hydros[].exit_stage_id` | — *(sempre nulo)* | Sempre nulo: usinas hidrelétricas do NEWAVE não saem de operação dentro do horizonte. |

### `system/hydro_production_models.json`

**Lê:** `dger.dat`, `hidr.dat`, `modif.dat` (opcional), `confhd.dat`, `exph.dat` (opcional), `volref_saz.dat` (opcional), `tratamento-fpha.csv` (opcional)  
**Quando:** sempre.  
**Esquema:** [production_models.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/production_models.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/hydro/productivity.py`

Uma entrada por usina ativa selecionando o modelo de produção (FPHA ou produtividade constante) e o volume de referência; nenhuma produtibilidade numérica é escrita aqui, ela vai para `system/hydro_energy_productivity.parquet`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `production_models[].hydro_id` | `confhd.dat` › `codigo_usina` *(derivado)* | Id Novomodelo da usina (mesma numeração de `hydros[].id`). |
| `production_models[].selection_mode` | `dger.dat` › `funcao_producao_uhe`, `volref_saz.dat` › `valor` | `seasonal` quando o caso é FPHA (`funcao_producao_uhe = 0`) e a usina tem linha não nula em `volref_saz.dat` (FPHA ou não, pois o Novomodelo lê o volume de referência da usina de jusante para o remanso); `stage_ranges` nos demais casos. |
| `production_models[].stage_ranges[].start_stage_id` | — *(constante)* | Sempre 0: uma única faixa novomodelo todo o horizonte. |
| `production_models[].stage_ranges[].end_stage_id` | — *(sempre nulo)* | Sempre nulo: a faixa única é aberta até o fim do horizonte. |
| `production_models[].stage_ranges[].model` | `dger.dat` › `funcao_producao_uhe`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4`, `hidr.dat` › `produtibilidade_especifica` | `fpha` para usina elegível à FPHA (mesmo critério de `hydros[].generation.model`); `constant_productivity` caso contrário. |
| `production_models[].stage_ranges[].fpha_config.source` | — *(constante)* | Sempre `computed`: o Novomodelo ajusta a FPHA a partir da geometria, das curvas de jusante e do rendimento. Presente só em faixas `fpha`. |
| `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3` | `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_referencia`, `modif.dat` › `VOLMIN · volume` | Limite inferior da janela de ajuste da FPHA: `volume_minimo` para regulação `M`; `volume_referencia` para `D`/`S` (ajuste em volume único, como o NEWAVE). |
| `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3` | `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `volume_referencia`, `modif.dat` › `VOLMAX · volume` | Limite superior da janela de ajuste: `volume_maximo` para regulação `M`; `volume_referencia` para `D`/`S`. |
| `production_models[].stage_ranges[].reference_volume.percentile` | — *(constante)* | Sempre 0.65 (altura a 65% do volume útil, convenção do NEWAVE e padrão do Novomodelo). Presente só em faixas `fpha` de usina sem linha em `volref_saz.dat`. |
| `production_models[].default_model` | — *(constante; condicional: somente em caso FPHA para usina com linha não nula em `volref_saz.dat`)* | Sempre `constant_productivity`: as doze estações são listadas, logo o padrão nunca é consultado. |
| `production_models[].seasons[].season_id` | `volref_saz.dat` › `mes` *(derivado; condicional: somente em caso FPHA para usina com linha não nula em `volref_saz.dat`)* | `mes − 1` (0 = janeiro), alinhado ao mapa de estações de `stages.json`. |
| `production_models[].seasons[].model` | `dger.dat` › `funcao_producao_uhe`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4`, `hidr.dat` › `produtibilidade_especifica` *(condicional: somente em caso FPHA para usina com linha não nula em `volref_saz.dat`)* | `fpha` ou `constant_productivity`, igual em todas as estações, pelo critério de elegibilidade da usina. |
| `production_models[].seasons[].reference_volume.volume_hm3` | `volref_saz.dat` › `valor`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `VOLMAX · volume` *(condicional: somente em caso FPHA para usina com linha não nula em `volref_saz.dat`)* | Volume de referência absoluto do mês: `volume_minimo + valor` (volume útil sazonal; mês ausente ou zero significa operar em `volume_minimo`), limitado a [`volume_minimo`, `volume_maximo`]. |
| `production_models[].seasons[].fpha_config.source` | — *(constante; condicional: somente em caso FPHA para usina `fpha` com linha não nula em `volref_saz.dat`)* | Sempre `computed`. |
| `production_models[].seasons[].fpha_config.fitting_window.volume_min_hm3` | `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_referencia`, `modif.dat` › `VOLMIN · volume` *(condicional: somente em caso FPHA para usina `fpha` com linha não nula em `volref_saz.dat`)* | Igual a `stage_ranges[].fpha_config.fitting_window.volume_min_hm3`, repetido em cada estação. |
| `production_models[].seasons[].fpha_config.fitting_window.volume_max_hm3` | `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `volume_referencia`, `modif.dat` › `VOLMAX · volume` *(condicional: somente em caso FPHA para usina `fpha` com linha não nula em `volref_saz.dat`)* | Igual a `stage_ranges[].fpha_config.fitting_window.volume_max_hm3`, repetido em cada estação. |
| `fpha_plane_reduction.method` | `tratamento-fpha.csv` › `HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO / HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO` *(condicional: somente em caso FPHA com ao menos uma usina `fpha` e `tratamento-fpha.csv` com linha de método ativa)* | `angle` para a linha `…-ANGULO-PADRAO`, `distance` para `…-DISTANCIA-PADRAO`; linhas iniciadas por `&` são comentários. Com mais de uma linha ativa, a primeira vale e um aviso é emitido. |
| `fpha_plane_reduction.tolerance_deg` | `tratamento-fpha.csv` › `HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO` *(condicional: somente quando o método ativo em `tratamento-fpha.csv` é o de ângulo)* | Tolerância angular (graus) lida após o `;` da linha. |
| `fpha_plane_reduction.tolerance_pct` | `tratamento-fpha.csv` › `HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO` *(condicional: somente quando o método ativo em `tratamento-fpha.csv` é o de distância)* | Tolerância de distância lida após o `;` da linha. |
| `fpha_plane_reduction.n_samples` | — *(constante; condicional: somente quando o método ativo em `tratamento-fpha.csv` é o de distância)* | Sempre 100: o NEWAVE não define o número de amostras da distância quadrática média exigido pelo Novomodelo, e a ponte fornece este padrão. |

### `system/hydro_geometry.parquet`

**Lê:** `hidr.dat`, `modif.dat` (opcional), `confhd.dat`, `exph.dat` (opcional)  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/newave/converters/hydro/geometry.py`

Curva volume-cota-área amostrada por usina ativa: 100 pontos uniformes entre `volume_minimo` e `volume_maximo` (um único ponto quando iguais); usinas com polinômio cota-volume nulo são omitidas com aviso.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `confhd.dat` › `codigo_usina` *(derivado)* | Id Novomodelo da usina, repetido em cada ponto amostrado; a tabela é ordenada por usina e depois por volume. |
| `volume_hm3` | `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `VOLMAX · volume` | Grade de 100 volumes igualmente espaçados em [`volume_minimo`, `volume_maximo`] do cadastro corrigido por `modif.dat`; um único ponto em `volume_minimo` quando a faixa é nula. Não aplica o colapso das usinas `D`/`S` feito em `hydros.json`. |
| `height_m` | `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4` | Cota de montante (m): polinômio `a0 + a1·V + a2·V² + a3·V³ + a4·V⁴` (coeficientes sobrescritos pelo registro permanente `VOLCOTA` de `modif.dat`) avaliado em cada volume, com valores negativos levados a 0. |
| `area_km2` | `hidr.dat` › `a0_cota_area … a4_cota_area`, `modif.dat` › `COTAREA · coeficientes a0 a a4` | Área do espelho d'água (km²): polinômio cota-área (coeficientes sobrescritos pelo registro permanente `COTAREA` de `modif.dat`) avaliado na cota calculada, com valores negativos levados a 0. |

### `system/hydro_energy_productivity.parquet`

**Lê:** `hidr.dat`, `modif.dat` (opcional), `confhd.dat`, `exph.dat` (opcional), `volref_saz.dat` (opcional), `dger.dat`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/newave/converters/hydro/productivity.py`

Produtibilidade equivalente ρ_eq (MW por m³/s) por usina ativa, fonte única do coeficiente `geração = ρ_eq · Q` (inclusive para usinas FPHA): uma linha padrão com `stage_id` nulo por usina sem variação temporal, ou uma linha por estágio para usina com `CFUGA`/`CMONT` em `modif.dat` ou linha em `volref_saz.dat`.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `confhd.dat` › `codigo_usina` *(derivado)* | Id Novomodelo da usina. |
| `stage_id` | `modif.dat` › `CFUGA · data_inicio`, `modif.dat` › `CMONT · data_inicio`, `volref_saz.dat` › `valor`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Nulo (valor padrão para todo o horizonte) quando a usina não tem `CFUGA`/`CMONT` nem linha não nula em `volref_saz.dat`; caso contrário, um estágio 0-based por mês do estudo e do pós-estudo. |
| `equivalent_productivity_mw_per_m3s` | `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `tipo_perda`, `hidr.dat` › `perdas`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `volume_referencia`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `VOLMAX · volume`, `modif.dat` › `CFUGA · nivel`, `modif.dat` › `CMONT · nivel`, `volref_saz.dat` › `valor`, `dger.dat` › `sazonaliza_cfuga_cmont`, `dger.dat` › `mes_inicio_estudo`, `confhd.dat` › `codigo_usina_jusante`, `confhd.dat` › `posto` | `ρ_esp · (cota(V_ref) − canal de fuga − perdas)`, com V_ref = `volume_minimo` + 65% do volume útil para regulação `M` (altura 65 do NEWAVE) e `volume_referencia` para `D`/`S`. Por estágio, `CFUGA` substitui o canal de fuga e `CMONT` fixa a cota de montante (função degrau a partir de `data_inicio`, repetida sazonalmente após o último registro se `sazonaliza_cfuga_cmont = 1`), e a linha de `volref_saz.dat` fixa V_ref = `volume_minimo + valor` do mês civil. Soma-se a ρ_eq das usinas fictícias atravessadas na cascata (zero nos decks reais). |
| `reference_outflow_m3s` | — *(sempre nulo)* | Sempre nulo: o volume de referência é declarado em `system/hydro_production_models.json`, não por vazão de referência. |
| `specific_productivity_mw_per_m3s_per_m` | — *(sempre nulo)* | Sempre nulo aqui: a produtibilidade específica é declarada em `hydros[].specific_productivity_mw_per_m3s_per_m`. |

### `system/tailrace_curves.parquet`

**Lê:** `polinjus.csv` (opcional), `confhd.dat`, `hidr.dat`, `exph.dat` (opcional)  
**Quando:** somente quando o caso traz `polinjus.csv` e ao menos um segmento pertence a uma usina convertida  
**Código:** `src/novomodelo_bridge/newave/converters/tailrace.py`

Famílias de curvas de nível de jusante (polinômios por partes na vazão defluente total) de `polinjus.csv`, uma linha por segmento, ordenadas por (`hydro_id`, `family_id`, `segment_id`); os índices 1-based do NEWAVE são mantidos e segmentos de usinas fora do mapa de ids (fictícias) são descartados.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · codigo_usina` *(derivado)* | Id Novomodelo da usina correspondente ao `codigo_usina` do segmento. |
| `family_id` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · indice_familia` | `indice_familia` tal como no arquivo (1-based); o Novomodelo o trata como chave opaca de agrupamento por usina. |
| `downstream_reference_level_m` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE · nivel_montante_referencia` | Nível de referência da usina de jusante que identifica a família (`nivel_montante_referencia` do registro de família, associado ao segmento por `codigo_usina` e `indice_familia`); nulo quando o registro de família não existe. |
| `segment_id` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · indice_polinomio` | `indice_polinomio` tal como no arquivo (1-based). |
| `outflow_min_m3s` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · limite_inferior_vazao_jusante` | Limite inferior de vazão defluente (m³/s) do segmento, sem transformação. |
| `outflow_max_m3s` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · limite_superior_vazao_jusante` | Limite superior de vazão defluente (m³/s) do segmento, sem transformação. |
| `coefficient_0` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a0` | `coeficiente_a0` do polinômio do segmento, sem transformação. |
| `coefficient_1` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a1` | `coeficiente_a1` do polinômio do segmento, sem transformação. |
| `coefficient_2` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a2` | `coeficiente_a2` do polinômio do segmento, sem transformação. |
| `coefficient_3` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a3` | `coeficiente_a3` do polinômio do segmento, sem transformação. |
| `coefficient_4` | `polinjus.csv` › `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a4` | `coeficiente_a4` do polinômio do segmento, sem transformação. |

### `system/non_controllable_sources.json`

**Lê:** `sistema.dat`, `dger.dat`, `ree.dat`  
**Quando:** sempre.  
**Esquema:** [non_controllable_sources.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/non_controllable_sources.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/network.py`

Uma fonte não controlável por par (`codigo_submercado`, `indice_bloco`) da geração de usinas não simuladas de `sistema.dat` dentro do horizonte.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `non_controllable_sources[].id` | `sistema.dat` › `geracao_usinas_nao_simuladas · codigo_submercado`, `sistema.dat` › `geracao_usinas_nao_simuladas · indice_bloco` *(derivado)* | Grupos (`codigo_submercado`, `indice_bloco`) das linhas com data no horizonte de estudo/pós-estudo (ano 9999 incluído), ordenados e numerados de 0; grupo cujo submercado não está no mapa é pulado com aviso. |
| `non_controllable_sources[].name` | `sistema.dat` › `geracao_usinas_nao_simuladas · fonte`, `sistema.dat` › `geracao_usinas_nao_simuladas · codigo_submercado` *(derivado)* | `{fonte}_{codigo_submercado}`, com a primeira `fonte` não nula do grupo (`NCS` quando não há). |
| `non_controllable_sources[].operational_start_date` | `dger.dat` › `ano_inicial_historico` *(derivado)* | 1º de janeiro de `ano_inicial_historico`, como nas barras. |
| `non_controllable_sources[].bus_id` | `sistema.dat` › `geracao_usinas_nao_simuladas · codigo_submercado` *(derivado)* | Id Novomodelo do submercado do grupo. |
| `non_controllable_sources[].max_generation_mw` | `sistema.dat` › `geracao_usinas_nao_simuladas · valor` | Maior `valor` não nulo do grupo dentro do horizonte; a disponibilidade por estágio (`valor` / máximo) vai para `scenarios/non_controllable_stats.parquet`. |
| `non_controllable_sources[].allow_curtailment` | — *(constante)* | Sempre `false`: o NEWAVE abate a geração não simulada do mercado antes do despacho, logo ela é obrigatória; permitir corte desviava o despacho hidráulico. |

### `scenarios/inflow_history.parquet`

**Lê:** `vazoes.dat`, `confhd.dat`, `dger.dat`, `hidr.dat`, `exph.dat` (opcional)  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/newave/converters/inflow_windows.py`

Histórico de vazões incrementais mensais de `vazoes.dat`, uma linha por usina e mês-calendário, de janeiro do primeiro ano do histórico até o mês anterior ao início do estudo. A largura da matriz (320 ou 600 postos) não é declarada em nenhum lugar do deck e vem do tamanho do arquivo, descartada a largura cujo histórico, contado a partir de `ano_inicial_historico`, terminaria depois do ano de início do estudo; um tamanho que ainda sirva às duas larguras é lido como 320, com aviso.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `confhd.dat` › `codigo_usina`, `confhd.dat` › `posto` *(derivado)* | Id Novomodelo 0-based da usina (ordem crescente de `codigo_usina` entre as usinas ativas). A coluna de `vazoes.dat` é escolhida pelo `posto` da usina em `confhd.dat`; a série é emitida uma vez por posto, de modo que, se dois códigos ativos compartilham o posto, só o último na ordem de `confhd.dat` recebe linhas. |
| `start_date` | `dger.dat` › `ano_inicial_historico` *(derivado)* | Primeiro dia do mês. A linha i de `vazoes.dat` é tomada como o mês i contado a partir de janeiro de `ano_inicial_historico`. |
| `end_date` | `dger.dat` › `ano_inicial_historico` *(derivado)* | Primeiro dia do mês seguinte (janela semiaberta `[start_date, end_date)`). |
| `value_m3s` | `vazoes.dat` › `1..N (coluna por posto)`, `confhd.dat` › `posto`, `confhd.dat` › `codigo_usina_jusante`, `confhd.dat` › `usina_existente`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo` | Vazão incremental (m³/s): vazão natural do posto menos a soma das vazões naturais dos postos imediatamente a montante na cascata de `confhd.dat` (usinas NE/NC no caminho são atravessadas; usinas NE com enchimento em `exph.dat` contam como nó da cascata). A série é truncada no mês anterior ao início do estudo. |

### `scenarios/inflow_seasonal_stats.parquet`

**Lê:** `vazoes.dat`, `confhd.dat`, `dger.dat`, `hidr.dat`, `exph.dat` (opcional)  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/newave/converters/stochastic.py`

Média e desvio-padrão da vazão incremental histórica por usina e estágio, calculados por mês-calendário sobre `vazoes.dat`. A largura da matriz (320 ou 600 postos) não é declarada em nenhum lugar do deck e vem do tamanho do arquivo, descartada a largura cujo histórico, contado a partir de `ano_inicial_historico`, terminaria depois do ano de início do estudo; um tamanho que ainda sirva às duas larguras é lido como 320, com aviso.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `confhd.dat` › `codigo_usina` *(derivado)* | Id Novomodelo 0-based da usina (ordem crescente de `codigo_usina` entre as usinas ativas). Toda usina ativa recebe uma linha por estágio. |
| `stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Índice 0-based do estágio ao longo de todo o horizonte (estudo mais pós-estudo) definido em `dger.dat`; o estágio determina o mês-calendário usado na estatística. |
| `mean_m3s` | `vazoes.dat` › `1..N (coluna por posto)`, `confhd.dat` › `posto`, `confhd.dat` › `codigo_usina_jusante`, `dger.dat` › `ano_inicial_historico`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo` | Média, ao longo dos anos do histórico (até o mês anterior ao início do estudo), da vazão incremental do posto da usina no mês-calendário do estágio. A vazão incremental é a natural menos as naturais dos postos imediatamente a montante em `confhd.dat`. Usina cujo posto não tem coluna em `vazoes.dat` recebe 0. |
| `std_m3s` | `vazoes.dat` › `1..N (coluna por posto)`, `confhd.dat` › `posto`, `confhd.dat` › `codigo_usina_jusante` | Desvio-padrão amostral (n − 1) da mesma amostra de vazões incrementais do mês-calendário; 0 quando há uma única observação ou quando o posto não tem série. |

### `scenarios/load_seasonal_stats.parquet`

**Lê:** `sistema.dat`, `c_adic.dat` (opcional), `dger.dat`, `ree.dat`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/newave/converters/stochastic.py`

Mercado de energia por submercado e estágio (MWmédio) de `sistema.dat`, acrescido das cargas adicionais de `c_adic.dat`.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `bus_id` | `sistema.dat` › `mercado_energia · codigo_submercado` *(derivado)* | Id Novomodelo 0-based do submercado (ordem crescente de `codigo_submercado`, incluindo submercados fictícios de `custo_deficit`). Submercados sem mercado em `sistema.dat` também recebem linhas. |
| `stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Índice 0-based do estágio ao longo de todo o horizonte (estudo mais pós-estudo) de `dger.dat`. |
| `mean_mw` | `dger.dat` › `considera_carga_adicional`, `sistema.dat` › `mercado_energia · valor`, `sistema.dat` › `mercado_energia · data`, `c_adic.dat` › `valor`, `c_adic.dat` › `data`, `c_adic.dat` › `codigo_submercado` | Mercado do submercado no mês (MWmédio, já convertido pelo leitor) somado às cargas adicionais de `c_adic.dat` do mesmo submercado e mês, todas as razões. Pós-estudo: valor do ano 9999 de `sistema.dat` ou, na falta, o do último ano de estudo no mesmo mês-calendário, mais a parcela POS de `c_adic.dat`. Submercado sem mercado (fictício) recebe 0. |
| `std_mw` | — *(constante)* | Sempre 0.0: o mercado do NEWAVE é determinístico, sem dispersão por estágio. |

### `scenarios/load_factors.json`

**Lê:** `patamar.dat`, `dger.dat`, `sistema.dat`, `ree.dat`  
**Quando:** sempre.  
**Esquema:** [load_factors.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/load_factors.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/stochastic.py`

Fatores de carga por patamar (p.u.) de `patamar.dat`, uma entrada por submercado e estágio.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `load_factors[].bus_id` | `patamar.dat` › `carga_patamares · codigo_submercado` *(derivado)* | Id Novomodelo 0-based do submercado (ordem crescente de `codigo_submercado`). Só os submercados presentes em `carga_patamares` recebem entradas. |
| `load_factors[].stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Índice 0-based do estágio ao longo de todo o horizonte (estudo mais pós-estudo) de `dger.dat`. |
| `load_factors[].block_factors[].block_id` | `patamar.dat` › `carga_patamares · patamar`, `patamar.dat` › `numero_patamares` *(derivado)* | Patamar 0-based. O índice `patamar` de `carga_patamares` é global (corrido entre submercados) e é normalizado módulo `numero_patamares` antes de subtrair 1. |
| `load_factors[].block_factors[].factor` | `patamar.dat` › `carga_patamares · valor`, `patamar.dat` › `carga_patamares · data` | Fator de carga (p.u.) do patamar no mês do estágio. No pós-estudo repete o último ano de estudo com dado no mesmo mês-calendário; 1.0 quando o mês ou patamar não tem registro. |

### `scenarios/non_controllable_stats.parquet`

**Lê:** `sistema.dat`, `dger.dat`, `ree.dat`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/newave/converters/network.py`

Disponibilidade da geração não simulada por bloco e estágio, como fração da geração máxima do bloco.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `ncs_id` | `sistema.dat` › `geracao_usinas_nao_simuladas · codigo_submercado`, `sistema.dat` › `geracao_usinas_nao_simuladas · indice_bloco` *(derivado)* | Id 0-based do par (`codigo_submercado`, `indice_bloco`), o mesmo de `system/non_controllable_sources.json`. |
| `stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Índice 0-based do estágio ao longo de todo o horizonte de `dger.dat`. Estágios sem valor de geração (nem no mês nem no mesmo mês-calendário de outro ano) não geram linha. |
| `mean` | `sistema.dat` › `geracao_usinas_nao_simuladas · valor`, `sistema.dat` › `geracao_usinas_nao_simuladas · data` | Razão entre a geração não simulada do bloco no mês (MWmédio) e o maior valor do bloco em todo o arquivo (o `max_generation_mw` da fonte), limitada a [0, 1]. Pós-estudo e meses sem registro: valor do último ano com dado no mesmo mês-calendário. 0 quando o máximo do bloco é 0. |
| `std` | — *(constante)* | Sempre 0.0: a geração não simulada do NEWAVE é determinística. |

### `scenarios/non_controllable_factors.json`

**Lê:** `patamar.dat`, `sistema.dat`, `dger.dat`, `ree.dat`  
**Quando:** sempre.  
**Esquema:** [non_controllable_factors.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/non_controllable_factors.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/network.py`

Fatores por patamar (p.u.) da geração não simulada de `patamar.dat`, uma entrada por bloco de usinas não simuladas e estágio.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `non_controllable_factors[].ncs_id` | `sistema.dat` › `geracao_usinas_nao_simuladas · codigo_submercado`, `sistema.dat` › `geracao_usinas_nao_simuladas · indice_bloco` *(derivado)* | Id 0-based atribuído a cada par (`codigo_submercado`, `indice_bloco`) de `geracao_usinas_nao_simuladas` de `sistema.dat`, na ordem crescente do par, considerando só registros com data dentro do horizonte; o mesmo id de `system/non_controllable_sources.json`. |
| `non_controllable_factors[].stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Índice 0-based do estágio ao longo de todo o horizonte (estudo mais pós-estudo) de `dger.dat`. |
| `non_controllable_factors[].block_factors[].block_id` | `patamar.dat` › `usinas_nao_simuladas · patamar`, `patamar.dat` › `numero_patamares` *(derivado)* | Patamar 0-based. O índice `patamar` de `usinas_nao_simuladas` é global (corrido entre blocos) e é normalizado módulo `numero_patamares` antes de subtrair 1. |
| `non_controllable_factors[].block_factors[].factor` | `patamar.dat` › `usinas_nao_simuladas · valor`, `patamar.dat` › `usinas_nao_simuladas · data`, `patamar.dat` › `usinas_nao_simuladas · codigo_submercado`, `patamar.dat` › `usinas_nao_simuladas · indice_bloco` | Fator (p.u.) do patamar para o bloco no mês do estágio. No pós-estudo, e em meses de estudo sem registro, usa o último ano com dado no mesmo mês-calendário; 1.0 quando ausente. Piso de 1e-6. |

### `constraints/hydro_bounds.parquet`

**Lê:** `dsvagua.dat` (opcional), `dger.dat`, `confhd.dat`, `hidr.dat`, `modif.dat` (opcional), `ghmin.dat` (opcional), `exph.dat` (opcional), `volref_saz.dat` (opcional)  
**Quando:** somente quando ao menos uma fonte produz linhas: `dsvagua.dat` com desvios (e `outros_usos_da_agua` de `dger.dat` diferente de 0), registros temporais `VMAXT`/`VMINT`/`TURBMAXT`/`TURBMINT`/`VAZMINT` em `modif.dat`, `ghmin.dat`, usina NE com enchimento ou usina EE com entrada de máquinas em `exph.dat`, ou usina com `CFUGA`/`CMONT`/linha em `volref_saz.dat`  
**Código:** `src/novomodelo_bridge/newave/converters/hydro/bounds.py`

Limites por (usina, estágio) que apertam o envelope de `hydros.json`, resultado da junção externa de três tabelas: retirada de água (`dsvagua.dat`), limites temporais (`modif.dat`, `ghmin.dat`, rampa de máquinas de `exph.dat`) e turbinamento máximo corrigido pela queda por estágio (`CFUGA`/`CMONT`/`volref_saz.dat`). Colunas de máximo acima do valor declarado em `hydros.json` são limitadas ao declarado, com aviso. A coluna `max_generation_mw` só existe quando há usina NE em enchimento ou usina EE com entrada de máquinas.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `confhd.dat` › `codigo_usina` *(derivado)* | Id Novomodelo da usina. Desvios de `dsvagua.dat` em usina NC são atribuídos à primeira usina existente a jusante na cadeia `codigo_usina_jusante`; desvios em usina fictícia são descartados. |
| `stage_id` | `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo`, `dsvagua.dat` › `data`, `modif.dat` › `VMAXT · data_inicio`, `ghmin.dat` › `data`, `exph.dat` › `data_entrada_operacao` *(derivado)* | Estágio 0-based contado a partir de `mes_inicio_estudo`/`ano_inicio_estudo`, cobrindo estudo e pós-estudo; cada fonte ocupa os estágios em que tem valor ativo e a junção mantém as linhas de qualquer uma delas. |
| `water_withdrawal_m3s` | `dsvagua.dat` › `codigo_usina`, `dsvagua.dat` › `data`, `dsvagua.dat` › `valor`, `dger.dat` › `outros_usos_da_agua`, `confhd.dat` › `codigo_usina_jusante` | Retirada de água (m³/s) de `dsvagua.dat`: soma de `valor` por (`codigo_usina`, `data`) com o sinal invertido (no NEWAVE retirada é negativa). Ignorado quando `outros_usos_da_agua = 0`. Estágios de pós-estudo repetem o padrão mensal do último ano civil presente no arquivo para a usina. Nulo nas linhas vindas das demais fontes. |
| `min_storage_hm3` | `modif.dat` › `VMINT · volume, unidade`, `modif.dat` › `VMINT · data_inicio`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `VOLMAX · volume`, `dger.dat` › `sazonaliza_vmint` | Armazenamento mínimo (hm³) de `VMINT`: o `volume` em hm³ quando `unidade` = h ou, com `%`, `volume_minimo + (volume / 100) · volume útil`; função degrau a partir de `data_inicio` até o próximo registro; valor maior ou igual a 99990 restaura o padrão (linha sem valor). No pós-estudo repete o padrão mensal do último ano de estudo se `sazonaliza_vmint = 1`, senão congela o último estágio. Ignorado para usina sem volume útil. |
| `max_storage_hm3` | `modif.dat` › `VMAXT · volume, unidade`, `modif.dat` › `VMAXT · data_inicio`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `VOLMAX · volume`, `dger.dat` › `sazonaliza_vmaxt` | Armazenamento máximo (hm³) de `VMAXT` (volume de espera): o `volume` em hm³ quando `unidade` = h ou, com `%`, `volume_minimo + (volume / 100) · volume útil`; função degrau a partir de `data_inicio`; valor maior ou igual a 99990 restaura o padrão. No pós-estudo repete o padrão mensal do último ano de estudo se `sazonaliza_vmaxt = 1`, senão congela o último estágio. |
| `min_turbined_m3s` | `dger.dat` › `restricao_turbinamento`, `modif.dat` › `TURBMINT · turbinamento`, `modif.dat` › `TURBMINT · data_inicio` | Turbinamento mínimo (m³/s) de `TURBMINT`, valor absoluto em função degrau a partir de `data_inicio`; valor maior ou igual a 99990 (família 99999) restaura o padrão. No pós-estudo congela o último estágio de estudo. |
| `max_turbined_m3s` | `dger.dat` › `restricao_turbinamento`, `modif.dat` › `TURBMAXT · turbinamento`, `modif.dat` › `TURBMAXT · data_inicio`, `modif.dat` › `CFUGA · nivel`, `modif.dat` › `CMONT · nivel`, `volref_saz.dat` › `valor`, `dger.dat` › `sazonaliza_cfuga_cmont`, `hidr.dat` › `maquinas_conjunto_1 … maquinas_conjunto_5`, `hidr.dat` › `vazao_nominal_conjunto_1 … vazao_nominal_conjunto_5`, `hidr.dat` › `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5`, `modif.dat` › `POTEFE · potencia, conjunto`, `hidr.dat` › `queda_nominal_conjunto_1 … queda_nominal_conjunto_5`, `hidr.dat` › `teif`, `hidr.dat` › `ip`, `hidr.dat` › `tipo_turbina`, `hidr.dat` › `produtibilidade_especifica`, `exph.dat` › `data_entrada_operacao`, `exph.dat` › `conjunto_maquina_entrada`, `exph.dat` › `data_inicio_enchimento`, `exph.dat` › `duracao_enchimento` | Três origens, nesta precedência: (1) `TURBMAXT` de `modif.dat`, valor absoluto em degrau (valor maior ou igual a 99990 restaura o padrão; pós-estudo congelado); (2) para usina NE em enchimento, o engolimento corrigido pela queda com apenas as máquinas já em operação (`data_entrada_operacao` por conjunto, nunca antes do fim do enchimento), 0 antes de entrar em operação, e esta linha prevalece sobre qualquer outra no mesmo estágio; (3) onde ainda vazio, o engolimento corrigido pela queda do estágio (`h = ρ_eq / ρ_esp`, com `CFUGA`/`CMONT` e `volref_saz.dat`), pela mesma fórmula de `hydros[].generation.max_turbined_m3s`, com as máquinas em operação no estágio para usina `EE`; usina `EE` sem queda por estágio recebe linha só nos estágios anteriores à entrada da última máquina de `exph.dat`, na mesma queda de referência do valor declarado. Valores acima do declarado em `hydros.json` são reduzidos ao declarado. |
| `min_outflow_m3s` | `dger.dat` › `desconsidera_vazao_minima`, `modif.dat` › `VAZMINT · vazao`, `modif.dat` › `VAZMINT · data_inicio`, `modif.dat` › `VAZMINT · periodo` | Vazão defluente mínima (m³/s) de `VAZMINT`, valor absoluto em função degrau a partir de `data_inicio`; valor maior ou igual a 99990 (família 99999) restaura o padrão. Registros com ano `PRE` ou `POS` formam degraus sobre um ano que se repete: o valor `PRE` vigente no mês anterior ao início do estudo vale até o primeiro registro datado, e os degraus `POS` definem cada mês do pós-estudo. Sem `POS`, o pós-estudo congela o último estágio de estudo. |
| `min_generation_mw` | `dger.dat` › `considera_ghmin`, `ghmin.dat` › `codigo_usina`, `ghmin.dat` › `data`, `ghmin.dat` › `patamar`, `ghmin.dat` › `geracao`, `exph.dat` › `data_inicio_enchimento` | Geração mínima (MW) de `ghmin.dat`, somente linhas com `patamar = 0` (média dos patamares): função degrau a partir de `data` até o próximo registro; registros de ano 9999 valem para o mês civil correspondente no pós-estudo, e meses sem registro repetem o último ano de estudo. Linhas de rampa de usina NE em enchimento recebem 0. |
| `max_generation_mw` | `exph.dat` › `data_entrada_operacao`, `exph.dat` › `conjunto_maquina_entrada`, `exph.dat` › `data_inicio_enchimento`, `exph.dat` › `duracao_enchimento`, `confhd.dat` › `usina_existente`, `hidr.dat` › `maquinas_conjunto_1 … maquinas_conjunto_5`, `hidr.dat` › `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5`, `modif.dat` › `POTEFE · potencia, conjunto`, `modif.dat` › `NUMCNJ · numero`, `modif.dat` › `NUMMAQ · numero_maquinas` *(condicional: somente quando o caso tem usina NE com registro de enchimento ou usina EE com entrada de máquinas em `exph.dat` (a coluna é omitida nos demais casos))* | Potência nominal (MW) das máquinas em operação em cada estágio anterior à entrada da última máquina: soma de `n_online · potencia_nominal_conjunto` por conjunto. Para usina NE, 0 antes do fim do enchimento; para usina EE, a configuração de `modif.dat` no início do estudo mais as máquinas de `exph.dat` já entradas, na mesma linha de qualquer limite de `modif.dat`/`ghmin.dat` do estágio. Nulo nas demais linhas de `modif.dat`/`ghmin.dat`. Valores acima do declarado em `hydros.json` são reduzidos ao declarado. |

### `constraints/thermal_bounds.parquet`

**Lê:** `conft.dat`, `clast.dat`, `term.dat`, `expt.dat` (opcional), `manutt.dat` (opcional), `dger.dat`  
**Quando:** somente quando os limites de alguma térmica variam entre estágios (e diferem, portanto, do envelope publicado em `system/thermals.json`), ou quando alguma térmica tem custo variável por ano de estudo ou modificação datada em `clast.dat`  
**Código:** `src/novomodelo_bridge/newave/converters/thermal.py`

Limites de geração por (térmica, estágio) seguindo a ordem do sintetizador do NEWAVE (IP, EXPT, disponibilidade POTEF/GTMIN, MANUTT), com sobrescrita de custo por estágio das térmicas de custo variável. Quando a tabela é emitida, toda térmica de `term.dat`, `expt.dat` ou `manutt.dat` recebe linhas; cada linha fica dentro do envelope de `system/thermals.json`.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `thermal_id` | `conft.dat` › `usinas · codigo_usina` *(derivado)* | Id Novomodelo da térmica; códigos de `expt.dat`, `manutt.dat` ou `term.dat` ausentes de `conft.dat` são ignorados. |
| `stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Todo estágio do estudo e do pós-estudo. |
| `min_generation_mw` | `term.dat` › `usinas · geracao_minima (mes 1 a 12)`, `term.dat` › `usinas · geracao_minima (mes = 13)`, `expt.dat` › `expansoes · tipo = GTMIN · modificacao`, `expt.dat` › `expansoes · data_inicio`, `expt.dat` › `expansoes · data_fim`, `conft.dat` › `usinas · usina_existente`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `mes_inicio_estudo` | `geracao_minima` de `term.dat` em dois regimes: nos estágios do primeiro ano de estudo (o ano de `ano_inicio_estudo`), a coluna do mês calendário do estágio; nos anos seguintes, inclusive o pós-estudo, o valor único dos demais anos (que a inewave entrega como `mes = 13`), independentemente de `num_anos_manutencao_utes`. Usina com esse valor em branco conserva a coluna do mês calendário nos anos seguintes. Em cima disso entram as janelas GTMIN de `expt.dat`. Para usina `EE` ou `NE` em `conft.dat`, o NEWAVE descarta o mínimo de cadastro: fora das janelas, e para usina sem nenhuma janela, vale 0. Usina `EX` conserva o valor do `term.dat` onde o `expt.dat` não declara nada. No pós-estudo congela na configuração do último estágio de estudo (ou do estágio final, para usina que só entra no pós-estudo). |
| `max_generation_mw` | `term.dat` › `usinas · potencia_instalada`, `term.dat` › `usinas · fator_capacidade_maximo`, `term.dat` › `usinas · teif`, `term.dat` › `usinas · indisponibilidade_programada`, `expt.dat` › `expansoes · tipo = POTEF · modificacao`, `expt.dat` › `expansoes · tipo = FCMAX · modificacao`, `expt.dat` › `expansoes · tipo = TEIFT · modificacao`, `expt.dat` › `expansoes · tipo = IPTER · modificacao`, `manutt.dat` › `manutencoes · data_inicio`, `manutt.dat` › `manutencoes · duracao`, `manutt.dat` › `manutencoes · potencia`, `conft.dat` › `usinas · usina_existente`, `dger.dat` › `num_anos_manutencao_utes` | `potencia` × FCMAX/100 × (100 − IP)/100 × (100 − TEIF)/100, com POTEF/FCMAX/TEIFT/IPTER de `expt.dat` aplicados por janela. Para usina `EE` ou `NE` em `conft.dat`, o NEWAVE descarta a potência de cadastro: fora de toda janela POTEF, e sem nenhuma janela, a potência é 0; usina `EX` conserva a do `term.dat`. IP é zerado e a redução de `manutt.dat` (potência × fração do mês em manutenção) subtraída só nos estágios antes do fim de `num_anos_manutencao_utes`; se GTMIN excede a capacidade, o máximo é elevado ao mínimo com aviso. |
| `cost_per_mwh` | `clast.dat` › `usinas · valor`, `clast.dat` › `usinas · indice_ano_estudo`, `clast.dat` › `modificacoes · custo`, `clast.dat` › `modificacoes · data_inicio`, `clast.dat` › `modificacoes · data_fim` | Nulo para térmica de custo constante. Para as demais, o `valor` do ano de estudo do estágio, sobrescrito em ordem de arquivo pelas modificações cujo intervalo contém a data do estágio; o pós-estudo congela no custo do último estágio de estudo. |

### `constraints/line_bounds.parquet`

**Lê:** `sistema.dat`, `patamar.dat`, `dger.dat`  
**Quando:** sempre.  
**Código:** `src/novomodelo_bridge/newave/converters/network.py`

Uma linha base por (linha, estágio) com os limites de intercâmbio do mês, mais uma linha por patamar onde o fator de `patamar.dat` difere de 1.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `line_id` | `sistema.dat` › `limites_intercambio · submercado_de`, `sistema.dat` › `limites_intercambio · submercado_para` *(derivado)* | Mesmo mapa de pares canônicos de `system/lines.json`. |
| `stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `ano_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Todo estágio do estudo e do pós-estudo, para toda linha. |
| `block_id` | `patamar.dat` › `intercambio_patamares · patamar` | Nulo na linha base do estágio; `patamar` − 1 nas linhas de sobrescrita por patamar, emitidas só quando o fator direto ou reverso é diferente de 1.0. |
| `direct_mw` | `sistema.dat` › `limites_intercambio · valor`, `sistema.dat` › `limites_intercambio · sentido`, `patamar.dat` › `intercambio_patamares · valor` | Linha base: limite menor→maior código no (ano, mês) do estágio, com recuo ao último ano do arquivo no mesmo mês; no pós-estudo congela no valor do último estágio de estudo. Linha por patamar: base × fator de `intercambio_patamares` (de→para com de < para), fator do pós-estudo repetindo o padrão mensal do último ano. |
| `reverse_mw` | `sistema.dat` › `limites_intercambio · valor`, `sistema.dat` › `limites_intercambio · sentido`, `patamar.dat` › `intercambio_patamares · valor` | Mesma regra no sentido maior→menor código (fator de `intercambio_patamares` com de > para). |

### `constraints/penalty_overrides_bus.parquet`

**Lê:** `sistema.dat`, `dger.dat`, `ree.dat`  
**Quando:** somente quando `sistema.dat` marca algum submercado como fictício (`ficticio`) e há custo de déficit positivo  
**Código:** `src/novomodelo_bridge/newave/converters/network.py`

Sobrescrita do custo de excesso nas barras fictícias em todo estágio, para proibir na prática o excesso de energia nesses nós de passagem.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `bus_id` | `sistema.dat` › `custo_deficit · ficticio`, `sistema.dat` › `custo_deficit · codigo_submercado` *(derivado)* | Id Novomodelo de cada submercado com `ficticio` verdadeiro, em ordem crescente de código. |
| `stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Todo estágio do estudo e do pós-estudo, por barra fictícia. |
| `excess_cost` | `sistema.dat` › `custo_deficit · custo` | Primeiro custo de déficit positivo do arquivo, o mesmo que `system/buses.json` atribui às barras fictícias, para que excesso e déficit custem o mesmo ali. |

### `constraints/penalty_overrides_hydro.parquet`

**Lê:** `sistema.dat`, `penalid.dat` (opcional), `hidr.dat`, `confhd.dat`, `modif.dat` (opcional), `exph.dat` (opcional), `dger.dat`  
**Quando:** somente quando alguma usina ativa tem registro CFUGA ou CMONT em `modif.dat` que faça a PROD_MEDIA_SIN de algum estágio diferir da global; a tabela é esparsa e só carrega as colunas que diferem  
**Código:** `src/novomodelo_bridge/newave/converters/network.py`

Recalcula o bloco `hydro` de `penalties.json` com a PROD_MEDIA_SIN de cada estágio (uniforme para todas as usinas, como a constante única do NEWAVE) e emite só o que difere do valor global.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `hydro_id` | `confhd.dat` › `usinas · codigo_usina` *(derivado)* | Toda usina hidráulica ativa, em ordem crescente de id, repetida para cada estágio que difere. |
| `stage_id` | `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Somente os estágios em que ao menos uma coluna difere do valor global. |
| `spillage_cost` | `modif.dat` › `CFUGA`, `modif.dat` › `CMONT`, `hidr.dat` › `cadastro · produtibilidade_especifica` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | 0.000300 × PROD_MEDIA_SIN do estágio (PRODT média com o canal de fuga/cota de montante de CFUGA/CMONT vigentes). |
| `turbined_cost` | `modif.dat` › `CFUGA`, `modif.dat` › `CMONT`, `hidr.dat` › `cadastro · produtibilidade_especifica` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | 0.000333 × PROD_MEDIA_SIN do estágio. |
| `diversion_cost` | `modif.dat` › `CFUGA`, `modif.dat` › `CMONT`, `hidr.dat` › `cadastro · produtibilidade_especifica` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | 0.000300 × PROD_MEDIA_SIN do estágio. |
| `turbined_violation_below_cost` | `penalid.dat` › `TURBMN · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `modif.dat` › `CFUGA`, `modif.dat` › `CMONT` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | TURBMN (ou 10 × maior custo de déficit) × PROD_MEDIA_SIN do estágio. |
| `outflow_violation_below_cost` | `penalid.dat` › `VAZMIN · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `modif.dat` › `CFUGA`, `modif.dat` › `CMONT` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | VAZMIN (ou 10 × maior custo de déficit) × PROD_MEDIA_SIN do estágio. |
| `outflow_violation_above_cost` | `penalid.dat` › `TURBMX · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `modif.dat` › `CFUGA`, `modif.dat` › `CMONT` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | TURBMX (ou 10 × maior custo de déficit) × PROD_MEDIA_SIN do estágio. |
| `storage_violation_below_cost` | `penalid.dat` › `VOLMIN · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica`, `confhd.dat` › `usinas · codigo_usina_jusante` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | Mesma fórmula de `penalties.json` com a MAX_PRODTACUM_SIN do estágio. Como o pipeline mantém essa constante fixa em todo o horizonte, na prática a coluna nunca difere e não é emitida. |
| `evaporation_violation_cost` | `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica`, `confhd.dat` › `usinas · codigo_usina_jusante` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | 10 × maior custo de déficit × MAX_PRODTACUM_SIN do estágio; constante no horizonte, logo na prática não emitida. |
| `water_withdrawal_violation_cost` | `penalid.dat` › `DESVIO · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica`, `confhd.dat` › `usinas · codigo_usina_jusante` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | DESVIO (ou 10 × maior custo de déficit) × MAX_PRODTACUM_SIN do estágio; constante no horizonte, logo na prática não emitida. |
| `inflow_nonnegativity_cost` | `penalid.dat` › `DESVIO · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo`, `hidr.dat` › `cadastro · produtibilidade_especifica` *(condicional: somente quando o valor do estágio difere de `penalties.json`)* | `water_withdrawal_violation_cost` do estágio + 1; segue a constante MAX_PRODTACUM_SIN, logo na prática não emitida. |

### `constraints/generic_constraints.json`

**Lê:** `curva.dat` (opcional), `dger.dat`, `confhd.dat`, `hidr.dat`, `modif.dat` (opcional), `ree.dat`, `exph.dat` (opcional), `restricao-eletrica.csv`, `re.dat` (opcional), `sistema.dat`, `patamar.dat`, `penalid.dat` (opcional), `agrint.dat` (opcional)  
**Quando:** somente quando alguma família gera restrições: curva de segurança (`curva.dat` presente, `curva_aversao` de `dger.dat` diferente de 0 e ao menos um REE com reservatório de regularização mensal com volume útil), restrições elétricas (`restricao-eletrica.csv` apontado por `indices.csv`, ou `re.dat`, com ao menos um limite válido) ou agrupamentos de intercâmbio (`agrint.dat` com grupos e limites); cada família só é lida se o flag correspondente de `dger.dat` estiver ligado (`agrupamento_livre`, `restricoes_eletricas`, `restricoes_eletricas_especiais`)  
**Esquema:** [generic_constraints.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/generic_constraints.schema.json) · **Código:** `src/novomodelo_bridge/newave/converters/constraints.py`

Restrições genéricas de três famílias concatenadas nesta ordem: curva de segurança (VminOP, uma por REE de `curva.dat`), restrições elétricas (`restricao-eletrica.csv` e `re.dat`) e agrupamentos de intercâmbio (`agrint.dat`). Os limites ficam em `constraints/generic_constraint_bounds.parquet`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `constraints[].id` | `curva.dat` › `curva_seguranca · codigo_ree`, `restricao-eletrica.csv` › `RE · cod_rest`, `re.dat` › `usinas_conjuntos · conjunto`, `agrint.dat` › `agrupamentos · agrupamento` *(derivado)* | Id 0-based contíguo, compartilhado pelas três famílias na ordem de emissão: uma restrição por REE da curva (ordem crescente de `codigo_ree`); para cada código de restrição elétrica (ordem crescente), uma restrição para o teto e, se houver piso, outra de mesmo nome; uma por grupo de `agrint.dat` (ordem crescente). Restrições sem nenhum limite não recebem id. |
| `constraints[].name` | `ree.dat` › `nome`, `restricao-eletrica.csv` › `RE · cod_rest`, `re.dat` › `usinas_conjuntos · conjunto`, `agrint.dat` › `agrupamentos · agrupamento` *(derivado)* | `VminOP_<nome do REE>` (nome de `ree.dat`; o código quando ausente), `RE_<código>` ou `AGRINT_<grupo>`. |
| `constraints[].description` | `curva.dat` › `curva_seguranca · codigo_ree`, `ree.dat` › `nome`, `restricao-eletrica.csv` › `RE · cod_rest`, `re.dat` › `usinas_conjuntos · conjunto`, `agrint.dat` › `agrupamentos · agrupamento` *(derivado)* | Texto fixo em inglês com o código da origem: energia armazenada mínima do REE (código e nome), restrição elétrica (código) ou grupo de intercâmbio (número). |
| `constraints[].expression` | `dger.dat` › `agrupamento_livre · restricoes_eletricas · restricoes_eletricas_especiais`, `confhd.dat` › `ree`, `confhd.dat` › `codigo_usina`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `produtibilidade_especifica`, `modif.dat` › `VOLMAX · volume`, `modif.dat` › `VOLMIN · volume`, `restricao-eletrica.csv` › `RE · formula`, `re.dat` › `usinas_conjuntos · codigo_usina`, `agrint.dat` › `agrupamentos · submercado_de`, `agrint.dat` › `agrupamentos · submercado_para`, `agrint.dat` › `agrupamentos · coeficiente`, `sistema.dat` › `limites_intercambio · submercado_de`, `sistema.dat` › `limites_intercambio · submercado_para` | Curva de segurança: soma de `@rho_acum_h<id> * hydro_storage_final(<id>)` sobre as usinas do REE (`ree` de `confhd.dat`) que são reservatório de regularização mensal (`tipo_regulacao` M) com volume útil positivo (após VOLMAX/VOLMIN de `modif.dat`) e produtibilidade acumulada positiva. Restrição elétrica: termos `ger_usih(cod)` da fórmula viram `hydro_generation(id)` e `ener_interc(A, B)` viram `line_direct(id)` ou `line_reverse(id)` conforme A < B, com o coeficiente e o sinal preservados; termos com usina fora do caso ou par sem linha em `sistema.dat` são descartados. Restrição só em `re.dat`: soma de `hydro_generation` das usinas do conjunto. Agrupamento: soma ponderada por `coeficiente` de `line_direct`/`line_reverse` do par de submercados. |
| `constraints[].slack.enabled` | `penalid.dat` › `ELETRI · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo` *(derivado)* | Verdadeiro para a curva de segurança; para restrições elétricas, verdadeiro quando há penalidade ELETRI em `penalid.dat` ou, na falta, custo de déficit positivo em `sistema.dat`; falso para agrupamentos de intercâmbio (limite rígido). |
| `constraints[].slack.penalty` | `curva.dat` › `custos_penalidades · penalidade`, `penalid.dat` › `ELETRI · valor_R$_MWh`, `sistema.dat` › `custo_deficit · custo` *(condicional: somente quando `slack.enabled` é verdadeiro (curva de segurança e restrições elétricas))* | Curva de segurança: `penalidade` do REE em `curva.dat` (R$/MWh); 1000.0 quando o REE não tem entrada ou o valor declarado não é positivo. Restrição elétrica: valor ELETRI de `penalid.dat` (primeiro patamar de penalidade) ou, na falta, 10 vezes o maior custo de déficit de `sistema.dat`. |

### `constraints/generic_constraint_bounds.parquet`

**Lê:** `curva.dat` (opcional), `dger.dat`, `confhd.dat`, `hidr.dat`, `modif.dat` (opcional), `ree.dat`, `exph.dat` (opcional), `restricao-eletrica.csv`, `re.dat` (opcional), `sistema.dat`, `patamar.dat`, `penalid.dat` (opcional), `agrint.dat` (opcional)  
**Quando:** escrito junto com `constraints/generic_constraints.json`, sob a mesma condição  
**Código:** `src/novomodelo_bridge/newave/converters/constraints.py`

Limites por restrição, estágio e patamar das três famílias, concatenados na mesma ordem de `constraints/generic_constraints.json`. O sentido é dado pelo preenchimento: só `bound_lower` para maior-ou-igual, só `bound_upper` para menor-ou-igual.

| Coluna | Origem | Transformação |
| --- | --- | --- |
| `constraint_id` | `curva.dat` › `curva_seguranca · codigo_ree`, `restricao-eletrica.csv` › `RE · cod_rest`, `re.dat` › `usinas_conjuntos · conjunto`, `agrint.dat` › `agrupamentos · agrupamento` *(derivado)* | O `id` da restrição em `constraints/generic_constraints.json`. |
| `stage_id` | `curva.dat` › `curva_seguranca · data`, `curva.dat` › `configuracoes_penalizacao`, `restricao-eletrica.csv` › `RE-LIM-FORM-PER-PAT · PerIni`, `restricao-eletrica.csv` › `RE-LIM-FORM-PER-PAT · PerFin`, `restricao-eletrica.csv` › `RE-HORIZ-PER · PerIni`, `restricao-eletrica.csv` › `RE-HORIZ-PER · PerFin`, `re.dat` › `restricoes · mes_inicio`, `re.dat` › `restricoes · ano_inicio`, `re.dat` › `restricoes · mes_fim`, `re.dat` › `restricoes · ano_fim`, `ree.dat` › `mes_fim_individualizado`, `ree.dat` › `ano_fim_individualizado`, `agrint.dat` › `limites_agrupamentos · data_inicio`, `agrint.dat` › `limites_agrupamentos · data_fim`, `dger.dat` › `mes_inicio_estudo`, `dger.dat` › `num_anos_estudo`, `dger.dat` › `num_anos_pos_estudo` *(derivado)* | Índice 0-based do estágio de `dger.dat`. Curva: estágio da `data` de cada registro; estágios de estudo sem registro e o pós-estudo repetem o último ano por mês-calendário quando o terceiro campo da linha de penalização de `curva.dat` é 1, senão o pós-estudo congela o percentual do último estágio de estudo. Elétricas: somente os estágios cobertos pela janela de vigência do próprio limite — `mes_inicio`/`ano_inicio` a `mes_fim`/`ano_fim` no `re.dat`, `PerIni`/`PerFin` do `RE-LIM-FORM-PER-PAT` interseccionado com o `RE-HORIZ-PER` no CSV; estágio fora de toda janela declarada não recebe limite. Os limites de `re.dat` valem só a partir do fim do período individualizado (`ree.dat`). Agrupamentos: estágios de estudo cobertos pelos limites (fim aberto vai até o fim do horizonte) e congelamento do último estágio de estudo no pós-estudo. |
| `block_id` | `restricao-eletrica.csv` › `RE-LIM-FORM-PER-PAT · pat`, `re.dat` › `restricoes · patamar`, `patamar.dat` › `numero_patamares` *(derivado)* | Nulo para a curva de segurança (limite por estágio, todos os patamares). Elétricas: `pat` menos 1; `patamar` 0 de `re.dat` expande para todos os `numero_patamares` patamares. Agrupamentos: posição da coluna de limite na linha de `agrint.dat` (uma por patamar; colunas faltantes repetem a última). |
| `bound_lower` | `curva.dat` › `curva_seguranca · valor`, `hidr.dat` › `volume_minimo`, `hidr.dat` › `volume_maximo`, `hidr.dat` › `tipo_regulacao`, `hidr.dat` › `produtibilidade_especifica`, `hidr.dat` › `a0_volume_cota..a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4`, `hidr.dat` › `canal_fuga_medio`, `hidr.dat` › `tipo_perda`, `hidr.dat` › `perdas`, `hidr.dat` › `volume_referencia`, `confhd.dat` › `codigo_usina_jusante`, `modif.dat` › `VOLMAX · volume`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `CFUGA · nivel`, `modif.dat` › `CMONT · nivel`, `dger.dat` › `sazonaliza_cfuga_cmont`, `restricao-eletrica.csv` › `RE-LIM-FORM-PER-PAT · lim_inf` | Curva de segurança: energia armazenada mínima do REE em MWmês, `valor` (%) aplicado ao volume útil de cada reservatório do REE mais a parcela do volume mínimo, ambos ponderados pela produtibilidade acumulada integrada da cascata (vol. mínimo ao máximo, com FICT dobradas na usina real a montante, e CFUGA/CMONT de `modif.dat` por estágio), convertida de MW/(m³/s)·hm³ para MWmês com as horas reais do mês do estágio. Elétricas: `lim_inf` de `restricao-eletrica.csv` quando maior que −1e29 (valores abaixo são ilimitados). Nulo para tetos elétricos, `re.dat` e agrupamentos. |
| `bound_upper` | `restricao-eletrica.csv` › `RE-LIM-FORM-PER-PAT · lim_sup`, `re.dat` › `restricoes · restricao`, `agrint.dat` › `limites_agrupamentos · valor` | Elétricas: `lim_sup` de `restricao-eletrica.csv` (quando maior que −1e29) nos estágios individualizados; a partir do fim do período individualizado prevalece `restricao` de `re.dat`, propagado até o fim do horizonte. Agrupamentos: limite do patamar (MWmédio) de `agrint.dat`. Nulo para a curva de segurança e para pisos elétricos. |

### `constraints/generic_parameters.json`

**Lê:** `confhd.dat`, `hidr.dat`, `exph.dat` (opcional), `modif.dat` (opcional), `curva.dat` (opcional), `dger.dat`, `ree.dat`  
**Quando:** sempre.  
**Esquema:** [generic_parameters.schema.json](https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/generic_parameters.schema.json) · **Código:** `src/novomodelo_bridge/novomodelo/scalar_parameters.py`

Declara `rho_eq_h{id}` e `rho_acum_h{id}` para toda usina hidráulica ativa; a produtibilidade acumulada vira `per_stage` nas usinas referenciadas pelas restrições VminOP de `curva.dat`.

| Campo | Origem | Transformação |
| --- | --- | --- |
| `scalar_parameters[].id` | `confhd.dat` › `usinas · codigo_usina` *(derivado)* | Sequencial de 0, dois por usina (ρ_eq e depois ρ_acum), em ordem crescente de id Novomodelo. |
| `scalar_parameters[].name` | `confhd.dat` › `usinas · codigo_usina` *(derivado)* | `rho_eq_h{id}` e `rho_acum_h{id}` com o id Novomodelo da usina; são os tokens `@name` usados em `constraints/generic_constraints.json`. |
| `scalar_parameters[].kind` | `curva.dat` › `curva_seguranca · codigo_ree`, `dger.dat` › `curva_aversao` | `computed` para todo `rho_eq_h{id}` e, por padrão, para `rho_acum_h{id}`; `per_stage` no `rho_acum_h{id}` das usinas que entram numa expressão VminOP (`curva.dat` presente, `curva_aversao` ≠ 0). |
| `scalar_parameters[].computed_spec.tag` | — *(constante)* | `equivalent_productivity` na entrada ρ_eq e `accumulated_productivity` na entrada ρ_acum; o Novomodelo calcula o valor a partir da geometria em tempo de solução. |
| `scalar_parameters[].computed_spec.hydro_id` | `confhd.dat` › `usinas · codigo_usina` *(derivado)* | Id Novomodelo da usina a que o parâmetro se refere. |
| `scalar_parameters[].values` | `curva.dat` › `curva_seguranca · codigo_ree`, `confhd.dat` › `usinas · ree`, `confhd.dat` › `usinas · codigo_usina_jusante`, `hidr.dat` › `cadastro · produtibilidade_especifica`, `hidr.dat` › `cadastro · a0_volume_cota … a4_volume_cota`, `modif.dat` › `VOLCOTA · coeficientes a0 a a4`, `hidr.dat` › `cadastro · canal_fuga_medio`, `hidr.dat` › `cadastro · perdas`, `hidr.dat` › `cadastro · tipo_perda`, `hidr.dat` › `cadastro · volume_minimo`, `hidr.dat` › `cadastro · volume_maximo`, `hidr.dat` › `cadastro · tipo_regulacao`, `modif.dat` › `CFUGA`, `modif.dat` › `CMONT`, `modif.dat` › `VOLMIN · volume`, `modif.dat` › `VOLMAX · volume`, `dger.dat` › `mes_inicio_estudo` *(condicional: somente nas entradas `rho_acum_h{id}` de usinas referenciadas por uma restrição VminOP)* | Pares [estágio, valor] com a produtibilidade acumulada de cascata na convenção de energia armazenada (média integrada da cota entre `volume_minimo` e `volume_maximo`, como o `produtibilidade_acumulada_calculo_earm` do pmo.dat), recalculada nos estágios com CFUGA/CMONT ativos. Dividida por 2.628 × horas do mês / 730 para ficar em MWmês/hm³ e casar com o lado direito da restrição. |

## Índice por origem

Para cada arquivo do deck (e, no DECOMP, cada registro): o que ele alimenta e o que nele ainda não é convertido.

### `dger.dat`

**Estado:** convertido em parte. **Lido por:** 24 arquivos gerados; ver a visão geral.

O bridge lê cerca de trinta campos (horizonte, aberturas, forwards, iterações, CVaR, curva de aversão, simulação final, reamostragem, seleção de cortes, GNL antecipado, FPHA, outros usos, sazonalização de VMAXT/VMINT/CFUGA-CMONT, estados dos cortes). Os demais flags são agrupados abaixo por tema.

- `ordem_maxima_parp` → `config.json` › `estimation.max_order`
- `consideracao_media_anual_afluencias` → `config.json` › `estimation.order_selection`
- `impressao_estados_geracao_cortes` → `config.json` › `exports.states`
- `tipo_execucao` → `config.json` › `simulation.enabled`
- `tipo_simulacao_final` → `config.json` › `simulation.enabled`
- `num_series_sinteticas` → `config.json` › `simulation.selection.num_scenarios`
- `tipo_simulacao_final` → `config.json` › `simulation.scenario_source.inflow.scheme`
- `considera_reamostragem_cenarios` → `config.json` › `simulation.scenario_source.inflow.scheme`
- `ano_inicial_historico` → `config.json` › `simulation.scenario_source.historical_years`
- `ano_inicio_estudo` → `config.json` › `simulation.scenario_source.historical_years`
- `num_forwards` → `config.json` › `training.selection.forward_passes`
- `num_max_iteracoes` → `config.json` › `training.stopping_rules[].limit`
- `num_aberturas` → `config.json` › `training.parallelism.backward_scheduler.block_size`
- `considera_reamostragem_cenarios` → `config.json` › `training.scenario_source.inflow.scheme`
- `num_forwards` → `config.json` › `training.scenario_source.inflow.scheme`
- `num_aberturas` → `config.json` › `training.scenario_source.inflow.scheme`
- `tipo_simulacao_final` → `config.json` › `training.scenario_source.inflow.scheme`
- `ano_inicial_historico` → `config.json` › `training.scenario_source.historical_years`
- `tipo_execucao` → `config.json` › `training.enabled`
- `taxa_de_desconto` → `stages.json` › `policy_graph.annual_discount_rate`
- `mes_inicio_estudo` → `stages.json` › `policy_graph.transitions[].source_id`
- `num_anos_estudo` → `stages.json` › `policy_graph.transitions[].source_id`
- `num_anos_pos_estudo` → `stages.json` › `policy_graph.transitions[].source_id`
- `mes_inicio_estudo` → `stages.json` › `policy_graph.transitions[].target_id`
- `num_anos_estudo` → `stages.json` › `policy_graph.transitions[].target_id`
- `num_anos_pos_estudo` → `stages.json` › `policy_graph.transitions[].target_id`
- `mes_inicio_estudo` → `stages.json` › `stages[].id`
- `ano_inicio_estudo` → `stages.json` › `stages[].id`
- `num_anos_estudo` → `stages.json` › `stages[].id`
- `num_anos_pos_estudo` → `stages.json` › `stages[].id`
- `mes_inicio_estudo` → `stages.json` › `stages[].start_date`
- `ano_inicio_estudo` → `stages.json` › `stages[].start_date`
- `mes_inicio_estudo` → `stages.json` › `stages[].end_date`
- `ano_inicio_estudo` → `stages.json` › `stages[].end_date`
- `mes_inicio_estudo` → `stages.json` › `stages[].season_id`
- `num_aberturas` → `stages.json` › `stages[].num_openings`
- `cvar` → `stages.json` › `stages[].risk_measure`
- `cvar` → `stages.json` › `stages[].risk_measure.cvar.alpha`
- `cvar` → `stages.json` › `stages[].risk_measure.cvar.lambda`
- `num_forwards` → `stages.json` › `stages[].state_variables.inflow_lags`
- `num_aberturas` → `stages.json` › `stages[].state_variables.inflow_lags`
- `tipo_simulacao_final` → `stages.json` › `stages[].state_variables.inflow_lags`
- `num_forwards` → `stages.json` › `stages[].sampling_method`
- `num_aberturas` → `stages.json` › `stages[].sampling_method`
- `tipo_simulacao_final` → `stages.json` › `stages[].sampling_method`
- `num_anos_pre_estudo` → `stages.json` › `pre_study_stages[].id`
- `num_anos_pre_estudo` → `stages.json` › `pre_study_stages[].start_date`
- `mes_inicio_estudo` → `stages.json` › `pre_study_stages[].start_date`
- `ano_inicio_estudo` → `stages.json` › `pre_study_stages[].start_date`
- `num_anos_pre_estudo` → `stages.json` › `pre_study_stages[].end_date`
- `mes_inicio_estudo` → `stages.json` › `pre_study_stages[].end_date`
- `ano_inicio_estudo` → `stages.json` › `pre_study_stages[].end_date`
- `num_anos_pre_estudo` → `stages.json` › `pre_study_stages[].season_id`
- `despacho_antecipado_gnl` → `initial_conditions.json` › `past_anticipated_commitments[].thermal_id`
- `mes_inicio_estudo` → `initial_conditions.json` › `past_anticipated_commitments[].start_date`
- `ano_inicio_estudo` → `initial_conditions.json` › `past_anticipated_commitments[].start_date`
- `mes_inicio_estudo` → `initial_conditions.json` › `past_anticipated_commitments[].end_date`
- `ano_inicio_estudo` → `initial_conditions.json` › `past_anticipated_commitments[].end_date`
- `mes_inicio_estudo` → `initial_conditions.json` › `recent_observations[].start_date`
- `ano_inicio_estudo` → `initial_conditions.json` › `recent_observations[].start_date`
- `mes_inicio_estudo` → `initial_conditions.json` › `recent_observations[].end_date`
- `ano_inicio_estudo` → `initial_conditions.json` › `recent_observations[].end_date`
- `ano_inicial_historico` → `system/buses.json` › `buses[].operational_start_date`
- `ano_inicial_historico` → `system/lines.json` › `lines[].operational_start_date`
- `mes_inicio_estudo` → `system/lines.json` › `lines[].capacity.direct_mw`
- `ano_inicio_estudo` → `system/lines.json` › `lines[].capacity.direct_mw`
- `mes_inicio_estudo` → `system/lines.json` › `lines[].capacity.reverse_mw`
- `ano_inicio_estudo` → `system/lines.json` › `lines[].capacity.reverse_mw`
- `ano_inicial_historico` → `system/thermals.json` › `thermals[].operational_start_date`
- `ano_inicio_estudo` → `system/thermals.json` › `thermals[].generation.min_mw`
- `mes_inicio_estudo` → `system/thermals.json` › `thermals[].generation.min_mw`
- `num_anos_manutencao_utes` → `system/thermals.json` › `thermals[].generation.max_mw`
- `despacho_antecipado_gnl` → `system/thermals.json` › `thermals[].anticipated_config`
- `ano_inicial_historico` → `system/non_controllable_sources.json` › `non_controllable_sources[].operational_start_date`
- `curva_aversao` → `constraints/generic_parameters.json` › `scalar_parameters[].kind`
- `mes_inicio_estudo` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `mes_inicio_estudo` → `constraints/line_bounds.parquet` › `stage_id`
- `ano_inicio_estudo` → `constraints/line_bounds.parquet` › `stage_id`
- `num_anos_estudo` → `constraints/line_bounds.parquet` › `stage_id`
- `num_anos_pos_estudo` → `constraints/line_bounds.parquet` › `stage_id`
- `mes_inicio_estudo` → `constraints/thermal_bounds.parquet` › `stage_id`
- `ano_inicio_estudo` → `constraints/thermal_bounds.parquet` › `stage_id`
- `num_anos_estudo` → `constraints/thermal_bounds.parquet` › `stage_id`
- `num_anos_pos_estudo` → `constraints/thermal_bounds.parquet` › `stage_id`
- `ano_inicio_estudo` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `mes_inicio_estudo` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `num_anos_manutencao_utes` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `mes_inicio_estudo` → `constraints/penalty_overrides_bus.parquet` › `stage_id`
- `num_anos_estudo` → `constraints/penalty_overrides_bus.parquet` › `stage_id`
- `num_anos_pos_estudo` → `constraints/penalty_overrides_bus.parquet` › `stage_id`
- `mes_inicio_estudo` → `constraints/penalty_overrides_hydro.parquet` › `stage_id`
- `num_anos_estudo` → `constraints/penalty_overrides_hydro.parquet` › `stage_id`
- `num_anos_pos_estudo` → `constraints/penalty_overrides_hydro.parquet` › `stage_id`
- `ano_inicial_historico` → `system/hydros.json` › `hydros[].operational_start_date`
- `desconsidera_vazao_minima` → `system/hydros.json` › `hydros[].outflow.min_outflow_m3s`
- `funcao_producao_uhe` → `system/hydros.json` › `hydros[].generation.model`
- `sazonaliza_cfuga_cmont` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `ano_inicio_estudo` → `system/hydros.json` › `hydros[].filling.start_stage_id`
- `mes_inicio_estudo` → `system/hydros.json` › `hydros[].filling.start_stage_id`
- `ano_inicio_estudo` → `system/hydros.json` › `hydros[].filling.filling_min_rate_m3s`
- `mes_inicio_estudo` → `system/hydros.json` › `hydros[].filling.filling_min_rate_m3s`
- `funcao_producao_uhe` → `system/hydros.json` › `hydros[].efficiency`
- `ano_inicio_estudo` → `system/hydros.json` › `hydros[].entry_stage_id`
- `mes_inicio_estudo` → `system/hydros.json` › `hydros[].entry_stage_id`
- `funcao_producao_uhe` → `system/hydro_production_models.json` › `production_models[].selection_mode`
- `funcao_producao_uhe` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].model`
- `funcao_producao_uhe` → `system/hydro_production_models.json` › `production_models[].seasons[].model`
- `ano_inicio_estudo` → `system/hydro_energy_productivity.parquet` › `stage_id`
- `mes_inicio_estudo` → `system/hydro_energy_productivity.parquet` › `stage_id`
- `num_anos_estudo` → `system/hydro_energy_productivity.parquet` › `stage_id`
- `num_anos_pos_estudo` → `system/hydro_energy_productivity.parquet` › `stage_id`
- `sazonaliza_cfuga_cmont` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `mes_inicio_estudo` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `ano_inicio_estudo` → `constraints/hydro_bounds.parquet` › `stage_id`
- `mes_inicio_estudo` → `constraints/hydro_bounds.parquet` › `stage_id`
- `num_anos_estudo` → `constraints/hydro_bounds.parquet` › `stage_id`
- `num_anos_pos_estudo` → `constraints/hydro_bounds.parquet` › `stage_id`
- `outros_usos_da_agua` → `constraints/hydro_bounds.parquet` › `water_withdrawal_m3s`
- `sazonaliza_vmint` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `sazonaliza_vmaxt` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `restricao_turbinamento` → `constraints/hydro_bounds.parquet` › `min_turbined_m3s`
- `restricao_turbinamento` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `sazonaliza_cfuga_cmont` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `desconsidera_vazao_minima` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `considera_ghmin` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `ano_inicial_historico` → `scenarios/inflow_history.parquet` › `start_date`
- `ano_inicial_historico` → `scenarios/inflow_history.parquet` › `end_date`
- `ano_inicio_estudo` → `scenarios/inflow_history.parquet` › `value_m3s`
- `mes_inicio_estudo` → `scenarios/inflow_history.parquet` › `value_m3s`
- `mes_inicio_estudo` → `scenarios/inflow_seasonal_stats.parquet` › `stage_id`
- `num_anos_estudo` → `scenarios/inflow_seasonal_stats.parquet` › `stage_id`
- `num_anos_pos_estudo` → `scenarios/inflow_seasonal_stats.parquet` › `stage_id`
- `ano_inicial_historico` → `scenarios/inflow_seasonal_stats.parquet` › `mean_m3s`
- `ano_inicio_estudo` → `scenarios/inflow_seasonal_stats.parquet` › `mean_m3s`
- `mes_inicio_estudo` → `scenarios/inflow_seasonal_stats.parquet` › `mean_m3s`
- `mes_inicio_estudo` → `scenarios/load_seasonal_stats.parquet` › `stage_id`
- `num_anos_estudo` → `scenarios/load_seasonal_stats.parquet` › `stage_id`
- `num_anos_pos_estudo` → `scenarios/load_seasonal_stats.parquet` › `stage_id`
- `considera_carga_adicional` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `mes_inicio_estudo` → `scenarios/load_factors.json` › `load_factors[].stage_id`
- `num_anos_estudo` → `scenarios/load_factors.json` › `load_factors[].stage_id`
- `num_anos_pos_estudo` → `scenarios/load_factors.json` › `load_factors[].stage_id`
- `mes_inicio_estudo` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].stage_id`
- `num_anos_estudo` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].stage_id`
- `num_anos_pos_estudo` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].stage_id`
- `mes_inicio_estudo` → `scenarios/non_controllable_stats.parquet` › `stage_id`
- `num_anos_estudo` → `scenarios/non_controllable_stats.parquet` › `stage_id`
- `num_anos_pos_estudo` → `scenarios/non_controllable_stats.parquet` › `stage_id`
- `agrupamento_livre · restricoes_eletricas · restricoes_eletricas_especiais` → `constraints/generic_constraints.json` › `constraints[].expression`
- `mes_inicio_estudo` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `num_anos_estudo` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `num_anos_pos_estudo` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `sazonaliza_cfuga_cmont` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `execução, impressão e memória (nome_caso, duracao_periodo, imprime_*, impressao_operacao/ena/convergencia/cortes_ativos_sim_final, gera_arquivo_cortes_unico, mantem_arquivos_*, alocacao_memoria_*, memoria_calculo_cortes, armazenamento_local_arquivos_temporarios, utiliza_gerenciamento_pls, comunicacao_dois_niveis, intervalo_para_gravar, tamanho_registro_arquivo_historico, consulta_fcf)` — *não lido.* Controles de relatório, disco e memória do executável NEWAVE. Não têm contraparte no caso Novomodelo.
- `convergência e gestão de cortes (tolerancia, delta_zinf, delta_zsup, deltas_consecutivos, converge_no_zero, num_minimo_iteracoes, inicio_teste_convergencia, desconsidera_convergencia_estatistica, considera_zsup_min_convergencia, iteracao_para_simulacao_final, aproveitamento_bases_backward, janela_de_cortes, periodos_manutencao_cortes, eliminacao_cortes, fcf_pos_estudo, mes_inicio_pre_estudo)` — *não lido.* O bridge fixa apenas `iteration_limit` (de `num_max_iteracoes`) e o liga/desliga da seleção de cortes; critérios de parada por gap e janelas de cortes ficam nos defaults do Novomodelo. `num_anos_pre_estudo` é convertido em `pre_study_stages`, mas `mes_inicio_pre_estudo` não é lido.
- `modelo estocástico de afluências (tipo_geracao_enas, matriz_correlacao_espacial, reducao_automatica_ordem, considera_tendencia_hidrologica_calculo_politica, considera_tendencia_hidrologica_sim_final, tipo_reamostragem_cenarios, passo_reamostragem_cenarios, momento_reamostragem, aberturas_variaveis, num_anos_pos_sim_final, agregacao_simulacao_final, simulacao_final_com_data, representacao_agregacao, el_nino, enso)` — *não lido.* O Novomodelo estima o PAR(p) a partir de `scenarios/inflow_history.parquet` com sua própria seleção de ordem (`estimation.order_selection`); só `ordem_maxima_parp`, `consideracao_media_anual_afluencias`, `num_series_sinteticas` e `considera_reamostragem_cenarios` são espelhados. A tendência hidrológica é sempre emitida (`recent_observations`) quando `vazpast.dat` existe, sem consultar os flags.
- `flags de recursos não convertidos (restricao_defluencia, restricao_itaipu, restricoes_rhq, restricoes_rhv, restricao_lpp_*, restricoes_emissao_gee, restricoes_fornecimento_gas, sar, bid, modif_automatica_adterm, sazonaliza_vminp, canal_desvio, correcao_desvio)` — *não lido.* Ativam recursos que o conversor não converte (vazão máxima `VAZMAXT`, RHQ/RHV, LPP, SAR, GEE, gás, canal de desvio, ADTERM automático), logo não há o que ligar ou desligar. Os flags de `ghmin.dat`, `c_adic.dat`, `agrint.dat`, `re.dat`, `restricao-eletrica.csv`, vazão mínima e turbinamento são honrados: um arquivo presente com o flag desligado é ignorado e reportado (diagnóstico `dger-switch-off`); linha ausente conta como ligada.
- `operação e penalidades internas (perdas_rede_transmissao, considera_geracao_eolica, penalidade_corte_geracao_eolica, estacoes_bombeamento, representacao_submotorizacao, calcula_volume_inicial, volume_inicial_subsistema, ordenacao_automatica, calcula_prodt_media_sin, racionamento_preventivo, primeira_profundidade_risco_deficit, segunda_profundidade_risco_deficit, equalizacao_penal_intercambio)` — *não lido.* As micro-penalidades de intercâmbio, vertimento, turbinamento, corte de eólica e excesso são escritas com os defaults internos do manual NEWAVE v30, não com o valor de `penalidade_corte_geracao_eolica`. O volume inicial vem de `confhd.dat` (`volume_inicial_percentual`); `calcula_volume_inicial`/`volume_inicial_subsistema` não são lidos.

### `confhd.dat`

**Estado:** convertido em parte. **Lido por:** 14 arquivos gerados; ver a visão geral.

Entram no caso as usinas em operação e não fictícias — `usina_existente` igual a `EX` ou a `EE` (existente em expansão) —, mais as usinas `NE` com registro de enchimento de volume morto em `exph.dat`. A usina `EE` opera desde o primeiro estágio e sua capacidade cresce conforme as máquinas de `exph.dat` entram.

- `usinas · codigo_usina_jusante` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `usinas · codigo_usina_jusante` → `penalties.json` › `hydro.evaporation_violation_cost`
- `usinas · codigo_usina_jusante` → `penalties.json` › `hydro.storage_violation_below_cost`
- `usinas · codigo_usina_jusante` → `penalties.json` › `hydro.filling_target_violation_cost`
- `usinas · codigo_usina` → `initial_conditions.json` › `storage[].hydro_id`
- `usinas · volume_inicial_percentual` → `initial_conditions.json` › `storage[].value_hm3`
- `usinas · usina_existente` → `initial_conditions.json` › `filling_storage`
- `usinas · codigo_usina` → `initial_conditions.json` › `filling_storage[].hydro_id`
- `usinas · usina_existente` → `initial_conditions.json` › `filling_storage[].hydro_id`
- `usinas · posto` → `initial_conditions.json` › `recent_observations[].hydro_id`
- `usinas · posto` → `initial_conditions.json` › `recent_observations[].value_m3s`
- `usinas · codigo_usina_jusante` → `initial_conditions.json` › `recent_observations[].value_m3s`
- `usinas · codigo_usina` → `constraints/generic_parameters.json` › `scalar_parameters[].id`
- `usinas · codigo_usina` → `constraints/generic_parameters.json` › `scalar_parameters[].name`
- `usinas · codigo_usina` → `constraints/generic_parameters.json` › `scalar_parameters[].computed_spec.hydro_id`
- `usinas · ree` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `usinas · codigo_usina_jusante` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `usinas · codigo_usina` → `constraints/penalty_overrides_hydro.parquet` › `hydro_id`
- `usinas · codigo_usina_jusante` → `constraints/penalty_overrides_hydro.parquet` › `storage_violation_below_cost`
- `usinas · codigo_usina_jusante` → `constraints/penalty_overrides_hydro.parquet` › `evaporation_violation_cost`
- `usinas · codigo_usina_jusante` → `constraints/penalty_overrides_hydro.parquet` › `water_withdrawal_violation_cost`
- `codigo_usina` → `system/hydros.json` › `hydros[].id`
- `usina_existente` → `system/hydros.json` › `hydros[].id`
- `posto` → `system/hydros.json` › `hydros[].id`
- `nome_usina` → `system/hydros.json` › `hydros[].name`
- `codigo_usina_jusante` → `system/hydros.json` › `hydros[].downstream_id`
- `usina_existente` → `system/hydros.json` › `hydros[].downstream_id`
- `posto` → `system/hydros.json` › `hydros[].downstream_id`
- `usina_existente` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `usina_existente` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `nome_usina` → `system/hydros.json` › `hydros[].unit_groups[].name`
- `ree` → `system/hydros.json` › `hydros[].unit_groups[].bus_id`
- `codigo_usina` → `system/hydro_production_models.json` › `production_models[].hydro_id`
- `codigo_usina` → `system/hydro_geometry.parquet` › `hydro_id`
- `codigo_usina` → `system/hydro_energy_productivity.parquet` › `hydro_id`
- `codigo_usina_jusante` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `posto` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `hydro_id`
- `codigo_usina_jusante` → `constraints/hydro_bounds.parquet` › `water_withdrawal_m3s`
- `usina_existente` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `codigo_usina` → `scenarios/inflow_history.parquet` › `hydro_id`
- `posto` → `scenarios/inflow_history.parquet` › `hydro_id`
- `posto` → `scenarios/inflow_history.parquet` › `value_m3s`
- `codigo_usina_jusante` → `scenarios/inflow_history.parquet` › `value_m3s`
- `usina_existente` → `scenarios/inflow_history.parquet` › `value_m3s`
- `codigo_usina` → `scenarios/inflow_seasonal_stats.parquet` › `hydro_id`
- `posto` → `scenarios/inflow_seasonal_stats.parquet` › `mean_m3s`
- `codigo_usina_jusante` → `scenarios/inflow_seasonal_stats.parquet` › `mean_m3s`
- `posto` → `scenarios/inflow_seasonal_stats.parquet` › `std_m3s`
- `codigo_usina_jusante` → `scenarios/inflow_seasonal_stats.parquet` › `std_m3s`
- `ree` → `constraints/generic_constraints.json` › `constraints[].expression`
- `codigo_usina` → `constraints/generic_constraints.json` › `constraints[].expression`
- `codigo_usina_jusante` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `usinas fictícias (produtibilidade_especifica = 0 no mesmo posto de uma usina geradora)` — *adiado.* Excluídas do mapa de ids com o diagnóstico `fictitious-plants-excluded`. São nós contábeis de cascata energética; o bridge preserva a topologia religando `downstream_id` à próxima usina real e somando a produtibilidade da cadeia fictícia à usina de montante.
- `usinas NE sem registro de enchimento em exph.dat e usinas NC` — *não lido.* Não entram no LP; a cascata é religada através delas (log informativo apenas). Converter uma expansão sem enchimento exigiria `entry_stage_id` sem bloco `filling`.
- `usina_modificada` — *não lido.* Indica se a usina tem registros em `modif.dat`; o bridge aplica os registros de `modif.dat` a toda usina cadastrada, sem consultar o flag.
- `ano_inicio_historico · ano_fim_historico` — *não lido.* Janela do histórico de vazões por usina. O bridge usa o histórico completo de `vazoes.dat` desde `ano_inicial_historico` até o mês anterior ao início do estudo para todas as usinas.

### `conft.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/thermal_bounds.parquet`, `initial_conditions.json`, `system/thermals.json`.

- `usinas · codigo_usina` → `system/thermals.json` › `thermals[].id`
- `usinas · nome_usina` → `system/thermals.json` › `thermals[].name`
- `usinas · submercado` → `system/thermals.json` › `thermals[].bus_id`
- `usinas · usina_existente` → `system/thermals.json` › `thermals[].generation.min_mw`
- `usinas · usina_existente` → `system/thermals.json` › `thermals[].generation.max_mw`
- `usinas · codigo_usina` → `constraints/thermal_bounds.parquet` › `thermal_id`
- `usinas · usina_existente` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `usinas · usina_existente` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `usina_existente · classe` — *não lido.* Toda usina listada vira uma térmica do Novomodelo; o estado `EX`/`NE`/`EE` e a classe térmica não são consultados. A disponibilidade efetiva vem das janelas POTEF de `expt.dat`.

### `sistema.dat`

**Estado:** convertido em parte. **Lido por:** 15 arquivos gerados; ver a visão geral.

- `custo_deficit · custo` → `penalties.json` › `bus.deficit_segments[].cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.generation_violation_below_cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.evaporation_violation_cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.storage_violation_below_cost`
- `custo_deficit · custo` → `penalties.json` › `hydro.filling_target_violation_cost`
- `custo_deficit · codigo_submercado` → `system/buses.json` › `buses[].id`
- `custo_deficit · nome_submercado` → `system/buses.json` › `buses[].name`
- `custo_deficit · custo` → `system/buses.json` › `buses[].deficit_segments[].cost`
- `custo_deficit · patamar_deficit` → `system/buses.json` › `buses[].deficit_segments[].cost`
- `custo_deficit · corte` → `system/buses.json` › `buses[].deficit_segments[].depth_mw`
- `limites_intercambio · submercado_de` → `system/lines.json` › `lines[].id`
- `limites_intercambio · submercado_para` → `system/lines.json` › `lines[].id`
- `limites_intercambio · submercado_de` → `system/lines.json` › `lines[].name`
- `limites_intercambio · submercado_para` → `system/lines.json` › `lines[].name`
- `limites_intercambio · submercado_de` → `system/lines.json` › `lines[].source_bus_id`
- `limites_intercambio · submercado_para` → `system/lines.json` › `lines[].target_bus_id`
- `limites_intercambio · valor` → `system/lines.json` › `lines[].capacity.direct_mw`
- `limites_intercambio · sentido` → `system/lines.json` › `lines[].capacity.direct_mw`
- `limites_intercambio · valor` → `system/lines.json` › `lines[].capacity.reverse_mw`
- `limites_intercambio · sentido` → `system/lines.json` › `lines[].capacity.reverse_mw`
- `custo_deficit · ficticio` → `system/lines.json` › `lines[].exchange_cost`
- `geracao_usinas_nao_simuladas · codigo_submercado` → `system/non_controllable_sources.json` › `non_controllable_sources[].id`
- `geracao_usinas_nao_simuladas · indice_bloco` → `system/non_controllable_sources.json` › `non_controllable_sources[].id`
- `geracao_usinas_nao_simuladas · fonte` → `system/non_controllable_sources.json` › `non_controllable_sources[].name`
- `geracao_usinas_nao_simuladas · codigo_submercado` → `system/non_controllable_sources.json` › `non_controllable_sources[].name`
- `geracao_usinas_nao_simuladas · codigo_submercado` → `system/non_controllable_sources.json` › `non_controllable_sources[].bus_id`
- `geracao_usinas_nao_simuladas · valor` → `system/non_controllable_sources.json` › `non_controllable_sources[].max_generation_mw`
- `limites_intercambio · submercado_de` → `constraints/line_bounds.parquet` › `line_id`
- `limites_intercambio · submercado_para` → `constraints/line_bounds.parquet` › `line_id`
- `limites_intercambio · valor` → `constraints/line_bounds.parquet` › `direct_mw`
- `limites_intercambio · sentido` → `constraints/line_bounds.parquet` › `direct_mw`
- `limites_intercambio · valor` → `constraints/line_bounds.parquet` › `reverse_mw`
- `limites_intercambio · sentido` → `constraints/line_bounds.parquet` › `reverse_mw`
- `custo_deficit · ficticio` → `constraints/penalty_overrides_bus.parquet` › `bus_id`
- `custo_deficit · codigo_submercado` → `constraints/penalty_overrides_bus.parquet` › `bus_id`
- `custo_deficit · custo` → `constraints/penalty_overrides_bus.parquet` › `excess_cost`
- `custo_deficit · custo` → `constraints/penalty_overrides_hydro.parquet` › `turbined_violation_below_cost`
- `custo_deficit · custo` → `constraints/penalty_overrides_hydro.parquet` › `outflow_violation_below_cost`
- `custo_deficit · custo` → `constraints/penalty_overrides_hydro.parquet` › `outflow_violation_above_cost`
- `custo_deficit · custo` → `constraints/penalty_overrides_hydro.parquet` › `storage_violation_below_cost`
- `custo_deficit · custo` → `constraints/penalty_overrides_hydro.parquet` › `evaporation_violation_cost`
- `custo_deficit · custo` → `constraints/penalty_overrides_hydro.parquet` › `water_withdrawal_violation_cost`
- `custo_deficit · custo` → `constraints/penalty_overrides_hydro.parquet` › `inflow_nonnegativity_cost`
- `custo_deficit · codigo_submercado` → `system/hydros.json` › `hydros[].unit_groups[].bus_id`
- `mercado_energia · codigo_submercado` → `scenarios/load_seasonal_stats.parquet` › `bus_id`
- `mercado_energia · valor` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `mercado_energia · data` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `geracao_usinas_nao_simuladas · codigo_submercado` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].ncs_id`
- `geracao_usinas_nao_simuladas · indice_bloco` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].ncs_id`
- `geracao_usinas_nao_simuladas · codigo_submercado` → `scenarios/non_controllable_stats.parquet` › `ncs_id`
- `geracao_usinas_nao_simuladas · indice_bloco` → `scenarios/non_controllable_stats.parquet` › `ncs_id`
- `geracao_usinas_nao_simuladas · valor` → `scenarios/non_controllable_stats.parquet` › `mean`
- `geracao_usinas_nao_simuladas · data` → `scenarios/non_controllable_stats.parquet` › `mean`
- `limites_intercambio · submercado_de` → `constraints/generic_constraints.json` › `constraints[].expression`
- `limites_intercambio · submercado_para` → `constraints/generic_constraints.json` › `constraints[].expression`
- `custo_deficit · custo` → `constraints/generic_constraints.json` › `constraints[].slack.enabled`
- `custo_deficit · custo` → `constraints/generic_constraints.json` › `constraints[].slack.penalty`
- `numero_patamares_deficit` — *não lido.* Os segmentos de déficit são derivados das linhas de `custo_deficit` (`patamar_deficit`, `custo`, `corte`); a contagem declarada não é lida.
- `limites_intercambio · flag` — *não lido.* Coluna de flag do bloco de limites de intercâmbio; o bridge usa apenas `submercado_de`, `submercado_para`, `sentido`, `data` e `valor`.

### `clast.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/thermal_bounds.parquet`, `system/thermals.json`.

- `usinas · valor (indice_ano_estudo = 1)` → `system/thermals.json` › `thermals[].cost_per_mwh`
- `usinas · valor` → `constraints/thermal_bounds.parquet` › `cost_per_mwh`
- `usinas · indice_ano_estudo` → `constraints/thermal_bounds.parquet` › `cost_per_mwh`
- `modificacoes · custo` → `constraints/thermal_bounds.parquet` › `cost_per_mwh`
- `modificacoes · data_inicio` → `constraints/thermal_bounds.parquet` › `cost_per_mwh`
- `modificacoes · data_fim` → `constraints/thermal_bounds.parquet` › `cost_per_mwh`
- `tipo_combustivel` — *não lido.* Combustível da classe térmica. O Novomodelo não distingue combustível; só o custo por ano de estudo e as modificações por data são convertidos.

### `term.dat`

**Estado:** convertido. **Lido por:** `constraints/thermal_bounds.parquet`, `initial_conditions.json`, `system/thermals.json`.

Todas as colunas de `usinas` são lidas (potência, FCMAX, TEIF, IP, GTMIN por mês do primeiro ano de estudo e GTMIN dos demais anos). Para usina `EE` ou `NE` em `conft.dat`, o NEWAVE descarta o GTMIN de `term.dat` e só valem as janelas GTMIN de `expt.dat`.

- `usinas · potencia_instalada` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `usinas · fator_capacidade_maximo` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `usinas · geracao_minima` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `usinas · geracao_minima (mes 1 a 12)` → `system/thermals.json` › `thermals[].generation.min_mw`
- `usinas · geracao_minima (mes = 13)` → `system/thermals.json` › `thermals[].generation.min_mw`
- `usinas · potencia_instalada` → `system/thermals.json` › `thermals[].generation.max_mw`
- `usinas · fator_capacidade_maximo` → `system/thermals.json` › `thermals[].generation.max_mw`
- `usinas · teif` → `system/thermals.json` › `thermals[].generation.max_mw`
- `usinas · indisponibilidade_programada` → `system/thermals.json` › `thermals[].generation.max_mw`
- `usinas · geracao_minima (mes 1 a 12)` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `usinas · geracao_minima (mes = 13)` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `usinas · potencia_instalada` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `usinas · fator_capacidade_maximo` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `usinas · teif` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `usinas · indisponibilidade_programada` → `constraints/thermal_bounds.parquet` › `max_generation_mw`

### `ree.dat`

**Estado:** convertido em parte. **Lido por:** 13 arquivos gerados; ver a visão geral.

- `rees · submercado` → `system/buses.json` › `buses[].id`
- `codigo` → `system/hydros.json` › `hydros[].unit_groups[].bus_id`
- `submercado` → `system/hydros.json` › `hydros[].unit_groups[].bus_id`
- `nome` → `constraints/generic_constraints.json` › `constraints[].name`
- `nome` → `constraints/generic_constraints.json` › `constraints[].description`
- `mes_fim_individualizado` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `ano_fim_individualizado` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `remocao_ficticias` — *não lido.* Flag de remoção de usinas fictícias do REE; o bridge identifica as fictícias estruturalmente (posto compartilhado e ρ_esp = 0), sem consultar o flag.

### `patamar.dat`

**Estado:** convertido. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/line_bounds.parquet`, `initial_conditions.json`, `scenarios/load_factors.json`, `scenarios/non_controllable_factors.json`, `stages.json`, `system/thermals.json`.

- `numero_patamares` → `stages.json` › `stages[].blocks[].id`
- `numero_patamares` → `stages.json` › `stages[].blocks[].name`
- `duracao_mensal_patamares · valor` → `stages.json` › `stages[].blocks[].hours`
- `duracao_mensal_patamares · valor` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `intercambio_patamares · patamar` → `constraints/line_bounds.parquet` › `block_id`
- `intercambio_patamares · valor` → `constraints/line_bounds.parquet` › `direct_mw`
- `intercambio_patamares · valor` → `constraints/line_bounds.parquet` › `reverse_mw`
- `carga_patamares · codigo_submercado` → `scenarios/load_factors.json` › `load_factors[].bus_id`
- `carga_patamares · patamar` → `scenarios/load_factors.json` › `load_factors[].block_factors[].block_id`
- `numero_patamares` → `scenarios/load_factors.json` › `load_factors[].block_factors[].block_id`
- `carga_patamares · valor` → `scenarios/load_factors.json` › `load_factors[].block_factors[].factor`
- `carga_patamares · data` → `scenarios/load_factors.json` › `load_factors[].block_factors[].factor`
- `usinas_nao_simuladas · patamar` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].block_id`
- `numero_patamares` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].block_id`
- `usinas_nao_simuladas · valor` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].factor`
- `usinas_nao_simuladas · data` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].factor`
- `usinas_nao_simuladas · codigo_submercado` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].factor`
- `usinas_nao_simuladas · indice_bloco` → `scenarios/non_controllable_factors.json` › `non_controllable_factors[].block_factors[].factor`
- `numero_patamares` → `constraints/generic_constraint_bounds.parquet` › `block_id`

### `hidr.dat`

**Estado:** convertido em parte. **Lido por:** 14 arquivos gerados; ver a visão geral.

Cadastro binário lido integralmente pelo `inewave`; as colunas abaixo não alimentam nenhuma saída.

- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.spillage_cost`
- `cadastro · volume_minimo` → `penalties.json` › `hydro.spillage_cost`
- `cadastro · volume_maximo` → `penalties.json` › `hydro.spillage_cost`
- `cadastro · canal_fuga_medio` → `penalties.json` › `hydro.spillage_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.turbined_cost`
- `cadastro · volume_minimo` → `penalties.json` › `hydro.turbined_cost`
- `cadastro · volume_maximo` → `penalties.json` › `hydro.turbined_cost`
- `cadastro · canal_fuga_medio` → `penalties.json` › `hydro.turbined_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.diversion_cost`
- `cadastro · volume_minimo` → `penalties.json` › `hydro.diversion_cost`
- `cadastro · volume_maximo` → `penalties.json` › `hydro.diversion_cost`
- `cadastro · canal_fuga_medio` → `penalties.json` › `hydro.diversion_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.evaporation_violation_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.storage_violation_below_cost`
- `cadastro · produtibilidade_especifica` → `penalties.json` › `hydro.filling_target_violation_cost`
- `cadastro · volume_minimo` → `initial_conditions.json` › `storage[].value_hm3`
- `cadastro · volume_maximo` → `initial_conditions.json` › `storage[].value_hm3`
- `cadastro · tipo_regulacao` → `initial_conditions.json` › `storage[].value_hm3`
- `cadastro · volume_referencia` → `initial_conditions.json` › `storage[].value_hm3`
- `cadastro · volume_minimo` → `initial_conditions.json` › `filling_storage[].value_hm3`
- `cadastro · produtibilidade_especifica` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `cadastro · a0_volume_cota … a4_volume_cota` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `cadastro · canal_fuga_medio` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `cadastro · perdas` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `cadastro · tipo_perda` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `cadastro · volume_minimo` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `cadastro · volume_maximo` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `cadastro · tipo_regulacao` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `cadastro · produtibilidade_especifica` → `constraints/penalty_overrides_hydro.parquet` › `spillage_cost`
- `cadastro · produtibilidade_especifica` → `constraints/penalty_overrides_hydro.parquet` › `turbined_cost`
- `cadastro · produtibilidade_especifica` → `constraints/penalty_overrides_hydro.parquet` › `diversion_cost`
- `cadastro · produtibilidade_especifica` → `constraints/penalty_overrides_hydro.parquet` › `storage_violation_below_cost`
- `cadastro · produtibilidade_especifica` → `constraints/penalty_overrides_hydro.parquet` › `evaporation_violation_cost`
- `cadastro · produtibilidade_especifica` → `constraints/penalty_overrides_hydro.parquet` › `water_withdrawal_violation_cost`
- `cadastro · produtibilidade_especifica` → `constraints/penalty_overrides_hydro.parquet` › `inflow_nonnegativity_cost`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].id`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].downstream_id`
- `volume_minimo` → `system/hydros.json` › `hydros[].reservoir.min_storage_hm3`
- `tipo_regulacao` → `system/hydros.json` › `hydros[].reservoir.min_storage_hm3`
- `volume_referencia` → `system/hydros.json` › `hydros[].reservoir.min_storage_hm3`
- `volume_maximo` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `volume_minimo` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `tipo_regulacao` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `volume_referencia` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `vazao_minima_historica` → `system/hydros.json` › `hydros[].outflow.min_outflow_m3s`
- `a0_volume_cota … a4_volume_cota` → `system/hydros.json` › `hydros[].generation.model`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].generation.model`
- `numero_conjuntos_maquinas` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `maquinas_conjunto_1 … maquinas_conjunto_5` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `vazao_nominal_conjunto_1 … vazao_nominal_conjunto_5` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `queda_nominal_conjunto_1 … queda_nominal_conjunto_5` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `teif` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `ip` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `tipo_turbina` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `tipo_regulacao` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `volume_minimo` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `volume_maximo` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `volume_referencia` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `a0_volume_cota … a4_volume_cota` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `canal_fuga_medio` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `tipo_perda` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `perdas` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `numero_conjuntos_maquinas` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `maquinas_conjunto_1 … maquinas_conjunto_5` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `numero_conjuntos_maquinas` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `maquinas_conjunto_1 … maquinas_conjunto_5` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `numero_conjuntos_maquinas` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `maquinas_conjunto_1 … maquinas_conjunto_5` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `vazao_nominal_conjunto_1 … vazao_nominal_conjunto_5` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `queda_nominal_conjunto_1 … queda_nominal_conjunto_5` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `teif` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `ip` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `tipo_turbina` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `a0_volume_cota … a4_volume_cota` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `canal_fuga_medio` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].specific_productivity_mw_per_m3s_per_m`
- `evaporacao_JAN … evaporacao_DEZ` → `system/hydros.json` › `hydros[].evaporation`
- `evaporacao_JAN … evaporacao_DEZ` → `system/hydros.json` › `hydros[].evaporation.coefficients_mm[]`
- `volume_minimo` → `system/hydros.json` › `hydros[].evaporation.reference_volumes_hm3[]`
- `volume_maximo` → `system/hydros.json` › `hydros[].evaporation.reference_volumes_hm3[]`
- `canal_fuga_medio` → `system/hydros.json` › `hydros[].tailrace.coefficients[]`
- `volume_minimo` → `system/hydros.json` › `hydros[].filling.filling_min_rate_m3s`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].efficiency`
- `a0_volume_cota … a4_volume_cota` → `system/hydros.json` › `hydros[].efficiency`
- `produtibilidade_especifica` → `system/hydros.json` › `hydros[].efficiency.value`
- `tipo_perda` → `system/hydros.json` › `hydros[].hydraulic_losses`
- `perdas` → `system/hydros.json` › `hydros[].hydraulic_losses`
- `tipo_perda` → `system/hydros.json` › `hydros[].hydraulic_losses.type`
- `perdas` → `system/hydros.json` › `hydros[].hydraulic_losses.value`
- `perdas` → `system/hydros.json` › `hydros[].hydraulic_losses.value_m`
- `a0_volume_cota … a4_volume_cota` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].model`
- `produtibilidade_especifica` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].model`
- `tipo_regulacao` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `volume_minimo` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `volume_referencia` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `tipo_regulacao` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `volume_maximo` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `volume_referencia` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `a0_volume_cota … a4_volume_cota` → `system/hydro_production_models.json` › `production_models[].seasons[].model`
- `produtibilidade_especifica` → `system/hydro_production_models.json` › `production_models[].seasons[].model`
- `volume_minimo` → `system/hydro_production_models.json` › `production_models[].seasons[].reference_volume.volume_hm3`
- `volume_maximo` → `system/hydro_production_models.json` › `production_models[].seasons[].reference_volume.volume_hm3`
- `tipo_regulacao` → `system/hydro_production_models.json` › `production_models[].seasons[].fpha_config.fitting_window.volume_min_hm3`
- `volume_minimo` → `system/hydro_production_models.json` › `production_models[].seasons[].fpha_config.fitting_window.volume_min_hm3`
- `volume_referencia` → `system/hydro_production_models.json` › `production_models[].seasons[].fpha_config.fitting_window.volume_min_hm3`
- `tipo_regulacao` → `system/hydro_production_models.json` › `production_models[].seasons[].fpha_config.fitting_window.volume_max_hm3`
- `volume_maximo` → `system/hydro_production_models.json` › `production_models[].seasons[].fpha_config.fitting_window.volume_max_hm3`
- `volume_referencia` → `system/hydro_production_models.json` › `production_models[].seasons[].fpha_config.fitting_window.volume_max_hm3`
- `volume_minimo` → `system/hydro_geometry.parquet` › `volume_hm3`
- `volume_maximo` → `system/hydro_geometry.parquet` › `volume_hm3`
- `a0_volume_cota … a4_volume_cota` → `system/hydro_geometry.parquet` › `height_m`
- `a0_cota_area … a4_cota_area` → `system/hydro_geometry.parquet` › `area_km2`
- `produtibilidade_especifica` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `a0_volume_cota … a4_volume_cota` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `canal_fuga_medio` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `tipo_perda` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `perdas` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `tipo_regulacao` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `volume_minimo` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `volume_maximo` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `volume_referencia` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `volume_minimo` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `volume_maximo` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `volume_minimo` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `volume_maximo` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `maquinas_conjunto_1 … maquinas_conjunto_5` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `vazao_nominal_conjunto_1 … vazao_nominal_conjunto_5` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `queda_nominal_conjunto_1 … queda_nominal_conjunto_5` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `teif` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `ip` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `tipo_turbina` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `produtibilidade_especifica` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `maquinas_conjunto_1 … maquinas_conjunto_5` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `potencia_nominal_conjunto_1 … potencia_nominal_conjunto_5` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `tipo_regulacao` → `constraints/generic_constraints.json` › `constraints[].expression`
- `volume_minimo` → `constraints/generic_constraints.json` › `constraints[].expression`
- `volume_maximo` → `constraints/generic_constraints.json` › `constraints[].expression`
- `produtibilidade_especifica` → `constraints/generic_constraints.json` › `constraints[].expression`
- `volume_minimo` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `volume_maximo` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `tipo_regulacao` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `produtibilidade_especifica` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `a0_volume_cota..a4_volume_cota` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `canal_fuga_medio` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `tipo_perda` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `perdas` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `volume_referencia` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `posto · submercado · codigo_usina_jusante (do cadastro)` — *não lido.* Posto, submercado e jusante são tomados de `confhd.dat` (jusante) e de `ree.dat` via o REE de `confhd.dat` (barra); as colunas homônimas do cadastro são ignoradas.
- `numero_polinomios_jusante · a0..a4_jusante_1..6 · referencia_jusante_1..6` — *não lido.* Polinômios cota de jusante × defluência do cadastro. O bridge escreve `tailrace` como constante `canal_fuga_medio` em `hydros.json` e, quando há `polinjus.csv`, as famílias por partes em `system/tailrace_curves.parquet`; os polinômios do `hidr.dat` não são usados.
- `influencia_vertimento_canal_fuga` — *não lido.* Flag de influência do vertimento no canal de fuga. O Novomodelo avalia as curvas de jusante sobre a defluência total; o flag não é consultado.
- `fator_carga_maximo · fator_carga_minimo` — *não lido.* Fatores de carga da usina hidráulica; o limite de turbinamento vem do engolimento corrigido pela queda e o de geração da potência nominal dos conjuntos.
- `volume_vertedouro · volume_desvio · cota_minima · cota_maxima · desvio` — *não lido.* Volumes de vertedouro e de desvio, cotas limite e o código de desvio. O Novomodelo não modela vertedouro nem canal de desvio (`diversion` é sempre nulo); as cotas são avaliadas pelo polinômio cota × volume.
- `numero_unidades_base · representacao_conjunto · empresa · observacao · data` — *não lido.* Dados cadastrais sem contraparte: unidades-base para submotorização, forma de representação dos conjuntos, empresa proprietária e observações.

### `vazoes.dat`

**Estado:** convertido. **Lido por:** `scenarios/inflow_history.parquet`, `scenarios/inflow_seasonal_stats.parquet`.

- `1..N (coluna por posto)` → `scenarios/inflow_history.parquet` › `value_m3s`
- `1..N (coluna por posto)` → `scenarios/inflow_seasonal_stats.parquet` › `mean_m3s`
- `1..N (coluna por posto)` → `scenarios/inflow_seasonal_stats.parquet` › `std_m3s`

### `modif.dat`

**Estado:** convertido em parte. **Lido por:** 11 arquivos gerados; ver a visão geral.

Registros permanentes `VAZMIN`, `VOLMAX`, `VOLMIN`, `NUMCNJ`, `NUMMAQ`, `POTEFE`, `VOLCOTA`, `COTAREA` são aplicados ao cadastro; `VMAXT`, `VMINT`, `TURBMAXT`, `TURBMINT`, `VAZMINT` viram limites por estágio e `CFUGA`/`CMONT` alteram a produtibilidade por estágio.

- `VOLMIN · volume` → `initial_conditions.json` › `storage[].value_hm3`
- `VOLMAX · volume` → `initial_conditions.json` › `storage[].value_hm3`
- `VOLCOTA · coeficientes a0 a a4` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `CFUGA` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `CMONT` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `VOLMIN · volume` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `VOLMAX · volume` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `CFUGA` → `constraints/penalty_overrides_hydro.parquet` › `spillage_cost`
- `CMONT` → `constraints/penalty_overrides_hydro.parquet` › `spillage_cost`
- `CFUGA` → `constraints/penalty_overrides_hydro.parquet` › `turbined_cost`
- `CMONT` → `constraints/penalty_overrides_hydro.parquet` › `turbined_cost`
- `CFUGA` → `constraints/penalty_overrides_hydro.parquet` › `diversion_cost`
- `CMONT` → `constraints/penalty_overrides_hydro.parquet` › `diversion_cost`
- `CFUGA` → `constraints/penalty_overrides_hydro.parquet` › `turbined_violation_below_cost`
- `CMONT` → `constraints/penalty_overrides_hydro.parquet` › `turbined_violation_below_cost`
- `CFUGA` → `constraints/penalty_overrides_hydro.parquet` › `outflow_violation_below_cost`
- `CMONT` → `constraints/penalty_overrides_hydro.parquet` › `outflow_violation_below_cost`
- `CFUGA` → `constraints/penalty_overrides_hydro.parquet` › `outflow_violation_above_cost`
- `CMONT` → `constraints/penalty_overrides_hydro.parquet` › `outflow_violation_above_cost`
- `VOLMIN · volume, unidade` → `system/hydros.json` › `hydros[].reservoir.min_storage_hm3`
- `VOLMAX · volume, unidade` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `VOLMIN · volume` → `system/hydros.json` › `hydros[].reservoir.max_storage_hm3`
- `VAZMIN · vazao` → `system/hydros.json` › `hydros[].outflow.min_outflow_m3s`
- `VOLCOTA · coeficientes a0 a a4` → `system/hydros.json` › `hydros[].generation.model`
- `POTEFE · potencia, conjunto` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `VOLCOTA · coeficientes a0 a a4` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `NUMCNJ · numero` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `NUMMAQ · numero_maquinas` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `VOLMIN · volume` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `VOLMAX · volume` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `CFUGA · nivel` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `CMONT · nivel` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `POTEFE · potencia, conjunto` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `NUMCNJ · numero` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `NUMMAQ · numero_maquinas` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `POTEFE · potencia, conjunto` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `NUMCNJ · numero` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `NUMMAQ · numero_maquinas` → `system/hydros.json` › `hydros[].unit_groups[].max_generation_mw`
- `POTEFE · potencia, conjunto` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `VOLCOTA · coeficientes a0 a a4` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `CFUGA · nivel` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `CMONT · nivel` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `VOLCOTA · coeficientes a0 a a4` → `system/hydros.json` › `hydros[].efficiency`
- `VOLCOTA · coeficientes a0 a a4` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].model`
- `VOLMIN · volume` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_min_hm3`
- `VOLMAX · volume` → `system/hydro_production_models.json` › `production_models[].stage_ranges[].fpha_config.fitting_window.volume_max_hm3`
- `VOLCOTA · coeficientes a0 a a4` → `system/hydro_production_models.json` › `production_models[].seasons[].model`
- `VOLMIN · volume` → `system/hydro_production_models.json` › `production_models[].seasons[].reference_volume.volume_hm3`
- `VOLMAX · volume` → `system/hydro_production_models.json` › `production_models[].seasons[].reference_volume.volume_hm3`
- `VOLMIN · volume` → `system/hydro_production_models.json` › `production_models[].seasons[].fpha_config.fitting_window.volume_min_hm3`
- `VOLMAX · volume` → `system/hydro_production_models.json` › `production_models[].seasons[].fpha_config.fitting_window.volume_max_hm3`
- `VOLMIN · volume` → `system/hydro_geometry.parquet` › `volume_hm3`
- `VOLMAX · volume` → `system/hydro_geometry.parquet` › `volume_hm3`
- `VOLCOTA · coeficientes a0 a a4` → `system/hydro_geometry.parquet` › `height_m`
- `COTAREA · coeficientes a0 a a4` → `system/hydro_geometry.parquet` › `area_km2`
- `CFUGA · data_inicio` → `system/hydro_energy_productivity.parquet` › `stage_id`
- `CMONT · data_inicio` → `system/hydro_energy_productivity.parquet` › `stage_id`
- `VOLCOTA · coeficientes a0 a a4` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `VOLMIN · volume` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `VOLMAX · volume` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `CFUGA · nivel` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `CMONT · nivel` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `VMAXT · data_inicio` → `constraints/hydro_bounds.parquet` › `stage_id`
- `VMINT · volume, unidade` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `VMINT · data_inicio` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `VOLMIN · volume` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `VOLMAX · volume` → `constraints/hydro_bounds.parquet` › `min_storage_hm3`
- `VMAXT · volume, unidade` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `VMAXT · data_inicio` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `VOLMIN · volume` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `VOLMAX · volume` → `constraints/hydro_bounds.parquet` › `max_storage_hm3`
- `TURBMINT · turbinamento` → `constraints/hydro_bounds.parquet` › `min_turbined_m3s`
- `TURBMINT · data_inicio` → `constraints/hydro_bounds.parquet` › `min_turbined_m3s`
- `TURBMAXT · turbinamento` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `TURBMAXT · data_inicio` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `CFUGA · nivel` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `CMONT · nivel` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `POTEFE · potencia, conjunto` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `VAZMINT · vazao` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `VAZMINT · data_inicio` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `VAZMINT · periodo` → `constraints/hydro_bounds.parquet` › `min_outflow_m3s`
- `POTEFE · potencia, conjunto` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `NUMCNJ · numero` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `NUMMAQ · numero_maquinas` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `VOLMAX · volume` → `constraints/generic_constraints.json` › `constraints[].expression`
- `VOLMIN · volume` → `constraints/generic_constraints.json` › `constraints[].expression`
- `VOLCOTA · coeficientes a0 a a4` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `VOLMAX · volume` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `VOLMIN · volume` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `CFUGA · nivel` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `CMONT · nivel` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `VMINP` — *adiado.* Volume mínimo com penalidade (meta operativa suave por estágio). Reportado por `modif-permanent-override-unsupported` e ignorado; converter exigiria uma restrição genérica com folga sobre `hydro_storage_final` penalizada pelo custo VOLMIN de `penalid.dat`, honrando `sazonaliza_vminp`.
- `VAZMAXT` — *adiado.* Defluência máxima por período. Reportado por `modif-permanent-override-unsupported` e ignorado; o Novomodelo aceita `max_outflow_m3s`, mas o bridge escreve sempre nulo em `hydros.json` e não emite coluna correspondente em `constraints/hydro_bounds.parquet`.

### `ghmin.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/hydro_bounds.parquet`.

- `data` → `constraints/hydro_bounds.parquet` › `stage_id`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `data` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `patamar` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `geracao` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `linhas com patamar ≠ 0` — *não lido.* Só as linhas de patamar 0 (média dos patamares) alimentam `min_generation_mw` em `constraints/hydro_bounds.parquet`; valores por patamar de carga são descartados porque a tabela é por estágio, não por bloco.

### `penalid.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/penalty_overrides_hydro.parquet`, `penalties.json`.

O bridge toma, por variável, o primeiro valor `R$/MWh` não nulo do primeiro patamar de penalidade e o aplica ao SIN inteiro após multiplicar pela produtibilidade média ou máxima acumulada.

- `TURBMN · valor_R$_MWh` → `penalties.json` › `hydro.turbined_violation_below_cost`
- `VAZMIN · valor_R$_MWh` → `penalties.json` › `hydro.outflow_violation_below_cost`
- `TURBMX · valor_R$_MWh` → `penalties.json` › `hydro.outflow_violation_above_cost`
- `GHMIN · valor_R$_MWh` → `penalties.json` › `hydro.generation_violation_below_cost`
- `DESVIO · valor_R$_MWh` → `penalties.json` › `hydro.water_withdrawal_violation_cost`
- `DESVIO · valor_R$_MWh` → `penalties.json` › `hydro.inflow_nonnegativity_cost`
- `VOLMIN · valor_R$_MWh` → `penalties.json` › `hydro.storage_violation_below_cost`
- `TURBMN · valor_R$_MWh` → `constraints/penalty_overrides_hydro.parquet` › `turbined_violation_below_cost`
- `VAZMIN · valor_R$_MWh` → `constraints/penalty_overrides_hydro.parquet` › `outflow_violation_below_cost`
- `TURBMX · valor_R$_MWh` → `constraints/penalty_overrides_hydro.parquet` › `outflow_violation_above_cost`
- `VOLMIN · valor_R$_MWh` → `constraints/penalty_overrides_hydro.parquet` › `storage_violation_below_cost`
- `DESVIO · valor_R$_MWh` → `constraints/penalty_overrides_hydro.parquet` › `water_withdrawal_violation_cost`
- `DESVIO · valor_R$_MWh` → `constraints/penalty_overrides_hydro.parquet` › `inflow_nonnegativity_cost`
- `ELETRI · valor_R$_MWh` → `constraints/generic_constraints.json` › `constraints[].slack.enabled`
- `ELETRI · valor_R$_MWh` → `constraints/generic_constraints.json` › `constraints[].slack.penalty`
- `codigo_ree_submercado` — *não lido.* Diferenciação da penalidade por REE ou submercado. O leitor por REE existe no código mas não é chamado; `penalties.json` recebe um único valor por variável.
- `patamar_penalidade = 2` — *não lido.* Segundo patamar de penalidade (custo por violação adicional). O Novomodelo tem uma única folga linear por variável; só o primeiro patamar é lido.
- `patamar_carga · valor_R$_hm3` — *não lido.* Penalidade específica por patamar de carga e o valor já convertido em R$/hm³. O bridge parte sempre de `valor_R$_MWh` e faz a própria conversão de unidades.
- `variáveis além de DESVIO, VAZMIN, GHMIN, TURBMN, TURBMX, VOLMIN e ELETRI` — *não lido.* Qualquer outra variável presente (por exemplo penalidades de intercâmbio ou de excesso) é ignorada; essas micro-penalidades são escritas com os defaults internos do NEWAVE.

### `vazpast.dat`

**Estado:** convertido. **Lido por:** `initial_conditions.json`.

Lido integralmente: as vazões naturais por posto são convertidas em incrementais e recortadas aos 12 meses anteriores ao início do estudo para `initial_conditions.recent_observations`.

- `tendencia · codigo_usina` → `initial_conditions.json` › `recent_observations[].hydro_id`
- `tendencia · mes` → `initial_conditions.json` › `recent_observations[].start_date`
- `tendencia · mes` → `initial_conditions.json` › `recent_observations[].end_date`
- `tendencia · valor` → `initial_conditions.json` › `recent_observations[].value_m3s`

### `dsvagua.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/hydro_bounds.parquet`.

Ignorado por inteiro quando `outros_usos_da_agua = 0` em `dger.dat`.

- `data` → `constraints/hydro_bounds.parquet` › `stage_id`
- `codigo_usina` → `constraints/hydro_bounds.parquet` › `water_withdrawal_m3s`
- `data` → `constraints/hydro_bounds.parquet` › `water_withdrawal_m3s`
- `valor` → `constraints/hydro_bounds.parquet` › `water_withdrawal_m3s`
- `considera_desvio_usina_NC` — *não lido.* Flag que decide se o desvio de uma usina `NC` é aplicado à usina existente a jusante. O bridge propaga sempre o desvio de uma usina fora do LP para a primeira usina existente a jusante, sem consultar o flag.
- `comentario` — *não lido.* Texto livre da linha; sem contraparte.

### `curva.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`, `constraints/generic_parameters.json`.

A curva de segurança vira uma restrição genérica `VminOP_<REE>` por REE com folga penalizada pelo custo de `custos_penalidades`.

- `curva_seguranca · codigo_ree` → `constraints/generic_parameters.json` › `scalar_parameters[].kind`
- `curva_seguranca · codigo_ree` → `constraints/generic_parameters.json` › `scalar_parameters[].values`
- `curva_seguranca · codigo_ree` → `constraints/generic_constraints.json` › `constraints[].id`
- `curva_seguranca · codigo_ree` → `constraints/generic_constraints.json` › `constraints[].description`
- `custos_penalidades · penalidade` → `constraints/generic_constraints.json` › `constraints[].slack.penalty`
- `curva_seguranca · codigo_ree` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `curva_seguranca · data` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `configuracoes_penalizacao` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `curva_seguranca · valor` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `configuracoes_penalizacao · TIPO DE PENALIZACAO ≠ 0` — *adiado.* Penalização iterativa/variável da curva (etapa 2). O bridge modela apenas a penalização FIXA e emite `vminop-penalization-not-fixa` quando o deck seleciona outro tipo; a diferença na penalidade de violação é esperada.
- `configuracoes_penalizacao · MES PENALIZACAO` — *não lido.* Mês a partir do qual a penalização passa a valer. A restrição é emitida em todos os estágios cobertos pela curva; só o primeiro campo (tipo) e o terceiro (sazonalização no pós-estudo) são lidos.
- `iteracao_a_partir_etapa2 · maximo_iteracoes_etapa2 · tolerancia_processo_etapa2 · impressao_relatorio_etapa2` — *não lido.* Parâmetros do processo iterativo da etapa 2 da curva; sem contraparte na penalização fixa.

### `expt.dat`

**Estado:** convertido. **Lido por:** `constraints/thermal_bounds.parquet`, `system/thermals.json`.

Os cinco tipos (`POTEF`, `FCMAX`, `TEIFT`, `GTMIN`, `IPTER`) são aplicados em `constraints/thermal_bounds.parquet` na ordem do arquivo; usinas presentes sem `POTEF` são tratadas como não instaladas (`thermal-expt-without-potef`). A coluna `nome_usina` não é lida.

- `expansoes · tipo = GTMIN · modificacao` → `system/thermals.json` › `thermals[].generation.min_mw`
- `expansoes · data_inicio` → `system/thermals.json` › `thermals[].generation.min_mw`
- `expansoes · data_fim` → `system/thermals.json` › `thermals[].generation.min_mw`
- `expansoes · tipo = POTEF · modificacao` → `system/thermals.json` › `thermals[].generation.max_mw`
- `expansoes · tipo = FCMAX · modificacao` → `system/thermals.json` › `thermals[].generation.max_mw`
- `expansoes · tipo = TEIFT · modificacao` → `system/thermals.json` › `thermals[].generation.max_mw`
- `expansoes · tipo = IPTER · modificacao` → `system/thermals.json` › `thermals[].generation.max_mw`
- `expansoes · tipo = GTMIN · modificacao` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `expansoes · data_inicio` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `expansoes · data_fim` → `constraints/thermal_bounds.parquet` › `min_generation_mw`
- `expansoes · tipo = POTEF · modificacao` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `expansoes · tipo = FCMAX · modificacao` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `expansoes · tipo = TEIFT · modificacao` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `expansoes · tipo = IPTER · modificacao` → `constraints/thermal_bounds.parquet` › `max_generation_mw`

### `exph.dat`

**Estado:** convertido em parte. **Lido por:** 14 arquivos gerados; ver a visão geral.

Só o registro de enchimento (`data_inicio_enchimento`, `duracao_enchimento`, `volume_morto`) e as datas de entrada de máquinas (`data_entrada_operacao`, `conjunto_maquina_entrada`) são lidos. A usina EE não traz registro de enchimento: apenas linhas de entrada de máquina.

- `expansoes · data_inicio_enchimento` → `initial_conditions.json` › `filling_storage`
- `expansoes · data_inicio_enchimento` → `initial_conditions.json` › `filling_storage[].hydro_id`
- `expansoes · volume_morto` → `initial_conditions.json` › `filling_storage[].value_hm3`
- `data_inicio_enchimento` → `system/hydros.json` › `hydros[].id`
- `data_inicio_enchimento` → `system/hydros.json` › `hydros[].operational_start_date`
- `duracao_enchimento` → `system/hydros.json` › `hydros[].operational_start_date`
- `data_inicio_enchimento` → `system/hydros.json` › `hydros[].downstream_id`
- `data_entrada_operacao` → `system/hydros.json` › `hydros[].generation.model`
- `data_entrada_operacao` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `conjunto_maquina_entrada` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `data_entrada_operacao` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `conjunto_maquina_entrada` → `system/hydros.json` › `hydros[].generation.max_generation_mw`
- `data_inicio_enchimento` → `system/hydros.json` › `hydros[].filling`
- `duracao_enchimento` → `system/hydros.json` › `hydros[].filling`
- `data_inicio_enchimento` → `system/hydros.json` › `hydros[].filling.start_stage_id`
- `volume_morto` → `system/hydros.json` › `hydros[].filling.filling_min_rate_m3s`
- `data_inicio_enchimento` → `system/hydros.json` › `hydros[].filling.filling_min_rate_m3s`
- `duracao_enchimento` → `system/hydros.json` › `hydros[].filling.filling_min_rate_m3s`
- `data_inicio_enchimento` → `system/hydros.json` › `hydros[].entry_stage_id`
- `duracao_enchimento` → `system/hydros.json` › `hydros[].entry_stage_id`
- `data_entrada_operacao` → `constraints/hydro_bounds.parquet` › `stage_id`
- `data_entrada_operacao` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `conjunto_maquina_entrada` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `data_inicio_enchimento` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `duracao_enchimento` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`
- `data_inicio_enchimento` → `constraints/hydro_bounds.parquet` › `min_generation_mw`
- `data_entrada_operacao` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `conjunto_maquina_entrada` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `data_inicio_enchimento` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `duracao_enchimento` → `constraints/hydro_bounds.parquet` › `max_generation_mw`
- `potencia_instalada · maquina_entrada` — *não lido.* Potência e número da máquina que entra. A rampa de capacidade é recalculada a partir dos conjuntos do `hidr.dat` (máquinas por conjunto, vazão e potência nominais), contando as máquinas por `conjunto_maquina_entrada`. Vale para a usina NE em enchimento e para a usina EE em expansão; nesta a contagem nunca passa do cadastro de `hidr.dat`.

### `manutt.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/thermal_bounds.parquet`, `system/thermals.json`.

Cada linha reduz a potência da usina proporcionalmente aos dias de sobreposição com o mês, somando unidades em manutenção simultânea; só até `num_anos_manutencao_utes`.

- `manutencoes · potencia` → `system/thermals.json` › `thermals[].generation.max_mw`
- `manutencoes · data_inicio` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `manutencoes · duracao` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `manutencoes · potencia` → `constraints/thermal_bounds.parquet` › `max_generation_mw`
- `codigo_empresa · nome_empresa · codigo_unidade · nome_usina` — *não lido.* Identificação da empresa e da unidade geradora; as reduções são somadas por usina sem distinguir a unidade.

### `c_adic.dat`

**Estado:** convertido em parte. **Lido por:** `scenarios/load_seasonal_stats.parquet`.

- `valor` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `data` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `codigo_submercado` → `scenarios/load_seasonal_stats.parquet` › `mean_mw`
- `razao · nome_submercado` — *não lido.* Motivo da carga adicional. Todas as razões do mesmo submercado e mês são somadas à carga em `scenarios/load_seasonal_stats.parquet`; a razão não é preservada.

### `cvar.dat`

**Estado:** convertido. **Lido por:** `stages.json`.

Lido integralmente: `valores_constantes` para `cvar = 1` e `alfa_variavel`/`lambda_variavel` para `cvar = 2` em `dger.dat`; linhas do ano 9999 (pós-estudo) recaem nos valores constantes.

- `valores_constantes` → `stages.json` › `stages[].risk_measure`
- `valores_constantes` → `stages.json` › `stages[].risk_measure.cvar.alpha`
- `alfa_variavel · valor` → `stages.json` › `stages[].risk_measure.cvar.alpha`
- `valores_constantes` → `stages.json` › `stages[].risk_measure.cvar.lambda`
- `lambda_variavel · valor` → `stages.json` › `stages[].risk_measure.cvar.lambda`

### `agrint.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`.

Lido por um parser próprio de texto, não pelo `inewave`. Limites pós-estudo do arquivo são ignorados: o bridge congela o último estágio de estudo, como o NEWAVE.

- `agrupamentos · agrupamento` → `constraints/generic_constraints.json` › `constraints[].id`
- `agrupamentos · agrupamento` → `constraints/generic_constraints.json` › `constraints[].name`
- `agrupamentos · agrupamento` → `constraints/generic_constraints.json` › `constraints[].description`
- `agrupamentos · submercado_de` → `constraints/generic_constraints.json` › `constraints[].expression`
- `agrupamentos · submercado_para` → `constraints/generic_constraints.json` › `constraints[].expression`
- `agrupamentos · coeficiente` → `constraints/generic_constraints.json` › `constraints[].expression`
- `agrupamentos · agrupamento` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `limites_agrupamentos · data_inicio` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `limites_agrupamentos · data_fim` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `limites_agrupamentos · valor` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `limites_agrupamentos · comentario` — *não lido.* Descrição livre do limite; sem contraparte.

### `re.dat`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`.

Fornece o conjunto de usinas e os limites superiores das restrições elétricas no período pós-individualizado; limites inferiores só existem em `restricao-eletrica.csv`.

- `usinas_conjuntos · conjunto` → `constraints/generic_constraints.json` › `constraints[].id`
- `usinas_conjuntos · conjunto` → `constraints/generic_constraints.json` › `constraints[].name`
- `usinas_conjuntos · conjunto` → `constraints/generic_constraints.json` › `constraints[].description`
- `usinas_conjuntos · codigo_usina` → `constraints/generic_constraints.json` › `constraints[].expression`
- `usinas_conjuntos · conjunto` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `restricoes · mes_inicio` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `restricoes · ano_inicio` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `restricoes · mes_fim` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `restricoes · ano_fim` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `restricoes · patamar` → `constraints/generic_constraint_bounds.parquet` › `block_id`
- `restricoes · restricao` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `restricoes · motivo` — *não lido.* Justificativa textual da restrição; sem contraparte.

### `volref_saz.dat`

**Estado:** convertido. **Lido por:** `constraints/hydro_bounds.parquet`, `system/hydro_energy_productivity.parquet`, `system/hydro_production_models.json`, `system/hydros.json`.

Lido integralmente. Uma linha inteiramente zerada é tratada como ausência de referência sazonal (a usina recai em `volume_referencia`/altura 65%).

- `valor` → `system/hydros.json` › `hydros[].generation.max_turbined_m3s`
- `valor` → `system/hydros.json` › `hydros[].unit_groups[].max_turbined_m3s`
- `mes` → `system/hydros.json` › `hydros[].evaporation.reference_volumes_hm3[]`
- `valor` → `system/hydros.json` › `hydros[].evaporation.reference_volumes_hm3[]`
- `valor` → `system/hydro_production_models.json` › `production_models[].selection_mode`
- `mes` → `system/hydro_production_models.json` › `production_models[].seasons[].season_id`
- `valor` → `system/hydro_production_models.json` › `production_models[].seasons[].reference_volume.volume_hm3`
- `valor` → `system/hydro_energy_productivity.parquet` › `stage_id`
- `valor` → `system/hydro_energy_productivity.parquet` › `equivalent_productivity_mw_per_m3s`
- `valor` → `constraints/hydro_bounds.parquet` › `max_turbined_m3s`

### `shist.dat`

**Estado:** convertido. **Lido por:** `config.json`, `stages.json`.

Lido integralmente. `varredura`, `ano_inicio_varredura` e `anos_inicio_simulacoes` definem `historical_years` e o modo determinístico em `config.json`.

- `anos_inicio_simulacoes` → `config.json` › `simulation.selection.num_scenarios`
- `ano_inicio_varredura` → `config.json` › `simulation.selection.num_scenarios`
- `varredura` → `config.json` › `simulation.scenario_source.historical_years`
- `anos_inicio_simulacoes` → `config.json` › `simulation.scenario_source.historical_years`
- `ano_inicio_varredura` → `config.json` › `simulation.scenario_source.historical_years`
- `varredura` → `config.json` › `training.scenario_source.inflow.scheme`
- `anos_inicio_simulacoes` → `config.json` › `training.scenario_source.inflow.scheme`
- `varredura` → `config.json` › `training.scenario_source.historical_years`
- `anos_inicio_simulacoes` → `config.json` › `training.scenario_source.historical_years`
- `ano_inicio_varredura` → `config.json` › `training.scenario_source.historical_years`
- `varredura` → `stages.json` › `stages[].state_variables.inflow_lags`
- `anos_inicio_simulacoes` → `stages.json` › `stages[].state_variables.inflow_lags`
- `varredura` → `stages.json` › `stages[].sampling_method`
- `anos_inicio_simulacoes` → `stages.json` › `stages[].sampling_method`

### `adterm.dat`

**Estado:** convertido. **Lido por:** `initial_conditions.json`, `system/thermals.json`.

Lido integralmente quando `despacho_antecipado_gnl ≠ 0`; despachos por patamar são ponderados pela duração dos patamares e lags além do horizonte são truncados com aviso.

- `despachos · codigo_usina` → `initial_conditions.json` › `past_anticipated_commitments[].thermal_id`
- `despachos · lag` → `initial_conditions.json` › `past_anticipated_commitments[].start_date`
- `despachos · lag` → `initial_conditions.json` › `past_anticipated_commitments[].end_date`
- `despachos · valor` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `despachos · patamar` → `initial_conditions.json` › `past_anticipated_commitments[].value_mw`
- `despachos · codigo_usina` → `system/thermals.json` › `thermals[].anticipated_config`
- `despachos · lag` → `system/thermals.json` › `thermals[].anticipated_config.lead_stages`

### `polinjus.csv`

**Estado:** convertido em parte. **Lido por:** `system/tailrace_curves.parquet`.

Só as famílias de curva de jusante (`HidreletricaCurvaJusante` e `HidreletricaCurvaJusantePolinomioPorPartesSegmento`) alimentam `system/tailrace_curves.parquet`.

- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · codigo_usina` → `system/tailrace_curves.parquet` › `hydro_id`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · indice_familia` → `system/tailrace_curves.parquet` › `family_id`
- `HIDRELETRICA-CURVAJUSANTE · nivel_montante_referencia` → `system/tailrace_curves.parquet` › `downstream_reference_level_m`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · indice_polinomio` → `system/tailrace_curves.parquet` › `segment_id`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · limite_inferior_vazao_jusante` → `system/tailrace_curves.parquet` › `outflow_min_m3s`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · limite_superior_vazao_jusante` → `system/tailrace_curves.parquet` › `outflow_max_m3s`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a0` → `system/tailrace_curves.parquet` › `coefficient_0`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a1` → `system/tailrace_curves.parquet` › `coefficient_1`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a2` → `system/tailrace_curves.parquet` › `coefficient_2`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a3` → `system/tailrace_curves.parquet` › `coefficient_3`
- `HIDRELETRICA-CURVAJUSANTE-POLINOMIOPORPARTES-SEGMENTO · coeficiente_a4` → `system/tailrace_curves.parquet` › `coefficient_4`
- `HidreletricaCurvaJusanteAfogamentoExplicitoPadrao · HidreletricaCurvaJusanteAfogamentoExplicitoUsina` — *não lido.* Flag de afogamento explícito do canal de fuga. O Novomodelo escolhe a família pelo nível de referência de jusante; o flag não é consultado.
- `HidreletricaPerdaHidraulicaGrade · HidreletricaProdutibilidadeEspecificaGrade` — *não lido.* Grades de perda hidráulica e de produtibilidade específica por turbinamento. O bridge usa os valores constantes do `hidr.dat` (`perdas`, `tipo_perda`, `produtibilidade_especifica`).
- `VolumeReferencialPeriodo · VolumeReferencialTipoPadrao · EstacaoBombeamento*` — *não lido.* Volumes de referência por período e estações de bombeamento. O volume de referência vem de `hidr.dat`/`volref_saz.dat`; bombeamento não é modelado.

### `tratamento-fpha.csv`

**Estado:** convertido em parte. **Lido por:** `system/hydro_production_models.json`.

Só a linha ativa terminada em `ANGULO-PADRAO` ou `DISTANCIA-PADRAO` vira `fpha_plane_reduction` em `system/hydro_production_models.json`; o número de amostras do método por distância é uma constante do bridge.

- `HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO / HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO` → `system/hydro_production_models.json` › `fpha_plane_reduction.method`
- `HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO` → `system/hydro_production_models.json` › `fpha_plane_reduction.tolerance_deg`
- `HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO` → `system/hydro_production_models.json` › `fpha_plane_reduction.tolerance_pct`
- `linhas por usina (sufixo diferente de -PADRAO)` — *não lido.* Ajustes do método de redução de cortes por usina; o Novomodelo só aceita um método por caso.

### `caso.dat`

**Estado:** arquivo de índice, lido para localizar os demais.

Nomeia o arquivo de índice (`arquivos.dat` por padrão). Lido apenas para localizar os demais arquivos do caso.

### `arquivos.dat`

**Estado:** arquivo de índice, lido para localizar os demais.

Índice que nomeia cada arquivo de dados do caso. Lido apenas para localizar os demais arquivos; os nomes que aponta são resolvidos sem distinção de maiúsculas.

### `restricao-eletrica.csv`

**Estado:** convertido em parte. **Lido por:** `constraints/generic_constraint_bounds.parquet`, `constraints/generic_constraints.json`.

Localizado pela chave `RESTRICAO-ELETRICA-ESPECIAL` de `indices.csv` e lido por um parser próprio; as demais chaves de `indices.csv` não são seguidas.

- `RE · cod_rest` → `constraints/generic_constraints.json` › `constraints[].id`
- `RE · cod_rest` → `constraints/generic_constraints.json` › `constraints[].name`
- `RE · cod_rest` → `constraints/generic_constraints.json` › `constraints[].description`
- `RE · formula` → `constraints/generic_constraints.json` › `constraints[].expression`
- `RE · cod_rest` → `constraints/generic_constraint_bounds.parquet` › `constraint_id`
- `RE-LIM-FORM-PER-PAT · PerIni` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `RE-LIM-FORM-PER-PAT · PerFin` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `RE-HORIZ-PER · PerIni` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `RE-HORIZ-PER · PerFin` → `constraints/generic_constraint_bounds.parquet` › `stage_id`
- `RE-LIM-FORM-PER-PAT · pat` → `constraints/generic_constraint_bounds.parquet` › `block_id`
- `RE-LIM-FORM-PER-PAT · lim_inf` → `constraints/generic_constraint_bounds.parquet` › `bound_lower`
- `RE-LIM-FORM-PER-PAT · lim_sup` → `constraints/generic_constraint_bounds.parquet` › `bound_upper`
- `funções de fórmula além de ger_usih e ener_interc` — *não lido.* Só termos `ger_usih(usina)` e `ener_interc(de, para)` são traduzidos para `hydro_generation` e `line_direct`/`line_reverse`; qualquer outra função (térmica, eólica, carga) é descartada silenciosamente pelo parser de fórmula.
- `registros RHE, RHQ, RHV e seus RHE/RHQ/RHV-HORIZ-PER, -LIM-FORM-PER, -LS-LPP-*` — *não lido.* Restrições hidráulicas de energia, vazão e volume (inclusive as LPP lineares por partes). O parser ignora toda linha cujo primeiro campo não seja `RE`, `RE-HORIZ-PER` ou `RE-LIM-FORM-PER-PAT`; converter exigiria restrições genéricas sobre `hydro_storage_final`/`hydro_outflow`.

### `indices.csv`

**Estado:** arquivo de índice, lido para localizar os demais.

Índice dos arquivos LIBs. Lido apenas para localizar `restricao-eletrica.csv`.

### `arquivos de saída do NEWAVE`

**Estado:** não lido.

Entradas de `arquivos.dat` que são resultados de uma execução, não dados do caso: `pmo.dat`, `parp.dat`, `forward.dat`, `forwardh.dat`, `newdesp.dat`, `cortes.dat`, `cortesh.dat`, `cortes_pos_estudo`, `cortesh_pos_estudo` e `dados_simulacao_final`. A conversão não os consome; o comando de comparação de resultados lê `pmo.dat` e os CSV `MEDIAS-*` apenas para confrontar com a saída do Novomodelo.

### `gtminpat.dat`

**Estado:** não lido.

Geração térmica mínima por patamar de carga (usina × mês × patamar). O bridge só lê o GTMIN mensal de `expt.dat`; converter exigiria emitir `min_generation_mw` por bloco em `constraints/thermal_bounds.parquet`, que hoje é por estágio.

### `itaipu.dat`

**Estado:** não lido.

Restrições operativas de Itaipu (parcelas 50 Hz/60 Hz e limites de intercâmbio associados). Ativado por `restricao_itaipu` em `dger.dat`; o bridge trata Itaipu como usina comum, sem partição de geração.

### `perda.dat`

**Estado:** não lido.

Fatores de perda na rede de transmissão por patamar. Ativado por `perdas_rede_transmissao` em `dger.dat`; o Novomodelo não modela perdas de intercâmbio e o bridge não emite nenhum equivalente.

### `tecno.dat`

**Estado:** não lido.

Tecnologia e combustível das classes térmicas (base para emissões e gás). Sem uso: o Novomodelo não distingue tecnologia térmica.

### `clasgas.dat`

**Estado:** não lido.

Restrições de fornecimento de gás natural por classe térmica. Ativado por `restricoes_fornecimento_gas`; converter exigiria restrições genéricas sobre `thermal_generation` agrupadas por classe.

### `gee.dat`

**Estado:** não lido.

Limites de emissão de gases de efeito estufa por submercado ou SIN. Ativado por `restricoes_emissao_gee`; não há contraparte no Novomodelo.

### `sar.dat`

**Estado:** não lido.

Superfície de aversão a risco (SAR). Ativado por `sar` em `dger.dat`; o bridge só representa aversão a risco via CVaR (`cvar.dat`) e curva de segurança (`curva.dat`).

### `bid.dat`

**Estado:** não lido.

Dados de ofertas (bid) de geração. Ativado por `bid` em `dger.dat`; sem uso no bridge.

### `abertura.dat`

**Estado:** não lido.

Número de aberturas variável por período. O bridge lê apenas `num_aberturas` de `dger.dat` e escreve o mesmo `num_openings` em todos os estágios; converter exigiria mapear a tabela por período para `stages[].num_openings`.

### `elnino.dat`

**Estado:** não lido.

Índices de El Niño para o modelo de afluências. Ativado por `el_nino` em `dger.dat`; o Novomodelo estima o PAR(p) sem covariáveis climáticas.

### `ensoaux.dat`

**Estado:** não lido.

Dados auxiliares de ENSO. Ativado por `enso` em `dger.dat`; mesma situação de `elnino.dat`.

### `eliminacao-cortes.dat`

**Estado:** não lido.

Parâmetros do algoritmo de eliminação de cortes de Benders. O bridge espelha apenas o liga/desliga da seleção de cortes (`selecao_de_cortes_*` de `dger.dat`) em `config.json`; os parâmetros finos ficam nos defaults do Novomodelo.

### `volumes-referencia.csv`

**Estado:** não lido.

Arquivo LIBs apontado por `indices.csv` (chave `HIDRELETRICA-CADASTRO-RESERVATORIO`) com `VOLUME-REFERENCIAL-TIPO-PADRAO` e `CADH-VOL-REF-PER` (volume de referência por usina e período). O bridge deriva o volume de referência de `hidr.dat` (`volume_referencia`) e de `volref_saz.dat`; a chave não é seguida em `indices.csv`.

## Ainda não convertido

Tudo o que o conversor deixa de fora, reunido em um só lugar. Cada item aparece também no índice acima, junto do arquivo a que pertence.

### Arquivos do deck não lidos

- `arquivos de saída do NEWAVE` — *não lido.* Entradas de `arquivos.dat` que são resultados de uma execução, não dados do caso: `pmo.dat`, `parp.dat`, `forward.dat`, `forwardh.dat`, `newdesp.dat`, `cortes.dat`, `cortesh.dat`, `cortes_pos_estudo`, `cortesh_pos_estudo` e `dados_simulacao_final`. A conversão não os consome; o comando de comparação de resultados lê `pmo.dat` e os CSV `MEDIAS-*` apenas para confrontar com a saída do Novomodelo.
- `gtminpat.dat` — *não lido.* Geração térmica mínima por patamar de carga (usina × mês × patamar). O bridge só lê o GTMIN mensal de `expt.dat`; converter exigiria emitir `min_generation_mw` por bloco em `constraints/thermal_bounds.parquet`, que hoje é por estágio.
- `itaipu.dat` — *não lido.* Restrições operativas de Itaipu (parcelas 50 Hz/60 Hz e limites de intercâmbio associados). Ativado por `restricao_itaipu` em `dger.dat`; o bridge trata Itaipu como usina comum, sem partição de geração.
- `perda.dat` — *não lido.* Fatores de perda na rede de transmissão por patamar. Ativado por `perdas_rede_transmissao` em `dger.dat`; o Novomodelo não modela perdas de intercâmbio e o bridge não emite nenhum equivalente.
- `tecno.dat` — *não lido.* Tecnologia e combustível das classes térmicas (base para emissões e gás). Sem uso: o Novomodelo não distingue tecnologia térmica.
- `clasgas.dat` — *não lido.* Restrições de fornecimento de gás natural por classe térmica. Ativado por `restricoes_fornecimento_gas`; converter exigiria restrições genéricas sobre `thermal_generation` agrupadas por classe.
- `gee.dat` — *não lido.* Limites de emissão de gases de efeito estufa por submercado ou SIN. Ativado por `restricoes_emissao_gee`; não há contraparte no Novomodelo.
- `sar.dat` — *não lido.* Superfície de aversão a risco (SAR). Ativado por `sar` em `dger.dat`; o bridge só representa aversão a risco via CVaR (`cvar.dat`) e curva de segurança (`curva.dat`).
- `bid.dat` — *não lido.* Dados de ofertas (bid) de geração. Ativado por `bid` em `dger.dat`; sem uso no bridge.
- `abertura.dat` — *não lido.* Número de aberturas variável por período. O bridge lê apenas `num_aberturas` de `dger.dat` e escreve o mesmo `num_openings` em todos os estágios; converter exigiria mapear a tabela por período para `stages[].num_openings`.
- `elnino.dat` — *não lido.* Índices de El Niño para o modelo de afluências. Ativado por `el_nino` em `dger.dat`; o Novomodelo estima o PAR(p) sem covariáveis climáticas.
- `ensoaux.dat` — *não lido.* Dados auxiliares de ENSO. Ativado por `enso` em `dger.dat`; mesma situação de `elnino.dat`.
- `eliminacao-cortes.dat` — *não lido.* Parâmetros do algoritmo de eliminação de cortes de Benders. O bridge espelha apenas o liga/desliga da seleção de cortes (`selecao_de_cortes_*` de `dger.dat`) em `config.json`; os parâmetros finos ficam nos defaults do Novomodelo.
- `volumes-referencia.csv` — *não lido.* Arquivo LIBs apontado por `indices.csv` (chave `HIDRELETRICA-CADASTRO-RESERVATORIO`) com `VOLUME-REFERENCIAL-TIPO-PADRAO` e `CADH-VOL-REF-PER` (volume de referência por usina e período). O bridge deriva o volume de referência de `hidr.dat` (`volume_referencia`) e de `volref_saz.dat`; a chave não é seguida em `indices.csv`.

### Registros e campos do deck não convertidos

- `dger.dat` › `execução, impressão e memória (nome_caso, duracao_periodo, imprime_*, impressao_operacao/ena/convergencia/cortes_ativos_sim_final, gera_arquivo_cortes_unico, mantem_arquivos_*, alocacao_memoria_*, memoria_calculo_cortes, armazenamento_local_arquivos_temporarios, utiliza_gerenciamento_pls, comunicacao_dois_niveis, intervalo_para_gravar, tamanho_registro_arquivo_historico, consulta_fcf)` — *não lido.* Controles de relatório, disco e memória do executável NEWAVE. Não têm contraparte no caso Novomodelo.
- `dger.dat` › `convergência e gestão de cortes (tolerancia, delta_zinf, delta_zsup, deltas_consecutivos, converge_no_zero, num_minimo_iteracoes, inicio_teste_convergencia, desconsidera_convergencia_estatistica, considera_zsup_min_convergencia, iteracao_para_simulacao_final, aproveitamento_bases_backward, janela_de_cortes, periodos_manutencao_cortes, eliminacao_cortes, fcf_pos_estudo, mes_inicio_pre_estudo)` — *não lido.* O bridge fixa apenas `iteration_limit` (de `num_max_iteracoes`) e o liga/desliga da seleção de cortes; critérios de parada por gap e janelas de cortes ficam nos defaults do Novomodelo. `num_anos_pre_estudo` é convertido em `pre_study_stages`, mas `mes_inicio_pre_estudo` não é lido.
- `dger.dat` › `modelo estocástico de afluências (tipo_geracao_enas, matriz_correlacao_espacial, reducao_automatica_ordem, considera_tendencia_hidrologica_calculo_politica, considera_tendencia_hidrologica_sim_final, tipo_reamostragem_cenarios, passo_reamostragem_cenarios, momento_reamostragem, aberturas_variaveis, num_anos_pos_sim_final, agregacao_simulacao_final, simulacao_final_com_data, representacao_agregacao, el_nino, enso)` — *não lido.* O Novomodelo estima o PAR(p) a partir de `scenarios/inflow_history.parquet` com sua própria seleção de ordem (`estimation.order_selection`); só `ordem_maxima_parp`, `consideracao_media_anual_afluencias`, `num_series_sinteticas` e `considera_reamostragem_cenarios` são espelhados. A tendência hidrológica é sempre emitida (`recent_observations`) quando `vazpast.dat` existe, sem consultar os flags.
- `dger.dat` › `flags de recursos não convertidos (restricao_defluencia, restricao_itaipu, restricoes_rhq, restricoes_rhv, restricao_lpp_*, restricoes_emissao_gee, restricoes_fornecimento_gas, sar, bid, modif_automatica_adterm, sazonaliza_vminp, canal_desvio, correcao_desvio)` — *não lido.* Ativam recursos que o conversor não converte (vazão máxima `VAZMAXT`, RHQ/RHV, LPP, SAR, GEE, gás, canal de desvio, ADTERM automático), logo não há o que ligar ou desligar. Os flags de `ghmin.dat`, `c_adic.dat`, `agrint.dat`, `re.dat`, `restricao-eletrica.csv`, vazão mínima e turbinamento são honrados: um arquivo presente com o flag desligado é ignorado e reportado (diagnóstico `dger-switch-off`); linha ausente conta como ligada.
- `dger.dat` › `operação e penalidades internas (perdas_rede_transmissao, considera_geracao_eolica, penalidade_corte_geracao_eolica, estacoes_bombeamento, representacao_submotorizacao, calcula_volume_inicial, volume_inicial_subsistema, ordenacao_automatica, calcula_prodt_media_sin, racionamento_preventivo, primeira_profundidade_risco_deficit, segunda_profundidade_risco_deficit, equalizacao_penal_intercambio)` — *não lido.* As micro-penalidades de intercâmbio, vertimento, turbinamento, corte de eólica e excesso são escritas com os defaults internos do manual NEWAVE v30, não com o valor de `penalidade_corte_geracao_eolica`. O volume inicial vem de `confhd.dat` (`volume_inicial_percentual`); `calcula_volume_inicial`/`volume_inicial_subsistema` não são lidos.
- `confhd.dat` › `usinas fictícias (produtibilidade_especifica = 0 no mesmo posto de uma usina geradora)` — *adiado.* Excluídas do mapa de ids com o diagnóstico `fictitious-plants-excluded`. São nós contábeis de cascata energética; o bridge preserva a topologia religando `downstream_id` à próxima usina real e somando a produtibilidade da cadeia fictícia à usina de montante.
- `confhd.dat` › `usinas NE sem registro de enchimento em exph.dat e usinas NC` — *não lido.* Não entram no LP; a cascata é religada através delas (log informativo apenas). Converter uma expansão sem enchimento exigiria `entry_stage_id` sem bloco `filling`.
- `confhd.dat` › `usina_modificada` — *não lido.* Indica se a usina tem registros em `modif.dat`; o bridge aplica os registros de `modif.dat` a toda usina cadastrada, sem consultar o flag.
- `confhd.dat` › `ano_inicio_historico · ano_fim_historico` — *não lido.* Janela do histórico de vazões por usina. O bridge usa o histórico completo de `vazoes.dat` desde `ano_inicial_historico` até o mês anterior ao início do estudo para todas as usinas.
- `conft.dat` › `usina_existente · classe` — *não lido.* Toda usina listada vira uma térmica do Novomodelo; o estado `EX`/`NE`/`EE` e a classe térmica não são consultados. A disponibilidade efetiva vem das janelas POTEF de `expt.dat`.
- `sistema.dat` › `numero_patamares_deficit` — *não lido.* Os segmentos de déficit são derivados das linhas de `custo_deficit` (`patamar_deficit`, `custo`, `corte`); a contagem declarada não é lida.
- `sistema.dat` › `limites_intercambio · flag` — *não lido.* Coluna de flag do bloco de limites de intercâmbio; o bridge usa apenas `submercado_de`, `submercado_para`, `sentido`, `data` e `valor`.
- `clast.dat` › `tipo_combustivel` — *não lido.* Combustível da classe térmica. O Novomodelo não distingue combustível; só o custo por ano de estudo e as modificações por data são convertidos.
- `ree.dat` › `remocao_ficticias` — *não lido.* Flag de remoção de usinas fictícias do REE; o bridge identifica as fictícias estruturalmente (posto compartilhado e ρ_esp = 0), sem consultar o flag.
- `hidr.dat` › `posto · submercado · codigo_usina_jusante (do cadastro)` — *não lido.* Posto, submercado e jusante são tomados de `confhd.dat` (jusante) e de `ree.dat` via o REE de `confhd.dat` (barra); as colunas homônimas do cadastro são ignoradas.
- `hidr.dat` › `numero_polinomios_jusante · a0..a4_jusante_1..6 · referencia_jusante_1..6` — *não lido.* Polinômios cota de jusante × defluência do cadastro. O bridge escreve `tailrace` como constante `canal_fuga_medio` em `hydros.json` e, quando há `polinjus.csv`, as famílias por partes em `system/tailrace_curves.parquet`; os polinômios do `hidr.dat` não são usados.
- `hidr.dat` › `influencia_vertimento_canal_fuga` — *não lido.* Flag de influência do vertimento no canal de fuga. O Novomodelo avalia as curvas de jusante sobre a defluência total; o flag não é consultado.
- `hidr.dat` › `fator_carga_maximo · fator_carga_minimo` — *não lido.* Fatores de carga da usina hidráulica; o limite de turbinamento vem do engolimento corrigido pela queda e o de geração da potência nominal dos conjuntos.
- `hidr.dat` › `volume_vertedouro · volume_desvio · cota_minima · cota_maxima · desvio` — *não lido.* Volumes de vertedouro e de desvio, cotas limite e o código de desvio. O Novomodelo não modela vertedouro nem canal de desvio (`diversion` é sempre nulo); as cotas são avaliadas pelo polinômio cota × volume.
- `hidr.dat` › `numero_unidades_base · representacao_conjunto · empresa · observacao · data` — *não lido.* Dados cadastrais sem contraparte: unidades-base para submotorização, forma de representação dos conjuntos, empresa proprietária e observações.
- `modif.dat` › `VMINP` — *adiado.* Volume mínimo com penalidade (meta operativa suave por estágio). Reportado por `modif-permanent-override-unsupported` e ignorado; converter exigiria uma restrição genérica com folga sobre `hydro_storage_final` penalizada pelo custo VOLMIN de `penalid.dat`, honrando `sazonaliza_vminp`.
- `modif.dat` › `VAZMAXT` — *adiado.* Defluência máxima por período. Reportado por `modif-permanent-override-unsupported` e ignorado; o Novomodelo aceita `max_outflow_m3s`, mas o bridge escreve sempre nulo em `hydros.json` e não emite coluna correspondente em `constraints/hydro_bounds.parquet`.
- `ghmin.dat` › `linhas com patamar ≠ 0` — *não lido.* Só as linhas de patamar 0 (média dos patamares) alimentam `min_generation_mw` em `constraints/hydro_bounds.parquet`; valores por patamar de carga são descartados porque a tabela é por estágio, não por bloco.
- `penalid.dat` › `codigo_ree_submercado` — *não lido.* Diferenciação da penalidade por REE ou submercado. O leitor por REE existe no código mas não é chamado; `penalties.json` recebe um único valor por variável.
- `penalid.dat` › `patamar_penalidade = 2` — *não lido.* Segundo patamar de penalidade (custo por violação adicional). O Novomodelo tem uma única folga linear por variável; só o primeiro patamar é lido.
- `penalid.dat` › `patamar_carga · valor_R$_hm3` — *não lido.* Penalidade específica por patamar de carga e o valor já convertido em R$/hm³. O bridge parte sempre de `valor_R$_MWh` e faz a própria conversão de unidades.
- `penalid.dat` › `variáveis além de DESVIO, VAZMIN, GHMIN, TURBMN, TURBMX, VOLMIN e ELETRI` — *não lido.* Qualquer outra variável presente (por exemplo penalidades de intercâmbio ou de excesso) é ignorada; essas micro-penalidades são escritas com os defaults internos do NEWAVE.
- `dsvagua.dat` › `considera_desvio_usina_NC` — *não lido.* Flag que decide se o desvio de uma usina `NC` é aplicado à usina existente a jusante. O bridge propaga sempre o desvio de uma usina fora do LP para a primeira usina existente a jusante, sem consultar o flag.
- `dsvagua.dat` › `comentario` — *não lido.* Texto livre da linha; sem contraparte.
- `curva.dat` › `configuracoes_penalizacao · TIPO DE PENALIZACAO ≠ 0` — *adiado.* Penalização iterativa/variável da curva (etapa 2). O bridge modela apenas a penalização FIXA e emite `vminop-penalization-not-fixa` quando o deck seleciona outro tipo; a diferença na penalidade de violação é esperada.
- `curva.dat` › `configuracoes_penalizacao · MES PENALIZACAO` — *não lido.* Mês a partir do qual a penalização passa a valer. A restrição é emitida em todos os estágios cobertos pela curva; só o primeiro campo (tipo) e o terceiro (sazonalização no pós-estudo) são lidos.
- `curva.dat` › `iteracao_a_partir_etapa2 · maximo_iteracoes_etapa2 · tolerancia_processo_etapa2 · impressao_relatorio_etapa2` — *não lido.* Parâmetros do processo iterativo da etapa 2 da curva; sem contraparte na penalização fixa.
- `exph.dat` › `potencia_instalada · maquina_entrada` — *não lido.* Potência e número da máquina que entra. A rampa de capacidade é recalculada a partir dos conjuntos do `hidr.dat` (máquinas por conjunto, vazão e potência nominais), contando as máquinas por `conjunto_maquina_entrada`. Vale para a usina NE em enchimento e para a usina EE em expansão; nesta a contagem nunca passa do cadastro de `hidr.dat`.
- `manutt.dat` › `codigo_empresa · nome_empresa · codigo_unidade · nome_usina` — *não lido.* Identificação da empresa e da unidade geradora; as reduções são somadas por usina sem distinguir a unidade.
- `c_adic.dat` › `razao · nome_submercado` — *não lido.* Motivo da carga adicional. Todas as razões do mesmo submercado e mês são somadas à carga em `scenarios/load_seasonal_stats.parquet`; a razão não é preservada.
- `agrint.dat` › `limites_agrupamentos · comentario` — *não lido.* Descrição livre do limite; sem contraparte.
- `re.dat` › `restricoes · motivo` — *não lido.* Justificativa textual da restrição; sem contraparte.
- `polinjus.csv` › `HidreletricaCurvaJusanteAfogamentoExplicitoPadrao · HidreletricaCurvaJusanteAfogamentoExplicitoUsina` — *não lido.* Flag de afogamento explícito do canal de fuga. O Novomodelo escolhe a família pelo nível de referência de jusante; o flag não é consultado.
- `polinjus.csv` › `HidreletricaPerdaHidraulicaGrade · HidreletricaProdutibilidadeEspecificaGrade` — *não lido.* Grades de perda hidráulica e de produtibilidade específica por turbinamento. O bridge usa os valores constantes do `hidr.dat` (`perdas`, `tipo_perda`, `produtibilidade_especifica`).
- `polinjus.csv` › `VolumeReferencialPeriodo · VolumeReferencialTipoPadrao · EstacaoBombeamento*` — *não lido.* Volumes de referência por período e estações de bombeamento. O volume de referência vem de `hidr.dat`/`volref_saz.dat`; bombeamento não é modelado.
- `tratamento-fpha.csv` › `linhas por usina (sufixo diferente de -PADRAO)` — *não lido.* Ajustes do método de redução de cortes por usina; o Novomodelo só aceita um método por caso.
- `restricao-eletrica.csv` › `funções de fórmula além de ger_usih e ener_interc` — *não lido.* Só termos `ger_usih(usina)` e `ener_interc(de, para)` são traduzidos para `hydro_generation` e `line_direct`/`line_reverse`; qualquer outra função (térmica, eólica, carga) é descartada silenciosamente pelo parser de fórmula.
- `restricao-eletrica.csv` › `registros RHE, RHQ, RHV e seus RHE/RHQ/RHV-HORIZ-PER, -LIM-FORM-PER, -LS-LPP-*` — *não lido.* Restrições hidráulicas de energia, vazão e volume (inclusive as LPP lineares por partes). O parser ignora toda linha cujo primeiro campo não seja `RE`, `RE-HORIZ-PER` ou `RE-LIM-FORM-PER-PAT`; converter exigiria restrições genéricas sobre `hydro_storage_final`/`hydro_outflow`.

### Campos do Novomodelo sem origem no deck

- `penalties.json` › `bus.deficit_segments[].depth_mw` — Sempre nulo: o segmento global é ilimitado; as profundidades por patamar ficam em `system/buses.json`.
- `system/thermals.json` › `thermals[].entry_stage_id` — Sempre nulo: térmicas do NEWAVE não entram dentro do horizonte; a expansão de `expt.dat` é expressa por limites por estágio.
- `system/thermals.json` › `thermals[].exit_stage_id` — Sempre nulo: térmicas do NEWAVE não saem dentro do horizonte.
- `system/hydros.json` › `hydros[].outflow.max_outflow_m3s` — Sempre nulo: o NEWAVE não impõe vazão defluente máxima estática; o registro `VAZMAXT` de `modif.dat` não é convertido.
- `system/hydros.json` › `hydros[].diversion` — Sempre nulo: os desvios de água de `dsvagua.dat` são convertidos como retirada por estágio (`water_withdrawal_m3s` em `constraints/hydro_bounds.parquet`), não como canal de desvio.
- `system/hydros.json` › `hydros[].penalties` — Sempre nulo: as penalidades de `penalid.dat` são convertidas uma única vez, com a produtibilidade média do SIN, para os valores globais de `penalties.json`, sem sobrescrita por usina.
- `system/hydros.json` › `hydros[].exit_stage_id` — Sempre nulo: usinas hidrelétricas do NEWAVE não saem de operação dentro do horizonte.
- `system/hydro_production_models.json` › `production_models[].stage_ranges[].end_stage_id` — Sempre nulo: a faixa única é aberta até o fim do horizonte.
- `system/hydro_energy_productivity.parquet` › `reference_outflow_m3s` — Sempre nulo: o volume de referência é declarado em `system/hydro_production_models.json`, não por vazão de referência.
- `system/hydro_energy_productivity.parquet` › `specific_productivity_mw_per_m3s_per_m` — Sempre nulo aqui: a produtibilidade específica é declarada em `hydros[].specific_productivity_mw_per_m3s_per_m`.
