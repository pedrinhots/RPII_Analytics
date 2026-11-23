# Relatório de Tratamento de Dados

ACH0042 – Grupo 11 · *Predição de criminalidade contra pessoas trans e travestis*

Este relatório documenta a etapa de tratamento de dados completa: da base bruta do Disque 100 até a matriz numérica de entrada dos modelos. O detalhamento variável a variável do Disque 100 está em `claude/tratamento_dados_disque100.md`.

---

## 1. Visão geral

O tratamento juntou seis fontes num painel **município × ano** com 66.780 linhas: 5.565 municípios × 12 anos (2011 a 2022). Cada linha reúne as denúncias do Disque 100 daquele município no ano e os indicadores socioeconômicos, demográficos, de violência e institucionais do mesmo município.

```
Disque 100 (lgbt.csv, 50.045 linhas)
   └─ limpeza_disque100.py ──► base_limpa.csv (27.441 registros) ──┐
                                                                  │ agregação município × ano
IBGE/SIDRA + Ipea ─┐                                              ▼
MUNIC 2014/19/23 ──┼─ coletar_dados_municipais.py ─► montar_painel.py ─► painel_municipio_ano.csv (66.780 × 39)
Atlas Brasil 2010 ─┘                                 ▲            └──► matriz_painel_ml.csv (66.780 × 65)
SIA/SIH-SUS (Processo Transexualizador) ─────────────┘
```

| Script | Função |
|---|---|
| `limpeza_disque100.py` | limpa, unifica os três formatos e codifica o Disque 100 |
| `coletar_dados_municipais.py` | baixa os indicadores municipais (APIs do IBGE e do Ipea, FTP da MUNIC, Atlas Brasil) |
| `montar_painel.py` | harmoniza, interpola, deflaciona, junta tudo e codifica o painel |

## 2. Disque 100

Resumo do que está detalhado em `tratamento_dados_disque100.md`:

- **Três formatos unificados:** 2011–2019, 1º semestre de 2020, e de jul/2020 a 2022. No último, uma linha é uma violação, não uma denúncia, e foi preciso colapsar.
- **Unidade final:** denúncia × vítima. A base sai com 27.441 registros, depois de remover 561 duplicatas exatas e colapsar 28.362 linhas de violação em 6.319 registros.
- **Identificação de vítima trans:** cada formato guarda essa informação num campo diferente. Entre jul/2020 e dez/2021 o formulário não tinha categoria trans, então a variável fica **ausente** nesse período (e não zero).
- **Categorias recodificadas:** cerca de 240 descrições de violação viraram 9 macro-categorias, e local, relação com o suspeito, faixa etária, raça/cor, denunciante e canal foram agrupados.
- **Município:** 97,6% dos registros têm o município identificado (código IBGE de 6 dígitos).
- **Codificação:** one-hot nas variáveis categóricas e multi-hot nas violações, o que dá uma matriz de 27.441 × 105.

## 3. Indicadores municipais: coleta

A coleta é automatizada e reproduzível pelo script `coletar_dados_municipais.py`. Ele foi rodado em 22/11/2026, em 1,9 minuto e sem erros. Todas as consultas foram testadas antes com um município (São Paulo).

| Variável | Fonte | Anos |
|---|---|---|
| População | SIDRA 6579 (estimativas); para 2022, SIDRA 4714 (Censo) | 2011–2022 |
| Área e densidade | SIDRA 4714 | 2022 |
| PIB | SIDRA 5938 | 2011–2022 |
| Renda domiciliar per capita | SIDRA 3578 (2010) e 10295 (2022) | 2010, 2022 |
| Cor ou raça | SIDRA 9605 | 2010, 2022 |
| População de 15 a 29 anos | SIDRA 9606 | 2010, 2022 |
| Domicílios com casal do mesmo sexo | SIDRA 9881 | 2022 |
| Homicídios registrados | Ipea, Atlas da Violência, série 328 (API) | 2011–2022 |
| Políticas de direitos humanos e LGBT | IBGE, MUNIC (bases completas) | 2014, 2019, 2023 |
| IDHM e Gini | Atlas Brasil (PNUD/Ipea/FJP), base do Censo | 2010 |
| Procedimentos do Processo Transexualizador | SIA/SIH-SUS (extração do grupo), 26 municípios | 2008–2026 |

**Tratamentos feitos já na coleta:**
- **Traço "-" do SIDRA:** na convenção do IBGE, "-" significa zero absoluto, e foi convertido para 0. Na primeira execução ele foi lido como ausente, o que deixava 6,9% dos municípios sem o indicador de casais do mesmo sexo. Isso foi corrigido.
- **Homicídios:** a série do Ipea só lista municípios com pelo menos um homicídio no ano. Município ausente na série = 0.
- **Sem estimativa para 2022:** a série 6579 não tem população estimada para o ano do Censo. Em 2022 foi usada a população recenseada.

## 4. Harmonização da MUNIC

A MUNIC muda os códigos das variáveis entre edições: 2014 usa o prefixo `A`, e 2019 e 2023 usam `MDHU`. Foram escolhidas as 7 perguntas que existem nas três edições, todas binárias:

| Variável harmonizada | 2014 | 2019 / 2023 | % de municípios com "Sim" (2014 → 2019) |
|---|---|---|---|
| `munic_estrutura_dh`: tem órgão gestor de direitos humanos | A392 | MDHU01 | 44,1 → 44,5 |
| `munic_tema_lgbt`: órgão trata proteção dos direitos LGBT | A428 | MDHU109 | 0,6 → 1,7 |
| `munic_nome_social`: órgão trata reconhecimento do nome social | A430 | MDHU1010 | 0,5 → 1,2 |
| `munic_conselho_dh`: conselho municipal de direitos humanos | A442 | MDHU12 | 5,8 → 1,6 |
| `munic_conselho_lgbt`: conselho municipal de direitos LGBT | A482 | MDHU36 | 0,4 → 0,9 |
| `munic_programa_lgbt`: programa de promoção de direitos LGBT | A640 | MDHU576 | 6,2 → 11,4 |
| `munic_ouvidoria_dh`: ouvidoria de direitos humanos | A625 | MDHU731 | 5,8 → 8,3 |

**Regras de codificação:**
- "Sim" = 1 e "Não" = 0.
- "-" = 0: a pergunta não se aplica porque o município não tem a estrutura.
- "Recusa" e "Não informou" ficaram ausentes e depois foram preenchidos com a resposta do mesmo município na outra edição (32 linhas).
- Órgão gestor: 0 se "Não possui estrutura", 1 para qualquer forma de estrutura.

**Ressalva de comparabilidade:** o conselho municipal de direitos humanos cai de 5,8% (2014) para 1,6% (2019) e sobe para 2,1% em 2023. Uma queda dessa ordem em cinco anos indica mudança no enunciado ou no critério da pergunta, não uma mudança real. Por isso, `munic_conselho_dh` não deve ser interpretada como série temporal. As demais variáveis têm trajetória coerente entre as edições.

**Índice de política LGBT** (`munic_indice_lgbt`, de 0 a 4): soma de tema LGBT, nome social, conselho LGBT e programa LGBT.

**Atribuição por ano:**
- A edição de 2014 vale para 2011–2018. Não há edição anterior com essas perguntas, então 2011–2013 recebe 2014 por aproximação.
- A edição de 2019 vale para 2019–2022.
- A de 2023 é posterior ao período e fica só no arquivo `munic_harmonizada.csv`, para análises de robustez.

## 5. Montagem do painel

1. **Base:** todos os municípios × anos de 2011 a 2022. Seis municípios saíram por não terem população em todos os anos: cinco criados em 2013 (Mojuí dos Campos/PA, Pescaria Brava/SC, Balneário Rincão/SC, Pinto Bandeira/RS e Paraíso das Águas/MS) e um criado depois do Censo 2022. Ficam 5.565 municípios.
2. **Disque 100:** os registros da base limpa foram agregados por município e ano. Município sem registro no ano recebe **0**. Os 2,4% de registros sem município ficam fora do painel, mas continuam nas análises nacionais.
3. **Contagem trans comparável ao total LGBT** (decisão do grupo):
   - `registros_lgbt_janela_trans` conta só os registros LGBT feitos em períodos em que a vítima trans era identificável.
   - `prop_trans` = `registros_trans` ÷ `registros_lgbt_janela_trans`.
   - `cobertura_trans` = 1 na maioria dos anos, 0,5 em 2020 (só jan–jun) e 0 em 2021, quando a contagem trans fica ausente.
4. **Inflação:** PIB e renda foram convertidos para R$ de julho de 2022 pelo IPCA (SIDRA 1737), usando julho como mês de referência, que é o mesmo dos Censos.
5. **Variáveis censitárias:** densidade, renda, % de pretos e pardos e % de 15 a 29 anos foram interpoladas linearmente entre 2010 e 2022. A densidade de 2010 usa a área de 2022. IDHM, Gini (2010) e casais do mesmo sexo (2022) ficam fixos no período.
6. **PIB negativo:** um caso, resultado de impostos líquidos negativos num município muito pequeno. Foi tratado como ausente e interpolado com os anos vizinhos do mesmo município.
7. **Processo Transexualizador:** o arquivo do SIA/SIH traz o número de procedimentos por município e ano. Foram criadas duas variáveis:
   - `processo_transexualizador` = 1 no ano em que o município teve ao menos um procedimento. Não foi usado "ativo a partir do primeiro ano" porque há interrupções: Goiânia teve produção de 2010 a 2016, nenhuma de 2017 a 2022 e voltou em 2023.
   - `procedimentos_pt`: volume de procedimentos no ano, 0 se nenhum.

   Dos 26 municípios com serviço, só **10 tiveram atividade dentro de 2011–2022**: Porto Alegre, São Paulo, Goiânia, Rio de Janeiro, Recife, Curitiba, Uberlândia, Salvador, Vitória e João Pessoa. Os outros 16 começaram em 2023 ou depois e entram no painel com 0. O número de municípios ativos por ano vai de 3 (2011) a 9 (2020–2022).
8. **Marcadores de tempo:**
   - `pos_stf` = 1 em 2020–2022, os anos completos após a decisão de 13/06/2019.
   - `ano_stf_parcial` = 1 em 2019.
   - `pandemia` = 1 em 2020–2021.

**Conferência:** o painel soma 26.772 registros LGBT e 4.406 registros trans, os mesmos números da base limpa, descontados os registros sem município. A proporção trans no período identificável é de **17,8%** dos registros LGBT.

## 6. Codificação para os modelos (`matriz_painel_ml.csv`)

Mesma lógica aplicada ao Disque 100, adaptada a variáveis contínuas:

| Tipo | Tratamento | Variáveis |
|---|---|---|
| Contínuas assimétricas | log(1 + x) e depois z-score (média 0, desvio 1) | população, densidade, PIB per capita, renda per capita, taxa de homicídios, procedimentos do Processo Transexualizador |
| Contínuas | z-score | IDHM, Gini, % de 15 a 29 anos, % de pretos e pardos, % de casais do mesmo sexo, índice de política LGBT |
| Binárias | mantidas em 0/1 | 7 variáveis da MUNIC, capital, Processo Transexualizador, `pos_stf`, `ano_stf_parcial`, `pandemia` |
| Categóricas | one-hot | região (referência: Sudeste) e UF (27 colunas) |
| Exposição | `offset_log_pop` = log(população) | termo de exposição dos modelos de contagem |
| Tempo | `ano_centrado` = ano − 2016 | tendência |
| Respostas | sem transformação | `registros_lgbt`, `registros_trans`, `registros_lgbt_janela_trans`, `prop_trans` |

Resultado: 66.780 × 65. Depois da codificação, só ficam ausentes as respostas trans de 2021, por definição.

**Uso de região e UF:** os dois conjuntos de colunas não devem entrar juntos no mesmo GLM, porque um é combinação do outro. Use região no GLM e UF nos modelos de árvore.

## 7. Diagnósticos para a modelagem

**Excesso de zeros:** 91,2% das linhas município × ano têm zero registros, e 3.495 dos 5.565 municípios (63%) nunca registraram nenhuma denúncia LGBT no período.

**Sobredispersão:** a variância de `registros_lgbt` é 44,6 vezes a média. Uma regressão de Poisson pura subestimaria os erros-padrão. O modelo de referência deve ser **Binomial Negativa**, comparado com uma versão inflada de zeros. Nos modelos de árvore, usar perda de Poisson ou Tweedie.

**Colinearidade:** correlação de Spearman e VIF entre os preditores contínuos, calculados em 2022:

| Par | ρ de Spearman |
|---|---|
| renda per capita × IDHM | 0,90 |
| PIB per capita × renda per capita | 0,77 |
| renda per capita × % de pretos e pardos | −0,75 |
| IDHM × PIB per capita | 0,72 |

| Variável | VIF |
|---|---|
| renda per capita | 8,4 |
| IDHM | 7,0 |
| todas as demais | ≤ 3,0 |

**Recomendação:** no GLM base, manter a renda per capita e deixar o IDHM para um modelo alternativo. Nos modelos de árvore, a colinearidade não atrapalha a predição, só a leitura da importância das variáveis.

O proxy de população LGBT (% de casais do mesmo sexo) se correlaciona moderadamente com população (0,46), densidade (0,42) e IDHM (0,42), sem redundância forte. Ele traz informação própria.

## 8. Pendências e limitações

| Item | Situação | Impacto |
|---|---|---|
| Processo Transexualizador | Resolvido com os dados do SIA/SIH, mas só 10 municípios têm serviço ativo no período, todos capitais ou cidades grandes. | O efeito do serviço se confunde com o porte da cidade. A comparação deve controlar população e capital, ou se limitar a municípios parecidos, como só as capitais. |
| MUNIC 2011–2013 | Usa a edição de 2014. | Pode superestimar a política municipal nesses anos. |
| Variáveis censitárias | Interpolação linear entre 2010 e 2022. | Não captura choques entre os Censos. |
| Casais do mesmo sexo | Só existe para 2022 no nível municipal. | Fica como característica fixa do município. |
| Registros sem município | 2,4% ficam fora do painel. | Pequena subcontagem municipal. |
| Denúncia ≠ ocorrência | O Disque 100 mede violência e disposição/capacidade de denunciar ao mesmo tempo. | É o que motiva os controles institucionais e de violência geral. |

## 9. Arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `painel_municipio_ano.csv` | painel legível, 66.780 × 39 |
| `matriz_painel_ml.csv` | painel codificado para os modelos, 66.780 × 65 |
| `processo_transexualizador.csv` | procedimentos do SIA/SIH por município e ano (2008–2026), formato longo |
| `munic_harmonizada.csv` | 7 variáveis da MUNIC + índice LGBT, nas três edições |
| `dicionario_painel.csv` | descrição das colunas |
| `base_limpa.csv`, `matriz_ml.csv` | Disque 100 por registro (limpo e codificado) |
| `serie_mensal.csv` | série mensal LGBT e trans, para a análise do STF |
