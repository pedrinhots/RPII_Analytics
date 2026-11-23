# Resumo dos resultados: para apresentar ao grupo

**Em uma frase:** conseguimos identificar quais cidades têm violência LGBT notificada acima do esperado e prever, melhor que o acaso, onde podem surgir ondas de denúncias. Também mostramos que a violência contra pessoas trans é mais grave e mais pública, e que os dados não permitem medir o efeito da decisão do STF.

---

## O que tínhamos

- **27.441 denúncias** de violência contra pessoas LGBT no Disque 100, de 2011 a 2022, em 2.070 municípios.
- **Dados de todos os 5.565 municípios:** população, renda, PIB, homicídios, casais do mesmo sexo, políticas municipais LGBT (MUNIC) e Processo Transexualizador (SUS).
- **Um problema de fundo:** o Disque 100 trocou de sistema e de formulário várias vezes. Os grandes saltos no total nacional vêm dessas trocas, não da violência. Por isso, comparamos as cidades entre si, e não o total do país ao longo do tempo.

---

## As quatro análises

### 1. Onde estão os números alarmantes e quais cidades tendem a ter ondas (análise principal)

**Como fizemos:** para cada cidade, calculamos quantas denúncias seriam esperadas pelo tamanho da população e comparamos com o que aconteceu de fato.

| Algoritmo | Para que serviu | O que trouxe |
|---|---|---|
| **Suavização bayesiana** | Evitar falso alarme em cidades pequenas (1 denúncia numa cidade de 3 mil habitantes parece enorme) | Lista de **141 cidades alarmantes**, a maioria pequena ou média |
| **Getis-Ord (Gi\*)** | Ver se as cidades alarmantes formam regiões | **61 cidades em regiões de risco:** Zona da Mata e Sul de Minas, região metropolitana do Rio, interior de SP, Goiás e entorno de João Pessoa |
| **Modelo endêmico-epidêmico** (vem da vigilância de epidemias) | Prever e explicar: separa o que vem da estrutura da cidade, da inércia (o próprio histórico) e do contágio dos vizinhos | 74% vem da estrutura, 26% da inércia e **só 0,3% dos vizinhos**: as ondas são locais |
| **LightGBM** (aprendizado de máquina com árvores) | Ter a previsão mais precisa possível | **Melhor modelo:** erro 17% menor que a regra "só população". Acerta 37% das cidades que aponta como risco de onda, contra 15% da regra simples, ou seja, **2,5 vezes melhor** |
| **Extra Trees** | Comparar com o método usado na literatura | Funciona (erro 11% menor), mas fica abaixo dos outros dois |
| **Alerta de onda** (lógica de vigilância de surtos) | Marcar a cidade quando o número observado sai do que o modelo considerava plausível | De 22 a 52 alertas por ano. Uma cidade em alerta tem **quase 6 vezes** mais chance de entrar em alerta de novo no ano seguinte |

**O que aumenta as denúncias:**
- mais casais do mesmo sexo (+13%), um indicador do tamanho da população LGBT visível;
- maior densidade (+12%);
- **ter ouvidoria municipal de direitos humanos (+23%):** um canal de denúncia ajuda a denunciar.

**O que não fez diferença:** renda, PIB, homicídios e Processo Transexualizador.

### 2. A violência atinge mais as pessoas trans em certas cidades?

**Como fizemos:** regressão binomial sobre a fração de vítimas trans entre as denúncias LGBT de cada cidade.

**O que encontramos:**
- **18%** das denúncias LGBT têm vítima trans.
- A fração é maior no **Nordeste** (+54% de chance).
- As cidades alarmantes **não** têm mais vítimas trans que as outras: as ondas atingem a comunidade LGBT como um todo.
- O Processo Transexualizador não alterou essa proporção.

### 3. A violência contra pessoas trans é diferente?

**Como fizemos:** regressão logística sobre cada denúncia, perguntando o que torna mais provável que a vítima seja trans, conferida com o LightGBM.

**O que encontramos:** a violência contra pessoas trans é **mais grave e mais pública**.
- **Violação contra a vida:** chance 3 vezes maior.
- **Violência sexual:** 1,7 vez.
- **Agressor desconhecido:** 1,7 vez.
- **Agressor profissional ou institucional:** 2,3 vezes.
- **Acontece mais na rua e menos em casa.**
- **Os registros de vítimas trans são mais incompletos:** falta raça/cor em 30% deles, contra 16% nos demais.

### 4. A decisão do STF (junho de 2019) mudou as denúncias?

**Como fizemos:** série temporal interrompida, que compara o que aconteceu depois da decisão com o que a tendência anterior previa, mais um **teste de placebo** que testa a mesma quebra em todos os meses de 2018 e 2019.

**O que encontramos:**
- **No total LGBT, nenhuma mudança.**
- **Nas denúncias trans, houve uma queda, mas ela começou em fevereiro de 2019, quatro meses antes da decisão.** Ela coincide com o sumiço da categoria "travesti" nos registros.
- **Conclusão:** não dá para medir o efeito do STF com o Disque 100. As mudanças de registro pesam mais do que a lei.

---

## Os algoritmos "aprenderam"? Rodaram instantaneamente?

**Aprenderam, sim, e não foi instantâneo.** O processo é que foi rápido, porque os dados são pequenos para o computador (no máximo 55 mil linhas por treino).

- **LightGBM:** monta árvores de decisão uma a uma, cada uma corrigindo os erros das anteriores. Usamos de **130 a 942 árvores** por treino. Uma parte dos dados ficava separada para conferência, e o treino **parava sozinho** quando o erro nela deixava de cair, para o modelo não "decorar" os dados.
- **Modelo endêmico-epidêmico:** um otimizador ajusta os 24 parâmetros passo a passo, em **70 a 95 iterações** por treino, até não conseguir melhorar mais.
- **Extra Trees:** treina **300 árvores** e faz a média delas.
- **Validação honesta:** cada modelo foi treinado **7 vezes**, sempre aprendendo só com o passado e prevendo o ano seguinte (2016 a 2022), como se o futuro fosse desconhecido.
- **Tempo total:** a análise principal levou **1 min e 29 s**; as outras três, de 1,5 a 11 segundos.


## Por que confiar no modelo de aprendizado de máquina (aprofundamento da Análise 1)

- **Ele aprende e sabe parar:** o erro nos dados de treino cai sempre, mas o erro no ano reservado para conferência cai até um ponto e depois volta a subir (até +11%). O treino é interrompido exatamente nesse ponto, o que evita que o modelo "decore" o passado. *(Figura das curvas de aprendizado.)*
- **O ganho não é sorte:** reamostrando os municípios 2.000 vezes, o ganho do LightGBM fica entre **13% e 20%**. Ele vence a regra simples em **7 de 7 anos** e o modelo estatístico em 6 de 7.
- **Não depende de ajuste fino:** mudando os parâmetros do algoritmo, o ganho fica sempre entre 15,7% e 17,3%.
- **De onde vem o ganho:** as características da cidade e o seu histórico, juntos. Os vizinhos quase não contribuem (+0,15 ponto), o que confirma que as ondas são locais.
- **É útil para priorizar:** olhando só 1% dos municípios por ano (~56 cidades) escolhidos pelo modelo, capturamos **31% de todas as ondas**, contra 22% escolhendo pela população.
- **O que ele aprendeu faz sentido:** histórico de denúncias acima do esperado, mais casais do mesmo sexo e maior densidade aumentam a previsão, as mesmas relações do modelo estatístico, encontradas por outro caminho.
- **Com quantos dados aprende:** o LightGBM chega perto do seu melhor com só 2 a 3 anos de dados. O modelo estatístico precisa de séries mais longas para alcançá-lo.

---

## O que **não** podemos afirmar

- **Mais denúncia não é, necessariamente, mais violência:** pode ser mais facilidade para denunciar. O efeito da ouvidoria sugere que parte é isso.
- **Não prevemos o total do Brasil:** ele depende de mudanças no próprio Disque 100. Prevemos **onde**, dado um volume nacional.
- **O Processo Transexualizador tem pouco poder estatístico:** só 10 cidades tinham o serviço ativo no período.
- **O efeito do STF não é mensurável com esses dados.**

## Próximos passos

1. Revisão crítica: o que cada análise responde e o que fica em aberto.
2. Conclusão do relatório e resumo.
3. Revisar a palavra-chave "SINAN" no relatório (a base é o Disque 100).
