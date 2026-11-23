# Dados de entrada das análises

Pasta lida por `analises/comum.py` (`DADOS = analises/dados`).

| Arquivo | Linhas | Origem |
|---|---|---|
| `painel_municipio_ano.csv` | 66.780 (5.565 municípios × 2011–2022) | gerado por `montar_painel.py` |
| `base_limpa.csv` | 27.441 (vítima × denúncia) | gerado por `limpeza_disque100.py` |
| `serie_mensal.csv` | 144 meses (2011–2022) | gerado por `limpeza_disque100.py` |
| `municipios_coord.csv` | 5.570 municípios (código IBGE, latitude, longitude) | repositório público kelvins/municipios-brasileiros (GitHub) |
| `geo/municipios.geojson`, `geo/ufs.geojson` | malhas para o mapa da Figura 3 | tbrugz/geodata-br e codeforgermany/click_that_hood (GitHub); `figuras.py` baixa de novo se faltarem |

Separador dos CSV gerados: ponto e vírgula; codificação UTF-8 com BOM.
