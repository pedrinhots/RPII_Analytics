"""
Montagem do painel município x ano (2011-2022) e codificação para os modelos
ACH0042 - Grupo 11 - EACH-USP

Uso:
    python montar_painel.py <pasta dados_municipais> <pasta com (disque100_)base_limpa.csv> <pasta de saida>
    Ex. (dentro de dados_municipais):  python scripts/montar_painel.py . painel painel
    [opcional] processo_transexualizador.csv na pasta dados_municipais, com colunas
               cod_mun_ibge7;ano_inicio  (município e ano em que passou a ter serviço habilitado)

Entradas:
    dados_municipais/  painel_anual.csv, censo_2010_2022.csv, municipios.csv,
                       brutos/MUNIC_2014.zip, MUNIC_2019.xlsx, MUNIC_2023.xlsx, atlas_censo_1991_2010.xlsx
    saida da limpeza:  base_limpa.csv (gerada por limpeza_disque100.py)

Saídas:
    painel_municipio_ano.csv   painel completo, legível (1 linha por município x ano)
    matriz_painel_ml.csv       painel codificado (one-hot + log + z-score), pronto para os modelos
    munic_harmonizada.csv      as 7 variáveis da MUNIC nas três edições (2014, 2019, 2023)
    dicionario_painel.csv      descrição de cada coluna
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ANOS = list(range(2011, 2023))
# IPCA, número-índice de julho (SIDRA 1737, v2266): mês de referência dos Censos
IPCA_JUL = {2010: 3111.05, 2011: 3324.86, 2012: 3497.70, 2013: 3717.03, 2014: 3958.72, 2015: 4337.11,
            2016: 4715.99, 2017: 4843.87, 2018: 5061.11, 2019: 5224.18, 2020: 5344.63, 2021: 5825.37,
            2022: 6411.95}
DEFLATOR = {a: IPCA_JUL[2022] / v for a, v in IPCA_JUL.items()}  # leva valores do ano a para R$ de 2022

# Variáveis da MUNIC presentes nas três edições: nome harmonizado -> (código 2014, código 2019/2023)
MUNIC_VARS = {
    "munic_estrutura_dh":   ("A392", "MDHU01"),    # órgão gestor de direitos humanos (≠ "não possui estrutura")
    "munic_tema_lgbt":      ("A428", "MDHU109"),   # órgão trata proteção dos direitos LGBT
    "munic_nome_social":    ("A430", "MDHU1010"),  # órgão trata reconhecimento do nome social
    "munic_conselho_dh":    ("A442", "MDHU12"),    # conselho municipal de direitos humanos
    "munic_conselho_lgbt":  ("A482", "MDHU36"),    # conselho municipal de direitos LGBT
    "munic_programa_lgbt":  ("A640", "MDHU576"),   # política/programa de promoção de direitos LGBT
    "munic_ouvidoria_dh":   ("A625", "MDHU731"),   # ouvidoria de direitos humanos (canal de denúncia)
}
INDICE_LGBT = ["munic_tema_lgbt", "munic_nome_social", "munic_conselho_lgbt", "munic_programa_lgbt"]


def ler_csv(p, **kw):
    return pd.read_csv(p, sep=";", encoding="utf-8-sig", dtype={"cod_mun_ibge7": str, "cod_mun_ibge6": str}, **kw)


# ---------------------------------------------------------------------------
# MUNIC
# ---------------------------------------------------------------------------
def sim_nao(s, estrutura=False):
    s = s.astype(str).str.strip()
    if estrutura:
        out = np.where(s.str.contains("Não possui", case=False), 0, 1)
        return pd.Series(np.where(s.str.contains("informou|informado|Recusa|nan", case=False), np.nan, out), index=s.index)
    # "-" = pergunta não se aplica (município sem estrutura/conselho) -> 0
    return s.map({"Sim": 1, "Não": 0, "-": 0}).astype(float)


def ler_munic(pasta_brutos, tmp):
    zipfile.ZipFile(pasta_brutos / "MUNIC_2014.zip").extractall(tmp)
    x14 = pd.ExcelFile(next(Path(tmp).rglob("*.xls")), engine="calamine")
    m14 = pd.concat([pd.read_excel(x14, sheet_name=s, dtype=str).set_index("A1")
                     for s in ("Direitos humanos 1", "Direitos humanos 2")], axis=1)
    m19 = pd.read_excel(pasta_brutos / "MUNIC_2019.xlsx", sheet_name="Direitos humanos", dtype=str,
                        engine="calamine").set_index("CodMun")
    m23 = pd.read_excel(pasta_brutos / "MUNIC_2023.xlsx", sheet_name="Direitos Humanos", dtype=str,
                        engine="calamine").set_index("Cod Munic")
    edicoes = []
    for ano, df, idx in ((2014, m14, 0), (2019, m19, 1), (2023, m23, 1)):
        e = pd.DataFrame(index=df.index.astype(str).str[:6])
        for nome, cods in MUNIC_VARS.items():
            e[nome] = sim_nao(df[cods[idx]], estrutura=(nome == "munic_estrutura_dh")).values
        e["munic_indice_lgbt"] = e[INDICE_LGBT].sum(axis=1, min_count=1)  # 0 a 4
        e["munic_edicao"] = ano
        edicoes.append(e.rename_axis("cod_mun_ibge6").reset_index())
    return pd.concat(edicoes, ignore_index=True)


# ---------------------------------------------------------------------------
# Atlas Brasil (2010)
# ---------------------------------------------------------------------------
def ler_atlas(pasta_brutos):
    a = pd.read_excel(pasta_brutos / "atlas_censo_1991_2010.xlsx", sheet_name="MUN 91-00-10", engine="calamine",
                      usecols=["ANO", "Codmun7", "IDHM", "GINI"])
    a = a[a["ANO"] == 2010]
    return pd.DataFrame({"cod_mun_ibge7": a["Codmun7"].astype("Int64").astype(str),
                         "idhm_2010": a["IDHM"].values, "gini_2010": a["GINI"].values})


# ---------------------------------------------------------------------------
# Disque 100 agregado por município x ano
# ---------------------------------------------------------------------------
def agregar_disque100(base_limpa):
    b = ler_csv(base_limpa, usecols=["cod_mun_ibge6", "ano", "is_trans", "trans_identificavel"])
    b = b.dropna(subset=["cod_mun_ibge6"])
    b["trans"] = b["is_trans"].fillna(0)
    g = b.groupby(["cod_mun_ibge6", "ano"])
    d = pd.DataFrame({
        "registros_lgbt": g.size(),
        # denominador comparável: só registros de períodos em que dava para identificar vítima trans
        "registros_lgbt_janela_trans": g["trans_identificavel"].sum(),
        "registros_trans": g["trans"].sum(),
    }).reset_index()
    return d


# ---------------------------------------------------------------------------
def interpolar(v2010, v2022, ano):
    return v2010 + (v2022 - v2010) * (ano - 2010) / 12


def montar(pasta_dados, pasta_disque, pasta_saida):
    pasta_dados, pasta_saida = Path(pasta_dados), Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    brutos = pasta_dados / "brutos"

    mun = ler_csv(pasta_dados / "municipios.csv")
    painel = ler_csv(pasta_dados / "painel_anual.csv")
    censo = ler_csv(pasta_dados / "censo_2010_2022.csv")

    # municípios sem população em algum ano (criados após o Censo 2022) saem do painel
    sem_pop = painel.groupby("cod_mun_ibge7")["populacao"].apply(lambda s: s.isna().any())
    excluidos = sem_pop[sem_pop].index.tolist()
    print(f"municípios excluídos por falta de população: {excluidos}")
    p = painel[~painel["cod_mun_ibge7"].isin(excluidos)].copy()
    p = p.merge(mun[["cod_mun_ibge7", "municipio", "uf", "regiao", "capital"]], on="cod_mun_ibge7", how="left")

    # --- valores monetários em R$ de 2022 ---
    p["pib_per_capita_r2022"] = p["pib_per_capita"] * p["ano"].map(DEFLATOR)
    # PIB negativo (impostos líquidos negativos em município muito pequeno) -> interpola com anos vizinhos
    neg = p["pib_per_capita_r2022"] <= 0
    print(f"PIB per capita <= 0 tratado como ausente e interpolado: {int(neg.sum())} caso(s)")
    p.loc[neg, "pib_per_capita_r2022"] = np.nan
    p["pib_per_capita_r2022"] = (p.sort_values("ano").groupby("cod_mun_ibge7")["pib_per_capita_r2022"]
                                  .transform(lambda s: s.interpolate(limit_direction="both")))

    # --- variáveis censitárias interpoladas (2010 -> 2022) ---
    c = censo.set_index("cod_mun_ibge7")
    c["densidade_2010"] = c["pop_total_2010"] / c["area_km2_2022"]
    c["renda_pc_r2022_2010"] = c["renda_pc_2010"] * DEFLATOR[2010]
    c["renda_pc_r2022_2022"] = c["renda_pc_2022"]
    pares = {"densidade": ("densidade_2010", "densidade_2022"),
             "renda_pc_r2022": ("renda_pc_r2022_2010", "renda_pc_r2022_2022"),
             "pct_pretos_pardos": ("pct_pretos_pardos_2010", "pct_pretos_pardos_2022"),
             "pct_15_29": ("pct_15_29_2010", "pct_15_29_2022")}
    for nome, (c10, c22) in pares.items():
        v10 = p["cod_mun_ibge7"].map(c[c10])
        v22 = p["cod_mun_ibge7"].map(c[c22])
        v10 = v10.fillna(v22)  # municípios criados após 2010: usa o valor de 2022
        p[nome] = interpolar(v10, v22, p["ano"])
    p["pct_casais_mesmo_sexo_2022"] = p["cod_mun_ibge7"].map(c["pct_casais_mesmo_sexo_2022"])
    p["area_km2"] = p["cod_mun_ibge7"].map(c["area_km2_2022"])

    # --- Atlas Brasil ---
    p = p.merge(ler_atlas(brutos), on="cod_mun_ibge7", how="left")

    # --- MUNIC: 2014 vale para 2011-2018; 2019 para 2019-2022 (2023 fica só no arquivo harmonizado) ---
    munic = ler_munic(brutos, pasta_saida / "_tmp_munic")
    munic.to_csv(pasta_saida / "munic_harmonizada.csv", index=False, sep=";", encoding="utf-8-sig")
    p["munic_edicao"] = np.where(p["ano"] <= 2018, 2014, 2019)
    p = p.merge(munic[munic["munic_edicao"] < 2023], on=["cod_mun_ibge6", "munic_edicao"], how="left")
    # sem resposta numa edição (recusa / não informou): usa a resposta da outra edição do mesmo município
    for col in [*MUNIC_VARS, "munic_indice_lgbt"]:
        n0 = p[col].isna().sum()
        p[col] = p[col].fillna(p.groupby("cod_mun_ibge7")[col].transform("max"))
        print(f"{col}: {n0} ausentes -> {p[col].isna().sum()} após usar a outra edição")

    # --- Processo Transexualizador (SIA/SIH-SUS: procedimentos por município e ano) ---
    # serviço "ativo" no ano = houve ao menos 1 procedimento (Goiânia, p.ex., parou em 2017-2022)
    arq_pt = pasta_dados / "processo_transexualizador.csv"
    if arq_pt.exists():
        pt = ler_csv(arq_pt)[["cod_mun_ibge6", "ano", "procedimentos_pt"]]
        p = p.merge(pt, on=["cod_mun_ibge6", "ano"], how="left")
        p["procedimentos_pt"] = p["procedimentos_pt"].fillna(0)
        p["processo_transexualizador"] = (p["procedimentos_pt"] > 0).astype(int)
        print(f"Processo Transexualizador: {p.loc[p.processo_transexualizador == 1, 'cod_mun_ibge7'].nunique()} "
              f"municípios ativos em algum ano do período")
    else:
        print("AVISO: processo_transexualizador.csv não encontrado; colunas preenchidas com NaN")
        p["procedimentos_pt"] = np.nan
        p["processo_transexualizador"] = np.nan

    # --- Disque 100 ---
    pasta_disque = Path(pasta_disque)
    arq_base = next(f for f in (pasta_disque / "disque100_base_limpa.csv", pasta_disque / "base_limpa.csv") if f.exists())
    d = agregar_disque100(arq_base)
    p = p.merge(d, on=["cod_mun_ibge6", "ano"], how="left")
    for col in ("registros_lgbt", "registros_lgbt_janela_trans", "registros_trans"):
        p[col] = p[col].fillna(0)
    # cobertura da identificação trans no ano: 1 = ano todo; 0,5 = só jan-jun/2020; 0 = 2021
    p["cobertura_trans"] = p["ano"].map(lambda a: 0.5 if a == 2020 else 0.0 if a == 2021 else 1.0)
    p.loc[p["cobertura_trans"] == 0, ["registros_trans", "registros_lgbt_janela_trans"]] = np.nan
    p["prop_trans"] = p["registros_trans"] / p["registros_lgbt_janela_trans"].replace(0, np.nan)
    p["taxa_lgbt_100mil"] = p["registros_lgbt"] / p["populacao"] * 1e5
    p["pos_stf"] = (p["ano"] >= 2020).astype(int)        # 1º ano completo após 13/06/2019
    p["ano_stf_parcial"] = (p["ano"] == 2019).astype(int)
    p["pandemia"] = p["ano"].isin([2020, 2021]).astype(int)

    cols = ["cod_mun_ibge7", "cod_mun_ibge6", "municipio", "uf", "regiao", "capital", "ano",
            # resposta
            "registros_lgbt", "registros_trans", "registros_lgbt_janela_trans", "prop_trans", "taxa_lgbt_100mil",
            "cobertura_trans",
            # exposição e socioeconômicas
            "populacao", "area_km2", "densidade", "pib_per_capita_r2022", "renda_pc_r2022", "idhm_2010", "gini_2010",
            # demografia e proxy LGBT
            "pct_15_29", "pct_pretos_pardos", "pct_casais_mesmo_sexo_2022",
            # violência
            "homicidios", "taxa_homicidios_100mil",
            # institucionais
            *MUNIC_VARS, "munic_indice_lgbt", "munic_edicao", "processo_transexualizador", "procedimentos_pt",
            # tempo
            "pos_stf", "ano_stf_parcial", "pandemia"]
    p = p[cols].sort_values(["cod_mun_ibge7", "ano"]).reset_index(drop=True)
    p.to_csv(pasta_saida / "painel_municipio_ano.csv", index=False, sep=";", encoding="utf-8-sig")

    m = codificar(p)
    m.to_csv(pasta_saida / "matriz_painel_ml.csv", index=False, sep=";", encoding="utf-8-sig")
    pd.DataFrame(DICIONARIO.items(), columns=["coluna", "descricao"]).to_csv(
        pasta_saida / "dicionario_painel.csv", index=False, sep=";", encoding="utf-8-sig")
    print(f"painel: {p.shape} | matriz: {m.shape}")
    return p, m


# ---------------------------------------------------------------------------
# Codificação para os modelos
# ---------------------------------------------------------------------------
ASSIMETRICAS = ["populacao", "densidade", "pib_per_capita_r2022", "renda_pc_r2022", "taxa_homicidios_100mil",
               "procedimentos_pt"]
CONTINUAS = ["idhm_2010", "gini_2010", "pct_15_29", "pct_pretos_pardos", "pct_casais_mesmo_sexo_2022",
             "munic_indice_lgbt"]


def codificar(p):
    m = p[["cod_mun_ibge7", "ano", "registros_lgbt", "registros_trans", "registros_lgbt_janela_trans",
           "prop_trans", "cobertura_trans", "populacao"]].copy()
    m["offset_log_pop"] = np.log(p["populacao"])  # termo de exposição dos modelos de contagem
    # log(1+x) nas assimétricas, depois z-score; contínuas só z-score
    for c in ASSIMETRICAS:
        m[f"z_log_{c}"] = zscore(np.log1p(p[c]))
    for c in CONTINUAS:
        m[f"z_{c}"] = zscore(p[c])
    # binárias ficam 0/1
    for c in [*MUNIC_VARS, "capital", "processo_transexualizador", "pos_stf", "ano_stf_parcial", "pandemia"]:
        m[c] = p[c]
    m["ano_centrado"] = p["ano"] - 2016
    # one-hot de região (referência: Sudeste) e UF
    m = pd.concat([m, pd.get_dummies(p["regiao"], prefix="regiao", dtype=int).drop(columns="regiao_Sudeste"),
                   pd.get_dummies(p["uf"], prefix="uf", dtype=int)], axis=1)
    return m.round(6)


def zscore(s):
    return (s - s.mean()) / s.std()


DICIONARIO = {
    "registros_lgbt": "Registros (denúncia x vítima) LGBT do Disque 100 no município e ano; 0 se nenhum",
    "registros_trans": "Registros com vítima travesti/transexual/transgênero; vazio em 2021 (não identificável)",
    "registros_lgbt_janela_trans": "Registros LGBT feitos em períodos em que a vítima trans era identificável (denominador de prop_trans)",
    "prop_trans": "registros_trans / registros_lgbt_janela_trans; vazio se não houve registro",
    "taxa_lgbt_100mil": "Registros LGBT por 100 mil habitantes",
    "cobertura_trans": "Fração do ano com identificação de vítima trans: 1; 0,5 em 2020 (jan-jun); 0 em 2021",
    "populacao": "População: estimativas IBGE (SIDRA 6579) 2011-2021; Censo 2022 (SIDRA 4714)",
    "densidade": "hab/km²; interpolação linear entre 2010 (pop. Censo 2010 / área 2022) e 2022 (SIDRA 4714)",
    "pib_per_capita_r2022": "PIB per capita (SIDRA 5938 / população), em R$ de jul/2022 pelo IPCA",
    "renda_pc_r2022": "Renda domiciliar per capita (SIDRA 3578 em 2010, 10295 em 2022) em R$ de 2022, interpolada",
    "idhm_2010": "IDHM 2010 (Atlas Brasil), fixo no período",
    "gini_2010": "Índice de Gini 2010 (Atlas Brasil), fixo no período",
    "pct_15_29": "% da população com 15 a 29 anos (SIDRA 9606), interpolado 2010-2022",
    "pct_pretos_pardos": "% de pretos e pardos (SIDRA 9605), interpolado 2010-2022",
    "pct_casais_mesmo_sexo_2022": "% de domicílios com casal do mesmo sexo (SIDRA 9881, Censo 2022), fixo",
    "homicidios": "Homicídios registrados (Ipea, Atlas da Violência, série 328); 0 se o município não aparece",
    "taxa_homicidios_100mil": "Homicídios por 100 mil habitantes",
    "munic_*": "Variáveis da MUNIC (1 = Sim): edição 2014 para 2011-2018 e edição 2019 para 2019-2022",
    "munic_indice_lgbt": "Soma de tema LGBT + nome social + conselho LGBT + programa LGBT (0 a 4)",
    "processo_transexualizador": "1 se o município teve ao menos 1 procedimento do Processo Transexualizador no ano (SIA/SIH-SUS)",
    "procedimentos_pt": "Procedimentos do Processo Transexualizador no município e ano (SIA/SIH-SUS); 0 se nenhum",
    "pos_stf": "1 nos anos completos após a decisão do STF (2020-2022)",
    "ano_stf_parcial": "1 em 2019 (decisão em 13/06/2019)",
    "pandemia": "1 em 2020 e 2021",
    "offset_log_pop": "log(população): termo de exposição (offset) dos modelos de contagem",
    "z_log_*": "log(1+x) padronizado (média 0, desvio 1)",
    "z_*": "variável padronizada (média 0, desvio 1)",
    "regiao_*": "one-hot da região (referência: Sudeste)",
    "uf_*": "one-hot da UF",
}

if __name__ == "__main__":
    montar(*sys.argv[1:4])
