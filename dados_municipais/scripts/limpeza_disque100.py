"""
Limpeza, unificação e codificação da base Disque 100 (recorte LGBT, 2011-2022)
ACH0042 - Grupo 11 - EACH-USP

Uso:
    python limpeza_disque100.py lgbt.csv saida/

Gera em saida/:
    base_limpa.csv           1 linha por registro (denúncia x vítima), categorias legíveis
    matriz_ml.csv            mesma base, só com números (one-hot / multi-hot), pronta para modelos
    contagem_municipio_ano.csv  registros LGBT e trans por município (cód. IBGE 6 dígitos) e ano
    serie_mensal.csv         registros LGBT e trans por mês (para a série temporal interrompida)
    dicionario_variaveis.csv descrição de cada coluna da base limpa

A base original junta 4 layouts do Disque 100:
    A) 2011-2019          1 linha = 1 registro; violações numa célula separadas por ';'
    B) 2020 jan-jun       1 linha = 1 registro; sem hash; município só em 'municipio denuncia'
    C) 2020 jul-2022      1 linha = 1 violação; várias linhas por 'hash' (denúncia)
O script harmoniza as colunas equivalentes, colapsa o layout C para 1 linha por
denúncia x vítima e mapeia todas as categorias para um vocabulário comum.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DESC = "desconhecido"  # categoria explícita para ausente (o ausente também é informação)
TOKENS_AUSENTE = {
    "", "NAN", "N/D", "NAO INFORMADO", "NAO SE APLICA", "NAO SABE INFORMAR",
    "DENUNCIANTE NAO SOUBE INFORMAR", "NAO SABE", "SEM INFORMACAO", "NULL",
}


# ----------------------------------------------------------------------------
# utilitários
# ----------------------------------------------------------------------------
def norm(s: pd.Series) -> pd.Series:
    """Maiúsculas, sem acento, sem espaços duplicados; tokens de ausência -> NaN."""
    def _n(x):
        if not isinstance(x, str):
            return np.nan
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
        x = re.sub(r"\s+", " ", x).strip().upper()
        return np.nan if x in TOKENS_AUSENTE else x
    return s.map(_n)


def mapear(s: pd.Series, regras: list[tuple[str, str]], outro="outro") -> pd.Series:
    """Aplica a primeira regex que casar; ausente -> 'desconhecido'; sem regra -> outro."""
    def _m(x):
        if not isinstance(x, str):
            return DESC
        for padrao, cat in regras:
            if re.search(padrao, x):
                return cat
        return outro
    return s.map(_m)


def coalesce(df, *cols):
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for c in cols:
        out = out.fillna(df[c])
    return out


# ----------------------------------------------------------------------------
# dicionários de harmonização
# ----------------------------------------------------------------------------
# macro-categorias de violação (vale para os dois vocabulários: "X.Y;..." e "A>B>C").
# Cada violação é classificada sozinha, pela PRIMEIRA regra que casar (a ordem importa:
# ex. "Discriminação por orientação sexual" não pode cair em violência sexual).
REGRAS_VIOLACAO = {
    "viol_vida": r"HOMICID|FEMINICID|SUICID|AUTOMUTILA|CRIME_VIDA|^VIDA>",
    "viol_sexual": r"VIOLENCIA SEXUAL|>SEXUAL>|ESTUPRO|ABUSO SEXUAL|ASSEDIO SEXUAL|EXPLORACAO SEXUAL|IMPORTUNACAO",
    "viol_institucional": r"INSTITUCIONAL|ABUSO DE AUTORIDADE|VIOLENCIA POLICIAL|OMISSAO|RECUSA DE ATENDIMENTO|"
                          r"^DIREITOS SOCIAIS|^DIREITOS CIVIS|AUSENCIA DE ACESSO",
    "viol_discriminacao": r"DISCRIMINA|RACISMO|RACIAL|HOMOFOBIA",
    "viol_patrimonial": r"PATRIMONI|FINANCEIRO",
    "viol_negligencia": r"NEGLIGENCIA|ABANDONO|INSUBSISTENCIA|SITUACAO DE RUA",
    "viol_liberdade": r"^LIBERDADE|CARCERE|ESCRAV|TRAFICO|SEQUESTRO|EXPLORACAO|AGRESSAO_DIREITO_LIBERDADE|LIBERDADE_CIVIL",
    "viol_fisica": r"FISICA|LESAO|AGRESSAO OU VIAS|VIAS DE FATO|MAUS TRATOS|TORTURA E OUTROS",
    "viol_psicologica": r"PSICOLOGICA|PSIQUICA|AMEACA|HUMILHA|HOSTILIZA|CONSTRANG|INJURIA|CALUNIA|"
                        r"DIFAMA|HONRA|PERSEGUI|STALKING|ASSEDIO MORAL|BULLYING|CHANTAGEM|EXPOSICAO",
}


def classificar_violacao(token: str) -> str:
    for col, padrao in REGRAS_VIOLACAO.items():
        if re.search(padrao, token):
            return col
    return "viol_outra"


REGRAS_IDADE_TEXTO = [
    (r"CRIANCA|ADOLESCENTE|NASCITURO|RECEM", "0-17"),
    (r"IDOSO|90\+|91 ANOS", "60+"),
]


def faixa_etaria(x):
    if not isinstance(x, str):
        return DESC
    for p, c in REGRAS_IDADE_TEXTO:
        if re.search(p, x):
            return c
    nums = re.findall(r"\d+", x)
    if not nums:
        return DESC
    lo = int(nums[0])  # limite inferior da faixa original
    for lim, cat in [(18, "0-17"), (25, "18-24"), (30, "25-29"), (40, "30-39"),
                     (50, "40-49"), (60, "50-59")]:
        if lo < lim:
            return cat
    return "60+"


REGRAS_RACA = [(r"PARD", "parda"), (r"PRET|NEGR", "preta"), (r"BRANC", "branca"),
               (r"AMAREL", "amarela"), (r"INDIGEN", "indigena")]

REGRAS_CENARIO = [
    (r"VIRTUAL|INTERNET", "virtual"),
    (r"CASA|RESID", "residencia"),
    (r"RUA|VIA PUBLICA|ONIBUS|TRANSPORTE|METRO|TAXI", "via_publica_transporte"),
    (r"TRABALHO", "trabalho"),
    (r"ESCOLA|ENSINO|CRECHE|BERCARIO", "ensino"),
    (r"HOSPITAL|SAUDE|MANICOMIO", "saude"),
    (r"DELEGACIA|PRISIONAL|ORGAO|ABRIG|ALBERGUE|ILPI|SOCIO|CENTRO DE REFERENCIA|IGREJA|INSTITUICAO",
     "instituicao_publica_ou_religiosa"),
    (r"BAR|LOJA|SHOPPING|LAZER|EVENTO|NOTURNA|CLUBE|CINEMA|COMERCIAL|CENTRO DE TREINAMENTO|FINANCEIRA",
     "comercio_lazer"),
]

REGRAS_RELACAO = [
    (r"^NAO SE APLICA|PROPRIA VITIMA", DESC),
    (r"DESCONHECID", "desconhecido_da_vitima"),
    (r"EX-|COMPANHEIR|NAMORAD|MARIDO|ESPOS", "parceiro_ou_ex"),
    (r"MAE|PAI|IRMA|TIO|PRIM|CUNHAD|PADRAST|MADRAST|SOGR|SOBRINH|FILH|AV[OÓ]|GENRO|NORA|"
     r"ENTEAD|NET[OA]|PADRINHO|FAMILI", "familia"),
    (r"VIZINH", "vizinho"),
    (r"AMIG", "amigo_conhecido"),
    (r"EMPREGA|PATRAO|COLEGA|SUBORDINADO|HIERARQUIC|PRESTADOR", "trabalho"),
    (r"PROFESSOR|DIRETOR|PROFISSIONAL|LIDER RELIGIOSO|FUNCIONARIO|CUIDADOR|GESTOR|EDUCACAO",
     "institucional_profissional"),
]

REGRAS_DENUNCIANTE = [(r"PROPRIA VITIMA", "propria_vitima"), (r"TERCEIRO", "terceiro"),
                      (r"ANONIMO", "anonimo")]

REGRAS_CANAL = [(r"TELEF", "telefone"), (r"WHATSAPP|TELEGRAM", "mensageiro"),
                (r"WEBCHAT|PORTAL|E-MAIL|MOBILE|APLICATIVO", "internet_app")]

REGRAS_ORIENTACAO = [
    (r"^NAO$", DESC),
    (r"HOMOSSEXUAL|GAY|LESBICA", "homossexual"), (r"BISSEXUAL", "bissexual"),
    (r"HETEROSSEXUAL", "heterossexual"), (r"PANSEXUAL|ASSEXUAL|OUTRO", "outra"),
]


# ----------------------------------------------------------------------------
# etapas
# ----------------------------------------------------------------------------
def carregar(caminho):
    df = pd.read_csv(caminho, sep=";", dtype=str, low_memory=False)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df["data_cadastro"] = pd.to_datetime(df["data de cadastro"].str[:19], format="mixed",
                                         dayfirst=True, errors="coerce")
    df["layout"] = np.select(
        [df["data_cadastro"] < "2020-01-01", df["hash"].isna()],
        ["A_2011_2019", "B_2020_s1"], default="C_2020s2_2022")
    print(f"linhas: {n0} -> {len(df)} após remover {n0 - len(df)} duplicatas exatas")
    return df


def identidade_e_orientacao(df):
    """Cria identidade_trans, is_trans, trans_identificavel, orientacao_sexual, sexo."""
    g = norm(df["genero da vitima"])
    o_antiga = norm(df["vitima_orientacao_sexual"])
    o_2020 = norm(df["orientacao_sexual"])
    o_nova = norm(df["orientacao sexual da vitima"])
    lay = df["layout"]

    ident = pd.Series(DESC, index=df.index)
    # A: trans registrado no campo de gênero
    a = lay == "A_2011_2019"
    ident[a & g.notna()] = "nao_trans"
    ident[a & (g == "TRAVESTI")] = "travesti"
    ident[a & (g == "TRANSEXUAL")] = "transexual"
    # B: trans registrado na coluna orientacao_sexual
    b = lay == "B_2020_s1"
    ident[b & o_2020.notna()] = "nao_trans"
    ident[b & (o_2020 == "TRAVESTI")] = "travesti"
    ident[b & (o_2020 == "TRANSEXUAL")] = "transexual"
    # C: só é possível identificar a partir de jul/2022 (categoria PESSOA TRANS.* na orientação)
    c22 = (lay == "C_2020s2_2022") & (df["data_cadastro"] >= "2022-07-01")
    ident[c22 & o_nova.notna()] = "nao_trans"
    ident[c22 & o_nova.str.contains("TRAVESTI", na=False)] = "travesti"
    ident[c22 & o_nova.str.contains("TRANSEXUAL", na=False)] = "transexual"
    ident[c22 & o_nova.str.contains("TRANSGENERO", na=False)] = "transgenero"
    ident[c22 & (g == "INTERSEXO")] = "intersexo"

    df["trans_identificavel"] = (a | b | c22).astype(int)
    df["identidade_trans"] = ident
    df["is_trans"] = np.where(ident.isin(["travesti", "transexual", "transgenero"]), 1.0,
                              np.where(ident == DESC, np.nan, 0.0))

    # orientação sexual: no layout A parte vem no campo de gênero (GAY/LESBICA/BISSEXUAL)
    orient_txt = coalesce(pd.DataFrame({"a": o_antiga, "b": o_2020, "c": o_nova,
                                        "g": g.where(g.isin(["GAY", "LESBICA", "BISSEXUAL", "HETEROSSEXUAL"]))}),
                          "a", "b", "c", "g")
    orient_txt = orient_txt.where(~orient_txt.str.contains("TRANS|TRAVESTI", na=False))
    df["orientacao_sexual"] = mapear(orient_txt, REGRAS_ORIENTACAO, outro=DESC)

    sexo_txt = coalesce(pd.DataFrame({"s": norm(df["vitima_sexo"]),
                                      "g": g.where(g.str.match(r"^(MASCULIN|FEMININ)", na=False))}), "s", "g")
    df["sexo_genero_vitima"] = mapear(sexo_txt, [(r"^MASC", "masculino"), (r"^FEM", "feminino")])
    df.loc[g.isin(["GAY"]) & (df["sexo_genero_vitima"] == DESC), "sexo_genero_vitima"] = "masculino"
    df.loc[g.isin(["LESBICA"]) & (df["sexo_genero_vitima"] == DESC), "sexo_genero_vitima"] = "feminino"
    return df


def municipio(df):
    """Código IBGE de 6 dígitos (sem dígito verificador) + UF."""
    cod = df["vitima_cod_municipio"].str.extract(r"^(\d{6})")[0]
    mun_txt = df["municipio"].fillna(df["municipio denuncia"])
    cod = cod.fillna(mun_txt.str.extract(r"^(\d{6}) \|")[0])
    df["fonte_municipio"] = np.where(df["municipio"].notna(), "municipio_vitima",
                                     np.where(df["municipio denuncia"].notna(), "municipio_denuncia", DESC))

    # nome limpo (remove "(UF)", "123456 | ") e UF
    nome = norm(mun_txt.str.replace(r"^\d{6} \|", "", regex=True)
                       .str.replace(r"\(\w{2}\)\s*$", "", regex=True))
    uf = df["uf"].fillna(df["uf denuncia"]).fillna(mun_txt.str.extract(r"\((\w{2})\)\s*$")[0])
    uf = norm(uf).where(lambda s: s.str.len() == 2)

    # completa códigos ausentes com o par (nome, UF) aprendido nos registros que têm código
    ref = (pd.DataFrame({"nome": nome, "uf": uf, "cod": cod}).dropna()
             .groupby(["nome", "uf"])["cod"].agg(lambda s: s.mode().iat[0]))
    faltando = cod.isna() & nome.notna() & uf.notna()
    chaves = list(zip(nome[faltando], uf[faltando]))
    cod.loc[faltando] = [ref.get(k, np.nan) for k in chaves]
    # a UF sai do código quando ele existe (2 primeiros dígitos = código da UF)
    UF_COD = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
              "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
              "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
              "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}
    df["cod_mun_ibge6"] = cod
    df["municipio_nome"] = nome
    df["uf"] = cod.str[:2].map(UF_COD).fillna(uf)
    return df


def violacoes_multi_hot(df):
    texto = norm(coalesce(df, "violacoes", "violacao"))
    cols = list(REGRAS_VIOLACAO) + ["viol_outra"]
    cache = {}

    def _cats(x):
        if not isinstance(x, str):
            return set()
        if x not in cache:
            cache[x] = {classificar_violacao(t.strip()) for t in re.split(r"[;,]", x) if t.strip()}
        return cache[x]

    cats = texto.map(_cats)
    for col in cols:
        df[col] = cats.map(lambda s, c=col: int(c in s))
    df["viol_desconhecida"] = texto.isna().astype(int)
    df["n_tipos_violacao"] = df[cols].sum(axis=1)
    return df


def demais_categorias(df):
    df["faixa_etaria"] = norm(coalesce(df, "faixa etaria da vitima", "idade_vitima")).map(faixa_etaria)
    df["raca_cor"] = mapear(norm(df["raca cor da vitima"]), REGRAS_RACA)
    df["cenario"] = mapear(norm(coalesce(df, "cenario_local", "cenario da violacao")), REGRAS_CENARIO)
    df["relacao_suspeito"] = mapear(norm(coalesce(df, "relacao vitima suspeito", "relacao_suspeito",
                                                   "Relação_Suspeito_x_Vítima")), REGRAS_RELACAO)
    df["denunciante"] = mapear(norm(df["denunciante"]), REGRAS_DENUNCIANTE)
    df["canal"] = mapear(norm(df["canal de atendimento"]), REGRAS_CANAL)
    return df


COLS_PERFIL = ["identidade_trans", "orientacao_sexual", "sexo_genero_vitima", "faixa_etaria",
               "raca_cor", "cod_mun_ibge6", "uf"]


def colapsar_layout_c(df):
    """Layout C: 1 linha por violação -> 1 linha por denúncia (hash) x perfil da vítima."""
    viol = [c for c in df.columns if c.startswith("viol_")]
    c = df["layout"] == "C_2020s2_2022"
    resto, lc = df[~c].copy(), df[c].copy()
    resto["id_registro"] = "R" + resto.index.astype(str)
    resto["n_linhas_origem"] = 1

    chave = ["hash"] + COLS_PERFIL
    lc[COLS_PERFIL] = lc[COLS_PERFIL].fillna("__na__")
    agg = {col: "max" for col in viol}
    outras = [col for col in resto.columns if col not in viol + chave + ["id_registro", "n_linhas_origem"]]
    agg.update({col: "first" for col in outras if col in lc.columns})
    g = lc.groupby(chave, sort=False)
    lc2 = g.agg(agg)
    lc2["n_linhas_origem"] = g.size()
    lc2 = lc2.reset_index()
    lc2[COLS_PERFIL] = lc2[COLS_PERFIL].replace("__na__", np.nan)
    lc2["id_registro"] = "H" + lc2["hash"].str[:12] + "_" + lc2.groupby("hash").cumcount().astype(str)
    lc2["n_tipos_violacao"] = lc2[[v for v in viol if v != "viol_desconhecida"]].sum(axis=1)
    lc2["hash"] = lc2["hash"].astype(str)
    lc2.loc[lc2["n_tipos_violacao"] > 0, "viol_desconhecida"] = 0
    print(f"layout C: {c.sum()} linhas (violações) -> {len(lc2)} registros")
    return pd.concat([resto, lc2], ignore_index=True)


COLS_SAIDA = (["id_registro", "hash", "layout", "data_cadastro", "ano", "mes", "pos_stf",
               "pandemia", "cod_mun_ibge6", "municipio_nome", "uf", "fonte_municipio",
               "trans_identificavel", "identidade_trans", "is_trans", "orientacao_sexual",
               "sexo_genero_vitima", "faixa_etaria", "raca_cor", "cenario", "relacao_suspeito",
               "denunciante", "canal"]
              + list(REGRAS_VIOLACAO) + ["viol_outra", "viol_desconhecida", "n_tipos_violacao",
                                         "n_linhas_origem"])

CATEGORICAS = ["identidade_trans", "orientacao_sexual", "sexo_genero_vitima", "faixa_etaria",
               "raca_cor", "cenario", "relacao_suspeito", "denunciante", "canal", "uf"]


def finalizar(df):
    df["ano"] = df["data_cadastro"].dt.year
    df["mes"] = df["data_cadastro"].dt.month
    df["pos_stf"] = (df["data_cadastro"] >= "2019-06-13").astype(int)
    df["pandemia"] = df["data_cadastro"].between("2020-03-01", "2021-12-31").astype(int)
    df["uf"] = df["uf"].fillna(DESC)
    # denunciante/canal não existem antes de 2020: marcar como "nao_coletado" em vez de desconhecido
    antes = df["layout"] == "A_2011_2019"
    df.loc[antes, ["denunciante", "canal"]] = "nao_coletado"
    return df[COLS_SAIDA].sort_values("data_cadastro").reset_index(drop=True)


def matriz_ml(base):
    num = ["ano", "mes", "pos_stf", "pandemia", "trans_identificavel", "n_tipos_violacao"]
    viol = [c for c in base.columns if c.startswith("viol_")]
    dummies = pd.get_dummies(base[CATEGORICAS], prefix=CATEGORICAS, dtype=int)
    m = pd.concat([base[["id_registro", "cod_mun_ibge6"]], base[num], base[viol],
                   base[["is_trans"]], dummies], axis=1)
    return m


def agregados(base):
    base = base.assign(trans=base["is_trans"].fillna(0))
    mun = (base.dropna(subset=["cod_mun_ibge6"])
               .groupby(["cod_mun_ibge6", "uf", "ano"])
               .agg(registros_lgbt=("id_registro", "size"), registros_trans=("trans", "sum"),
                    trans_identificavel=("trans_identificavel", "max"))
               .reset_index())
    serie = (base.groupby(base["data_cadastro"].dt.to_period("M"))
                 .agg(registros_lgbt=("id_registro", "size"), registros_trans=("trans", "sum"),
                      trans_identificavel=("trans_identificavel", "min"))
                 .reset_index().rename(columns={"data_cadastro": "mes"}))
    serie["pos_stf"] = (serie["mes"] >= pd.Period("2019-06", "M")).astype(int)
    serie["pandemia"] = serie["mes"].between(pd.Period("2020-03", "M"), pd.Period("2021-12", "M")).astype(int)
    # meses sem identificação de trans: contagem trans vira ausente (não é zero!)
    serie.loc[serie["trans_identificavel"] == 0, "registros_trans"] = np.nan
    mun.loc[mun["trans_identificavel"] == 0, "registros_trans"] = np.nan
    return mun, serie


DICIONARIO = {
    "id_registro": "Identificador do registro (denúncia x vítima) criado na limpeza",
    "hash": "Identificador da denúncia no Disque 100 (só 2020 jul em diante)",
    "layout": "Formato de origem: A_2011_2019, B_2020_s1, C_2020s2_2022",
    "data_cadastro": "Data/hora de cadastro da denúncia",
    "pos_stf": "1 se cadastrada a partir de 13/06/2019 (decisão do STF, ADO 26)",
    "pandemia": "1 se cadastrada entre mar/2020 e dez/2021",
    "cod_mun_ibge6": "Código IBGE do município da vítima, 6 dígitos (sem verificador)",
    "fonte_municipio": "De onde veio o município: da vítima ou do local da denúncia (2020 s1)",
    "trans_identificavel": "1 se o formulário do período permite identificar vítima trans (0 = jul/2020 a jun/2022)",
    "identidade_trans": "travesti / transexual / transgenero / intersexo / nao_trans / desconhecido",
    "is_trans": "1 = vítima trans ou travesti; 0 = não; vazio = não identificável ou não informado",
    "orientacao_sexual": "homossexual / bissexual / heterossexual / outra / desconhecido",
    "sexo_genero_vitima": "masculino / feminino / desconhecido (sexo ou gênero binário informado)",
    "faixa_etaria": "0-17, 18-24, 25-29, 30-39, 40-49, 50-59, 60+ (pelo limite inferior da faixa original)",
    "raca_cor": "branca / preta / parda / amarela / indigena / desconhecido",
    "cenario": "Local da violação agrupado em macro-categorias",
    "relacao_suspeito": "Relação do suspeito com a vítima, agrupada",
    "denunciante": "propria_vitima / terceiro / anonimo / outro (nao_coletado antes de 2020)",
    "canal": "telefone / mensageiro / internet_app / outro (nao_coletado antes de 2020)",
    "viol_*": "1 se a denúncia contém ao menos uma violação daquela macro-categoria (multi-hot)",
    "n_tipos_violacao": "Quantas macro-categorias de violação a denúncia contém",
    "n_linhas_origem": "Quantas linhas da base original foram colapsadas neste registro",
}


def main(entrada, saida):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    df = carregar(entrada)
    df = identidade_e_orientacao(df)
    df = municipio(df)
    df = violacoes_multi_hot(df)
    df = demais_categorias(df)
    df = colapsar_layout_c(df)
    base = finalizar(df)
    m = matriz_ml(base)
    mun, serie = agregados(base)

    kw = dict(index=False, sep=";", encoding="utf-8-sig")
    base.to_csv(saida / "base_limpa.csv", **kw)
    m.to_csv(saida / "matriz_ml.csv", **kw)
    mun.to_csv(saida / "contagem_municipio_ano.csv", **kw)
    serie.astype({"mes": str}).to_csv(saida / "serie_mensal.csv", **kw)
    pd.DataFrame(DICIONARIO.items(), columns=["coluna", "descricao"]).to_csv(
        saida / "dicionario_variaveis.csv", **kw)
    print(f"base_limpa: {base.shape} | matriz_ml: {m.shape} | municípios-ano: {len(mun)} | meses: {len(serie)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lgbt.csv",
         sys.argv[2] if len(sys.argv) > 2 else "saida")
