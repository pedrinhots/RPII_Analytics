"""
Análise 2 - Proporção de vítimas trans entre as notificações LGBT, por município e ano
ACH0042 - Grupo 11

Pergunta: que características do município se associam a uma fatia maior de vítimas trans?
Modelo: regressão binomial (registros trans de n registros LGBT da janela identificável), com
        dispersão estimada (quase-binomial) e erros-padrão agrupados por município; efeitos fixos de ano
        (a forma de registrar identidade de gênero muda entre os formatos do Disque 100).
Robustez: só capitais; sem efeito fixo de ano.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from comum import SAIDA, ler_painel

p = ler_painel()
risco = pd.read_csv(SAIDA / "a1_risco_municipal.csv", sep=";", dtype={"cod_mun_ibge7": str})
p = p.merge(risco[["cod_mun_ibge7", "alarmante_2020_2022", "hotspot_2011_2022"]], on="cod_mun_ibge7", how="left")
d = p[(p["cobertura_trans"] > 0) & (p["registros_lgbt_janela_trans"] > 0)].copy()

d["log_pop"] = np.log(d["populacao"])
d["log_dens"] = np.log(d["densidade"])
d["log_renda"] = np.log(d["renda_pc_r2022"])
d["log_hom"] = np.log1p(d["taxa_homicidios_100mil"])
d["log_proc_pt"] = np.log1p(d["procedimentos_pt"])
CONT = ["log_pop", "log_dens", "log_renda", "log_hom", "pct_15_29", "pct_pretos_pardos",
        "pct_casais_mesmo_sexo_2022", "munic_indice_lgbt"]
BIN = ["processo_transexualizador", "munic_ouvidoria_dh", "capital"]
for c in CONT:
    d["z_" + c] = (d[c] - d[c].mean()) / d[c].std()
regioes = pd.get_dummies(d["regiao"], prefix="reg", dtype=float).drop(columns="reg_Sudeste")
anos = pd.get_dummies(d["ano"], prefix="ano", dtype=float).drop(columns="ano_2011")


def ajustar(df, com_ano=True, cols=None):
    cols = cols or (["z_" + c for c in CONT] + BIN)
    X = pd.concat([df[cols], regioes.loc[df.index]] + ([anos.loc[df.index]] if com_ano else []), axis=1)
    X = sm.add_constant(X.loc[:, X.std() > 0])
    y = df["registros_trans"] / df["registros_lgbt_janela_trans"]
    m = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=df["registros_lgbt_janela_trans"]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["cod_mun_ibge7"])[0]})
    disp = m.pearson_chi2 / m.df_resid
    t = pd.DataFrame({"coef": m.params, "ep": m.bse, "p": m.pvalues})
    t["razao_chances"] = np.exp(t["coef"])
    t["ic95_inf"], t["ic95_sup"] = np.exp(t["coef"] - 1.96 * t["ep"]), np.exp(t["coef"] + 1.96 * t["ep"])
    return t, disp, len(df), df["cod_mun_ibge7"].nunique()


res = {}
res["principal"] = ajustar(d)
res["sem_efeito_ano"] = ajustar(d, com_ano=False)
res["so_capitais"] = ajustar(d[d["capital"] == 1], cols=["z_" + c for c in CONT] + ["processo_transexualizador",
                                                                                     "munic_ouvidoria_dh"])
for nome, (t, disp, n, nm) in res.items():
    t.to_csv(SAIDA / f"a2_coeficientes_{nome}.csv", sep=";", encoding="utf-8-sig")
    print(f"\n== {nome}: {n} município-anos, {nm} municípios, dispersão de Pearson = {disp:.2f}")
    print(t.drop(index=[i for i in t.index if i.startswith("ano_")]).round(3).to_string())

# descritivo: proporção trans por grupo de município
desc = []
for nome, mask in (("todos", d.index == d.index), ("alarmantes 2020-2022", d["alarmante_2020_2022"] == 1),
                   ("demais", d["alarmante_2020_2022"] == 0), ("aglomerado 2011-2022", d["hotspot_2011_2022"] == 1),
                   ("capitais", d["capital"] == 1), ("capitais com Proc. Transexualizador ativo",
                                                    (d["capital"] == 1) & (d["processo_transexualizador"] == 1)),
                   ("capitais sem", (d["capital"] == 1) & (d["processo_transexualizador"] == 0))):
    s = d[mask]
    desc.append((nome, int(s["registros_trans"].sum()), int(s["registros_lgbt_janela_trans"].sum()),
                 s["registros_trans"].sum() / s["registros_lgbt_janela_trans"].sum() * 100))
desc = pd.DataFrame(desc, columns=["grupo", "trans", "lgbt_janela", "proporcao_trans_%"])
desc.to_csv(SAIDA / "a2_proporcao_por_grupo.csv", index=False, sep=";", encoding="utf-8-sig")
print(desc.round(1).to_string(index=False))
por_ano = d.groupby("ano")[["registros_trans", "registros_lgbt_janela_trans"]].sum()
por_ano["proporcao_%"] = por_ano.iloc[:, 0] / por_ano.iloc[:, 1] * 100
por_ano.to_csv(SAIDA / "a2_proporcao_por_ano.csv", sep=";", encoding="utf-8-sig")
print(por_ano.round(1).to_string())
