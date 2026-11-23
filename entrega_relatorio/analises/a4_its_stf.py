"""
Análise 4 - Série temporal interrompida: decisão do STF (13/06/2019, ADO 26 / MI 4733)
ACH0042 - Grupo 11

Séries mensais (Disque 100):
  (a) total LGBT, jan/2011 a dez/2022, com controles de pandemia (mar/2020-dez/2021), de troca de sistema
      (a partir de jul/2020) e da nova ficha de registro (a partir de jul/2022)
  (b) vítimas trans, jan/2011 a jun/2020 (último mês em que a identidade trans é registrada antes da lacuna)
  (c) proporção trans entre as notificações LGBT, mesmo período de (b): compara trans com LGBT não trans
Modelo: log E[Y_t] = b0 + b1 t + b2 STF_t + b3 (t - t0) STF_t + controles + sazonalidade (Fourier, 12 meses)
  (a), (b): Binomial Negativa; (c): binomial. Erros-padrão de Newey-West (HAC, 6 defasagens).
Robustez: excluindo 2012 (pico atípico de volume nacional).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from comum import DADOS, SAIDA

s = pd.read_csv(DADOS / "serie_mensal.csv", sep=";", encoding="utf-8-sig")
s["data"] = pd.PeriodIndex(s["mes"], freq="M")
s = s.sort_values("data").reset_index(drop=True)
T0 = pd.Period("2019-06", "M")
s["t"] = np.arange(len(s))
t0 = s.loc[s["data"] == T0, "t"].iat[0]
s["stf"] = (s["data"] >= T0).astype(int)
s["t_pos"] = (s["t"] - t0) * s["stf"]
s["pandemia"] = ((s["data"] >= pd.Period("2020-03", "M")) & (s["data"] <= pd.Period("2021-12", "M"))).astype(int)
s["sistema_novo"] = (s["data"] >= pd.Period("2020-07", "M")).astype(int)
s["formulario_2022"] = (s["data"] >= pd.Period("2022-07", "M")).astype(int)  # nova ficha (inclui "pessoa trans")
mes = s["data"].dt.month
for k in (1, 2):
    s[f"sen{k}"], s[f"cos{k}"] = np.sin(2 * np.pi * k * mes / 12), np.cos(2 * np.pi * k * mes / 12)
FOURIER = ["sen1", "cos1", "sen2", "cos2"]


def nb_its(df, y, cols):
    X = sm.add_constant(df[cols])
    m = sm.NegativeBinomial(df[y], X).fit(disp=0, maxiter=500, cov_type="HAC", cov_kwds={"maxlags": 6})
    return m


def tabela(m, nome):
    t = pd.DataFrame({"coef": m.params, "ep": m.bse, "p": m.pvalues})
    t["razao"] = np.exp(t["coef"])
    t["ic95_inf"], t["ic95_sup"] = np.exp(t["coef"] - 1.96 * t["ep"]), np.exp(t["coef"] + 1.96 * t["ep"])
    t["modelo"] = nome
    return t.loc[[i for i in t.index if i in ("t", "stf", "t_pos", "pandemia", "sistema_novo", "formulario_2022")]]


res = []
BASE = ["t", "stf", "t_pos"] + FOURIER
# (a) LGBT total
ma = nb_its(s, "registros_lgbt", BASE + ["pandemia", "sistema_novo", "formulario_2022"])
res.append(tabela(ma, "a) LGBT total 2011-2022"))
sa = s[s["data"].dt.year != 2012]
res.append(tabela(nb_its(sa, "registros_lgbt", BASE + ["pandemia", "sistema_novo", "formulario_2022"]), "a') LGBT total, sem 2012"))
# janela comparável à série trans
w = s[s["data"] <= pd.Period("2020-06", "M")].copy()
res.append(tabela(nb_its(w, "registros_lgbt", BASE + ["pandemia"]), "a'') LGBT total 2011-jun/2020"))
# (b) trans
mb = nb_its(w, "registros_trans", BASE + ["pandemia"])
res.append(tabela(mb, "b) Vítimas trans 2011-jun/2020"))
res.append(tabela(nb_its(w[w["data"].dt.year != 2012], "registros_trans", BASE + ["pandemia"]),
                  "b') Vítimas trans, sem 2012"))
# (c) proporção trans (binomial)
Xc = sm.add_constant(w[BASE + ["pandemia"]])
mc = sm.GLM(w["registros_trans"] / w["registros_lgbt"], Xc, family=sm.families.Binomial(),
            var_weights=w["registros_lgbt"]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
res.append(tabela(mc, "c) Proporção trans 2011-jun/2020 (razão de chances)"))

out = pd.concat(res)
out.to_csv(SAIDA / "a4_its_coeficientes.csv", sep=";", encoding="utf-8-sig")
print(out.round(3).to_string())

# contrafactual (série b): previsão sem a decisão, para o gráfico
for nome, m, df, y in (("lgbt", ma, s, "registros_lgbt"), ("trans", mb, w, "registros_trans")):
    X = sm.add_constant(df[[c for c in m.params.index if c not in ("const", "alpha")]], has_constant="add")
    X0 = X.copy()
    X0["stf"], X0["t_pos"] = 0, 0
    df = df.assign(ajustado=m.predict(X), contrafactual=m.predict(X0))
    df[["mes", y, "ajustado", "contrafactual", "stf", "pandemia"]].to_csv(
        SAIDA / f"a4_serie_{nome}.csv", index=False, sep=";", encoding="utf-8-sig")
    pos = df[df["stf"] == 1]
    if nome == "trans":
        print(f"{nome}: observado pós-STF={pos[y].sum():.0f}, contrafactual={pos['contrafactual'].sum():.1f}")
pre, pos = w[(w["data"] >= pd.Period("2018-06", "M")) & (w["data"] < T0)], w[w["stf"] == 1]
print("média mensal trans 12 meses antes:", round(pre["registros_trans"].mean(), 1),
      "| depois (13 meses):", round(pos["registros_trans"].mean(), 1))

# ---------------------------------------------------------------------------
# Teste de placebo: em que mês a quebra se ajusta melhor? (série trans, jan/2011-jun/2020)
# Se o melhor ponto for anterior a jun/2019, a queda não pode ser atribuída à decisão.
# ---------------------------------------------------------------------------
placebo = []
for quebra in pd.period_range("2018-01", "2019-12", freq="M"):
    v = w.copy()
    tq = v.loc[v["data"] == quebra, "t"].iat[0]
    v["stf"] = (v["data"] >= quebra).astype(int)
    v["t_pos"] = (v["t"] - tq) * v["stf"]
    m = sm.NegativeBinomial(v["registros_trans"], sm.add_constant(v[BASE + ["pandemia"]])).fit(disp=0, maxiter=500)
    placebo.append((str(quebra), m.llf, np.exp(m.params["stf"])))
placebo = pd.DataFrame(placebo, columns=["mes_quebra", "log_verossimilhanca", "razao_nivel"])
placebo["delta_vs_jun2019"] = placebo["log_verossimilhanca"] - placebo.loc[placebo.mes_quebra == "2019-06",
                                                                            "log_verossimilhanca"].iat[0]
placebo.to_csv(SAIDA / "a4_placebo_quebra.csv", index=False, sep=";", encoding="utf-8-sig")
print(placebo.round(3).to_string(index=False))
# mudança no registro: travesti x transexual no campo de gênero, por trimestre de 2018-2019
b = pd.read_csv(DADOS / "base_limpa.csv", sep=";", usecols=["data_cadastro", "identidade_trans"], low_memory=False)
b["tri"] = pd.PeriodIndex(pd.to_datetime(b["data_cadastro"]), freq="Q")
reg = b[(b["tri"] >= pd.Period("2018Q1", "Q")) & (b["tri"] <= pd.Period("2020Q2", "Q"))]
tab = pd.crosstab(reg["tri"], reg["identidade_trans"])
tab.to_csv(SAIDA / "a4_registro_identidade_trimestre.csv", sep=";", encoding="utf-8-sig")
print(tab.to_string())
