"""
Análise 1 - Previsão de notificações e alerta de ondas por município
ACH0042 - Grupo 11

Pergunta: onde estão os números alarmantes e quais cidades tendem a ter ondas de notificações?

Componentes
  1. Suavização Bayesiana empírica (Poisson-Gama) -> mapa de risco relativo e P(RR > 1)
  2. Estatística local Gi* (Getis-Ord) -> aglomerados regionais (corredores) de risco
  3. Previsão um ano à frente, validação por janela móvel (2016-2022):
       referências: só população | participação histórica | ano anterior
       modelo endêmico-epidêmico Binomial Negativo (forma aditiva do hhh4)
       LightGBM (perda Tweedie, offset do cenário nacional)
       Extra Trees (comparação com a literatura)
  4. Alertas de onda (limiar preditivo P95/P99, à la Farrington) e persistência
  5. Ajuste final 2012-2022: coeficientes, decomposição endêmico/autorregressivo/vizinhança,
     tendência relativa das cidades e lista de cidades em risco de onda em 2023 (cenários)
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score

from a1_preparacao import (ANOS, ARVORE, CONTINUAS, ENDEMICAS, LIM_ONDA_MIN, LIM_ONDA_RR, N_MUN, E, EndemicoEpidemico,
                           Y, codigos, coords, idx_viz, nac, p, padronizar)
from comum import SAIDA, bayes_empirico, desvio_poisson, nb_quantil, nb_sf, topk_recall

# ---------------------------------------------------------------------------
# 3. Validação por janela móvel: treina até t-1, prevê t
# ---------------------------------------------------------------------------
linhas, prev_todas, iteracoes, log_treino = [], [], [], []
for T in range(2016, 2023):
    tr, te = p[p["ano"] < T].copy(), p[p["ano"] == T].copy()
    y = te["registros_lgbt"].values
    onda = ((y >= LIM_ONDA_MIN) & (y >= LIM_ONDA_RR * te["e"].values)).astype(int)
    Xtr, Xte = padronizar(tr, te, ENDEMICAS)

    prev = {}
    prev["Só população"] = te["e"].values
    anteriores = [a for a in ANOS if a < T]
    share = Y[anteriores].sum(axis=1) / Y[anteriores].sum().sum()
    prev["Participação histórica"] = te["cod_mun_ibge7"].map(share).values * nac[T] + 1e-3
    prev["Ano anterior"] = np.clip(te["y_lag"].values * te["r_nac"].values, 0.05, None)

    t0 = time.perf_counter()
    ee = EndemicoEpidemico(ENDEMICAS).ajustar(tr, Xtr)
    t_ee = time.perf_counter() - t0
    prev["Endêmico-epidêmico (BN)"] = ee.prever(te, Xte)

    # LightGBM: offset = log(e) no init_score; validação interna = último ano de treino
    trA, trB = tr[tr["ano"] < T - 1], tr[tr["ano"] == T - 1]
    dA = lgb.Dataset(trA[ARVORE], trA["registros_lgbt"], init_score=np.log(trA["e"]))
    dB = lgb.Dataset(trB[ARVORE], trB["registros_lgbt"], init_score=np.log(trB["e"]), reference=dA)
    params = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.03, num_leaves=15,
                  min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=42)
    t0 = time.perf_counter()
    gbm = lgb.train(params, dA, 2000, valid_sets=[dB], callbacks=[lgb.early_stopping(100, verbose=False)])
    gbm_full = lgb.train(params, lgb.Dataset(tr[ARVORE], tr["registros_lgbt"], init_score=np.log(tr["e"])),
                         gbm.best_iteration)
    t_gbm = time.perf_counter() - t0
    prev["LightGBM (Tweedie)"] = np.exp(gbm_full.predict(te[ARVORE], raw_score=True) + np.log(te["e"].values))

    et = ExtraTreesRegressor(n_estimators=300, criterion="poisson", min_samples_leaf=20, max_features=0.5,
                             n_jobs=-1, random_state=42)
    t0 = time.perf_counter()
    et.fit(np.c_[tr[ARVORE].values, np.log(tr["e"].values)], tr["registros_lgbt"].values)
    t_et = time.perf_counter() - t0
    prev["Extra Trees"] = np.clip(et.predict(np.c_[te[ARVORE].values, np.log(te["e"].values)]), 1e-3, None)

    for nome, mu in prev.items():
        ponto = y >= LIM_ONDA_MIN
        limiar = np.maximum(LIM_ONDA_MIN, np.ceil(LIM_ONDA_RR * te["e"].values))
        prob = nb_sf(limiar, mu, ee.size)
        linhas.append(dict(ano=T, modelo=nome, desvio=desvio_poisson(y, mu), mae=np.mean(np.abs(y - mu)),
                           top50=topk_recall(y, mu, 50), n_ondas=onda.sum(),
                           ap_onda=average_precision_score(onda, prob) if onda.sum() else np.nan,
                           prec20_onda=onda[np.argsort(-prob)[:20]].mean()))
    out = te[["cod_mun_ibge7", "municipio", "uf", "ano", "registros_lgbt", "e"]].copy()
    out["onda"] = onda
    for nome, mu in prev.items():
        out["mu_" + nome] = mu
    out["size_ee"] = ee.size
    prev_todas.append(out)
    iteracoes.append(gbm.best_iteration)
    log_treino.append(dict(ano_previsto=T, anos_treino=f"2012-{T - 1}", n_treino=len(tr), n_teste=len(te),
                           ee_iteracoes_otimizador=ee.n_iter, ee_avaliacoes_verossimilhanca=ee.n_aval,
                           ee_convergiu=ee.convergiu, ee_segundos=round(t_ee, 2),
                           lgbm_arvores_escolhidas=gbm.best_iteration, lgbm_segundos=round(t_gbm, 2),
                           et_arvores=300, et_segundos=round(t_et, 2)))
    print(f"{T}: ok (EE convergiu={ee.convergiu}, LGBM it={gbm.best_iteration})")

pd.DataFrame(log_treino).to_csv(SAIDA / "a1_log_treinamento.csv", index=False, sep=";", encoding="utf-8-sig")
val = pd.DataFrame(linhas)
val.to_csv(SAIDA / "a1_validacao_por_ano.csv", index=False, sep=";", encoding="utf-8-sig")
resumo = val.groupby("modelo")[["desvio", "mae", "top50", "ap_onda", "prec20_onda"]].mean()
ref = resumo.loc["Só população", "desvio"]
resumo["ganho_desvio_vs_pop_%"] = (1 - resumo["desvio"] / ref) * 100
resumo = resumo.sort_values("desvio")
resumo.to_csv(SAIDA / "a1_validacao_resumo.csv", sep=";", encoding="utf-8-sig")
print(resumo.round(3).to_string())
prevs = pd.concat(prev_todas)
prevs.to_csv(SAIDA / "a1_previsoes_validacao.csv", index=False, sep=";", encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 4. Alertas de onda (limiar preditivo do modelo endêmico-epidêmico) e persistência
# ---------------------------------------------------------------------------
mu_ee = prevs["mu_Endêmico-epidêmico (BN)"].values
prevs["q95"] = nb_quantil(0.95, mu_ee, prevs["size_ee"].values)
prevs["q99"] = nb_quantil(0.99, mu_ee, prevs["size_ee"].values)
prevs["alerta"] = ((prevs["registros_lgbt"] > prevs["q95"]) & (prevs["registros_lgbt"] >= 3)).astype(int)
prevs["alerta_forte"] = ((prevs["registros_lgbt"] > prevs["q99"]) & (prevs["registros_lgbt"] >= 3)).astype(int)
al = prevs.pivot(index="cod_mun_ibge7", columns="ano", values="alerta")
pares = [(al[t], al[t + 1]) for t in range(2016, 2022)]
p_rep = sum((a & b).sum() for a, b in pares) / sum(a.sum() for a, _ in pares)
p_base = sum(b.sum() for _, b in pares) / (len(pares) * len(al))
alertas = {"alertas_por_ano": prevs.groupby("ano")["alerta"].sum().to_dict(),
           "alertas_fortes_por_ano": prevs.groupby("ano")["alerta_forte"].sum().to_dict(),
           "P(alerta t+1 | alerta t)": round(p_rep, 3), "P(alerta t+1) base": round(p_base, 4),
           "razao_persistencia": round(p_rep / p_base, 1)}
rep = (prevs.groupby(["cod_mun_ibge7", "municipio", "uf"])["alerta"].sum().sort_values(ascending=False)
       .reset_index().rename(columns={"alerta": "anos_em_alerta_2016_2022"}))
rep.head(30).to_csv(SAIDA / "a1_cidades_alerta_recorrente.csv", index=False, sep=";", encoding="utf-8-sig")
print(json.dumps(alertas, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 5. Ajuste final (2012-2022): coeficientes, decomposição, SHAP
# ---------------------------------------------------------------------------
Xall, _ = padronizar(p, p, ENDEMICAS)
ee = EndemicoEpidemico(ENDEMICAS).ajustar(p, Xall, erros=True)
nomes = ["intercepto"] + ENDEMICAS + ["AR: intercepto", "AR: log pop", "Vizinhança: intercepto",
                                       "Vizinhança: log pop", "log tamanho (BN)"]
coef = pd.DataFrame({"parametro": nomes, "estimativa": ee.theta, "erro_padrao": ee.ep})
coef["z"] = coef["estimativa"] / coef["erro_padrao"]
coef["p_valor"] = 2 * (1 - __import__("scipy").stats.norm.cdf(coef["z"].abs()))
coef["razao_taxa"] = np.exp(coef["estimativa"])
coef["ic95_inf"] = np.exp(coef["estimativa"] - 1.96 * coef["erro_padrao"])
coef["ic95_sup"] = np.exp(coef["estimativa"] + 1.96 * coef["erro_padrao"])
coef.to_csv(SAIDA / "a1_coeficientes_endemico_epidemico.csv", index=False, sep=";", encoding="utf-8-sig")
print(coef.round(3).to_string())

end, aut, viz = ee.prever(p, Xall, partes=True)
p["c_end"], p["c_aut"], p["c_viz"] = end, aut, viz
tot = end.sum() + aut.sum() + viz.sum()
decomp = {"endemico_%": end.sum() / tot * 100, "autorregressivo_%": aut.sum() / tot * 100,
          "vizinhanca_%": viz.sum() / tot * 100}
decomp_ano = (p.groupby("ano")[["c_end", "c_aut", "c_viz"]].sum()
              .pipe(lambda d: d.div(d.sum(axis=1), axis=0) * 100))
decomp_ano.to_csv(SAIDA / "a1_decomposicao_por_ano.csv", sep=";", encoding="utf-8-sig")
print("decomposição:", {k: round(v, 1) for k, v in decomp.items()})

params_final = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.03, num_leaves=15,
                    min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                    lambda_l2=1.0, verbose=-1, seed=42)
n_it = int(np.median(iteracoes))  # mediana das iterações escolhidas na validação
gbm_final = lgb.train(params_final, lgb.Dataset(p[ARVORE], p["registros_lgbt"], init_score=np.log(p["e"])), n_it)
import shap  # noqa: E402

amostra = p.sample(5000, random_state=42)
sv = shap.TreeExplainer(gbm_final).shap_values(amostra[ARVORE])
imp = pd.Series(np.abs(sv).mean(axis=0), index=ARVORE).sort_values(ascending=False)
imp.rename("shap_medio_abs").to_csv(SAIDA / "a1_importancia_shap.csv", sep=";", encoding="utf-8-sig")
print(imp.round(3).head(12).to_string())

# ---------------------------------------------------------------------------
# 1-2. Mapa de risco (Bayes empírico) e aglomerados Gi*
# ---------------------------------------------------------------------------
import libpysal  # noqa: E402
from esda.getisord import G_Local  # noqa: E402

W = libpysal.weights.W({i: list(idx_viz[i]) for i in range(N_MUN)})
W.transform = "r"
risco = pd.DataFrame({"cod_mun_ibge7": codigos})
risco = risco.merge(p[p.ano == 2022][["cod_mun_ibge7", "municipio", "uf", "regiao", "populacao"]], how="left")
for nome, anos in (("2011_2022", ANOS), ("2020_2022", [2020, 2021, 2022])):
    yy, ee_ = Y[anos].sum(axis=1).values, E[anos].sum(axis=1).values
    rr, pac, _, _ = bayes_empirico(yy, ee_)
    risco[f"registros_{nome}"], risco[f"esperado_{nome}"] = yy, ee_
    risco[f"rr_bruto_{nome}"], risco[f"rr_eb_{nome}"], risco[f"p_rr_maior1_{nome}"] = yy / ee_, rr, pac
    # Gi* exige variável não negativa: usa o RR suavizado (não o log)
    g = G_Local(rr, W, star=True, permutations=999, seed=42)
    risco[f"gi_z_{nome}"], risco[f"gi_p_{nome}"] = g.Zs, g.p_sim
    # aglomerado alto: vizinhança com RR alto (Gi* > 0, p < 0,05) E o próprio município acima da média
    risco[f"hotspot_{nome}"] = ((g.Zs > 0) & (g.p_sim < 0.05) & (rr > 1)).astype(int)
risco = pd.concat([risco, coords], axis=1)
risco["alarmante_2020_2022"] = (risco["p_rr_maior1_2020_2022"] > 0.95).astype(int)

# tendência relativa (cidades com >= 30 registros): Poisson y ~ offset log(e) + ano, erro quase-Poisson
import statsmodels.api as sm  # noqa: E402

tend = []
for cod in risco.loc[risco["registros_2011_2022"] >= 30, "cod_mun_ibge7"]:
    yy, ee_ = Y.loc[cod].values, E.loc[cod].values
    X = sm.add_constant(np.array(ANOS) - 2016.5)
    m = sm.GLM(yy, X, family=sm.families.Poisson(), offset=np.log(ee_)).fit(scale="X2")
    tend.append((cod, m.params[1], m.bse[1], m.pvalues[1]))
tend = pd.DataFrame(tend, columns=["cod_mun_ibge7", "inclinacao", "ep", "p"])
tend["variacao_anual_relativa_%"] = (np.exp(tend["inclinacao"]) - 1) * 100
tend["tendencia"] = np.select([(tend.p < 0.05) & (tend.inclinacao > 0), (tend.p < 0.05) & (tend.inclinacao < 0)],
                              ["alta", "queda"], "estável")
risco = risco.merge(tend, on="cod_mun_ibge7", how="left")
risco.to_csv(SAIDA / "a1_risco_municipal.csv", index=False, sep=";", encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Lista de vigilância 2023: P(onda) sob cenários do volume nacional
# ---------------------------------------------------------------------------
u = p[p["ano"] == 2022].copy()
u["y_lag"], u["y_lag_viz"] = Y[2022].loc[u["cod_mun_ibge7"]].values, Y.values[idx_viz][:, :, -1].mean(axis=1)[
    pd.Index(codigos).get_indexer(u["cod_mun_ibge7"])]
viz_list = []
for cen_nome, N23 in (("baixo (média 2015-2021)", nac.loc[2015:2021].mean()), ("igual a 2022", nac[2022])):
    v = u.copy()
    v["r_nac"] = N23 / nac[2022]
    v["e"] = N23 * v["populacao"] / v["populacao"].sum()
    Xv = (v[ENDEMICAS] - p[ENDEMICAS].mean()) / p[ENDEMICAS].std().replace(0, 1)
    mu = ee.prever(v, Xv)
    lim = np.maximum(LIM_ONDA_MIN, np.ceil(LIM_ONDA_RR * v["e"].values))
    v["mu_2023"], v["p_onda_2023"], v["cenario"] = mu, nb_sf(lim, mu, ee.size), cen_nome
    viz_list.append(v[["cod_mun_ibge7", "municipio", "uf", "cenario", "registros_lgbt", "mu_2023", "p_onda_2023"]])
vig = pd.concat(viz_list).rename(columns={"registros_lgbt": "registros_2022"})
vig.to_csv(SAIDA / "a1_vigilancia_2023.csv", index=False, sep=";", encoding="utf-8-sig")

json.dump({"decomposicao": decomp, "alertas": alertas, "n_hotspots": {
    k: int(risco[f"hotspot_{k}"].sum()) for k in ("2011_2022", "2020_2022")},
    "n_alarmantes_2020_2022": int(risco["alarmante_2020_2022"].sum()),
    "tendencias": tend["tendencia"].value_counts().to_dict(), "size_bn": float(ee.size)},
    open(SAIDA / "a1_resumo.json", "w"), ensure_ascii=False, indent=1, default=float)
print("fim")
