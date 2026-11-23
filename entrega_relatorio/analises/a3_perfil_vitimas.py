"""
Análise 3 - Perfil das notificações com vítima trans versus demais vítimas LGBT
ACH0042 - Grupo 11

Unidade: registro (denúncia x vítima) com identidade conhecida (is_trans = 0/1).
Modelos:
  - Regressão logística com efeitos fixos de ano e erros agrupados por município -> razões de chance
  - LightGBM classificador, validação cruzada agrupada por município (AUC), importância SHAP
Excluídas por vazamento: identidade_trans, orientação sexual e sexo/gênero (em 2022 a identidade trans
foi registrada no campo de orientação). Denunciante e canal só existem a partir de 2020 e ficam de fora.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from comum import DADOS, SAIDA

b = pd.read_csv(DADOS / "base_limpa.csv", sep=";", encoding="utf-8-sig", dtype={"cod_mun_ibge6": str, "hash": str},
                low_memory=False)
b = b[b["is_trans"].notna()].copy()
b["grupo_mun"] = b["cod_mun_ibge6"].fillna(pd.Series("sem_municipio_" + b.index.astype(str), index=b.index))
VIOL = ["viol_vida", "viol_sexual", "viol_institucional", "viol_discriminacao", "viol_patrimonial",
        "viol_negligencia", "viol_liberdade", "viol_fisica", "viol_psicologica"]
REF = {"cenario": "residencia", "relacao_suspeito": "familia", "faixa_etaria": "18-24", "raca_cor": "branca"}
dums = []
for col, ref in REF.items():
    dd = pd.get_dummies(b[col], prefix=col, dtype=float)
    dums.append(dd.drop(columns=f"{col}_{ref}"))
regiao_uf = {"N": ["RO", "AC", "AM", "RR", "PA", "AP", "TO"], "NE": ["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"],
             "SE": ["MG", "ES", "RJ", "SP"], "S": ["PR", "SC", "RS"], "CO": ["MS", "MT", "GO", "DF"]}
mapa = {uf: r for r, ufs in regiao_uf.items() for uf in ufs}
reg = pd.get_dummies(b["uf"].map(mapa).fillna("desconhecida"), prefix="regiao", dtype=float).drop(columns="regiao_SE")
anos = pd.get_dummies(b["ano"], prefix="ano", dtype=float).drop(columns="ano_2011")
X = pd.concat([b[VIOL].astype(float)] + dums + [reg, anos], axis=1)
X = X.loc[:, X.sum() >= 20]  # categorias com menos de 20 registros não são estimáveis
y = b["is_trans"].astype(int)
print(f"registros: {len(b)} | vítimas trans: {y.sum()} ({y.mean() * 100:.1f}%) | preditores: {X.shape[1]}")

# --- Logística ---
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=200, cov_type="cluster",
                                         cov_kwds={"groups": pd.factorize(b["grupo_mun"])[0]})
t = pd.DataFrame({"coef": m.params, "ep": m.bse, "p": m.pvalues})
t["razao_chances"] = np.exp(t["coef"])
t["ic95_inf"], t["ic95_sup"] = np.exp(t["coef"] - 1.96 * t["ep"]), np.exp(t["coef"] + 1.96 * t["ep"])
t["n_registros"] = [len(b)] + [int(X[c].sum()) for c in X.columns]
t.to_csv(SAIDA / "a3_logistica_razoes_chance.csv", sep=";", encoding="utf-8-sig")
print(t.drop(index=[i for i in t.index if i.startswith("ano_")]).round(3).to_string())
print("pseudo-R2 McFadden:", round(m.prsquared, 3))

# --- LightGBM com validação agrupada por município ---
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
oof_gbm, oof_log = np.zeros(len(b)), np.zeros(len(b))
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(X, y, b["grupo_mun"]):
    d = lgb.Dataset(X.iloc[tr], y.iloc[tr])
    mod = lgb.train(params, d, 400)
    oof_gbm[te] = mod.predict(X.iloc[te])
    ml = sm.Logit(y.iloc[tr], sm.add_constant(X.iloc[tr])).fit(disp=0, maxiter=200)
    oof_log[te] = ml.predict(sm.add_constant(X.iloc[te], has_constant="add"))
auc = {"logistica": roc_auc_score(y, oof_log), "lightgbm": roc_auc_score(y, oof_gbm),
       "so_efeito_ano": roc_auc_score(y, sm.Logit(y, sm.add_constant(anos)).fit(disp=0).predict())}
print("AUC (validação agrupada por município):", {k: round(v, 3) for k, v in auc.items()})
final = lgb.train(params, lgb.Dataset(X, y), 400)
sv = shap.TreeExplainer(final).shap_values(X)
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
imp.rename("shap_medio_abs").to_csv(SAIDA / "a3_importancia_shap.csv", sep=";", encoding="utf-8-sig")
print(imp.head(15).round(3).to_string())
pd.Series(auc).to_csv(SAIDA / "a3_auc.csv", sep=";", encoding="utf-8-sig")

# descritivo: % de cada característica entre trans e demais
desc = pd.concat([b[VIOL].astype(float)] + [pd.get_dummies(b[c], prefix=c, dtype=float) for c in REF], axis=1)
comp = desc.groupby(y).mean().T * 100
comp.columns = ["demais_LGBT_%", "trans_%"]
comp.to_csv(SAIDA / "a3_perfil_descritivo.csv", sep=";", encoding="utf-8-sig")
print(comp.round(1).to_string())
