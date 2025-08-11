"""
prototype_newindex.py — Optimized for Negotiation Benefits
Includes:
- Additional KPIs for tariff negotiation scenarios
- Works with updated streamlit_app.py
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from datetime import date, datetime
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Leistung(BaseModel):
    leistung_code: str = Field(..., description="Codierter Leistungsschlüssel (z.B. L100)")
    name: Optional[str] = None
    basispreis: float = Field(..., description="aktueller Basispreis pro Einheit (CHF)")
    einheit: str = Field("Stück", description="Einheit, z.B. 'Stück', 'Min.', 'kB'")
    tarifart: Optional[str] = None

class Arzt(BaseModel):
    arzt_id: int
    kanton: Optional[str] = None
    fachgebiet: Optional[str] = None
    praxis_typ: Optional[str] = None

class Abrechnung(BaseModel):
    record_id: int
    arzt_id: int
    leistung_code: str
    datum: date
    menge: int
    preis_pro_einheit: float
    total: float
    kanton: Optional[str] = None
    fachgebiet: Optional[str] = None

class SzenarioAenderung(BaseModel):
    leistung_code: str
    preis_multiplikator: Optional[float] = None
    menge_multiplikator: Optional[float] = None

class Szenario(BaseModel):
    szenario_id: int
    name: str
    beschreibung: Optional[str] = None
    aenderungen: List[SzenarioAenderung]
    erstellt_am: datetime = Field(default_factory=datetime.utcnow)

class PrognoseResult(BaseModel):
    szenario_id: int
    aggregierter_alt_betrag: float
    aggregierter_neu_betrag: float
    absolut_diff: float
    rel_diff_pct: float
    anzahl_betroffene_aerzte: int
    modell_version: Optional[str] = None
    kommentar: Optional[str] = None
    ersparnis_pro_arzt: Optional[float] = None
    hochrechnung_5jahre: Optional[float] = None
    top_leistungen_impact: Optional[List[str]] = None

def generate_synthetic_data(n_aerzte: int = 500, n_leistungen: int = 30, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    leistungen = []
    for i in range(n_leistungen):
        code = f"L{i+1:03d}"
        basispreis = float(np.round(20 + np.random.rand()*180, 2))
        leistungen.append({"leistung_code": code, "basispreis": basispreis, "name": f"Leistung {code}"})
    services_df = pd.DataFrame(leistungen)

    kantone = ["ZH","BE","LU","UR","SZ","OW","NW","GL","ZG","FR","SO","BS","BL","SH","AR","AI","SG","GR","AG","TG","TI","VD","VS","NE","GE","JU"]
    fachgebiet_list = ["Allgemein","Innere","Chirurgie","Gyn","Pediatrie","Radiologie"]
    aerzte = []
    for a in range(1, n_aerzte+1):
        aerzte.append({"arzt_id": a,
                       "kanton": np.random.choice(kantone),
                       "fachgebiet": np.random.choice(fachgebiet_list)})
    aerzte_df = pd.DataFrame(aerzte)

    rows = []
    record_id = 1
    for _, ar in aerzte_df.iterrows():
        n_used = np.random.randint(5, min(20, n_leistungen))
        used = services_df.sample(n_used, replace=False)
        for _, svc in used.iterrows():
            lam = np.clip( np.random.rand()*30 + 5, 1, 200 )
            menge = int(max(0, np.random.poisson(lam)))
            preis = float(svc["basispreis"] * (0.9 + 0.2*np.random.rand()))
            total = round(menge * preis, 2)
            rows.append({
                "record_id": record_id,
                "arzt_id": int(ar["arzt_id"]),
                "leistung_code": svc["leistung_code"],
                "datum": date(2024, 1, 1),
                "menge": int(menge),
                "preis_pro_einheit": float(preis),
                "total": float(total),
                "kanton": ar["kanton"],
                "fachgebiet": ar["fachgebiet"]
            })
            record_id += 1

    records_df = pd.DataFrame(rows)
    return records_df, services_df, aerzte_df

def build_aggregate_features(records_df: pd.DataFrame, top_n_services: int = 20):
    pivot_menge = records_df.pivot_table(index="arzt_id", columns="leistung_code", values="menge", aggfunc="sum", fill_value=0)
    pivot_spend = records_df.pivot_table(index="arzt_id", columns="leistung_code", values="total", aggfunc="sum", fill_value=0)

    agg = pd.DataFrame(index=pivot_menge.index)
    total_menge_global = pivot_menge.sum().sort_values(ascending=False)
    top_services = list(total_menge_global.head(top_n_services).index)

    for s in top_services:
        agg[f"menge_{s}"] = pivot_menge.get(s, 0)
        agg[f"spend_{s}"] = pivot_spend.get(s, 0)

    agg["total_spend"] = pivot_spend.sum(axis=1)
    agg["num_services_used"] = (pivot_menge > 0).sum(axis=1)

    meta = records_df.groupby("arzt_id").agg({"kanton":"first","fachgebiet":"first"})
    agg = agg.merge(meta, left_index=True, right_index=True)

    agg = pd.get_dummies(agg, columns=["kanton","fachgebiet"], drop_first=True)
    X = agg.drop(columns=["total_spend"])
    y = agg["total_spend"]
    return X, y, top_services

def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return model, rmse, X_train.columns.tolist()

def deterministic_scenario_eval(records_df: pd.DataFrame, scenario: List[Dict[str, Any]]):
    df = records_df.copy()
    changes = {c["leistung_code"]: c for c in scenario}

    def adj_price(row):
        ch = changes.get(row["leistung_code"])
        if ch and ch.get("preis_multiplikator") is not None:
            return row["preis_pro_einheit"] * float(ch["preis_multiplikator"])
        return row["preis_pro_einheit"]

    def adj_menge(row):
        ch = changes.get(row["leistung_code"])
        if ch and ch.get("menge_multiplikator") is not None:
            return math.floor(row["menge"] * float(ch["menge_multiplikator"]))
        return row["menge"]

    df["new_preis"] = df.apply(adj_price, axis=1)
    df["new_menge"] = df.apply(adj_menge, axis=1)
    df["new_total"] = df["new_preis"] * df["new_menge"]

    old_agg = df.groupby("arzt_id")["total"].sum().rename("old_total")
    new_agg = df.groupby("arzt_id")["new_total"].sum().rename("new_total")
    comp = pd.concat([old_agg, new_agg], axis=1).fillna(0)
    comp["diff_abs"] = comp["new_total"] - comp["old_total"]
    comp["diff_rel_pct"] = (comp["diff_abs"] / comp["old_total"].replace(0, np.nan)).fillna(0)*100

    summary = {
        "aggregierter_alt_betrag": float(comp["old_total"].sum()),
        "aggregierter_neu_betrag": float(comp["new_total"].sum()),
        "absolut_diff": float(comp["diff_abs"].sum()),
        "rel_diff_pct": float((comp["new_total"].sum() - comp["old_total"].sum()) / comp["old_total"].sum() * 100) if comp["old_total"].sum() != 0 else 0.0,
        "anzahl_betroffene_aerzte": int((comp["diff_abs"]!=0).sum())
    }

    # Additional negotiation KPIs
    summary["ersparnis_pro_arzt"] = summary["absolut_diff"] / max(1, summary["anzahl_betroffene_aerzte"])
    summary["hochrechnung_5jahre"] = summary["absolut_diff"] * 5
    leistung_diff = df.groupby("leistung_code")[["total","new_total"]].sum()
    leistung_diff["diff"] = leistung_diff["new_total"] - leistung_diff["total"]
    summary["top_leistungen_impact"] = leistung_diff["diff"].sort_values().head(3).index.tolist()

    return comp, summary

def ml_scenario_eval(model, records_df: pd.DataFrame, scenario: List[Dict[str, Any]], top_services: List[str], model_feature_order: List[str]):
    df = records_df.copy()
    changes = {c["leistung_code"]: c for c in scenario}
    df["preis_eff"] = df.apply(lambda r: r["preis_pro_einheit"] * (changes.get(r["leistung_code"], {}).get("preis_multiplikator", 1.0)), axis=1)
    df["menge_eff"] = df.apply(lambda r: math.floor(r["menge"] * (changes.get(r["leistung_code"], {}).get("menge_multiplikator", 1.0))), axis=1)
    df["total_eff"] = df["preis_eff"] * df["menge_eff"]

    pivot_menge = df.pivot_table(index="arzt_id", columns="leistung_code", values="menge_eff", aggfunc="sum", fill_value=0)
    pivot_spend = df.pivot_table(index="arzt_id", columns="leistung_code", values="total_eff", aggfunc="sum", fill_value=0)

    agg = pd.DataFrame(index=pivot_menge.index)
    for s in top_services:
        agg[f"menge_{s}"] = pivot_menge.get(s, 0)
        agg[f"spend_{s}"] = pivot_spend.get(s, 0)
    agg["total_spend"] = pivot_spend.sum(axis=1)
    agg["num_services_used"] = (pivot_menge > 0).sum(axis=1)
    meta = df.groupby("arzt_id").agg({"kanton":"first","fachgebiet":"first"})
    agg = agg.merge(meta, left_index=True, right_index=True)
    agg = pd.get_dummies(agg, columns=["kanton","fachgebiet"], drop_first=True)

    Xnew = agg.reindex(columns=model_feature_order, fill_value=0)
    preds = model.predict(Xnew)
    return preds, Xnew

def run_demo():
    print("Generating synthetic data...")
    records_df, services_df, aerzte_df = generate_synthetic_data(n_aerzte=600, n_leistungen=40)
    print("Building features...")
    X, y, top_services = build_aggregate_features(records_df, top_n_services=20)
    print("Training model...")
    model, rmse, model_feature_order = train_model(X, y)
    print(f"Model trained. RMSE: {rmse:.2f} CHF")

    scenario = [
        {"leistung_code":"L005", "preis_multiplikator":1.20},
        {"leistung_code":"L012", "menge_multiplikator":0.90}
    ]

    comp, summary_det = deterministic_scenario_eval(records_df, scenario)
    print("Deterministic summary:", summary_det)

    preds_new, _ = ml_scenario_eval(model, records_df, scenario, top_services, model_feature_order)
    prognose_ml_total = float(preds_new.sum())
    print("ML predicted total:", round(prognose_ml_total,2))

    report = PrognoseResult(
        szenario_id=1,
        aggregierter_alt_betrag=summary_det["aggregierter_alt_betrag"],
        aggregierter_neu_betrag=summary_det["aggregierter_neu_betrag"],
        absolut_diff=summary_det["absolut_diff"],
        rel_diff_pct=summary_det["rel_diff_pct"],
        anzahl_betroffene_aerzte=summary_det["anzahl_betroffene_aerzte"],
        modell_version="gbm-v1",
        kommentar="Deterministische und ML-Sicht wurden erzeugt.",
        ersparnis_pro_arzt=summary_det["ersparnis_pro_arzt"],
        hochrechnung_5jahre=summary_det["hochrechnung_5jahre"],
        top_leistungen_impact=summary_det["top_leistungen_impact"]
    )
    print(report.model_dump_json(indent=2))

if __name__ == "__main__":
    run_demo()