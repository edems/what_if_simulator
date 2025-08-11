"""
streamlit_app.py — CSV-driven UI with data quality checks, tariff type in PDF
"""
import streamlit as st
import pandas as pd
import numpy as np
import importlib.util
import io
import os
import altair as alt
from datetime import datetime
import math

CURRENT_DIR = os.path.dirname(__file__)
CATALOG_PATH = os.path.join(CURRENT_DIR, "services_catalog.csv")
PROTO_PATH = os.path.join(CURRENT_DIR, "prototype_newindex.py")

# Dynamically import the prototype module
spec = importlib.util.spec_from_file_location("prototype_newindex", PROTO_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore

st.set_page_config(page_title="Was-wäre-wenn Simulator", layout="wide")
st.title("💡 Was-wäre-wenn Simulator (Prototyp)")
st.markdown("CSV-basiert: Leistungen werden aus `services_catalog.csv` geladen. Ergebnisse sind verhandlungstauglich (Benefits, PDF, Export).")

# ---- Load & validate catalog
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["Leistung_Code","Leistung_Name","Tarifart","Basispreis_CHF","Beispielmenge_pro_Jahr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Fehlende Spalten in services_catalog.csv: {missing}")
        st.stop()
    # Data quality checks
    issues = []
    if df["Leistung_Code"].duplicated().any():
        dups = df[df["Leistung_Code"].duplicated()]["Leistung_Code"].tolist()
        issues.append(f"Doppelte Leistungscodes: {dups}")
    if (df["Basispreis_CHF"]<=0).any() or df["Basispreis_CHF"].isna().any():
        issues.append("Ungültige Basispreise (<=0 oder NaN) gefunden.")
    if (df["Beispielmenge_pro_Jahr"]<=0).any() or df["Beispielmenge_pro_Jahr"].isna().any():
        issues.append("Ungültige Beispielmengen (<=0 oder NaN) gefunden.")
    if issues:
        st.warning("⚠️ Datenqualitäts-Hinweise:\n- " + "\n- ".join(issues))
    return df

services_df = load_catalog(CATALOG_PATH)

# ---- Core data & model
with st.spinner("📊 Generiere synthetische Daten..."):
    records_df, _, aerzte_df = module.generate_synthetic_data(n_aerzte=300, seed=42, catalog_path=CATALOG_PATH)
    X, y, top_services = module.build_aggregate_features(records_df, top_n_services=20)
    model, rmse, model_feature_order = module.train_model(X, y)

# ---- Sidebar controls
st.sidebar.header("⚙️ Szenario Einstellungen")
# Build label map Code -> "Code — Name (Tarifart)"
label_map = {row["Leistung_Code"]: f"{row['Leistung_Code']} — {row['Leistung_Name']} ({row['Tarifart']})" for _, row in services_df.iterrows()}
# Only list top services for convenience
top_labels = [label_map.get(code, code) for code in top_services]
selected_label = st.sidebar.selectbox("Leistung", top_labels)
selected_code = selected_label.split(" — ")[0]

preis_mult = st.sidebar.slider("Preis-Multiplikator", 0.5, 1.5, 1.0, step=0.01)
menge_mult = st.sidebar.slider("Mengen-Multiplikator", 0.0, 2.0, 1.0, step=0.01)
apply_btn = st.sidebar.button("Szenario anwenden")

st.sidebar.markdown(f"**Train RMSE:** {rmse:.2f} CHF")
st.sidebar.markdown(f"**Train Size:** {len(X)} Ärzte")

# ---- PDF helper (with Tarifart)
def generate_szenario_pdf(summary_det, selected_code, preis_mult, menge_mult, df_adjusted, services_df, scenario=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Was-wäre-wenn Szenario – Tarifverhandlungen", styles['Title']))
    elements.append(Paragraph(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Scenario header
    selected_row = services_df.set_index("Leistung_Code").loc[selected_code]
    elements.append(Paragraph(f"<b>Szenario:</b> {selected_code} — {selected_row['Leistung_Name']} ({selected_row['Tarifart']})", styles['Normal']))
    elements.append(Paragraph(f"Preis-Multiplikator: {preis_mult} | Mengen-Multiplikator: {menge_mult}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # KPIs table
    kpi_data = [
        ["Alter Gesamtbetrag (CHF)", f"{summary_det['aggregierter_alt_betrag']:.2f}"],
        ["Neuer Gesamtbetrag (CHF)", f"{summary_det['aggregierter_neu_betrag']:.2f}"],
        ["Absolute Änderung (CHF)", f"{summary_det['absolut_diff']:.2f}"],
        ["Relative Änderung (%)", f"{summary_det['rel_diff_pct']:.2f}%"],
        ["Ersparnis pro Arzt (CHF)", f"{summary_det.get('ersparnis_pro_arzt', 0):.2f}"],
        ["Hochrechnung auf 5 Jahre (CHF)", f"{summary_det.get('hochrechnung_5jahre', 0):.2f}"],
    ]
    table = Table(kpi_data, hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
                               ('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Top-3 Leistungen list
    elements.append(Paragraph("<b>Top-3 Leistungen mit größtem Sparpotenzial:</b>", styles['Heading3']))
    for code in summary_det.get('top_leistungen_impact', []):
        if code in services_df["Leistung_Code"].values:
            row = services_df.set_index("Leistung_Code").loc[code]
            elements.append(Paragraph(f"- {code} — {row['Leistung_Name']} ({row['Tarifart']})", styles['Normal']))
        else:
            elements.append(Paragraph(f"- {code}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Build per-service diff table with Tarifart
    svc_map = services_df.set_index("Leistung_Code")[["Leistung_Name","Tarifart"]]
    svc_diff = df_adjusted.groupby("leistung_code")[["total","new_total"]].sum()
    svc_diff["diff"] = svc_diff["new_total"] - svc_diff["total"]
    svc_diff = svc_diff.sort_values("diff").head(10).reset_index()
    data = [["Leistung", "Tarifart", "Alt (CHF)", "Neu (CHF)", "Diff (CHF)"]]
    for _, r in svc_diff.iterrows():
        code = r["leistung_code"]
        name = svc_map.loc[code]["Leistung_Name"] if code in svc_map.index else code
        cat = svc_map.loc[code]["Tarifart"] if code in svc_map.index else ""
        data.append([f"{code} — {name}", f"{cat}", f"{r['total']:.2f}", f"{r['new_total']:.2f}", f"{r['diff']:.2f}"])
    from reportlab.platypus import Table as RLTable
    from reportlab.platypus import TableStyle as RLTableStyle
    from reportlab.lib import colors as RLcolors
    table2 = RLTable(data, hAlign='LEFT')
    table2.setStyle(RLTableStyle([('GRID',(0,0),(-1,-1),0.5,RLcolors.grey)]))
    elements.append(table2)

    # Conclusion
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Fazit für Entscheider:</b>", styles['Heading3']))
    if summary_det['absolut_diff'] < 0:
        elements.append(Paragraph("Das Szenario zeigt ein Einsparpotenzial und könnte zur Kostensenkung beitragen.", styles['Normal']))
    else:
        elements.append(Paragraph("Das Szenario führt zu Mehrkosten und sollte kritisch bewertet werden.", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---- Apply scenario
if apply_btn:
    scenario = [{"leistung_code": selected_code, "preis_multiplikator": float(preis_mult), "menge_multiplikator": float(menge_mult)}]
    comp, summary_det, df_adjusted = module.deterministic_scenario_eval(records_df, scenario)
    preds_new, _ = module.ml_scenario_eval(model, records_df, scenario, top_services, model_feature_order)
    prognose_ml_total = float(preds_new.sum())

    # KPIs
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💰 Alter Gesamtbetrag (Det.)", f"{summary_det['aggregierter_alt_betrag']:.2f} CHF")
        st.metric("💰 Neuer Gesamtbetrag (Det.)", f"{summary_det['aggregierter_neu_betrag']:.2f} CHF",
                  delta=f"{summary_det['absolut_diff']:.2f} CHF ({summary_det['rel_diff_pct']:.2f}%)",
                  delta_color="normal" if summary_det['absolut_diff'] > 0 else "inverse")
    with col2:
        st.metric("🤖 Neuer Gesamtbetrag (ML)", f"{prognose_ml_total:.2f} CHF",
                  delta=f"{prognose_ml_total - summary_det['aggregierter_alt_betrag']:.2f} CHF",
                  delta_color="normal" if (prognose_ml_total - summary_det['aggregierter_alt_betrag']) > 0 else "inverse")

    # Benefit box
    st.info(f"""
**Benefits für Tarifverhandlungen**
- **Ersparnis/Mehrkosten pro Arzt:** {summary_det.get('ersparnis_pro_arzt',0):.2f} CHF  
- **Hochrechnung 5 Jahre:** {summary_det.get('hochrechnung_5jahre',0):,.2f} CHF  
- **Top-3 Sparpotenzial-Leistungen:** {', '.join([label_map.get(c, c) for c in summary_det.get('top_leistungen_impact', [])])}
""")

    # Per-doctor changes
    df_comp = comp.reset_index().rename(columns={"old_total":"alt","new_total":"neu","diff_abs":"diff"})
    df_comp["diff_rel_pct"] = (df_comp["diff"] / df_comp["alt"].replace(0, np.nan) * 100).fillna(0)

    # Charts
    st.subheader("📈 Änderungen nach Arzt (Top 10)")
    top_abs = df_comp.sort_values("diff", ascending=False).head(10)
    chart = alt.Chart(top_abs).mark_bar().encode(
        x=alt.X('arzt_id:O', title='Arzt-ID'),
        y=alt.Y('diff:Q', title='Absolute Änderung (CHF)'),
        tooltip=['arzt_id', alt.Tooltip('diff:Q', format='.2f'), alt.Tooltip('diff_rel_pct:Q', format='.2f')]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.subheader("🏥 Top-Leistungen nach absoluter Änderung (Top 10)")
    svc_map = services_df.set_index("Leistung_Code")[["Leistung_Name","Tarifart"]]
    svc_diff = df_adjusted.groupby("leistung_code")[["total","new_total"]].sum()
    svc_diff["diff"] = svc_diff["new_total"] - svc_diff["total"]
    svc_diff = svc_diff.sort_values("diff").head(10).reset_index()
    svc_diff["label"] = svc_diff["leistung_code"].apply(
        lambda c: f"{c} — {svc_map.loc[c]['Leistung_Name']} ({svc_map.loc[c]['Tarifart']})" if c in svc_map.index else c)
    chart2 = alt.Chart(svc_diff).mark_bar().encode(
        x=alt.X('label:N', sort=None, title='Leistung'),
        y=alt.Y('diff:Q', title='Diff (CHF)'),
        tooltip=[alt.Tooltip('label:N'), alt.Tooltip('diff:Q', format='.2f'),
                 alt.Tooltip('total:Q', title='Alt', format='.2f'), alt.Tooltip('new_total:Q', title='Neu', format='.2f')]
    ).properties(height=300).interactive()
    st.altair_chart(chart2, use_container_width=True)

    # Table
    st.subheader("📋 Detailansicht (Top 15 absolute Änderungen je Arzt)")
    df_top = df_comp.sort_values("diff", ascending=False).head(15)
    st.dataframe(df_top.style.format({"alt":"{:.2f}","neu":"{:.2f}","diff":"{:.2f}","diff_rel_pct":"{:.2f}"}))

    # PDF & CSV download
    pdf_buffer = generate_szenario_pdf(summary_det, selected_code, preis_mult, menge_mult, df_adjusted, services_df, scenario=None)
    st.download_button('📄 PDF herunterladen', pdf_buffer, file_name='szenario_report.pdf', mime='application/pdf')

    csv_buf = io.StringIO()
    df_comp.to_csv(csv_buf, index=False)
    st.download_button("📥 CSV (Änderungen je Arzt)", csv_buf.getvalue(), file_name="szenario_ergebnisse_aerzte.csv", mime="text/csv")

else:
    st.info("⬅️ Wähle eine Leistung und Multiplikatoren und klicke **Szenario anwenden**.")
