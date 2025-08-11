def generate_szenario_pdf(summary_det, selected_service, preis_mult, menge_mult, records_df):  # Hinzugefügt: records_df
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from io import BytesIO
    from datetime import datetime

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Was-wäre-wenn Szenario – Tarifverhandlungen", styles['Title']))
    elements.append(Paragraph(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Szenario Params
    elements.append(Paragraph(f"<b>Szenario:</b> Leistung {selected_service}, Preis-Multiplikator {preis_mult}, Mengen-Multiplikator {menge_mult}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Main KPIs
    elements.append(Paragraph("<b>Top-3 Leistungen mit größtem Sparpotenzial:</b>", styles['Heading3']))
    for t in summary_det.get('top_leistungen_impact', []):
        elements.append(Paragraph(f"- {t}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Table of top impacted services – fixed mit records_df
    elements.append(Paragraph("<b>Top-betroffene Leistungen (absolute Änderung in CHF):</b>", styles['Heading3']))
    df = records_df.copy()  # Verwende records_df für Leistungs-Daten
    # Wiederhole Anpassungen aus deterministic_eval (falls nötig, aber hier angenommen, dass df schon angepasst ist – passe bei Bedarf)
    leistung_diff = df.groupby("leistung_code")[["total","new_total"]].sum()
    leistung_diff["diff"] = leistung_diff["new_total"] - leistung_diff["total"]
    top_services = leistung_diff.sort_values("diff").head(10).reset_index()
    data = [["Leistung", "Alt (CHF)", "Neu (CHF)", "Diff (CHF)"]]
    for _, row in top_services.iterrows():
        data.append([row['leistung_code'], f"{row['total']:.2f}", f"{row['new_total']:.2f}", f"{row['diff']:.2f}"])
    table2 = Table(data, hAlign='LEFT')
    table2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    elements.append(table2)
    elements.append(Spacer(1, 12))

    # Conclusion
    elements.append(Paragraph("<b>Fazit für Entscheider:</b>", styles['Heading3']))
    if summary_det['absolut_diff'] < 0:
        elements.append(Paragraph("Das Szenario zeigt ein Einsparpotenzial und könnte zur Kostensenkung beitragen.", styles['Normal']))
    else:
        elements.append(Paragraph("Das Szenario führt zu Mehrkosten und sollte kritisch bewertet werden.", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer
"""
streamlit_app.py — Enhanced Version

Interactive "Was-wäre-wenn" simulator for NewIndex with charts, CSV export, and improved layout.
Angepasst mit realen TARMED-Leistungen für bessere Verständlichkeit.
Verbessert mit Benefits-Box, Zusammenfassung und Filtern für einfachere Ablesbarkeit.
Farblich angepasst: Helle, positive Farben (heller Hintergrund, Akzente in Grün/Blau).
Angepasst: Sidebar-Text dunkel, Header hell, Dropdown hell – alle weißen Schriftstellen entfernt.
"""
import streamlit as st
import pandas as pd
import numpy as np
import importlib.util
import io
import os
import altair as alt  # Für interaktive Charts (pip install altair)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime
import math

# Dateipfad relativ zum Streamlit-Skript
current_dir = os.path.dirname(__file__)
proto_path = os.path.join(current_dir, "prototype_newindex.py")

spec = importlib.util.spec_from_file_location("prototype_newindex", proto_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore

st.set_page_config(page_title="NewIndex - Was-wäre-wenn Simulator", layout="wide")


st.title("💡 NewIndex — Was-wäre-wenn Simulator (Prototyp)")
st.markdown("Simuliere Änderungen von Preisen/Mengen für Leistungen und analysiere die finanziellen Auswirkungen. Mit realen TARMED-Beispielen für Ärzte.")

# Load synthetic data
with st.spinner("📊 Generiere Beispieldaten..."):
    records_df, services_df, aerzte_df = module.generate_synthetic_data(n_aerzte=300, n_leistungen=30, seed=42)
    X, y, top_services = module.build_aggregate_features(records_df, top_n_services=20)
    model, rmse, model_feature_order = module.train_model(X, y)

# Sidebar controls
st.sidebar.header("⚙️ Szenario Einstellungen")

# Zeige Leistungen mit Code und Name
leistungen_dict = services_df.set_index('leistung_code')['name'].to_dict()
top_services_with_names = [f"{code} - {leistungen_dict.get(code, 'Unbekannt')}" for code in top_services]
selected_service_str = st.sidebar.selectbox("Leistung (Code - Name)", top_services_with_names)
selected_service = selected_service_str.split(" - ")[0]  # Extrahiere Code

preis_mult = st.sidebar.slider("Preis-Multiplikator", 0.5, 1.5, 1.0, step=0.01)
menge_mult = st.sidebar.slider("Mengen-Multiplikator", 0.0, 2.0, 1.0, step=0.01)
apply_btn = st.sidebar.button("Szenario anwenden")

# Model info
st.sidebar.markdown(f"**Train RMSE:** {rmse:.2f} CHF")
st.sidebar.markdown(f"**Train Size:** {len(X)} Ärzte")

def generate_szenario_pdf(summary_det, selected_service, preis_mult, menge_mult, records_df, scenario):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from io import BytesIO
    from datetime import datetime

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Was-wäre-wenn Szenario – Tarifverhandlungen", styles['Title']))
    elements.append(Paragraph(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Szenario Params
    elements.append(Paragraph(f"<b>Szenario:</b> Leistung {selected_service}, Preis-Multiplikator {preis_mult}, Mengen-Multiplikator {menge_mult}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Main KPIs
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

    # Top 3 Leistungen
    top3 = summary_det.get('top_leistungen_impact', [])
    elements.append(Paragraph("<b>Top-3 Leistungen mit größtem Sparpotenzial:</b>", styles['Heading3']))
    for t in top3:
        elements.append(Paragraph(f"- {t}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Table of top impacted services – fixed mit Anpassungen auf records_df
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

    leistung_diff = df.groupby("leistung_code")[["total","new_total"]].sum()
    leistung_diff["diff"] = leistung_diff["new_total"] - leistung_diff["total"]
    top_services = leistung_diff.sort_values("diff").head(10).reset_index()
    data = [["Leistung", "Alt (CHF)", "Neu (CHF)", "Diff (CHF)"]]
    for _, row in top_services.iterrows():
        data.append([row['leistung_code'], f"{row['total']:.2f}", f"{row['new_total']:.2f}", f"{row['diff']:.2f}"])
    table2 = Table(data, hAlign='LEFT')
    table2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    elements.append(table2)
    elements.append(Spacer(1, 12))

    # Conclusion
    elements.append(Paragraph("<b>Fazit für Entscheider:</b>", styles['Heading3']))
    if summary_det['absolut_diff'] < 0:
        elements.append(Paragraph("Das Szenario zeigt ein Einsparpotenzial und könnte zur Kostensenkung beitragen.", styles['Normal']))
    else:
        elements.append(Paragraph("Das Szenario führt zu Mehrkosten und sollte kritisch bewertet werden.", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

if apply_btn:
    scenario = [
        {"leistung_code": selected_service, "preis_multiplikator": float(preis_mult), "menge_multiplikator": float(menge_mult)}
    ]
    comp, summary_det = module.deterministic_scenario_eval(records_df, scenario)
    preds_new, _ = module.ml_scenario_eval(model, records_df, scenario, top_services, model_feature_order)
    prognose_ml_total = float(preds_new.sum())

    # Metrics mit Benefits
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💰 Alter Gesamtbetrag (Deterministisch)", f"{summary_det['aggregierter_alt_betrag']:.2f} CHF")
        st.metric("💰 Neuer Gesamtbetrag (Deterministisch)", f"{summary_det['aggregierter_neu_betrag']:.2f} CHF", 
                  delta=f"{summary_det['absolut_diff']:.2f} CHF ({summary_det['rel_diff_pct']:.2f}%)",
                  delta_color="normal" if summary_det['absolut_diff'] > 0 else "inverse")
    with col2:
        st.metric("🤖 Neuer Gesamtbetrag (ML-Prognose)", f"{prognose_ml_total:.2f} CHF", 
                  delta=f"{prognose_ml_total - summary_det['aggregierter_alt_betrag']:.2f} CHF",
                  delta_color="normal" if (prognose_ml_total - summary_det['aggregierter_alt_betrag']) > 0 else "inverse")

    # Neue Benefits-Box
    st.info("""
    **Benefits für Tarifverhandlungen:**
    - **Evidenzbasierte Argumente**: Zeige, wie Änderungen (z. B. +{summary_det['rel_diff_pct']:.2f}%) Kosten steuern – ideal für FMH-Verhandlungen zu TARDOC (13 Mrd. CHF Umverteilung ab 2026).
    - **Fairness & Risiken minimieren**: Analysiere Verteilung, um Ungleichgewichte (z. B. in Kantonen) zu vermeiden.
    - **Kostenkontrolle**: Simuliere Ersparnisse/Mehreinnahmen – stärkt Position gegenüber Versicherern/Bundesrat.
    - **Schnelle Insights**: Teste Alternativen live, ohne manuelle Rechnungen.
    """)

    # Prepare DataFrame
    df_comp = comp.reset_index().rename(columns={"old_total":"alt","new_total":"neu","diff_abs":"diff"})
    df_comp["diff_rel_pct"] = (df_comp["diff"] / df_comp["alt"].replace(0, np.nan) * 100).fillna(0)

    # Filter für Kanton (neu)
    unique_kantone = df_comp.merge(aerzte_df[['arzt_id', 'kanton']], on='arzt_id')['kanton'].unique()
    selected_kanton = st.selectbox("Filter nach Kanton", ["Alle"] + list(unique_kantone))
    if selected_kanton != "Alle":
        df_comp = df_comp.merge(aerzte_df[['arzt_id', 'kanton']], on='arzt_id')
        df_comp = df_comp[df_comp['kanton'] == selected_kanton].drop(columns=['kanton'])

    # Charts
    st.subheader("📈 Änderungen nach Arzt (Top 10)")
    top_abs = df_comp.sort_values("diff", ascending=False).head(10)
    chart = alt.Chart(top_abs).mark_bar(color='#90ee90' if summary_det['absolut_diff'] > 0 else '#ffcccb').encode(
        x=alt.X('arzt_id:O', axis=alt.Axis(title='Arzt-ID')),
        y=alt.Y('diff:Q', axis=alt.Axis(title='Absolute Änderung (CHF)')),
        tooltip=['arzt_id', 'diff', 'diff_rel_pct']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption("Benefit: Identifiziert Gewinner/Verlierer – hilft, faire Anpassungen in Verhandlungen zu argumentieren.")

    st.subheader("📊 Verteilung der relativen Änderungen (%)")
    hist_data = pd.DataFrame({'rel_diff_pct': df_comp['diff_rel_pct']})
    chart_hist = alt.Chart(hist_data).mark_bar(color='#add8e6').encode(
        alt.X("rel_diff_pct:Q", bin=True, axis=alt.Axis(title='Relative Änderung (%)')),
        y=alt.Y('count():Q', axis=alt.Axis(title='Anzahl Ärzte')),
        tooltip=['count()']
    ).interactive()
    st.altair_chart(chart_hist, use_container_width=True)
    st.caption("Benefit: Zeigt Fairness – z. B. ob Änderung viele Ärzte belastet, für evidenzbasierte Verhandlungen.")

    # Table
    st.subheader("📋 Detailansicht (Top 15 absolute Änderungen)")
    df_top = df_comp.sort_values("diff", ascending=False).head(15)
    df_top['Implikation'] = df_top['diff_rel_pct'].apply(lambda x: "Starke Steigerung: Potenzial für Mehreinnahmen" if x > 5 else "Geringe Änderung: Stabil" if abs(x) < 5 else "Reduktion: Kosteneinsparung")
    st.dataframe(df_top.style.format({"alt":"{:.2f}","neu":"{:.2f}","diff":"{:.2f}","diff_rel_pct":"{:.2f}"}))
    st.caption("Benefit: Direkt ablesbare Implikationen – z. B. für faire Tarifverteilung in FMH-Verhandlungen.")

    # PDF Export – fixed
    pdf_buffer = generate_szenario_pdf(summary_det, selected_service, preis_mult, menge_mult, records_df, scenario)  # Änderung: records_df und scenario übergeben
    st.download_button('📄 PDF herunterladen', pdf_buffer, file_name='szenario_report.pdf', mime='application/pdf')

    # CSV Export
    csv_buf = io.StringIO()
    df_comp.to_csv(csv_buf, index=False)
    st.download_button("📥 CSV herunterladen", csv_buf.getvalue(), file_name="szenario_ergebnisse.csv", mime="text/csv")

else:
    st.info("⬅️ Wähle eine Leistung und Multiplikatoren und klicke **Szenario anwenden**.")