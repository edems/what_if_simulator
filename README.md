# What-if Simulator

## Beschreibung
Dieser Prototyp ist ein interaktives Tool zur Simulation von "Was-wäre-wenn"-Szenarien in der Tarifanalyse für die Schweizer Ärzteschaft. Basierend auf synthetischen Daten (inspiriert von NAKO/OBELISC), ermöglicht es die Anpassung von Preisen und Mengen für TARMED-Leistungen und berechnet die finanziellen Auswirkungen (deterministisch und ML-basiert). Es unterstützt evidenzbasierte Tarifverhandlungen, z. B. für FMH oder Kantonsärztevereine, indem es Kostensteigerungen, Fairness und Risiken visualisiert.

**Key Features:**
- Auswahl realer TARMED-Leistungen (z. B. "Ärztliches Gespräch / Beratung").
- Slider für Preis-/Mengen-Multiplikatoren.
- Metrics für Gesamtbeträge und Differenzen.
- Interaktive Charts (Top-Änderungen, Verteilung).
- Export als CSV oder PDF-Report.
- ML-Prognosen mit Gradient Boosting für robuste Insights.

Das Tool ist DSG-konform (anonymisiert, synthetisch) und modular erweiterbar für reale Daten.

## Installation
1. Klonen des Repos: `git clone https://github.com/dein-repo/what-if-simulator.git`
2. Abhängigkeiten installieren: `pip install -r requirements.txt` (enthält streamlit, pandas, numpy, scikit-learn, altair, reportlab).
3. App starten: `streamlit run streamlit_app.py`

**Hinweis:** Keine Internetzugang nötig; alle Bibliotheken sind offline.

## Usage
1. Öffne die App im Browser (nach Start: http://localhost:8501).
2. In der Sidebar: Wähle eine Leistung, passe Multiplikatoren an und klicke "Szenario anwenden".
3. Sieh Ergebnisse: Metrics, Charts, Tabelle.
4. Exportiere PDF/CSV für Berichte.
5. Filtere nach Kanton für regionale Analysen.

Beispiel-Szenario: Erhöhe Preis für "Röntgen Thorax" um 20% – sieh Auswirkungen auf Gesamtkosten und Fairness.

## Benefits für Tarifverhandlungen
- **Evidenzbasierte Argumente**: Simuliere Änderungen und zeige, wie sie Kosten steuern – ideal für FMH-Verhandlungen.
- **Fairness-Analyse**: Verteilungscharts decken Ungleichgewichte auf (z. B. kleine vs. große Praxen, Kantone).
- **Risikominimierung**: Prognostiziere Ersparnisse/Mehreinnahmen, inkl. 5-Jahres-Hochrechnung und Ersparnis pro Arzt.
- **Schnelle Iteration**: Teste Alternativen live – stärkt Position gegenüber Versicherern/Bundesrat.
- **Datenschutz**: Nur synthetische Daten – erweiterbar für reale NAKO-Daten.

## Warnung
Dies ist ein Prototyp mit synthetischen Daten – keine Garantie für Realwelt-Genauigkeit. Für produktiven Einsatz: Mit realen Daten validieren und rechtlich prüfen (DSG). Feedback willkommen!

## Kontakt
Adam Horn – info@adamhorn-ki.ch
