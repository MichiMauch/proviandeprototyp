# 📊 Chat-Analyse Dashboard

Ein umfassendes Analytics-Dashboard zur Analyse von Chat-Daten mit interaktiven Visualisierungen und detaillierten Insights.

## 🚀 Features

### 📅 Übersicht
- **Key Metrics:** Gesamt Fragen, Ø Fragen/Tag, Ø Fragenlänge, Aktivste Stunde
- **Zeitverlauf:** Interaktive Visualisierung der Aktivität über Zeit

### 🔤 Top-Themen & Keywords
- **Keyword-Analyse** mit deutschen Stoppwort-Filterung
- **Word Cloud** für visuelle Darstellung
- Top 25 häufigste Begriffe

### ⏰ Zeitliche Muster
- **Wochentag-Analyse:** Wann sind User am aktivsten?
- **Tageszeit-Verteilung:** Morgen, Mittag, Nachmittag, Abend, Nacht
- **Heatmap:** Detaillierte Aktivitätsmuster nach Tag und Stunde

### 📏 Fragenlängen-Analyse
- **Statistiken:** Min, Max, Durchschnitt, Median
- **Kategorisierung:** Kurz, Mittel, Lang
- **Histogramme** für Zeichen- und Wortverteilung

### 📈 Trends & Kategorien
- **Automatische Kategorisierung:** Rezepte, Fleischsorten, Zubereitungsarten
- **Zeitliche Entwicklung** der Kategorien
- **Korrelationsanalysen** zwischen Länge und Themen

### 💬 Durchsuchbare Frage-Antwort-Datenbank
- **Volltext-Suche** in allen Fragen
- **Datumsfilter** für gezielte Analysen
- **Detailansicht** mit Metadaten

## 🛠️ Installation

### Voraussetzungen
- Python 3.8 oder höher
- pip (Python Package Manager)

### Schritt-für-Schritt Installation

1. **Repository klonen**
   ```bash
   git clone git@github.com:MichiMauch/proviandeprototyp.git
   cd proviandeprototyp
   ```

2. **Virtuelle Umgebung erstellen (empfohlen)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Auf Windows: .venv\Scripts\activate
   ```

3. **Dependencies installieren**
   ```bash
   pip install -r requirements.txt
   ```

4. **App starten**
   ```bash
   streamlit run app.py
   ```

5. **Dashboard öffnen**
   - Die App öffnet sich automatisch im Browser
   - Oder besuche: `http://localhost:8501`

## 📊 Datenformat

Das Dashboard erwartet eine CSV-Datei namens `bonneviande_ai_chat.csv` mit folgendem Format:
- **Spalte "Erstellt":** Zeitstempel im Format `YYYY-MM-DD HH:MM:SS`
- **Spalte "Raw Chat Data":** JSON-formatierte Chat-Nachrichten mit User-Fragen und Assistant-Antworten

## 🚀 Deployment

### Streamlit Cloud
1. Repository zu GitHub pushen
2. Bei [Streamlit Cloud](https://streamlit.io/cloud) anmelden
3. Repository verknüpfen und deployen
4. Live-URL wird automatisch generiert

### Lokale Entwicklung
```bash
# Entwicklungsmodus mit Auto-Reload
streamlit run app.py --server.runOnSave true
```

## 📈 Analysierte Metriken

- **330+ Frage-Antwort-Paare** (Juli-September 2025)
- **Kategorien:** Rezepte, Poulet, Rind, Schwein, Grillen, Braten
- **Zeitraum:** 21.07.2025 bis 01.09.2025
- **Keywords:** Automatische Erkennung relevanter Begriffe

## 🔧 Technische Details

### Dependencies
- **Streamlit** 1.49+ - Web-App Framework
- **Pandas** 2.0+ - Datenverarbeitung
- **Plotly** 6.0+ - Interaktive Visualisierungen
- **WordCloud** 1.9+ - Word Cloud Generation
- **Matplotlib** 3.8+ - Plotting
- **NumPy** 1.24+ - Numerische Berechnungen

### Architektur
- **Modular aufgebaut** mit separaten Funktionen für Datenextraktion und -analyse
- **Tab-basierte UI** für übersichtliche Navigation
- **Caching** für optimierte Performance
- **Responsive Design** für verschiedene Bildschirmgrößen

## 📝 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

## 🤝 Beitragen

1. Fork das Repository
2. Erstelle einen Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Committe deine Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Pushe zum Branch (`git push origin feature/AmazingFeature`)
5. Öffne eine Pull Request

## 📞 Support

Bei Fragen oder Problemen erstelle bitte ein [Issue](https://github.com/MichiMauch/proviandeprototyp/issues) im Repository.

---
*Erstellt mit ❤️ und [Streamlit](https://streamlit.io)*