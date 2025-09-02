# app.py
import streamlit as st
import pandas as pd
import re
from datetime import datetime
import csv
import sys
import json
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Erhöhe CSV Field Size Limit für große Felder
csv.field_size_limit(sys.maxsize)

# Deutsche Stoppwörter
GERMAN_STOPWORDS = {
    'der', 'die', 'das', 'den', 'dem', 'des', 'und', 'oder', 'aber', 'in', 'im', 'ist', 'sind',
    'war', 'waren', 'sein', 'haben', 'hat', 'hatte', 'werden', 'wird', 'wurde', 'wurden',
    'mit', 'bei', 'zu', 'zum', 'zur', 'für', 'auf', 'an', 'von', 'vom', 'als', 'es', 'ein',
    'eine', 'einen', 'einem', 'einer', 'ich', 'du', 'er', 'sie', 'wir', 'ihr', 'nicht',
    'noch', 'da', 'dann', 'so', 'wie', 'was', 'wer', 'wo', 'wann', 'wenn', 'dass', 'kann',
    'will', 'muss', 'soll', 'darf', 'möchte', 'würde', 'mir', 'mich', 'dich', 'dir', 'uns',
    'euch', 'sich', 'man', 'gibt', 'geben', 'um', 'aus', 'nach', 'vor', 'über', 'unter',
    'zwischen', 'durch', 'bis', 'seit', 'mal', 'also', 'doch', 'nur', 'schon', 'auch', 'mehr',
    'sehr', 'viel', 'ganz', 'ganz', 'alle', 'alles', 'immer', 'wieder', 'hier', 'dort'
}

def extract_qa_pairs_with_time(path):
    all_records = []
    
    # Lese CSV direkt
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        csv_reader = csv.DictReader(f)
        
        for row in csv_reader:
            try:
                # Extrahiere Datum und Zeit aus der "Erstellt" Spalte
                erstellt = row.get("Erstellt", "")
                if erstellt and erstellt != "Erstellt":  # Skip header
                    datetime_obj = datetime.strptime(erstellt, "%Y-%m-%d %H:%M:%S")
                    date = datetime_obj.date()
                    
                    # Extrahiere Raw Chat Data
                    raw_data = row.get("Raw Chat Data", "")
                    if raw_data:
                        # Versuche JSON zu parsen
                        try:
                            # Ersetze doppelte Anführungszeichen für JSON-Parsing
                            json_str = raw_data.replace('""', '"')
                            chat_data = json.loads(json_str)
                            messages = chat_data.get("messages", [])
                            
                            # Finde User-Frage und Assistant-Antwort
                            user_text = None
                            assistant_text = None
                            
                            for i, msg in enumerate(messages):
                                if msg.get("role") == "user" and not user_text:
                                    parts = msg.get("parts", [])
                                    for part in parts:
                                        if part.get("type") == "text":
                                            user_text = part.get("text", "")
                                            break
                                
                                elif msg.get("role") == "assistant" and user_text and not assistant_text:
                                    parts = msg.get("parts", [])
                                    for part in parts:
                                        if part.get("type") == "text":
                                            assistant_text = part.get("text", "")
                                            break
                                    break
                            
                            if user_text:
                                all_records.append({
                                    "Datum": date,
                                    "Datetime": datetime_obj,
                                    "Wochentag": datetime_obj.strftime("%A"),
                                    "Stunde": datetime_obj.hour,
                                    "Frage": user_text,
                                    "Antwort": assistant_text or "Keine Antwort gefunden",
                                    "Fragenlänge_Zeichen": len(user_text),
                                    "Fragenlänge_Wörter": len(user_text.split())
                                })
                        
                        except (json.JSONDecodeError, Exception):
                            # Fallback: Nutze Regex-Extraktion
                            user_text = None
                            assistant_text = None
                            
                            # Extrahiere User-Text
                            if '""text""' in raw_data:
                                matches = re.findall(r'""text"":""([^"]+)""[^}]*\}[^}]*""role"":""user""', raw_data)
                                if matches:
                                    user_text = matches[0]
                            elif '"text"' in raw_data:
                                matches = re.findall(r'"text":"([^"]+)"[^}]*\}[^}]*"role":"user"', raw_data)
                                if matches:
                                    user_text = matches[0]
                            
                            # Extrahiere Assistant-Text (erste Antwort nach User)
                            if user_text:
                                if '""role"":""assistant""' in raw_data:
                                    pattern = r'""role"":""assistant"".*?""text"":""([^"]+)""'
                                    matches = re.findall(pattern, raw_data)
                                    if matches:
                                        assistant_text = matches[0]
                                elif '"role":"assistant"' in raw_data:
                                    pattern = r'"role":"assistant".*?"text":"([^"]+)"'
                                    matches = re.findall(pattern, raw_data)
                                    if matches:
                                        assistant_text = matches[0]
                                
                                all_records.append({
                                    "Datum": date,
                                    "Datetime": datetime_obj,
                                    "Wochentag": datetime_obj.strftime("%A"),
                                    "Stunde": datetime_obj.hour,
                                    "Frage": user_text,
                                    "Antwort": assistant_text or "Keine Antwort gefunden",
                                    "Fragenlänge_Zeichen": len(user_text),
                                    "Fragenlänge_Wörter": len(user_text.split())
                                })
                        
            except Exception as e:
                pass
    
    # Erstelle DataFrame
    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.dropna(subset=["Datum"])
        df = df.sort_values("Datetime")
        df = df.reset_index(drop=True)
        
        # Deutsche Wochentage
        weekday_map = {
            'Monday': 'Montag', 'Tuesday': 'Dienstag', 'Wednesday': 'Mittwoch',
            'Thursday': 'Donnerstag', 'Friday': 'Freitag', 
            'Saturday': 'Samstag', 'Sunday': 'Sonntag'
        }
        df['Wochentag'] = df['Wochentag'].map(weekday_map)
        
        # Tageszeit-Kategorien
        def get_tageszeit(hour):
            if 5 <= hour < 10:
                return "Morgen (5-10)"
            elif 10 <= hour < 14:
                return "Mittag (10-14)"
            elif 14 <= hour < 18:
                return "Nachmittag (14-18)"
            elif 18 <= hour < 22:
                return "Abend (18-22)"
            else:
                return "Nacht (22-5)"
        
        df['Tageszeit'] = df['Stunde'].apply(get_tageszeit)
        
        # Längenkategorien
        def get_length_category(length):
            if length < 50:
                return "Kurz (<50)"
            elif length < 150:
                return "Mittel (50-150)"
            else:
                return "Lang (>150)"
        
        df['Längenkategorie'] = df['Fragenlänge_Zeichen'].apply(get_length_category)
    
    return df

def extract_keywords(df, top_n=20):
    """Extrahiere die häufigsten Keywords aus allen Fragen"""
    all_words = []
    for frage in df['Frage']:
        # Normalisiere Text
        words = re.findall(r'\b[a-züäö]+\b', frage.lower())
        # Filtere Stoppwörter und kurze Wörter
        words = [w for w in words if w not in GERMAN_STOPWORDS and len(w) > 2]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)

def categorize_questions(df):
    """Kategorisiere Fragen nach Themen"""
    categories = []
    for frage in df['Frage']:
        frage_lower = frage.lower()
        cat = []
        
        # Rezepte
        if any(word in frage_lower for word in ['rezept', 'kochen', 'zubereiten', 'zubereitung', 'machen']):
            cat.append('Rezepte')
        
        # Fleischsorten
        if any(word in frage_lower for word in ['poulet', 'huhn', 'hähnchen', 'geflügel']):
            cat.append('Poulet')
        if any(word in frage_lower for word in ['rind', 'steak', 'filet', 'entrecote']):
            cat.append('Rind')
        if any(word in frage_lower for word in ['schwein', 'schnitzel', 'kotelett']):
            cat.append('Schwein')
        
        # Zubereitungsarten
        if any(word in frage_lower for word in ['grill', 'grillieren', 'bbq']):
            cat.append('Grillen')
        if any(word in frage_lower for word in ['braten', 'anbraten', 'pfanne']):
            cat.append('Braten')
        if any(word in frage_lower for word in ['schmoren', 'schmorbraten']):
            cat.append('Schmoren')
        
        categories.append(', '.join(cat) if cat else 'Sonstiges')
    
    return categories

# Streamlit App
st.set_page_config(page_title="Chat-Analyse Dashboard", layout="wide")

# Datei laden
file_path = "bonneviande_ai_chat.csv"
df = extract_qa_pairs_with_time(file_path)

if df.empty:
    st.error("Keine Frage-Antwort-Paare gefunden – bitte Datei prüfen.")
else:
    st.title("📊 Chat-Analyse Dashboard")
    st.write(f"Analysiere {len(df)} Frage-Antwort-Paare vom {df['Datum'].min().strftime('%d.%m.%Y')} bis {df['Datum'].max().strftime('%d.%m.%Y')}")
    
    # Tabs erstellen
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Übersicht", 
        "🔤 Top-Themen", 
        "⏰ Zeitliche Muster",
        "📏 Fragenlängen",
        "📈 Trends",
        "💬 Fragen & Antworten"
    ])
    
    with tab1:
        st.header("Übersicht")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gesamt Fragen", len(df))
        with col2:
            st.metric("Ø Fragen/Tag", f"{len(df) / (df['Datum'].max() - df['Datum'].min()).days:.1f}")
        with col3:
            st.metric("Ø Fragenlänge", f"{df['Fragenlänge_Zeichen'].mean():.0f} Zeichen")
        with col4:
            st.metric("Aktivste Stunde", f"{df['Stunde'].mode().values[0]}:00 Uhr")
        
        st.subheader("Fragen pro Tag")
        fragen_pro_tag = df.groupby("Datum").size().reset_index(name="Anzahl")
        fig = px.bar(fragen_pro_tag, x="Datum", y="Anzahl", 
                     title="Anzahl Fragen pro Tag",
                     labels={"Anzahl": "Anzahl Fragen", "Datum": "Datum"})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Top-Themen & Keywords")
        
        # Extrahiere Keywords
        keywords = extract_keywords(df, top_n=25)
        
        # Erstelle Balkendiagramm
        words, counts = zip(*keywords) if keywords else ([], [])
        fig = px.bar(x=counts, y=words, orientation='h',
                     title="Top 25 Keywords",
                     labels={"x": "Häufigkeit", "y": "Keyword"})
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Word Cloud
        if keywords:
            st.subheader("Word Cloud")
            text = ' '.join([word for word, count in keywords for _ in range(count)])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with tab3:
        st.header("Zeitliche Muster")
        
        # Wochentag-Analyse
        st.subheader("Aktivität nach Wochentag")
        weekday_order = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
        weekday_counts = df['Wochentag'].value_counts().reindex(weekday_order, fill_value=0)
        fig = px.bar(x=weekday_order, y=weekday_counts.values,
                     title="Fragen nach Wochentag",
                     labels={"x": "Wochentag", "y": "Anzahl Fragen"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tageszeit-Analyse
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Aktivität nach Tageszeit")
            tageszeit_order = ["Morgen (5-10)", "Mittag (10-14)", "Nachmittag (14-18)", "Abend (18-22)", "Nacht (22-5)"]
            tageszeit_counts = df['Tageszeit'].value_counts().reindex(tageszeit_order, fill_value=0)
            fig = px.pie(values=tageszeit_counts.values, names=tageszeit_order,
                        title="Verteilung nach Tageszeit")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Aktivität nach Stunde")
            hour_counts = df['Stunde'].value_counts().sort_index()
            fig = px.bar(x=hour_counts.index, y=hour_counts.values,
                        title="Fragen nach Uhrzeit",
                        labels={"x": "Stunde", "y": "Anzahl Fragen"})
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap Wochentag x Stunde
        st.subheader("Heatmap: Wochentag × Stunde")
        pivot_table = df.pivot_table(values='Frage', index='Wochentag', columns='Stunde', aggfunc='count', fill_value=0)
        pivot_table = pivot_table.reindex(weekday_order, fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=[f"{h}:00" for h in pivot_table.columns],
            y=pivot_table.index,
            colorscale='Blues',
            text=pivot_table.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Anzahl<br>Fragen")
        ))
        fig.update_layout(
            title="Aktivität nach Wochentag und Stunde",
            xaxis_title="Uhrzeit",
            yaxis_title="Wochentag",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Fragenlängen-Analyse")
        
        # Statistiken
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min. Länge", f"{df['Fragenlänge_Zeichen'].min()} Zeichen")
        with col2:
            st.metric("Max. Länge", f"{df['Fragenlänge_Zeichen'].max()} Zeichen")
        with col3:
            st.metric("Durchschnitt", f"{df['Fragenlänge_Zeichen'].mean():.0f} Zeichen")
        with col4:
            st.metric("Median", f"{df['Fragenlänge_Zeichen'].median():.0f} Zeichen")
        
        # Histogramm
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Fragenlänge_Zeichen', nbins=30,
                              title="Verteilung der Fragenlänge (Zeichen)",
                              labels={"Fragenlänge_Zeichen": "Anzahl Zeichen", "count": "Häufigkeit"})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='Fragenlänge_Wörter', nbins=20,
                              title="Verteilung der Fragenlänge (Wörter)",
                              labels={"Fragenlänge_Wörter": "Anzahl Wörter", "count": "Häufigkeit"})
            st.plotly_chart(fig, use_container_width=True)
        
        # Längenkategorien
        st.subheader("Kategorisierung nach Länge")
        length_cats = df['Längenkategorie'].value_counts()
        fig = px.pie(values=length_cats.values, names=length_cats.index,
                    title="Verteilung der Fragenlängen-Kategorien")
        st.plotly_chart(fig, use_container_width=True)
        
        # Längste und kürzeste Fragen
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Kürzeste Fragen")
            shortest = df.nsmallest(5, 'Fragenlänge_Zeichen')[['Frage', 'Fragenlänge_Zeichen']]
            for _, row in shortest.iterrows():
                st.write(f"**{row['Fragenlänge_Zeichen']} Zeichen:** {row['Frage']}")
        
        with col2:
            st.subheader("Längste Fragen")
            longest = df.nlargest(5, 'Fragenlänge_Zeichen')[['Frage', 'Fragenlänge_Zeichen']]
            for _, row in longest.iterrows():
                st.write(f"**{row['Fragenlänge_Zeichen']} Zeichen:** {row['Frage'][:100]}...")
    
    with tab5:
        st.header("Trends & Kategorien")
        
        # Kategorisiere Fragen
        df['Kategorie'] = categorize_questions(df)
        
        # Zeige Kategorieverteilung
        st.subheader("Verteilung der Fragenkategorien")
        cat_counts = pd.Series([cat for cats in df['Kategorie'].str.split(', ') for cat in cats if cat]).value_counts()
        fig = px.bar(x=cat_counts.values, y=cat_counts.index, orientation='h',
                    title="Häufigkeit der Kategorien",
                    labels={"x": "Anzahl", "y": "Kategorie"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Trends über Zeit
        st.subheader("Kategorien-Trends über Zeit")
        
        # Erstelle tägliche Aggregation für Hauptkategorien
        main_categories = ['Rezepte', 'Poulet', 'Rind', 'Schwein', 'Grillen']
        trend_data = []
        
        for date in df['Datum'].unique():
            day_data = df[df['Datum'] == date]
            for cat in main_categories:
                count = sum(cat in str(cats) for cats in day_data['Kategorie'])
                trend_data.append({'Datum': date, 'Kategorie': cat, 'Anzahl': count})
        
        trend_df = pd.DataFrame(trend_data)
        
        if not trend_df.empty:
            fig = px.line(trend_df, x='Datum', y='Anzahl', color='Kategorie',
                         title="Entwicklung der Hauptkategorien über Zeit",
                         labels={"Anzahl": "Anzahl Fragen", "Datum": "Datum"})
            st.plotly_chart(fig, use_container_width=True)
        
        # Korrelation zwischen Länge und Kategorie
        st.subheader("Durchschnittliche Fragenlänge nach Kategorie")
        avg_length_by_cat = []
        for cat in cat_counts.index[:10]:  # Top 10 Kategorien
            cat_questions = df[df['Kategorie'].str.contains(cat, na=False)]
            if not cat_questions.empty:
                avg_length_by_cat.append({
                    'Kategorie': cat,
                    'Durchschnittslänge': cat_questions['Fragenlänge_Zeichen'].mean()
                })
        
        if avg_length_by_cat:
            avg_df = pd.DataFrame(avg_length_by_cat)
            fig = px.bar(avg_df, x='Kategorie', y='Durchschnittslänge',
                        title="Durchschnittliche Fragenlänge nach Kategorie",
                        labels={"Durchschnittslänge": "Ø Zeichen"})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("Fragen & Antworten durchsuchen")
        
        # Suchfunktion
        search_term = st.text_input("🔍 Suche in Fragen:", "")
        
        # Datumsfilter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Von:", value=df['Datum'].min())
        with col2:
            end_date = st.date_input("Bis:", value=df['Datum'].max())
        
        # Filtere Daten
        filtered_df = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)]
        
        if search_term:
            filtered_df = filtered_df[filtered_df['Frage'].str.contains(search_term, case=False, na=False)]
        
        st.write(f"Gefundene Einträge: {len(filtered_df)}")
        
        # Zeige Frage-Antwort-Paare
        for idx, row in filtered_df.head(50).iterrows():  # Limitiere auf 50 für Performance
            with st.expander(f"📅 {row['Datum']} | 💬 {row['Frage'][:100]}..."):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**Datum:** {row['Datum']}")
                    st.write(f"**Uhrzeit:** {row['Datetime'].strftime('%H:%M')}")
                    st.write(f"**Wochentag:** {row['Wochentag']}")
                    st.write(f"**Länge:** {row['Fragenlänge_Zeichen']} Zeichen")
                    if row['Kategorie'] != 'Sonstiges':
                        st.write(f"**Kategorie:** {row['Kategorie']}")
                
                with col2:
                    st.markdown("**Frage:**")
                    st.write(row['Frage'])
                    st.markdown("**Antwort:**")
                    # Begrenze Antwortlänge für bessere Lesbarkeit
                    antwort = row['Antwort']
                    if len(antwort) > 1000:
                        antwort = antwort[:1000] + "..."
                    st.write(antwort)