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
import os
from dotenv import load_dotenv
import openai
import time

# Erhöhe CSV Field Size Limit für große Felder
csv.field_size_limit(sys.maxsize)

# Lade Umgebungsvariablen
load_dotenv('.env.local')

# OpenAI Konfiguration
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
openai_max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '500'))
openai_temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))

# OpenAI Client initialisieren wenn API Key vorhanden
if openai_api_key and openai_api_key != 'your-openai-api-key-here':
    client = openai.OpenAI(api_key=openai_api_key)
    use_openai = True
else:
    use_openai = False

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

# Keyword-Listen für Klassifizierung
FOOD_KEYWORDS = {
    'rezept', 'kochen', 'zubereiten', 'zubereitung', 'braten', 'grillen', 'grillieren', 
    'schmoren', 'backen', 'fleisch', 'poulet', 'huhn', 'hähnchen', 'geflügel', 
    'rind', 'rindfleisch', 'steak', 'filet', 'entrecote', 'schwein', 'schweinefleisch',
    'schnitzel', 'kotelett', 'wurst', 'schinken', 'speck', 'hackfleisch', 'gulasch',
    'sauce', 'marinade', 'gewürze', 'salz', 'pfeffer', 'kräuter', 'essen', 'mahlzeit',
    'mittag', 'abend', 'dinner', 'lunch', 'menü', 'teller', 'portion', 'servieren',
    'geschmack', 'würzig', 'scharf', 'mild', 'süss', 'sauer', 'salzig', 'bitter',
    'metzgete', 'blutwurst', 'leberwurst', 'gnagi', 'schnörrli', 'rippli', 'bratwurst'
}

AGGRESSIVE_KEYWORDS = {
    'scheiss', 'scheiß', 'mist', 'verdammt', 'verflucht', 'blöd', 'dumm', 'idiot', 
    'doof', 'bescheuert', 'beschissen', 'kacke', 'dreck', 'müll', 'schrott',
    'arsch', 'kotzen', 'kotzt', 'nervt', 'nervig', 'ätzend', 'zum kotzen',
    'schlecht', 'furchtbar', 'schrecklich', 'grausam', 'widerlich', 'ekelig',
    'hass', 'hasse', 'verachten', 'töten', 'sterben', 'kaputtgehen'
}

POSITIVE_KEYWORDS = {
    'super', 'toll', 'wunderbar', 'fantastisch', 'großartig', 'perfekt', 'excellent', 
    'ausgezeichnet', 'hervorragend', 'brilliant', 'genial', 'klasse', 'prima',
    'gut', 'sehr gut', 'lecker', 'köstlich', 'delikat', 'schmackhaft', 'fein',
    'danke', 'dankeschön', 'vielen dank', 'freue', 'freut', 'glücklich', 'zufrieden',
    'liebe', 'mag', 'gefällt', 'begeistert', 'empfehlen', 'empfehle', 'top',
    'wow', 'cool', 'geil', 'hammer', 'stark', 'spitze', 'bombig'
}

NEGATIVE_KEYWORDS = {
    'schlecht', 'übel', 'furchtbar', 'schrecklich', 'grausam', 'widerlich', 'ekelig',
    'enttäuscht', 'enttäuschend', 'langweilig', 'fade', 'geschmacklos', 'trocken',
    'zäh', 'hart', 'kalt', 'versalzen', 'verbrennt', 'verbrannt', 'bitter',
    'problem', 'fehler', 'falsch', 'nicht gut', 'ungeniessbar', 'ungeniessen',
    'beschwerde', 'reklamation', 'unzufrieden', 'ärgern', 'ärgerlich', 'frustriert',
    'hilfe', 'klappt nicht', 'funktioniert nicht', 'geht nicht', 'verstehe nicht'
}

def classify_message_with_ai(message_text, role, max_retries=3):
    """Klassifiziert eine Nachricht mit OpenAI API mit Retry-Logik für Rate Limits"""
    if not use_openai:
        return classify_message_rule_based(message_text, role)
    
    for attempt in range(max_retries):
        try:
            prompt = f"""Du bist ein Analysesystem. Klassifiziere diese Nachricht:

Nachricht: {message_text}
Rolle: {role}

Antworte im JSON-Format:
{{
    "kategorie": "essen/fleisch" | "neutral" | "aggressiv" | "anderes",
    "sentiment": "positiv" | "neutral" | "negativ"
}}

Kategorien:
- essen/fleisch: Fragen zu Rezepten, Fleisch, Kochen, Zubereitung
- aggressiv: Schimpfwörter, Beleidigungen, aggressive Sprache
- neutral: Normale, sachliche Nachrichten
- anderes: Alles andere

Sentiment:
- positiv: Lob, Dank, Freude, Begeisterung
- negativ: Kritik, Beschwerden, Probleme, Unzufriedenheit
- neutral: Sachliche, emotionslose Nachrichten"""

            response = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "system", "content": prompt}],
                temperature=openai_temperature,
                max_tokens=openai_max_tokens
            )
            
            # Parse JSON aus der Antwort
            response_text = response.choices[0].message.content
            # Versuche JSON zu extrahieren, auch wenn Text drumherum ist
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback auf manuelle Extraktion
                result = {"kategorie": "anderes", "sentiment": "neutral"}
            return {
                "frage": message_text,
                "rolle": role,
                "kategorie": result.get("kategorie", "anderes"),
                "sentiment": result.get("sentiment", "neutral")
            }
            
        except Exception as e:
            error_str = str(e)
            
            # Rate Limit Behandlung (429 Error)
            if "429" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    if 'rate_limit_info' not in st.session_state:
                        st.session_state.rate_limit_info = True
                        st.info(f"⏳ Rate Limit erreicht. Warte {wait_time}s und versuche erneut... (Versuch {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Nach allen Versuchen: Zeige Warnung und nutze Fallback
                    if 'rate_limit_final' not in st.session_state:
                        st.session_state.rate_limit_final = True
                        st.warning("⚠️ GPT-4 Rate Limit erreicht. Nutze regelbasierte Klassifizierung als Fallback.")
                    return classify_message_rule_based(message_text, role)
            else:
                # Andere Fehler: Zeige Warnung und nutze Fallback
                if 'openai_error_shown' not in st.session_state:
                    st.session_state.openai_error_shown = True
                    st.sidebar.error(f"OpenAI API Fehler: {error_str[:100]}... Nutze regelbasierte Klassifizierung.")
                return classify_message_rule_based(message_text, role)
    
    # Fallback falls alle Versuche fehlschlagen
    return classify_message_rule_based(message_text, role)

def classify_message_rule_based(message_text, role):
    """Regelbasierte Klassifizierung als Fallback"""
    if not message_text or not isinstance(message_text, str):
        return {
            "frage": message_text or "",
            "rolle": role,
            "kategorie": "anderes",
            "sentiment": "neutral"
        }
    
    text_lower = message_text.lower()
    
    # Kategorie bestimmen
    kategorie = "neutral"
    
    # Check für Food/Fleisch Keywords
    if any(keyword in text_lower for keyword in FOOD_KEYWORDS):
        kategorie = "essen/fleisch"
    
    # Check für aggressive Keywords (überschreibt andere)
    if any(keyword in text_lower for keyword in AGGRESSIVE_KEYWORDS):
        kategorie = "aggressiv"
    
    # Wenn weder Food noch aggressiv, aber spezifische andere Inhalte
    if kategorie == "neutral":
        # Sehr kurze Nachrichten oder nur Zahlen/Symbole
        if len(text_lower.strip()) < 5 or text_lower.isdigit():
            kategorie = "anderes"
    
    # Sentiment bestimmen
    positive_count = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text_lower)
    negative_count = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text_lower)
    
    if positive_count > negative_count and positive_count > 0:
        sentiment = "positiv"
    elif negative_count > positive_count and negative_count > 0:
        sentiment = "negativ"
    else:
        sentiment = "neutral"
    
    # Aggressive Kategorie impliziert meist negatives Sentiment
    if kategorie == "aggressiv" and sentiment == "neutral":
        sentiment = "negativ"
    
    return {
        "frage": message_text,
        "rolle": role,
        "kategorie": kategorie,
        "sentiment": sentiment
    }

def get_latest_messages(df, limit=20):
    """Extrahiert die neuesten Nachrichten für Klassifizierung"""
    if df.empty:
        return []
    
    # Sortiere nach Datum/Zeit (neueste zuerst)
    df_sorted = df.sort_values('Datetime', ascending=False).head(limit)
    
    messages = []
    for _, row in df_sorted.iterrows():
        # User-Nachricht
        messages.append({
            'text': row['Frage'],
            'role': 'user',
            'datetime': row['Datetime']
        })
        
        # Assistant-Nachricht (falls vorhanden)
        if row['Antwort'] and row['Antwort'] != "Keine Antwort gefunden":
            messages.append({
                'text': row['Antwort'],
                'role': 'assistant', 
                'datetime': row['Datetime']
            })
    
    # Nach Zeit sortieren (neueste zuerst) und auf limit begrenzen
    messages = sorted(messages, key=lambda x: x['datetime'], reverse=True)[:limit]
    
    return messages

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📅 Übersicht", 
        "🔤 Top-Themen", 
        "⏰ Zeitliche Muster",
        "📏 Fragenlängen",
        "📈 Trends",
        "💬 Fragen & Antworten",
        "🤖 KI-Klassifizierung"
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
    
    with tab7:
        st.header("🤖 KI-Klassifizierung der neuesten Nachrichten")
        
        # OpenAI Status Anzeige
        if use_openai:
            st.success(f"✅ OpenAI API aktiv (Modell: {openai_model})")
        else:
            st.info("ℹ️ Regelbasierte Klassifizierung aktiv. Füge deinen OpenAI API Key in .env.local hinzu für KI-basierte Analyse.")
        
        # Konfiguration
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            message_limit = st.slider("Anzahl Nachrichten analysieren:", 5, 50, 20)
        with col2:
            if st.button("🔄 Neu analysieren"):
                st.experimental_rerun()
        with col3:
            show_details = st.checkbox("Details anzeigen", value=True)
        
        # Extrahiere neueste Nachrichten
        latest_messages = get_latest_messages(df, limit=message_limit)
        
        if not latest_messages:
            st.warning("Keine Nachrichten zum Analysieren gefunden.")
        else:
            # Klassifiziere alle Nachrichten
            classified_messages = []
            with st.spinner('Analysiere Nachrichten...'):
                for msg in latest_messages:
                    classification = classify_message_with_ai(msg['text'], msg['role'])
                    classification['datetime'] = msg['datetime']
                    classified_messages.append(classification)
            
            # Erstelle DataFrame für Analyse
            class_df = pd.DataFrame(classified_messages)
            
            # Übersichts-Metriken
            st.subheader("📊 Klassifizierungs-Übersicht")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_msgs = len(classified_messages)
                st.metric("Analysierte Nachrichten", total_msgs)
            
            with col2:
                food_count = sum(1 for c in classified_messages if c['kategorie'] == 'essen/fleisch')
                st.metric("🍖 Essen/Fleisch", food_count, f"{food_count/total_msgs*100:.1f}%")
            
            with col3:
                positive_count = sum(1 for c in classified_messages if c['sentiment'] == 'positiv')
                st.metric("😊 Positiv", positive_count, f"{positive_count/total_msgs*100:.1f}%")
            
            with col4:
                aggressive_count = sum(1 for c in classified_messages if c['kategorie'] == 'aggressiv')
                st.metric("🔴 Aggressiv", aggressive_count, f"{aggressive_count/total_msgs*100:.1f}%" if total_msgs > 0 else "0%")
            
            # Visualisierungen
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Kategorieverteilung")
                cat_counts = class_df['kategorie'].value_counts()
                
                # Farben für Kategorien
                color_map = {
                    'essen/fleisch': '#28a745',  # Grün
                    'neutral': '#007bff',        # Blau
                    'aggressiv': '#dc3545',      # Rot
                    'anderes': '#ffc107'         # Gelb
                }
                colors = [color_map.get(cat, '#6c757d') for cat in cat_counts.index]
                
                fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                           title="Verteilung der Kategorien",
                           color_discrete_sequence=colors)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Sentiment-Analyse")
                sent_counts = class_df['sentiment'].value_counts()
                
                # Farben für Sentiment
                sent_colors = {
                    'positiv': '#28a745',    # Grün
                    'neutral': '#6c757d',    # Grau
                    'negativ': '#dc3545'     # Rot
                }
                colors = [sent_colors.get(sent, '#6c757d') for sent in sent_counts.index]
                
                fig = px.bar(x=sent_counts.index, y=sent_counts.values,
                           title="Sentiment-Verteilung",
                           color=sent_counts.index,
                           color_discrete_map=sent_colors,
                           labels={"x": "Sentiment", "y": "Anzahl"})
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment-Timeline
            if len(classified_messages) > 5:
                st.subheader("📈 Sentiment-Entwicklung über Zeit")
                
                # Erstelle numerische Sentiment-Werte für Trend
                sentiment_values = []
                for c in classified_messages:
                    if c['sentiment'] == 'positiv':
                        sentiment_values.append(1)
                    elif c['sentiment'] == 'negativ':
                        sentiment_values.append(-1)
                    else:
                        sentiment_values.append(0)
                
                timeline_df = pd.DataFrame({
                    'Nachricht': range(len(classified_messages), 0, -1),  # Rückwärts zählen
                    'Sentiment_Score': sentiment_values,
                    'Sentiment': [c['sentiment'] for c in classified_messages],
                    'Zeit': [c['datetime'] for c in classified_messages]
                })
                
                fig = px.line(timeline_df, x='Nachricht', y='Sentiment_Score',
                             title="Sentiment-Trend (neueste → älteste Nachrichten)",
                             labels={"Nachricht": "Nachrichtennummer", "Sentiment_Score": "Sentiment (-1 bis +1)"},
                             markers=True)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                fig.update_traces(line_color='#007bff')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detaillierte Tabelle
            if show_details:
                st.subheader("📋 Detaillierte Klassifizierung")
                
                # Filter-Optionen
                col1, col2 = st.columns(2)
                with col1:
                    filter_category = st.selectbox("Filter nach Kategorie:", 
                                                 ["Alle"] + list(class_df['kategorie'].unique()))
                with col2:
                    filter_sentiment = st.selectbox("Filter nach Sentiment:",
                                                  ["Alle"] + list(class_df['sentiment'].unique()))
                
                # Filtere DataFrame
                display_df = class_df.copy()
                if filter_category != "Alle":
                    display_df = display_df[display_df['kategorie'] == filter_category]
                if filter_sentiment != "Alle":
                    display_df = display_df[display_df['sentiment'] == filter_sentiment]
                
                # Formatiere für Anzeige
                display_df['Zeit'] = pd.to_datetime(display_df['datetime']).dt.strftime('%d.%m.%Y %H:%M')
                display_df = display_df[['Zeit', 'rolle', 'kategorie', 'sentiment', 'frage']]
                display_df.columns = ['🕐 Zeit', '👤 Rolle', '📂 Kategorie', '😊 Sentiment', '💬 Nachricht']
                
                # Zeige Tabelle mit Farbkodierung
                for idx, row in display_df.iterrows():
                    # Farbe basierend auf Kategorie
                    if row['📂 Kategorie'] == 'essen/fleisch':
                        color = "background-color: #d4edda"
                    elif row['📂 Kategorie'] == 'aggressiv':
                        color = "background-color: #f8d7da"
                    elif row['📂 Kategorie'] == 'neutral':
                        color = "background-color: #d1ecf1"
                    else:
                        color = "background-color: #fff3cd"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="{color}; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #007bff;">
                            <strong>{row['🕐 Zeit']}</strong> | 
                            <span style="background: white; padding: 2px 6px; border-radius: 3px;">{row['👤 Rolle']}</span> | 
                            <span style="background: white; padding: 2px 6px; border-radius: 3px;">{row['📂 Kategorie']}</span> | 
                            <span style="background: white; padding: 2px 6px; border-radius: 3px;">{row['😊 Sentiment']}</span>
                            <br><br>
                            "{row['💬 Nachricht'][:200]}{'...' if len(row['💬 Nachricht']) > 200 else ''}"
                        </div>
                        """, unsafe_allow_html=True)
            
            # Export-Möglichkeit
            st.subheader("📥 Export")
            if st.button("📊 Klassifizierung als JSON exportieren"):
                json_data = json.dumps(classified_messages, indent=2, default=str, ensure_ascii=False)
                st.download_button(
                    label="💾 JSON herunterladen",
                    data=json_data,
                    file_name=f"klassifizierung_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )