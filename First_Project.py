import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

st.title('🏀 NBA Player Stats Explorer & Dream Team Generator')

st.markdown("""
This app performs simple webscraping of NBA player stats data and builds **Dream Teams** using player statistics!  
* **Python libraries:** base64, pandas, streamlit  
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

# Sidebar Year Selection
st.sidebar.header('User Input Features')
year = datetime.now().year
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, year))))

# Use new Streamlit cache method
@st.cache_data
def load_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df['Age'] == 'Age'].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(columns=['Rk'])
    return playerstats

playerstats = load_data(selected_year)

# Sidebar filters
sorted_unique_team = sorted(playerstats['Team'].dropna().astype(str).unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtered dataset
df_selected_team = playerstats[
    (playerstats['Team'].isin(selected_team)) & (playerstats['Pos'].isin(selected_pos))
]

st.header('Display Player Stats of Selected Team(s)')
st.write(f"Data Dimension: {df_selected_team.shape[0]} rows and {df_selected_team.shape[1]} columns.")
st.dataframe(df_selected_team)

# CSV download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap section
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    numeric_df = df_selected_team.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns to compute a correlation matrix.")
    else:
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, mask=mask, vmax=1, square=True, ax=ax)
            st.pyplot(f)

# -------------------------------------------------------
# DREAM TEAM GENERATOR SECTION
# -------------------------------------------------------
st.subheader("🏆 Dream Team Generator")

# Convert important columns to numeric
for col in ['PTS', 'AST', 'TRB', 'STL', 'BLK']:
    if col in playerstats.columns:
        playerstats[col] = pd.to_numeric(playerstats[col], errors='coerce')

# Helper function to build dream team
def build_dream_team(df, seed=0):
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    team_players = []
    for pos in positions:
        players = df[df['Pos'] == pos].nlargest(4, 'PTS')  # top 4 scorers per position
        team_players.append(players)
    team_df = pd.concat(team_players).sample(10, random_state=seed)
    return team_df

# Create three dream teams
dream_team_1 = build_dream_team(playerstats, seed=1)
dream_team_2 = build_dream_team(playerstats, seed=2)
dream_team_3 = build_dream_team(playerstats, seed=3)

# Display dream teams
st.markdown("### 🥇 Dream Team 1")
st.dataframe(dream_team_1[['Player', 'Pos', 'Team', 'PTS', 'AST', 'TRB']])

st.markdown("### 🥈 Dream Team 2")
st.dataframe(dream_team_2[['Player', 'Pos', 'Team', 'PTS', 'AST', 'TRB']])

st.markdown("### 🥉 Dream Team 3")
st.dataframe(dream_team_3[['Player', 'Pos', 'Team', 'PTS', 'AST', 'TRB']])

# -------------------------------------------------------
# 📈 Win Percentage Estimation
# -------------------------------------------------------
def team_strength(team_df):
    weights = {'PTS': 0.4, 'AST': 0.2, 'TRB': 0.2, 'STL': 0.1, 'BLK': 0.1}
    score = sum(team_df[col].mean() * w for col, w in weights.items() if col in team_df)
    return score

strengths = {
    'Team 1': team_strength(dream_team_1),
    'Team 2': team_strength(dream_team_2),
    'Team 3': team_strength(dream_team_3)
}

# Normalize to get win %
total_strength = sum(strengths.values())
win_prob = {team: round((val / total_strength) * 100, 2) for team, val in strengths.items()}

st.markdown("## 🧮 Estimated Win Percentages")
st.bar_chart(pd.DataFrame(list(win_prob.items()), columns=['Team', 'Win %']).set_index('Team'))

# -------------------------------------------------------
#  Line Chart 
# -------------------------------------------------------
st.subheader("📊 Total Points Scored vs Team (Streamlit Line Chart)")
if 'PTS' in df_selected_team.columns:
    team_points = df_selected_team.groupby('Team')['PTS'].sum().reset_index()
    team_points = team_points.sort_values(by='PTS', ascending=False)
    st.line_chart(team_points.set_index('Team'))
else:
    st.warning("PTS column not available for visualization.")


# -------------------------------------------------------
# 🧠 NBA Insights RAG Agent (Gemini + FAISS)
# -------------------------------------------------------
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.markdown("---")
st.header("🏀 NBA Insights RAG Agent")

# Get API key from Streamlit secrets
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning("Gemini API key not found. Please add it in Streamlit Secrets.")
else:
    genai.configure(api_key=api_key)

    # Convert your playerstats DataFrame to text
    st.write("Preparing dataset embeddings... (first-time load may take 20s)")
    data_text = ""
    for _, row in playerstats.iterrows():
        data_text += f"Player: {row.get('Player', '')}, Position: {row.get('Pos', '')}, Team: {row.get('Team', '')}, PTS: {row.get('PTS', '')}, AST: {row.get('AST', '')}, TRB: {row.get('TRB', '')}, STL: {row.get('STL', '')}, BLK: {row.get('BLK', '')}\n"

    # Chunk data for embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(data_text)]

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    # Chat interface
    st.subheader("💬 Ask any basketball query based on this dataset:")
    user_query = st.text_input("Type your question:")

    if user_query:
        with st.spinner("Fetching insights..."):
            # Retrieve top matching chunks
            matched_docs = db.similarity_search(user_query, k=4)
            context = "\n".join([doc.page_content for doc in matched_docs])

            # Generate response using Gemini
            model = genai.GenerativeModel("gemini-pro")
            prompt = (
                "You are an expert NBA data analyst. "
                "Use only the data provided below to answer queries. "
                "Give clear and factual basketball insights.\n\n"
                f"Context:\n{context}\n\nUser Question: {user_query}"
            )

            try:
                response = model.generate_content(prompt)
                st.success("🏀 Insight:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error generating response: {e}")



