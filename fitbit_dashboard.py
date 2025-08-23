import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import os

import firebase_admin
from firebase_admin import credentials, firestore

# ---- Initialize Firebase Admin SDK ----
if not firebase_admin._apps:
    firebase_cred_dict = dict(st.secrets["firebase"])
    firebase_cred_dict["private_key"] = firebase_cred_dict["private_key"].replace("\\n", "\n").strip()
    cred = credentials.Certificate(firebase_cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---- OAuth / Secrets ----
FITBIT_CLIENT_ID = st.secrets["FITBIT_CLIENT_ID"]
FITBIT_CLIENT_SECRET = st.secrets["FITBIT_CLIENT_SECRET"]
GOOGLE_CLIENT_ID = st.secrets["google_fit"]["client_id"]
GOOGLE_CLIENT_SECRET = st.secrets["google_fit"]["client_secret"]
GOOGLE_REDIRECT_URI = st.secrets["google_fit"]["redirect_uri"]

# Fitbit
FITBIT_AUTH_URL = (
    f"https://www.fitbit.com/oauth2/authorize?"
    f"response_type=code&client_id={FITBIT_CLIENT_ID}&redirect_uri={GOOGLE_REDIRECT_URI}"
    f"&scope=weight&expires_in=604800&prompt=login"
)
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
FITBIT_TOKENS_DOC = "fitbit/tokens"

# Google Fit
GOOGLE_AUTH_URL = (
    f"https://accounts.google.com/o/oauth2/v2/auth?"
    f"client_id={GOOGLE_CLIENT_ID}"
    f"&redirect_uri={GOOGLE_REDIRECT_URI}"
    f"&response_type=code"
    f"&scope=https://www.googleapis.com/auth/fitness.activity.read"
    f"&access_type=offline"
    f"&prompt=consent"
)
GOOGLE_TOKENS_DOC = "google_fit/tokens"

# ---- Helper Functions ----
def save_tokens(doc_path, tokens):
    db.document(doc_path).set(tokens)

def load_tokens(doc_path):
    doc = db.document(doc_path).get()
    if doc.exists:
        return doc.to_dict()
    return {}

def kg_to_lbs(kg):
    return kg * 2.20462

def lbs_to_st_lbs(lbs):
    stn = int(lbs // 14)
    rem_lbs = lbs % 14
    return f"{stn}st {rem_lbs:.1f}lbs"

def st_to_lbs(stone):
    return stone * 14

# ---- Fitbit OAuth / API ----
def get_fitbit_token_from_code(code):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "client_id": FITBIT_CLIENT_ID,
    }
    response = requests.post(FITBIT_TOKEN_URL, data=data, headers=headers, auth=(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET))
    return response

def refresh_fitbit_token(refresh_token):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": FITBIT_CLIENT_ID,
    }
    response = requests.post(FITBIT_TOKEN_URL, data=data, headers=headers, auth=(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET))
    return response

def is_fitbit_token_valid(token):
    test_url = "https://api.fitbit.com/1/user/-/profile.json"
    resp = requests.get(test_url, headers={"Authorization": f"Bearer {token}"})
    return resp.status_code == 200

def fetch_fitbit_weight(access_token):
    start_date = datetime(2025, 5, 12)
    end_date = datetime.today()
    all_data = []
    headers = {"Authorization": f"Bearer {access_token}"}
    while start_date <= end_date:
        chunk_end = min(start_date + timedelta(days=30), end_date)
        url = f"https://api.fitbit.com/1/user/-/body/log/weight/date/{start_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}.json"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            st.error(f"Fitbit fetch error: {resp.status_code}")
            break
        data_chunk = resp.json()
        if "weight" in data_chunk:
            all_data.extend(data_chunk["weight"])
        start_date = chunk_end + timedelta(days=1)
    return all_data

# ---- Google Fit OAuth / API ----
def get_google_token_from_code(code):
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    resp = requests.post("https://oauth2.googleapis.com/token", data=data)
    return resp.json()

def refresh_google_token(refresh_token):
    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    resp = requests.post("https://oauth2.googleapis.com/token", data=data)
    return resp.json()

def fetch_google_steps(access_token, days=7):
    # Fetch aggregated step count from Google Fit
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    url = "https://fitness.googleapis.com/fitness/v1/users/me/dataset:aggregate"
    body = {
        "aggregateBy": [{"dataTypeName": "com.google.step_count.delta"}],
        "bucketByTime": {"durationMillis": 24*60*60*1000},
        "startTimeMillis": start_time,
        "endTimeMillis": end_time
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(url, headers=headers, json=body)
    if resp.status_code != 200:
        return []
    data = resp.json()
    steps = []
    for bucket in data.get("bucket", []):
        date = datetime.fromtimestamp(int(bucket["startTimeMillis"])/1000).date()
        total = sum([int(dp["value"][0]["intVal"]) for dp in bucket["dataset"][0].get("point", [])])
        steps.append({"date": date, "steps": total})
    return steps

# ---- Streamlit App ----
st.set_page_config(page_title="Fatboard Tracker", layout="centered")
st.title("Fat Packer Tracker")

query_params = st.experimental_get_query_params()
fitbit_code = query_params.get("fitbit_code", [None])[0]
google_code = query_params.get("google_code", [None])[0]

# ---- Fitbit Tokens ----
fitbit_tokens = load_tokens(FITBIT_TOKENS_DOC)
fitbit_access = fitbit_tokens.get("access_token")
fitbit_refresh = fitbit_tokens.get("refresh_token")

if fitbit_code and not fitbit_access:
    resp = get_fitbit_token_from_code(fitbit_code)
    if resp.status_code == 200:
        fitbit_tokens = resp.json()
        save_tokens(FITBIT_TOKENS_DOC, fitbit_tokens)
        fitbit_access = fitbit_tokens["access_token"]
    st.experimental_set_query_params()

if fitbit_access and not is_fitbit_token_valid(fitbit_access):
    resp = refresh_fitbit_token(fitbit_refresh)
    if resp.status_code == 200:
        fitbit_tokens = resp.json()
        save_tokens(FITBIT_TOKENS_DOC, fitbit_tokens)
        fitbit_access = fitbit_tokens["access_token"]

if not fitbit_access:
    st.markdown(f"[Connect your Fitbit account]({FITBIT_AUTH_URL})")
    st.stop()

# ---- Google Fit Tokens ----
google_tokens = load_tokens(GOOGLE_TOKENS_DOC)
google_access = google_tokens.get("access_token")
google_refresh = google_tokens.get("refresh_token")

if google_code and not google_access:
    google_tokens = get_google_token_from_code(google_code)
    save_tokens(GOOGLE_TOKENS_DOC, google_tokens)
    google_access = google_tokens["access_token"]
    st.experimental_set_query_params()

if google_access is None:
    st.markdown(f"[Connect Google Fit]({GOOGLE_AUTH_URL})")
    google_steps = []
else:
    try:
        google_steps = fetch_google_steps(google_access)
    except:
        google_steps = []

# ---- Fetch Fitbit Weight ----
fitbit_data = fetch_fitbit_weight(fitbit_access)
if not fitbit_data:
    st.error("No Fitbit weight data found.")
    st.stop()

df = pd.DataFrame(fitbit_data)
df["dateTime"] = pd.to_datetime(df["date"])
df = df.sort_values("dateTime")
df["weight_lbs"] = df["weight"].apply(kg_to_lbs)
df["weight_stlbs"] = df["weight_lbs"].apply(lbs_to_st_lbs)

journey_start = datetime(2025, 5, 12)
df_after_start = df[df["dateTime"] >= journey_start]
start_weight = df_after_start.iloc[0]["weight_lbs"]
current_weight = df.iloc[-1]["weight_lbs"]
loss = start_weight - current_weight
days = (datetime.today() - journey_start).days
avg_per_day = loss / days if days>0 else 0
goal_stone = 15
goal_lbs = st_to_lbs(goal_stone)
goal_date = datetime.today() + timedelta(days=(current_weight-goal_lbs)/avg_per_day) if current_weight>goal_lbs else None
countdown_days = (goal_date - datetime.today()).days if goal_date else None

# ---- Metrics Tiles CSS ----
tile_style = """
<style>
.metric-box {
    background-color: #3C3C3C;
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin-bottom: 10px;
}
.metric-label {
    font-size: 16px;
    font-weight: normal;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: bold;
}
</style>
"""
st.markdown(tile_style, unsafe_allow_html=True)

# ---- Top Row Metrics ----
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Total Weight Lost</div><div class="metric-value">{lbs_to_st_lbs(loss)}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Days Not Being Fat</div><div class="metric-value">{days} days</div></div>', unsafe_allow_html=True)
with col3:
    if google_steps:
        today_steps = google_steps[-1]["steps"]
    else:
        today_steps = 0
    st.markdown(f'<div class="metric-box"><div class="metric-label">Today\'s Steps</div><div class="metric-value">{today_steps}</div></div>', unsafe_allow_html=True)

# ---- Fitbit Daily Graph ----
y_min = int(df["weight_lbs"].min())-1
y_max = int(df["weight_lbs"].max())+1
y_ticks = list(range(y_min, y_max+1))
y_tick_text = [lbs_to_st_lbs(i) for i in y_ticks]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["dateTime"], y=df["weight_lbs"], mode="lines+markers", name="Actual Weight", customdata=df["weight_stlbs"], hovertemplate="Date: %{x|%d-%m-%Y}<br>Weight: %{customdata}<extra></extra>", line=dict(color="cyan"), marker=dict(color="cyan")))
fig.add_trace(go.Scatter(x=[journey_start, datetime(2026,1,1)], y=[start_weight, goal_lbs], mode="lines", name="Goal Trendline", line=dict(color="red", dash="dash")))
fig.update_layout(plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white", margin=dict(l=40,r=40,t=40,b=40), xaxis_title="Date", yaxis=dict(range=[y_min,y_max], tickvals=y_ticks, ticktext=y_tick_text, gridcolor="#555"))
st.markdown('<div class="section-title">📅 Daily Weight Over Time</div>', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)

# ---- Google Fit Steps Graph ----
if google_steps:
    df_steps = pd.DataFrame(google_steps)
    fig_steps = go.Figure()
    fig_steps.add_trace(go.Bar(x=df_steps["date"], y=df_steps["steps"], marker_color="cyan"))
    fig_steps.update_layout(plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white", margin=dict(l=40,r=40,t=40,b=40), yaxis_title="Steps", xaxis_title="Date")
    st.markdown('<div class="section-title">📅 Steps Last 7 Days</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_steps, use_container_width=True)

# ---- Weekly Weight Loss ----
df['week'] = df['dateTime'].dt.to_period('W').apply(lambda r: r.start_time)
weekly = df.groupby('week')['weight_lbs'].mean().reset_index()
weekly['weight_loss'] = weekly['weight_lbs'].shift(1) - weekly['weight_lbs']
weekly = weekly.dropna(subset=['weight_loss'])
weekly['text'] = weekly['weight_loss'].apply(lambda x: f"-{abs(x):.1f}" if x>0 else (f"+{abs(x):.1f}" if x<0 else "0.0"))
weekly['color'] = weekly['weight_loss'].apply(lambda v: 'green' if v>0 else ('red' if v<0 else 'gray'))
figw = go.Figure()
for color, dfc in weekly.groupby('color'):
    figw.add_trace(go.Bar(x=dfc['week'], y=dfc['weight_loss'], marker_color=color, text=dfc['text'], textposition='outside'))
figw.update_layout(plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white", margin=dict(l=40,r=40,t=40,b=40), xaxis_title="Week Starting", xaxis=dict(tickformat="%d-%m-%Y", gridcolor="#555"), yaxis=dict(title="Weight Change (lbs)", tickvals=y_ticks, ticktext=y_tick_text, zeroline=True, zerolinecolor="white", zerolinewidth=2, gridcolor="#555"))
st.markdown('<div class="section-title">📊 Weekly Weight Loss</div>', unsafe_allow_html=True)
st.plotly_chart(figw, use_container_width=True)

# ---- Raw Weight Log ----
df_sorted = df.sort_values(by='dateTime', ascending=False)
df_sorted['date'] = df_sorted['dateTime'].dt.date
st.markdown('<div class="section-title">📋 Raw Weight Log</div>', unsafe_allow_html=True)
st.dataframe(df_sorted[['date','weight_stlbs']].rename(columns={'date':'Date','weight_stlbs':'Weight'}).reset_index(drop=True))
