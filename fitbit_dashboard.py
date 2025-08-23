import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import os

import firebase_admin
from firebase_admin import credentials, firestore

# ------------------ Firebase Setup ------------------
if not firebase_admin._apps:
    firebase_cred_dict = dict(st.secrets["firebase"])
    firebase_cred_dict["private_key"] = firebase_cred_dict["private_key"].replace("\\n", "\n").strip()
    cred = credentials.Certificate(firebase_cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ------------------ Fitbit Setup ------------------
FITBIT_CLIENT_ID = st.secrets["FITBIT_CLIENT_ID"]
FITBIT_CLIENT_SECRET = st.secrets["FITBIT_CLIENT_SECRET"]
REDIRECT_URI = "https://fatboard.streamlit.app"
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
AUTH_URL = (
    f"https://www.fitbit.com/oauth2/authorize?"
    f"response_type=code&client_id={FITBIT_CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    f"&scope=weight&expires_in=604800&prompt=login"
)
TOKENS_DOC = "fitbit/tokens"

def save_tokens(tokens):
    db.document(TOKENS_DOC).set(tokens)

def load_tokens():
    doc = db.document(TOKENS_DOC).get()
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

def get_token_from_code(code):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": FITBIT_CLIENT_ID,
    }
    response = requests.post(
        TOKEN_URL,
        data=data,
        headers=headers,
        auth=(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET),
    )
    return response

def refresh_access_token(refresh_token):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": FITBIT_CLIENT_ID,
    }
    response = requests.post(
        TOKEN_URL,
        data=data,
        headers=headers,
        auth=(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET),
    )
    return response

def fetch_weight_data(access_token):
    start_date = datetime(2025, 5, 12)
    end_date = datetime.today()
    all_data = []
    headers = {"Authorization": f"Bearer {access_token}"}
    while start_date <= end_date:
        chunk_end = min(start_date + timedelta(days=30), end_date)
        url = (
            f"https://api.fitbit.com/1/user/-/body/log/weight/date/"
            f"{start_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}.json"
        )
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.error(f"Error fetching data: {response.status_code}")
            st.json(response.json())
            break
        data_chunk = response.json()
        if "weight" in data_chunk:
            all_data.extend(data_chunk["weight"])
        start_date = chunk_end + timedelta(days=1)
    return {"weight": all_data}

# ------------------ Google Fit Setup ------------------
GOOGLE_FIT_CLIENT_ID = st.secrets["google_fit"]["client_id"]
GOOGLE_FIT_CLIENT_SECRET = st.secrets["google_fit"]["client_secret"]
GOOGLE_FIT_REDIRECT_URI = st.secrets["google_fit"]["redirect_uri"]
GOOGLE_FIT_SCOPES = "https://www.googleapis.com/auth/fitness.activity.read"

# Token storage for Google Fit
GOOGLE_FIT_TOKENS_DOC = "google_fit/tokens"

def save_google_tokens(tokens):
    db.document(GOOGLE_FIT_TOKENS_DOC).set(tokens)

def load_google_tokens():
    doc = db.document(GOOGLE_FIT_TOKENS_DOC).get()
    if doc.exists:
        return doc.to_dict()
    return {}

def refresh_google_token(refresh_token):
    url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": GOOGLE_FIT_CLIENT_ID,
        "client_secret": GOOGLE_FIT_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    resp = requests.post(url, data=data)
    return resp.json()

def fetch_google_fit_steps(access_token, days=7):
    headers = {"Authorization": f"Bearer {access_token}"}
    end_time = int(datetime.utcnow().timestamp() * 1e9)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1e9)
    body = {
        "aggregateBy": [{"dataTypeName": "com.google.step_count.delta"}],
        "bucketByTime": {"durationMillis": 86400000},
        "startTimeMillis": start_time // 1_000_000,
        "endTimeMillis": end_time // 1_000_000
    }
    url = "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate"
    resp = requests.post(url, headers=headers, json=body)
    data = resp.json()
    steps = []
    if "bucket" in data:
        for b in data["bucket"]:
            date = datetime.fromtimestamp(int(b["startTimeMillis"])/1000).date()
            step_count = sum([pt["value"][0]["intVal"] for ds in b["dataset"] for pt in ds.get("point", [])])
            steps.append({"date": date, "steps": step_count})
    return pd.DataFrame(steps)

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="Fatboard Tracker", layout="centered")
st.title("Fat Packer Tracker")

# ------------------ Fitbit Token Handling ------------------
code = st.query_params.get("code", [None])[0]
tokens = load_tokens()
access_token = tokens.get("access_token")
refresh_token_val = tokens.get("refresh_token")

def is_token_valid(token):
    test_url = "https://api.fitbit.com/1/user/-/profile.json"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(test_url, headers=headers)
    return resp.status_code == 200

if access_token and not is_token_valid(access_token):
    response = refresh_access_token(refresh_token_val)
    if response.status_code == 200:
        tokens = response.json()
        save_tokens(tokens)
        access_token = tokens["access_token"]
        refresh_token_val = tokens.get("refresh_token")
    else:
        st.error("❌ Failed to refresh token, please reconnect Fitbit.")
        st.markdown(f"[Connect your Fitbit account]({AUTH_URL})")
        st.stop()

if not access_token and not code:
    st.markdown(f"[Connect your Fitbit account]({AUTH_URL})")
    st.stop()

if code and not access_token:
    response = get_token_from_code(code)
    if response.status_code != 200:
        st.error("❌ Failed to authenticate with Fitbit.")
        st.markdown(f"[Click here to reconnect your Fitbit account]({AUTH_URL})")
        st.stop()
    tokens = response.json()
    save_tokens(tokens)
    access_token = tokens["access_token"]
    refresh_token_val = tokens.get("refresh_token")
    st.experimental_set_query_params()

elif refresh_token_val and not access_token:
    response = refresh_access_token(refresh_token_val)
    if response.status_code == 200:
        tokens = response.json()
        save_tokens(tokens)
        access_token = tokens["access_token"]
        refresh_token_val = tokens.get("refresh_token")

# ------------------ Fetch Fitbit Weight Data ------------------
data = fetch_weight_data(access_token)
if "weight" not in data or len(data["weight"]) == 0:
    st.error("No weight data found.")
    st.stop()

weights = data["weight"]
df = pd.DataFrame(weights)
df["dateTime"] = pd.to_datetime(df["date"])
df = df.sort_values("dateTime")
df["date"] = df["dateTime"].dt.strftime("%d-%m-%Y")
df["weight_lbs"] = df["weight"].apply(kg_to_lbs)
df["weight_stlbs"] = df["weight_lbs"].apply(lbs_to_st_lbs)

journey_start_date = datetime(2025, 5, 12)
df_after_start = df[df["dateTime"] >= journey_start_date]
start_weight = df_after_start.iloc[0]["weight_lbs"]
current_weight = df.iloc[-1]["weight_lbs"]
latest_date = df.iloc[-1]["dateTime"].strftime("%d-%m-%Y")
days = (datetime.today() - journey_start_date).days
loss = start_weight - current_weight
avg_per_day = loss / days if days > 0 else 0

goal_stone = 15
goal = st_to_lbs(goal_stone)
if current_weight > goal and avg_per_day > 0:
    days_left = (current_weight - goal) / avg_per_day
    goal_date = datetime.today() + timedelta(days=days_left)
    countdown_days = (goal_date - datetime.today()).days
else:
    goal_date = None
    countdown_days = None

# ------------------ Google Fit Steps ------------------
google_tokens = load_google_tokens()
google_access_token = google_tokens.get("access_token")
google_refresh_token = google_tokens.get("refresh_token")
today_steps = 0
df_steps = pd.DataFrame()

if google_access_token:
    try:
        df_steps = fetch_google_fit_steps(google_access_token, days=7)
        if not df_steps.empty:
            today_steps = int(df_steps[df_steps["date"]==datetime.today().date()]["steps"].sum())
    except:
        today_steps = 0

# ------------------ Metrics Display ------------------
st.subheader("📌 Latest Weigh-In")
st.metric("Latest Weight", lbs_to_st_lbs(current_weight), delta=f"{current_weight - start_weight:.1f} lbs")

st.subheader("👣 Today's Steps")
st.metric("Steps Today", f"{today_steps:,}")

# ------------------ CSS for Metric Boxes ------------------
progress_style = """
<style>
.metric-box { background-color: #3C3C3C; padding: 20px; border-radius: 10px; color: white; text-align: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-bottom: 10px; }
.metric-label { font-size: 16px; margin-bottom: 4px; font-weight: normal; }
.metric-value { font-size: 22px; font-weight: bold; }
.section-title { font-size: 20px; font-weight: bold; color: white; margin-top: 20px; margin-bottom: 10px; }
</style>
"""
st.markdown(progress_style, unsafe_allow_html=True)

# ------------------ Weight & Steps Graphs ------------------
# (Keep your existing graphs here)
# Add a new steps last 7 days graph if df_steps is not empty
if not df_steps.empty:
    fig_steps = go.Figure()
    fig_steps.add_trace(go.Scatter(
        x=df_steps["date"],
        y=df_steps["steps"],
        mode="lines+markers",
        name="Steps",
        line=dict(color="orange"),
        marker=dict(color="orange"),
        hovertemplate="Date: %{x}<br>Steps: %{y}<extra></extra>"
    ))
    fig_steps.update_layout(
        plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white",
        margin=dict(l=40,r=40,t=40,b=40),
        xaxis_title="Date", yaxis_title="Steps",
        xaxis=dict(gridcolor="#555"), yaxis=dict(gridcolor="#555")
    )
    st.markdown('<div class="section-title">📅 Steps Last 7 Days</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_steps, use_container_width=True)

# ------------------ Raw Weight Log ------------------
df = df.dropna(subset=['dateTime'])
df['original_index'] = df.index
df_sorted = df.sort_values(by=['dateTime','original_index'], ascending=[False,False])
df_sorted['date'] = df_sorted['dateTime'].dt.date
st.markdown('<div class="section-title">📋 Raw Weight Log</div>', unsafe_allow_html=True)
st.dataframe(df_sorted[['date','weight_stlbs']].rename(columns={'date':'Date','weight_stlbs':'Weight'}).reset_index(drop=True))
