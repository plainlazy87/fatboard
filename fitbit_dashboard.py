import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import os

import firebase_admin
from firebase_admin import credentials, firestore

# ------------------- Initialize Firebase -------------------
if not firebase_admin._apps:
    firebase_cred_dict = dict(st.secrets["firebase"])
    firebase_cred_dict["private_key"] = firebase_cred_dict["private_key"].replace("\\n", "\n").strip()
    cred = credentials.Certificate(firebase_cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ------------------- Fitbit OAuth -------------------
FITBIT_CLIENT_ID = st.secrets["FITBIT_CLIENT_ID"]
FITBIT_CLIENT_SECRET = st.secrets["FITBIT_CLIENT_SECRET"]
FITBIT_REDIRECT_URI = "https://fatboard.streamlit.app"
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
FITBIT_AUTH_URL = (
    f"https://www.fitbit.com/oauth2/authorize?"
    f"response_type=code&client_id={FITBIT_CLIENT_ID}&redirect_uri={FITBIT_REDIRECT_URI}"
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
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": FITBIT_REDIRECT_URI, "client_id": FITBIT_CLIENT_ID}
    return requests.post(FITBIT_TOKEN_URL, data=data, headers=headers, auth=(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET))

def refresh_access_token(refresh_token):
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": FITBIT_CLIENT_ID}
    return requests.post(FITBIT_TOKEN_URL, data=data, headers=headers, auth=(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET))

def fetch_weight_data(access_token):
    start_date = datetime(2025, 5, 12)
    end_date = datetime.today()
    all_data = []
    headers = {"Authorization": f"Bearer {access_token}"}
    while start_date <= end_date:
        chunk_end = min(start_date + timedelta(days=30), end_date)
        url = f"https://api.fitbit.com/1/user/-/body/log/weight/date/{start_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}.json"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.error(f"Error fetching Fitbit data: {response.status_code}")
            st.json(response.json())
            break
        data_chunk = response.json()
        if "weight" in data_chunk:
            all_data.extend(data_chunk["weight"])
        start_date = chunk_end + timedelta(days=1)
    return {"weight": all_data}

def is_token_valid(token):
    test_url = "https://api.fitbit.com/1/user/-/profile.json"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(test_url, headers=headers)
    return resp.status_code == 200

# ------------------- Google Fit OAuth -------------------
GOOGLE_FIT_CLIENT_ID = st.secrets["google_fit"]["client_id"]
GOOGLE_FIT_CLIENT_SECRET = st.secrets["google_fit"]["client_secret"]
GOOGLE_FIT_REDIRECT_URI = st.secrets["google_fit"]["redirect_uri"]
GOOGLE_FIT_SCOPES = "https://www.googleapis.com/auth/fitness.activity.read"

GOOGLE_TOKENS_DOC = "google_fit/tokens"

def save_google_tokens(tokens):
    db.document(GOOGLE_TOKENS_DOC).set(tokens)

def load_google_tokens():
    doc = db.document(GOOGLE_TOKENS_DOC).get()
    if doc.exists:
        return doc.to_dict()
    return {}

def refresh_google_token(refresh_token):
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": GOOGLE_FIT_CLIENT_ID,
        "client_secret": GOOGLE_FIT_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    })
    if resp.status_code == 200:
        tokens = resp.json()
        save_google_tokens(tokens)
        return tokens.get("access_token")
    return None

def fetch_google_steps(access_token):
    end_time = int(datetime.now().timestamp() * 1000000000)  # in ns
    start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000000000)
    url = f"https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    body = {
        "aggregateBy":[{"dataTypeName":"com.google.step_count.delta"}],
        "bucketByTime":{"durationMillis":86400000},
        "startTimeMillis": start_time // 1000000,
        "endTimeMillis": end_time // 1000000
    }
    resp = requests.post(url, headers=headers, json=body)
    steps_list = []
    if resp.status_code == 200:
        data = resp.json()
        for bucket in data.get("bucket", []):
            steps = sum(int(ds['intVal']) for ds in bucket['dataset'][0]['point'][0].get('value', [{'intVal':0}]))
            date = datetime.fromtimestamp(int(bucket['startTimeMillis'])/1000).date()
            steps_list.append({"date": date, "steps": steps})
    return pd.DataFrame(steps_list)

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="Fatboard Tracker", layout="centered")
st.title("Fat Packer Tracker")

# --- Fitbit Authentication ---
code = st.query_params.get("code", [None])[0]
tokens = load_tokens()
access_token = tokens.get("access_token")
refresh_token_val = tokens.get("refresh_token")

if access_token and not is_token_valid(access_token):
    response = refresh_access_token(refresh_token_val)
    if response.status_code == 200:
        tokens = response.json()
        save_tokens(tokens)
        access_token = tokens["access_token"]
        refresh_token_val = tokens.get("refresh_token")
    else:
        st.error("❌ Fitbit token refresh failed, reconnect.")
        st.markdown(f"[Connect Fitbit]({FITBIT_AUTH_URL})")
        st.stop()

if not access_token and not code:
    st.markdown(f"[Connect Fitbit]({FITBIT_AUTH_URL})")
    st.stop()

if code and not access_token:
    response = get_token_from_code(code)
    if response.status_code == 200:
        tokens = response.json()
        save_tokens(tokens)
        access_token = tokens["access_token"]
        refresh_token_val = tokens.get("refresh_token")
        st.experimental_set_query_params()
    else:
        st.error("❌ Fitbit authentication failed.")
        st.markdown(f"[Reconnect Fitbit]({FITBIT_AUTH_URL})")
        st.stop()

# Fetch Fitbit weight data
data = fetch_weight_data(access_token)
if "weight" not in data or len(data["weight"])==0:
    st.error("No Fitbit weight data found.")
    st.stop()

weights = data["weight"]
df = pd.DataFrame(weights)
df["dateTime"] = pd.to_datetime(df["date"])
df = df.sort_values("dateTime")
df["date"] = df["dateTime"].dt.strftime("%d-%m-%Y")
df["weight_lbs"] = df["weight"].apply(kg_to_lbs)
df["weight_stlbs"] = df["weight_lbs"].apply(lbs_to_st_lbs)

# --- Google Fit Authentication ---
google_code = st.query_params.get("google_code", [None])[0]
google_tokens = load_google_tokens()
google_access_token = google_tokens.get("access_token")
google_refresh_token = google_tokens.get("refresh_token")

if not google_access_token and not google_code:
    oauth_url = f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id={GOOGLE_FIT_CLIENT_ID}&redirect_uri={GOOGLE_FIT_REDIRECT_URI}&scope={GOOGLE_FIT_SCOPES}&access_type=offline&prompt=consent"
    st.markdown(f"[Connect Google Fit]({oauth_url})")
    df_steps = pd.DataFrame()
else:
    if google_code and not google_access_token:
        resp = requests.post("https://oauth2.googleapis.com/token", data={
            "code": google_code,
            "client_id": GOOGLE_FIT_CLIENT_ID,
            "client_secret": GOOGLE_FIT_CLIENT_SECRET,
            "redirect_uri": GOOGLE_FIT_REDIRECT_URI,
            "grant_type": "authorization_code"
        })
        if resp.status_code == 200:
            google_tokens = resp.json()
            save_google_tokens(google_tokens)
            google_access_token = google_tokens.get("access_token")
            google_refresh_token = google_tokens.get("refresh_token")
            st.experimental_set_query_params()
        else:
            st.error("Google Fit authentication failed.")
            df_steps = pd.DataFrame()
    if google_access_token:
        df_steps = fetch_google_steps(google_access_token)

# ------------------- Dashboard Metrics -------------------
journey_start_date = datetime(2025,5,12)
df_after_start = df[df["dateTime"]>=journey_start_date]
start_weight = df_after_start.iloc[0]["weight_lbs"]
current_weight = df.iloc[-1]["weight_lbs"]
latest_date = df.iloc[-1]["dateTime"].strftime("%d-%m-%Y")
days = (datetime.today() - journey_start_date).days
loss = start_weight - current_weight
avg_per_day = loss/days if days>0 else 0
goal_stone = 15
goal = st_to_lbs(goal_stone)
if current_weight>goal and avg_per_day>0:
    days_left = (current_weight - goal)/avg_per_day
    goal_date = datetime.today()+timedelta(days=days_left)
    countdown_days = (goal_date - datetime.today()).days
else:
    goal_date=None
    countdown_days=None

# ---- Tiles ----
st.subheader("📌 Latest Weigh-In")
st.metric("Latest Weight", lbs_to_st_lbs(current_weight), delta=f"{current_weight-start_weight:.1f} lbs")
if not df_steps.empty:
    st.metric("Today's Steps", df_steps.iloc[-1]["steps"])

progress_style = """
<style>
.metric-box { background-color: #3C3C3C; padding:20px; border-radius:10px; color:white; text-align:center; font-family:'Segoe UI'; margin-bottom:10px; }
.metric-label{ font-size:16px; margin-bottom:4px; font-weight:normal;}
.metric-value{ font-size:22px; font-weight:bold;}
</style>
"""
st.markdown(progress_style, unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-box'><div class='metric-label'>Total Weight Lost</div><div class='metric-value'>{lbs_to_st_lbs(loss)}</div></div>",unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><div class='metric-label'>Days Not Being Fat</div><div class='metric-value'>{days} days</div></div>",unsafe_allow_html=True)
# col3 left blank

col4,col5,col6 = st.columns(3)
with col4:
    st.markdown(f"<div class='metric-box'><div class='metric-label'>Goal Weight</div><div class='metric-value'>{goal_stone}st ({goal} lbs)</div></div>",unsafe_allow_html=True)
with col5:
    if goal_date and countdown_days is not None:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Estimated Goal Date</div><div class='metric-value'>{goal_date.strftime('%d-%m-%Y')}</div></div>",unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>🎯 Goal reached!</div><div class='metric-value'></div></div>",unsafe_allow_html=True)
with col6:
    if goal_date and countdown_days is not None:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Days Until Goal</div><div class='metric-value'>{countdown_days} days</div></div>",unsafe_allow_html=True)

# ------------------- Graphs -------------------
def lbs_to_stlbs_ticks(lbs):
    total_lbs = round(lbs)
    stn = total_lbs//14
    lbs_left = total_lbs%14
    return f"{stn}st {lbs_left}lbs"

y_min = int(df["weight_lbs"].min())-1
y_max = int(df["weight_lbs"].max())+1
y_ticks = list(range(y_min,y_max+1))
y_tick_text = [lbs_to_stlbs_ticks(i) for i in y_ticks]

# Last 7 Days Weight
df_7day = df[df["dateTime"]>=(datetime.today()-timedelta(days=7))]
fig_7day = go.Figure()
fig_7day.add_trace(go.Scatter(x=df_7day["dateTime"], y=df_7day["weight_lbs"], mode="lines+markers", name="Last 7 Days", customdata=df_7day["weight_stlbs"], hovertemplate="Date: %{x|%d-%m-%Y}<br>Weight: %{customdata}<extra></extra>", line=dict(color="cyan"), marker=dict(color="cyan")))
fig_7day.add_trace(go.Scatter(x=[journey_start_date, datetime(2026,1,1)], y=[start_weight,goal], mode="lines", name="Goal Trendline", line=dict(color="red", dash="dash")))
fig_7day.update_layout(plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white", margin=dict(l=40,r=40,t=40,b=40), xaxis_title="Date", xaxis=dict(range=[df_7day["dateTime"].min()-timedelta(days=0.5), df_7day["dateTime"].max()+timedelta(days=0.5)], tickformat="%d-%m-%Y", gridcolor="#555"), yaxis=dict(range=[int(df_7day["weight_lbs"].min())-1,int(df_7day["weight_lbs"].max())+1], tickvals=list(range(int(df_7day["weight_lbs"].min())-1,int(df_7day["weight_lbs"].max())+2)), ticktext=[lbs_to_stlbs_ticks(i) for i in range(int(df_7day["weight_lbs"].min())-1,int(df_7day["weight_lbs"].max())+2)], gridcolor="#555"), legend=dict(bgcolor="#3C3C3C", bordercolor="#222", borderwidth=1, font=dict(color="white"), orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1))
st.markdown('<div class="section-title">📅 Last 7 Days Weight</div>', unsafe_allow_html=True)
st.plotly_chart(fig_7day, use_container_width=True)

# Daily Weight Over Time
fig_all = go.Figure()
fig_all.add_trace(go.Scatter(x=df["dateTime"], y=df["weight_lbs"], mode="lines+markers", name="Actual Weight", customdata=df["weight_stlbs"], hovertemplate="Date: %{x|%d-%m-%Y}<br>Weight: %{customdata}<extra></extra>", line=dict(color="cyan"), marker=dict(color="cyan")))
fig_all.add_trace(go.Scatter(x=[journey_start_date, datetime(2026,1,1)], y=[start_weight,goal], mode="lines", name="Goal Trendline", line=dict(color="red", dash="dash")))
fig_all.update_layout(plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white", margin=dict(l=40,r=40,t=40,b=40), xaxis_title="Date", yaxis_title="Weight", xaxis=dict(range=[df["dateTime"].min()-timedelta(days=0.5), df["dateTime"].max()+timedelta(days=0.5)], tickformat="%d-%m-%Y"), yaxis=dict(range=[y_min,y_max], tickvals=y_ticks, ticktext=y_tick_text, gridcolor="#555"), legend=dict(bgcolor="#3C3C3C", bordercolor="#222", borderwidth=1, font=dict(color="white"), orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1))
st.markdown('<div class="section-title">📅 Daily Weight Over Time</div>', unsafe_allow_html=True)
st.plotly_chart(fig_all, use_container_width=True)

# Weekly Weight Loss
df['week'] = df['dateTime'].dt.to_period('W').apply(lambda r: r.start_time)
weekly = df.groupby('week')['weight_lbs'].mean().reset_index()
weekly['weight_loss'] = weekly['weight_lbs'].shift(1)-weekly['weight_lbs']
weekly = weekly.dropna(subset=['weight_loss'])
weekly['text'] = weekly['weight_loss'].apply(lambda x: f"-{x:.1f}" if x>0 else (f"+{abs(x):.1f}" if x<0 else "0.0"))
figw = go.Figure()
if not weekly[weekly['weight_loss']>0].empty:
    figw.add_trace(go.Bar(x=weekly[weekly['weight_loss']>0]['week'], y=weekly[weekly['weight_loss']>0]['weight_loss'], marker_color='green', text=weekly[weekly['weight_loss']>0]['text'], textposition='outside', name="Weight Lost"))
if not weekly[weekly['weight_loss']<0].empty:
    figw.add_trace(go.Bar(x=weekly[weekly['weight_loss']<0]['week'], y=weekly[weekly['weight_loss']<0]['weight_loss'], marker_color='red', text=weekly[weekly['weight_loss']<0]['text'], textposition='outside', name="Weight Gained"))
figw.update_layout(plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white", margin=dict(l=40,r=40,t=40,b=40), xaxis_title="Week Starting", xaxis=dict(tickformat="%d-%m-%Y", gridcolor="#555"), yaxis=dict(title="Weight Change (lbs)", tickvals=y_ticks, ticktext=y_tick_text, zeroline=True, zerolinecolor="white", zerolinewidth=2, gridcolor="#555"), legend=dict(bgcolor="#3C3C3C", bordercolor="#222", borderwidth=1, font=dict(color="white"), orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1))
st.markdown('<div class="section-title">📊 Weekly Weight Loss</div>', unsafe_allow_html=True)
st.plotly_chart(figw, use_container_width=True)

# Google Fit Steps Graph
if not df_steps.empty:
    fig_steps = go.Figure()
    fig_steps.add_trace(go.Bar(x=df_steps["date"], y=df_steps["steps"], marker_color='cyan'))
    fig_steps.update_layout(plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white", margin=dict(l=40,r=40,t=40,b=40), yaxis_title="Steps", xaxis_title="Date")
    st.markdown('<div class="section-title">📊 Steps Last 7 Days</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_steps, use_container_width=True)

# Raw Weight Log
df_sorted = df.sort_values(by='dateTime', ascending=False)
df_sorted['date'] = df_sorted['dateTime'].dt.date
st.markdown('<div class="section-title">📋 Raw Weight Log</div>', unsafe_allow_html=True)
st.dataframe(df_sorted[['date','weight_stlbs']].rename(columns={'date':'Date','weight_stlbs':'Weight'}).reset_index(drop=True))
