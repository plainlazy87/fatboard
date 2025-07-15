import streamlit as st
import requests
import json
import os
import time
from datetime import datetime, timedelta
from streamlit.runtime.scriptrunner import ScriptRerunException

# Custom rerun function to replace deprecated st.experimental_rerun()
def rerun():
    raise ScriptRerunException

# ---- Fitbit OAuth2 Credentials ----
CLIENT_ID = st.secrets["FITBIT_CLIENT_ID"]
CLIENT_SECRET = st.secrets["FITBIT_CLIENT_SECRET"]
REDIRECT_URI = "https://fatboard.streamlit.app"
TOKEN_URL = "https://api.fitbit.com/oauth2/token"

AUTH_URL = (
    f"https://www.fitbit.com/oauth2/authorize?"
    f"response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    f"&scope=weight&expires_in=604800&prompt=login"
)

TOKEN_FILE = "fitbit_tokens.json"

def save_tokens(tokens):
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f)

def load_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    return None

def exchange_code_for_tokens(code):
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
            "code": code,
        },
        auth=(CLIENT_ID, CLIENT_SECRET),
    )
    if response.status_code == 200:
        tokens = response.json()
        tokens["expires_at"] = int(time.time()) + tokens["expires_in"]
        save_tokens(tokens)
        return tokens
    else:
        st.error(f"Failed to exchange code: {response.text}")
        return None

def refresh_access_token(refresh_token):
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        },
        auth=(CLIENT_ID, CLIENT_SECRET),
    )
    if response.status_code == 200:
        tokens = response.json()
        tokens["expires_at"] = int(time.time()) + tokens["expires_in"]
        save_tokens(tokens)
        return tokens
    else:
        st.error(f"Failed to refresh token: {response.text}")
        return None

def get_valid_access_token():
    tokens = load_tokens()
    if not tokens:
        return None

    # Refresh if expired or about to expire in 60 seconds
    if int(time.time()) >= tokens.get("expires_at", 0) - 60:
        tokens = refresh_access_token(tokens["refresh_token"])
        if not tokens:
            return None

    return tokens["access_token"]

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

def main():
    st.title("FatBoard Fitbit Weight Dashboard")

    tokens = load_tokens()

    if not tokens:
        st.write("Please authorize the app by following these steps:")
        st.markdown(f"1. Go to this URL and log in:\n\n`{AUTH_URL}`")
        auth_code = st.text_input("Enter the authorization code here:", type="password")
        if auth_code:
            tokens = exchange_code_for_tokens(auth_code)
            if tokens:
                st.success("Authorization successful! Tokens saved. Reloading...")
                rerun()  # <-- use custom rerun()
    else:
        access_token = get_valid_access_token()
        if access_token:
            data = fetch_weight_data(access_token)
            st.write("Fetched weight data:")
            st.json(data)
        else:
            st.error("Failed to get a valid access token. Please delete the token file and reauthorize.")
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
                rerun()

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------------
# ---- Streamlit App ----
st.set_page_config(page_title="Fitbit Weight Loss Dashboard", layout="centered")

# Custom CSS (unchanged) ...
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2E2E2E;
        color: white;
    }
    .section-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .plotly-graph-div {
        background-color: #3C3C3C !important;
        border-radius: 8px;
        padding: 10px;
    }
    .legend-bg {
        background-color: #3C3C3C;
        padding: 10px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        width: 160px;
        margin-left: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-container {
        background-color: #3C3C3C;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📉 Leon's Weight Loss Dashboard")

# === IMPORTANT FIX HERE ===
# Use ONLY experimental_get_query_params for reading query params
query_params = st.query_params
code = query_params.get("code", [None])[0]

# Load tokens from file/session
tokens = load_tokens()
access_token = tokens.get("access_token") if tokens else None

# If no access token and no code, show login link and stop
if not access_token and not code:
    st.markdown(f"[🔒 Connect your Fitbit account]({AUTH_URL})")
    st.stop()

# If code is present (first auth), exchange for tokens and save
if code and not access_token:
    tokens = get_token_from_code(code)
    if "access_token" not in tokens:
        st.error("❌ Failed to authenticate with Fitbit.")
        st.json(tokens)
        st.stop()
    save_tokens(tokens)
    access_token = tokens["access_token"]

# If tokens have refresh token, try to refresh
if tokens and "refresh_token" in tokens:
    tokens = refresh_token(tokens["refresh_token"])
    if "access_token" in tokens:
        save_tokens(tokens)
        access_token = tokens["access_token"]

# Fetch data
data = fetch_weight_data(access_token)

if "weight" not in data or len(data["weight"]) == 0:
    st.error("No weight data found. Have you logged your weight recently in the Fitbit app?")
    st.json(data)
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

if df_after_start.empty:
    st.error("No weight data found on or after your journey start date (12th May 2025).")
    st.stop()

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

# ---- Metrics Display ----
st.subheader("📌 Latest Weigh-In")
st.metric("Latest Weight", lbs_to_st_lbs(current_weight), delta=f"{current_weight - start_weight:.1f} lbs")

progress_style = """
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
    margin-bottom: 4px;
    font-weight: normal;  /* Title line is now normal weight */
}
.metric-value {
    font-size: 22px;
    font-weight: bold;    /* Value remains bold */
}
</style>
"""
st.markdown(progress_style, unsafe_allow_html=True)

# (The rest of your UI code remains unchanged, just remove all st.query_params references!)

# --- Replace any remaining st.query_params.get(...) calls with query_params.get(...)

# You can keep the rest of your script as is, just be sure not to mix st.query_params with experimental_get_query_params

# For brevity, I’m not repeating the full UI code here as it’s unchanged.



# Row 1 (3 columns): show 2 boxes, skip 3rd
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Total Weight Lost</div>
        <div class="metric-value">{lbs_to_st_lbs(loss)}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Days Not Being Fat</div>
        <div class="metric-value">{days} days</div>
    </div>
    """, unsafe_allow_html=True)
# col3 intentionally left blank (no box)

# Row 2 (3 columns): show final 3 boxes
col4, col5, col6 = st.columns(3)
with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Goal Weight</div>
        <div class="metric-value">{goal_stone}st ({goal} lbs)</div>
    </div>
    """, unsafe_allow_html=True)
with col5:
    if goal_date and countdown_days is not None:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Estimated Goal Date</div>
            <div class="metric-value">{goal_date.strftime("%d-%m-%Y")}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">🎯 Goal reached!</div>
            <div class="metric-value"></div>
        </div>
        """, unsafe_allow_html=True)
with col6:
    if goal_date and countdown_days is not None:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Days Until Goal</div>
            <div class="metric-value">{countdown_days} days</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# --- Helper function for stone + lbs ticks ---
def lbs_to_stlbs_ticks(lbs):
    total_lbs = round(lbs)
    st = total_lbs // 14
    lbs_left = total_lbs % 14
    return f"{st}st {lbs_left}lbs"

# Overall weight range ticks (for other charts)
y_min = int(df["weight_lbs"].min()) - 1
y_max = int(df["weight_lbs"].max()) + 1
y_ticks = list(range(y_min, y_max + 1))
y_tick_text = [lbs_to_stlbs_ticks(i) for i in y_ticks]

# ---- Last 7 Days Weight Graph ----
with st.container():
    st.markdown('<div class="section-title">📅 Last 7 Days Weight</div>', unsafe_allow_html=True)

    df_7day = df[df["dateTime"] >= (datetime.today() - timedelta(days=7))]

    y_min_7 = int(df_7day["weight_lbs"].min()) - 1
    y_max_7 = int(df_7day["weight_lbs"].max()) + 1

    # Use 1 lb interval for ticks
    y_ticks_7 = list(range(y_min_7, y_max_7 + 1))
    y_tick_text_7 = [lbs_to_stlbs_ticks(tick) for tick in y_ticks_7]

    fig_7day = go.Figure()
    fig_7day.add_trace(go.Scatter(
        x=df_7day["dateTime"],
        y=df_7day["weight_lbs"],
        mode="lines+markers",
        name="Last 7 Days",
        customdata=df_7day["weight_stlbs"],
        hovertemplate="Date: %{x|%d-%m-%Y}<br>Weight: %{customdata}<extra></extra>",
        line=dict(color="cyan"),
        marker=dict(color="cyan"),
    ))

    fig_7day.add_trace(go.Scatter(
        x=[journey_start_date, datetime(2026, 1, 1)],
        y=[start_weight, goal],
        mode="lines",
        name="Goal Trendline",
        line=dict(color="red", dash="dash"),
    ))

    fig_7day.update_layout(
        plot_bgcolor="#3C3C3C",
        paper_bgcolor="#3C3C3C",
        font_color="white",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Date",
        xaxis=dict(
            range=[
                df_7day["dateTime"].min() - timedelta(days=0.5),
                df_7day["dateTime"].max() + timedelta(days=0.5)
            ],
            tickformat="%d-%m-%Y",
            gridcolor="#555",
        ),
        yaxis=dict(
            range=[y_min_7, y_max_7],
            tickvals=y_ticks_7,
            ticktext=y_tick_text_7,
            gridcolor="#555",
        ),
        legend=dict(
            bgcolor="#3C3C3C",
            bordercolor="#222",
            borderwidth=1,
            font=dict(color="white"),
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_7day, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)








# ---- Daily Weight Over Time Graph ----
with st.container():
    st.markdown('<div class="section-title">📅 Daily Weight Over Time</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["dateTime"],
        y=df["weight_lbs"],
        mode="lines+markers",
        name="Actual Weight",
        customdata=df["weight_stlbs"],
        hovertemplate="Date: %{x|%d-%m-%Y}<br>Weight: %{customdata}<extra></extra>",
        line=dict(color="cyan"),
        marker=dict(color="cyan"),
    ))

    fig.add_trace(go.Scatter(
        x=[journey_start_date, datetime(2026, 1, 1)],
        y=[start_weight, goal],
        mode="lines",
        name="Goal Trendline",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(
        plot_bgcolor="#3C3C3C",
        paper_bgcolor="#3C3C3C",
        font_color="white",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Date",
        yaxis_title="Weight",
        xaxis=dict(
    range=[
        df["dateTime"].min() - timedelta(days=0.5),
        df["dateTime"].max() + timedelta(days=0.5)
    ],
                   
                   tickformat="%d-%m-%Y"),
        yaxis=dict(
            range=[y_min, y_max],
            tickvals=y_ticks,
            ticktext=y_tick_text,
            gridcolor="#555",
        ),
legend=dict(
    bgcolor="#3C3C3C",
    bordercolor="#222",
    borderwidth=1,
    font=dict(color="white"),
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1
)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

# --- Weekly Weight Loss ---
with st.container():
    st.markdown('<div class="section-title">📊 Weekly Weight Loss</div>', unsafe_allow_html=True)
    df['week'] = df['dateTime'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = df.groupby('week')['weight_lbs'].mean().reset_index()
    weekly['weight_loss'] = weekly['weight_lbs'].shift(1) - weekly['weight_lbs']
    weekly = weekly.dropna(subset=['weight_loss'])

    def format_label(x):
        if x > 0:
            return f"-{abs(x):.1f}"
        elif x < 0:
            return f"+{abs(x):.1f}"
        else:
            return "0.0"

    weekly['text'] = weekly['weight_loss'].apply(format_label)
    weekly['color'] = weekly['weight_loss'].apply(lambda v: 'green' if v > 0 else ('red' if v < 0 else 'gray'))

    lost = weekly[weekly['weight_loss'] > 0]
    gained = weekly[weekly['weight_loss'] < 0]
    no_change = weekly[weekly['weight_loss'] == 0]

    figw = go.Figure()
    if not lost.empty:
        figw.add_trace(go.Bar(
            x=lost['week'],
            y=lost['weight_loss'],
            marker_color='green',
            text=lost['text'],
            textposition='outside',
            name="Weight Lost",
            hovertemplate="%{x|%d-%m-%Y}<br>Change: %{text}<extra></extra>"
        ))
    if not gained.empty:
        figw.add_trace(go.Bar(
            x=gained['week'],
            y=gained['weight_loss'],
            marker_color='red',
            text=gained['text'],
            textposition='outside',
            name="Weight Gained",
            hovertemplate="%{x|%d-%m-%Y}<br>Change: %{text}<extra></extra>"
        ))
    if not no_change.empty:
        figw.add_trace(go.Bar(
            x=no_change['week'],
            y=no_change['weight_loss'],
            marker_color='gray',
            text=no_change['text'],
            textposition='outside',
            name="No Change",
            hovertemplate="%{x|%d-%m-%Y}<br>Change: %{text}<extra></extra>"
        ))

    figw.update_layout(
        plot_bgcolor="#3C3C3C", paper_bgcolor="#3C3C3C", font_color="white",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Week Starting",
        xaxis=dict(tickformat="%d-%m-%Y", gridcolor="#555"),
        yaxis=dict(
            title="Weight Change (lbs)",
            tickvals=y_ticks,
            ticktext=y_tick_text,
            zeroline=True, zerolinecolor="white", zerolinewidth=2,
            gridcolor="#555"
        ),
        legend=dict(
            bgcolor="#3C3C3C",
            bordercolor="#222",
            borderwidth=1,
            font=dict(color="white"),
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(figw, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Last table ---

with st.container():
    st.markdown('<div class="section-title">📋 Raw Weight Log</div>', unsafe_allow_html=True)

    # Drop rows with invalid 'dateTime'
    df = df.dropna(subset=['dateTime'])

    # Add helper column for original order
    df['original_index'] = df.index

    # Sort by 'dateTime' descending (newest first), then original index descending to keep entry order on same datetime
    df_sorted = df.sort_values(by=['dateTime', 'original_index'], ascending=[False, False])

    # Display date only without time
    df_sorted['date'] = df_sorted['dateTime'].dt.date

    # Show the cleaned dataframe with Date and Weight columns
    st.dataframe(
        df_sorted[['date', 'weight_stlbs']].rename(columns={
            'date': 'Date',
            'weight_stlbs': 'Weight'
        }).reset_index(drop=True)
    )
