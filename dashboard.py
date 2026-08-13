from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from io import StringIO
from typing import Any

import pandas as pd
import requests
import streamlit as st

NFFC_ADP_URL = "https://nfc.shgn.com/adp/football"
NFFC_ADP_DATA_URL = "https://nfc.shgn.com/adp.data.php"
SLEEPER_PROJECTIONS_URL = "https://api.sleeper.app/projections/nfl/{season}"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_DRAFT_PICKS_URL = "https://api.sleeper.app/v1/draft/{draft_id}/picks"
POSITIONS = ("QB", "RB", "WR", "TE")

LEAGUE_SIZE = 12


def generate_user_picks(draft_position: int, league_size: int = LEAGUE_SIZE, rounds: int = 25) -> list[int]:
    """Generates the user's specific pick numbers across a snake draft."""
    return [
        (r - 1) * league_size + draft_position if r % 2 != 0 else r * league_size - draft_position + 1
        for r in range(1, rounds + 1)
    ]


def extract_draft_id(input_str: str) -> str:
    """Extracts a Sleeper draft ID from a raw ID string or a Sleeper draft URL."""
    input_str = input_str.strip()
    if not input_str:
        return ""
    url_match = re.search(r"draft/(?:nfl/)?([a-zA-Z0-9]+)", input_str, re.IGNORECASE)
    if url_match:
        return url_match.group(1)
    id_match = re.search(r"\b\d{15,20}\b", input_str)
    if id_match:
        return id_match.group(0)
    return re.sub(r"[^a-zA-Z0-9]", "", input_str)


def get_draft_horizon(current_pick: int, user_picks: list[int]) -> tuple[int, int]:
    """
    Returns (upcoming_user_pick, horizon_user_pick).
    Example for Pick 10 slot:
      - At Pick 3: Upcoming = 10, Horizon = 15
      - At Pick 10: Upcoming = 10, Horizon = 15
      - At Pick 12: Upcoming = 15, Horizon = 34
    """
    upcoming_idx = 0
    for idx, p in enumerate(user_picks):
        if p >= current_pick:
            upcoming_idx = idx
            break

    upcoming_pick = user_picks[upcoming_idx]

    if upcoming_idx + 1 < len(user_picks):
        horizon_pick = user_picks[upcoming_idx + 1]
    else:
        horizon_pick = upcoming_pick + 15

    return upcoming_pick, horizon_pick


def adp_to_pick_format(adp: float | int | None) -> str:
    """Converts ADP number to Round.Pick format for a 12-team league (e.g., 37 -> 4.1)."""
    if pd.isna(adp) or adp is None or adp <= 0:
        return "-"
    round_num = int((adp - 1) // LEAGUE_SIZE) + 1
    pick_num = int((adp - 1) % LEAGUE_SIZE) + 1
    return f"{round_num}.{pick_num}"


def normalize_name(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.lower().replace("'", "")
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def first_number(value: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def choose_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", col.lower()): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for key, original in normalized.items():
        if any(candidate in key for candidate in candidates):
            return original
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_nffc(days_back: int = 7) -> pd.DataFrame:
    today = datetime.now()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    response = requests.post(
        NFFC_ADP_DATA_URL,
        data={
            "team_id": "0",
            "from_date": from_date,
            "to_date": to_date,
            "num_teams": "12",
            "draft_type": "0",
            "sport": "football",
            "position": "",
            "league_teams": "0",
        },
        timeout=25,
        headers={"User-Agent": "Mozilla/5.0", "Referer": NFFC_ADP_URL},
    )
    response.raise_for_status()

    if "No ADP Information Available" in response.text:
        raise ValueError(
            f"No NFFC 12-team drafts were completed between {from_date} and {to_date}. "
            "Please widen the 'NFFC ADP Window' slider in the sidebar."
        )

    table_html = (
        """<table><thead><tr><th>Rk</th><th>Player</th><th>Team</th><th>Position(s)</th><th>ADP / AAV</th></tr></thead><tbody>"""
        + response.text
        + "</tbody></table>"
    )
    tables = pd.read_html(StringIO(table_html))
    candidates: list[pd.DataFrame] = []
    for table in tables:
        table.columns = [str(c).strip() for c in table.columns]
        player_col = choose_column(list(table.columns), ("player", "playername", "name"))
        adp_col = choose_column(list(table.columns), ("adp", "averagepick", "avgpick"))
        if player_col and adp_col:
            position_col = choose_column(list(table.columns), ("position", "pos"))
            player_values = table[player_col].astype(str)
            parsed_position = player_values.str.upper().str.extract(r"\b(QB|RB|WR|TE)\b", expand=False)
            frame = pd.DataFrame({
                "Player": table[player_col],
                "NFFC ADP": table[adp_col].map(first_number),
                "Position": table[position_col] if position_col else parsed_position,
            })
            candidates.append(frame)

    if not candidates:
        raise ValueError("Could not parse player/ADP table from NFFC response.")

    result = pd.concat(candidates, ignore_index=True).dropna(subset=["NFFC ADP"])
    result["Position"] = result["Position"].astype(str).str.upper().str.extract(r"(QB|RB|WR|TE)", expand=False)
    return result.dropna(subset=["Position"])


@st.cache_data(ttl=900, show_spinner=False)
def fetch_sleeper(season: int) -> pd.DataFrame:
    records = []
    try:
        response = requests.get(
            SLEEPER_PROJECTIONS_URL.format(season=season),
            params={"season_type": "regular"},
            timeout=25,
        )
        response.raise_for_status()
        rows = response.json()
        if isinstance(rows, list):
            for player in rows:
                stats = player.get("stats") or {}
                player_info = player.get("player") or {}
                adp = next((stats.get(key, player.get(key)) for key in ("adp_ppr", "adp_half_ppr", "adp") if stats.get(key, player.get(key)) is not None), None)
                position = str(player_info.get("position") or player.get("position") or player.get("fantasy_position") or "").upper()
                name = (
                    player_info.get("full_name")
                    or player.get("full_name")
                    or player.get("player_name")
                    or player.get("name")
                    or " ".join(filter(None, (player_info.get("first_name"), player_info.get("last_name"))))
                )
                adp_value = first_number(adp)
                if name and position in POSITIONS and adp_value is not None and adp_value < 999:
                    records.append({"Player": name, "Sleeper ADP": adp_value, "Position": position})
    except Exception:
        pass

    if not records:
        response = requests.get(SLEEPER_PLAYERS_URL, timeout=25)
        response.raise_for_status()
        players = response.json()
        for _, info in players.items():
            if not isinstance(info, dict):
                continue
            position = str(info.get("position", "")).upper()
            rank = info.get("search_rank")
            full_name = info.get("full_name")
            if full_name and position in POSITIONS and rank is not None and rank < 300:
                records.append({"Player": full_name, "Sleeper ADP": float(rank), "Position": position})

    if not records:
        raise ValueError("Sleeper returned no usable ADP or rank values.")
    return pd.DataFrame(records)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_sleeper_draft_picks(draft_id: str) -> list[dict]:
    if not draft_id.strip():
        return []
    try:
        response = requests.get(SLEEPER_DRAFT_PICKS_URL.format(draft_id=draft_id.strip()), timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return []


def build_board(
    nffc: pd.DataFrame, sleeper: pd.DataFrame, current_pick: int, user_picks: list[int]
) -> tuple[pd.DataFrame, int, int]:
    nffc = nffc.copy()
    sleeper = sleeper.copy()
    nffc["key"] = nffc["Player"].map(normalize_name)
    sleeper["key"] = sleeper["Player"].map(normalize_name)
    nffc = nffc.sort_values("NFFC ADP").drop_duplicates("key")
    sleeper = sleeper.sort_values("Sleeper ADP").drop_duplicates("key")

    board = nffc.merge(sleeper[["key", "Sleeper ADP"]], on="key", how="inner")
    board["Value"] = (board["Sleeper ADP"] - board["NFFC ADP"]).round(1)

    board["NFFC Pick"] = board["NFFC ADP"].map(adp_to_pick_format)
    board["Sleeper Pick"] = board["Sleeper ADP"].map(adp_to_pick_format)

    upcoming_pick, target_horizon_pick = get_draft_horizon(current_pick, user_picks)

    def assign_risk_status(row):
        nffc_adp = row["NFFC ADP"]
        sleeper_adp = row["Sleeper ADP"]

        if nffc_adp <= target_horizon_pick:
            if sleeper_adp <= target_horizon_pick:
                return "🚨 AT RISK"
            elif sleeper_adp <= target_horizon_pick + 5:
                return "👀 ON WATCH"
            else:
                return "⏳ CAN WAIT"
        return "OK"

    board["Draft Status"] = board.apply(assign_risk_status, axis=1)

    cols = [
        "Draft Status", "Player", "Position",
        "NFFC ADP", "NFFC Pick",
        "Sleeper ADP", "Sleeper Pick",
        "Value", "key"
    ]
    return board[cols].sort_values("NFFC ADP", ascending=True), upcoming_pick, target_horizon_pick


def color_value(val: float | int) -> str:
    if pd.isna(val):
        return ""
    if val >= 10.0:
        return "background-color: #2e7d32; color: #ffffff; font-weight: bold;"
    elif val > 0.0:
        return "background-color: #c8e6c9; color: #1b5e20;"
    elif val <= -10.0:
        return "background-color: #c62828; color: #ffffff; font-weight: bold;"
    elif val < 0.0:
        return "background-color: #ffcdd2; color: #b71c1c;"
    return ""


# Streamlit Page Config
st.set_page_config(page_title="Draft Turn Strategy Board", page_icon="🏈", layout="wide")

if "drafted" not in st.session_state:
    st.session_state.drafted = set()

with st.sidebar:
    st.header("Draft Controls")

    # Select draft slot (1 to LEAGUE_SIZE)
    draft_position = st.selectbox(
        "Your Draft Position / Slot",
        options=list(range(1, LEAGUE_SIZE + 1)),
        index=9,  # Default to Pick 10 (0-indexed 9)
        help="Select your draft slot in the 12-team order."
    )

    raw_draft_input = st.text_input(
        "Sleeper Draft ID or Link (Live Sync)",
        help="Paste your Sleeper draft ID or full draft URL (e.g. https://sleeper.app/draft/nfl/123456789...)"
    )

    nffc_days_back = st.slider(
        "NFFC ADP Window (Days)",
        min_value=1,
        max_value=14,
        value=7,
        step=1,
        help="Fetch NFFC drafts from the last X days."
    )

    manual_pick = st.number_input("Active Draft Pick #", min_value=1, max_value=250, value=draft_position, step=1)
    season = st.number_input("Sleeper season", min_value=2020, max_value=2035, value=datetime.now().year, step=1)
    hide_drafted = st.checkbox("Hide drafted players", value=True)
    position = st.selectbox("Position", ("ALL", *POSITIONS))
    search = st.text_input("Search player")

    if st.button("Refresh live ADP & Sync", use_container_width=True):
        fetch_nffc.clear()
        fetch_sleeper.clear()
        fetch_sleeper_draft_picks.clear()
        st.rerun()

    if st.button("Reset draft board", type="secondary", use_container_width=True):
        st.session_state.drafted = set()
        st.rerun()

# Dynamic Title
st.title(f"🏈 Pick {draft_position} Turn Strategy Board (12-Team)")

# Compute user picks based on chosen slot
user_picks = generate_user_picks(int(draft_position))

# Extract draft ID from raw string input or URL
draft_id = extract_draft_id(raw_draft_input)

# Process Sleeper Live Sync
synced_picks = []
if draft_id:
    synced_picks = fetch_sleeper_draft_picks(draft_id)
    if synced_picks:
        for pick in synced_picks:
            meta = pick.get("metadata", {})
            first_name = meta.get("first_name", "")
            last_name = meta.get("last_name", "")
            full_name = f"{first_name} {last_name}".strip()
            if full_name:
                st.session_state.drafted.add(normalize_name(full_name))
        current_pick = len(synced_picks) + 1
        st.sidebar.success(f"⚡ Live Sync Active: Pick {current_pick} (ID: {draft_id})")
    else:
        st.sidebar.warning(f"Could not fetch picks for Draft ID: {draft_id}")
        current_pick = manual_pick
else:
    current_pick = manual_pick

try:
    with st.spinner("Fetching live ADP sources…"):
        board, upcoming_pick, next_target_pick = build_board(
            fetch_nffc(days_back=int(nffc_days_back)),
            fetch_sleeper(int(season)),
            int(current_pick),
            user_picks
        )

    csv = board.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("📥 Download CSV Backup", csv, "adp_board.csv", "text/csv")
except Exception as exc:
    st.error(f"Live data could not be loaded: {exc}")
    st.stop()

# Info Banner showing active draft pick, upcoming selection, and target survival horizon
st.info(
    f"📍 **Active Pick:** #{current_pick} | "
    f"🎯 **Upcoming Pick:** #{upcoming_pick} ({adp_to_pick_format(upcoming_pick)}) | "
    f"🔭 **Target Horizon:** Evaluating survival through Pick **#{next_target_pick}** ({adp_to_pick_format(next_target_pick)})"
)

filtered = board.copy()
if position != "ALL":
    filtered = filtered[filtered["Position"] == position]
if search:
    filtered = filtered[filtered["Player"].str.contains(search, case=False, na=False)]
if hide_drafted:
    filtered = filtered[~filtered["key"].isin(st.session_state.drafted)]

df_display = filtered.copy()
df_display["Drafted"] = df_display["key"].isin(st.session_state.drafted)
df_display = df_display[[
    "Drafted", "Draft Status", "Player", "Position",
    "NFFC ADP", "NFFC Pick",
    "Sleeper ADP", "Sleeper Pick",
    "Value", "key"
]]

st.session_state["rendered_keys"] = df_display["key"].tolist()

def update_drafted_state():
    editor_state = st.session_state.get("draft_table_editor", {})
    edited_rows = editor_state.get("edited_rows", {})
    rendered_keys = st.session_state.get("rendered_keys", [])

    for row_idx, changes in edited_rows.items():
        if "Drafted" in changes and row_idx < len(rendered_keys):
            player_key = rendered_keys[row_idx]
            if changes["Drafted"]:
                st.session_state.drafted.add(player_key)
            else:
                st.session_state.drafted.discard(player_key)

styled_df = df_display.style.map(color_value, subset=["Value"])

st.data_editor(
    styled_df,
    column_config={
        "Drafted": st.column_config.CheckboxColumn("Drafted?", help="Check off player as drafted", default=False),
        "Draft Status": st.column_config.TextColumn("Draft Status", help="Risk of being drafted before your next-next pick horizon"),
        "NFFC ADP": st.column_config.NumberColumn("NFFC ADP", format="%.1f"),
        "NFFC Pick": st.column_config.TextColumn("NFFC Pick"),
        "Sleeper ADP": st.column_config.NumberColumn("Sleeper ADP", format="%.1f"),
        "Sleeper Pick": st.column_config.TextColumn("Sleeper Pick"),
        "Value": st.column_config.NumberColumn("Value", format="%+.1f"),
        "key": None,
    },
    disabled=["Draft Status", "Player", "Position", "NFFC ADP", "NFFC Pick", "Sleeper ADP", "Sleeper Pick", "Value"],
    hide_index=True,
    use_container_width=True,
    height=800,
    key="draft_table_editor",
    on_change=update_drafted_state,
)

if st.session_state.drafted:
    with st.expander("📋 Drafted Players Log / Undraft Manager"):
        drafted_board = board[board["key"].isin(st.session_state.drafted)].sort_values("Player")
        for _, row in drafted_board.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• **{row['Player']}** ({row['Position']}) — NFFC: {row['NFFC Pick']} ({row['NFFC ADP']}) | Sleeper: {row['Sleeper Pick']} ({row['Sleeper ADP']})")
            with col2:
                if st.button("Restore", key=f"restore_{row['key']}", use_container_width=True):
                    st.session_state.drafted.remove(row['key'])
                    st.rerun()