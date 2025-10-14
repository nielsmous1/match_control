import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import os
from pathlib import Path
from collections import defaultdict
from mplsoccer import Pitch, VerticalPitch
import numpy as np

# Page config
st.set_page_config(page_title="Match Control Analysis", layout="wide")

# Initialize session state for colors
if 'home_color' not in st.session_state:
    st.session_state.home_color = '#1f77b4'
if 'away_color' not in st.session_state:
    st.session_state.away_color = '#ff7f0e'

# Title
st.title("Eredivisie 2025/2026 Data-Analyse")

# Sidebar for appearance
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.home_color = st.color_picker("Kleur Thuis", 
                                                      st.session_state.home_color)
    with col2:
        st.session_state.away_color = st.color_picker("Kleur Uit", 
                                                      st.session_state.away_color)

# Main screen: folder-only selection with team and match dropdowns
events_data = None
file_name = None

# Resolve paths relative to this file so it works on Streamlit Cloud
BASE_DIR = Path(__file__).parent
match_folder = str((BASE_DIR / "MatchEvents").resolve())

def load_json_lenient(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        # Fallbacks for BOM, concatenated JSON, or NDJSON
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        raw = raw.lstrip('\ufeff').strip()
        # Try as a single JSON with trimming to outermost braces
        try:
            return json.loads(raw)
        except Exception:
            pass
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except Exception:
                pass
        # Try NDJSON: one JSON object per line, collect into list under 'data'
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                continue
        if items:
            return { 'data': items }
        raise

def parse_teams_from_filename(name: str):
    # Extract: date (8 digits), then everything up to 'SciSportsEvents' => "Home vs Away"
    try:
        base = name
        if base.lower().endswith('.json'):
            base = base[:-5]
        # Expect leading date and a space
        date_part = base[:8]
        if not date_part.isdigit():
            return None, None, None
        rest = base[8:].lstrip()
        lower_rest = rest.lower()
        marker = 'scisportsevents'
        idx = lower_rest.find(marker)
        if idx == -1:
            return None, None, date_part
        middle = rest[:idx].strip()
        # Now middle should be "Home vs Away" exactly once
        if ' vs ' in middle:
            home, away = middle.split(' vs ', 1)
            home = home.strip()
            away = away.strip()
            return home or None, away or None, date_part
        return None, None, date_part
    except Exception:
        return None, None, None

available_teams = {}
files_info = []  # list of dicts: {path, name, home, away, date, label}

if os.path.exists(match_folder):
    json_files = sorted([p for p in Path(match_folder).glob("*.json")])
    for p in json_files:
        home, away, yyyymmdd = parse_teams_from_filename(p.name)
        # Only use filename-derived teams to avoid partial tokens from metadata
        # Build friendly label like: "Home - Away DD-MM-YYYY"
        label = p.name
        if home and away and yyyymmdd and len(yyyymmdd) == 8:
            dd = yyyymmdd[6:8]
            mm = yyyymmdd[4:6]
            yyyy = yyyymmdd[0:4]
            label = f"{home} - {away}, {dd}-{mm}-{yyyy}"
        # Build canonical team map (case-insensitive dedupe)
        if home:
            key = home.strip().lower()
            if key not in available_teams:
                available_teams[key] = home
        if away:
            key = away.strip().lower()
            if key not in available_teams:
                available_teams[key] = away
        files_info.append({
            'path': str(p),
            'name': p.name,
            'home': home,
            'away': away,
            'date': yyyymmdd,
            'label': label
        })
else:
    st.warning("Folder not found")

selected_team = None
selected_match = None

if available_teams:
    team_options = sorted(available_teams.values(), key=lambda s: s.lower())
    selected_team = st.selectbox("Selecteer een team", team_options)
    team_matches = []
    if selected_team:
        for info in files_info:
            if (
                (info['home'] and info['home'].strip().lower() == selected_team.strip().lower()) or
                (info['away'] and info['away'].strip().lower() == selected_team.strip().lower())
            ):
                team_matches.append(info)
        # Reverse order to show most recent matches first
        team_matches.reverse()
        # Build friendly labels
        match_labels = [info['label'] for info in team_matches]
        if match_labels:
            choice = st.selectbox("Selecteer een wedstrijd", match_labels)
            if choice:
                sel = next((i for i in team_matches if i['label'] == choice), None)
                if sel:
                    file_name = sel['name']
                    try:
                        events_data = load_json_lenient(sel['path'])
                    except Exception as e:
                        st.error(f"Failed to load JSON: {e}")
                        events_data = None
        else:
            st.info("No matches found for the selected team in this folder.")
else:
    st.info("No JSON files found in the specified folder")

# Load custom icons unconditionally (relative to repo)
try:
    icons_dir = BASE_DIR / "icons"
    ball_icon_path = icons_dir / "football.png"
    sub_icon_path = icons_dir / "subicon.png"
    redcard_icon_path = icons_dir / "red_card.png"
    ball_icon = mpimg.imread(str(ball_icon_path))
    sub_icon = mpimg.imread(str(sub_icon_path))
    redcard_icon = mpimg.imread(str(redcard_icon_path))
except Exception:
    st.warning("Icon files not found in './icons'. Using default markers.")
    ball_icon = None
    sub_icon = None
    redcard_icon = None

def load_team_logo(team_name):
    """Load team logo from logos folder"""
    try:
        logos_dir = BASE_DIR / "logos"
        logo_path = logos_dir / f"{team_name}.png"
        
        if logo_path.exists():
            return mpimg.imread(str(logo_path))
        
        return None
    except Exception:
        return None

# Make colors available globally for the function
home_color = st.session_state.home_color
away_color = st.session_state.away_color

def calculate_game_control_and_domination(data, home_team_override=None, away_team_override=None):
    """
    Calculate both game control (possession metrics) and domination (threat creation).
    
    Control: Shown as filled areas with outline - includes successful passes,
             final third passes, interceptions, tackles, dribbles, and recoveries.
    Domination: Shown as dashed lines in team colors - includes goals, shots, and
                dangerous passes to/in box with xG incorporated.
    """
    
    # Handle different data structures
    if isinstance(data, dict) and 'data' in data:
        events = data['data']
    elif isinstance(data, dict) and 'events' in data:
        events = data['events']
    elif isinstance(data, list):
        events = data
    else:
        events = []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    events = v
                    break
    
    # Get team names from metadata if present
    home_team = home_team_override
    away_team = away_team_override
    
    if home_team is None or away_team is None:
        if isinstance(data, dict) and 'metaData' in data and isinstance(data['metaData'], dict):
            metadata = data['metaData']
            home_meta = metadata.get('homeTeamName') or metadata.get('homeTeam') or metadata.get('home')
            away_meta = metadata.get('awayTeamName') or metadata.get('awayTeam') or metadata.get('away')
            
            if home_team is None and home_meta:
                home_team = home_meta
            if away_team is None and away_meta:
                away_team = away_meta
    
    # Normalized team strings for matching
    def _norm(s):
        return s.lower().strip() if isinstance(s, str) else ''
    
    home_norm = _norm(home_team)
    away_norm = _norm(away_team)
    
    # DOMINATION weights (threat/danger creation)
    DOMINATION_WEIGHTS = {
        'GOAL': 8,
        'SHOT_ON_TARGET': 8,
        'SHOT_POST': 6,
        'INTERCEPTION_FINAL_THIRD': 5,
        'PASS_TO_BOX': 4,
        'SHOT_BLOCKED': 3,
        'PASS_IN_BOX': 5,
        'SHOT_WIDE': 3,
        'DRIBBLE_TO_BOX': 4,
        'XG_MULTIPLIER': 20
    }
    
    # CONTROL weights (possession/ball control)
    CONTROL_WEIGHTS = {
        'PASS_KEY': 2.5,
        'PASS_TO_FINAL_THIRD': 2,
        'PASS_IN_FINAL_THIRD': 2.5,
        'INTERCEPTION': 2,
        'DRIBBLE_SUCCESSFUL': 2.5,
        'COUNTER_INTERCEPTION': 2.5,
    }
    
    # SciSports type mappings
    BASE_TYPE = {
        'PASS': 1,
        'DRIBBLE': 2,
        'TACKLE': 3,
        'INTERCEPTION': 5,
        'SHOT': 6,
        'BALL_RECOVERY': 9,
        'PERIOD': 14,
        'SUBSTITUTE': 16,
    }
    
    SUB_TYPE = {
        'OWN_GOAL': 1101,
        'END_PERIOD': 1401,
        'START_PERIOD': 1400,
        'SUBBED_OUT': 1600,
        'SUBBED_IN': 1601,
        'KEY_PASS': 101,
        'COUNTER_INTERCEPTION': 196,
    }
    
    RESULT = {
        'UNSUCCESSFUL': 0,
        'SUCCESSFUL': 1,
    }
    
    SHOT_TYPE = {
        'WIDE': 1,
        'POST': 2,
        'ON_TARGET': 3,
        'BLOCKED': 4,
    }
    
    # Helper to match team
    def match_event_team(event_team_str):
        team_norm = _norm(event_team_str)
        if not team_norm:
            return None
        
        if home_norm and (home_norm in team_norm or team_norm in home_norm):
            return home_team
        if away_norm and (away_norm in team_norm or team_norm in away_norm):
            return away_team
        
        if home_norm and any(part in team_norm for part in home_norm.split()):
            return home_team
        if away_norm and any(part in team_norm for part in away_norm.split()):
            return away_team
        
        return None
    
    # Initialize event lists
    first_half_domination_events = []
    first_half_control_events = []
    second_half_domination_events = []
    second_half_control_events = []
    first_half_goals = []
    second_half_goals = []
    first_half_subs = []
    second_half_subs = []
    first_half_cards = []
    second_half_cards = []
    
    # Process events
    for event in events:
        team_field = event.get('teamName') or event.get('team') or event.get('team_name') or event.get('teamNameFormatted')
        matched_team = match_event_team(team_field)
        
        if matched_team is None:
            continue
        
        team = matched_team
        
        base_type_id = event.get('baseTypeId')
        sub_type_id = event.get('subTypeId')
        result_id = event.get('resultId')
        shot_type_id = event.get('shotTypeId')
        
        start_x = event.get('startPosXM')
        end_x = event.get('endPosXM')
        start_y = event.get('startPosYM')
        end_y = event.get('endPosYM')
        
        time_ms = event.get('startTimeMs', 0) or event.get('timeMs', 0) or event.get('timestampMs', 0)
        minute = time_ms / 1000 / 60 if time_ms else 0
        
        if not time_ms:
            continue
        
        domination_value = 0
        control_value = 0
        domination_type = None
        control_type = None
        
        # Determine half
        part_id = event.get('partId')
        part_name = event.get('partName', '').upper()
        
        if part_id == 1 or part_name == 'FIRST_HALF' or part_name == 'FIRST HALF':
            target_domination_events = first_half_domination_events
            target_control_events = first_half_control_events
            target_goals = first_half_goals
            target_subs = first_half_subs
            target_cards = first_half_cards
        elif part_id == 2 or part_name == 'SECOND_HALF' or part_name == 'SECOND HALF':
            target_domination_events = second_half_domination_events
            target_control_events = second_half_control_events
            target_goals = second_half_goals
            target_subs = second_half_subs
            target_cards = second_half_cards
        else:
            continue
        
        # Check for goals
        if base_type_id == BASE_TYPE['SHOT'] and result_id == RESULT['SUCCESSFUL']:
            target_goals.append({
                'team': team,
                'minute': minute,
                'player': event.get('playerName', 'Unknown')
            })
            # Check if it's a penalty shot and use xG of 0.76
            is_penalty = (base_type_id == 6 and sub_type_id == 602)
            xg_value = 0.76 if is_penalty else event.get('metrics', {}).get('xG', 0.0)
            domination_value = DOMINATION_WEIGHTS['GOAL'] + (xg_value * DOMINATION_WEIGHTS['XG_MULTIPLIER'])
            domination_type = 'GOAL'
        
        # Check for own goals
        elif sub_type_id == SUB_TYPE['OWN_GOAL']:
            opposing_team = away_team if team == home_team else home_team
            target_goals.append({
                'team': opposing_team,
                'minute': minute,
                'player': f"OG: {event.get('playerName', 'Unknown')}"
            })
            target_domination_events.append({
                'team': opposing_team,
                'minute': minute,
                'value': DOMINATION_WEIGHTS['GOAL'],
                'type': 'OWN_GOAL'
            })
            continue
        
        # Other shot types
        elif base_type_id == BASE_TYPE['SHOT']:
            # Check if it's a penalty shot and use xG of 0.76
            is_penalty = (base_type_id == 6 and sub_type_id == 602)
            xg_value = (0.76 if is_penalty else event.get('metrics', {}).get('xG', 0.0)) * DOMINATION_WEIGHTS['XG_MULTIPLIER']
            if shot_type_id == SHOT_TYPE['ON_TARGET']:
                domination_value = DOMINATION_WEIGHTS['SHOT_ON_TARGET'] + xg_value
                domination_type = 'SHOT_ON_TARGET'
            elif shot_type_id == SHOT_TYPE['POST']:
                domination_value = DOMINATION_WEIGHTS['SHOT_POST'] + xg_value
                domination_type = 'SHOT_POST'
            elif shot_type_id == SHOT_TYPE['BLOCKED']:
                domination_value = DOMINATION_WEIGHTS['SHOT_BLOCKED'] + xg_value
                domination_type = 'SHOT_BLOCKED'
            elif shot_type_id == SHOT_TYPE['WIDE']:
                domination_value = DOMINATION_WEIGHTS['SHOT_WIDE'] + xg_value
                domination_type = 'SHOT_WIDE'
        
        # Substitutions
        elif base_type_id == BASE_TYPE['SUBSTITUTE']:
            if sub_type_id == SUB_TYPE['SUBBED_IN']:
                player_out = 'Unknown'
                for other_event in events:
                    if (other_event.get('baseTypeId') == BASE_TYPE['SUBSTITUTE'] and
                        other_event.get('subTypeId') == SUB_TYPE['SUBBED_OUT'] and
                        abs((other_event.get('startTimeMs', 0) or 0) - time_ms) < 1000 and
                        match_event_team(other_event.get('teamName') or other_event.get('team')) == team):
                        player_out = other_event.get('playerName', 'Unknown')
                        break
                
                target_subs.append({
                    'team': team,
                    'minute': minute,
                    'player_in': event.get('playerName', 'Unknown'),
                    'player_out': player_out
                })
        
        # Cards
        if base_type_id == 15:
            if sub_type_id in (1501, 1502):
                target_cards.append({
                    'team': team,
                    'minute': minute,
                    'player': event.get('playerName', 'Unknown'),
                    'type': 'RED'
                })
        
        # Passes
        if base_type_id == BASE_TYPE['PASS']:
            if result_id == RESULT['SUCCESSFUL']:
                if sub_type_id == SUB_TYPE.get('KEY_PASS'):
                    control_value += CONTROL_WEIGHTS['PASS_KEY']
                    control_type = 'PASS_KEY'
                
                if start_x is not None and end_x is not None:
                    in_final_third = (start_x >= 17.5) and (end_x > 17.5)
                    to_final_third = (start_x < 17.5) and (end_x > 17.5)
                    
                    if to_final_third:
                        control_value += CONTROL_WEIGHTS['PASS_TO_FINAL_THIRD']
                        control_type = 'PASS_TO_FINAL_THIRD'
                    elif in_final_third:
                        control_value += CONTROL_WEIGHTS['PASS_IN_FINAL_THIRD']
                        control_type = 'PASS_IN_FINAL_THIRD'
            
            if result_id == RESULT['SUCCESSFUL'] and start_x is not None and end_x is not None:
                in_box = (start_x >= 36) and (end_x > 36) and (end_y is not None and abs(end_y) < 20.15)
                to_box = (start_x < 36) and (end_x > 36) and (end_y is not None and abs(end_y) < 20.15)
                
                if to_box:
                    domination_value = DOMINATION_WEIGHTS['PASS_TO_BOX']
                    domination_type = 'PASS_TO_BOX'
                elif in_box:
                    domination_value = DOMINATION_WEIGHTS['PASS_IN_BOX']
                    domination_type = 'PASS_IN_BOX'
        
        # Interceptions
        elif base_type_id == BASE_TYPE['INTERCEPTION']:
            if sub_type_id == SUB_TYPE.get('COUNTER_INTERCEPTION'):
                control_value = CONTROL_WEIGHTS['COUNTER_INTERCEPTION']
                control_type = 'COUNTER_INTERCEPTION'
            else:
                control_value = CONTROL_WEIGHTS['INTERCEPTION']
                control_type = 'INTERCEPTION'
            
            if start_x is not None and start_x > 17.5:
                domination_value = DOMINATION_WEIGHTS['INTERCEPTION_FINAL_THIRD']
                domination_type = 'INTERCEPTION_FINAL_THIRD'
        
        # Dribbles
        elif base_type_id == BASE_TYPE['DRIBBLE'] and result_id == RESULT['SUCCESSFUL']:
            to_box = (start_x < 36) and (end_x > 36) and (end_y is not None and abs(end_y) < 20.15)
            
            if to_box:
                domination_value = DOMINATION_WEIGHTS['DRIBBLE_TO_BOX']
                domination_type = 'DRIBBLE_TO_BOX'
            else:
                control_value = CONTROL_WEIGHTS['DRIBBLE_SUCCESSFUL']
                control_type = 'DRIBBLE_SUCCESSFUL'
        
        # Ball recoveries
        elif base_type_id == BASE_TYPE.get('BALL_RECOVERY'):
            if sub_type_id == SUB_TYPE.get('COUNTER_INTERCEPTION'):
                control_value = CONTROL_WEIGHTS['COUNTER_INTERCEPTION']
                control_type = 'COUNTER_INTERCEPTION'
            else:
                control_value = CONTROL_WEIGHTS.get('BALL_RECOVERY', 2)
                control_type = 'BALL_RECOVERY'
        
        # Add to events lists
        if domination_value > 0:
            target_domination_events.append({
                'team': team,
                'minute': minute,
                'value': domination_value,
                'type': domination_type
            })
        
        if control_value > 0:
            target_control_events.append({
                'team': team,
                'minute': minute,
                'value': control_value,
                'type': control_type
            })
    
    # Function to calculate metrics for a specific half
    def calculate_half_metrics(domination_events, control_events, start_minute, end_minute):
        if not domination_events and not control_events:
            return [], [], [], [], [], [], []
        
        minutes = np.arange(start_minute, end_minute + 0.5, 0.5)
        home_domination = []
        away_domination = []
        net_domination = []
        home_control = []
        away_control = []
        net_control = []
        window_size = 5
        
        for current_minute in minutes:
            window_start = max(start_minute, current_minute - window_size)
            window_end = current_minute
            
            home_dom_sum = sum(e['value'] for e in domination_events
                          if e['team'] == home_team and window_start <= e['minute'] <= window_end)
            away_dom_sum = sum(e['value'] for e in domination_events
                          if e['team'] == away_team and window_start <= e['minute'] <= window_end)
            
            home_domination.append(home_dom_sum)
            away_domination.append(away_dom_sum)
            net_domination.append(home_dom_sum - away_dom_sum)
            
            home_ctrl_sum = sum(e['value'] for e in control_events
                          if e['team'] == home_team and window_start <= e['minute'] <= window_end)
            away_ctrl_sum = sum(e['value'] for e in control_events
                          if e['team'] == away_team and window_start <= e['minute'] <= window_end)
            
            home_control.append(home_ctrl_sum)
            away_control.append(away_ctrl_sum)
            net_control.append(home_ctrl_sum - away_ctrl_sum)
        
        total_home_control = sum(e['value'] for e in control_events if e['team'] == home_team)
        total_away_control = sum(e['value'] for e in control_events if e['team'] == away_team)
        
        return minutes, net_domination, net_control, home_domination, away_domination, home_control, away_control
    
    # Calculate metrics for each half
    first_half_start = min([e['minute'] for e in first_half_domination_events + first_half_control_events]) if (first_half_domination_events or first_half_control_events) else 0
    first_half_end = max([e['minute'] for e in first_half_domination_events + first_half_control_events]) if (first_half_domination_events or first_half_control_events) else 45
    
    second_half_start = min([e['minute'] for e in second_half_domination_events + second_half_control_events]) if (second_half_domination_events or second_half_control_events) else 45
    second_half_end = max([e['minute'] for e in second_half_domination_events + second_half_control_events]) if (second_half_domination_events or second_half_control_events) else 90
    
    first_half_minutes, first_half_net_dom, first_half_net_ctrl, first_half_home_dom, first_half_away_dom, first_half_home_ctrl, first_half_away_ctrl = calculate_half_metrics(
        first_half_domination_events, first_half_control_events, first_half_start, first_half_end
    )
    
    second_half_minutes, second_half_net_dom, second_half_net_ctrl, second_half_home_dom, second_half_away_dom, second_half_home_ctrl, second_half_away_ctrl = calculate_half_metrics(
        second_half_domination_events, second_half_control_events, second_half_start, second_half_end
    )
    
    # Calculate durations
    first_half_duration = first_half_end - first_half_start if (first_half_domination_events or first_half_control_events) else 45
    second_half_duration = second_half_end - second_half_start if (second_half_domination_events or second_half_control_events) else 45
    
    total_duration = first_half_duration + second_half_duration
    first_half_ratio = first_half_duration / total_duration if total_duration > 0 else 0.5
    second_half_ratio = second_half_duration / total_duration if total_duration > 0 else 0.5
    
    # Create visualization
    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[first_half_ratio, second_half_ratio],
                          height_ratios=[5, 1], hspace=0.25, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])
    
    home_plot_color = home_color
    away_plot_color = away_color
    
    # Set danger line colors - default is red and black
    home_danger_line_color = 'red'
    away_danger_line_color = 'black'
    
    # If home team is red (#e50000), use black for home and red for away
    if home_color.lower().replace('#', '') == 'e50000':
        home_danger_line_color = 'black'
        away_danger_line_color = 'red'
    
    # If away team is black (#000000), use red for away and black for home
    if away_color.lower().replace('#', '') == '000000':
        away_danger_line_color = 'red'
        home_danger_line_color = 'black'
    
    # If home team is black (#000000), use red for home and black for away
    if home_color.lower().replace('#', '') == '000000':
        home_danger_line_color = 'red'
        away_danger_line_color = 'black'
    
    # If away team is red (#e50000), use black for away and red for home
    if away_color.lower().replace('#', '') == 'e50000':
        away_danger_line_color = 'black'
        home_danger_line_color = 'red'
    
    # Plot function
    def plot_half(ax, minutes, home_domination, away_domination, net_control, home_control, away_control,
                  goals, subs, cards, half_name, home_color, away_color, home_team_name, away_team_name):
        ax.set_facecolor('#f5f5f5')
        
        if len(minutes) == 0:
            ax.text(0.5, 0.5, f'No data for {half_name}', ha='center', va='center', transform=ax.transAxes)
            return
        
        ax.axhline(y=0, color='#95A5A6', linestyle='-', linewidth=1)
        
        control_minutes = minutes
        control_values = net_control
        
        ax.plot(control_minutes, control_values, color='#2E4053', linewidth=1.5, zorder=4)
        
        for i in range(len(control_minutes)-1):
            if control_values[i] > 0 or control_values[i+1] > 0:
                ax.fill_between([control_minutes[i], control_minutes[i+1]], 0,
                               [control_values[i], control_values[i+1]],
                               color=home_color, alpha=0.6, zorder=2)
            elif control_values[i] < 0 or control_values[i+1] < 0:
                ax.fill_between([control_minutes[i], control_minutes[i+1]], 0,
                               [control_values[i], control_values[i+1]],
                               color=away_color, alpha=0.6, zorder=2)
        
        if len(minutes) > 1 and len(home_domination) > 1 and len(away_domination) > 1:
            domination_indices = list(range(0, len(minutes), 1))
            if domination_indices[-1] != len(minutes) - 1:
                domination_indices.append(len(minutes) - 1)
            
            smooth_minutes = np.array([minutes[i] for i in domination_indices])
            smooth_home_domination = [home_domination[i] for i in domination_indices]
            smooth_away_domination = [-away_domination[i] for i in domination_indices]
        else:
            smooth_minutes = minutes
            smooth_home_domination = home_domination
            smooth_away_domination = [-away_domination[i] for i in range(len(away_domination))]
        
        ax.plot(smooth_minutes, smooth_home_domination, color=home_danger_line_color, linewidth=2,
                linestyle='--', zorder=6, label=f'{home_team_name} Danger', alpha=0.9, dashes=(5, 3))
        ax.plot(smooth_minutes, smooth_away_domination, color=away_danger_line_color, linewidth=2,
                linestyle='--', zorder=6, label=f'{away_team_name} Danger', alpha=0.9, dashes=(5, 3))
        
        y_limit = 80
        ax.set_ylim(-y_limit, y_limit)
        
        # Add goal markers
        home_goals_half = [g for g in goals if g['team'] == home_team_name]
        away_goals_half = [g for g in goals if g['team'] == away_team_name]
        
        for goal in home_goals_half:
            ax.axvline(x=goal['minute'], ymin=0.5, ymax=1, color=home_color,
                      linestyle='--', linewidth=1.5, alpha=0.7)
            if ball_icon is not None:
                ax.imshow(ball_icon, extent=(goal['minute'] - 0.75, goal['minute'] + 0.75,
                         y_limit * 0.8, y_limit * 0.9), aspect="auto", zorder=10)
            else:
                ax.scatter(goal['minute'], y_limit * 0.85, s=300, marker='o',
                          color='white', edgecolor=home_color, linewidth=2, zorder=10)
                ax.text(goal['minute'], y_limit * 0.85, 'G', fontsize=12, ha='center', va='center',
                       fontweight='bold', color=home_color)
        
        for goal in away_goals_half:
            ax.axvline(x=goal['minute'], ymin=0, ymax=0.5, color=away_color,
                      linestyle='--', linewidth=1.5, alpha=0.7)
            if ball_icon is not None:
                ax.imshow(ball_icon, extent=(goal['minute'] - 0.75, goal['minute'] + 0.75,
                         -y_limit * 0.9, -y_limit * 0.8), aspect="auto", zorder=10)
            else:
                ax.scatter(goal['minute'], -y_limit * 0.85, s=300, marker='o',
                          color='white', edgecolor=away_color, linewidth=2, zorder=10)
                ax.text(goal['minute'], -y_limit * 0.85, 'G', fontsize=12, ha='center', va='center',
                       fontweight='bold', color=away_color)
        
        # Add substitution markers
        for sub in subs:
            sub_minute = sub['minute']
            if half_name == 'First Half' and abs(sub_minute - minutes[-1]) < 1:
                continue
            elif half_name == 'Second Half' and abs(sub_minute - minutes[0]) < 1:
                sub_minute = minutes[0]
            
            if sub['team'] == home_team_name:
                ax.axvline(x=sub_minute, ymin=0.5, ymax=1, color='#7F8C8D',
                          linestyle='--', linewidth=1, alpha=0.5)
                y_pos_bottom = y_limit * 0.65
                y_pos_top = y_limit * 0.75
            else:
                ax.axvline(x=sub_minute, ymin=0, ymax=0.5, color='#7F8C8D',
                          linestyle='--', linewidth=1, alpha=0.5)
                y_pos_bottom = -y_limit * 0.75
                y_pos_top = -y_limit * 0.65
            
            if sub_icon is not None:
                ax.imshow(sub_icon, extent=(sub_minute - 0.75, sub_minute + 0.75,
                         y_pos_bottom, y_pos_top), aspect="auto", zorder=9)
            else:
                ax.scatter(sub_minute, (y_pos_bottom + y_pos_top)/2, s=150, marker='o',
                          color='white', edgecolor='#7F8C8D', linewidth=2, zorder=9)
                ax.text(sub_minute, (y_pos_bottom + y_pos_top)/2, 'S', fontsize=10, ha='center', va='center',
                       color='#7F8C8D', fontweight='bold')
        
        # Add red card markers
        for card in cards:
            card_minute = card['minute']
            if card['team'] == home_team_name:
                ax.axvline(x=card_minute, ymin=0.5, ymax=1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                y_bottom = y_limit * 0.70
                y_top = y_limit * 0.80
                if redcard_icon is not None:
                    ax.imshow(redcard_icon, extent=(card_minute - 0.5, card_minute + 0.5, y_bottom, y_top),
                              aspect='auto', zorder=11)
                else:
                    ax.scatter(card_minute, y_limit * 0.75, s=220, marker='s', color='red', zorder=11)
                    ax.text(card_minute, y_limit * 0.75, 'RC', fontsize=9, ha='center', va='center', color='white', fontweight='bold')
            else:
                ax.axvline(x=card_minute, ymin=0, ymax=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                y_bottom = -y_limit * 0.80
                y_top = -y_limit * 0.70
                if redcard_icon is not None:
                    ax.imshow(redcard_icon, extent=(card_minute - 0.5, card_minute + 0.5, y_bottom, y_top),
                              aspect='auto', zorder=11)
                else:
                    ax.scatter(card_minute, -y_limit * 0.75, s=220, marker='s', color='red', zorder=11)
                    ax.text(card_minute, -y_limit * 0.75, 'RC', fontsize=9, ha='center', va='center', color='white', fontweight='bold')
        
        ax.grid(True, alpha=0.2, color='white', linewidth=1)
        ax.set_axisbelow(True)
        
        x_padding = 1
        ax.set_xlim(minutes[0] - x_padding, minutes[-1] + x_padding)
        
        step = 5 if (minutes[-1] - minutes[0]) >= 30 else max(1, int((minutes[-1] - minutes[0]) // 5))
        ticks = np.arange(minutes[0], minutes[-1] + 1, step)
        
        if half_name == 'Second Half' and len(minutes) > 0:
            offset = 45 - minutes[0]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(t + offset)}'" for t in ticks])
        else:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(t)}'" for t in ticks])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        
        ax.set_title(half_name, fontsize=14, fontweight='bold', pad=10)
        
        if half_name == 'First Half':
            ax.text(0.02, 0.95, 'Control:', transform=ax.transAxes, fontsize=9, ha='left', va='center')
            
            home_rect = Rectangle((0.12, 0.945), 0.02, 0.012,
                                 facecolor=home_color, alpha=0.6,
                                 transform=ax.transAxes)
            ax.add_patch(home_rect)
            
            away_rect = Rectangle((0.14, 0.945), 0.02, 0.012,
                                facecolor=away_color, alpha=0.6,
                                transform=ax.transAxes)
            ax.add_patch(away_rect)
            
            ax.text(0.02, 0.91, 'Danger:', transform=ax.transAxes, fontsize=9, ha='left', va='center')
            
            ax.plot([0.12, 0.14], [0.91, 0.91],
                   color='red', linewidth=2,
                   linestyle='--', alpha=0.9, transform=ax.transAxes, dashes=(3, 1.5))
            
            ax.plot([0.14, 0.16], [0.91, 0.91],
                   color='black', linewidth=2,
                   linestyle='--', alpha=0.9, transform=ax.transAxes, dashes=(3, 1.5))
    
    # Plot both halves
    plot_half(ax1, first_half_minutes, first_half_home_dom, first_half_away_dom,
              first_half_net_ctrl, first_half_home_ctrl, first_half_away_ctrl,
              first_half_goals, first_half_subs, first_half_cards, 'First Half', 
              home_plot_color, away_plot_color, home_team, away_team)
    
    plot_half(ax2, second_half_minutes, second_half_home_dom, second_half_away_dom,
              second_half_net_ctrl, second_half_home_ctrl, second_half_away_ctrl,
              second_half_goals, second_half_subs, second_half_cards, 'Second Half', 
              home_plot_color, away_plot_color, home_team, away_team)
    
    # Calculate overall percentages
    all_control_events = first_half_control_events + second_half_control_events
    all_domination_events = first_half_domination_events + second_half_domination_events
    all_cards = first_half_cards + second_half_cards
    
    match_start = first_half_start
    match_end = second_half_end
    match_total_duration = max(1.0, (match_end - match_start))
    
    def compute_pct_by_team(events_list, start_time, end_time, teamA, teamB):
        home_sum = sum(e['value'] for e in events_list if e['team'] == teamA and start_time <= e['minute'] < end_time)
        away_sum = sum(e['value'] for e in events_list if e['team'] == teamB and start_time <= e['minute'] < end_time)
        total = home_sum + away_sum
        if total <= 0:
            return 50.0, 50.0
        return (home_sum / total * 100.0), (away_sum / total * 100.0)
    
    # Handle red card splits - sort all cards chronologically and filter those not too close to edges
    if len(all_cards) > 0:
        # Sort cards by minute
        sorted_cards = sorted(all_cards, key=lambda c: c['minute'])
        
        # Filter cards that are not too close to start or end (at least 5 minutes away)
        valid_card_minutes = []
        for card in sorted_cards:
            card_minute = card['minute']
            card_minute = max(match_start, min(card_minute, match_end - 1e-6))
            too_close_to_edges = ((card_minute - match_start) < 5) or ((match_end - card_minute) < 5)
            if not too_close_to_edges:
                valid_card_minutes.append(card_minute)
        
        # Remove cards that are too close to each other (at least 5 minutes apart)
        filtered_card_minutes = []
        for card_minute in valid_card_minutes:
            if not filtered_card_minutes or (card_minute - filtered_card_minutes[-1]) >= 5:
                filtered_card_minutes.append(card_minute)
        
        too_close_to_edges = len(filtered_card_minutes) == 0

        if too_close_to_edges:
            # Standard bars (no split)
            total_home_control_points = sum(e['value'] for e in all_control_events if e['team'] == home_team)
            total_away_control_points = sum(e['value'] for e in all_control_events if e['team'] == away_team)
            total_control_points = total_home_control_points + total_away_control_points
            if total_control_points > 0:
                home_control_pct = (total_home_control_points / total_control_points) * 100
                away_control_pct = (total_away_control_points / total_control_points) * 100
            else:
                home_control_pct = away_control_pct = 50

            total_home_danger_points = sum(e['value'] for e in all_domination_events if e['team'] == home_team)
            total_away_danger_points = sum(e['value'] for e in all_domination_events if e['team'] == away_team)
            total_danger_points = total_home_danger_points + total_away_danger_points
            if total_danger_points > 0:
                home_danger_pct = (total_home_danger_points / total_danger_points) * 100
                away_danger_pct = (total_away_danger_points / total_danger_points) * 100
            else:
                home_danger_pct = away_danger_pct = 50

            ax_bar.barh([1], [home_control_pct], color=home_plot_color, alpha=0.8, height=0.85)
            ax_bar.barh([1], [away_control_pct], left=[home_control_pct], color=away_plot_color, alpha=0.8, height=0.85)
            ax_bar.barh([0], [home_danger_pct], color=home_plot_color, alpha=0.8, height=0.85)
            ax_bar.barh([0], [away_danger_pct], left=[home_danger_pct], color=away_plot_color, alpha=0.8, height=0.85)
            ax_bar.set_xlim(0, 100)
            ax_bar.set_ylim(-0.65, 1.65)
            ax_bar.set_yticks([0, 1])
            ax_bar.set_yticklabels(['Danger', 'Control'], fontsize=12, fontweight='bold')
            ax_bar.set_xticks([])
            ax_bar.set_xlabel('')
            ax_bar.text(home_control_pct/2, 1, f'{home_control_pct:.0f}%',
                        ha='center', va='center', color='white', fontweight='bold', fontsize=13)
            ax_bar.text(home_control_pct + away_control_pct/2, 1, f'{away_control_pct:.0f}%',
                        ha='center', va='center', color='white', fontweight='bold', fontsize=13)
            ax_bar.text(home_danger_pct/2, 0, f'{home_danger_pct:.0f}%',
                        ha='center', va='center', color='white', fontweight='bold', fontsize=13)
            ax_bar.text(home_danger_pct + away_danger_pct/2, 0, f'{away_danger_pct:.0f}%',
                        ha='center', va='center', color='white', fontweight='bold', fontsize=13)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['bottom'].set_visible(False)
        else:
            # Multiple red cards - create segments between each card
            # Compute bar split based on plotted spans of first/second half
            first_half_plotted_duration = (first_half_minutes[-1] - first_half_minutes[0]) if (hasattr(first_half_minutes, 'size') and first_half_minutes.size > 0) else 0
            second_half_plotted_duration = (second_half_minutes[-1] - second_half_minutes[0]) if (hasattr(second_half_minutes, 'size') and second_half_minutes.size > 0) else 0
            total_plotted_span_actual = first_half_plotted_duration + second_half_plotted_duration

            # Create time segments: [match_start, card1, card2, ..., match_end]
            segment_times = [match_start] + filtered_card_minutes + [match_end]
            
            # Calculate percentage position of each card on the bar
            card_split_positions = []
            for card_minute in filtered_card_minutes:
                if total_plotted_span_actual <= 0:
                    split_pct = 50.0
                else:
                    if (card_minute <= first_half_end) and (hasattr(first_half_minutes, 'size') and first_half_minutes.size > 0):
                        time_before_card_on_plot = card_minute - first_half_minutes[0]
                    elif hasattr(second_half_minutes, 'size') and second_half_minutes.size > 0:
                        time_before_card_on_plot = first_half_plotted_duration + (card_minute - second_half_minutes[0])
                    else:
                        time_before_card_on_plot = 0
                    split_pct = (time_before_card_on_plot / total_plotted_span_actual) * 100.0
                    split_pct = float(np.clip(split_pct, 0.0, 100.0))
                card_split_positions.append(split_pct)
            
            # Calculate control and danger percentages for each segment
            segment_data = []
            for i in range(len(segment_times) - 1):
                start_time = segment_times[i]
                end_time = segment_times[i + 1]
                
                ctrl_home_pct, ctrl_away_pct = compute_pct_by_team(all_control_events, start_time, end_time, home_team, away_team)
                dang_home_pct, dang_away_pct = compute_pct_by_team(all_domination_events, start_time, end_time, home_team, away_team)
                
                segment_data.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'ctrl_home_pct': ctrl_home_pct,
                    'ctrl_away_pct': ctrl_away_pct,
                    'dang_home_pct': dang_home_pct,
                    'dang_away_pct': dang_away_pct
                })
            
            # Calculate segment positions on bar (0-100%)
            segment_positions = [0.0] + card_split_positions + [100.0]

            ax_bar.clear()
            ax_bar.set_xlim(0, 100)
            ax_bar.set_ylim(-0.65, 1.65)
            ax_bar.set_yticks([0, 1])
            ax_bar.set_yticklabels(['Danger', 'Control'], fontsize=12, fontweight='bold')
            ax_bar.set_xticks([])

            def maybe_text(x_start, width, y, txt):
                if width >= 3:
                    ax_bar.text(x_start + width / 2.0, y, txt, ha='center', va='center', color='white', fontweight='bold', fontsize=11)

            # Draw each segment
            for i in range(len(segment_data)):
                seg = segment_data[i]
                seg_start_pct = segment_positions[i]
                seg_end_pct = segment_positions[i + 1]
                seg_width = seg_end_pct - seg_start_pct
                
                # Calculate absolute widths for home/away in this segment
                home_ctrl_width = seg['ctrl_home_pct'] / 100.0 * seg_width
                away_ctrl_width = seg['ctrl_away_pct'] / 100.0 * seg_width
                
                home_dang_width = seg['dang_home_pct'] / 100.0 * seg_width
                away_dang_width = seg['dang_away_pct'] / 100.0 * seg_width
                
                # Control row
                ax_bar.barh([1], [home_ctrl_width], left=[seg_start_pct], color=home_plot_color, alpha=0.8, height=0.85)
                ax_bar.barh([1], [away_ctrl_width], left=[seg_start_pct + home_ctrl_width], color=away_plot_color, alpha=0.8, height=0.85)
                
                # Danger row
                ax_bar.barh([0], [home_dang_width], left=[seg_start_pct], color=home_plot_color, alpha=0.8, height=0.85)
                ax_bar.barh([0], [away_dang_width], left=[seg_start_pct + home_dang_width], color=away_plot_color, alpha=0.8, height=0.85)
                
                # Add percentage labels
                maybe_text(seg_start_pct, home_ctrl_width, 1, f"{seg['ctrl_home_pct']:.0f}%")
                maybe_text(seg_start_pct + home_ctrl_width, away_ctrl_width, 1, f"{seg['ctrl_away_pct']:.0f}%")
                maybe_text(seg_start_pct, home_dang_width, 0, f"{seg['dang_home_pct']:.0f}%")
                maybe_text(seg_start_pct + home_dang_width, away_dang_width, 0, f"{seg['dang_away_pct']:.0f}%")
            
            # Draw red card markers at split positions
            for split_pct in card_split_positions:
                ax_bar.axvline(x=split_pct, color='red', linestyle=':', linewidth=1.2, zorder=5)

            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['bottom'].set_visible(False)
    
    else:
        # No red card - standard bars
        total_home_control_points = sum(e['value'] for e in all_control_events if e['team'] == home_team)
        total_away_control_points = sum(e['value'] for e in all_control_events if e['team'] == away_team)
        total_control_points = total_home_control_points + total_away_control_points
        
        if total_control_points > 0:
            home_control_pct = (total_home_control_points / total_control_points) * 100
            away_control_pct = (total_away_control_points / total_control_points) * 100
        else:
            home_control_pct = away_control_pct = 50
        
        total_home_danger_points = sum(e['value'] for e in all_domination_events if e['team'] == home_team)
        total_away_danger_points = sum(e['value'] for e in all_domination_events if e['team'] == away_team)
        total_danger_points = total_home_danger_points + total_away_danger_points
        
        if total_danger_points > 0:
            home_danger_pct = (total_home_danger_points / total_danger_points) * 100
            away_danger_pct = (total_away_danger_points / total_danger_points) * 100
        else:
            home_danger_pct = away_danger_pct = 50
        
        ax_bar.barh([1], [home_control_pct], color=home_plot_color, alpha=0.8, height=0.85)
        ax_bar.barh([1], [away_control_pct], left=[home_control_pct], color=away_plot_color, alpha=0.8, height=0.85)
        
        ax_bar.barh([0], [home_danger_pct], color=home_plot_color, alpha=0.8, height=0.85)
        ax_bar.barh([0], [away_danger_pct], left=[home_danger_pct], color=away_plot_color, alpha=0.8, height=0.85)
        
        ax_bar.set_xlim(0, 100)
        ax_bar.set_ylim(-0.65, 1.65)
        ax_bar.set_yticks([0, 1])
        ax_bar.set_yticklabels(['Danger', 'Control'], fontsize=12, fontweight='bold')
        ax_bar.set_xticks([])
        ax_bar.set_xlabel('')
        
        ax_bar.text(home_control_pct/2, 1, f'{home_control_pct:.0f}%',
                   ha='center', va='center', color='white', fontweight='bold', fontsize=13)
        ax_bar.text(home_control_pct + away_control_pct/2, 1, f'{away_control_pct:.0f}%',
                   ha='center', va='center', color='white', fontweight='bold', fontsize=13)
        
        ax_bar.text(home_danger_pct/2, 0, f'{home_danger_pct:.0f}%',
                   ha='center', va='center', color='white', fontweight='bold', fontsize=13)
        ax_bar.text(home_danger_pct + away_danger_pct/2, 0, f'{away_danger_pct:.0f}%',
                   ha='center', va='center', color='white', fontweight='bold', fontsize=13)
        
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['bottom'].set_visible(False)
    
    # Add score
    total_home_goals = len([g for g in first_half_goals + second_half_goals if g['team'] == home_team])
    total_away_goals = len([g for g in first_half_goals + second_half_goals if g['team'] == away_team])
    
    fig.suptitle(f'{home_team} {total_home_goals} - {total_away_goals} {away_team}',
                 fontsize=18, fontweight='bold', y=1.02)
    
    return fig, {
        'first_half': {
            'minutes': first_half_minutes,
            'net_domination': first_half_net_dom,
            'net_control': first_half_net_ctrl,
            'domination_events': first_half_domination_events,
            'control_events': first_half_control_events,
            'goals': first_half_goals,
            'substitutions': first_half_subs,
            'cards': first_half_cards
        },
        'second_half': {
            'minutes': second_half_minutes,
            'net_domination': second_half_net_dom,
            'net_control': second_half_net_ctrl,
            'domination_events': second_half_domination_events,
            'control_events': second_half_control_events,
            'goals': second_half_goals,
            'substitutions': second_half_subs,
            'cards': second_half_cards
        }
    }

# Main app
if events_data is not None:
    with st.spinner("Analyzing match control..."):
        fig, control_data = calculate_game_control_and_domination(
            events_data, 
            None,
            None
        )

        tab1, tab2, tab9, tab11, tab3, tab8, tab10, tab5, tab6, tab4 = st.tabs(["Controle & Gevaar", "Schoten", "Multi Match Schoten", "Temporary", "xG Verloop", "Voorzetten", "Multi Match Voorzetten", "Gemiddelde Posities", "Samenvatting", "Eredivisie Tabel"])

        with tab1:
            st.pyplot(fig)
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 Download",
                data=buf,
                file_name=f"match_control_{file_name.replace('.json', '')}.png",
                mime="image/png"
            )

        # ---------- Shot Map Tab ----------
        def draw_pitch(ax):
            pitch = patches.Rectangle((-52.5, -34), 105, 68, linewidth=2,
                                     edgecolor='gray', facecolor='white')
            ax.add_patch(pitch)
            ax.plot([0, 0], [-34, 34], 'gray', linestyle='-', linewidth=2)
            center_circle = patches.Circle((0, 0), 9.15, linewidth=2,
                                         edgecolor='gray', fill=False)
            ax.add_patch(center_circle)
            left_penalty = patches.Rectangle((-52.5, -20.16), 16.5, 40.32,
                                           linewidth=2, edgecolor='gray', fill=False)
            ax.add_patch(left_penalty)
            right_penalty = patches.Rectangle((36, -20.16), 16.5, 40.32,
                                            linewidth=2, edgecolor='gray', fill=False)
            ax.add_patch(right_penalty)
            left_goal_area = patches.Rectangle((-52.5, -9.18), 5.5, 18.36,
                                             linewidth=2, edgecolor='gray', fill=False)
            ax.add_patch(left_goal_area)
            right_goal_area = patches.Rectangle((47, -9.18), 5.5, 18.36,
                                              linewidth=2, edgecolor='gray', fill=False)
            ax.add_patch(right_goal_area)
            left_goal = patches.Rectangle((-54.5, -3.66), 2, 7.32,
                                        linewidth=2, edgecolor='gray', fill=False)
            ax.add_patch(left_goal)
            right_goal = patches.Rectangle((52.5, -3.66), 2, 7.32,
                                         linewidth=2, edgecolor='gray', fill=False)
            ax.add_patch(right_goal)

        def find_shot_events(events, team_name=None):
            shot_events = []
            SHOT_LABELS = [128, 143, 144, 142]
            GOAL_LABELS = [146, 147, 148, 149, 150, 151]
            for event in events:
                is_shot = 'shot' in str(event.get('baseTypeName', '')).lower()
                event_labels = event.get('labels', []) or []
                has_shot_label = any(label in event_labels for label in SHOT_LABELS)
                if is_shot or has_shot_label:
                    if team_name is None or event.get('teamName') == team_name:
                        is_goal = any(label in event_labels for label in GOAL_LABELS)
                        # Check if it's a penalty shot and use xG of 0.76
                        is_penalty = (event.get('baseTypeId') == 6 and event.get('subTypeId') == 602)
                        xg_value = 0.76 if is_penalty else event.get('metrics', {}).get('xG', 0.0)
                        
                        shot_info = {
                            'team': event.get('teamName', 'Unknown'),
                            'player': event.get('playerName', 'Unknown'),
                            'x': event.get('startPosXM', 0.0),
                            'y': event.get('startPosYM', 0.0),
                            'xG': xg_value,
                            'PSxG': event.get('metrics', {}).get('PSxG', None),
                            'is_goal': is_goal,
                            'result': event.get('resultName', 'Unknown'),
                            'type': event.get('subTypeName', 'Unknown'),
                            'is_penalty': is_penalty,
                            'time': int((event.get('startTimeMs', 0) or 0) / 1000 / 60),
                            'partId': event.get('partId', 1)
                        }
                        shot_events.append(shot_info)
            return shot_events

        def count_own_goals(events, team_name):
            OWN_GOAL_LABEL = 205
            own_goals = 0
            for event in events:
                event_labels = event.get('labels', []) or []
                if OWN_GOAL_LABEL in event_labels and event.get('teamName') == team_name:
                    own_goals += 1
            return own_goals

        # Removed Schoten tab per request
        # with tab2:
        if False:
            metadata = events_data.get('metaData', {}) if isinstance(events_data, dict) else {}
            home_team = metadata.get('homeTeamName') or metadata.get('homeTeam') or metadata.get('home') or 'Home'
            away_team = metadata.get('awayTeamName') or metadata.get('awayTeam') or metadata.get('away') or 'Away'
            events = events_data.get('data', []) if isinstance(events_data, dict) else []

            all_shots = find_shot_events(events)
            home_shots = [s for s in all_shots if s['team'] == home_team]
            away_shots = [s for s in all_shots if s['team'] == away_team]

            home_own_goals = count_own_goals(events, home_team)
            away_own_goals = count_own_goals(events, away_team)

            def calculate_shot_intervals(shots):
                intervals = [0] * 6
                second_half_start_time = 45
                for event in events:
                    if event.get('baseTypeId') == 14 and event.get('subTypeId') == 1400 and event.get('partId') == 2:
                        second_half_start_time = int((event.get('startTimeMs', 0) or 0) / 1000 / 60)
                        break
                for shot in shots:
                    minute = shot['time']
                    part_id = shot['partId']
                    if part_id == 1:
                        if minute < 15:
                            intervals[0] += 1
                        elif minute < 30:
                            intervals[1] += 1
                        else:
                            intervals[2] += 1
                    elif part_id == 2:
                        relative_minute = minute - second_half_start_time
                        if relative_minute < 15:
                            intervals[3] += 1
                        elif relative_minute < 30:
                            intervals[4] += 1
                        else:
                            intervals[5] += 1
                return intervals

            home_shot_intervals = calculate_shot_intervals(home_shots)
            away_shot_intervals = calculate_shot_intervals(away_shots)
            max_shots = max(max(home_shot_intervals) if home_shot_intervals else 0,
                            max(away_shot_intervals) if away_shot_intervals else 0)

            fig_shots = plt.figure(figsize=(22, 12))
            gs2 = gridspec.GridSpec(1, 3, width_ratios=[0.7, 3, 0.7], wspace=0.1)
            ax_home_bars = fig_shots.add_subplot(gs2[0])
            ax_pitch = fig_shots.add_subplot(gs2[1])
            ax_away_bars = fig_shots.add_subplot(gs2[2])
            draw_pitch(ax_pitch)

            stats = {
                home_team: {'shots': 0, 'goals': 0, 'xG': 0.0, 'PSxG': 0.0, 'shots_on_target': 0, 'penalties': 0, 'pen_xG': 0.0, 'pen_PSxG': 0.0},
                away_team: {'shots': 0, 'goals': 0, 'xG': 0.0, 'PSxG': 0.0, 'shots_on_target': 0, 'penalties': 0, 'pen_xG': 0.0, 'pen_PSxG': 0.0}
            }

            for shot in home_shots:
                # Always flip home team shots so they appear on the left
                x = -shot['x']; y = -shot['y']
                marker_size = 50 + (shot['xG'] * 450)
                if shot['is_goal']:
                    face_color = home_color; edge_color = home_color; edge_width = 2; stats[home_team]['goals'] += 1
                else:
                    face_color = 'white'; edge_color = home_color; edge_width = 2
                ax_pitch.scatter(x, y, s=marker_size, c=face_color, alpha=1.0,
                                 edgecolors=edge_color, linewidths=edge_width, zorder=5)
                stats[home_team]['shots'] += 1
                if shot.get('is_penalty'):
                    stats[home_team]['penalties'] += 1
                    stats[home_team]['pen_xG'] += shot['xG']
                    if shot['PSxG']:
                        stats[home_team]['pen_PSxG'] += shot['PSxG']
                        stats[home_team]['shots_on_target'] += 1
                else:
                    stats[home_team]['xG'] += shot['xG']
                    if shot['PSxG']:
                        stats[home_team]['PSxG'] += shot['PSxG']
                        stats[home_team]['shots_on_target'] += 1

            for shot in away_shots:
                # Away shots remain as-is (shown on the right)
                x = shot['x']; y = shot['y']
                marker_size = 50 + (shot['xG'] * 450)
                if shot['is_goal']:
                    face_color = away_color; edge_color = away_color; edge_width = 2; stats[away_team]['goals'] += 1
                else:
                    face_color = 'white'; edge_color = away_color; edge_width = 2
                ax_pitch.scatter(x, y, s=marker_size, c=face_color, alpha=1.0,
                                 edgecolors=edge_color, linewidths=edge_width, zorder=5)
                stats[away_team]['shots'] += 1
                if shot.get('is_penalty'):
                    stats[away_team]['penalties'] += 1
                    stats[away_team]['pen_xG'] += shot['xG']
                    if shot['PSxG']:
                        stats[away_team]['pen_PSxG'] += shot['PSxG']
                        stats[away_team]['shots_on_target'] += 1
                else:
                    stats[away_team]['xG'] += shot['xG']
                    if shot['PSxG']:
                        stats[away_team]['PSxG'] += shot['PSxG']
                        stats[away_team]['shots_on_target'] += 1

            home_total_goals = stats[home_team]['goals'] + away_own_goals
            away_total_goals = stats[away_team]['goals'] + home_own_goals

            # Label adjustments if penalties exist
            has_penalties = (stats[home_team]['penalties'] > 0) or (stats[away_team]['penalties'] > 0)
            xg_label = "xG (zonder penalty's)" if has_penalties else 'xG'
            xgot_label = "xGOT (zonder penalty's)" if has_penalties else 'xGOT'
            stats_labels = ['Doelpunten', 'Schoten', 'Schoten op doel', xg_label, xgot_label]
            home_values = [
                f"{home_total_goals}",
                f"{stats[home_team]['shots']}",
                f"{stats[home_team]['shots_on_target']}",
                f"{stats[home_team]['xG']:.2f}",
                f"{stats[home_team]['PSxG']:.2f}",
            ]
            away_values = [
                f"{away_total_goals}",
                f"{stats[away_team]['shots']}",
                f"{stats[away_team]['shots_on_target']}",
                f"{stats[away_team]['xG']:.2f}",
                f"{stats[away_team]['PSxG']:.2f}",
            ]

            # Append penalties xG row only if present
            if has_penalties:
                stats_labels.append("xG uit penalty's")
                home_values.append(f"{stats[home_team]['pen_xG']:.2f}")
                away_values.append(f"{stats[away_team]['pen_xG']:.2f}")

            ax_pitch.text(-20, 56, home_team, fontsize=14, fontweight='bold', color=home_color, ha='center', va='center')
            ax_pitch.text(20, 56, away_team, fontsize=14, fontweight='bold', color=away_color, ha='center', va='center')

            y_start = 52
            for i, (label, home_val, away_val) in enumerate(zip(stats_labels, home_values, away_values)):
                y_pos = y_start - (i * 3)
                ax_pitch.text(-20, y_pos, home_val, fontsize=12, fontweight='bold', color='black', ha='center', va='center')
                ax_pitch.text(0, y_pos, label, fontsize=10, color='gray', ha='center', va='center')
                ax_pitch.text(20, y_pos, away_val, fontsize=12, fontweight='bold', color='black', ha='center', va='center')
        with tab2:
            metadata = events_data.get('metaData', {}) if isinstance(events_data, dict) else {}
            home_team = metadata.get('homeTeamName') or metadata.get('homeTeam') or metadata.get('home') or 'Home'
            away_team = metadata.get('awayTeamName') or metadata.get('awayTeam') or metadata.get('away') or 'Away'
            events = events_data.get('data', []) if isinstance(events_data, dict) else []

            all_shots = find_shot_events(events)
            home_shots = [s for s in all_shots if s['team'] == home_team]
            away_shots = [s for s in all_shots if s['team'] == away_team]

            # Calculate shot intervals (duplicate from Schoten tab)
            def calculate_shot_intervals_imp(shots):
                intervals = [0] * 6
                second_half_start_time = 45
                for event in events:
                    if event.get('baseTypeId') == 14 and event.get('subTypeId') == 1400 and event.get('partId') == 2:
                        second_half_start_time = int((event.get('startTimeMs', 0) or 0) / 1000 / 60)
                        break
                for shot in shots:
                    minute = shot['time']
                    part_id = shot['partId']
                    if part_id == 1:
                        if minute < 15:
                            intervals[0] += 1
                        elif minute < 30:
                            intervals[1] += 1
                        else:
                            intervals[2] += 1
                    elif part_id == 2:
                        relative_minute = minute - second_half_start_time
                        if relative_minute < 15:
                            intervals[3] += 1
                        elif relative_minute < 30:
                            intervals[4] += 1
                        else:
                            intervals[5] += 1
                return intervals

            home_shot_intervals = calculate_shot_intervals_imp(home_shots)
            away_shot_intervals = calculate_shot_intervals_imp(away_shots)
            max_shots = max(max(home_shot_intervals) if home_shot_intervals else 0,
                            max(away_shot_intervals) if away_shot_intervals else 0)

            # Build figure layout with mplsoccer pitch only (remove second/original graph)
            fig_shots_imp = plt.figure(figsize=(22, 12))
            # Increase pitch size and slightly reduce spacing from bars
            gs_imp = gridspec.GridSpec(1, 3, width_ratios=[0.55, 3.6, 0.55], wspace=0.13)
            ax_home_bars_imp = fig_shots_imp.add_subplot(gs_imp[0])
            ax_pitch_imp = fig_shots_imp.add_subplot(gs_imp[1])
            ax_away_bars_imp = fig_shots_imp.add_subplot(gs_imp[2])
            # Slightly shift bar axes upward so their centers align better with pitch center
            try:
                pos_left = ax_home_bars_imp.get_position()
                ax_home_bars_imp.set_position([pos_left.x0, pos_left.y0 + 0.0475, pos_left.width, pos_left.height])
                pos_right = ax_away_bars_imp.get_position()
                ax_away_bars_imp.set_position([pos_right.x0, pos_right.y0 + 0.0475, pos_right.width, pos_right.height])
            except Exception:
                pass
            # Draw mplsoccer pitch (default size), spacing handled by GridSpec wspace
            Pitch(pitch_type='impect').draw(ax=ax_pitch_imp)

            # Stats aggregation identical to Schoten tab (already excludes penalties from xG/xGOT)
            stats = {
                home_team: {'shots': 0, 'goals': 0, 'xG': 0.0, 'PSxG': 0.0, 'shots_on_target': 0, 'penalties': 0, 'pen_xG': 0.0, 'pen_PSxG': 0.0},
                away_team: {'shots': 0, 'goals': 0, 'xG': 0.0, 'PSxG': 0.0, 'shots_on_target': 0, 'penalties': 0, 'pen_xG': 0.0, 'pen_PSxG': 0.0}
            }

            # Plot shots on impect pitch and update stats
            for shot in home_shots:
                # Always flip home shots to left on impect pitch
                x = -shot['x']; y = -shot['y']
                marker_size = 50 + (shot['xG'] * 450)
                if shot['is_goal']:
                    face_color = home_color; edge_color = home_color; edge_width = 2; stats[home_team]['goals'] += 1
                else:
                    face_color = 'white'; edge_color = home_color; edge_width = 2
                ax_pitch_imp.scatter(x, y, s=marker_size, c=face_color, alpha=1.0,
                                     edgecolors=edge_color, linewidths=edge_width, zorder=5)
                stats[home_team]['shots'] += 1
                if shot.get('is_penalty'):
                    stats[home_team]['penalties'] += 1
                    stats[home_team]['pen_xG'] += shot['xG']
                    if shot['PSxG']:
                        stats[home_team]['pen_PSxG'] += shot['PSxG']
                        stats[home_team]['shots_on_target'] += 1
                else:
                    stats[home_team]['xG'] += shot['xG']
                    if shot['PSxG']:
                        stats[home_team]['PSxG'] += shot['PSxG']
                        stats[home_team]['shots_on_target'] += 1

            for shot in away_shots:
                # Away shots stay as-is on impect pitch
                x = shot['x']; y = shot['y']
                marker_size = 50 + (shot['xG'] * 450)
                if shot['is_goal']:
                    face_color = away_color; edge_color = away_color; edge_width = 2; stats[away_team]['goals'] += 1
                else:
                    face_color = 'white'; edge_color = away_color; edge_width = 2
                ax_pitch_imp.scatter(x, y, s=marker_size, c=face_color, alpha=1.0,
                                     edgecolors=edge_color, linewidths=edge_width, zorder=5)
                stats[away_team]['shots'] += 1
                if shot.get('is_penalty'):
                    stats[away_team]['penalties'] += 1
                    stats[away_team]['pen_xG'] += shot['xG']
                    if shot['PSxG']:
                        stats[away_team]['pen_PSxG'] += shot['PSxG']
                        stats[away_team]['shots_on_target'] += 1
                else:
                    stats[away_team]['xG'] += shot['xG']
                    if shot['PSxG']:
                        stats[away_team]['PSxG'] += shot['PSxG']
                        stats[away_team]['shots_on_target'] += 1

            home_total_goals = stats[home_team]['goals'] + count_own_goals(events, home_team)
            away_total_goals = stats[away_team]['goals'] + count_own_goals(events, away_team)

            has_penalties = (stats[home_team]['penalties'] > 0) or (stats[away_team]['penalties'] > 0)
            xg_label = "xG (zonder penalty's)" if has_penalties else 'xG'
            xgot_label = "xGOT (zonder penalty's)" if has_penalties else 'xGOT'
            stats_labels = ['Doelpunten', 'Schoten', 'Schoten op doel', xg_label, xgot_label]
            home_values = [
                f"{home_total_goals}",
                f"{stats[home_team]['shots']}",
                f"{stats[home_team]['shots_on_target']}",
                f"{stats[home_team]['xG']:.2f}",
                f"{stats[home_team]['PSxG']:.2f}",
            ]
            away_values = [
                f"{away_total_goals}",
                f"{stats[away_team]['shots']}",
                f"{stats[away_team]['shots_on_target']}",
                f"{stats[away_team]['xG']:.2f}",
                f"{stats[away_team]['PSxG']:.2f}",
            ]
            if has_penalties:
                stats_labels.append("xG uit penalty's")
                home_values.append(f"{stats[home_team]['pen_xG']:.2f}")
                away_values.append(f"{stats[away_team]['pen_xG']:.2f}")

            # Render stats texts in center panel
            ax_pitch_imp.text(-20, 56, home_team, fontsize=14, fontweight='bold', color=home_color, ha='center', va='center')
            ax_pitch_imp.text(20, 56, away_team, fontsize=14, fontweight='bold', color=away_color, ha='center', va='center')
            y_start = 52
            for i, (label, home_val, away_val) in enumerate(zip(stats_labels, home_values, away_values)):
                # Slightly smaller vertical spacing between rows vs previous
                y_pos = y_start - (i * 2.5)
                ax_pitch_imp.text(-20, y_pos, home_val, fontsize=12, fontweight='bold', color='black', ha='center', va='center')
                ax_pitch_imp.text(0, y_pos, label, fontsize=12, color='gray', ha='center', va='center')
                ax_pitch_imp.text(20, y_pos, away_val, fontsize=12, fontweight='bold', color='black', ha='center', va='center')

            # Side bar charts
            y_pos = np.arange(6)
            bar_height = 0.6
            ax_home_bars_imp.barh(y_pos, home_shot_intervals, bar_height, color=home_color, alpha=1)
            ax_home_bars_imp.set_yticks(y_pos)
            ax_home_bars_imp.set_yticklabels(["0-15'", "15-30'", "30-45+'", "45-60'", "60-75'", "75-90+'"]) 
            ax_home_bars_imp.invert_xaxis()
            ax_home_bars_imp.set_xlabel("Aantal schoten")
            for spine in ['right','top','bottom','left']:
                ax_home_bars_imp.spines[spine].set_visible(False)
            ax_home_bars_imp.yaxis.tick_right()
            ax_home_bars_imp.tick_params(axis='y', which='major', pad=10)
            ax_home_bars_imp.set_xlim(max_shots + 0.5, 0)
            # Dashed guides at every tick (same logic as original)
            xticks = ax_home_bars_imp.get_xticks()
            ymin = y_pos[0] - bar_height * 0.6
            ymax = y_pos[-1] + bar_height * 1.2
            for tx in xticks:
                ax_home_bars_imp.vlines(x=tx, ymin=ymin, ymax=ymax, linestyles='--', colors='gray', linewidth=0.7, zorder=0, alpha=0.55)

            ax_away_bars_imp.barh(y_pos, away_shot_intervals, bar_height, color=away_color, alpha=1)
            ax_away_bars_imp.set_yticks(y_pos)
            ax_away_bars_imp.set_yticklabels(["0-15'", "15-30'", "30-45+'", "45-60'", "60-75'", "75-90+'"]) 
            ax_away_bars_imp.set_xlabel("Aantal schoten")
            for spine in ['left','top','bottom','right']:
                ax_away_bars_imp.spines[spine].set_visible(False)
            ax_away_bars_imp.yaxis.tick_left()
            ax_away_bars_imp.tick_params(axis='y', which='major', pad=10)
            ax_away_bars_imp.set_xlim(0, max_shots + 0.5)
            # Dashed guides at every tick (same logic as original)
            xticks = ax_away_bars_imp.get_xticks()
            ymin = y_pos[0] - bar_height * 0.6
            ymax = y_pos[-1] + bar_height * 1.2
            for tx in xticks:
                ax_away_bars_imp.vlines(x=tx, ymin=ymin, ymax=ymax, linestyles='--', colors='gray', linewidth=0.7, zorder=0, alpha=0.55)

            # xG scale guide (position relative to current pitch limits)
            # Ensure there is space below the pitch for xG scale
            xmin, xmax = ax_pitch_imp.get_xlim()
            ymin_data, ymax_data = ax_pitch_imp.get_ylim()
            height = ymax_data - ymin_data
            extra_space = height * 0.08
            ax_pitch_imp.set_ylim(ymin_data - extra_space, ymax_data)
            # Recompute with updated limits
            xmin, xmax = ax_pitch_imp.get_xlim()
            ymin_data, ymax_data = ax_pitch_imp.get_ylim()
            x_center = (xmin + xmax) / 2
            width = xmax - xmin
            height = ymax_data - ymin_data
            # Position title/dots/text below pitch with adjusted spacing
            # Move title slightly up, ticks (labels) slightly down
            title_y = ymin_data + height * 0.07
            dots_y = ymin_data + height * 0.035
            labels_y = ymin_data + height * 0.004
            ax_pitch_imp.text(x_center, title_y, "xG Schaal", fontsize=10, fontweight='bold', ha='center', va='center')
            scale_xg_values = [0.1, 0.3, 0.5, 0.7, 0.9]
            # place dots centered under title
            dot_spacing = width * 0.095
            start_x = x_center - dot_spacing * 2
            for i, xg in enumerate(scale_xg_values):
                size = 50 + (xg * 450)
                x_pos = start_x + (i * dot_spacing)
                ax_pitch_imp.scatter(x_pos, dots_y, s=size, c='white', alpha=1, edgecolors='black', linewidths=2, zorder=6, clip_on=False)
                ax_pitch_imp.text(x_pos, labels_y, f'{xg:.1f}', ha='center', va='center', fontsize=8, zorder=6, clip_on=False)

            plt.tight_layout()
            st.pyplot(fig_shots_imp)

        # ---------- Multi Match Schoten Tab ----------
        with tab9:
            st.subheader("🎯 Multi Match Schoten")
            
            # Allow user to select multiple matches
            if team_matches and len(team_matches) > 0:
                match_labels_multi = [info['label'] for info in team_matches]
                selected_multi_matches = st.multiselect(
                    "Selecteer wedstrijden voor analyse",
                    match_labels_multi,
                    default=[team_matches[0]['label']] if len(team_matches) > 0 else []
                )
                
                if selected_multi_matches and len(selected_multi_matches) > 0:
                    # Load all selected matches
                    all_multi_shots_for = []  # Shots by selected team
                    all_multi_shots_against = []  # Shots against selected team
                    
                    for match_label in selected_multi_matches:
                        match_info = next((m for m in team_matches if m['label'] == match_label), None)
                        if match_info:
                            try:
                                match_data = load_json_lenient(match_info['path'])
                                events = match_data.get('data', []) if isinstance(match_data, dict) else []
                                metadata = match_data.get('metaData', {}) if isinstance(match_data, dict) else {}
                                
                                # Find second half start for THIS match
                                second_half_start_time = 45
                                for event in events:
                                    if event.get('baseTypeId') == 14 and event.get('subTypeId') == 1400 and event.get('partId') == 2:
                                        second_half_start_time = int((event.get('startTimeMs', 0) or 0) / 1000 / 60)
                                        break
                                
                                # Find all shots and store the second half start time with each shot
                                all_shots_m = find_shot_events(events)
                                for shot in all_shots_m:
                                    # Add the match's second half start time to the shot
                                    shot['match_second_half_start'] = second_half_start_time
                                    
                                    if shot['team'] == selected_team:
                                        all_multi_shots_for.append(shot)
                                    else:
                                        all_multi_shots_against.append(shot)
                            except Exception as e:
                                st.warning(f"Error loading {match_label}: {e}")
                    
                    # Calculate shot intervals and xG per interval - each shot uses its own match's second half start
                    def calculate_multi_shot_intervals(shots):
                        shot_intervals = [0] * 6
                        xg_intervals = [0.0] * 6
                        for shot in shots:
                            minute = int((shot.get('time', 0) or 0))
                            part_id = shot.get('partId', 1)
                            xg_value = shot.get('xG', 0.0)
                            second_half_start_time = shot.get('match_second_half_start', 45)
                            
                            if part_id == 1:
                                if minute < 15:
                                    shot_intervals[0] += 1
                                    xg_intervals[0] += xg_value
                                elif minute < 30:
                                    shot_intervals[1] += 1
                                    xg_intervals[1] += xg_value
                                else:
                                    shot_intervals[2] += 1
                                    xg_intervals[2] += xg_value
                            elif part_id == 2:
                                # Use THIS shot's match-specific second half start time
                                relative_minute = minute - second_half_start_time
                                if relative_minute < 15:
                                    shot_intervals[3] += 1
                                    xg_intervals[3] += xg_value
                                elif relative_minute < 30:
                                    shot_intervals[4] += 1
                                    xg_intervals[4] += xg_value
                                else:
                                    shot_intervals[5] += 1
                                    xg_intervals[5] += xg_value
                        return shot_intervals, xg_intervals
                    
                    for_intervals, for_xg_intervals = calculate_multi_shot_intervals(all_multi_shots_for)
                    against_intervals, against_xg_intervals = calculate_multi_shot_intervals(all_multi_shots_against)
                    
                    # --- Figure 1: Shots For ---
                    st.subheader(f"Schoten van {selected_team}")
                    fig_for = plt.figure(figsize=(18, 10))
                    gs_for = gridspec.GridSpec(1, 3, width_ratios=[0.8, 2.5, 1.2], wspace=0.15)
                    ax_bars_for = fig_for.add_subplot(gs_for[0])
                    ax_pitch_for = fig_for.add_subplot(gs_for[1])
                    ax_stats_for = fig_for.add_subplot(gs_for[2])
                    
                    # Draw pitch
                    pitch = VerticalPitch(pitch_type='impect', pitch_color='white', line_color='gray',
                                         linewidth=2, half=True, pad_bottom=0)
                    pitch.draw(ax=ax_pitch_for)
                    
                    # Add title
                    ax_pitch_for.set_title(f"{selected_team} - Schoten", fontsize=14, fontweight='bold', pad=10)
                    
                    # Plot shots
                    import math
                    for shot in all_multi_shots_for:
                        sx = shot.get('x', 0.0)
                        sy = shot.get('y', 0.0)
                        if sx < 0:
                            x = -sx
                            y = -sy
                        else:
                            x = sx
                            y = sy
                        
                        marker_size = 50 + (shot.get('xG', 0.0) * 500)
                        
                        if shot.get('is_goal'):
                            face_color = home_color
                            edge_color = home_color
                            alpha = 1.0
                            edge_width = 2
                            zorder = 10
                        else:
                            face_color = 'white'
                            edge_color = home_color
                            alpha = 1.0
                            edge_width = 2
                            zorder = 5
                        
                        pitch.scatter(x, y, s=marker_size, c=face_color,
                                     alpha=alpha, edgecolors=edge_color,
                                     linewidths=edge_width, zorder=zorder, ax=ax_pitch_for)
                    
                    # Bars (left)
                    y_pos = np.arange(6)
                    bar_height = 0.7
                    interval_labels = ['0-15\'', '15-30\'', '30-45+\'', '45-60\'', '60-75\'', '75-90+\'']
                    bars = ax_bars_for.barh(y_pos, for_intervals, bar_height, color=home_color, alpha=1, zorder=5)
                    ax_bars_for.set_yticks(y_pos)
                    ax_bars_for.set_yticklabels(interval_labels)
                    ax_bars_for.set_xlabel("Aantal schoten")
                    
                    for spine in ['left','top','bottom','right']:
                        ax_bars_for.spines[spine].set_visible(False)
                    ax_bars_for.yaxis.tick_left()
                    ax_bars_for.tick_params(axis='y', which='major', pad=10)
                    
                    max_shots_for = max(for_intervals) if for_intervals else 0
                    n_intervals = 4
                    step = max(1, int(math.ceil((max_shots_for + 1) / float(n_intervals))))
                    last_tick = n_intervals * step
                    xticks = np.arange(0, last_tick + 1, step)
                    ax_bars_for.set_xlim(0, last_tick)
                    ax_bars_for.set_xticks(xticks)
                    ax_bars_for.set_xticklabels([str(int(t)) for t in xticks])
                    
                    ymin = y_pos[0] - bar_height * 0.6
                    ymax = y_pos[-1] + bar_height * 1.2
                    for tx in xticks:
                        ax_bars_for.vlines(x=tx, ymin=ymin, ymax=ymax, linestyles='--',
                                          colors='gray', linewidth=0.7, zorder=0, alpha=0.55)
                    
                    # Add xG text to bars
                    for i, (bar, xg_val) in enumerate(zip(bars, for_xg_intervals)):
                        if xg_val > 0:
                            bar_width = bar.get_width()
                            bar_y = bar.get_y() + bar.get_height() / 2
                            xg_text = f'{xg_val:.2f} xG'
                            
                            # Estimate text width (rough approximation: 6 units per character)
                            text_width = len(xg_text) * 0.5
                            
                            # Check if text fits horizontally (bar width > text width + margin)
                            if bar_width > text_width + 0.5:
                                # Horizontal text
                                ax_bars_for.text(bar_width / 2, bar_y, xg_text,
                                               ha='center', va='center', color='white',
                                               fontsize=9, fontweight='bold', zorder=10)
                            else:
                                # Vertical text
                                ax_bars_for.text(bar_width / 2, bar_y, xg_text,
                                               ha='center', va='center', color='white',
                                               fontsize=9, fontweight='bold', rotation=90, zorder=10)
                    
                    # xG scale under pitch (moved lower)
                    title_axes_y = -0.08
                    scatter_axes_y = -0.12
                    scale_xg_values = [0.1, 0.3, 0.5, 0.7, 0.9]
                    n = len(scale_xg_values)
                    spacing = 0.15
                    total_width = spacing * (n - 1)
                    scale_x_start = 0.5 - total_width / 2.0
                    
                    ax_pitch_for.text(0.5, title_axes_y, 'xG Schaal', fontsize=10, fontweight='bold',
                                     ha='center', transform=ax_pitch_for.transAxes)
                    
                    for i, xg in enumerate(scale_xg_values):
                        scale_marker_size = 50 + (xg * 500)
                        adjusted_scale_marker_size = scale_marker_size + 20
                        x_pos = scale_x_start + (i * spacing)
                        ax_pitch_for.scatter([x_pos], [scatter_axes_y], s=adjusted_scale_marker_size, c='white', alpha=1,
                                            edgecolors='black', linewidths=2, clip_on=False,
                                            transform=ax_pitch_for.transAxes, zorder=20)
                        ax_pitch_for.text(x_pos, scatter_axes_y - 0.06, f'{xg:.1f}', ha='center',
                                         transform=ax_pitch_for.transAxes, fontsize=8)
                    
                    # Statistics table (right)
                    ax_stats_for.axis('off')
                    num_matches = len(selected_multi_matches)
                    
                    goals_for = sum(1 for s in all_multi_shots_for if s['is_goal'])
                    shots_for = len(all_multi_shots_for)
                    on_target_for = sum(1 for s in all_multi_shots_for if s['PSxG'])
                    xg_for = sum(s['xG'] for s in all_multi_shots_for if not s.get('is_penalty'))
                    xgot_for = sum(s['PSxG'] for s in all_multi_shots_for if s['PSxG'] and not s.get('is_penalty'))
                    penalties_for = sum(1 for s in all_multi_shots_for if s.get('is_penalty'))
                    penalty_goals_for = sum(1 for s in all_multi_shots_for if s.get('is_penalty') and s['is_goal'])
                    
                    avg_goals = goals_for / num_matches
                    avg_shots = shots_for / num_matches
                    avg_on_target = on_target_for / num_matches
                    avg_xg = xg_for / num_matches
                    avg_xgot = xgot_for / num_matches
                    
                    stats_data = [
                        ('', '', ''),
                        ('', 'Totaal', 'Per wedstrijd'),
                        ("Doelpunten\n(waarvan penalty's)", f'{int(round(goals_for))}({penalty_goals_for})', f'{avg_goals:.2f}'),
                        ('Schoten', f'{shots_for}', f'{avg_shots:.1f}'),
                        ('Schoten op doel', f'{on_target_for}', f'{avg_on_target:.1f}'),
                        ("xG\n(zonder penalty's)", f'{xg_for:.2f}', f'{avg_xg:.2f}'),
                        ("xGOT\n(zonder penalty's)", f'{xgot_for:.2f}', f'{avg_xgot:.2f}'),
                    ]
                    
                    table_y = 0.95
                    table_step = 0.08
                    for idx, row in enumerate(stats_data):
                        if len(row) == 3:
                            # For xG/xGOT rows (with newline), align values with top of text (where "xG" is)
                            if '\n' in row[0]:
                                # Split the label to show main part normal, comment smaller
                                parts = row[0].split('\n')
                                ax_stats_for.text(0.05, table_y, parts[0], ha='left', fontsize=10,
                                                 transform=ax_stats_for.transAxes, fontweight='normal', va='top')
                                if len(parts) > 1:
                                    ax_stats_for.text(0.05, table_y - table_step * 0.4, parts[1], ha='left', fontsize=8,
                                                     transform=ax_stats_for.transAxes, fontweight='normal', va='top', color='gray')
                                # Values at same y position as label (top-aligned)
                                ax_stats_for.text(0.50, table_y, row[1], ha='center', fontsize=10,
                                                 transform=ax_stats_for.transAxes, fontweight='bold', va='top')
                                ax_stats_for.text(0.85, table_y, row[2], ha='center', fontsize=10,
                                                 transform=ax_stats_for.transAxes, fontweight='bold', va='top')
                            else:
                                ax_stats_for.text(0.05, table_y, row[0], ha='left', fontsize=10,
                                                 transform=ax_stats_for.transAxes, fontweight='bold' if row[0] == '' else 'normal')
                                ax_stats_for.text(0.50, table_y, row[1], ha='center', fontsize=10,
                                                 transform=ax_stats_for.transAxes, fontweight='bold')
                                ax_stats_for.text(0.85, table_y, row[2], ha='center', fontsize=10,
                                                 transform=ax_stats_for.transAxes, fontweight='bold')
                        table_y -= table_step
                    
                    plt.tight_layout()
                    st.pyplot(fig_for)
                    
                    # --- Figure 2: Shots Against ---
                    st.subheader(f"Schoten tegen {selected_team}")
                    fig_against = plt.figure(figsize=(18, 10))
                    gs_against = gridspec.GridSpec(1, 3, width_ratios=[0.8, 2.5, 1.2], wspace=0.15)
                    ax_bars_against = fig_against.add_subplot(gs_against[0])
                    ax_pitch_against = fig_against.add_subplot(gs_against[1])
                    ax_stats_against = fig_against.add_subplot(gs_against[2])
                    
                    # Draw pitch
                    pitch_against = VerticalPitch(pitch_type='impect', pitch_color='white', line_color='gray',
                                                  linewidth=2, half=True, pad_bottom=0)
                    pitch_against.draw(ax=ax_pitch_against)
                    
                    # Add title
                    ax_pitch_against.set_title(f"{selected_team} - Schoten Tegen", fontsize=14, fontweight='bold', pad=10)
                    
                    # Plot shots against
                    for shot in all_multi_shots_against:
                        sx = shot.get('x', 0.0)
                        sy = shot.get('y', 0.0)
                        if sx < 0:
                            x = -sx
                            y = -sy
                        else:
                            x = sx
                            y = sy
                        
                        marker_size = 50 + (shot.get('xG', 0.0) * 500)
                        
                        if shot.get('is_goal'):
                            face_color = away_color
                            edge_color = away_color
                            alpha = 1.0
                            edge_width = 2
                            zorder = 10
                        else:
                            face_color = 'white'
                            edge_color = away_color
                            alpha = 1.0
                            edge_width = 2
                            zorder = 5
                        
                        pitch_against.scatter(x, y, s=marker_size, c=face_color,
                                             alpha=alpha, edgecolors=edge_color,
                                             linewidths=edge_width, zorder=zorder, ax=ax_pitch_against)
                    
                    # Bars (left)
                    bars_against = ax_bars_against.barh(y_pos, against_intervals, bar_height, color=away_color, alpha=1, zorder=5)
                    ax_bars_against.set_yticks(y_pos)
                    ax_bars_against.set_yticklabels(interval_labels)
                    ax_bars_against.set_xlabel("Aantal schoten")
                    
                    for spine in ['left','top','bottom','right']:
                        ax_bars_against.spines[spine].set_visible(False)
                    ax_bars_against.yaxis.tick_left()
                    ax_bars_against.tick_params(axis='y', which='major', pad=10)
                    
                    max_shots_against = max(against_intervals) if against_intervals else 0
                    step_against = max(1, int(math.ceil((max_shots_against + 1) / float(n_intervals))))
                    last_tick_against = n_intervals * step_against
                    xticks_against = np.arange(0, last_tick_against + 1, step_against)
                    ax_bars_against.set_xlim(0, last_tick_against)
                    ax_bars_against.set_xticks(xticks_against)
                    ax_bars_against.set_xticklabels([str(int(t)) for t in xticks_against])
                    
                    for tx in xticks_against:
                        ax_bars_against.vlines(x=tx, ymin=ymin, ymax=ymax, linestyles='--',
                                              colors='gray', linewidth=0.7, zorder=0, alpha=0.55)
                    
                    # Add xG text to bars
                    for i, (bar, xg_val) in enumerate(zip(bars_against, against_xg_intervals)):
                        if xg_val > 0:
                            bar_width = bar.get_width()
                            bar_y = bar.get_y() + bar.get_height() / 2
                            xg_text = f'{xg_val:.2f} xG'
                            
                            # Estimate text width (rough approximation: 6 units per character)
                            text_width = len(xg_text) * 0.5
                            
                            # Check if text fits horizontally (bar width > text width + margin)
                            if bar_width > text_width + 0.5:
                                # Horizontal text
                                ax_bars_against.text(bar_width / 2, bar_y, xg_text,
                                                   ha='center', va='center', color='white',
                                                   fontsize=9, fontweight='bold', zorder=10)
                            else:
                                # Vertical text
                                ax_bars_against.text(bar_width / 2, bar_y, xg_text,
                                                   ha='center', va='center', color='white',
                                                   fontsize=9, fontweight='bold', rotation=90, zorder=10)
                    
                    # xG scale under pitch (moved lower)
                    ax_pitch_against.text(0.5, title_axes_y, 'xG Schaal', fontsize=10, fontweight='bold',
                                         ha='center', transform=ax_pitch_against.transAxes)
                    
                    for i, xg in enumerate(scale_xg_values):
                        scale_marker_size = 50 + (xg * 500)
                        adjusted_scale_marker_size = scale_marker_size + 20
                        x_pos = scale_x_start + (i * spacing)
                        ax_pitch_against.scatter([x_pos], [scatter_axes_y], s=adjusted_scale_marker_size, c='white', alpha=1,
                                                edgecolors='black', linewidths=2, clip_on=False,
                                                transform=ax_pitch_against.transAxes, zorder=20)
                        ax_pitch_against.text(x_pos, scatter_axes_y - 0.06, f'{xg:.1f}', ha='center',
                                             transform=ax_pitch_against.transAxes, fontsize=8)
                    
                    # Statistics table (right) for shots against
                    ax_stats_against.axis('off')
                    
                    goals_against = sum(1 for s in all_multi_shots_against if s['is_goal'])
                    shots_against = len(all_multi_shots_against)
                    on_target_against = sum(1 for s in all_multi_shots_against if s['PSxG'])
                    xg_against = sum(s['xG'] for s in all_multi_shots_against if not s.get('is_penalty'))
                    xgot_against = sum(s['PSxG'] for s in all_multi_shots_against if s['PSxG'] and not s.get('is_penalty'))
                    penalties_against = sum(1 for s in all_multi_shots_against if s.get('is_penalty'))
                    penalty_goals_against = sum(1 for s in all_multi_shots_against if s.get('is_penalty') and s['is_goal'])
                    
                    avg_goals_ag = goals_against / num_matches
                    avg_shots_ag = shots_against / num_matches
                    avg_on_target_ag = on_target_against / num_matches
                    avg_xg_ag = xg_against / num_matches
                    avg_xgot_ag = xgot_against / num_matches
                    
                    stats_data_against = [
                        ('', '', ''),
                        ('', 'Totaal', 'Per wedstrijd'),
                        ("Doelpunten\n(waarvan penalty's)", f'{int(round(goals_against))}({penalty_goals_against})', f'{avg_goals_ag:.2f}'),
                        ('Schoten', f'{shots_against}', f'{avg_shots_ag:.1f}'),
                        ('Schoten op doel', f'{on_target_against}', f'{avg_on_target_ag:.1f}'),
                        ("xG\n(zonder penalty's)", f'{xg_against:.2f}', f'{avg_xg_ag:.2f}'),
                        ("xGOT\n(zonder penalty's)", f'{xgot_against:.2f}', f'{avg_xgot_ag:.2f}'),
                    ]
                    
                    table_y_ag = 0.95
                    for idx, row in enumerate(stats_data_against):
                        if len(row) == 3:
                            # For xG/xGOT rows (with newline), align values with top of text (where "xG" is)
                            if '\n' in row[0]:
                                # Split the label to show main part normal, comment smaller
                                parts = row[0].split('\n')
                                ax_stats_against.text(0.05, table_y_ag, parts[0], ha='left', fontsize=10,
                                                     transform=ax_stats_against.transAxes, fontweight='normal', va='top')
                                if len(parts) > 1:
                                    ax_stats_against.text(0.05, table_y_ag - table_step * 0.4, parts[1], ha='left', fontsize=8,
                                                         transform=ax_stats_against.transAxes, fontweight='normal', va='top', color='gray')
                                # Values at same y position as label (top-aligned)
                                ax_stats_against.text(0.50, table_y_ag, row[1], ha='center', fontsize=10,
                                                     transform=ax_stats_against.transAxes, fontweight='bold', va='top')
                                ax_stats_against.text(0.85, table_y_ag, row[2], ha='center', fontsize=10,
                                                     transform=ax_stats_against.transAxes, fontweight='bold', va='top')
                            else:
                                ax_stats_against.text(0.05, table_y_ag, row[0], ha='left', fontsize=10,
                                                     transform=ax_stats_against.transAxes, fontweight='bold' if row[0] == '' else 'normal')
                                ax_stats_against.text(0.50, table_y_ag, row[1], ha='center', fontsize=10,
                                                     transform=ax_stats_against.transAxes, fontweight='bold')
                                ax_stats_against.text(0.85, table_y_ag, row[2], ha='center', fontsize=10,
                                                     transform=ax_stats_against.transAxes, fontweight='bold')
                        table_y_ag -= table_step
                    
                    plt.tight_layout()
                    st.pyplot(fig_against)
                else:
                    st.info("Selecteer minstens één wedstrijd voor multi-match analyse.")
            else:
                st.info("Geen wedstrijden beschikbaar voor dit team.")

        # ---------- Temporary Tab ----------
        with tab11:
            st.subheader("Temporary Analysis")
            
            # Check if we have team_matches
            try:
                has_team_matches = team_matches and len(team_matches) > 0
            except NameError:
                has_team_matches = False
            
            if has_team_matches:
                match_labels_temp = [info['label'] for info in team_matches]
                selected_temp_matches = st.multiselect(
                    "Selecteer wedstrijden voor temporary analyse",
                    match_labels_temp,
                    default=match_labels_temp if len(match_labels_temp) <= 5 else match_labels_temp[:5]
                )
                
                if selected_temp_matches and len(selected_temp_matches) > 0:
                    # Initialize counters for two time periods
                    stats_0_75 = {
                        'goals_for': 0, 'goals_against': 0,
                        'shots_for': 0, 'shots_against': 0,
                        'xg_for': 0.0, 'xg_against': 0.0
                    }
                    stats_75_plus = {
                        'goals_for': 0, 'goals_against': 0,
                        'shots_for': 0, 'shots_against': 0,
                        'xg_for': 0.0, 'xg_against': 0.0
                    }
                    
                    # Process each match individually
                    for match_label in selected_temp_matches:
                        match_info = next((m for m in team_matches if m['label'] == match_label), None)
                        if match_info:
                            try:
                                match_data = load_json_lenient(match_info['path'])
                                events = match_data.get('data', []) if isinstance(match_data, dict) else []
                                
                                # Find second half start time for THIS match
                                second_half_start_time = 45
                                for event in events:
                                    if event.get('baseTypeId') == 14 and event.get('subTypeId') == 1400 and event.get('partId') == 2:
                                        second_half_start_time = int((event.get('startTimeMs', 0) or 0) / 1000 / 60)
                                        break
                                
                                # Find all shots in this match
                                all_shots_temp = find_shot_events(events)
                                
                                # Calculate the 75th minute for THIS match (45 + 30 minutes into 2nd half)
                                minute_75 = second_half_start_time + 30
                                
                                for shot in all_shots_temp:
                                    minute = shot.get('time', 0)
                                    xg = shot.get('xG', 0.0)
                                    is_goal = shot.get('is_goal', False)
                                    is_for_team = shot.get('team') == selected_team
                                    
                                    # Check if shot is before or after the 75th minute of THIS match
                                    if minute < minute_75:
                                        # 0-75 minutes
                                        if is_for_team:
                                            stats_0_75['shots_for'] += 1
                                            stats_0_75['xg_for'] += xg
                                            if is_goal:
                                                stats_0_75['goals_for'] += 1
                                        else:
                                            stats_0_75['shots_against'] += 1
                                            stats_0_75['xg_against'] += xg
                                            if is_goal:
                                                stats_0_75['goals_against'] += 1
                                    else:
                                        # 75+ minutes
                                        if is_for_team:
                                            stats_75_plus['shots_for'] += 1
                                            stats_75_plus['xg_for'] += xg
                                            if is_goal:
                                                stats_75_plus['goals_for'] += 1
                                        else:
                                            stats_75_plus['shots_against'] += 1
                                            stats_75_plus['xg_against'] += xg
                                            if is_goal:
                                                stats_75_plus['goals_against'] += 1
                                
                            except Exception as e:
                                st.warning(f"Error loading {match_label}: {e}")
                    
                    # Create the visualization
                    fig_temp, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 8))
                    
                    # Top section: 0-75 minutes
                    categories = ['Doelpunten', 'Schoten', 'xG']
                    y_pos = np.arange(len(categories))[::-1]  # Reverse order so Doelpunten is on top
                    bar_height = 0.6
                    bar_length = 100  # All bars same length
                    
                    # For vs Against data for 0-75 (reversed order)
                    for_values_0_75 = [stats_0_75['goals_for'], stats_0_75['shots_for'], stats_0_75['xg_for']]
                    against_values_0_75 = [stats_0_75['goals_against'], stats_0_75['shots_against'], stats_0_75['xg_against']]
                    
                    # Plot 0-75 bars with proportional coloring
                    for i, (for_val, against_val) in enumerate(zip(for_values_0_75, against_values_0_75)):
                        total = for_val + against_val
                        if total > 0:
                            for_proportion = (for_val / total) * bar_length
                            against_proportion = (against_val / total) * bar_length
                            
                            # Draw the bar split proportionally
                            ax_top.barh(y_pos[i], for_proportion, bar_height, 
                                       left=0, color=home_color, alpha=0.8)
                            ax_top.barh(y_pos[i], against_proportion, bar_height, 
                                       left=for_proportion, color=away_color, alpha=0.8)
                            
                            # Add text values inside bars
                            # Format values based on category
                            if i == 2:  # xG - show 2 decimals
                                for_text = f'{for_val:.2f}'
                                against_text = f'{against_val:.2f}'
                            else:  # Goals and Shots - show integers
                                for_text = f'{int(for_val)}'
                                against_text = f'{int(against_val)}'
                            
                            # Add text for "Voor" side
                            if for_proportion > 10:  # Only show if bar is wide enough
                                ax_top.text(for_proportion / 2, y_pos[i], for_text,
                                          ha='center', va='center', color='white',
                                          fontsize=12, fontweight='bold')
                            
                            # Add text for "Tegen" side
                            if against_proportion > 10:  # Only show if bar is wide enough
                                ax_top.text(for_proportion + against_proportion / 2, y_pos[i], against_text,
                                          ha='center', va='center', color='white',
                                          fontsize=12, fontweight='bold')
                        else:
                            # Empty bar if no data
                            ax_top.barh(y_pos[i], bar_length, bar_height, color='lightgray', alpha=0.3)
                    
                    ax_top.set_yticks(y_pos)
                    ax_top.set_yticklabels(categories)
                    ax_top.set_xlim(-15, bar_length)
                    ax_top.set_title("Tot de 75e minuut", fontsize=14, fontweight='bold', pad=5)
                    ax_top.axis('off')
                    ax_top.set_ylim(-0.5, len(categories) - 0.5)
                    
                    # Add y-labels manually with bold font (centered and closer to bars)
                    for i, cat in enumerate(categories):
                        ax_top.text(-7.5, y_pos[i], cat, ha='center', va='center', fontsize=11, fontweight='bold')
                    
                    # Bottom section: 75+ minutes
                    for_values_75_plus = [stats_75_plus['goals_for'], stats_75_plus['shots_for'], stats_75_plus['xg_for']]
                    against_values_75_plus = [stats_75_plus['goals_against'], stats_75_plus['shots_against'], stats_75_plus['xg_against']]
                    
                    # Plot 75+ bars with proportional coloring
                    for i, (for_val, against_val) in enumerate(zip(for_values_75_plus, against_values_75_plus)):
                        total = for_val + against_val
                        if total > 0:
                            for_proportion = (for_val / total) * bar_length
                            against_proportion = (against_val / total) * bar_length
                            
                            # Draw the bar split proportionally
                            ax_bottom.barh(y_pos[i], for_proportion, bar_height, 
                                          left=0, color=home_color, alpha=0.8)
                            ax_bottom.barh(y_pos[i], against_proportion, bar_height, 
                                          left=for_proportion, color=away_color, alpha=0.8)
                            
                            # Add text values inside bars
                            # Format values based on category
                            if i == 2:  # xG - show 2 decimals
                                for_text = f'{for_val:.2f}'
                                against_text = f'{against_val:.2f}'
                            else:  # Goals and Shots - show integers
                                for_text = f'{int(for_val)}'
                                against_text = f'{int(against_val)}'
                            
                            # Add text for "Voor" side
                            if for_proportion > 10:  # Only show if bar is wide enough
                                ax_bottom.text(for_proportion / 2, y_pos[i], for_text,
                                             ha='center', va='center', color='white',
                                             fontsize=12, fontweight='bold')
                            
                            # Add text for "Tegen" side
                            if against_proportion > 10:  # Only show if bar is wide enough
                                ax_bottom.text(for_proportion + against_proportion / 2, y_pos[i], against_text,
                                             ha='center', va='center', color='white',
                                             fontsize=12, fontweight='bold')
                        else:
                            # Empty bar if no data
                            ax_bottom.barh(y_pos[i], bar_length, bar_height, color='lightgray', alpha=0.3)
                    
                    ax_bottom.set_yticks(y_pos)
                    ax_bottom.set_yticklabels(categories)
                    ax_bottom.set_xlim(-15, bar_length)
                    ax_bottom.set_title("Na de 75e minuut", fontsize=14, fontweight='bold', pad=5)
                    ax_bottom.axis('off')
                    ax_bottom.set_ylim(-0.5, len(categories) - 0.5)
                    
                    # Add y-labels manually with bold font (centered and closer to bars)
                    for i, cat in enumerate(categories):
                        ax_bottom.text(-7.5, y_pos[i], cat, ha='center', va='center', fontsize=11, fontweight='bold')
                    
                    # Add indicators for selected team and opponents
                    ax_bottom.text(25, y_pos[0] + 1.2, f'{selected_team}', ha='center', va='center', 
                                  fontsize=11, fontweight='bold', color=home_color)
                    ax_bottom.text(75, y_pos[0] + 1.2, 'Tegenstanders', ha='center', va='center', 
                                  fontsize=11, fontweight='bold', color=away_color)
                    
                    plt.tight_layout()
                    st.pyplot(fig_temp)
                else:
                    st.info("Selecteer minstens één wedstrijd voor temporary analyse.")
            else:
                st.info("Geen wedstrijden beschikbaar voor dit team.")

        # ---------- xG Verloop Tab ----------
        def get_halftime_offset(events):
            """Calculate the time offset for the second half to account for halftime break"""
            first_half_end = 45
            second_half_start = 45
            for event in events:
                if (event.get('baseTypeId') == 14 and
                    event.get('subTypeId') == 1401 and
                    event.get('partId') == 1):
                    first_half_end = event.get('startTimeMs', 0) / 1000 / 60
                if (event.get('baseTypeId') == 14 and
                    event.get('subTypeId') == 1400 and
                    event.get('partId') == 2):
                    second_half_start = event.get('startTimeMs', 0) / 1000 / 60
            offset = second_half_start - first_half_end
            return offset, second_half_start

        def find_shot_events_xg(events, team_name=None):
            """Find all shot events for xG plot with adjusted timing"""
            shot_events = []
            halftime_offset, second_half_start = get_halftime_offset(events)
            SHOT_LABELS = [128, 143, 144, 142]
            GOAL_LABELS = [146, 147, 148, 149, 150, 151]
            for event in events:
                is_shot = 'shot' in str(event.get('baseTypeName', '')).lower()
                event_labels = event.get('labels', []) or []
                has_shot_label = any(label in event_labels for label in SHOT_LABELS)
                if is_shot or has_shot_label:
                    if team_name is None or event.get('teamName') == team_name:
                        is_goal = any(label in event_labels for label in GOAL_LABELS)
                        event_time = event.get('startTimeMs', 0) / 1000 / 60
                        part_id = event.get('partId', 1)
                        if part_id == 2:
                            adjusted_time = event_time - halftime_offset
                        else:
                            adjusted_time = event_time
                        # Check if it's a penalty shot and use xG of 0.76
                        is_penalty = (event.get('baseTypeId') == 6 and event.get('subTypeId') == 602)
                        xg_value = 0.76 if is_penalty else event.get('metrics', {}).get('xG', 0.0)
                        
                        shot_info = {
                            'team': event.get('teamName', 'Unknown'),
                            'player': event.get('playerName', 'Unknown'),
                            'xG': xg_value,
                            'is_goal': is_goal,
                            'time': adjusted_time,
                            'partId': part_id
                        }
                        shot_events.append(shot_info)
            return shot_events

        def count_own_goals_xg(events, team_name):
            """Count own goals with adjusted timing"""
            OWN_GOAL_LABEL = 205
            own_goal_events = []
            halftime_offset, second_half_start = get_halftime_offset(events)
            for event in events:
                event_labels = event.get('labels', []) or []
                if OWN_GOAL_LABEL in event_labels and event.get('teamName') == team_name:
                    event_time = event.get('startTimeMs', 0) / 1000 / 60
                    part_id = event.get('partId', 1)
                    if part_id == 2:
                        adjusted_time = event_time - halftime_offset
                    else:
                        adjusted_time = event_time
                    own_goal_info = {
                        'team': event.get('teamName', 'Unknown'),
                        'player': event.get('playerName', 'Unknown'),
                        'time': adjusted_time,
                        'is_own_goal': True
                    }
                    own_goal_events.append(own_goal_info)
            return own_goal_events

        def simulate_match(home_shots, away_shots, num_simulations=10000):
            """Simulate match outcomes based on xG for each shot"""
            import random
            home_wins = 0
            away_wins = 0
            draws = 0
            for _ in range(num_simulations):
                home_goals_sim = 0
                away_goals_sim = 0
                for shot in home_shots:
                    if random.random() <= shot['xG']:
                        home_goals_sim += 1
                for shot in away_shots:
                    if random.random() <= shot['xG']:
                        away_goals_sim += 1
                if home_goals_sim > away_goals_sim:
                    home_wins += 1
                elif away_goals_sim > home_goals_sim:
                    away_wins += 1
                else:
                    draws += 1
            total_simulations = home_wins + away_wins + draws
            home_win_prob = (home_wins / total_simulations) * 100 if total_simulations > 0 else 0
            draw_prob = (draws / total_simulations) * 100 if total_simulations > 0 else 0
            away_win_prob = (away_wins / total_simulations) * 100 if total_simulations > 0 else 0
            return home_win_prob, draw_prob, away_win_prob

        with tab3:
            # Get teams and events for xG plot
            metadata = events_data.get('metaData', {}) if isinstance(events_data, dict) else {}
            home_team_xg = metadata.get('homeTeamName') or metadata.get('homeTeam') or metadata.get('home') or 'Home'
            away_team_xg = metadata.get('awayTeamName') or metadata.get('awayTeam') or metadata.get('away') or 'Away'
            events_xg = events_data.get('data', []) if isinstance(events_data, dict) else []
            
            # Load team logos
            home_logo = load_team_logo(home_team_xg)
            away_logo = load_team_logo(away_team_xg)

            all_shots_xg = find_shot_events_xg(events_xg)
            home_shots_xg = sorted([s for s in all_shots_xg if s['team'] == home_team_xg], key=lambda x: x['time'])
            away_shots_xg = sorted([s for s in all_shots_xg if s['team'] == away_team_xg], key=lambda x: x['time'])

            home_own_goal_events_xg = count_own_goals_xg(events_xg, home_team_xg)
            away_own_goal_events_xg = count_own_goals_xg(events_xg, away_team_xg)

            def create_cumulative_data(shots, max_time=90):
                times = [0]
                cumulative_xg = [0]
                goals = []
                for shot in shots:
                    times.append(shot['time'])
                    cumulative_xg.append(cumulative_xg[-1] + shot['xG'])
                    if shot['is_goal']:
                        goals.append({
                            'time': shot['time'],
                            'xg': cumulative_xg[-1],
                            'player': shot['player']
                        })
                if times and times[-1] < max_time:
                    times.append(max_time)
                    cumulative_xg.append(cumulative_xg[-1])
                elif not times:
                    times.append(max_time)
                    cumulative_xg.append(0)
                return times, cumulative_xg, goals

            max_time = 90
            if all_shots_xg:
                max_time = max([shot['time'] for shot in all_shots_xg]) + 2
                max_time = max(max_time, 90)

            home_times, home_cumulative, home_goals = create_cumulative_data(home_shots_xg, max_time)
            away_times, away_cumulative, away_goals = create_cumulative_data(away_shots_xg, max_time)

            # Add own goals to goal events
            for og in away_own_goal_events_xg:
                idx = np.searchsorted(home_times, og['time'])
                xg_at_time = home_cumulative[min(idx, len(home_cumulative)-1)] if home_cumulative else 0
                home_goals.append({
                    'time': og['time'],
                    'xg': xg_at_time,
                    'player': f"{og['player']} (OG)"
                })
            for og in home_own_goal_events_xg:
                idx = np.searchsorted(away_times, og['time'])
                xg_at_time = away_cumulative[min(idx, len(away_cumulative)-1)] if away_cumulative else 0
                away_goals.append({
                    'time': og['time'],
                    'xg': xg_at_time,
                    'player': f"{og['player']} (OG)"
                })

            home_goals = sorted(home_goals, key=lambda x: x['time'])
            away_goals = sorted(away_goals, key=lambda x: x['time'])

            home_total_goals = len(home_goals)
            away_total_goals = len(away_goals)
            home_total_xg = home_cumulative[-1] if home_cumulative else 0
            away_total_xg = away_cumulative[-1] if away_cumulative else 0

            home_win_prob, draw_prob, away_win_prob = simulate_match(home_shots_xg, away_shots_xg)

            fig_xg = plt.figure(figsize=(14, 12))
            gs_xg = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.1)
            ax_prob = fig_xg.add_subplot(gs_xg[0])
            ax_plot = fig_xg.add_subplot(gs_xg[1])

            # Plot probability bar (much smaller)
            prob_bar_height = 0.08  # Much shorter bar
            y_pos = [0]
            # Scale bar width to 50% of original (25-75 instead of 0-100)
            bar_width_scale = 0.5
            home_width_scaled = home_win_prob * bar_width_scale
            draw_width_scaled = draw_prob * bar_width_scale
            away_width_scaled = away_win_prob * bar_width_scale
            
            # Center the bar at x=50 (accounting for y-axis labels in the plot below)
            bar_start = 50 - (50 * bar_width_scale)  # 25
            bar_end = 50 + (50 * bar_width_scale)    # 75
            
            ax_prob.barh(y_pos, [home_width_scaled], left=[bar_start], color=home_color, height=prob_bar_height, label=f'{home_team_xg} Win ({home_win_prob:.1f}%)')
            ax_prob.barh(y_pos, [draw_width_scaled], left=[bar_start + home_width_scaled], color='gray', height=prob_bar_height, label=f'Draw ({draw_prob:.1f}%)')
            ax_prob.barh(y_pos, [away_width_scaled], left=[bar_start + home_width_scaled + draw_width_scaled], color=away_color, height=prob_bar_height, label=f'{away_team_xg} Win ({away_win_prob:.1f}%)')

            # Update percentage text positions for scaled bar (only if there's enough space)
            min_width_for_text = 8  # Minimum width needed to display text
            
            if home_win_prob > 0 and home_width_scaled >= min_width_for_text:
                ax_prob.text(bar_start + home_width_scaled/2, 0, f'{home_win_prob:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            if draw_prob > 0 and draw_width_scaled >= min_width_for_text:
                ax_prob.text(bar_start + home_width_scaled + draw_width_scaled/2, 0, f'{draw_prob:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            if away_win_prob > 0 and away_width_scaled >= min_width_for_text:
                ax_prob.text(bar_start + home_width_scaled + draw_width_scaled + away_width_scaled/2, 0, f'{away_win_prob:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=12)

            ax_prob.set_xlim(0, 100)
            ax_prob.set_ylim(-0.15, 0.25)  # Back to original working position
            ax_prob.set_yticks([])
            ax_prob.set_xticks([])
            ax_prob.spines['top'].set_visible(False)
            ax_prob.spines['right'].set_visible(False)
            ax_prob.spines['left'].set_visible(False)
            ax_prob.spines['bottom'].set_visible(False)
            
            # Add team logos on both sides of the bar
            logo_width = 12  # Width of the logo in x-axis units (doubled)
            logo_height = 0.24  # Height of the logo in y-axis units (doubled)
            logo_y_center = 0.0  # Center vertically with the bar
            
            # Home team logo (left side)
            if home_logo is not None:
                home_logo_x = 15  # Fixed position to the left
                # Calculate aspect ratio to maintain proportions
                logo_aspect = home_logo.shape[1] / home_logo.shape[0]  # width/height
                if logo_aspect > 1:  # Wider than tall
                    adjusted_width = logo_height * logo_aspect
                    ax_prob.imshow(home_logo, extent=(home_logo_x - adjusted_width/2, home_logo_x + adjusted_width/2, 
                                                    logo_y_center - logo_height/2, logo_y_center + logo_height/2), 
                                  aspect='auto', zorder=10)
                else:  # Taller than wide or square
                    ax_prob.imshow(home_logo, extent=(home_logo_x - logo_width/2, home_logo_x + logo_width/2, 
                                                    logo_y_center - logo_height/2, logo_y_center + logo_height/2), 
                                  aspect='auto', zorder=10)
            
            # Away team logo (right side)
            if away_logo is not None:
                away_logo_x = 85  # Fixed position to the right
                # Calculate aspect ratio to maintain proportions
                logo_aspect = away_logo.shape[1] / away_logo.shape[0]  # width/height
                if logo_aspect > 1:  # Wider than tall
                    adjusted_width = logo_height * logo_aspect
                    ax_prob.imshow(away_logo, extent=(away_logo_x - adjusted_width/2, away_logo_x + adjusted_width/2, 
                                                    logo_y_center - logo_height/2, logo_y_center + logo_height/2), 
                                  aspect='auto', zorder=10)
                else:  # Taller than wide or square
                    ax_prob.imshow(away_logo, extent=(away_logo_x - logo_width/2, away_logo_x + logo_width/2, 
                                                    logo_y_center - logo_height/2, logo_y_center + logo_height/2), 
                                  aspect='auto', zorder=10)
            
            # Scoreboard above the probability bar
            home_goals_count = len([g for g in home_goals if not g['player'].endswith('(OG)')])
            away_goals_count = len([g for g in away_goals if not g['player'].endswith('(OG)')])
            home_own_goals = len([g for g in home_goals if g['player'].endswith('(OG)')])
            away_own_goals = len([g for g in away_goals if g['player'].endswith('(OG)')])

            # Total goals including own goals
            home_total_goals_display = home_goals_count + away_own_goals
            away_total_goals_display = away_goals_count + home_own_goals

            # Scoreboard rows above the bar
            # Row 1: Doelpunten
            ax_prob.text(bar_start, 0.20, f"{home_total_goals_display}", ha='left', va='center', fontsize=14, fontweight='bold', color=home_color)
            ax_prob.text(50, 0.20, "Doelpunten", ha='center', va='center', fontsize=12, color='gray')
            ax_prob.text(bar_end, 0.20, f"{away_total_goals_display}", ha='right', va='center', fontsize=14, fontweight='bold', color=away_color)

            # Row 2: xG totals
            ax_prob.text(bar_start, 0.15, f"{home_total_xg:.2f}", ha='left', va='center', fontsize=14, fontweight='bold', color=home_color)
            ax_prob.text(50, 0.15, "xG", ha='center', va='center', fontsize=12, color='gray')
            ax_prob.text(bar_end, 0.15, f"{away_total_xg:.2f}", ha='right', va='center', fontsize=14, fontweight='bold', color=away_color)

            # Add "Verwacht resultaat o.b.v. kansen" below the bar
            ax_prob.text(50, -0.12, "Verwacht resultaat o.b.v. kansen", ha='center', va='center', fontsize=12, fontweight='bold')

            # Plot cumulative xG lines
            ax_plot.step(home_times, home_cumulative, where='post', color=home_color, linewidth=2.5, label=home_team_xg)
            ax_plot.step(away_times, away_cumulative, where='post', color=away_color, linewidth=2.5, label=away_team_xg)
            ax_plot.fill_between(home_times, 0, home_cumulative, step='post', alpha=0.3, color=home_color)
            ax_plot.fill_between(away_times, 0, away_cumulative, step='post', alpha=0.3, color=away_color)

            # Mark goals chronologically
            all_goals = []
            for goal in home_goals:
                all_goals.append({**goal, 'team': home_team_xg, 'is_home': True})
            for goal in away_goals:
                all_goals.append({**goal, 'team': away_team_xg, 'is_home': False})
            all_goals = sorted(all_goals, key=lambda x: x['time'])

            home_score = 0
            away_score = 0
            for goal in all_goals:
                if goal['is_home']:
                    home_score += 1
                    idx = np.searchsorted(home_times, goal['time'])
                    goal_xg = home_cumulative[min(idx, len(home_cumulative)-1)] if home_cumulative else 0
                    ax_plot.plot(goal['time'], goal_xg, 'o', color='white', markersize=12,
                                 markeredgecolor='black', markeredgewidth=2, zorder=6)
                    score_text = f"{home_score}-{away_score}"
                    player_text = goal['player'].split()[-1] if len(goal['player'].split()) > 1 else goal['player']
                    ax_plot.annotate(f"{score_text} | {player_text}",
                                     xy=(goal['time'], goal_xg),
                                     xytext=(goal['time'], goal_xg + 0.15),
                                     fontsize=9, fontweight='bold',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
                                     ha='center', va='bottom')
                else:
                    away_score += 1
                    idx = np.searchsorted(away_times, goal['time'])
                    goal_xg = away_cumulative[min(idx, len(away_cumulative)-1)] if away_cumulative else 0
                    ax_plot.plot(goal['time'], goal_xg, 'o', color='white', markersize=12,
                                 markeredgecolor='black', markeredgewidth=2, zorder=6)
                    score_text = f"{home_score}-{away_score}"
                    player_text = goal['player'].split()[-1] if len(goal['player'].split()) > 1 else goal['player']
                    ax_plot.annotate(f"{score_text} | {player_text}",
                                     xy=(goal['time'], goal_xg),
                                     xytext=(goal['time'], goal_xg + 0.15),
                                     fontsize=9, fontweight='bold',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
                                     ha='center', va='bottom')

            ax_plot.set_xlabel('Minuut', fontsize=12, fontweight='bold')
            ax_plot.set_ylabel('xG', fontsize=12, fontweight='bold')
            ax_plot.set_xlim(0, max_time)
            ax_plot.set_ylim(0, max(3.0, max(home_total_xg, away_total_xg) + 0.5))
            ax_plot.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax_plot.set_axisbelow(True)

            # Add halftime line
            first_half_end = None
            for event in events_xg:
                if (event.get('baseTypeId') == 14 and
                    event.get('subTypeId') == 1401 and
                    event.get('partId') == 1):
                    first_half_end = event.get('startTimeMs', 0) / 1000 / 60
                    break
            if first_half_end is not None:
                # Draw at the actual first-half end minute (no offset correction)
                ax_plot.axvline(x=first_half_end, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            tick_positions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
            tick_labels = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90']
            ax_plot.set_xticks(tick_positions)
            ax_plot.set_xticklabels(tick_labels)
            ax_plot.set_facecolor('#F8F8F8')

            st.pyplot(fig_xg)
            import io
            buf3 = io.BytesIO()
            fig_xg.savefig(buf3, format='png', dpi=150, bbox_inches='tight')
            buf3.seek(0)
            st.download_button(
                label="📥 Download",
                data=buf3,
                file_name=f"xg_verloop_{file_name.replace('.json', '')}.png",
                mime="image/png"
            )

        # ---------- Eredivisie Tabel Tab ----------
        with tab4:
            st.subheader("Eredivisie 2025/2026 Tabel")
            
            # Initialize league table data structure
            league_data = {}
            
            # Process all matches to build league table
            for file_info in files_info:
                try:
                    with open(file_info['path'], 'r', encoding='utf-8') as f:
                        match_data = load_json_lenient(file_info['path'])
                    
                    if not match_data:
                        continue
                        
                    metadata = match_data.get('metaData', {}) if isinstance(match_data, dict) else {}
                    home_team = metadata.get('homeTeamName') or metadata.get('homeTeam') or metadata.get('home') or 'Home'
                    away_team = metadata.get('awayTeamName') or metadata.get('awayTeam') or metadata.get('away') or 'Away'
                    events = match_data.get('data', []) if isinstance(match_data, dict) else []
                    
                    # Initialize teams if not exists
                    if home_team not in league_data:
                        league_data[home_team] = {
                            'matches_played': 0, 'points': 0, 'expected_points': 0,
                            'goals_for': 0, 'goals_against': 0, 'xg_for': 0, 'xg_against': 0
                        }
                    if away_team not in league_data:
                        league_data[away_team] = {
                            'matches_played': 0, 'points': 0, 'expected_points': 0,
                            'goals_for': 0, 'goals_against': 0, 'xg_for': 0, 'xg_against': 0
                        }
                    
                    # Get shots and goals for this match
                    all_shots_match = find_shot_events_xg(events)
                    home_shots_match = [s for s in all_shots_match if s['team'] == home_team]
                    away_shots_match = [s for s in all_shots_match if s['team'] == away_team]
                    
                    home_own_goals_match = count_own_goals_xg(events, home_team)
                    away_own_goals_match = count_own_goals_xg(events, away_team)
                    
                    # Calculate goals (including own goals)
                    home_goals_match = len([s for s in home_shots_match if s['is_goal']])
                    away_goals_match = len([s for s in away_shots_match if s['is_goal']])
                    home_own_goals_count = len(home_own_goals_match)
                    away_own_goals_count = len(away_own_goals_match)
                    
                    # Total goals including own goals
                    home_total_goals = home_goals_match + away_own_goals_count
                    away_total_goals = away_goals_match + home_own_goals_count
                    
                    # Calculate xG
                    home_xg = sum(shot['xG'] for shot in home_shots_match)
                    away_xg = sum(shot['xG'] for shot in away_shots_match)
                    
                    # Calculate points
                    if home_total_goals > away_total_goals:
                        home_points = 3
                        away_points = 0
                    elif away_total_goals > home_total_goals:
                        home_points = 0
                        away_points = 3
                    else:
                        home_points = 1
                        away_points = 1
                    
                    # Calculate expected points using simulation
                    home_win_prob, draw_prob, away_win_prob = simulate_match(home_shots_match, away_shots_match)
                    home_expected_points = (home_win_prob / 100) * 3 + (draw_prob / 100) * 1
                    away_expected_points = (away_win_prob / 100) * 3 + (draw_prob / 100) * 1
                    
                    # Update league data
                    league_data[home_team]['matches_played'] += 1
                    league_data[home_team]['points'] += home_points
                    league_data[home_team]['expected_points'] += home_expected_points
                    league_data[home_team]['goals_for'] += home_total_goals
                    league_data[home_team]['goals_against'] += away_total_goals
                    league_data[home_team]['xg_for'] += home_xg
                    league_data[home_team]['xg_against'] += away_xg
                    
                    league_data[away_team]['matches_played'] += 1
                    league_data[away_team]['points'] += away_points
                    league_data[away_team]['expected_points'] += away_expected_points
                    league_data[away_team]['goals_for'] += away_total_goals
                    league_data[away_team]['goals_against'] += home_total_goals
                    league_data[away_team]['xg_for'] += away_xg
                    league_data[away_team]['xg_against'] += home_xg
                    
                except Exception as e:
                    continue
            
            # Convert to DataFrame and sort
            if league_data:
                import pandas as pd
                
                table_data = []
                for team, stats in league_data.items():
                    goal_diff = stats['goals_for'] - stats['goals_against']
                    xg_diff = stats['xg_for'] - stats['xg_against']
                    
                    table_data.append({
                        'Team': team,
                        'MP': stats['matches_played'],
                        'Pts': stats['points'],
                        'xPts': round(stats['expected_points'], 1),
                        'GF': stats['goals_for'],
                        'GA': stats['goals_against'],
                        'xG': round(stats['xg_for'], 1),
                        'xGA': round(stats['xg_against'], 1),
                        'GD': goal_diff,
                        'xGD': round(xg_diff, 1)
                    })
                
                df = pd.DataFrame(table_data)
                df = df.sort_values(['Pts', 'GD', 'GF'], ascending=[False, False, False])
                df = df.reset_index(drop=True)
                df.index = df.index + 1
                
                # Display table
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=(len(df) + 1) * 35 + 3,  # Show all rows without scrolling
                    column_config={
                        "Team": st.column_config.TextColumn("Team", width="medium"),
                        "MP": st.column_config.NumberColumn("Wed", help="Wedstrijden gespeeld"),
                        "Pts": st.column_config.NumberColumn("Punten", help="Behaalde punten"),
                        "xPts": st.column_config.NumberColumn("xPunten", help="Verwachte punten"),
                        "GF": st.column_config.NumberColumn("Voor", help="Doelpunten voor"),
                        "GA": st.column_config.NumberColumn("Tegen", help="Doelpunten tegen"),
                        "xG": st.column_config.NumberColumn("xG", help="Expected Goals voor"),
                        "xGA": st.column_config.NumberColumn("xGA", help="Expected Goals tegen"),
                        "GD": st.column_config.NumberColumn("DV", help="Doelsaldo"),
                        "xGD": st.column_config.NumberColumn("xDV", help="Expected Goals saldo")
                    }
                )
                # Summary statistics removed per request
            else:
                st.warning("Geen wedstrijddata gevonden voor de tabel.")

        # ---------- Average Positions Tab ----------
        with tab5:
            st.header("Gemiddelde Posities")
            st.info("Deze tab is momenteel niet beschikbaar.")
        with tab6:
            st.subheader("Samenvatting Statistieken")
            
            if events_data is not None:
                events = events_data.get('data', []) if isinstance(events_data, dict) else []
                
                # High Recoveries Analysis
                st.subheader("🏃‍♂️ High Recoveries (x > -17.5)")
                
                # Define base type IDs for Interception and Ball Recovery
                INTERCEPTION_BASE_TYPE_ID = 5
                BALL_RECOVERY_BASE_TYPE_ID = 9
                SUCCESSFUL_RESULT_ID = 1
                RECOVERY_SUB_TYPE_ID_INTERCEPTION = 501
                
                # Filter for successful interceptions and recoveries in the middle and final third (x > -17.5)
                filtered_recoveries_interceptions = []

                for event in events:
                    base_type_id = event.get('baseTypeId')
                    sub_type_id = event.get('subTypeId')
                    result_id = event.get('resultId')
                    event_x = event.get('startPosXM')
                    
                    # Check if it's a successful event
                    if result_id == SUCCESSFUL_RESULT_ID:
                        # Check if it's an Interception or a Ball Recovery
                        is_interception = base_type_id == INTERCEPTION_BASE_TYPE_ID
                        is_ball_recovery = base_type_id == BALL_RECOVERY_BASE_TYPE_ID
                        is_interception_recovery = (base_type_id == INTERCEPTION_BASE_TYPE_ID and
                                                    sub_type_id == RECOVERY_SUB_TYPE_ID_INTERCEPTION)
                        
                        # If it's one of the relevant types and the location is correct (x > -17.5)
                        if (is_interception or is_ball_recovery or is_interception_recovery) and event_x is not None and event_x > -17.5:
                            filtered_recoveries_interceptions.append(event)
                
                # Prepare data for a DataFrame
                recovery_interception_data = []
                for event in filtered_recoveries_interceptions:
                    recovery_interception_data.append({
                        'Minute': int(event.get('startTimeMs', 0) / 1000 / 60),
                        'Team': event.get('teamName', 'Unknown Team'),
                        'Player': event.get('playerName', 'Unknown Player'),
                        'Base Type': event.get('baseTypeName', 'Unknown'),
                        'Sub Type': event.get('subTypeName', 'Unknown'),
                        'Start X': event.get('startPosXM'),
                        'Start Y': event.get('startPosYM'),
                        'Labels': event.get('labels', [])
                    })
                
                # Create and display the DataFrame
                recovery_interception_df = pd.DataFrame(recovery_interception_data)
                
                # Display team counts
                team_counts = recovery_interception_df['Team'].value_counts().reset_index()
                team_counts.columns = ['Team', 'Count of Recoveries/Interceptions (x > -17.5)']
                
                # Display the counts per team
                st.dataframe(team_counts)
                
                # Display detailed events
                st.subheader("📋 Gedetailleerde High Recoveries")
                
                # Show detailed events table
                if not recovery_interception_df.empty:
                    st.dataframe(recovery_interception_df)
                else:
                    st.info("Geen High Recoveries gevonden in deze wedstrijd.")
                
                # Final Third Entries Analysis
                st.subheader("⚽ Final Third Entries")
                
                # Define labels for final third entries
                DRIBBLE_CHANCE_CREATION_LABEL = 127
                FINAL_3RD_PASS_SUCCESS_LABEL = 69
                
                # Count final third entries per team
                final_third_entries = defaultdict(int)

                for event in events:
                    base_type_id = event.get('baseTypeId')
                    result_id = event.get('resultId')
                    labels = event.get('labels', [])
                    team_name = event.get('teamName')
                    
                    # Check for successful dribbles (baseTypeId 2, resultId 1, label 127)
                    if (base_type_id == 2 and result_id == 1 and DRIBBLE_CHANCE_CREATION_LABEL in labels):
                        final_third_entries[f"{team_name}_dribbles"] += 1
                    
                    # Check for successful final third passes (baseTypeId 1, resultId 1, label 69)
                    if (base_type_id == 1 and result_id == 1 and FINAL_3RD_PASS_SUCCESS_LABEL in labels):
                        final_third_entries[f"{team_name}_passes"] += 1
                
                # Create summary DataFrame
                final_third_df = pd.DataFrame([
                    {'Team': team, 'Chance Creation Dribbles': final_third_entries.get(f"{team}_dribbles", 0), 
                     'Final 3rd Passes': final_third_entries.get(f"{team}_passes", 0)}
                    for team in set([event.get('teamName', 'Unknown') for event in events])
                ])
                
                # Add total column
                final_third_df['Total Final Third Entries'] = final_third_df['Chance Creation Dribbles'] + final_third_df['Final 3rd Passes']
                
                # Display the DataFrame
                st.dataframe(final_third_df)
                
                # Match Summary Statistics
                st.subheader("📊 Wedstrijd Samenvatting")
                
                # Calculate summary metrics
                total_events = len(events)
                total_high_recoveries = len(filtered_recoveries_interceptions)
                total_final_third_entries = final_third_df['Total Final Third Entries'].sum()
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Totaal Events", total_events)
                
                with col2:
                    st.metric("High Recoveries", total_high_recoveries)
                
                with col3:
                    st.metric("Final Third Entries", total_final_third_entries)
                
                # Additional metrics
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    st.metric("Chance Creation Dribbles", final_third_df['Chance Creation Dribbles'].sum())
                
                with col5:
                    st.metric("Final 3rd Passes", final_third_df['Final 3rd Passes'].sum())
                
                with col6:
                    st.metric("Gem. Recovery X Positie", f"{filtered_recoveries_interceptions[0]['startPosXM']:.1f}" if filtered_recoveries_interceptions else "N/A")
            else:
                st.warning("Geen wedstrijddata gevonden voor de samenvatting.")

        def calculate_average_positions_during_sequences(positions_data, events_data, sequence_starts,
                                                        time_window_duration_ms=2000): # Analyze positions for the first 2 seconds of the sequence
            """
            Calculate average positions for each player during the start of defined sequences.
            """
            frames = positions_data.get('data', [])
            # Sort frames by time for efficient processing
            frames.sort(key=lambda x: x.get('t', 0))

            _, _, second_half_start_ms, _ = get_halftime_info(events_data.get('data', []))

            # Initialize position accumulators per team and zone
            average_positions_by_zone = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'x': [], 'y': [], 'shirt': None}))) # {team: {zone: {player_id: {x:[], y:[], shirt:''}}}}

            # Iterate through each defined sequence start
            for seq_info in sequence_starts:
                team = seq_info['team']
                zone = seq_info['zone']
                sequence_start_time = seq_info['start_time_ms']
                sequence_end_time = sequence_start_time + time_window_duration_ms # Define the end of the analysis window for this sequence

                # Find frames within this sequence's time window
                relevant_frames = [
                    frame for frame in frames
                    if sequence_start_time <= frame.get('t', 0) < sequence_end_time
                ]

                if not relevant_frames:
                    continue # Skip if no position data for this sequence start window

                # Process frames within the window
                for frame in relevant_frames:
                    frame_time = frame.get('t', 0)
                    is_second_half = frame_time >= second_half_start_ms if second_half_start_ms is not None else False

                    # Determine which players are on the field for this team at this time
                    # This is complex without explicit lineup/sub data in position frames.
                    # Simplification: Assume players listed in the frame for the team are on the field.
                    team_players_in_frame = []
                    if team == events_data.get('metaData', {}).get('homeTeamName', 'Home'):
                        team_players_in_frame = frame.get('h', [])
                    elif team == events_data.get('metaData', {}).get('awayTeamName', 'Away'):
                        team_players_in_frame = frame.get('a', [])
                    else:
                        # Try to match by group (Home/Away) if team names are not consistent
                        if team == events_data.get('metaData', {}).get('homeTeam', 'Home'):
                            team_players_in_frame = frame.get('h', [])
                        elif team == events_data.get('metaData', {}).get('awayTeam', 'Away'):
                            team_players_in_frame = frame.get('a', [])

                    for player in team_players_in_frame:
                        player_id = player.get('p')
                        if player_id:
                            x = player.get('x', 0)
                            y = player.get('y', 0)
                            shirt = player.get('s', '')

                            # Flip coordinates for second half (teams switch sides)
                            if is_second_half:
                                x = -x
                                y = -y

                            average_positions_by_zone[team][zone][player_id]['x'].append(x)
                            average_positions_by_zone[team][zone][player_id]['y'].append(y)
                            average_positions_by_zone[team][zone][player_id]['shirt'] = shirt # Store latest shirt number

            # Calculate averages for each player in each zone
            final_average_positions = defaultdict(lambda: defaultdict(dict)) # {team: {zone: {player_id: {x, y, shirt, appearances}}}}

            for team, zones_data in average_positions_by_zone.items():
                for zone, players_data in zones_data.items():
                    for player_id, coords in players_data.items():
                        if coords['x'] and coords['y']:
                            final_average_positions[team][zone][player_id] = {
                                'x': np.mean(coords['x']),
                                'y': np.mean(coords['y']),
                                'shirt': coords['shirt'],
                                'appearances': len(coords['x'])
                            }

            return final_average_positions

        def plot_average_positions_by_zone(events_data, positions_data, start_minute=None, end_minute=None):
            """
            Plots average player positions during the start of different ball start zones.
            """
            events = events_data.get('data', [])
            if not events:
                st.error("No event data found.")
                return

            metadata = events_data.get('metaData', {})
            home_team = metadata.get('homeTeamName', metadata.get('homeTeam', 'Home'))
            away_team = metadata.get('awayTeamName', metadata.get('awayTeam', 'Away'))

            # Identify players who were substituted in
            substituted_in_players = set()
            SUBSTITUTE_BASE_TYPE = 16
            SUBBED_IN_SUBTYPE = 1601

            for event in events:
                if event.get('baseTypeId') == SUBSTITUTE_BASE_TYPE and event.get('subTypeId') == SUBBED_IN_SUBTYPE:
                    player_id = event.get('playerId')
                    if player_id is not None and player_id != -1:
                        substituted_in_players.add(player_id)

            # Get and display half-time info
            match_start_ms, first_half_end_ms, second_half_start_ms, match_end_ms = get_halftime_info(events)
            st.write(f"Match Start: {int(match_start_ms / 60000)} minutes")
            st.write(f"First Half End: {int(first_half_end_ms / 60000)} minutes" if first_half_end_ms is not None else "First Half End: N/A")
            st.write(f"Second Half Start: {int(second_half_start_ms / 60000)} minutes" if second_half_start_ms is not None else "Second Half Start: N/A")
            st.write(f"Match End: {int(match_end_ms / 60000)} minutes" if match_end_ms is not None else "Match End: N/A")

            # Categorize sequence starts within the time window
            sequence_starts_categorized = categorize_sequence_starts(events, start_minute, end_minute)

            if not sequence_starts_categorized:
                st.warning(f"No relevant sequence starts found in the specified time window ({start_minute}-{end_minute} minutes).")
                return

            st.write(f"Found {len(sequence_starts_categorized)} relevant sequence starts.")

            # Calculate average positions during the start of these sequences
            average_positions_data = calculate_average_positions_during_sequences(positions_data, events_data, sequence_starts_categorized)

            if not average_positions_data:
                st.warning("No position data found for the calculated sequence starts.")
                return

            # Define the 4 zones for plotting order and titles
            zones_order = ['Doeltrap', 'Zone 1 (Defensive Third)', 'Zone 2 (Middle Third)', 'Zone 3 (Attacking Third)']
            zone_titles = {
                'Doeltrap': 'Doeltrap Start',
                'Zone 1 (Defensive Third)': 'Zone 1 Start',
                'Zone 2 (Middle Third)': 'Zone 2 Start',
                'Zone 3 (Attacking Third)': 'Zone 3 Start'
            }

            # Create figure with 2 rows and 4 columns (8 plots)
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

            # Plot configurations: (ax, team, color, attacking_side)
            plot_configs = []
            for i, zone in enumerate(zones_order):
                plot_configs.append((axes[i], home_team, home_color, 'right', zone)) # Home team in top row, attacks right
                plot_configs.append((axes[i + 4], away_team, away_color, 'left', zone)) # Away team in bottom row, attacks left

            # Set a minimum number of appearances in position data to plot a player
            # This prevents plotting players who only appeared for a very brief moment in the window
            min_appearances_threshold = 5 # Adjust as needed

            # Plot average positions for each team in each zone using mplsoccer Pitch
            for ax, team, color, attacking_side, zone in plot_configs:
                positions = average_positions_data.get(team, {}).get(zone, {})
                title = f"{team} - {zone_titles.get(zone, zone)}"

                # Create mplsoccer Pitch for this subplot
                # Using default vertical orientation
                pitch = Pitch(pitch_color='grass', line_color='white', pitch_type='impect') # Use 'impect' pitch type, no ax here

                # Redraw pitch on the specific axis
                pitch.draw(ax=ax) # Pass ax to draw method
                ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color=color)

                # Filter out players with very few appearances for this zone's sequences
                filtered_positions = {
                    pid: pos for pid, pos in positions.items()
                    if pos['appearances'] >= min_appearances_threshold
                }

                if filtered_positions:
                    for player_id, pos in filtered_positions.items():
                        x = pos['x']
                        y = pos['y']
                        shirt_number = pos.get('shirt', '?')

                        # Determine marker shape: circle for starters, square for substitutes
                        marker = 's' if player_id in substituted_in_players else 'o'

                        # Map SciSports coordinates to mplsoccer default vertical pitch (x, y)
                        # SciSports x (-52.5 to 52.5) -> mplsoccer y (0 to 105)
                        # SciSports y (-34 to 34) -> mplsoccer x (0 to 68)
                        mpl_x = x # SciSports y maps to mplsoccer x
                        mpl_y = y  # SciSports x maps to mplsoccer y

                        # Flip coordinates if the plot is showing away team (attacking downwards in default vertical)
                        # We want home to attack upwards and away to attack downwards in the plots
                        # For home (attacking upwards), use mpl_x, mpl_y directly
                        # For away (attacking downwards), flip mpl_x and mpl_y
                        if team == away_team: # Assuming home attacks upwards (+y), away attacks downwards (-y)
                            mpl_x = -mpl_x # Flip x
                            mpl_y = -mpl_y # Flip y

                        # Plot player position using mplsoccer pitch scatter
                        pitch.scatter(mpl_x, mpl_y, ax=ax, s=300, color=color, alpha=0.7,
                                      edgecolors='white', linewidth=1.5, zorder=5)

                        # Add shirt number
                        ax.text(mpl_x, mpl_y, str(shirt_number),
                               color='white', fontsize=9, fontweight='bold',
                               ha='center', va='center', zorder=6)

            # Add main title with time range
            if start_minute is not None and end_minute is not None:
                time_range = f' (Minuut {start_minute}-{end_minute})'
            elif start_minute is not None:
                time_range = f' (From minute {start_minute})'
            elif end_minute is not None:
                time_range = f' (Until minute {end_minute})'
            else:
                time_range = ' (Full Match)'

            fig.suptitle(f'Average Positions During Ball Start Zones{time_range}\n{home_team} vs {away_team}',
                        fontsize=16, fontweight='bold', y=1.02) # Adjust y for suptitle

            plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
            st.pyplot(fig)

    
        # ---------- Voorzetten Tab (Single Match - Both Teams) ----------
        with tab8:
            st.subheader("📮 Voorzetten per Zone")
            
            # Use current match data
            if events_data is not None:
                events = events_data.get('data', []) if isinstance(events_data, dict) else []
                metadata = events_data.get('metaData', {}) if isinstance(events_data, dict) else {}
                home_team_v = metadata.get('homeTeamName', 'Home')
                away_team_v = metadata.get('awayTeamName', 'Away')
                
                # IDs for crosses
                PASS_BASE_TYPE_ID = 2
                CROSS_SUB_TYPE_ID = 200
                CUTBACK_SUB_TYPE_ID = 204
                CROSS_LOW_SUB_TYPE_ID = 203
                SUCCESSFUL_RESULT_ID = 1
                
                # Zones definition
                zones = {
                    'Zone 18 (Wide Attacking Right)': {'x_min': 36, 'x_max': 52.5, 'y_min': -34, 'y_max': -20.16},
                    'Zone 16 (Wide Attacking Left)': {'x_min': 36, 'x_max': 52.5, 'y_min': 20.16, 'y_max': 34},
                    'Zone 17 LL (Box Left Deep)': {'x_min': 36, 'x_max': 47, 'y_min': 9.25, 'y_max': 20.16},
                    'Zone 17 LR (Box Right Deep)': {'x_min': 36, 'x_max': 47, 'y_min': -20.16, 'y_max': -9.25},
                    'Zone 17 HL (Box Left Attacking)': {'x_min': 47, 'x_max': 52.5, 'y_min': 9.25, 'y_max': 20.16},
                    'Zone 17 HR (Box Right Attacking)': {'x_min': 47, 'x_max': 52.5, 'y_min': -20.16, 'y_max': -9.25},
                    'Zone 13 (Wide Middle Left)': {'x_min': 17.5, 'x_max': 36, 'y_min': 20.16, 'y_max': 34},
                    'Zone 15 (Wide Middle Right)': {'x_min': 17.5, 'x_max': 36, 'y_min': -34, 'y_max': -20.16},
                    'Zone 14 R (Central Deep Left)': {'x_min': 17.5, 'x_max': 36, 'y_min': -20.16, 'y_max': -9.25},
                    'Zone 14 L (Central Deep Right)': {'x_min': 17.5, 'x_max': 36, 'y_min': 9.25, 'y_max': 20.16},
                    'Zone 14 M (Zone 14 Area)': {'x_min': 17.5, 'x_max': 36, 'y_min': -9.25, 'y_max': 9.25},
                }
                
                # Function to process crosses for a team
                def process_team_crosses(events, team_name):
                    cross_events = [
                        event for event in events
                        if event.get('teamName') == team_name and
                        event.get('baseTypeId') == PASS_BASE_TYPE_ID and
                        (event.get('subTypeId') == CROSS_SUB_TYPE_ID or 
                         event.get('subTypeId') == CUTBACK_SUB_TYPE_ID or 
                         event.get('subTypeId') == CROSS_LOW_SUB_TYPE_ID)
                    ]
                    
                    filtered = []
                    for event in cross_events:
                        start_x = event.get('startPosXM')
                        start_y = event.get('startPosYM')
                        end_x = event.get('endPosXM')
                        end_y = event.get('endPosYM')
                        if start_x is not None and start_y is not None and end_x is not None and end_y is not None:
                            distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                            if distance >= 0:
                                filtered.append(event)
                    return filtered
                
                def draw_voorzetten_pitch(cross_events, title):
                    zone_stats = {zone_name: {'total': 0, 'successful': 0} for zone_name in zones}
                    for event in cross_events:
                        start_x = event.get('startPosXM')
                        start_y = event.get('startPosYM')
                        is_successful = event.get('resultId') == SUCCESSFUL_RESULT_ID
                        if start_x is not None and start_y is not None:
                            for zone_name, coords in zones.items():
                                if (coords['x_min'] <= start_x < coords['x_max'] and
                                    coords['y_min'] <= start_y < coords['y_max']):
                                    zone_stats[zone_name]['total'] += 1
                                    if is_successful:
                                        zone_stats[zone_name]['successful'] += 1
                                    break
                    
                    pitch_v = VerticalPitch(half=True, pitch_type='impect')
                    fig_v, ax_v = pitch_v.draw(figsize=(8, 12))
                    
                    for zone_name, coords in zones.items():
                        total = zone_stats[zone_name]['total']
                        successful = zone_stats[zone_name]['successful']
                        percentage = (successful / total * 100) if total > 0 else 0

                        if total > 0 and percentage == 0:
                            facecolor = '#f8d7da'
                        elif percentage > 0 and percentage < 30:
                            facecolor = '#ffd8a8'
                        elif percentage >= 30 and percentage < 60:
                            facecolor = '#fff3bf'
                        elif percentage >= 60:
                            facecolor = '#d3f9d8'
                        else:
                            facecolor = 'none'

                        rect = patches.Rectangle((coords['y_min'], coords['x_min']),
                                                 coords['y_max'] - coords['y_min'],
                                                 coords['x_max'] - coords['x_min'],
                                                 linewidth=1.5, edgecolor='black', facecolor=facecolor, alpha=0.85, zorder=1)
                        ax_v.add_patch(rect)

                        ul_x = coords['x_min'] + 1.2
                        ul_y = coords['y_max'] - 0.4
                        zone_title = zone_name.split('(')[0].strip()
                        pitch_v.annotate(zone_title, (ul_x, ul_y), ax=ax_v,
                                       ha='left', va='top', fontsize=6, color='black',
                                       zorder=11)

                        center_x = (coords['x_min'] + coords['x_max']) / 2
                        center_y = (coords['y_min'] + coords['y_max']) / 2
                        text_string = f"{successful}/{total}\n{percentage:.0f}%"
                        pitch_v.annotate(text_string, (center_x, center_y), ax=ax_v,
                                       ha='center', va='center', fontsize=8, color='black',
                                       fontweight='bold', zorder=12)
                    
                    plt.title(title, fontsize=16, fontweight='bold')
                    return fig_v
                
                # Generate both team graphs
                home_crosses = process_team_crosses(events, home_team_v)
                away_crosses = process_team_crosses(events, away_team_v)
                
                fig_home = draw_voorzetten_pitch(home_crosses, f"{home_team_v} - Voorzetten")
                st.pyplot(fig_home)
                
                fig_away = draw_voorzetten_pitch(away_crosses, f"{away_team_v} - Voorzetten")
                st.pyplot(fig_away)
            else:
                st.info("Selecteer een wedstrijd om voorzetten te bekijken.")
        
        # ---------- Multi Match Voorzetten Tab ----------
        with tab10:
            st.subheader("📮 Multi Match Voorzetten")
            
            # Check if we have team_matches available
            try:
                has_team_matches = team_matches and len(team_matches) > 0
            except NameError:
                has_team_matches = False
            
            if has_team_matches:
                match_labels_voorzetten = [info['label'] for info in team_matches]
                selected_voorzetten_matches = st.multiselect(
                    "Selecteer wedstrijden voor voorzetten analyse",
                    match_labels_voorzetten,
                    default=[team_matches[0]['label']] if len(team_matches) > 0 else []
                )
                
                if selected_voorzetten_matches and len(selected_voorzetten_matches) > 0:
                    # Aggregate crosses from all selected matches
                    all_cross_events_for = []
                    all_cross_events_against = []
                    team_to_filter = selected_team if selected_team else 'Unknown'
                    
                    # IDs for crosses
                    PASS_BASE_TYPE_ID = 2
                    CROSS_SUB_TYPE_ID = 200
                    CUTBACK_SUB_TYPE_ID = 204
                    CROSS_LOW_SUB_TYPE_ID = 203
                    SUCCESSFUL_RESULT_ID = 1
                    SHOT_BASE_TYPE_ID = 6
                    DEFENSIVE_DUEL_BASE_TYPE_ID = 4
                    
                    # Function to check if a cross leads to a shot
                    def is_cross_successful(cross_event, events):
                        """
                        A cross is successful if:
                        1. It has resultId = 1 AND
                        2. Next event is a shot (baseTypeId 6) by the SAME team OR
                        3. Next TWO events are defensive duels AND the event after that is a shot by the SAME team
                        """
                        if cross_event.get('resultId') != SUCCESSFUL_RESULT_ID:
                            return False
                        
                        cross_team = cross_event.get('teamName')
                        if not cross_team:
                            return False
                        
                        # Find the index of this cross in the events list
                        cross_id = cross_event.get('eventId')
                        if not cross_id:
                            return False
                        
                        cross_idx = None
                        for idx, evt in enumerate(events):
                            if evt.get('eventId') == cross_id:
                                cross_idx = idx
                                break
                        
                        if cross_idx is None or cross_idx >= len(events) - 1:
                            return False
                        
                        # Check Condition 1: Immediately followed by a shot by the SAME team
                        next_event_1 = events[cross_idx + 1]
                        if (next_event_1.get('baseTypeId') == SHOT_BASE_TYPE_ID and
                            next_event_1.get('teamName') == cross_team):
                            return True
                        
                        # Check Condition 2: Followed by TWO defensive duels and then a shot by the SAME team
                        if cross_idx + 3 < len(events):
                            next_event_2 = events[cross_idx + 2]
                            next_event_3 = events[cross_idx + 3]
                            
                            if (next_event_1.get('baseTypeName') == 'DEFENSIVE_DUEL' and
                                next_event_2.get('baseTypeName') == 'DEFENSIVE_DUEL' and
                                next_event_3.get('baseTypeId') == SHOT_BASE_TYPE_ID and
                                next_event_3.get('teamName') == cross_team):
                                return True
                        
                        return False
                    
                    # Load all selected matches and collect crosses with success info
                    for match_label in selected_voorzetten_matches:
                        match_info = next((m for m in team_matches if m['label'] == match_label), None)
                        if match_info:
                            try:
                                match_data = load_json_lenient(match_info['path'])
                                events = match_data.get('data', []) if isinstance(match_data, dict) else []
                                
                                # Filter crosses by selected team
                                cross_events_for = [
                                    event for event in events
                                    if event.get('teamName') == team_to_filter and
                                    event.get('baseTypeId') == PASS_BASE_TYPE_ID and
                                    (event.get('subTypeId') == CROSS_SUB_TYPE_ID or 
                                     event.get('subTypeId') == CUTBACK_SUB_TYPE_ID or 
                                     event.get('subTypeId') == CROSS_LOW_SUB_TYPE_ID)
                                ]
                                
                                # Filter crosses against selected team
                                cross_events_against = [
                                    event for event in events
                                    if event.get('teamName') != team_to_filter and
                                    event.get('baseTypeId') == PASS_BASE_TYPE_ID and
                                    (event.get('subTypeId') == CROSS_SUB_TYPE_ID or 
                                     event.get('subTypeId') == CUTBACK_SUB_TYPE_ID or 
                                     event.get('subTypeId') == CROSS_LOW_SUB_TYPE_ID)
                                ]
                                
                                # Distance filter, check success, and add to aggregated lists
                                for event in cross_events_for:
                                    start_x = event.get('startPosXM')
                                    start_y = event.get('startPosYM')
                                    end_x = event.get('endPosXM')
                                    end_y = event.get('endPosYM')
                                    if start_x is not None and start_y is not None and end_x is not None and end_y is not None:
                                        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                                        if distance >= 0:
                                            # Add success flag to the event
                                            event['is_successful_cross'] = is_cross_successful(event, events)
                                            all_cross_events_for.append(event)
                                
                                for event in cross_events_against:
                                    start_x = event.get('startPosXM')
                                    start_y = event.get('startPosYM')
                                    end_x = event.get('endPosXM')
                                    end_y = event.get('endPosYM')
                                    if start_x is not None and start_y is not None and end_x is not None and end_y is not None:
                                        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                                        if distance >= 0:
                                            # Add success flag to the event
                                            event['is_successful_cross'] = is_cross_successful(event, events)
                                            all_cross_events_against.append(event)
                            except Exception as e:
                                st.warning(f"Error loading {match_label}: {e}")
                
                    # Zones definition
                    zones = {
                        'Zone 18 (Wide Attacking Right)': {'x_min': 36, 'x_max': 52.5, 'y_min': -34, 'y_max': -20.16},
                        'Zone 16 (Wide Attacking Left)': {'x_min': 36, 'x_max': 52.5, 'y_min': 20.16, 'y_max': 34},
                        'Zone 17 LL (Box Left Deep)': {'x_min': 36, 'x_max': 47, 'y_min': 9.25, 'y_max': 20.16},
                        'Zone 17 LR (Box Right Deep)': {'x_min': 36, 'x_max': 47, 'y_min': -20.16, 'y_max': -9.25},
                        'Zone 17 HL (Box Left Attacking)': {'x_min': 47, 'x_max': 52.5, 'y_min': 9.25, 'y_max': 20.16},
                        'Zone 17 HR (Box Right Attacking)': {'x_min': 47, 'x_max': 52.5, 'y_min': -20.16, 'y_max': -9.25},
                        'Zone 13 (Wide Middle Left)': {'x_min': 17.5, 'x_max': 36, 'y_min': 20.16, 'y_max': 34},
                        'Zone 15 (Wide Middle Right)': {'x_min': 17.5, 'x_max': 36, 'y_min': -34, 'y_max': -20.16},
                        'Zone 14 R (Central Deep Left)': {'x_min': 17.5, 'x_max': 36, 'y_min': -20.16, 'y_max': -9.25},
                        'Zone 14 L (Central Deep Right)': {'x_min': 17.5, 'x_max': 36, 'y_min': 9.25, 'y_max': 20.16},
                        'Zone 14 M (Zone 14 Area)': {'x_min': 17.5, 'x_max': 36, 'y_min': -9.25, 'y_max': 9.25},
                    }
                    
                    # Count crosses per zone for both teams
                    zone_stats_for = {zone_name: {'total': 0, 'successful': 0} for zone_name in zones}
                    zone_stats_against = {zone_name: {'total': 0, 'successful': 0} for zone_name in zones}
                    
                    for event in all_cross_events_for:
                        start_x = event.get('startPosXM')
                        start_y = event.get('startPosYM')
                        is_successful = event.get('is_successful_cross', False)
                        if start_x is not None and start_y is not None:
                            for zone_name, coords in zones.items():
                                if (coords['x_min'] <= start_x < coords['x_max'] and
                                    coords['y_min'] <= start_y < coords['y_max']):
                                    zone_stats_for[zone_name]['total'] += 1
                                    if is_successful:
                                        zone_stats_for[zone_name]['successful'] += 1
                                    break
                    
                    for event in all_cross_events_against:
                        start_x = event.get('startPosXM')
                        start_y = event.get('startPosYM')
                        is_successful = event.get('is_successful_cross', False)
                        if start_x is not None and start_y is not None:
                            for zone_name, coords in zones.items():
                                if (coords['x_min'] <= start_x < coords['x_max'] and
                                    coords['y_min'] <= start_y < coords['y_max']):
                                    zone_stats_against[zone_name]['total'] += 1
                                    if is_successful:
                                        zone_stats_against[zone_name]['successful'] += 1
                                    break
                    
                    # Draw full vertical pitch (both halves)
                    pitch_full = VerticalPitch(half=False, pitch_type='impect')
                    fig_full, ax_full = pitch_full.draw(figsize=(8, 16))
                    
                    # Draw zones for selected team's crosses (top half)
                    for zone_name, coords in zones.items():
                        total = zone_stats_for[zone_name]['total']
                        successful = zone_stats_for[zone_name]['successful']
                        percentage = (successful / total * 100) if total > 0 else 0

                        if total > 0 and percentage == 0:
                            facecolor = '#f8d7da'
                        elif percentage > 0 and percentage < 30:
                            facecolor = '#ffd8a8'
                        elif percentage >= 30 and percentage < 60:
                            facecolor = '#fff3bf'
                        elif percentage >= 60:
                            facecolor = '#d3f9d8'
                        else:
                            facecolor = 'none'

                        # Draw zone rectangle (top half, normal coords)
                        rect = patches.Rectangle((coords['y_min'], coords['x_min']),
                                                 coords['y_max'] - coords['y_min'],
                                                 coords['x_max'] - coords['x_min'],
                                                 linewidth=1.5, edgecolor='black', facecolor=facecolor, alpha=0.85, zorder=1)
                        ax_full.add_patch(rect)

                        # Zone name
                        ul_x = coords['x_min'] + 1.2
                        ul_y = coords['y_max'] - 0.4
                        zone_title = zone_name.split('(')[0].strip()
                        pitch_full.annotate(zone_title, (ul_x, ul_y), ax=ax_full,
                                           ha='left', va='top', fontsize=6, color='black',
                                           zorder=11)

                        # Centered stats text
                        center_x = (coords['x_min'] + coords['x_max']) / 2
                        center_y = (coords['y_min'] + coords['y_max']) / 2
                        text_string = f"{successful}/{total}\n{percentage:.0f}%"
                        pitch_full.annotate(text_string, (center_x, center_y), ax=ax_full,
                                           ha='center', va='center', fontsize=8, color='black',
                                           fontweight='bold', zorder=12)
                    
                    # Draw zones for conceded crosses (bottom half, inverted)
                    for zone_name, coords in zones.items():
                        total = zone_stats_against[zone_name]['total']
                        successful = zone_stats_against[zone_name]['successful']
                        percentage = (successful / total * 100) if total > 0 else 0

                        # Inverse color scheme: green for safe (low %), red for dangerous (high %)
                        if total > 0 and percentage == 0:
                            facecolor = '#d3f9d8'  # Green - safe, no successful crosses
                        elif percentage > 0 and percentage < 30:
                            facecolor = '#fff3bf'  # Yellow - relatively safe
                        elif percentage >= 30 and percentage < 60:
                            facecolor = '#ffd8a8'  # Orange - moderate danger
                        elif percentage >= 60:
                            facecolor = '#f8d7da'  # Red - dangerous, high success rate
                        else:
                            facecolor = 'none'

                        # Invert coordinates for bottom half: x -> -x, y -> -y
                        inverted_y_min = -coords['y_max']
                        inverted_y_max = -coords['y_min']
                        inverted_x_min = -coords['x_max']
                        inverted_x_max = -coords['x_min']
                        
                        # Draw zone rectangle (bottom half, inverted)
                        rect = patches.Rectangle((inverted_y_min, inverted_x_min),
                                                 inverted_y_max - inverted_y_min,
                                                 inverted_x_max - inverted_x_min,
                                                 linewidth=1.5, edgecolor='black', facecolor=facecolor, alpha=0.85, zorder=1)
                        ax_full.add_patch(rect)

                        # Zone name (inverted position) - use zone_name not zone_title from previous loop
                        zone_title_inv = zone_name.split('(')[0].strip()
                        ul_x_inv = inverted_x_min + 1.2
                        ul_y_inv = inverted_y_max - 0.4
                        pitch_full.annotate(zone_title_inv, (ul_x_inv, ul_y_inv), ax=ax_full,
                                           ha='left', va='top', fontsize=6, color='black',
                                           zorder=11)

                        # Centered stats text (inverted)
                        center_x_inv = (inverted_x_min + inverted_x_max) / 2
                        center_y_inv = (inverted_y_min + inverted_y_max) / 2
                        text_string = f"{successful}/{total}\n{percentage:.0f}%"
                        pitch_full.annotate(text_string, (center_x_inv, center_y_inv), ax=ax_full,
                                           ha='center', va='center', fontsize=8, color='black',
                                           fontweight='bold', zorder=12)
                    
                    # Add labels for each half (closer to the pitch)
                    fig_full.text(0.5, 0.88, f"{team_to_filter} - Voorzetten", fontsize=12, fontweight='bold',
                                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    fig_full.text(0.5, 0.85, "Voorzetten die leiden tot doelpoging", fontsize=9, fontstyle='italic',
                                ha='center', va='center', color='gray')
                    
                    fig_full.text(0.5, 0.12, f"{team_to_filter} - Voorzetten Tegen", fontsize=12, fontweight='bold',
                                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    fig_full.text(0.5, 0.09, "Voorzetten die leiden tot doelpoging", fontsize=9, fontstyle='italic',
                                ha='center', va='center', color='gray')
                    
                    st.pyplot(fig_full)
                else:
                    st.info("Selecteer minstens één wedstrijd voor voorzetten analyse.")
            else:
                st.info("Geen wedstrijden beschikbaar voor dit team.")
else:
    st.info("Please select a team and match on the main screen to begin analysis.")