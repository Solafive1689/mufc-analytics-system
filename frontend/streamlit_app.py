"""
Manchester United Analytics Dashboard
Streamlit Web Application for Data Visualization
Deploy for free on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta
import os
from analysis_modules import PerformanceAnalyzer, PlayerAnalyzer, TacticalAnalyzer

# Page configuration
st.set_page_config(
    page_title="Manchester United Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #FF0000;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF0000;
    }
    .sidebar-header {
        color: #FF0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
@st.cache_resource
def init_database_connection():
    """Initialize database connection with caching"""
    try:
        db_config = {
            'host': st.secrets.get("DB_HOST", os.getenv('DB_HOST')),
            'database': st.secrets.get("DB_NAME", os.getenv('DB_NAME', 'mufc_analytics')),
            'user': st.secrets.get("DB_USER", os.getenv('DB_USER', 'postgres')),
            'password': st.secrets.get("DB_PASSWORD", os.getenv('DB_PASSWORD')),
        }
        
        # Test connection
        conn = psycopg2.connect(**db_config)
        conn.close()
        
        return db_config
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.info("Using demo data mode")
        return None

# Initialize analyzers
@st.cache_resource
def get_analyzers():
    """Get analyzer instances with caching"""
    db_config = init_database_connection()
    if db_config:
        return {
            'performance': PerformanceAnalyzer(db_config),
            'player': PlayerAnalyzer(db_config),
            'tactical': TacticalAnalyzer(db_config)
        }
    return None

# Demo data for when database is unavailable
def get_demo_data():
    """Generate demo data for testing"""
    return {
        'home_away': [
            {'venue_type': 'Home', 'matches_played': 10, 'wins': 7, 'draws': 2, 'losses': 1, 'win_rate': 70, 'goals_for': 22, 'goals_against': 8},
            {'venue_type': 'Away', 'matches_played': 8, 'wins': 4, 'draws': 3, 'losses': 1, 'win_rate': 50, 'goals_for': 15, 'goals_against': 10}
        ],
        'players': pd.DataFrame({
            'name': ['Bruno Fernandes', 'Marcus Rashford', 'Casemiro', 'Harry Maguire'],
            'position': ['MID', 'FWD', 'MID', 'DEF'],
            'goals': [8, 12, 2, 1],
            'assists': [6, 4, 3, 0],
            'appearances': [18, 16, 15, 14],
            'avg_rating': [7.8, 7.5, 7.2, 6.9]
        }),
        'formations': pd.DataFrame({
            'formation': ['4-2-3-1', '4-3-3', '3-5-2'],
            'matches_played': [12, 4, 2],
            'wins': [8, 2, 1],
            'win_rate': [66.7, 50.0, 50.0],
            'avg_goals_for': [1.8, 1.5, 2.0]
        })
    }

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">Manchester United Analytics Dashboard 2025/26</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<h3 class="sidebar-header">Navigation</h3>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["Overview", "Performance Analysis", "Player Analysis", "Tactical Analysis", "Match Analysis"]
    )
    
    # Season info
    st.sidebar.markdown("---")
    st.sidebar.info("**Season:** 2025/26\n**Last Updated:** " + datetime.now().strftime("%d/%m/%Y %H:%M"))
    
    # Get analyzers or demo data
    analyzers = get_analyzers()
    demo_mode = analyzers is None
    
    if demo_mode:
        st.warning("⚠️ Demo Mode - Using sample data")
        data = get_demo_data()
    
    # Main content based on page selection
    if page == "Overview":
        show_overview(analyzers, demo_mode, data if demo_mode else None)
    elif page == "Performance Analysis":
        show_performance_analysis(analyzers, demo_mode, data if demo_mode else None)
    elif page == "Player Analysis":
        show_player_analysis(analyzers, demo_mode, data if demo_mode else None)
    elif page == "Tactical Analysis":
        show_tactical_analysis(analyzers, demo_mode, data if demo_mode else None)
    elif page == "Match Analysis":
        show_match_analysis(analyzers, demo_mode, data if demo_mode else None)

def show_overview(analyzers, demo_mode, demo_data):
    """Show overview dashboard"""
    
    st.subheader("🏆 Season Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if demo_mode:
        total_matches = sum([x['matches_played'] for x in demo_data['home_away']])
        total_wins = sum([x['wins'] for x in demo_data['home_away']])
        total_goals = sum([x['goals_for'] for x in demo_data['home_away']])
        win_rate = (total_wins / total_matches) * 100
    else:
        # Get real data
        form_data = analyzers['performance'].get_form_analysis(last_n_games=38)
        total_matches = form_data.get('matches_played', 0)
        total_wins = form_data.get('wins', 0)
        total_goals = form_data.get('goals_for', 0)
        win_rate = form_data.get('win_rate', 0)
    
    with col1:
        st.metric("Matches Played", total_matches)
    with col2:
        st.metric("Wins", total_wins)
    with col3:
        st.metric("Goals Scored", total_goals)
    with col4:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home vs Away Performance")
        if demo_mode:
            home_away_df = pd.DataFrame(demo_data['home_away'])
        else:
            home_away_data = analyzers['performance'].analyze_home_vs_away_performance()
            home_away_df = pd.DataFrame(home_away_data)
        
        if not home_away_df.empty:
            fig = px.bar(
                home_away_df, 
                x='venue_type', 
                y='win_rate',
                color='venue_type',
                color_discrete_map={'Home': '#FF0000', 'Away': '#FFD700'},
                title="Win Rate by Venue"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Top Performers")
        if demo_mode:
            top_players = demo_data['players'].head(5)
        else:
            top_players = analyzers['player'].get_top_performers().head(5)
        
        if not top_players.empty:
            fig = px.bar(
                top_players,
                x='name',
                y='goals',
                color='position',
                title="Goals by Top Players"
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent form
    st.subheader("📊 Recent Form")
    
    if not demo_mode and analyzers:
        recent_form = analyzers['performance'].get_form_analysis(last_n_games=10)
        if recent_form:
            form_string = recent_form.get('form_string', '')
            
            # Create form visualization
            form_colors = {'W': 'green', 'D': 'orange', 'L': 'red'}
            form_data = [{'Match': i+1, 'Result': result, 'Color': form_colors[result]} 
                        for i, result in enumerate(reversed(form_string))]
            
            if form_data:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Form string display
                    form_html = ""
                    for item in form_data:
                        color = item['Color']
                        form_html += f'<span style="background-color:{color};color:white;padding:5px 8px;margin:2px;border-radius:3px;font-weight:bold">{item["Result"]}</span>'
                    st.markdown(f"**Last 10 Games:** {form_html}", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Points in Last 10", recent_form.get('points', 0))
                
                with col3:
                    st.metric("Goal Difference", f"+{recent_form.get('goal_difference', 0)}" if recent_form.get('goal_difference', 0) >= 0 else str(recent_form.get('goal_difference', 0)))

def show_performance_analysis(analyzers, demo_mode, demo_data):
    """Show detailed performance analysis"""
    
    st.subheader("📈 Performance Analysis")
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Home vs Away", "Opposition Analysis", "Form Tracker"])
    
    with tab1:
        st.subheader("🏠 Home vs Away Detailed Analysis")
        
        if demo_mode:
            home_away_df = pd.DataFrame(demo_data['home_away'])
        else:
            home_away_data = analyzers['performance'].analyze_home_vs_away_performance()
            home_away_df = pd.DataFrame(home_away_data)
        
        if not home_away_df.empty:
            # Metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏠 Home Performance")
                home_data = home_away_df[home_away_df['venue_type'] == 'Home'].iloc[0] if len(home_away_df) > 0 else None
                if home_data is not None:
                    st.metric("Matches", int(home_data['matches_played']))
                    st.metric("Win Rate", f"{home_data['win_rate']:.1f}%")
                    st.metric("Goals For", int(home_data['goals_for']))
                    st.metric("Goals Against", int(home_data['goals_against']))
            
            with col2:
                st.markdown("### ✈️ Away Performance") 
                away_data = home_away_df[home_away_df['venue_type'] == 'Away'].iloc[0] if len(home_away_df) > 1 else None
                if away_data is not None:
                    st.metric("Matches", int(away_data['matches_played']))
                    st.metric("Win Rate", f"{away_data['win_rate']:.1f}%")
                    st.metric("Goals For", int(away_data['goals_for']))
                    st.metric("Goals Against", int(away_data['goals_against']))
            
            # Visualization
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Win Rate %', 'Goals For/Against'))
            
            fig.add_trace(
                go.Bar(x=home_away_df['venue_type'], y=home_away_df['win_rate'], 
                       name='Win Rate', marker_color=['#FF0000', '#FFD700']),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=home_away_df['venue_type'], y=home_away_df['goals_for'], 
                       name='Goals For', marker_color='green'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(x=home_away_df['venue_type'], y=home_away_df['goals_against'], 
                       name='Goals Against', marker_color='red'),
                row=1, col=2
            )
            
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Performance vs Opposition Strength")
        
        if not demo_mode and analyzers:
            opp_data = analyzers['performance'].analyze_top6_vs_bottom_performance()
            if opp_data:
                opp_df = pd.DataFrame(opp_data)
                
                # Display table
                st.dataframe(
                    opp_df[['opposition_strength', 'matches_played', 'wins', 'draws', 'win_rate', 'points_per_game']],
                    use_container_width=True
                )
                
                # Visualization
                fig = px.bar(
                    opp_df,
                    x='opposition_strength',
                    y='win_rate',
                    color='opposition_strength',
                    title="Win Rate vs Opposition Strength"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No opposition strength data available yet")
        else:
            st.info("Opposition analysis requires live database connection")
    
    with tab3:
        st.subheader("📊 Form Tracker")
        
        # Form length selector
        form_games = st.selectbox("Analyze last N games:", [5, 10, 15, 20], index=1)
        
        if not demo_mode and analyzers:
            form_data = analyzers['performance'].get_form_analysis(last_n_games=form_games)
            
            if form_data and form_data.get('recent_matches'):
                # Form metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Matches", form_data['matches_played'])
                with col2:
                    st.metric("Points", form_data['points'])
                with col3:
                    st.metric("PPG", f"{form_data['points_per_game']:.2f}")
                with col4:
                    st.metric("GD", f"+{form_data['goal_difference']}" if form_data['goal_difference'] >= 0 else str(form_data['goal_difference']))
                
                # Recent matches table
                st.subheader("Recent Matches")
                matches_df = pd.DataFrame(form_data['recent_matches'])
                matches_df['opponent'] = matches_df.apply(
                    lambda x: x['away_team'] if x['venue'] == 'Home' else x['home_team'], axis=1
                )
                matches_df['score'] = matches_df.apply(
                    lambda x: f"{x['goals_for']}-{x['goals_against']}", axis=1
                )
                
                display_matches = matches_df[['match_date', 'opponent', 'venue', 'result', 'score']]
                display_matches['match_date'] = pd.to_datetime(display_matches['match_date']).dt.strftime('%d/%m/%Y')
                
                st.dataframe(display_matches, use_container_width=True)

def show_player_analysis(analyzers, demo_mode, demo_data):
    """Show player analysis page"""
    
    st.subheader("👥 Player Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Top Performers", "Unsung Heroes", "Individual Analysis"])
    
    with tab1:
        st.subheader("⭐ Top Performers")
        
        # Position filter
        position_filter = st.selectbox("Filter by position:", ["All", "FWD", "MID", "DEF", "GK"])
        min_minutes = st.slider("Minimum minutes played:", 90, 1800, 180)
        
        if demo_mode:
            top_players = demo_data['players']
        else:
            pos_filter = None if position_filter == "All" else position_filter
            top_players = analyzers['player'].get_top_performers(position=pos_filter, min_minutes=min_minutes)
        
        if not top_players.empty:
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Goals chart
                fig = px.bar(
                    top_players.head(10),
                    x='name',
                    y='goals',
                    color='position',
                    title="Goals Scored"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Assists chart
                fig = px.bar(
                    top_players.head(10),
                    x='name',
                    y='assists' if 'assists' in top_players.columns else 'goals',
                    color='position',
                    title="Assists"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("Detailed Statistics")
            display_columns = ['name', 'position', 'appearances', 'goals', 'assists'] if demo_mode else \
                             ['name', 'position', 'appearances', 'total_minutes', 'goals', 'assists', 'goals_per_90', 'assists_per_90', 'avg_rating']
            
            available_columns = [col for col in display_columns if col in top_players.columns]
            st.dataframe(top_players[available_columns], use_container_width=True)
    
    with tab2:
        st.subheader("🛡️ Unsung Heroes (Defensive Contributors)")
        
        if not demo_mode and analyzers:
            unsung_heroes = analyzers['player'].get_unsung_heroes()
            
            if not unsung_heroes.empty:
                # Defensive actions chart
                fig = px.bar(
                    unsung_heroes.head(10),
                    x='name',
                    y='defensive_actions_per_90',
                    color='position',
                    title="Defensive Actions per 90 Minutes"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                display_cols = ['name', 'position', 'appearances', 'tackles', 'interceptions', 
                              'defensive_actions_per_90', 'avg_rating']
                st.dataframe(unsung_heroes[display_cols], use_container_width=True)
            else:
                st.info("No defensive statistics available yet")
        else:
            st.info("Unsung Heroes analysis requires live database connection")
    
    with tab3:
        st.subheader("🔍 Individual Player Analysis")
        
        # Player selector
        if demo_mode:
            player_names = demo_data['players']['name'].tolist()
        else:
            # Get player names from database
            player_names = ["Bruno Fernandes", "Marcus Rashford", "Casemiro"]  # Placeholder
        
        selected_player = st.selectbox("Select player:", player_names)
        
        if selected_player and not demo_mode and analyzers:
            player_form = analyzers['player'].analyze_player_form(selected_player)
            
            if player_form:
                # Player metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Rating", f"{player_form['avg_rating']:.1f}")
                with col2:
                    st.metric("Recent Goals", player_form['total_goals'])
                with col3:
                    st.metric("Recent Assists", player_form['total_assists'])
                with col4:
                    st.metric("Pass Accuracy", f"{player_form['avg_pass_accuracy']:.1f}%")
                
                # Recent form table
                if player_form['recent_form']:
                    st.subheader("Recent Performances")
                    recent_df = pd.DataFrame(player_form['recent_form'])
                    st.dataframe(recent_df, use_container_width=True)
        elif selected_player:
            st.info(f"Individual analysis for {selected_player} - requires live database")

def show_tactical_analysis(analyzers, demo_mode, demo_data):
    """Show tactical analysis page"""
    
    st.subheader("🧠 Tactical Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Formation Analysis", "Possession vs Results", "Shot Maps"])
    
    with tab1:
        st.subheader("⚽ Formation Effectiveness")
        
        if demo_mode:
            formations_df = demo_data['formations']
        else:
            formations_df = analyzers['tactical'].analyze_formation_effectiveness()
        
        if not formations_df.empty:
            # Formation effectiveness chart
            fig = px.bar(
                formations_df,
                x='formation',
                y='win_rate',
                color='matches_played',
                title="Win Rate by Formation",
                hover_data=['matches_played', 'wins']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed formation table
            st.subheader("Formation Statistics")
            display_cols = ['formation', 'matches_played', 'wins', 'win_rate'] if demo_mode else \
                          ['formation', 'matches_played', 'wins', 'win_rate', 'avg_goals_for', 'avg_possession']
            available_cols = [col for col in display_cols if col in formations_df.columns]
            st.dataframe(formations_df[available_cols], use_container_width=True)
    
    with tab2:
        st.subheader("📊 Possession vs Results Analysis")
        
        if not demo_mode and analyzers:
            possession_data = analyzers['tactical'].analyze_possession_vs_results()
            
            if possession_data and possession_data.get('raw_data'):
                poss_df = pd.DataFrame(possession_data['raw_data'])
                
                # Scatter plot
                fig = px.scatter(
                    poss_df,
                    x='possession_percent',
                    y='goals_for',
                    color='result',
                    title="Possession % vs Goals Scored",
                    color_discrete_map={'Win': 'green', 'Draw': 'orange', 'Loss': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation info
                if 'possession_goals_correlation' in possession_data:
                    st.metric("Possession-Goals Correlation", f"{possession_data['possession_goals_correlation']:.3f}")
        else:
            st.info("Possession analysis requires live database connection")
    
    with tab3:
        st.subheader("🎯 Shot Map Analysis")
        
        if not demo_mode and analyzers:
            shot_data = analyzers['tactical'].analyze_shot_maps()
            
            if shot_data and shot_data.get('shot_data'):
                shots_df = pd.DataFrame(shot_data['shot_data'])
                
                # Shot map visualization
                fig = px.scatter(
                    shots_df,
                    x='x_coordinate',
                    y='y_coordinate',
                    color='is_goal',
                    size='xg_value',
                    hover_data=['player_name', 'shot_type'],
                    title="Shot Map",
                    color_discrete_map={True: 'green', False: 'red'}
                )
                fig.update_layout(
                    xaxis_title="Pitch Length",
                    yaxis_title="Pitch Width",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Shot statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Shots", shot_data['total_shots'])
                with col2:
                    st.metric("Goals", shot_data['goals_scored'])
                with col3:
                    st.metric("Shot Accuracy", f"{shot_data['shot_accuracy']:.1f}%")
                with col4:
                    st.metric("Conversion Rate", f"{shot_data['conversion_rate']:.1f}%")
        else:
            st.info("Shot map analysis requires live database connection with shot coordinate data")

def show_match_analysis(analyzers, demo_mode, demo_data):
    """Show individual match analysis"""
    
    st.subheader("🔍 Match Analysis")
    st.info("Individual match analysis - Coming soon!")
    st.markdown("""
    This section will include:
    - Match-by-match performance breakdown
    - Player ratings and statistics per game
    - Tactical analysis for specific matches
    - Opposition scouting reports
    - Pre and post-match analytics
    """)

# Streamlit app entry point
if __name__ == "__main__":
    main()