"""
Manchester United Analytics - Analysis Modules
Performance, Player, and Tactical Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AnalyticsBase:
    """Base class for all analytics modules"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.mufc_team_id = self._get_mufc_team_id()
        self.current_season_id = self._get_current_season_id()
    
    def get_db_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def execute_query(self, query: str, params=None) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        with self.get_db_connection() as conn:
            return pd.read_sql(query, conn, params=params)
    
    def _get_mufc_team_id(self) -> int:
        query = "SELECT id FROM teams WHERE short_name = 'MUN'"
        result = self.execute_query(query)
        return result.iloc[0]['id'] if not result.empty else 1
    
    def _get_current_season_id(self) -> int:
        query = "SELECT id FROM seasons WHERE is_current = TRUE"
        result = self.execute_query(query)
        return result.iloc[0]['id'] if not result.empty else 1

class PerformanceAnalyzer(AnalyticsBase):
    """Analyze Manchester United's performance metrics"""
    
    def analyze_home_vs_away_performance(self) -> Dict:
        """Compare home vs away performance"""
        query = """
        SELECT 
            CASE 
                WHEN m.home_team_id = %s THEN 'Home'
                ELSE 'Away'
            END as venue_type,
            COUNT(*) as matches_played,
            SUM(CASE 
                WHEN (m.home_team_id = %s AND m.home_score > m.away_score)
                OR (m.away_team_id = %s AND m.away_score > m.home_score)
                THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN m.home_score = m.away_score THEN 1 ELSE 0 END) as draws,
            SUM(CASE 
                WHEN (m.home_team_id = %s AND m.home_score < m.away_score)
                OR (m.away_team_id = %s AND m.away_score < m.home_score)
                THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN m.home_team_id = %s THEN m.home_score ELSE m.away_score END) as goals_for,
            SUM(CASE WHEN m.home_team_id = %s THEN m.away_score ELSE m.home_score END) as goals_against
        FROM matches m
        WHERE (m.home_team_id = %s OR m.away_team_id = %s)
            AND m.status = 'finished'
            AND m.season_id = %s
        GROUP BY venue_type
        """
        params = [self.mufc_team_id] * 9 + [self.current_season_id]
        df = self.execute_query(query, params)
        
        # Calculate additional metrics
        df['win_rate'] = df['wins'] / df['matches_played'] * 100
        df['points'] = df['wins'] * 3 + df['draws']
        df['points_per_game'] = df['points'] / df['matches_played']
        df['goal_difference'] = df['goals_for'] - df['goals_against']
        
        return df.to_dict('records')
    
    def analyze_top6_vs_bottom_performance(self) -> Dict:
        """Analyze performance against different opponent strengths"""
        # Define top 6 teams (adjust based on current season)
        top6_teams = ['Manchester City', 'Arsenal', 'Liverpool', 
                      'Chelsea', 'Tottenham Hotspur', 'Newcastle United']
        
        query = """
        SELECT 
            CASE 
                WHEN (ht.name IN %s OR at.name IN %s) AND ht.name != 'Manchester United' AND at.name != 'Manchester United'
                THEN 'Top 6'
                WHEN pts.final_position BETWEEN 1 AND 6 AND pts.team_id != %s
                THEN 'Top 6'
                WHEN pts.final_position BETWEEN 7 AND 14 OR pts.final_position IS NULL
                THEN 'Mid Table'
                ELSE 'Bottom 6'
            END as opposition_strength,
            COUNT(*) as matches_played,
            SUM(CASE 
                WHEN (m.home_team_id = %s AND m.home_score > m.away_score)
                OR (m.away_team_id = %s AND m.away_score > m.home_score)
                THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN m.home_score = m.away_score THEN 1 ELSE 0 END) as draws,
            AVG(CASE WHEN m.home_team_id = %s THEN m.home_score ELSE m.away_score END) as avg_goals_for,
            AVG(CASE WHEN m.home_team_id = %s THEN m.away_score ELSE m.home_score END) as avg_goals_against
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN opposition_categories pts ON 
            (pts.team_id = CASE WHEN m.home_team_id = %s THEN m.away_team_id ELSE m.home_team_id END)
            AND pts.season_id = m.season_id
        WHERE (m.home_team_id = %s OR m.away_team_id = %s)
            AND m.status = 'finished'
            AND m.season_id = %s
        GROUP BY opposition_strength
        """
        
        params = [tuple(top6_teams), tuple(top6_teams)] + [self.mufc_team_id] * 7 + [self.current_season_id]
        df = self.execute_query(query, params)
        
        if not df.empty:
            df['win_rate'] = df['wins'] / df['matches_played'] * 100
            df['points_per_game'] = (df['wins'] * 3 + df['draws']) / df['matches_played']
        
        return df.to_dict('records')
    
    def get_form_analysis(self, last_n_games: int = 10) -> Dict:
        """Analyze recent form over last N games"""
        query = """
        SELECT 
            m.match_date,
            ht.short_name as home_team,
            at.short_name as away_team,
            m.home_score,
            m.away_score,
            CASE 
                WHEN m.home_team_id = %s THEN 'Home'
                ELSE 'Away'
            END as venue,
            CASE 
                WHEN (m.home_team_id = %s AND m.home_score > m.away_score)
                OR (m.away_team_id = %s AND m.away_score > m.home_score)
                THEN 'W'
                WHEN m.home_score = m.away_score THEN 'D'
                ELSE 'L'
            END as result,
            CASE WHEN m.home_team_id = %s THEN m.home_score ELSE m.away_score END as goals_for,
            CASE WHEN m.home_team_id = %s THEN m.away_score ELSE m.home_score END as goals_against
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE (m.home_team_id = %s OR m.away_team_id = %s)
            AND m.status = 'finished'
            AND m.season_id = %s
        ORDER BY m.match_date DESC
        LIMIT %s
        """
        
        params = [self.mufc_team_id] * 7 + [self.current_season_id, last_n_games]
        df = self.execute_query(query, params)
        
        if df.empty:
            return {}
        
        # Calculate form metrics
        form_string = ''.join(df['result'].values)
        wins = (df['result'] == 'W').sum()
        draws = (df['result'] == 'D').sum()
        losses = (df['result'] == 'L').sum()
        
        return {
            'form_string': form_string,
            'matches_played': len(df),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': (wins / len(df)) * 100,
            'points': wins * 3 + draws,
            'points_per_game': (wins * 3 + draws) / len(df),
            'goals_for': df['goals_for'].sum(),
            'goals_against': df['goals_against'].sum(),
            'goal_difference': df['goals_for'].sum() - df['goals_against'].sum(),
            'recent_matches': df.to_dict('records')
        }

class PlayerAnalyzer(AnalyticsBase):
    """Analyze individual player performances"""
    
    def get_top_performers(self, position: Optional[str] = None, min_minutes: int = 180) -> pd.DataFrame:
        """Get top performing players by various metrics"""
        position_filter = "AND p.position = %s" if position else ""
        params = [self.mufc_team_id, self.current_season_id, min_minutes]
        if position:
            params.append(position)
        
        query = f"""
        SELECT 
            p.name,
            p.position,
            p.squad_number,
            COUNT(pms.match_id) as appearances,
            SUM(pms.minutes_played) as total_minutes,
            AVG(pms.minutes_played) as avg_minutes_per_game,
            SUM(pms.goals) as goals,
            SUM(pms.assists) as assists,
            SUM(pms.goals + pms.assists) as goal_contributions,
            SUM(pms.key_passes) as key_passes,
            SUM(pms.shots_total) as shots,
            SUM(pms.shots_on_target) as shots_on_target,
            CASE WHEN SUM(pms.shots_total) > 0 
                THEN (SUM(pms.shots_on_target)::float / SUM(pms.shots_total) * 100)
                ELSE 0 END as shot_accuracy,
            SUM(pms.passes_total) as total_passes,
            SUM(pms.passes_accurate) as accurate_passes,
            CASE WHEN SUM(pms.passes_total) > 0 
                THEN (SUM(pms.passes_accurate)::float / SUM(pms.passes_total) * 100)
                ELSE 0 END as pass_accuracy,
            SUM(pms.tackles) as tackles,
            SUM(pms.interceptions) as interceptions,
            SUM(pms.clearances) as clearances,
            AVG(pms.rating) as avg_rating,
            -- Goals per 90 minutes
            CASE WHEN SUM(pms.minutes_played) > 0 
                THEN (SUM(pms.goals)::float / SUM(pms.minutes_played) * 90)
                ELSE 0 END as goals_per_90,
            -- Assists per 90 minutes  
            CASE WHEN SUM(pms.minutes_played) > 0 
                THEN (SUM(pms.assists)::float / SUM(pms.minutes_played) * 90)
                ELSE 0 END as assists_per_90,
            -- Key passes per 90
            CASE WHEN SUM(pms.minutes_played) > 0 
                THEN (SUM(pms.key_passes)::float / SUM(pms.minutes_played) * 90)
                ELSE 0 END as key_passes_per_90
        FROM players p
        JOIN player_match_stats pms ON p.id = pms.player_id
        JOIN matches m ON pms.match_id = m.id
        WHERE p.team_id = %s
            AND m.season_id = %s
            AND pms.minutes_played > 0
            {position_filter}
        GROUP BY p.id, p.name, p.position, p.squad_number
        HAVING SUM(pms.minutes_played) >= %s
        ORDER BY goal_contributions DESC, avg_rating DESC
        """
        
        return self.execute_query(query, params)
    
    def get_unsung_heroes(self, min_minutes: int = 450) -> pd.DataFrame:
        """Identify defensive players making significant contributions"""
        query = """
        SELECT 
            p.name,
            p.position,
            COUNT(pms.match_id) as appearances,
            SUM(pms.minutes_played) as total_minutes,
            SUM(pms.tackles) as tackles,
            SUM(pms.interceptions) as interceptions,
            SUM(pms.clearances) as clearances,
            SUM(pms.tackles + pms.interceptions) as defensive_actions,
            AVG(pms.rating) as avg_rating,
            -- Defensive actions per 90
            CASE WHEN SUM(pms.minutes_played) > 0 
                THEN ((SUM(pms.tackles) + SUM(pms.interceptions))::float / SUM(pms.minutes_played) * 90)
                ELSE 0 END as defensive_actions_per_90,
            -- Clean sheet percentage (for defenders)
            CASE WHEN p.position IN ('DEF', 'GK') THEN
                (SELECT COUNT(*)::float / COUNT(pms.match_id) * 100
                 FROM matches m2 
                 WHERE m2.id = pms.match_id 
                   AND ((m2.home_team_id = %s AND m2.away_score = 0) 
                        OR (m2.away_team_id = %s AND m2.home_score = 0)))
                ELSE NULL 
            END as clean_sheet_percentage
        FROM players p
        JOIN player_match_stats pms ON p.id = pms.player_id
        JOIN matches m ON pms.match_id = m.id
        WHERE p.team_id = %s
            AND m.season_id = %s
            AND pms.minutes_played > 0
            AND p.position IN ('DEF', 'MID', 'GK')
        GROUP BY p.id, p.name, p.position
        HAVING SUM(pms.minutes_played) >= %s
        ORDER BY defensive_actions_per_90 DESC, avg_rating DESC
        """
        
        params = [self.mufc_team_id, self.mufc_team_id, self.mufc_team_id, self.current_season_id, min_minutes]
        return self.execute_query(query, params)
    
    def analyze_player_form(self, player_name: str, last_n_games: int = 5) -> Dict:
        """Analyze individual player form over recent games"""
        query = """
        SELECT 
            m.match_date,
            ht.short_name as home_team,
            at.short_name as away_team,
            pms.minutes_played,
            pms.goals,
            pms.assists,
            pms.shots_total,
            pms.shots_on_target,
            pms.key_passes,
            pms.passes_total,
            pms.passes_accurate,
            pms.tackles,
            pms.interceptions,
            pms.rating
        FROM player_match_stats pms
        JOIN players p ON pms.player_id = p.id
        JOIN matches m ON pms.match_id = m.id
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE p.name = %s
            AND p.team_id = %s
            AND m.season_id = %s
            AND pms.minutes_played > 0
        ORDER BY m.match_date DESC
        LIMIT %s
        """
        
        params = [player_name, self.mufc_team_id, self.current_season_id, last_n_games]
        df = self.execute_query(query, params)
        
        if df.empty:
            return {}
        
        return {
            'player': player_name,
            'recent_form': df.to_dict('records'),
            'avg_rating': df['rating'].mean(),
            'total_goals': df['goals'].sum(),
            'total_assists': df['assists'].sum(),
            'avg_passes_per_game': df['passes_total'].mean(),
            'avg_pass_accuracy': (df['passes_accurate'].sum() / df['passes_total'].sum()) * 100 if df['passes_total'].sum() > 0 else 0
        }

class TacticalAnalyzer(AnalyticsBase):
    """Analyze tactical aspects and formations"""
    
    def analyze_formation_effectiveness(self) -> pd.DataFrame:
        """Analyze how different formations perform"""
        query = """
        SELECT 
            ms.formation,
            COUNT(*) as matches_played,
            SUM(CASE 
                WHEN (m.home_team_id = %s AND m.home_score > m.away_score)
                OR (m.away_team_id = %s AND m.away_score > m.home_score)
                THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN m.home_score = m.away_score THEN 1 ELSE 0 END) as draws,
            AVG(ms.possession_percent) as avg_possession,
            AVG(ms.shots_total) as avg_shots,
            AVG(ms.shots_on_target) as avg_shots_on_target,
            AVG(ms.pass_accuracy_percent) as avg_pass_accuracy,
            AVG(CASE WHEN m.home_team_id = %s THEN m.home_score ELSE m.away_score END) as avg_goals_for,
            AVG(CASE WHEN m.home_team_id = %s THEN m.away_score ELSE m.home_score END) as avg_goals_against
        FROM match_stats ms
        JOIN matches m ON ms.match_id = m.id
        WHERE ms.team_id = %s
            AND m.season_id = %s
            AND m.status = 'finished'
            AND ms.formation IS NOT NULL
        GROUP BY ms.formation
        HAVING COUNT(*) >= 3  -- Only formations used 3+ times
        ORDER BY wins DESC, avg_goals_for DESC
        """
        
        params = [self.mufc_team_id] * 5 + [self.current_season_id]
        df = self.execute_query(query, params)
        
        if not df.empty:
            df['win_rate'] = (df['wins'] / df['matches_played']) * 100
            df['points_per_game'] = (df['wins'] * 3 + df['draws']) / df['matches_played']
            df['goal_difference_per_game'] = df['avg_goals_for'] - df['avg_goals_against']
        
        return df
    
    def analyze_possession_vs_results(self) -> Dict:
        """Analyze correlation between possession and match results"""
        query = """
        SELECT 
            ms.possession_percent,
            CASE 
                WHEN (m.home_team_id = %s AND m.home_score > m.away_score)
                OR (m.away_team_id = %s AND m.away_score > m.home_score)
                THEN 'Win'
                WHEN m.home_score = m.away_score THEN 'Draw'
                ELSE 'Loss'
            END as result,
            CASE WHEN m.home_team_id = %s THEN m.home_score ELSE m.away_score END as goals_for,
            ms.shots_total,
            ms.shots_on_target,
            ms.pass_accuracy_percent
        FROM match_stats ms
        JOIN matches m ON ms.match_id = m.id
        WHERE ms.team_id = %s
            AND m.season_id = %s
            AND m.status = 'finished'
            AND ms.possession_percent IS NOT NULL
        ORDER BY ms.possession_percent DESC
        """
        
        params = [self.mufc_team_id] * 4 + [self.current_season_id]
        df = self.execute_query(query, params)
        
        if df.empty:
            return {}
        
        # Calculate correlations
        possession_goals_corr = df['possession_percent'].corr(df['goals_for'])
        possession_shots_corr = df['possession_percent'].corr(df['shots_total'])
        
        # Group by result
        result_analysis = df.groupby('result').agg({
            'possession_percent': ['mean', 'median'],
            'goals_for': 'mean',
            'shots_total': 'mean',
            'pass_accuracy_percent': 'mean'
        }).round(2)
        
        return {
            'possession_goals_correlation': possession_goals_corr,
            'possession_shots_correlation': possession_shots_corr,
            'by_result': result_analysis.to_dict(),
            'raw_data': df.to_dict('records')
        }
    
    def analyze_shot_maps(self) -> Dict:
        """Analyze shot locations and effectiveness"""
        query = """
        SELECT 
            s.x_coordinate,
            s.y_coordinate,
            s.shot_type,
            s.situation,
            s.is_goal,
            s.is_on_target,
            s.xg_value,
            p.name as player_name,
            p.position,
            m.match_date,
            ht.short_name as home_team,
            at.short_name as away_team
        FROM shots s
        JOIN players p ON s.player_id = p.id
        JOIN matches m ON s.match_id = m.id
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE s.team_id = %s
            AND m.season_id = %s
        ORDER BY m.match_date DESC
        """
        
        params = [self.mufc_team_id, self.current_season_id]
        df = self.execute_query(query, params)
        
        if df.empty:
            return {}
        
        # Analyze shot zones
        df['shot_zone'] = df.apply(self._categorize_shot_zone, axis=1)
        
        zone_analysis = df.groupby('shot_zone').agg({
            'is_goal': ['count', 'sum', 'mean'],
            'is_on_target': 'mean',
            'xg_value': 'mean'
        }).round(3)
        
        return {
            'total_shots': len(df),
            'goals_scored': df['is_goal'].sum(),
            'shot_accuracy': df['is_on_target'].mean() * 100,
            'conversion_rate': df['is_goal'].mean() * 100,
            'avg_xg_per_shot': df['xg_value'].mean(),
            'zone_analysis': zone_analysis.to_dict(),
            'shot_data': df.to_dict('records')
        }
    
    def _categorize_shot_zone(self, row) -> str:
        """Categorize shot based on coordinates"""
        x, y = row['x_coordinate'], row['y_coordinate']
        
        if pd.isna(x) or pd.isna(y):
            return 'Unknown'
        
        # Assuming pitch coordinates: x=0 is own goal, x=100 is opponent goal
        # y=0 is left side, y=100 is right side (from perspective)
        
        if x >= 83:  # Close to goal
            if 20 <= y <= 80:  # Central area
                return 'Six Yard Box'
            else:
                return 'Wide Close'
        elif x >= 67:  # Penalty area
            if 30 <= y <= 70:
                return 'Central Penalty Area'
            else:
                return 'Wide Penalty Area'
        elif x >= 50:  # Midfield attacking
            return 'Outside Box'
        else:
            return 'Long Range'

# Visualization Functions

def create_performance_dashboard(performance_analyzer: PerformanceAnalyzer):
    """Create performance visualization dashboard"""
    
    # Get data
    home_away = performance_analyzer.analyze_home_vs_away_performance()
    top6_bottom = performance_analyzer.analyze_top6_vs_bottom_performance()
    form = performance_analyzer.get_form_analysis()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Home vs Away Performance', 'Opposition Strength Analysis',
                       'Recent Form', 'Goals For/Against'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Home vs Away chart
    if home_away:
        ha_df = pd.DataFrame(home_away)
        fig.add_trace(
            go.Bar(x=ha_df['venue_type'], y=ha_df['win_rate'], 
                   name='Win Rate %', marker_color='red'),
            row=1, col=1
        )
    
    # Opposition strength chart  
    if top6_bottom:
        opp_df = pd.DataFrame(top6_bottom)
        fig.add_trace(
            go.Bar(x=opp_df['opposition_strength'], y=opp_df['win_rate'],
                   name='Win Rate vs Opposition', marker_color='blue'),
            row=1, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title_text="Manchester United Performance Analysis")
    return fig

# Usage Example
if __name__ == "__main__":
    # Example configuration
    db_config = {
        'host': 'localhost',
        'database': 'mufc_analytics',
        'user': 'postgres',
        'password': 'password'
    }
    
    # Initialize analyzers
    perf_analyzer = PerformanceAnalyzer(db_config)
    player_analyzer = PlayerAnalyzer(db_config)
    tactical_analyzer = TacticalAnalyzer(db_config)
    
    # Example analyses
    print("=== PERFORMANCE ANALYSIS ===")
    home_away = perf_analyzer.analyze_home_vs_away_performance()
    print(f"Home vs Away: {home_away}")
    
    print("\n=== PLAYER ANALYSIS ===")
    top_players = player_analyzer.get_top_performers()
    print(f"Top 5 performers:\n{top_players.head()}")
    
    print("\n=== TACTICAL ANALYSIS ===")
    formations = tactical_analyzer.analyze_formation_effectiveness()
    print(f"Formation effectiveness:\n{formations}")