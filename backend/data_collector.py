"""
Manchester United Premier League Data Collector
Zero-budget solution using free APIs and web scraping
Google Cloud Functions compatible
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchData:
    """Data structure for match information"""
    gameweek: int
    home_team: str
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]
    match_date: datetime
    venue: str
    status: str

class DatabaseManager:
    """Handle database connections and operations"""
    
    def __init__(self):
        # Google Cloud SQL connection
        # Set these as environment variables in Cloud Functions
        self.db_config = {
            'host': os.getenv('DB_HOST', '/cloudsql/your-project:region:instance'),
            'database': os.getenv('DB_NAME', 'mufc_analytics'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD'),
        }
    
    def get_connection(self):
        """Create database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def execute_query(self, query: str, params=None, fetch=False):
        """Execute database query"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                conn.commit()
    
    def log_collection(self, source: str, data_type: str, records: int, 
                      success: bool, error_msg: str = None, exec_time: float = 0):
        """Log data collection attempts"""
        query = """
        INSERT INTO data_collection_log 
        (source, data_type, records_collected, success, error_message, execution_time_seconds)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (source, data_type, records, success, error_msg, exec_time)
        self.execute_query(query, params)

class FPLAPICollector:
    """Collect data from Fantasy Premier League API (completely free)"""
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_bootstrap_data(self) -> Dict:
        """Get general info including teams and players"""
        start_time = time.time()
        try:
            response = self.session.get(f"{self.BASE_URL}/bootstrap-static/")
            response.raise_for_status()
            data = response.json()
            
            exec_time = time.time() - start_time
            self.db.log_collection('fpl_api', 'bootstrap', len(data.get('elements', [])), 
                                 True, None, exec_time)
            return data
        except Exception as e:
            exec_time = time.time() - start_time
            self.db.log_collection('fpl_api', 'bootstrap', 0, False, str(e), exec_time)
            logger.error(f"FPL bootstrap failed: {e}")
            return {}
    
    def get_fixtures(self) -> List[Dict]:
        """Get fixture list"""
        start_time = time.time()
        try:
            response = self.session.get(f"{self.BASE_URL}/fixtures/")
            response.raise_for_status()
            fixtures = response.json()
            
            exec_time = time.time() - start_time
            self.db.log_collection('fpl_api', 'fixtures', len(fixtures), 
                                 True, None, exec_time)
            return fixtures
        except Exception as e:
            exec_time = time.time() - start_time
            self.db.log_collection('fpl_api', 'fixtures', 0, False, str(e), exec_time)
            logger.error(f"FPL fixtures failed: {e}")
            return []
    
    def get_gameweek_data(self, gameweek: int) -> Dict:
        """Get specific gameweek data"""
        start_time = time.time()
        try:
            response = self.session.get(f"{self.BASE_URL}/event/{gameweek}/live/")
            response.raise_for_status()
            data = response.json()
            
            exec_time = time.time() - start_time
            self.db.log_collection('fpl_api', f'gameweek_{gameweek}', 
                                 len(data.get('elements', [])), True, None, exec_time)
            return data
        except Exception as e:
            exec_time = time.time() - start_time
            self.db.log_collection('fpl_api', f'gameweek_{gameweek}', 0, False, str(e), exec_time)
            logger.error(f"FPL gameweek {gameweek} failed: {e}")
            return {}

class FootballDataAPICollector:
    """Football-Data.org API (free tier: 10 calls/minute)"""
    
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self, db_manager: DatabaseManager, api_key: str):
        self.db = db_manager
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'X-Auth-Token': api_key,
            'User-Agent': 'MUFC-Analytics/1.0'
        })
        self.rate_limit_delay = 6  # 10 calls per minute = 6 second delay
    
    def get_premier_league_matches(self) -> List[Dict]:
        """Get Premier League matches"""
        start_time = time.time()
        try:
            # Premier League ID is 2021
            response = self.session.get(
                f"{self.BASE_URL}/competitions/2021/matches"
            )
            response.raise_for_status()
            data = response.json()
            
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            exec_time = time.time() - start_time
            self.db.log_collection('football_data_api', 'pl_matches', 
                                 len(data.get('matches', [])), True, None, exec_time)
            return data.get('matches', [])
        except Exception as e:
            exec_time = time.time() - start_time
            self.db.log_collection('football_data_api', 'pl_matches', 0, False, str(e), exec_time)
            logger.error(f"Football-Data API failed: {e}")
            return []
    
    def get_team_matches(self, team_id: int) -> List[Dict]:
        """Get specific team matches"""
        start_time = time.time()
        try:
            response = self.session.get(
                f"{self.BASE_URL}/teams/{team_id}/matches"
            )
            response.raise_for_status()
            data = response.json()
            
            time.sleep(self.rate_limit_delay)
            
            exec_time = time.time() - start_time
            self.db.log_collection('football_data_api', f'team_{team_id}_matches', 
                                 len(data.get('matches', [])), True, None, exec_time)
            return data.get('matches', [])
        except Exception as e:
            exec_time = time.time() - start_time
            self.db.log_collection('football_data_api', f'team_{team_id}_matches', 0, False, str(e), exec_time)
            return []

class BBCSportScraper:
    """Scrape BBC Sport for additional match data"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_premier_league_results(self) -> List[Dict]:
        """Scrape BBC Sport Premier League results"""
        start_time = time.time()
        try:
            # This is a simplified example - you'd need to parse HTML
            url = "https://www.bbc.co.uk/sport/football/premier-league/results"
            response = self.session.get(url)
            response.raise_for_status()
            
            # Here you would parse HTML using BeautifulSoup
            # For brevity, returning empty list
            results = []
            
            time.sleep(2)  # Be respectful with scraping
            
            exec_time = time.time() - start_time
            self.db.log_collection('bbc_scrape', 'results', len(results), True, None, exec_time)
            return results
        except Exception as e:
            exec_time = time.time() - start_time
            self.db.log_collection('bbc_scrape', 'results', 0, False, str(e), exec_time)
            logger.error(f"BBC scraping failed: {e}")
            return []

class DataProcessor:
    """Process and store collected data"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def process_fpl_teams(self, bootstrap_data: Dict):
        """Process FPL team data"""
        teams = bootstrap_data.get('teams', [])
        
        for team in teams:
            # Check if team exists
            check_query = "SELECT id FROM teams WHERE name = %s"
            existing = self.db.execute_query(check_query, (team['name'],), fetch=True)
            
            if not existing:
                insert_query = """
                INSERT INTO teams (name, short_name, founded_year, stadium)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING
                """
                params = (
                    team['name'],
                    team['short_name'],
                    None,  # FPL doesn't provide founding year
                    None   # FPL doesn't provide stadium
                )
                self.db.execute_query(insert_query, params)
    
    def process_fpl_players(self, bootstrap_data: Dict):
        """Process FPL player data"""
        players = bootstrap_data.get('elements', [])
        teams = {t['id']: t['name'] for t in bootstrap_data.get('teams', [])}
        
        for player in players:
            team_name = teams.get(player['team'])
            
            # Get team_id from database
            team_query = "SELECT id FROM teams WHERE name = %s"
            team_result = self.db.execute_query(team_query, (team_name,), fetch=True)
            
            if team_result:
                team_id = team_result[0]['id']
                
                insert_query = """
                INSERT INTO players (name, position, squad_number, team_id, is_active)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """
                # Map FPL positions (1=GK, 2=DEF, 3=MID, 4=FWD)
                position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                position = position_map.get(player['element_type'], 'UNK')
                
                params = (
                    f"{player['first_name']} {player['second_name']}",
                    position,
                    player.get('code', 0) % 100,  # Rough squad number approximation
                    team_id,
                    player['status'] != 'u'  # Available players
                )
                self.db.execute_query(insert_query, params)
    
    def process_fixtures(self, fixtures: List[Dict]):
        """Process fixture data"""
        for fixture in fixtures:
            if not fixture.get('started'):
                continue
                
            # Convert FPL team IDs to our team IDs
            home_team_query = "SELECT id FROM teams WHERE name = (SELECT name FROM teams WHERE id = %s)"
            away_team_query = "SELECT id FROM teams WHERE name = (SELECT name FROM teams WHERE id = %s)"
            
            # This is simplified - you'd need proper team ID mapping
            insert_query = """
            INSERT INTO matches 
            (season_id, gameweek, match_date, home_team_id, away_team_id, 
             home_score, away_score, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """
            
            # Process each fixture...

# Main Collection Functions (for Google Cloud Functions)

def collect_weekly_data(request=None):
    """Main function for weekly data collection"""
    logger.info("Starting weekly data collection...")
    
    db = DatabaseManager()
    
    try:
        # Initialize collectors
        fpl_collector = FPLAPICollector(db)
        
        # Get FPL data
        bootstrap_data = fpl_collector.get_bootstrap_data()
        fixtures = fpl_collector.get_fixtures()
        
        # Process data
        processor = DataProcessor(db)
        if bootstrap_data:
            processor.process_fpl_teams(bootstrap_data)
            processor.process_fpl_players(bootstrap_data)
        
        if fixtures:
            processor.process_fixtures(fixtures)
        
        logger.info("Weekly data collection completed successfully")
        return {"status": "success", "message": "Data collection completed"}
        
    except Exception as e:
        logger.error(f"Weekly collection failed: {e}")
        return {"status": "error", "message": str(e)}

def collect_match_day_data(request=None):
    """Function for match day data collection (more frequent)"""
    logger.info("Starting match day data collection...")
    
    # Similar structure but focused on live match data
    # Run every hour on match days
    
    return {"status": "success"}

# For local testing
if __name__ == "__main__":
    # Test the collectors locally
    collect_weekly_data()