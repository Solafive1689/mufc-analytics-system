-- Manchester United Premier League 2025/26 Analytics Database Schema
-- Optimized for Google Cloud SQL PostgreSQL Free Tier

-- Core Teams Table
CREATE TABLE teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    short_name VARCHAR(10),
    founded_year INTEGER,
    stadium VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seasons Table
CREATE TABLE seasons (
    id SERIAL PRIMARY KEY,
    season_name VARCHAR(20) NOT NULL, -- e.g., '2025/26'
    start_date DATE,
    end_date DATE,
    is_current BOOLEAN DEFAULT FALSE
);

-- Matches Table - Core fixture and result data
CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    season_id INTEGER REFERENCES seasons(id),
    gameweek INTEGER,
    match_date TIMESTAMP,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    home_score INTEGER,
    away_score INTEGER,
    venue VARCHAR(100),
    referee VARCHAR(100),
    attendance INTEGER,
    status VARCHAR(20) DEFAULT 'scheduled', -- scheduled, live, finished, postponed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Match Statistics - Team-level stats per match
CREATE TABLE match_stats (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    team_id INTEGER REFERENCES teams(id),
    is_home BOOLEAN,
    possession_percent DECIMAL(5,2),
    shots_total INTEGER,
    shots_on_target INTEGER,
    shots_off_target INTEGER,
    shots_blocked INTEGER,
    corners INTEGER,
    fouls INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    passes_total INTEGER,
    passes_accurate INTEGER,
    pass_accuracy_percent DECIMAL(5,2),
    formation VARCHAR(10), -- e.g., '4-2-3-1'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players Table
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    position VARCHAR(20), -- GK, DEF, MID, FWD
    nationality VARCHAR(50),
    age INTEGER,
    squad_number INTEGER,
    team_id INTEGER REFERENCES teams(id),
    market_value_millions DECIMAL(10,2),
    contract_until DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player Match Performance
CREATE TABLE player_match_stats (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    player_id INTEGER REFERENCES players(id),
    team_id INTEGER REFERENCES teams(id),
    started BOOLEAN DEFAULT FALSE,
    minutes_played INTEGER DEFAULT 0,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    shots_total INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    key_passes INTEGER DEFAULT 0,
    passes_total INTEGER DEFAULT 0,
    passes_accurate INTEGER DEFAULT 0,
    tackles INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    clearances INTEGER DEFAULT 0,
    rating DECIMAL(3,1), -- Match rating out of 10
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shot Map Data - For tactical analysis
CREATE TABLE shots (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    player_id INTEGER REFERENCES players(id),
    team_id INTEGER REFERENCES teams(id),
    minute INTEGER,
    x_coordinate DECIMAL(5,2), -- Pitch coordinates (0-100)
    y_coordinate DECIMAL(5,2), -- Pitch coordinates (0-100)
    shot_type VARCHAR(20), -- header, right_foot, left_foot, etc.
    body_part VARCHAR(20),
    situation VARCHAR(30), -- open_play, corner, free_kick, penalty, etc.
    is_goal BOOLEAN DEFAULT FALSE,
    is_on_target BOOLEAN DEFAULT FALSE,
    is_blocked BOOLEAN DEFAULT FALSE,
    xg_value DECIMAL(4,3), -- Expected Goals value
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- League Table Snapshots - Track position over time
CREATE TABLE league_table_snapshots (
    id SERIAL PRIMARY KEY,
    season_id INTEGER REFERENCES seasons(id),
    gameweek INTEGER,
    team_id INTEGER REFERENCES teams(id),
    position INTEGER,
    matches_played INTEGER,
    wins INTEGER,
    draws INTEGER,
    losses INTEGER,
    goals_for INTEGER,
    goals_against INTEGER,
    goal_difference INTEGER,
    points INTEGER,
    form VARCHAR(10), -- Last 5 results: WWDLL
    snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Opposition Analysis - Track performance vs different opposition types
CREATE TABLE opposition_categories (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id),
    season_id INTEGER REFERENCES seasons(id),
    category VARCHAR(20), -- 'top_6', 'mid_table', 'relegation_battle'
    final_position INTEGER, -- End of season position for historical categorization
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data Collection Log - Track scraping and API calls
CREATE TABLE data_collection_log (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50), -- 'fpl_api', 'bbc_scrape', 'football_data_api'
    data_type VARCHAR(50), -- 'match_results', 'player_stats', 'fixtures'
    records_collected INTEGER,
    success BOOLEAN,
    error_message TEXT,
    execution_time_seconds DECIMAL(8,3),
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance Optimization
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_teams ON matches(home_team_id, away_team_id);
CREATE INDEX idx_player_match_stats_match ON player_match_stats(match_id);
CREATE INDEX idx_player_match_stats_player ON player_match_stats(player_id);
CREATE INDEX idx_shots_match ON shots(match_id);
CREATE INDEX idx_league_table_gameweek ON league_table_snapshots(season_id, gameweek);
CREATE INDEX idx_data_log_source_date ON data_collection_log(source, collected_at);

-- Insert initial data
INSERT INTO seasons (season_name, start_date, end_date, is_current) 
VALUES ('2025/26', '2025-08-16', '2026-05-24', TRUE);

-- Insert Manchester United
INSERT INTO teams (name, short_name, founded_year, stadium) 
VALUES ('Manchester United', 'MUN', 1878, 'Old Trafford');

-- Views for Common Queries

-- Manchester United Home vs Away Performance
CREATE VIEW united_home_away_form AS
SELECT 
    CASE 
        WHEN m.home_team_id = (SELECT id FROM teams WHERE short_name = 'MUN') THEN 'Home'
        ELSE 'Away'
    END as venue_type,
    COUNT(*) as matches_played,
    SUM(CASE 
        WHEN (m.home_team_id = (SELECT id FROM teams WHERE short_name = 'MUN') AND m.home_score > m.away_score)
        OR (m.away_team_id = (SELECT id FROM teams WHERE short_name = 'MUN') AND m.away_score > m.home_score)
        THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN m.home_score = m.away_score THEN 1 ELSE 0 END) as draws,
    SUM(CASE 
        WHEN (m.home_team_id = (SELECT id FROM teams WHERE short_name = 'MUN') AND m.home_score < m.away_score)
        OR (m.away_team_id = (SELECT id FROM teams WHERE short_name = 'MUN') AND m.away_score < m.home_score)
        THEN 1 ELSE 0 END) as losses
FROM matches m
WHERE (m.home_team_id = (SELECT id FROM teams WHERE short_name = 'MUN') 
       OR m.away_team_id = (SELECT id FROM teams WHERE short_name = 'MUN'))
    AND m.status = 'finished'
    AND m.season_id = (SELECT id FROM seasons WHERE is_current = TRUE)
GROUP BY venue_type;

-- Top Performers View
CREATE VIEW united_top_performers AS
SELECT 
    p.name,
    p.position,
    COUNT(pms.match_id) as matches_played,
    SUM(pms.minutes_played) as total_minutes,
    SUM(pms.goals) as goals,
    SUM(pms.assists) as assists,
    SUM(pms.key_passes) as key_passes,
    SUM(pms.tackles) as tackles,
    SUM(pms.interceptions) as interceptions,
    AVG(pms.rating) as avg_rating
FROM players p
JOIN player_match_stats pms ON p.id = pms.player_id
JOIN matches m ON pms.match_id = m.id
WHERE p.team_id = (SELECT id FROM teams WHERE short_name = 'MUN')
    AND m.season_id = (SELECT id FROM seasons WHERE is_current = TRUE)
    AND pms.minutes_played > 0
GROUP BY p.id, p.name, p.position
ORDER BY goals DESC, assists DESC;