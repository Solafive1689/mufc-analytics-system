"""
Main entry point for Google Cloud Functions
All function entry points for the MUFC Analytics system
"""

from data_collector import collect_weekly_data, collect_match_day_data
from advanced_scraper import run_scraping_job
from ml_models import run_ml_training
from automated_reports import run_weekly_reporting, check_match_results
from monitoring_dashboard import run_system_monitoring

# Cloud Function entry points - these names must match deployment commands
def weekly_data_collection(request):
    """Weekly data collection endpoint"""
    return collect_weekly_data(request)

def matchday_data_collection(request):
    """Match day data collection endpoint"""
    return collect_match_day_data(request)

def advanced_scraping(request):
    """Advanced web scraping endpoint"""
    return run_scraping_job(request)

def ml_training(request):
    """ML model training endpoint"""
    return run_ml_training(request)

def weekly_reporting(request):
    """Weekly reporting endpoint"""
    return run_weekly_reporting(request)

def match_results_check(request):
    """Match results checking endpoint"""
    return check_match_results(request)

def system_monitoring(request):
    """System monitoring endpoint"""
    return run_system_monitoring(request)

# Health check endpoint
def health_check(request):
    """Simple health check for all functions"""
    return {
        'status': 'healthy',
        'message': 'MUFC Analytics system is running',
        'functions': [
            'weekly_data_collection',
            'matchday_data_collection', 
            'advanced_scraping',
            'ml_training',
            'weekly_reporting',
            'system_monitoring'
        ]
    }