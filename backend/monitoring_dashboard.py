"""
System Monitoring and Testing Suite
Health checks, data quality monitoring, and performance testing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import requests
import psycopg2
from typing import Dict, List, Optional, Tuple
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os

logger = logging.getLogger(__name__)

class SystemHealthChecker:
    """Monitor system health and data quality"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.health_status = {}
        
    def run_comprehensive_health_check(self) -> Dict:
        """Run all health checks and return status"""
        
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # Database connectivity
        health_results['checks']['database'] = self._check_database_health()
        
        # Data quality checks
        health_results['checks']['data_quality'] = self._check_data_quality()
        
        # API connectivity
        health_results['checks']['apis'] = self._check_api_health()
        
        # Data freshness
        health_results['checks']['data_freshness'] = self._check_data_freshness()
        
        # System performance
        health_results['checks']['performance'] = self._check_system_performance()
        
        # Determine overall status
        failed_checks = [name for name, result in health_results['checks'].items() 
                        if result.get('status') != 'healthy']
        
        if failed_checks:
            health_results['overall_status'] = 'warning' if len(failed_checks) <= 2 else 'critical'
            health_results['failed_checks'] = failed_checks
        
        logger.info(f"Health check completed. Status: {health_results['overall_status']}")
        return health_results
    
    def _check_database_health(