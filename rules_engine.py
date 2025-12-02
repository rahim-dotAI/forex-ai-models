#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
🎯 RULES ENGINE - Advanced Condition Evaluator
═══════════════════════════════════════════════════════════════

This module evaluates rule conditions from rules.json and
determines which rules should be triggered.

It supports:
- Simple conditions (wr < 40)
- Complex conditions (wr < 40 for 3 days)
- Time-based conditions (is_weekend)
- Trend detection (increasing, decreasing)
- Historical analysis

═══════════════════════════════════════════════════════════════
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class RulesEngine:
    """
    Evaluates rule conditions from rules.json
    """
    
    def __init__(self, memory: Dict = None):
        self.memory = memory or {}
        self.historical_metrics = self.load_historical_metrics()
    
    def load_historical_metrics(self):
        """Load historical metrics for trend analysis"""
        # In real implementation, would load from database
        return {
            'win_rates': [],
            'profits': [],
            'timestamps': []
        }
    
    def evaluate_condition(self, condition: str, current_metrics: Dict) -> bool:
        """
        Main entry point: Evaluate any rule condition
        
        Args:
            condition: The condition string from rules.json
            current_metrics: Current system metrics
        
        Returns:
            True if condition is met, False otherwise
        """
        
        # Remove extra whitespace
        condition = condition.strip()
        
        # Parse and evaluate different condition types
        
        # 1. Simple comparison (e.g., "weekday_wr < 40")
        if self._is_simple_comparison(condition):
            return self._evaluate_simple_comparison(condition, current_metrics)
        
        # 2. Consecutive occurrences (e.g., "wr < 40 for 3 days")
        elif " for " in condition and ("day" in condition or "weekend" in condition):
            return self._evaluate_consecutive_condition(condition, current_metrics)
        
        # 3. Trend detection (e.g., "atr_increasing_3_days")
        elif "_increasing" in condition or "_decreasing" in condition:
            return self._evaluate_trend_condition(condition, current_metrics)
        
        # 4. Time-based (e.g., "is_weekend")
        elif condition.startswith("is_"):
            return self._evaluate_time_condition(condition, current_metrics)
        
        # 5. Range-based (e.g., "price_within_1pct_of_20d_range")
        elif "_within_" in condition:
            return self._evaluate_range_condition(condition, current_metrics)
        
        # 6. Boolean conditions
        elif condition in ['always', 'true']:
            return True
        elif condition in ['never', 'false']:
            return False
        
        # 7. Complex conditions (use AND/OR)
        elif " and " in condition.lower() or " or " in condition.lower():
            return self._evaluate_complex_condition(condition, current_metrics)
        
        # Unknown condition type
        else:
            return False
    
    def _is_simple_comparison(self, condition: str) -> bool:
        """Check if condition is a simple comparison"""
        operators = ['<', '>', '<=', '>=', '==', '!=']
        return any(op in condition for op in operators)
    
    def _evaluate_simple_comparison(self, condition: str, metrics: Dict) -> bool:
        """
        Evaluate simple comparison
        Examples: "weekday_wr < 40", "total_profit > 500"
        """
        
        # Parse condition
        for op in ['<=', '>=', '==', '!=', '<', '>']:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    metric_path = parts[0].strip()
                    threshold = parts[1].strip()
                    
                    # Get metric value
                    value = self._get_metric_value(metric_path, metrics)
                    
                    if value is None:
                        return False
                    
                    # Convert threshold to number
                    try:
                        threshold_num = float(threshold)
                    except ValueError:
                        return False
                    
                    # Compare
                    if op == '<':
                        return value < threshold_num
                    elif op == '>':
                        return value > threshold_num
                    elif op == '<=':
                        return value <= threshold_num
                    elif op == '>=':
                        return value >= threshold_num
                    elif op == '==':
                        return value == threshold_num
                    elif op == '!=':
                        return value != threshold_num
        
        return False
    
    def _evaluate_consecutive_condition(self, condition: str, metrics: Dict) -> bool:
        """
        Evaluate consecutive occurrences
        Example: "weekday_wr < 40 for 3 consecutive days"
        """
        
        # Parse: "<base_condition> for <N> <unit>"
        match = re.search(r'(.+) for (\d+) (?:consecutive )?(\w+)', condition)
        
        if not match:
            return False
        
        base_condition = match.group(1).strip()
        count_needed = int(match.group(2))
        unit = match.group(3).strip()  # 'days', 'weekends', etc.
        
        # Would need historical data to properly evaluate
        # For now, simplified implementation
        
        # Check if we have historical data
        history = self.memory.get('historical_conditions', [])
        
        # Count how many times condition was true recently
        consecutive_count = 0
        for historical_state in reversed(history[-count_needed:]):
            if self.evaluate_condition(base_condition, historical_state):
                consecutive_count += 1
            else:
                break  # Streak broken
        
        return consecutive_count >= count_needed
    
    def _evaluate_trend_condition(self, condition: str, metrics: Dict) -> bool:
        """
        Evaluate trend conditions
        Example: "atr_increasing_3_days", "wr_decreasing"
        """
        
        # Parse condition
        match = re.search(r'(\w+)_(increasing|decreasing)(?:_(\d+)_days)?', condition)
        
        if not match:
            return False
        
        metric_name = match.group(1)
        direction = match.group(2)
        days = int(match.group(3)) if match.group(3) else 3
        
        # Get historical values
        historical = self.historical_metrics.get(metric_name, [])
        
        if len(historical) < days:
            return False
        
        # Check trend
        recent_values = historical[-days:]
        
        if direction == 'increasing':
            return all(recent_values[i] < recent_values[i+1] for i in range(len(recent_values)-1))
        else:  # decreasing
            return all(recent_values[i] > recent_values[i+1] for i in range(len(recent_values)-1))
    
    def _evaluate_time_condition(self, condition: str, metrics: Dict) -> bool:
        """
        Evaluate time-based conditions
        Examples: "is_weekend", "is_monday", "time_0000_0800_utc"
        """
        
        now = datetime.now()
        
        if condition == 'is_weekend':
            return now.weekday() in [5, 6]  # Saturday, Sunday
        
        elif condition.startswith('is_'):
            day_name = condition[3:]  # Remove "is_"
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2,
                'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
            }
            return now.weekday() == day_map.get(day_name.lower(), -1)
        
        elif condition.startswith('time_'):
            # Parse time range: time_0000_0800_utc
            match = re.search(r'time_(\d{4})_(\d{4})_utc', condition)
            if match:
                start_hour = int(match.group(1)[:2])
                end_hour = int(match.group(2)[:2])
                current_hour = now.hour
                
                if start_hour <= end_hour:
                    return start_hour <= current_hour < end_hour
                else:  # Wraps around midnight
                    return current_hour >= start_hour or current_hour < end_hour
        
        return False
    
    def _evaluate_range_condition(self, condition: str, metrics: Dict) -> bool:
        """
        Evaluate range-based conditions
        Example: "price_within_1pct_of_20d_range"
        """
        
        # Parse condition
        match = re.search(r'(\w+)_within_(\d+)pct_of_(\d+)d_range', condition)
        
        if not match:
            return False
        
        metric_name = match.group(1)
        percent = int(match.group(2))
        days = int(match.group(3))
        
        # Would need historical price data
        # Simplified for now
        return False
    
    def _evaluate_complex_condition(self, condition: str, metrics: Dict) -> bool:
        """
        Evaluate complex conditions with AND/OR
        Example: "weekday_wr < 40 and profit < -50"
        """
        
        condition_lower = condition.lower()
        
        if ' and ' in condition_lower:
            parts = condition.split(' and ')
            return all(self.evaluate_condition(part.strip(), metrics) for part in parts)
        
        elif ' or ' in condition_lower:
            parts = condition.split(' or ')
            return any(self.evaluate_condition(part.strip(), metrics) for part in parts)
        
        return False
    
    def _get_metric_value(self, metric_path: str, metrics: Dict) -> Optional[float]:
        """
        Get a metric value from nested dict
        Example: "beacon.win_rate_7day" or "weekday_wr"
        """
        
        # Handle dot notation (e.g., "beacon.win_rate_7day")
        if '.' in metric_path:
            parts = metric_path.split('.')
            value = metrics
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return self._to_number(value)
        
        # Handle direct access
        else:
            # Try different metric locations
            locations_to_check = [
                metrics,
                metrics.get('beacon', {}),
                metrics.get('pipeline', {}),
                metrics.get('data', {})
            ]
            
            for location in locations_to_check:
                if metric_path in location:
                    return self._to_number(location[metric_path])
        
        return None
    
    def _to_number(self, value: Any) -> Optional[float]:
        """Convert value to number"""
        if value is None:
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def update_historical_metrics(self, current_metrics: Dict):
        """
        Update historical metrics for trend analysis
        """
        timestamp = datetime.now().isoformat()
        
        # Extract key metrics
        beacon = current_metrics.get('beacon', {})
        
        if 'win_rates' not in self.historical_metrics:
            self.historical_metrics['win_rates'] = []
        if 'profits' not in self.historical_metrics:
            self.historical_metrics['profits'] = []
        if 'timestamps' not in self.historical_metrics:
            self.historical_metrics['timestamps'] = []
        
        # Append current values
        if 'win_rate_7day' in beacon:
            self.historical_metrics['win_rates'].append(beacon['win_rate_7day'])
        
        if 'total_profit' in beacon:
            self.historical_metrics['profits'].append(beacon['total_profit'])
        
        self.historical_metrics['timestamps'].append(timestamp)
        
        # Keep only last 30 days
        max_length = 30 * 12  # 12 samples per day (every 2 hours)
        for key in ['win_rates', 'profits', 'timestamps']:
            if len(self.historical_metrics[key]) > max_length:
                self.historical_metrics[key] = self.historical_metrics[key][-max_length:]


# ═══════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test the rules engine
    
    engine = RulesEngine()
    
    # Mock metrics
    test_metrics = {
        'beacon': {
            'win_rate_7day': 35,
            'total_profit': 250
        },
        'pipeline': {
            'weekday_wr': 45,
            'weekend_wr': 15
        }
    }
    
    # Test conditions
    test_conditions = [
        "weekday_wr < 40",
        "weekend_wr < 20",
        "total_profit > 200",
        "is_weekend",
        "always"
    ]
    
    print("Testing Rules Engine:")
    print("="*50)
    
    for condition in test_conditions:
        result = engine.evaluate_condition(condition, test_metrics)
        print(f"{condition:30} → {result}")
