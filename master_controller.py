#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
🧠 MASTER CONTROLLER v1.0 - The Autonomous AI System Manager
═══════════════════════════════════════════════════════════════

This is Cell 10 - The brain that controls everything else.

What it does:
- Monitors all components (Trade Beacon, Pipeline, Data)
- Makes optimization decisions automatically
- Adjusts parameters based on performance
- Handles emergencies without human intervention
- Only alerts you when truly necessary

What it DOESN'T do:
- Execute trades directly (Trade Beacon does that)
- Generate predictions (Pipeline does that)
- Process data (Data cells do that)

It's the CEO that manages the company!

═══════════════════════════════════════════════════════════════
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

# ═══════════════════════════════════════════════════════════════
# 🎛️ MASTER CONTROLLER TOGGLE
# ═══════════════════════════════════════════════════════════════
MASTER_ENABLED = True  # ← Set to False to disable Master Controller

if not MASTER_ENABLED:
    print("⏸️  MASTER CONTROLLER DISABLED")
    print("   System running in normal mode without Master oversight")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════
# 📁 PATH SETUP
# ═══════════════════════════════════════════════════════════════

# Get base path (repository root)
BASE_PATH = Path(os.getenv('GITHUB_WORKSPACE', os.getcwd()))

# Define all paths
PATHS = {
    'base': BASE_PATH,
    'reports': BASE_PATH / 'reports',
    'config': BASE_PATH / 'config',
    'commands': BASE_PATH / 'commands',
    'memory': BASE_PATH / 'memory',
    'logs': BASE_PATH / 'logs'
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 📝 LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(PATHS['logs'] / 'master_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MasterController')

# ═══════════════════════════════════════════════════════════════
# 🧠 MASTER CONTROLLER CLASS
# ═══════════════════════════════════════════════════════════════

class MasterController:
    """
    The Autonomous AI System Manager
    
    Responsibilities:
    1. Monitor all system components
    2. Analyze performance metrics
    3. Make optimization decisions
    4. Execute control actions
    5. Learn from outcomes
    6. Alert user when needed
    """
    
    def __init__(self):
        logger.info("="*70)
        logger.info("🧠 MASTER CONTROLLER v1.0 INITIALIZING")
        logger.info("="*70)
        
        self.paths = PATHS
        self.cycle_start = datetime.now()
        
        # Load configuration
        self.rules = self.load_rules()
        self.config = self.load_master_config()
        self.thresholds = self.load_thresholds()
        self.notification_settings = self.load_notification_settings()
        
        # Load memory (what Master has learned)
        self.memory = self.load_memory()
        
        # Current state
        self.current_health = {}
        self.triggered_rules = []
        self.actions_taken = []
        
        logger.info(f"✅ Loaded {len(self.rules)} rules")
        logger.info(f"✅ Memory contains {len(self.memory.get('changes', []))} historical changes")
        logger.info("✅ Master Controller initialized successfully")
        logger.info("")
    
    # ═══════════════════════════════════════════════════════════
    # 📥 LOADING METHODS
    # ═══════════════════════════════════════════════════════════
    
    def load_rules(self):
        """Load rules from rules.json"""
        rules_file = self.paths['config'] / 'rules.json'
        
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning("⚠️  rules.json not found, using empty ruleset")
            return {}
    
    def load_master_config(self):
        """Load Master Controller configuration"""
        config_file = self.paths['config'] / 'master_config.json'
        
        default_config = {
            'version': '1.0',
            'enabled': True,
            'check_interval_hours': 2,
            'learning_enabled': True,
            'auto_optimization': True,
            'emergency_controls': True
        }
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def load_thresholds(self):
        """Load alert thresholds"""
        threshold_file = self.paths['config'] / 'system_thresholds.json'
        
        default_thresholds = {
            'max_daily_loss': -100,
            'max_drawdown': -500,
            'max_consecutive_losses': 10,
            'min_win_rate_7day': 30,
            'min_data_quality': 80,
            'max_execution_time_seconds': 300
        }
        
        if threshold_file.exists():
            with open(threshold_file, 'r') as f:
                return json.load(f)
        else:
            with open(threshold_file, 'w') as f:
                json.dump(default_thresholds, f, indent=2)
            return default_thresholds
    
    def load_notification_settings(self):
        """Load notification preferences"""
        settings_file = self.paths['config'] / 'notification_settings.json'
        
        default_settings = {
            'email_enabled': True,
            'emergency_alerts': True,
            'daily_summary': False,
            'weekly_summary': True,
            'monthly_summary': True,
            'milestone_alerts': True,
            'intervention_alerts': True
        }
        
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                return json.load(f)
        else:
            with open(settings_file, 'w') as f:
                json.dump(default_settings, f, indent=2)
            return default_settings
    
    def load_memory(self):
        """Load Master's memory (learning history)"""
        memory_file = self.paths['memory'] / 'optimization_history.json'
        
        default_memory = {
            'changes': [],
            'discoveries': [],
            'rule_effectiveness': {},
            'best_parameters': {}
        }
        
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                return json.load(f)
        else:
            return default_memory
    
    # ═══════════════════════════════════════════════════════════
    # 📊 MONITORING METHODS
    # ═══════════════════════════════════════════════════════════
    
    def read_report(self, filename):
        """Read a report from a component"""
        report_path = self.paths['reports'] / filename
        
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Error reading {filename}: {e}")
                return None
        else:
            logger.warning(f"⚠️  Report not found: {filename}")
            return None
    
    def collect_all_reports(self):
        """Collect reports from all components"""
        logger.info("📊 Collecting reports from all components...")
        
        reports = {
            'beacon': self.read_report('beacon_report.json'),
            'pipeline': self.read_report('pipeline_report.json'),
            'data_quality': self.read_report('data_quality_report.json')
        }
        
        # Check if this is first run (no reports yet)
        if all(report is None for report in reports.values()):
            logger.info("   ℹ️  First run detected - no reports available yet")
            logger.info("   ℹ️  Will use default parameters for this cycle")
            self.is_first_run = True
            return reports
        
        self.is_first_run = False
        
        # Log what we got
        for component, report in reports.items():
            if report:
                logger.info(f"   ✅ {component.capitalize()}: Report received")
            else:
                logger.warning(f"   ⚠️  {component.capitalize()}: No report available")
        
        return reports
    
    def analyze_system_health(self, reports):
        """Analyze overall system health"""
        logger.info("")
        logger.info("🏥 Analyzing system health...")
        
        health = {
            'status': 'HEALTHY',
            'timestamp': datetime.now().isoformat(),
            'issues': [],
            'metrics': {}
        }
        
        # Check Trade Beacon
        if reports['beacon']:
            beacon = reports['beacon']
            health['metrics']['beacon'] = {
                'win_rate_7day': beacon.get('win_rate_7day', 0),
                'total_profit': beacon.get('total_profit', 0),
                'trades_today': beacon.get('trades_executed', 0),
                'filtered_trades': beacon.get('filtered_trades', 0)
            }
            
            # Check for issues
            if beacon.get('win_rate_7day', 100) < self.thresholds['min_win_rate_7day']:
                health['issues'].append('Low win rate detected')
                health['status'] = 'WARNING'
        
        # Check Pipeline
        if reports['pipeline']:
            pipeline = reports['pipeline']
            health['metrics']['pipeline'] = {
                'weekday_wr': pipeline.get('weekday_wr', 0),
                'weekend_wr': pipeline.get('weekend_wr', 0),
                'predictions_evaluated': pipeline.get('predictions_evaluated', 0)
            }
        
        # Check Data Quality
        if reports['data_quality']:
            data = reports['data_quality']
            health['metrics']['data'] = {
                'quality_score': data.get('average_quality', 0),
                'files_processed': data.get('files_processed', 0)
            }
            
            if data.get('average_quality', 100) < self.thresholds['min_data_quality']:
                health['issues'].append('Data quality below threshold')
                health['status'] = 'WARNING'
        
        # Overall status
        if len(health['issues']) == 0:
            logger.info("   ✅ System Status: HEALTHY")
        else:
            logger.warning(f"   ⚠️  System Status: {health['status']}")
            for issue in health['issues']:
                logger.warning(f"      - {issue}")
        
        self.current_health = health
        return health
    
    # ═══════════════════════════════════════════════════════════
    # 🎯 RULES ENGINE
    # ═══════════════════════════════════════════════════════════
    
    def check_all_rules(self, health):
        """Check all rules against current health"""
        logger.info("")
        logger.info("📋 Checking rules...")
        
        # Skip rule checking on first run (no data yet)
        if getattr(self, 'is_first_run', False):
            logger.info("   ℹ️  First run - skipping rule evaluation")
            logger.info("   ℹ️  Creating default config for components")
            self._create_default_config()
            return []
        
        triggered = []
        
        # Iterate through all rule categories
        for category_name, category_data in self.rules.items():
            if category_name == 'metadata':
                continue
            
            if 'rules' not in category_data:
                continue
            
            for rule_id, rule in category_data['rules'].items():
                # Skip disabled rules
                if not rule.get('enabled', True):
                    continue
                
                # Evaluate rule condition
                if self.evaluate_rule_condition(rule, health):
                    triggered.append({
                        'rule_id': rule_id,
                        'rule': rule,
                        'category': category_name
                    })
                    logger.info(f"   ⚡ Rule {rule_id} triggered: {rule['name']}")
        
        if len(triggered) == 0:
            logger.info("   ✅ No rules triggered - system operating normally")
        else:
            logger.info(f"   ⚡ {len(triggered)} rule(s) triggered")
        
        self.triggered_rules = triggered
        return triggered
    
    def _create_default_config(self):
        """Create default config for first run"""
        default_config = {
            'sl_multiplier': 1.5,
            'tp_multiplier': 2.0,
            'confidence_threshold': 0.65,
            'contrarian_mode': False,
            'trading_enabled': True,
            'risk_per_trade': 1.0,
            'created': 'first_run',
            'last_updated': datetime.now().isoformat(),
            'updated_by': 'MasterController'
        }
        
        config_file = self.paths['config'] / 'beacon_params.json'
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info("   ✅ Created default config for Trade Beacon")
    
    def evaluate_rule_condition(self, rule, health):
        """
        Evaluate if a rule's condition is met
        This is simplified - you'd expand this based on actual conditions
        """
        condition = rule.get('condition', '')
        metrics = health.get('metrics', {})
        
        # Example condition evaluations
        # You'll expand this based on your specific needs
        
        if condition == 'weekday_wr < 40 for 3 consecutive days':
            # Would need historical data to check this
            return False
        
        elif condition == 'weekend_wr < 20 for 2 consecutive weekends':
            pipeline = metrics.get('pipeline', {})
            return pipeline.get('weekend_wr', 100) < 20
        
        elif condition == 'data_quality < 80':
            data = metrics.get('data', {})
            return data.get('quality_score', 100) < 80
        
        # Add more condition evaluations as needed
        
        return False
    
    # ═══════════════════════════════════════════════════════════
    # 🎬 ACTION EXECUTION
    # ═══════════════════════════════════════════════════════════
    
    def decide_actions(self, triggered_rules):
        """Decide what actions to take based on triggered rules"""
        logger.info("")
        logger.info("🤔 Deciding actions...")
        
        actions = []
        
        for trigger in triggered_rules:
            rule = trigger['rule']
            action = {
                'rule_id': trigger['rule_id'],
                'type': rule['action'],
                'params': rule.get('params', {}),
                'priority': rule.get('priority', 'medium'),
                'timestamp': datetime.now().isoformat()
            }
            actions.append(action)
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        actions.sort(key=lambda x: priority_order.get(x['priority'], 999))
        
        logger.info(f"   📋 {len(actions)} action(s) planned")
        
        return actions
    
    def execute_actions(self, actions):
        """Execute the decided actions"""
        logger.info("")
        logger.info("⚙️  Executing actions...")
        
        for action in actions:
            try:
                self.execute_single_action(action)
                self.actions_taken.append(action)
            except Exception as e:
                logger.error(f"   ❌ Error executing action {action['type']}: {e}")
    
    def execute_single_action(self, action):
        """Execute a single action"""
        action_type = action['type']
        params = action['params']
        
        logger.info(f"   🔧 Executing: {action_type}")
        
        # Map actions to methods
        action_map = {
            'pause_all_trading': self.action_pause_trading,
            'force_contrarian_mode': self.action_force_contrarian,
            'reduce_tp_multiplier': self.action_adjust_tp,
            'increase_sl_multiplier': self.action_adjust_sl,
            'increase_confidence_threshold': self.action_adjust_confidence,
            # Add more action mappings as needed
        }
        
        if action_type in action_map:
            action_map[action_type](params)
        else:
            logger.warning(f"      ⚠️  Action type not implemented: {action_type}")
    
    # ═══════════════════════════════════════════════════════════
    # 🛠️ ACTION IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════
    
    def action_pause_trading(self, params):
        """Pause all trading"""
        command_file = self.paths['commands'] / 'pause_trading.flag'
        command_file.touch()
        logger.info("      ⏸️  Created pause_trading.flag")
        
        # Also update config
        self.update_beacon_config({'trading_enabled': False})
    
    def action_force_contrarian(self, params):
        """Force contrarian mode"""
        self.update_beacon_config({
            'contrarian_mode': True,
            'contrarian_percentage': 100
        })
        logger.info("      ✅ Enabled 100% contrarian mode")
    
    def action_adjust_tp(self, params):
        """Adjust TP multiplier"""
        current_config = self.read_beacon_config()
        current_tp = current_config.get('tp_multiplier', 2.0)
        delta = params.get('delta', -0.1)
        new_tp = max(1.0, min(3.0, current_tp + delta))
        
        self.update_beacon_config({'tp_multiplier': new_tp})
        logger.info(f"      ✅ Adjusted TP: {current_tp}x → {new_tp}x")
    
    def action_adjust_sl(self, params):
        """Adjust SL multiplier"""
        current_config = self.read_beacon_config()
        current_sl = current_config.get('sl_multiplier', 1.5)
        delta = params.get('delta', 0.1)
        new_sl = max(1.0, min(3.0, current_sl + delta))
        
        self.update_beacon_config({'sl_multiplier': new_sl})
        logger.info(f"      ✅ Adjusted SL: {current_sl}x → {new_sl}x")
    
    def action_adjust_confidence(self, params):
        """Adjust confidence threshold"""
        current_config = self.read_beacon_config()
        current_conf = current_config.get('confidence_threshold', 0.65)
        delta = params.get('delta', 0.05)
        new_conf = max(0.5, min(0.8, current_conf + delta))
        
        self.update_beacon_config({'confidence_threshold': new_conf})
        logger.info(f"      ✅ Adjusted confidence: {current_conf} → {new_conf}")
    
    # ═══════════════════════════════════════════════════════════
    # 📝 CONFIG MANAGEMENT
    # ═══════════════════════════════════════════════════════════
    
    def read_beacon_config(self):
        """Read Trade Beacon configuration"""
        config_file = self.paths['config'] / 'beacon_params.json'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def update_beacon_config(self, updates):
        """Update Trade Beacon configuration"""
        config_file = self.paths['config'] / 'beacon_params.json'
        
        # Read current config
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Apply updates
        config.update(updates)
        config['last_updated'] = datetime.now().isoformat()
        config['updated_by'] = 'MasterController'
        
        # Write back
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════
    # 🧠 LEARNING & MEMORY
    # ═══════════════════════════════════════════════════════════
    
    def learn_from_cycle(self):
        """Learn from this cycle's outcomes"""
        logger.info("")
        logger.info("🎓 Learning from cycle...")
        
        # Record actions taken
        for action in self.actions_taken:
            self.memory['changes'].append({
                'timestamp': action['timestamp'],
                'action': action['type'],
                'params': action['params'],
                'health_before': self.current_health.copy(),
                'rule_triggered': action['rule_id']
            })
        
        # Save memory
        self.save_memory()
        
        logger.info(f"   ✅ Recorded {len(self.actions_taken)} action(s) to memory")
    
    def save_memory(self):
        """Save Master's memory"""
        memory_file = self.paths['memory'] / 'optimization_history.json'
        
        with open(memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════
    # 📱 NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════
    
    def send_notifications(self):
        """Send notifications if needed"""
        logger.info("")
        logger.info("📬 Checking if notifications needed...")
        
        # Critical issues always notify
        if self.current_health['status'] in ['CRITICAL', 'ERROR']:
            self.send_notification(
                f"🚨 System Health: {self.current_health['status']}",
                urgency='critical'
            )
        
        # Weekly summary on Sundays
        if datetime.now().weekday() == 6 and self.notification_settings.get('weekly_summary'):
            self.send_weekly_summary()
        
        logger.info("   ✅ Notification check complete")
    
    def send_notification(self, message, urgency='normal'):
        """Send a notification to user"""
        # In real implementation, would send email/SMS
        logger.info(f"   📧 NOTIFICATION ({urgency}): {message}")
        
        # Save to notifications log
        notif_file = self.paths['logs'] / 'notifications.log'
        with open(notif_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} | {urgency.upper()} | {message}\n")
    
    def send_weekly_summary(self):
        """Generate and send weekly summary"""
        summary = f"""
        📊 WEEKLY SUMMARY
        
        System Health: {self.current_health['status']}
        Actions Taken This Week: {len(self.actions_taken)}
        Rules Triggered: {len(self.triggered_rules)}
        
        Performance Metrics:
        {json.dumps(self.current_health.get('metrics', {}), indent=2)}
        """
        
        self.send_notification(summary, urgency='normal')
    
    # ═══════════════════════════════════════════════════════════
    # 🔄 MAIN CYCLE
    # ═══════════════════════════════════════════════════════════
    
    def run_cycle(self):
        """Run one complete Master Controller cycle"""
        
        logger.info("")
        logger.info("="*70)
        logger.info("🧠 MASTER CONTROLLER CYCLE START")
        logger.info(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("="*70)
        
        try:
            # 1. Collect reports from all components
            reports = self.collect_all_reports()
            
            # 2. Analyze system health
            health = self.analyze_system_health(reports)
            
            # 3. Check all rules
            triggered = self.check_all_rules(health)
            
            # 4. Decide actions
            actions = self.decide_actions(triggered)
            
            # 5. Execute actions
            if actions:
                self.execute_actions(actions)
            else:
                logger.info("")
                logger.info("✅ No actions needed - system operating optimally")
            
            # 6. Learn from this cycle
            self.learn_from_cycle()
            
            # 7. Send notifications if needed
            self.send_notifications()
            
            # 8. Summary
            duration = (datetime.now() - self.cycle_start).total_seconds()
            logger.info("")
            logger.info("="*70)
            logger.info("✅ MASTER CONTROLLER CYCLE COMPLETE")
            logger.info(f"⏱️  Duration: {duration:.1f}s")
            logger.info(f"📊 Health: {health['status']}")
            logger.info(f"⚡ Rules Triggered: {len(triggered)}")
            logger.info(f"🎯 Actions Taken: {len(self.actions_taken)}")
            logger.info("="*70)
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Error in Master Controller cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════
# 🚀 MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        # Create and run Master Controller
        master = MasterController()
        master.run_cycle()
        
        logger.info("🎯 Master Controller execution complete")
        logger.info("📅 Next cycle: In 2 hours (GitHub Actions schedule)")
        
    except Exception as e:
        logger.error(f"💥 Fatal error in Master Controller: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
