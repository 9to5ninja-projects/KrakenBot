"""
Experiment tracking for AI model development
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentTracker:
    """
    Simple experiment tracking for AI model development.
    """
    
    def __init__(self, experiment_dir: str = "data/ai_experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiment data
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment = None
        
    def start_experiment(self, name: str, description: str = "", tags: Optional[list] = None) -> str:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Optional tags for categorization
            
        Returns:
            Experiment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        self.current_experiment = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "metrics": {},
            "parameters": {},
            "artifacts": [],
            "logs": []
        }
        
        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save initial experiment data
        self._save_experiment()
        
        return experiment_id
    
    def log_parameter(self, key: str, value: Any):
        """Log a parameter for the current experiment."""
        if self.current_experiment:
            self.current_experiment["parameters"][key] = value
            self._save_experiment()
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric for the current experiment."""
        if self.current_experiment:
            if key not in self.current_experiment["metrics"]:
                self.current_experiment["metrics"][key] = []
            
            metric_entry = {
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
            
            if step is not None:
                metric_entry["step"] = step
            
            self.current_experiment["metrics"][key].append(metric_entry)
            self._save_experiment()
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log an artifact (file, model, etc.) for the current experiment."""
        if self.current_experiment:
            artifact = {
                "path": artifact_path,
                "type": artifact_type,
                "timestamp": datetime.now().isoformat()
            }
            
            self.current_experiment["artifacts"].append(artifact)
            self._save_experiment()
    
    def log_message(self, message: str, level: str = "info"):
        """Log a message for the current experiment."""
        if self.current_experiment:
            log_entry = {
                "message": message,
                "level": level,
                "timestamp": datetime.now().isoformat()
            }
            
            self.current_experiment["logs"].append(log_entry)
            self._save_experiment()
    
    def end_experiment(self, status: str = "completed"):
        """End the current experiment."""
        if self.current_experiment:
            self.current_experiment["status"] = status
            self.current_experiment["end_time"] = datetime.now().isoformat()
            
            # Calculate duration
            start_time = datetime.fromisoformat(self.current_experiment["start_time"])
            end_time = datetime.fromisoformat(self.current_experiment["end_time"])
            duration = (end_time - start_time).total_seconds()
            self.current_experiment["duration_seconds"] = duration
            
            self._save_experiment()
            
            # Generate summary
            self._generate_experiment_summary()
            
            experiment_id = self.current_experiment["id"]
            self.current_experiment = None
            
            return experiment_id
    
    def _save_experiment(self):
        """Save current experiment data to file."""
        if self.current_experiment:
            exp_dir = self.experiment_dir / self.current_experiment["id"]
            exp_file = exp_dir / "experiment.json"
            
            with open(exp_file, 'w') as f:
                json.dump(self.current_experiment, f, indent=2)
    
    def _generate_experiment_summary(self):
        """Generate a summary of the experiment."""
        if not self.current_experiment:
            return
        
        exp_dir = self.experiment_dir / self.current_experiment["id"]
        summary_file = exp_dir / "summary.txt"
        
        # Calculate final metrics
        final_metrics = {}
        for metric_name, metric_values in self.current_experiment["metrics"].items():
            if metric_values:
                final_metrics[metric_name] = metric_values[-1]["value"]
        
        # Generate summary text
        summary = f"""
EXPERIMENT SUMMARY
==================

Name: {self.current_experiment['name']}
ID: {self.current_experiment['id']}
Description: {self.current_experiment['description']}
Status: {self.current_experiment['status']}

Duration: {self.current_experiment.get('duration_seconds', 0):.2f} seconds

PARAMETERS:
{self._format_dict(self.current_experiment['parameters'])}

FINAL METRICS:
{self._format_dict(final_metrics)}

ARTIFACTS:
{chr(10).join(f"- {a['type']}: {a['path']}" for a in self.current_experiment['artifacts'])}

LOGS:
{chr(10).join(f"[{l['level'].upper()}] {l['message']}" for l in self.current_experiment['logs'][-10:])}
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary.strip())
    
    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format dictionary for display."""
        if not d:
            return "None"
        
        return chr(10).join(f"  {k}: {v}" for k, v in d.items())
    
    def list_experiments(self) -> list:
        """List all experiments."""
        experiments = []
        
        for exp_dir in self.experiment_dir.iterdir():
            if exp_dir.is_dir():
                exp_file = exp_dir / "experiment.json"
                if exp_file.exists():
                    try:
                        with open(exp_file, 'r') as f:
                            exp_data = json.load(f)
                            experiments.append({
                                "id": exp_data["id"],
                                "name": exp_data["name"],
                                "status": exp_data["status"],
                                "start_time": exp_data["start_time"]
                            })
                    except:
                        continue
        
        return sorted(experiments, key=lambda x: x["start_time"], reverse=True)
    
    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment data by ID."""
        exp_dir = self.experiment_dir / experiment_id
        exp_file = exp_dir / "experiment.json"
        
        if exp_file.exists():
            try:
                with open(exp_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        
        return None
    
    def compare_experiments(self, experiment_ids: list) -> Dict[str, Any]:
        """Compare multiple experiments."""
        experiments = []
        
        for exp_id in experiment_ids:
            exp_data = self.load_experiment(exp_id)
            if exp_data:
                experiments.append(exp_data)
        
        if not experiments:
            return {"error": "No valid experiments found"}
        
        # Extract metrics for comparison
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp["metrics"].keys())
        
        comparison = {
            "experiments": len(experiments),
            "metrics_comparison": {},
            "parameter_comparison": {},
            "summary": []
        }
        
        # Compare metrics
        for metric in all_metrics:
            comparison["metrics_comparison"][metric] = {}
            
            for exp in experiments:
                exp_id = exp["id"]
                metric_values = exp["metrics"].get(metric, [])
                
                if metric_values:
                    final_value = metric_values[-1]["value"]
                    comparison["metrics_comparison"][metric][exp_id] = final_value
        
        # Compare parameters
        all_params = set()
        for exp in experiments:
            all_params.update(exp["parameters"].keys())
        
        for param in all_params:
            comparison["parameter_comparison"][param] = {}
            
            for exp in experiments:
                exp_id = exp["id"]
                param_value = exp["parameters"].get(param, "N/A")
                comparison["parameter_comparison"][param][exp_id] = param_value
        
        # Generate summary
        for exp in experiments:
            exp_summary = {
                "id": exp["id"],
                "name": exp["name"],
                "status": exp["status"],
                "duration": exp.get("duration_seconds", 0)
            }
            comparison["summary"].append(exp_summary)
        
        return comparison