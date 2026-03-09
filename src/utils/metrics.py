from typing import Dict, List
from datetime import datetime
from collections import defaultdict


class MetricsCollector:
    """Collect and aggregate system metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = datetime.now()
        
    def record(self, metric_name: str, value: float):
        """Record a metric value"""
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for a metric"""
        values = [m['value'] for m in self.metrics.get(metric_name, [])]
        return sum(values) / len(values) if values else 0.0
    
    def get_total(self, metric_name: str) -> float:
        """Get total sum for a metric"""
        values = [m['value'] for m in self.metrics.get(metric_name, [])]
        return sum(values)
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        return {
            metric_name: {
                'count': len(values),
                'average': self.get_average(metric_name),
                'total': self.get_total(metric_name)
            }
            for metric_name, values in self.metrics.items()
        }
    
    def export(self) -> Dict:
        """Export all metrics"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'metrics': dict(self.metrics),
            'summary': self.get_summary()
        }

