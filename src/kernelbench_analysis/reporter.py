from typing import Dict, List, Any
from pathlib import Path
import json
from datetime import datetime


class AnalysisReporter:
    """Generate reports and visualizations from analysis"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_full_report(self, analyses: List[Dict], create_visualizations: bool = True) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_operations': len(analyses),
                'version': '1.0'
            },
            'category_summary': self._summarize_by_category(analyses),
            'fp32_statistics': self._aggregate_fp32_stats(analyses),
            'memory_analysis': self._aggregate_memory_analysis(analyses),
            'operation_details': analyses
        }
        
        # Save report
        report_path = self.output_dir / f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate summary text
        self._generate_summary_text(report)
        
        # Generate visualizations
        if create_visualizations:
            from visualizer import AnalysisVisualizer
            visualizer = AnalysisVisualizer(self.output_dir / 'visualizations')
            visualizer.generate_all_visualizations(report)
        
        return report
    
    def _summarize_by_category(self, analyses: List[Dict]) -> Dict:
        """Summarize operations by category"""
        categories = {}
        
        for analysis in analyses:
            category = analysis.get('category', 'unknown')
            if category not in categories:
                categories[category] = {
                    'count': 0,
                    'total_memory_mb': 0,
                    'operations': []
                }
            
            categories[category]['count'] += 1
            categories[category]['operations'].append(analysis.get('filename'))
            
            if 'fp32_analysis' in analysis:
                mem_usage = analysis['fp32_analysis'].get('memory_usage', {})
                categories[category]['total_memory_mb'] += mem_usage.get('total_mb', 0)
        
        return categories
    
    def _aggregate_fp32_stats(self, analyses: List[Dict]) -> Dict:
        """Aggregate FP32 statistics across all operations"""
        total_memory = 0
        total_tensors = 0
        dimension_distribution = {}
        
        for analysis in analyses:
            if 'fp32_analysis' in analysis:
                fp32 = analysis['fp32_analysis']
                
                if 'memory_usage' in fp32:
                    total_memory += fp32['memory_usage'].get('total_mb', 0)
                    total_tensors += fp32['memory_usage'].get('tensor_count', 0)
                
                for tensor_info in fp32.get('tensor_info', []):
                    dims = tensor_info.get('dimensions', 0)
                    dimension_distribution[dims] = dimension_distribution.get(dims, 0) + 1
        
        return {
            'total_memory_mb': total_memory,
            'total_memory_gb': total_memory / 1024,
            'total_tensors': total_tensors,
            'avg_memory_per_operation_mb': total_memory / len(analyses) if analyses else 0,
            'dimension_distribution': dimension_distribution
        }
    
    def _aggregate_memory_analysis(self, analyses: List[Dict]) -> Dict:
        """Aggregate memory access pattern analysis"""
        pattern_counts = {
            'transpose': 0,
            'reshape': 0,
            'permute': 0,
            'contiguous': 0,
            'indexing': 0,
            'batching': 0
        }
        
        for analysis in analyses:
            if 'input_data' in analysis:
                patterns = analysis['input_data'].get('memory_patterns', {})
                if patterns.get('has_transpose'): pattern_counts['transpose'] += 1
                if patterns.get('has_reshape'): pattern_counts['reshape'] += 1
                if patterns.get('has_permute'): pattern_counts['permute'] += 1
                if patterns.get('has_contiguous'): pattern_counts['contiguous'] += 1
                if patterns.get('has_indexing'): pattern_counts['indexing'] += 1
                if patterns.get('has_batching'): pattern_counts['batching'] += 1
        
        return pattern_counts
    
    def _generate_summary_text(self, report: Dict):
        """Generate human-readable summary text"""
        summary_path = self.output_dir / 'analysis_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("KERNELBENCH LEVEL 1 OPERATIONS ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {report['metadata']['timestamp']}\n")
            f.write(f"Total Operations Analyzed: {report['metadata']['total_operations']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("CATEGORY SUMMARY\n")
            f.write("-" * 80 + "\n")
            for category, data in report['category_summary'].items():
                f.write(f"\n{category}:\n")
                f.write(f"  Operations: {data['count']}\n")
                f.write(f"  Total Memory: {data['total_memory_mb']:.2f} MB\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("FP32 STATISTICS\n")
            f.write("-" * 80 + "\n")
            fp32_stats = report['fp32_statistics']
            f.write(f"Total Memory Usage: {fp32_stats['total_memory_gb']:.2f} GB\n")
            f.write(f"Total Tensors: {fp32_stats['total_tensors']}\n")
            f.write(f"Avg Memory per Operation: {fp32_stats['avg_memory_per_operation_mb']:.2f} MB\n")
            f.write(f"\nDimension Distribution:\n")
            for dims, count in sorted(fp32_stats['dimension_distribution'].items()):
                f.write(f"  {dims}D tensors: {count}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("MEMORY ACCESS PATTERNS\n")
            f.write("-" * 80 + "\n")
            for pattern, count in report['memory_analysis'].items():
                f.write(f"{pattern.capitalize()}: {count} operations\n")
