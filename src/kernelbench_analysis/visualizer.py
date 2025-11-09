import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from collections import Counter


class AnalysisVisualizer:
    """Generate visualizations for KernelBench analysis results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def generate_all_visualizations(self, report: Dict):
        """Generate all visualization charts"""
        print("Generating visualizations...")
        
        self.plot_category_distribution(report)
        self.plot_memory_usage_by_category(report)
        self.plot_tensor_dimension_distribution(report)
        self.plot_memory_access_patterns(report)
        self.plot_operation_complexity(report)
        self.plot_memory_usage_histogram(report)
        self.plot_top_memory_operations(report)
        
        print(f"Visualizations saved to: {self.output_dir}")
    
    def plot_category_distribution(self, report: Dict):
        """Plot distribution of operations by category"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        categories = report['category_summary']
        names = list(categories.keys())
        counts = [cat['count'] for cat in categories.values()]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        bars = ax.barh(names, counts, color=colors)
        
        ax.set_xlabel('Number of Operations', fontsize=12, fontweight='bold')
        ax.set_ylabel('Category', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Operations by Category', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                   f'{count}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_memory_usage_by_category(self, report: Dict):
        """Plot total memory usage by category"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        categories = report['category_summary']
        names = list(categories.keys())
        memory_mb = [cat['total_memory_mb'] for cat in categories.values()]
        counts = [cat['count'] for cat in categories.values()]
        
        # Filter out categories with zero memory
        valid_data = [(name, mem, count) for name, mem, count in zip(names, memory_mb, counts) if mem > 0]
        
        if valid_data:
            valid_names, valid_memory_mb, valid_counts = zip(*valid_data)
            
            # Pie chart for memory distribution
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(valid_names)))
            wedges, texts, autotexts = ax1.pie(valid_memory_mb, labels=valid_names, autopct='%1.1f%%',
                                                colors=colors, startangle=90)
            ax1.set_title('Memory Usage Distribution by Category (MB)', fontsize=12, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
        else:
            ax1.text(0.5, 0.5, 'No memory data available', 
                    ha='center', va='center', fontsize=12, transform=ax1.transAxes)
            ax1.set_title('Memory Usage Distribution by Category (MB)', fontsize=12, fontweight='bold')
        
        # Bar chart for average memory per operation
        avg_memory = [mem / count if count > 0 else 0 
                     for mem, count in zip(memory_mb, counts)]
        
        if any(avg > 0 for avg in avg_memory):
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(names)))
            bars = ax2.bar(range(len(names)), avg_memory, color=colors)
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.set_ylabel('Average Memory (MB)', fontsize=11, fontweight='bold')
            ax2.set_title('Average Memory per Operation by Category', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            max_avg = max(avg_memory) if avg_memory else 1
            for i, (bar, val) in enumerate(zip(bars, avg_memory)):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_avg*0.01,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No memory data available', 
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.set_title('Average Memory per Operation by Category', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_tensor_dimension_distribution(self, report: Dict):
        """Plot distribution of tensor dimensions"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dim_dist = report['fp32_statistics']['dimension_distribution']
        
        if dim_dist:
            dimensions = sorted(dim_dist.keys())
            counts = [dim_dist[d] for d in dimensions]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(dimensions)))
            bars = ax.bar([f'{d}D' for d in dimensions], counts, color=colors, edgecolor='black', linewidth=1.5)
            
            ax.set_xlabel('Tensor Dimensions', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Tensors', fontsize=12, fontweight='bold')
            ax.set_title('Distribution of Tensor Dimensions', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            max_count = max(counts) if counts else 1
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_count*0.01,
                       f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No tensor dimension data available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Distribution of Tensor Dimensions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tensor_dimension_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_memory_access_patterns(self, report: Dict):
        """Plot memory access pattern analysis"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        patterns = report['memory_analysis']
        
        if patterns:
            pattern_names = list(patterns.keys())
            pattern_counts = list(patterns.values())
            
            # Sort by count
            sorted_pairs = sorted(zip(pattern_names, pattern_counts), key=lambda x: x[1], reverse=True)
            pattern_names, pattern_counts = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            if pattern_names and any(count > 0 for count in pattern_counts):
                colors = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(pattern_names)))
                bars = ax.barh(pattern_names, pattern_counts, color=colors, edgecolor='black', linewidth=1.5)
                
                ax.set_xlabel('Number of Operations', fontsize=12, fontweight='bold')
                ax.set_ylabel('Memory Access Pattern', fontsize=12, fontweight='bold')
                ax.set_title('Memory Access Patterns Across Operations', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                max_count = max(pattern_counts) if pattern_counts else 1
                for bar, count in zip(bars, pattern_counts):
                    if count > 0:
                        ax.text(bar.get_width() + max_count*0.01, bar.get_y() + bar.get_height()/2,
                               f'{count}', va='center', fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No memory access pattern data available', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title('Memory Access Patterns Across Operations', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No memory access pattern data available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Memory Access Patterns Across Operations', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_access_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_operation_complexity(self, report: Dict):
        """Plot computational complexity distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract complexity information
        complexity_counter = Counter()
        
        for op_detail in report['operation_details']:
            if 'fp32_analysis' in op_detail:
                complexity_data = op_detail['fp32_analysis'].get('computational_complexity', {})
                is_compute_intensive = complexity_data.get('is_compute_intensive', False)
                
                if is_compute_intensive:
                    complexity_counter['Compute Intensive'] += 1
                else:
                    complexity_counter['Memory Bound'] += 1
        
        if complexity_counter and sum(complexity_counter.values()) > 0:
            labels = list(complexity_counter.keys())
            sizes = list(complexity_counter.values())
            colors = ['#ff6b6b', '#4ecdc4']
            explode = (0.05, 0.05) if len(labels) == 2 else tuple([0.05] * len(labels))
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                               colors=colors[:len(labels)], explode=explode, startangle=90,
                                               textprops={'fontsize': 12, 'fontweight': 'bold'})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)
            
            ax.set_title('Operation Complexity Distribution', fontsize=14, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, 'No complexity data available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Operation Complexity Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'operation_complexity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_memory_usage_histogram(self, report: Dict):
        """Plot histogram of memory usage per operation"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        memory_values = []
        for op_detail in report['operation_details']:
            if 'fp32_analysis' in op_detail:
                mem_usage = op_detail['fp32_analysis'].get('memory_usage', {})
                total_mb = mem_usage.get('total_mb', 0)
                if total_mb > 0:
                    memory_values.append(total_mb)
        
        if memory_values:
            n, bins, patches = ax.hist(memory_values, bins=min(30, len(memory_values)), color='steelblue', 
                                       edgecolor='black', alpha=0.7)
            
            # Color gradient
            cm = plt.cm.plasma
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            if max(col) > 0:
                col /= max(col)
            
            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))
            
            ax.set_xlabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Operations', fontsize=12, fontweight='bold')
            ax.set_title('Distribution of Memory Usage per Operation', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add statistics text
            mean_mem = np.mean(memory_values)
            median_mem = np.median(memory_values)
            max_mem = np.max(memory_values)
            
            stats_text = f'Mean: {mean_mem:.2f} MB\nMedian: {median_mem:.2f} MB\nMax: {max_mem:.2f} MB'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No memory usage data available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Distribution of Memory Usage per Operation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_top_memory_operations(self, report: Dict):
        """Plot top 15 operations by memory usage"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Collect operation memory data
        op_memory = []
        for op_detail in report['operation_details']:
            if 'fp32_analysis' in op_detail:
                mem_usage = op_detail['fp32_analysis'].get('memory_usage', {})
                total_mb = mem_usage.get('total_mb', 0)
                filename = op_detail.get('filename', 'Unknown')
                category = op_detail.get('category', 'unknown')
                
                if total_mb > 0:
                    op_memory.append({
                        'filename': filename[:50],  # Truncate long names
                        'memory_mb': total_mb,
                        'category': category
                    })
        
        # Sort and get top 15
        op_memory.sort(key=lambda x: x['memory_mb'], reverse=True)
        top_ops = op_memory[:15]
        
        if top_ops:
            filenames = [op['filename'] for op in top_ops]
            memory_values = [op['memory_mb'] for op in top_ops]
            categories = [op['category'] for op in top_ops]
            
            # Create color map based on categories
            unique_categories = list(set(categories))
            color_map = {cat: plt.cm.tab10(i % 10) for i, cat in enumerate(unique_categories)}
            colors = [color_map[cat] for cat in categories]
            
            bars = ax.barh(filenames, memory_values, color=colors, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Operation', fontsize=12, fontweight='bold')
            ax.set_title('Top 15 Operations by Memory Usage', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            max_mem = max(memory_values) if memory_values else 1
            for bar, val in zip(bars, memory_values):
                ax.text(bar.get_width() + max_mem*0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
            
            # Add legend
            legend_patches = [mpatches.Patch(color=color_map[cat], label=cat) 
                            for cat in unique_categories]
            ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No memory data available for operations', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Top 15 Operations by Memory Usage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_memory_operations.png', dpi=300, bbox_inches='tight')
        plt.close()
