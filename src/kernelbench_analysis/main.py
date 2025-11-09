from pathlib import Path
from typing import Optional
import time
from operation_loader import OperationLoader
from input_extractor import InputExtractorAgent
from categorizer import OperationCategorizer
from fp32_analyzer import FP32Analyzer
from reporter import AnalysisReporter
import gc
import psutil
import os


class KernelBenchAnalysisPipeline:
    """Main pipeline for analyzing KernelBench Level 1 operations"""
    
    def __init__(
        self,
        dataset_path: Path,
        operation_types_path: Path,
        output_dir: Path,
        timeout_seconds: int = 30,
        skip_actual_inputs: bool = False
    ):
        self.loader = OperationLoader(dataset_path, operation_types_path)
        self.extractor = InputExtractorAgent(timeout_seconds=timeout_seconds)
        self.categorizer = OperationCategorizer(operation_types_path)
        self.fp32_analyzer = FP32Analyzer()
        self.reporter = AnalysisReporter(output_dir)
        self.skip_actual_inputs = skip_actual_inputs
        
        # Track memory usage
        self.process = psutil.Process(os.getpid())
        self.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def run_full_analysis(self, category_filter: Optional[str] = None, create_visualizations: bool = True):
        """Run complete analysis pipeline"""
        print("Loading operations...")
        operations = self.loader.get_all_operations()
        
        if category_filter:
            operations = [op for op in operations if op['category'] == category_filter]
            print(f"Filtered to {len(operations)} operations in category '{category_filter}'")
        
        analyses = []
        failed_operations = []
        timeout_operations = []
        start_time = time.time()
        
        print(f"Analyzing {len(operations)} operations...")
        print(f"Timeout per operation: {self.extractor.timeout_seconds}s")
        print(f"Skip actual inputs: {self.skip_actual_inputs}")
        print(f"Initial memory: {self.initial_memory_mb:.1f} MB")
        print()
        
        for i, operation in enumerate(operations, 1):
            op_start = time.time()
            filename_display = operation['filename'][:55] + '...' if len(operation['filename']) > 58 else operation['filename']
            print(f"[{i:3d}/{len(operations)}] {filename_display:<58}", end='', flush=True)
            
            try:
                # Extract input data
                if self.skip_actual_inputs:
                    # Only extract from source code
                    with open(operation['path'], 'r', encoding='utf-8') as f:
                        content = f.read()
                    input_data = {
                        'filepath': str(operation['path']),
                        'tensor_shapes': self.extractor._extract_tensor_shapes(content),
                        'actual_inputs': None,
                        'dtypes': self.extractor._extract_dtypes(content),
                        'memory_patterns': self.extractor._analyze_memory_patterns(content),
                        'operation_calls': self.extractor._extract_operation_calls(content),
                        'parameters': self.extractor._extract_parameters(content),
                    }
                else:
                    input_data = self.extractor.extract_from_file(operation['path'])
                
                # Check if we got actual inputs
                has_actual = input_data.get('actual_inputs') is not None
                
                # Categorize
                categorization = self.categorizer.categorize_operation(operation['filename'])
                
                # Analyze FP32 characteristics
                fp32_analysis = self.fp32_analyzer.analyze_tensor_characteristics(input_data)
                
                analyses.append({
                    'filename': operation['filename'],
                    'category': categorization['category'],
                    'subcategory': categorization['subcategory'],
                    'description': categorization['description'],
                    'input_data': input_data,
                    'fp32_analysis': fp32_analysis
                })
                
                elapsed = time.time() - op_start
                status = "✓" if has_actual else "○"
                
                # Show memory usage periodically
                if i % 10 == 0:
                    current_mem = self._get_memory_usage_mb()
                    mem_delta = current_mem - self.initial_memory_mb
                    print(f" {status} ({elapsed:.1f}s) [Mem: {current_mem:.0f}MB +{mem_delta:.0f}MB]")
                else:
                    print(f" {status} ({elapsed:.1f}s)")
                
                # Force garbage collection every 20 operations
                if i % 20 == 0:
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                
            except Exception as e:
                elapsed = time.time() - op_start
                error_msg = str(e)
                
                if 'timeout' in error_msg.lower() or elapsed >= self.extractor.timeout_seconds:
                    print(f" ⏱ ({elapsed:.1f}s)")
                    timeout_operations.append(operation['filename'])
                else:
                    print(f" ✗ ({elapsed:.1f}s)")
                    
                failed_operations.append({
                    'filename': operation['filename'],
                    'error': error_msg[:100]
                })
                
                # Force cleanup after errors
                gc.collect()
                continue
            
            # Clear input_data to free memory
            del input_data
        
        # Final cleanup
        self.extractor.cleanup()
        gc.collect()
        
        total_time = time.time() - start_time
        final_memory = self._get_memory_usage_mb()
        memory_increase = final_memory - self.initial_memory_mb
        
        print(f"\n{'='*80}")
        print(f"Analysis completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Memory: {final_memory:.1f} MB (increased by {memory_increase:.1f} MB)")
        print(f"Successful: {len(analyses)}/{len(operations)}")
        
        if timeout_operations:
            print(f"Timeouts: {len(timeout_operations)}")
        
        if failed_operations:
            print(f"Failed: {len(failed_operations)}")
            if len(failed_operations) <= 5:
                print("\nFailed operations:")
                for failed in failed_operations:
                    print(f"  - {failed['filename']}")
                    print(f"    Error: {failed['error']}")
        
        if analyses:
            print("\nGenerating report and visualizations...")
            report = self.reporter.generate_full_report(analyses, create_visualizations=create_visualizations)
            
            print(f"\n{'='*80}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*80}")
            print(f"Results saved to: {self.reporter.output_dir}")
            if create_visualizations:
                print(f"Visualizations saved to: {self.reporter.output_dir / 'visualizations'}")
            print(f"Total operations analyzed: {len(analyses)}")
            
            # Clear analyses to free memory before returning
            report_copy = report.copy()
            del analyses
            gc.collect()
            
            return report_copy
        else:
            print("\nNo operations were successfully analyzed!")
            return None


def main():
    """Entry point for analysis"""
    # Configure paths
    base_path = Path(r"c:\inżtest")
    dataset_path = base_path / "KernelBench-main" / "KernelBench" / "level1"
    operation_types_path = base_path / "src" / "kernelbench_analysis" / "operation_types.json"
    output_dir = base_path / "analysis_results"
    
    # Create and run pipeline with timeout
    pipeline = KernelBenchAnalysisPipeline(
        dataset_path=dataset_path,
        operation_types_path=operation_types_path,
        output_dir=output_dir,
        timeout_seconds=30,  # 5 second timeout per operation
        skip_actual_inputs=False  # Set to True to skip get_inputs() calls
    )
    
    # Run analysis
    report = pipeline.run_full_analysis(create_visualizations=True)


if __name__ == "__main__":
    main()
