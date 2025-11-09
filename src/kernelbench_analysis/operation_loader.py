import json
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util
import sys


class OperationLoader:
    """Load and manage Level 1 operations from KernelBench dataset"""
    
    def __init__(self, dataset_path: Path, operation_types_path: Path):
        self.dataset_path = Path(dataset_path)
        self.operation_types_path = Path(operation_types_path)
        self.operation_types = self._load_operation_types()
        
    def _load_operation_types(self) -> Dict:
        """Load operation types from JSON configuration"""
        with open(self.operation_types_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_operations(self) -> List[Dict]:
        """Get all operations with their metadata"""
        operations = []
        
        for category, data in self.operation_types['kernel_operations'].items():
            if 'files' in data:
                files = data['files']
            elif 'subcategories' in data:
                files = []
                for subcat_files in data['subcategories'].values():
                    files.extend(subcat_files)
            else:
                continue
                
            for filename in files:
                op_path = self.dataset_path / filename
                if op_path.exists():
                    operations.append({
                        'filename': filename,
                        'path': op_path,
                        'category': category,
                        'description': data['description']
                    })
                    
        return operations
    
    def load_operation_module(self, filepath: Path):
        """Dynamically load a Python operation file"""
        spec = importlib.util.spec_from_file_location("operation_module", filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules["operation_module"] = module
        spec.loader.exec_module(module)
        return module
    
    def get_operations_by_category(self, category: str) -> List[Dict]:
        """Get operations filtered by category"""
        all_ops = self.get_all_operations()
        return [op for op in all_ops if op['category'] == category]
