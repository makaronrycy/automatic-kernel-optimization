from typing import Dict, List
from pathlib import Path
import json


class OperationCategorizer:
    """Categorize operations according to operation types"""
    
    def __init__(self, operation_types_path: Path):
        with open(operation_types_path, 'r', encoding='utf-8') as f:
            self.operation_types = json.load(f)
    
    def categorize_operation(self, filename: str) -> Dict[str, str]:
        """Categorize a single operation by filename"""
        for category, data in self.operation_types['kernel_operations'].items():
            if 'files' in data and filename in data['files']:
                return {
                    'category': category,
                    'subcategory': None,
                    'description': data['description']
                }
            elif 'subcategories' in data:
                for subcat, files in data['subcategories'].items():
                    if filename in files:
                        return {
                            'category': category,
                            'subcategory': subcat,
                            'description': data['description']
                        }
        
        return {'category': 'unknown', 'subcategory': None, 'description': 'Unknown operation'}
    
    def get_category_statistics(self) -> Dict:
        """Get statistics for all categories"""
        return self.operation_types['summary']
    
    def get_operations_by_type(self, category: str) -> List[str]:
        """Get all operation files for a specific category"""
        if category not in self.operation_types['kernel_operations']:
            return []
        
        data = self.operation_types['kernel_operations'][category]
        
        if 'files' in data:
            return data['files']
        elif 'subcategories' in data:
            files = []
            for subcat_files in data['subcategories'].values():
                files.extend(subcat_files)
            return files
        
        return []
