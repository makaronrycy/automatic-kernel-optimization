from typing import Dict, List, Any
import numpy as np


class FP32Analyzer:
    """Analyze FP32 tensor characteristics and memory patterns"""
    
    def __init__(self):
        self.fp32_size = 4  # bytes
    
    def analyze_tensor_characteristics(self, input_data: Dict) -> Dict[str, Any]:
        """Analyze tensor characteristics for FP32 data type"""
        analysis = {
            'tensor_info': [],
            'memory_usage': {},
            'access_patterns': {},
            'computational_complexity': {}
        }
        
        # Try to use actual inputs first
        if 'actual_inputs' in input_data and input_data['actual_inputs']:
            analysis['tensor_info'] = self._analyze_actual_inputs(input_data['actual_inputs'])
        else:
            # Fall back to extracted shapes
            for tensor in input_data.get('tensor_shapes', []):
                if tensor['shape']:
                    tensor_analysis = self._analyze_single_tensor(tensor['shape'])
                    analysis['tensor_info'].append({
                        'shape': tensor['shape'],
                        **tensor_analysis
                    })
        
        analysis['memory_usage'] = self._calculate_memory_usage(input_data)
        analysis['access_patterns'] = self._analyze_access_patterns(input_data)
        analysis['computational_complexity'] = self._estimate_complexity(input_data)
        
        return analysis
    
    def _analyze_actual_inputs(self, actual_inputs: Dict) -> List[Dict]:
        """Analyze actual input tensors from get_inputs"""
        tensor_info = []
        
        # Handle different input structures
        if 'inputs' in actual_inputs and isinstance(actual_inputs['inputs'], list):
            # Handle list of inputs (most common case)
            for i, item in enumerate(actual_inputs['inputs']):
                if isinstance(item, dict) and 'shape' in item:
                    analysis = self._analyze_single_tensor(item['shape'])
                    analysis['name'] = f"input_{i}"
                    analysis['dtype'] = item.get('dtype', 'unknown')
                    analysis['device'] = item.get('device', 'unknown')
                    analysis['actual_memory_bytes'] = item.get('numel', 0) * item.get('element_size', 4)
                    analysis['actual_memory_mb'] = analysis['actual_memory_bytes'] / (1024 ** 2)
                    tensor_info.append(analysis)
        elif isinstance(actual_inputs, dict):
            # Handle dict of inputs
            for key, tensor_data in actual_inputs.items():
                if isinstance(tensor_data, dict) and 'shape' in tensor_data:
                    analysis = self._analyze_single_tensor(tensor_data['shape'])
                    analysis['name'] = key
                    analysis['dtype'] = tensor_data.get('dtype', 'unknown')
                    analysis['device'] = tensor_data.get('device', 'unknown')
                    analysis['actual_memory_bytes'] = tensor_data.get('numel', 0) * tensor_data.get('element_size', 4)
                    analysis['actual_memory_mb'] = analysis['actual_memory_bytes'] / (1024 ** 2)
                    tensor_info.append(analysis)
                elif isinstance(tensor_data, list):
                    for i, item in enumerate(tensor_data):
                        if isinstance(item, dict) and 'shape' in item:
                            analysis = self._analyze_single_tensor(item['shape'])
                            analysis['name'] = f"{key}[{i}]"
                            analysis['dtype'] = item.get('dtype', 'unknown')
                            analysis['device'] = item.get('device', 'unknown')
                            analysis['actual_memory_bytes'] = item.get('numel', 0) * item.get('element_size', 4)
                            analysis['actual_memory_mb'] = analysis['actual_memory_bytes'] / (1024 ** 2)
                            tensor_info.append(analysis)
        
        return tensor_info
    
    def _analyze_single_tensor(self, shape) -> Dict:
        """Analyze a single tensor's characteristics"""
        if isinstance(shape, (list, tuple)):
            total_elements = np.prod(shape)
            memory_bytes = total_elements * self.fp32_size
            
            return {
                'dimensions': len(shape),
                'total_elements': int(total_elements),
                'memory_bytes': int(memory_bytes),
                'memory_mb': memory_bytes / (1024 ** 2),
                'is_contiguous': True,  # Assumption for newly created tensors
                'shape_characteristics': self._classify_shape(shape)
            }
        
        return {}
    
    def _classify_shape(self, shape) -> Dict:
        """Classify shape characteristics"""
        shape_list = list(shape)
        
        return {
            'is_square': len(shape_list) >= 2 and shape_list[-2] == shape_list[-1],
            'is_batched': len(shape_list) > 2,
            'batch_size': shape_list[0] if len(shape_list) > 2 else None,
            'aspect_ratio': shape_list[-1] / shape_list[-2] if len(shape_list) >= 2 and shape_list[-2] != 0 else None
        }
    
    def _calculate_memory_usage(self, input_data: Dict) -> Dict:
        """Calculate total memory usage"""
        total_memory = 0
        tensor_count = 0
        
        # Use actual inputs if available
        if 'actual_inputs' in input_data and input_data['actual_inputs']:
            actual_inputs = input_data['actual_inputs']
            
            def count_tensors(data):
                nonlocal total_memory, tensor_count
                if isinstance(data, dict):
                    if 'shape' in data and 'numel' in data:
                        element_size = data.get('element_size', 4)
                        total_memory += data['numel'] * element_size
                        tensor_count += 1
                    else:
                        for value in data.values():
                            count_tensors(value)
                elif isinstance(data, list):
                    for item in data:
                        count_tensors(item)
            
            count_tensors(actual_inputs)
        else:
            # Fall back to extracted shapes
            for tensor in input_data.get('tensor_shapes', []):
                if tensor['shape']:
                    elements = np.prod(tensor['shape'])
                    total_memory += elements * self.fp32_size
                    tensor_count += 1
        
        return {
            'total_bytes': int(total_memory),
            'total_mb': total_memory / (1024 ** 2),
            'total_gb': total_memory / (1024 ** 3),
            'tensor_count': tensor_count,
            'avg_tensor_size_mb': (total_memory / tensor_count / (1024 ** 2)) if tensor_count > 0 else 0
        }
    
    def _analyze_access_patterns(self, input_data: Dict) -> Dict:
        """Analyze memory access patterns"""
        patterns = input_data.get('memory_patterns', {})
        
        complexity_score = 0
        if patterns.get('has_transpose'): complexity_score += 2
        if patterns.get('has_reshape'): complexity_score += 1
        if patterns.get('has_permute'): complexity_score += 3
        if patterns.get('has_indexing'): complexity_score += 2
        
        return {
            **patterns,
            'complexity_score': complexity_score,
            'requires_optimization': complexity_score > 3
        }
    
    def _estimate_complexity(self, input_data: Dict) -> Dict:
        """Estimate computational complexity"""
        operations = input_data.get('operation_calls', [])
        
        complexity_map = {
            'matmul': 'O(n³)',
            'mm': 'O(n³)',
            'bmm': 'O(b*n³)',
            'conv2d': 'O(n²*m²*k²)',
            'conv3d': 'O(n³*m³*k³)',
            'relu': 'O(n)',
            'sigmoid': 'O(n)',
            'softmax': 'O(n)',
        }
        
        estimated_complexity = []
        for op in operations:
            op_lower = op.lower()
            for key, value in complexity_map.items():
                if key in op_lower:
                    estimated_complexity.append({'operation': op, 'complexity': value})
                    break
        
        return {
            'operations': estimated_complexity,
            'is_compute_intensive': any(op in str(operations).lower() for op in ['matmul', 'mm', 'conv'])
        }
