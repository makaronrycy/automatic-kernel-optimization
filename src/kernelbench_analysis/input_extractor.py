import ast
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import re
import sys
import importlib.util
import traceback
import threading
import gc
import weakref
import ctypes
import multiprocessing as mp
import signal
class InputExtractorAgent:
    """Automated agent for extracting input characteristics from Level 1 operations"""
    
    def __init__(self, timeout_seconds: int = 30, max_tensor_size_mb: int = 2000):
        self.timeout_seconds = timeout_seconds
        self.max_tensor_size_mb = max_tensor_size_mb
        self.tensor_patterns = {
            'torch.randn': r'torch\.randn\((.*?)\)',
            'torch.rand': r'torch\.rand\((.*?)\)',
            'torch.zeros': r'torch\.zeros\((.*?)\)',
            'torch.ones': r'torch\.ones\((.*?)\)',
            'torch.empty': r'torch\.empty\((.*?)\)',
            'torch.full': r'torch\.full\((.*?)\)',
            'torch.arange': r'torch\.arange\((.*?)\)',
            'torch.linspace': r'torch\.linspace\((.*?)\)',
        }
    
    def extract_from_file(self, filepath: Path) -> Dict[str, Any]:
        """Extract input characteristics from operation file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                'filepath': str(filepath),
                'error': f"Failed to read file: {str(e)}"
            }
        
        # Try to get actual inputs by calling get_input IN A SEPARATE PROCESS
        actual_inputs = self._extract_actual_inputs(filepath)
        
        return {
            'filepath': str(filepath),
            'tensor_shapes': self._extract_tensor_shapes(content),
            'actual_inputs': actual_inputs,
            'dtypes': self._extract_dtypes(content),
            'memory_patterns': self._analyze_memory_patterns(content),
            'operation_calls': self._extract_operation_calls(content),
            'parameters': self._extract_parameters(content),
            'has_get_input': self._has_get_input_function(content)
        }
    
    def _has_get_input_function(self, content: str) -> bool:
        """Check if the file has a get_inputs function"""
        return bool(re.search(r'def\s+get_inputs\s*\(', content))
    
    def _extract_actual_inputs(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Extract actual input data using multiprocessing for complete isolation"""
        
        # Use multiprocessing instead of threading for true memory isolation
        ctx = mp.get_context('spawn')  # spawn creates fresh process
        result_queue = ctx.Queue()
        
        process = ctx.Process(
            target=self._worker_process,
            args=(filepath, result_queue),
            daemon=False  # Don't use daemon to ensure cleanup
        )
        
        process.start()
        process.join(timeout=self.timeout_seconds)
        
        if process.is_alive():
            # Process timed out - force terminate
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            
            # Clear the queue
            try:
                while not result_queue.empty():
                    result_queue.get_nowait()
            except:
                pass
            
            return None
        
        # Get result from queue
        try:
            if not result_queue.empty():
                result = result_queue.get_nowait()
                if result.get('success') is not None:
                    return result['success']
        except:
            pass
        
        return None
    
    @staticmethod
    def _worker_process(filepath: Path, result_queue):
        """Worker process to extract inputs - runs in completely separate memory space"""
        try:
            # Set a stricter timeout using signal (Unix only)
            if hasattr(signal, 'SIGALRM'):
                def timeout_handler(signum, frame):
                    raise TimeoutError("Extraction timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)  # 5 second alarm
            
            # Load the module
            module_name = "temp_operation_module"
            spec = importlib.util.spec_from_file_location(module_name, str(filepath))
            if spec is None or spec.loader is None:
                result_queue.put({'error': 'Failed to load module spec'})
                return
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Get the get_inputs function
            func = getattr(module, 'get_inputs', None)
            
            if func is None:
                result_queue.put({'error': 'No get_inputs function found'})
                return
            
            # Call the function
            inputs = func()
            
            # Analyze inputs WITHOUT keeping references
            input_info = InputExtractorAgent._serialize_inputs(inputs)
            
            result_queue.put({'success': input_info})
            
            # Cancel alarm if it was set
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            # Explicit cleanup
            del inputs
            del func
            del module
            if module_name in sys.modules:
                del sys.modules[module_name]
            gc.collect()
            
        except TimeoutError:
            result_queue.put({'error': 'Timeout'})
        except Exception as e:
            result_queue.put({'error': f"{type(e).__name__}: {str(e)}"})
    
    @staticmethod
    def _serialize_inputs(inputs) -> Dict[str, Any]:
        """Serialize inputs to basic Python types (no tensor references)"""
        if isinstance(inputs, dict):
            return {key: InputExtractorAgent._analyze_tensor_lightweight(value) 
                    for key, value in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            return {'inputs': [InputExtractorAgent._analyze_tensor_lightweight(inp) 
                              for inp in inputs]}
        else:
            return {'input': InputExtractorAgent._analyze_tensor_lightweight(inputs)}
    
    @staticmethod
    def _analyze_tensor_lightweight(tensor) -> Dict[str, Any]:
        """Analyze tensor without holding references to large data"""
        try:
            # Check if it's a PyTorch tensor
            tensor_type = type(tensor).__name__
            tensor_module = getattr(type(tensor), '__module__', '')
            
            if 'torch' in tensor_module:
                # Extract info WITHOUT keeping reference
                info = {
                    'type': 'torch_tensor',
                    'shape': tuple(int(s) for s in tensor.shape) if hasattr(tensor, 'shape') else None,
                    'dtype': str(tensor.dtype).replace('torch.', '') if hasattr(tensor, 'dtype') else 'unknown',
                    'device': str(tensor.device) if hasattr(tensor, 'device') else 'unknown',
                }
                
                # Calculate memory
                if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                    numel = int(tensor.numel())
                    element_size = int(tensor.element_size())
                    info['numel'] = numel
                    info['element_size'] = element_size
                    info['memory_mb'] = (numel * element_size) / (1024 ** 2)
                
                # Optional attributes
                if hasattr(tensor, 'requires_grad'):
                    info['requires_grad'] = bool(tensor.requires_grad)
                if hasattr(tensor, 'is_contiguous'):
                    try:
                        info['is_contiguous'] = bool(tensor.is_contiguous())
                    except:
                        pass
                
                return info
                
            # Check if it's a NumPy array
            elif 'numpy' in tensor_module:
                info = {
                    'type': 'numpy_array',
                    'shape': tuple(int(s) for s in tensor.shape),
                    'dtype': str(tensor.dtype),
                    'device': 'cpu',
                }
                
                if hasattr(tensor, 'size') and hasattr(tensor, 'itemsize'):
                    numel = int(tensor.size)
                    element_size = int(tensor.itemsize)
                    info['numel'] = numel
                    info['element_size'] = element_size
                    info['memory_mb'] = (numel * element_size) / (1024 ** 2)
                
                return info
            
            # For lists and tuples - limit depth to prevent recursion
            elif isinstance(tensor, (list, tuple)):
                if len(tensor) > 10:
                    return {
                        'type': tensor_type,
                        'length': len(tensor),
                        'note': 'Large collection - not fully analyzed'
                    }
                return {
                    'type': tensor_type,
                    'length': len(tensor),
                    'elements': [InputExtractorAgent._analyze_tensor_lightweight(t) 
                                for t in tensor[:3]]
                }
            
            # For dictionaries
            elif isinstance(tensor, dict):
                return {
                    'type': 'dict',
                    'keys': list(tensor.keys())[:10],
                    'values': {k: InputExtractorAgent._analyze_tensor_lightweight(v) 
                             for k, v in list(tensor.items())[:3]}
                }
            
            # For scalar values
            elif isinstance(tensor, (int, float, bool, str)):
                return {
                    'type': tensor_type,
                    'value': tensor if isinstance(tensor, (int, float, bool)) else str(tensor)[:100]
                }
            
            # For other types
            else:
                return {
                    'type': tensor_type,
                    'size': sys.getsizeof(tensor) if tensor is not None else 0
                }
                
        except Exception as e:
            return {
                'error': str(e)[:100], 
                'type': type(tensor).__name__ if tensor is not None else 'None'
            }
    
    def _extract_tensor_shapes(self, content: str) -> List[Dict]:
        """Extract tensor shape definitions using safer parsing"""
        shapes = []
        
        for func_name, pattern in self.tensor_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                args = match.group(1)
                shape_dict = {
                    'function': func_name,
                    'raw': args
                }
                
                shape = self._parse_shape_args(args)
                shape_dict['shape'] = shape
                    
                shapes.append(shape_dict)
        
        return shapes
    
    def _parse_shape_args(self, args: str) -> Optional[Union[tuple, list]]:
        """Safely parse shape arguments"""
        try:
            # Remove dtype and device arguments
            args_clean = re.sub(r',\s*(dtype|device|requires_grad)\s*=\s*[^,)]+', '', args)
            
            # Handle cases like (2, 3), [2, 3], or 2, 3
            shape_match = re.match(r'^[\[\(]?([\d,\s]+)[\]\)]?', args_clean.strip())
            if shape_match:
                shape_str = shape_match.group(1)
                shape = tuple(int(x.strip()) for x in shape_str.split(',') if x.strip().isdigit())
                if shape:
                    return shape
                    
            # Try AST parsing as fallback
            try:
                parsed = ast.literal_eval(f"({args_clean})")
                if isinstance(parsed, (tuple, list)) and all(isinstance(x, int) for x in parsed):
                    return tuple(parsed)
                elif isinstance(parsed, tuple) and len(parsed) > 0:
                    if isinstance(parsed[0], (tuple, list)):
                        return tuple(parsed[0])
            except:
                pass
                
        except Exception:
            pass
            
        return None
    
    def _extract_dtypes(self, content: str) -> List[str]:
        """Extract data types used in operations"""
        dtypes = set()
        
        dtype_patterns = [
            r'dtype\s*=\s*torch\.(\w+)',
            r'\.to\(torch\.(\w+)\)',
            r'\.type\(torch\.(\w+)\)',
            r'\.(\w+)_\(\)',
        ]
        
        for pattern in dtype_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                dtype = match.group(1)
                if dtype in ['float16', 'float32', 'float64', 'int8', 'int16', 
                           'int32', 'int64', 'uint8', 'bool', 'float', 'double', 
                           'half', 'long', 'int', 'short', 'byte']:
                    dtypes.add(dtype)
        
        return list(dtypes)
    
    def _analyze_memory_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze memory access patterns"""
        patterns = {
            'has_transpose': bool(re.search(r'\.t\(\)|\.transpose\(|\.T\b|\.mT\b', content)),
            'has_reshape': bool(re.search(r'\.reshape\(|\.view\(', content)),
            'has_permute': bool(re.search(r'\.permute\(', content)),
            'has_contiguous': bool(re.search(r'\.contiguous\(\)', content)),
            'has_indexing': bool(re.search(r'\[[\s\S]*?:[\s\S]*?\]', content)),
            'has_batching': self._detect_batching(content),
            'has_squeeze': bool(re.search(r'\.squeeze\(|\.unsqueeze\(', content)),
            'has_flatten': bool(re.search(r'\.flatten\(', content)),
            'has_expand': bool(re.search(r'\.expand\(|\.expand_as\(', content)),
            'has_repeat': bool(re.search(r'\.repeat\(|\.repeat_interleave\(', content)),
        }
        
        return patterns
    
    def _detect_batching(self, content: str) -> bool:
        """Detect if operation uses batching"""
        batch_indicators = [
            r'\bbatch\b', r'\bbatched\b', r'\bbmm\b', r'\bB,', 
            r'batch_size', r'\.batch_', r'baddbmm', r'batch_norm'
        ]
        return any(re.search(indicator, content, re.IGNORECASE) for indicator in batch_indicators)
    
    def _extract_operation_calls(self, content: str) -> List[str]:
        """Extract main operation function calls"""
        operations = set()
        
        op_patterns = [
            (r'torch\.(\w+)\(', 'torch'),
            (r'F\.(\w+)\(', 'F'),
            (r'nn\.(\w+)\(', 'nn'),
            (r'torch\.nn\.functional\.(\w+)\(', 'F'),
            (r'torch\.linalg\.(\w+)\(', 'linalg'),
            (r'torch\.sparse\.(\w+)\(', 'sparse'),
        ]
        
        for pattern, prefix in op_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                op_name = match.group(1)
                if op_name not in ['tensor', 'Tensor', 'cuda', 'cpu', 'device']:
                    operations.add(f"{prefix}.{op_name}" if prefix != 'torch' else op_name)
        
        return sorted(list(operations))
    
    def _extract_parameters(self, content: str) -> Dict[str, Any]:
        """Extract operation parameters"""
        params = {}
        
        param_patterns = {
            'kernel_size': r'kernel_size\s*=\s*(\d+|\([^)]+\))',
            'stride': r'stride\s*=\s*(\d+|\([^)]+\))',
            'padding': r'padding\s*=\s*(\d+|\([^)]+\)|"[^"]+")',
            'dilation': r'dilation\s*=\s*(\d+|\([^)]+\))',
            'groups': r'groups\s*=\s*(\d+)',
            'bias': r'bias\s*=\s*(True|False)',
            'dim': r'dim\s*=\s*(-?\d+|\([^)]+\))',
            'keepdim': r'keepdim\s*=\s*(True|False)',
            'reduction': r'reduction\s*=\s*["\'](\w+)["\']',
        }
        
        for param_name, pattern in param_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                value = match.group(1)
                try:
                    if value in ['True', 'False']:
                        params[param_name] = value == 'True'
                    elif value.isdigit() or (value[0] == '-' and value[1:].isdigit()):
                        params[param_name] = int(value)
                    elif value.startswith('(') and value.endswith(')'):
                        params[param_name] = ast.literal_eval(value)
                    else:
                        params[param_name] = value.strip('"\'')
                except:
                    params[param_name] = value
                break
        
        return params
    
    def batch_extract(self, directory: Path, pattern: str = "*.py") -> List[Dict[str, Any]]:
        """Extract from multiple files in a directory"""
        results = []
        
        for filepath in directory.glob(pattern):
            if filepath.is_file():
                result = self.extract_from_file(filepath)
                results.append(result)
                
                # Force garbage collection after each file
                gc.collect()
        
        return results
    
    def summarize_extraction(self, extraction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize extraction results across multiple files"""
        summary = {
            'total_files': len(extraction_results),
            'files_with_get_input': 0,
            'common_shapes': {},
            'all_dtypes': set(),
            'all_operations': set(),
            'memory_pattern_stats': {},
            'extraction_errors': []
        }
        
        for result in extraction_results:
            if result.get('has_get_input'):
                summary['files_with_get_input'] += 1
            
            if 'error' in result:
                summary['extraction_errors'].append({
                    'file': result['filepath'],
                    'error': result['error']
                })
            
            for shape_info in result.get('tensor_shapes', []):
                if shape_info.get('shape'):
                    shape_key = str(shape_info['shape'])
                    summary['common_shapes'][shape_key] = summary['common_shapes'].get(shape_key, 0) + 1
            
            summary['all_dtypes'].update(result.get('dtypes', []))
            summary['all_operations'].update(result.get('operation_calls', []))
            
            for pattern, value in result.get('memory_patterns', {}).items():
                if pattern not in summary['memory_pattern_stats']:
                    summary['memory_pattern_stats'][pattern] = 0
                if value:
                    summary['memory_pattern_stats'][pattern] += 1
        
        summary['all_dtypes'] = sorted(list(summary['all_dtypes']))
        summary['all_operations'] = sorted(list(summary['all_operations']))
        summary['common_shapes'] = dict(sorted(
            summary['common_shapes'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return summary
    
    def cleanup(self):
        """Clean up any cached resources"""
        gc.collect()