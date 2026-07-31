import importlib
for m in ['torch', 'torchvision', 'onnx', 'onnxruntime', 'numpy', 'PIL', 'sklearn', 'matplotlib']:
    try:
        mod = importlib.import_module(m)
        print(f'{m}: OK ({getattr(mod, "__version__", "?")})')
    except ImportError:
        print(f'{m}: MISSING')
