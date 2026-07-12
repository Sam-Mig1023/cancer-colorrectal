
import h5py
import json

h5_path = 'models/mobilenetv2_base_only.h5'

with h5py.File(h5_path, 'r') as f:
    # Check model config
    if 'model_config' in f.attrs:
        model_config = json.loads(f.attrs['model_config'])
        print("="*80)
        print("Full Model Config")
        print("="*80)
        print(json.dumps(model_config, indent=2))
    else:
        print("No model_config attribute found in H5 file.")
