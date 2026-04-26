# CapstoneL17

For running a Demixing model: 
1. Check that bs-roformer-infer package is installed or install it if it's not:

```!pip install -q git+https://github.com/openmirlab/bs-roformer-infer.git```

2. run the command:

```
bs-roformer-infer --config_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.yaml --model_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.ckpt --input_folder ./input_songs --store_dir ./outputs
```
Source: https://github.com/openmirlab/bs-roformer-infer
