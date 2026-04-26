# CapstoneL17

To run the demixing model, use the following command:

```
bs-roformer-infer --config_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.yaml --model_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.ckpt --input_folder ./input_songs --store_dir ./outputs
```

The required dependencies can be found in the ```requirements.txt``` file.

Source: https://github.com/openmirlab/bs-roformer-infer
