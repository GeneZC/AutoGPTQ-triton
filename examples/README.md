# Examples

To run example scripts in this folder, one must first install `auto_gptq` as described in [this](../README.md)

## Basic Usage
Run the following code to execute `basic_usage.py`:
```shell
python basic_usage.py
```

## Quantize with Alpaca

Then Execute `quant_with_alpaca.py` using command like this:
```shell
CUDA_VISIBLE_DEVICES=0 python quant_with_alpaca.py --pretrained_model_dir FreedomIntelligence/phoenix-inst-chat-7b --quantized_model_dir phoenix-inst-chat-7b-int4 --save_and_reload --fast_tokenizer
```

The alpaca dataset used in here is a cleaned version provided by **gururise** in [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)
