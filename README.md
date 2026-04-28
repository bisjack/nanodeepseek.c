# nano deepseek.c

Inspired by [llm.c](https://github.com/karpathy/llm.c), a pure C implementation of [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) 16B version. MoE forward is finished at this stage.

## Dependencies

We need the python program to generate input, params and ground truth:

```pip-requirements
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

The `kernel.py`,`configs` and `moe.py` is borrowed from [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), where `moe.py` is modified. 

## Compilation

As simple as

```bash
gcc model.c -o model -lm
```

## Verification

Lowered the params to give my cpu a relief.

```bash
python model.py
```
This will generate a bunch of txt files.

```bash
./model
```
This will output something like this:

```
Rate: 0.991974
Rate: 0.997009
Rate: 0.996307
```

## Future

Finish the attn part and piece them together, then implement it on GPU.

## license

MIT
