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
gcc moe.c -o moe -lm
```

## Verification

Lowered the params to give my cpu a relief. Actually the tolerance of relative error is a bit intolerant... Wish I can figure it out later.

```bash
python moe.py
```
This will generate a bunch of txt files.

```bash
./moe
```
This will output something like this:

```
moe forward took 0.867705 seconds to execute.
passed
```

## Future

Finish the attn part and piece them together, then implement it on GPU.

## license

MIT