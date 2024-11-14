# Llama with ExecuTorch.js

This is an example showing how to run Llama 3.2 with
[ExecuTorch.js](https://github.com/frost-beta/executorch.js), the code is not
production ready.

## How to use

Download model:

```console
$ npm install -g @frost-beta/huggingface
$ huggingface download frost-beta/llama3.2-instruct-executorch-kv-xnnpack-bf16
```

Run this script:

```console
$ npm install
$ node chat.mjs llama3.2-instruct-executorch-kv-xnnpack-bf16
You> Who are you?
Assistant> I'm an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."
You>
```
