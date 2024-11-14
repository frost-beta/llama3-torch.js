#!/usr/bin/env node

import readline from 'node:readline/promises'
import {styleText} from 'node:util'
import {readFileSync} from 'node:fs'
import {TokenizerLoader} from '@lenml/tokenizers'
import {DType, Module, Tensor, sample} from 'executorch'

const dir = process.argv[2]
if (!dir) {
  console.error('Usage: chat.mjs /path/to/weights/dir')
  process.exit(0)
}

// Load tokenizer.
const tokenizerConfig = readJsonSync(`${dir}/tokenizer_config.json`)
const tokenizer = TokenizerLoader.fromPreTrained({
  tokenizerConfig,
  tokenizerJSON: readJsonSync(`${dir}/tokenizer.json`),
})
tokenizer.eosToken = tokenizer.model.tokens_to_ids.get(tokenizerConfig.eos_token)

// Load model.
const mod = new Module(`${dir}/llama3.2-instruct-executorch-kv-xnnpack-bf16.pte`)
await mod.load()

// Enter chat loop.
chat(tokenizer, mod)

async function chat(tokenizer, mod) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  })
  rl.once('close', () => process.stdout.write('\n'))
  const messages = []
  const youPrompt = styleText('green', 'You> ')
  const botPrompt = styleText('blue', 'Assistant> ')
  while (!process.stdin.closed) {
    const question = await rl.question(youPrompt)
    messages.push({role: 'user', content: question})
    process.stdout.write(botPrompt)
    const reply = await talk(rl, tokenizer, mod, messages)
    messages.push({role: 'assistant', content: reply})
  }
}

async function talk(rl, tokenizer, mod, messages) {
  // Interrupt generation when Ctrl-C is pressed.
  const controller = new AbortController()
  const abort = () => controller.abort()
  const {signal} = controller
  rl.on('SIGINT', abort)
  // Encode messages into tokens.
  const promptTokens = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
  })
  // Decode predicted tokens.
  let text = ''
  for await (const token of step(signal, tokenizer, mod, promptTokens)) {
    const char = tokenizer.decode([token])
    text += char
    process.stdout.write(char)
  }
  process.stdout.write('\n')
  // Cleanup.
  rl.removeListener('SIGINT', abort)
  return text
}

async function* step(signal, tokenizer, mod, promptTokens) {
  let token
  let pos = promptTokens.length
  // Prefill prompt.
  const prefillStep = 512
  for (let i = 0; i < prefillStep; i += prefillStep) {
    if (signal.aborted)
      return
    const end = Math.min(i + prefillStep, promptTokens.length)
    const output = await mod.forward(new Tensor([ promptTokens.slice(i,  end) ], DType.Int64),
                                     new Tensor([ i * prefillStep ], DType.Int64))
    if (promptTokens.length < i + prefillStep)
      token = sample(output, {topP: 0.9})
  }
  // Forward.
  while (true) {
    if (signal.aborted)
      return
    if (token == tokenizer.eosToken)
      return
    yield token
    const output = await mod.forward(new Tensor([ [ token ]], DType.Int64),
                                     new Tensor([ pos ], DType.Int64))
    token = sample(output, {topP: 0.9})
    pos++
  }
}

function readJsonSync(path) {
  return JSON.parse(String(readFileSync(path)))
}
