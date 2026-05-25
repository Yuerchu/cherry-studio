import http from 'node:http'

import { loggerService } from '@main/services/LoggerService'

const logger = loggerService.withContext('OpenAIBridge')

// ─── Types ───────────────────────────────────────────────────────────────────

type AnyBlock = Record<string, unknown>

type OpenAIToolCall = {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

type OpenAIChatContentPart = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }

type OpenAIChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | OpenAIChatContentPart[] | null
  tool_call_id?: string
  tool_calls?: OpenAIToolCall[]
}

type OpenAIChatRequest = {
  model: string
  messages: OpenAIChatMessage[]
  stream?: boolean
  enable_thinking?: boolean
  thinking_budget?: number
  temperature?: number
  tools?: Array<{ type: 'function'; function: { name: string; description?: string; parameters?: unknown } }>
  tool_choice?: 'auto' | { type: 'function'; function: { name: string } }
  max_tokens?: number
}

type OpenAIStreamChunk = {
  id?: string
  model?: string
  choices?: Array<{
    index?: number
    delta?: {
      role?: 'assistant'
      content?: string | null
      reasoning_content?: string | null
      tool_calls?: Array<{
        index?: number
        id?: string
        type?: 'function'
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason?: string | null
  }>
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
}

export interface OpenAIBridgeConfig {
  apiKey: string
  baseURL: string
  model: string
}

export interface OpenAIBridgeHandle {
  port: number
  close: () => void
}

// ─── Request Conversion (Anthropic → OpenAI) ────────────────────────────────

function toBlocks(content: unknown): AnyBlock[] {
  return Array.isArray(content) ? (content as AnyBlock[]) : [{ type: 'text', text: content }]
}

function mapUserBlocksToOpenAI(blocks: AnyBlock[]): OpenAIChatContentPart[] {
  return blocks.flatMap<OpenAIChatContentPart>((block) => {
    if (block.type === 'text' && typeof block.text === 'string' && block.text.length > 0) {
      return [{ type: 'text' as const, text: block.text }]
    }
    if (
      block.type === 'image' &&
      block.source &&
      typeof block.source === 'object' &&
      (block.source as AnyBlock).type === 'base64' &&
      typeof (block.source as AnyBlock).media_type === 'string' &&
      typeof (block.source as AnyBlock).data === 'string'
    ) {
      const src = block.source as AnyBlock
      return [
        {
          type: 'image_url' as const,
          image_url: { url: `data:${src.media_type};base64,${src.data}` }
        }
      ]
    }
    return []
  })
}

function convertToolDefinitions(
  tools?: Array<{ name?: string; description?: string; input_schema?: unknown }>
): OpenAIChatRequest['tools'] | undefined {
  if (!tools || tools.length === 0) return undefined
  const mapped = tools.flatMap((tool) => {
    if (!tool.name) return []
    return [
      {
        type: 'function' as const,
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.input_schema
        }
      }
    ]
  })
  return mapped.length > 0 ? mapped : undefined
}

function convertAnthropicToOpenAI(body: AnyBlock, model: string): OpenAIChatRequest {
  const messages: OpenAIChatMessage[] = []

  if (body.system) {
    const systemText = Array.isArray(body.system)
      ? (body.system as Array<{ text?: string }>).map((b) => b.text ?? '').join('\n')
      : String(body.system)
    if (systemText) messages.push({ role: 'system', content: systemText })
  }

  for (const message of (body.messages as Array<{ role: string; content: unknown }>) ?? []) {
    if (message.role === 'user') {
      const blocks = toBlocks(message.content)
      const nonToolBlocks = blocks.filter((b) => b.type !== 'tool_result')
      const userContent = mapUserBlocksToOpenAI(nonToolBlocks)
      if (userContent.length > 0) {
        messages.push({ role: 'user', content: userContent })
      }

      for (const result of blocks.filter((b) => b.type === 'tool_result')) {
        messages.push({
          role: 'tool',
          tool_call_id: typeof result.tool_use_id === 'string' ? result.tool_use_id : undefined,
          content: typeof result.content === 'string' ? result.content : JSON.stringify(result.content)
        })
      }
      continue
    }

    if (message.role === 'assistant') {
      const blocks = Array.isArray(message.content) ? (message.content as AnyBlock[]) : []
      const text = blocks
        .filter((b) => b.type === 'text')
        .map((b) => (typeof b.text === 'string' ? b.text : ''))
        .join('')
      const toolCalls = blocks
        .filter((b) => b.type === 'tool_use')
        .map((b) => ({
          id: String(b.id),
          type: 'function' as const,
          function: {
            name: String(b.name),
            arguments: typeof b.input === 'string' ? b.input : JSON.stringify(b.input ?? {})
          }
        }))

      messages.push({
        role: 'assistant',
        content: text || null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {})
      })
    }
  }

  const thinking = body.thinking as { type?: string; budget_tokens?: number } | undefined
  const toolChoice = body.tool_choice as { type?: string; name?: string } | undefined

  return {
    model,
    messages,
    stream: true,
    enable_thinking: thinking?.type === 'enabled' || thinking?.type === 'adaptive',
    ...(thinking?.type === 'enabled' && typeof thinking.budget_tokens === 'number'
      ? { thinking_budget: thinking.budget_tokens }
      : {}),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
    ...(convertToolDefinitions(body.tools as Array<{ name?: string; description?: string; input_schema?: unknown }>)
      ? {
          tools: convertToolDefinitions(
            body.tools as Array<{ name?: string; description?: string; input_schema?: unknown }>
          )
        }
      : {}),
    ...(toolChoice?.type === 'tool'
      ? { tool_choice: { type: 'function' as const, function: { name: toolChoice.name! } } }
      : toolChoice?.type === 'auto'
        ? { tool_choice: 'auto' as const }
        : {})
  }
}

// ─── Response Conversion (OpenAI SSE → Anthropic SSE) ────────────────────────

function mapFinishReason(reason: string | null | undefined): string {
  if (reason === 'tool_calls') return 'tool_use'
  if (reason === 'length') return 'max_tokens'
  return 'end_turn'
}

function parseSSEChunks(buffer: string): { events: string[]; remainder: string } {
  const normalized = buffer.replace(/\r\n/g, '\n')
  const parts = normalized.split('\n\n')
  const remainder = parts.pop() ?? ''
  return { events: parts, remainder }
}

function writeAnthropicEvent(res: http.ServerResponse, eventType: string, data: unknown): void {
  res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`)
}

async function streamOpenAIToAnthropic(
  openaiResponse: globalThis.Response,
  res: http.ServerResponse,
  model: string
): Promise<void> {
  if (!openaiResponse.body) {
    throw new Error('OpenAI response has no body')
  }

  const reader = openaiResponse.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let started = false
  let textStarted = false
  let textContentIndex: number | null = null
  let thinkingStarted = false
  let thinkingContentIndex: number | null = null
  const toolIndexByOpenAIIndex = new Map<number, number>()
  let nextContentIndex = 0
  let promptTokens = 0
  let completionTokens = 0
  let emittedAnyContent = false
  const toolCallState = new Map<number, { id: string; name: string; arguments: string }>()

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parsed = parseSSEChunks(buffer)
      buffer = parsed.remainder

      for (const rawEvent of parsed.events) {
        const dataLines = rawEvent
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trim())

        for (const data of dataLines) {
          if (!data || data === '[DONE]') continue
          let chunk: OpenAIStreamChunk
          try {
            chunk = JSON.parse(data)
          } catch {
            logger.warn('Failed to parse OpenAI stream chunk', { data: data.slice(0, 200) })
            continue
          }

          const choice = chunk.choices?.[0]
          const delta = choice?.delta

          if (!choice) continue

          if (!started) {
            started = true
            promptTokens = chunk.usage?.prompt_tokens ?? 0
            writeAnthropicEvent(res, 'message_start', {
              type: 'message_start',
              message: {
                id: chunk.id ?? 'openai-bridge',
                type: 'message',
                role: 'assistant',
                model,
                content: [],
                stop_reason: null,
                stop_sequence: null,
                usage: { input_tokens: promptTokens, output_tokens: 0 }
              }
            })
          }

          if (delta?.content) {
            if (!textStarted) {
              textStarted = true
              textContentIndex = nextContentIndex++
              writeAnthropicEvent(res, 'content_block_start', {
                type: 'content_block_start',
                index: textContentIndex,
                content_block: { type: 'text', text: '' }
              })
            }
            writeAnthropicEvent(res, 'content_block_delta', {
              type: 'content_block_delta',
              index: textContentIndex ?? 0,
              delta: { type: 'text_delta', text: delta.content }
            })
            emittedAnyContent = true
          }

          if (delta?.reasoning_content) {
            if (!thinkingStarted) {
              thinkingStarted = true
              thinkingContentIndex = nextContentIndex++
              writeAnthropicEvent(res, 'content_block_start', {
                type: 'content_block_start',
                index: thinkingContentIndex,
                content_block: { type: 'thinking', thinking: '', signature: '' }
              })
            }
            writeAnthropicEvent(res, 'content_block_delta', {
              type: 'content_block_delta',
              index: thinkingContentIndex ?? 0,
              delta: { type: 'thinking_delta', thinking: delta.reasoning_content }
            })
            emittedAnyContent = true
          }

          for (const toolCall of delta?.tool_calls ?? []) {
            const openAIIndex = toolCall.index ?? 0
            let anthropicIndex = toolIndexByOpenAIIndex.get(openAIIndex)
            if (anthropicIndex === undefined) {
              anthropicIndex = nextContentIndex++
              toolIndexByOpenAIIndex.set(openAIIndex, anthropicIndex)
              const state = {
                id: toolCall.id ?? `toolu_${openAIIndex}`,
                name: toolCall.function?.name ?? '',
                arguments: ''
              }
              toolCallState.set(openAIIndex, state)
              writeAnthropicEvent(res, 'content_block_start', {
                type: 'content_block_start',
                index: anthropicIndex,
                content_block: { type: 'tool_use', id: state.id, name: state.name, input: '' }
              })
            }

            const state = toolCallState.get(openAIIndex)
            if (!state) continue
            if (toolCall.id) state.id = toolCall.id
            if (toolCall.function?.name) state.name = toolCall.function.name
            if (toolCall.function?.arguments) {
              state.arguments += toolCall.function.arguments
              writeAnthropicEvent(res, 'content_block_delta', {
                type: 'content_block_delta',
                index: anthropicIndex,
                delta: { type: 'input_json_delta', partial_json: toolCall.function.arguments }
              })
              emittedAnyContent = true
            }
          }

          if (choice.finish_reason) {
            if (!emittedAnyContent) {
              writeAnthropicEvent(res, 'content_block_start', {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'text', text: '' }
              })
              writeAnthropicEvent(res, 'content_block_stop', { type: 'content_block_stop', index: 0 })
            }

            completionTokens = chunk.usage?.completion_tokens ?? completionTokens

            if (textStarted && textContentIndex !== null) {
              writeAnthropicEvent(res, 'content_block_stop', { type: 'content_block_stop', index: textContentIndex })
            }
            if (thinkingStarted && thinkingContentIndex !== null) {
              writeAnthropicEvent(res, 'content_block_stop', {
                type: 'content_block_stop',
                index: thinkingContentIndex
              })
            }
            for (const idx of toolIndexByOpenAIIndex.values()) {
              writeAnthropicEvent(res, 'content_block_stop', { type: 'content_block_stop', index: idx })
            }

            writeAnthropicEvent(res, 'message_delta', {
              type: 'message_delta',
              delta: { stop_reason: mapFinishReason(choice.finish_reason), stop_sequence: null },
              usage: { output_tokens: completionTokens }
            })
            writeAnthropicEvent(res, 'message_stop', { type: 'message_stop' })
            return
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

// ─── Bridge Server ───────────────────────────────────────────────────────────

function resolveCompletionsUrl(rawBaseURL: string): string {
  const base = rawBaseURL.replace(/\/$/, '')
  if (base.endsWith('/chat/completions')) return base
  if (base.endsWith('/v1')) return `${base}/chat/completions`
  return `${base}/v1/chat/completions`
}

export async function startOpenAIBridge(config: OpenAIBridgeConfig): Promise<OpenAIBridgeHandle> {
  const completionsUrl = resolveCompletionsUrl(config.baseURL)

  const server = http.createServer(async (req, res) => {
    if (req.method !== 'POST' || !req.url?.includes('/v1/messages')) {
      res.writeHead(404, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: 'Not found' }))
      return
    }

    const chunks: Buffer[] = []
    for await (const chunk of req) {
      chunks.push(chunk as Buffer)
    }

    let anthropicBody: AnyBlock
    try {
      anthropicBody = JSON.parse(Buffer.concat(chunks).toString())
    } catch {
      res.writeHead(400, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: 'Invalid JSON body' }))
      return
    }

    const openaiRequest = convertAnthropicToOpenAI(anthropicBody, config.model)

    try {
      const openaiResponse = await fetch(completionsUrl, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${config.apiKey}`
        },
        body: JSON.stringify(openaiRequest)
      })

      if (!openaiResponse.ok) {
        const errorText = await openaiResponse.text().catch(() => '')
        logger.error('OpenAI provider returned error', {
          status: openaiResponse.status,
          body: errorText.slice(0, 500)
        })
        res.writeHead(openaiResponse.status, { 'content-type': 'application/json' })
        res.end(
          JSON.stringify({
            type: 'error',
            error: { type: 'api_error', message: `Upstream error ${openaiResponse.status}: ${errorText.slice(0, 200)}` }
          })
        )
        return
      }

      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive'
      })

      await streamOpenAIToAnthropic(openaiResponse, res, config.model)
    } catch (error) {
      logger.error('Bridge request failed', { error: error instanceof Error ? error.message : String(error) })
      if (!res.headersSent) {
        res.writeHead(500, { 'content-type': 'application/json' })
      }
      res.end(
        JSON.stringify({
          type: 'error',
          error: { type: 'api_error', message: error instanceof Error ? error.message : String(error) }
        })
      )
    } finally {
      res.end()
    }
  })

  return new Promise((resolve, reject) => {
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address()
      if (!addr || typeof addr === 'string') {
        reject(new Error('Failed to get bridge server address'))
        return
      }
      logger.info('OpenAI bridge started', { port: addr.port })
      resolve({
        port: addr.port,
        close: () => {
          server.close()
          logger.info('OpenAI bridge closed', { port: addr.port })
        }
      })
    })
    server.on('error', reject)
  })
}
