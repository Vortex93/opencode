import { BusEvent } from "@/bus/bus-event"
import { Bus } from "@/bus"
import { Session } from "."
import { Identifier } from "../id/id"
import { Instance } from "../project/instance"
import { Provider } from "../provider/provider"
import { MessageV2 } from "./message-v2"
import z from "zod"
import { SessionPrompt } from "./prompt"
import { Token } from "../util/token"
import { Log } from "../util/log"
import { SessionProcessor } from "./processor"
import { fn } from "@/util/fn"
import { Agent } from "@/agent/agent"
import { Plugin } from "@/plugin"
import { Config } from "@/config/config"

export namespace SessionCompaction {
  const log = Log.create({ service: "session.compaction" })

  const DEFAULT_SLIDING_THRESHOLD = 0.6
  const DEFAULT_SUMMARY_THRESHOLD = 1
  const MIN_KEEP_MESSAGES = 2

  type SlidingWindow = {
    messages: MessageV2.WithParts[]
    removed: number
    manual: number
    automatic: number
    remainingEstimate: number
    target: number
  }

  function normalizeThreshold(value: number | undefined, fallback: number) {
    if (!Number.isFinite(value)) return fallback
    const threshold = value ?? fallback
    if (threshold <= 0) return fallback
    return Math.min(1, Math.max(0.1, threshold))
  }

  async function getCompactionSettings() {
    const config = await Config.get()
    const mode = config.compaction?.mode ?? "sliding"
    const fallback = mode === "sliding" ? DEFAULT_SLIDING_THRESHOLD : DEFAULT_SUMMARY_THRESHOLD
    const threshold = normalizeThreshold(config.compaction?.threshold, fallback)
    return { config, mode, threshold }
  }

  function getUsableTokens(model: Provider.Model) {
    const context = model.limit.context
    if (context === 0) return 0
    const output = Math.min(model.limit.output, SessionPrompt.OUTPUT_TOKEN_MAX) || SessionPrompt.OUTPUT_TOKEN_MAX
    return model.limit.input || context - output
  }

  function manualSlides(session: Session.Info) {
    const count = session.sliding?.manual ?? 0
    return Number.isFinite(count) ? Math.max(0, Math.floor(count)) : 0
  }

  async function setManualSlides(sessionID: string, count: number) {
    const next = Math.max(0, Math.floor(count))
    await Session.update(sessionID, (draft) => {
      if (next === 0) {
        draft.sliding = undefined
        return
      }
      draft.sliding = {
        manual: next,
      }
    })
  }

  function computeSlidingWindow(input: {
    messages: MessageV2.WithParts[]
    model: Provider.Model
    threshold: number
    manual: number
  }): SlidingWindow {
    const usable = getUsableTokens(input.model)
    if (usable === 0)
      return {
        messages: input.messages,
        removed: 0,
        manual: 0,
        automatic: 0,
        remainingEstimate: 0,
        target: 0,
      }

    const target = Math.floor(usable * input.threshold)
    const maxRemovable = Math.max(0, input.messages.length - MIN_KEEP_MESSAGES)
    const manual = Math.min(Math.max(0, Math.floor(input.manual)), maxRemovable)
    let removed = manual
    let total = estimateMessagesTokens(input.messages.slice(removed))

    while (removed < maxRemovable && total > target) {
      total -= estimateMessageTokens(input.messages[removed])
      removed += 1
    }

    return {
      messages: input.messages.slice(removed),
      removed,
      manual,
      automatic: removed - manual,
      remainingEstimate: total,
      target,
    }
  }

  export const Event = {
    Compacted: BusEvent.define(
      "session.compacted",
      z.object({
        sessionID: z.string(),
      }),
    ),
  }

  export async function isOverflow(input: {
    tokens: MessageV2.Assistant["tokens"]
    model: Provider.Model
    sessionID?: string
    messages?: MessageV2.WithParts[]
  }) {
    const { config, mode, threshold } = await getCompactionSettings()
    if (config.compaction?.auto === false) return false
    const usable = getUsableTokens(input.model)
    if (usable === 0) return false
    if (mode === "sliding") {
      if (!input.messages || !input.sessionID) return false
      const session = await Session.get(input.sessionID)
      const result = computeSlidingWindow({
        messages: input.messages,
        model: input.model,
        threshold,
        manual: manualSlides(session),
      })
      return result.remainingEstimate > result.target
    }
    const count = input.tokens.input + input.tokens.cache.read + input.tokens.output
    return count > usable * threshold
  }

  export const PRUNE_MINIMUM = 20_000
  export const PRUNE_PROTECT = 40_000

  const PRUNE_PROTECTED_TOOLS = ["skill"]

  // goes backwards through parts until there are 40_000 tokens worth of tool
  // calls. then erases output of previous tool calls. idea is to throw away old
  // tool calls that are no longer relevant.
  export async function prune(input: { sessionID: string }) {
    const config = await Config.get()
    if (config.compaction?.prune === false) return
    log.info("pruning")
    const msgs = await Session.messages({ sessionID: input.sessionID })
    let total = 0
    let pruned = 0
    const toPrune = []
    let turns = 0

    loop: for (let msgIndex = msgs.length - 1; msgIndex >= 0; msgIndex--) {
      const msg = msgs[msgIndex]
      if (msg.info.role === "user") turns++
      if (turns < 2) continue
      if (msg.info.role === "assistant" && msg.info.summary) break loop
      for (let partIndex = msg.parts.length - 1; partIndex >= 0; partIndex--) {
        const part = msg.parts[partIndex]
        if (part.type === "tool")
          if (part.state.status === "completed") {
            if (PRUNE_PROTECTED_TOOLS.includes(part.tool)) continue

            if (part.state.time.compacted) break loop
            const estimate = Token.estimate(part.state.output)
            total += estimate
            if (total > PRUNE_PROTECT) {
              pruned += estimate
              toPrune.push(part)
            }
          }
      }
    }
    log.info("found", { pruned, total })
    if (pruned > PRUNE_MINIMUM) {
      for (const part of toPrune) {
        if (part.state.status === "completed") {
          part.state.time.compacted = Date.now()
          await Session.updatePart(part)
        }
      }
      log.info("pruned", { count: toPrune.length })
    }
  }

  export async function window(input: {
    sessionID: string
    messages: MessageV2.WithParts[]
    model: Provider.Model
  }) {
    const { mode, threshold } = await getCompactionSettings()
    const session = await Session.get(input.sessionID)
    const manual = manualSlides(session)
    if (mode !== "sliding") {
      const maxRemovable = Math.max(0, input.messages.length - MIN_KEEP_MESSAGES)
      const removed = Math.min(manual, maxRemovable)
      const messages = input.messages.slice(removed)
      const usable = getUsableTokens(input.model)
      const target = usable === 0 ? 0 : Math.floor(usable * threshold)
      return {
        messages,
        removed,
        manual: removed,
        automatic: 0,
        remainingEstimate: estimateMessagesTokens(messages),
        target,
      }
    }
    return computeSlidingWindow({
      messages: input.messages,
      model: input.model,
      threshold,
      manual,
    })
  }

  export const slide = fn(
    z.object({
      sessionID: Identifier.schema("session"),
      count: z.number().int().positive().optional().default(1),
    }),
    async (input) => {
      const session = await Session.get(input.sessionID)
      const current = manualSlides(session)
      const messages = await Session.messages({ sessionID: input.sessionID })
      const maxRemovable = Math.max(0, messages.length - MIN_KEEP_MESSAGES)
      const next = Math.min(maxRemovable, current + input.count)
      if (next !== current) {
        await setManualSlides(input.sessionID, next)
      }
      Bus.publish(Event.Compacted, { sessionID: input.sessionID })
      return {
        removed: next - current,
        manual: next,
      }
    },
  )

  export async function process(input: {
    parentID: string
    messages: MessageV2.WithParts[]
    sessionID: string
    abort: AbortSignal
    auto: boolean
  }) {
    const { mode } = await getCompactionSettings()
    if (mode === "sliding") {
      return processSliding(input)
    }
    return processSummarize(input)
  }

  async function removeMessageWithParts(message: MessageV2.WithParts) {
    for (const part of message.parts) {
      await Session.removePart({
        sessionID: message.info.sessionID,
        messageID: message.info.id,
        partID: part.id,
      })
    }
    await Session.removeMessage({
      sessionID: message.info.sessionID,
      messageID: message.info.id,
    })
  }

  function estimatePartTokens(part: MessageV2.Part) {
    switch (part.type) {
      case "text":
      case "reasoning":
        return Token.estimate(part.text)
      case "tool": {
        const input = Token.estimate(JSON.stringify(part.state.input ?? {}))
        if (part.state.status === "completed") return input + Token.estimate(part.state.output)
        if (part.state.status === "error") return input + Token.estimate(part.state.error)
        return input
      }
      case "file":
        if (part.source?.text?.value) return Token.estimate(part.source.text.value)
        if (part.filename) return Token.estimate(part.filename)
        return Token.estimate(part.url)
      case "patch":
        return Token.estimate(part.files.join(" "))
      case "subtask":
        return Token.estimate(`${part.prompt}\n${part.description}`)
      case "agent":
        return Token.estimate(part.name)
      case "retry":
        return Token.estimate(part.error.data.message)
      case "compaction":
      case "snapshot":
      case "step-start":
      case "step-finish":
        return 0
      default:
        return 0
    }
  }

  function estimateMessageTokens(message: MessageV2.WithParts) {
    let total = 0
    if (message.info.role === "user" && message.info.system) {
      total += Token.estimate(message.info.system)
    }
    for (const part of message.parts) {
      total += estimatePartTokens(part)
    }
    return total
  }

  function estimateMessagesTokens(messages: MessageV2.WithParts[]) {
    return messages.reduce((sum, msg) => sum + estimateMessageTokens(msg), 0)
  }

  async function slideSession(input: {
    sessionID: string
    messages: MessageV2.WithParts[]
    model: Provider.Model
    threshold: number
    removeMessageID?: string
  }) {
    const session = await Session.get(input.sessionID)
    const messages = input.removeMessageID
      ? input.messages.filter((msg) => msg.info.id !== input.removeMessageID)
      : input.messages
    return computeSlidingWindow({
      messages,
      model: input.model,
      threshold: input.threshold,
      manual: manualSlides(session),
    })
  }

  async function processSliding(input: {
    parentID: string
    messages: MessageV2.WithParts[]
    sessionID: string
    abort: AbortSignal
    auto: boolean
  }) {
    const { config, threshold } = await getCompactionSettings()
    if (config.compaction?.auto === false && input.auto) return "continue"
    const userMessage = input.messages.findLast((m) => m.info.id === input.parentID)?.info as MessageV2.User
    if (!userMessage) return "continue"

    const agent = await Agent.get("compaction")
    const model = agent.model
      ? await Provider.getModel(agent.model.providerID, agent.model.modelID)
      : await Provider.getModel(userMessage.model.providerID, userMessage.model.modelID)

    const result = await slideSession({
      sessionID: input.sessionID,
      messages: input.messages,
      model,
      threshold,
      removeMessageID: input.parentID,
    })

    if (!input.auto) {
      await setManualSlides(input.sessionID, result.removed)
    }

    const compactionMessage = input.messages.find((msg) => msg.info.id === input.parentID)
    if (compactionMessage && compactionMessage.parts.some((part) => part.type === "compaction")) {
      await removeMessageWithParts(compactionMessage)
    }

    log.info("sliding", {
      removed: result.removed,
      manual: result.manual,
      automatic: result.automatic,
      remainingEstimate: result.remainingEstimate,
      target: result.target,
    })
    Bus.publish(Event.Compacted, { sessionID: input.sessionID })
    return "continue"
  }

  async function processSummarize(input: {
    parentID: string
    messages: MessageV2.WithParts[]
    sessionID: string
    abort: AbortSignal
    auto: boolean
  }) {
    const userMessage = input.messages.findLast((m) => m.info.id === input.parentID)!.info as MessageV2.User
    const agent = await Agent.get("compaction")
    const model = agent.model
      ? await Provider.getModel(agent.model.providerID, agent.model.modelID)
      : await Provider.getModel(userMessage.model.providerID, userMessage.model.modelID)
    const msg = (await Session.updateMessage({
      id: Identifier.ascending("message"),
      role: "assistant",
      parentID: input.parentID,
      sessionID: input.sessionID,
      mode: "compaction",
      agent: "compaction",
      variant: userMessage.variant,
      summary: true,
      path: {
        cwd: Instance.directory,
        root: Instance.worktree,
      },
      cost: 0,
      tokens: {
        output: 0,
        input: 0,
        reasoning: 0,
        cache: { read: 0, write: 0 },
      },
      modelID: model.id,
      providerID: model.providerID,
      time: {
        created: Date.now(),
      },
    })) as MessageV2.Assistant
    const processor = SessionProcessor.create({
      assistantMessage: msg,
      sessionID: input.sessionID,
      model,
      abort: input.abort,
    })
    // Allow plugins to inject context or replace compaction prompt
    const compacting = await Plugin.trigger(
      "experimental.session.compacting",
      { sessionID: input.sessionID },
      { context: [], prompt: undefined },
    )
    const defaultPrompt =
      "Provide a detailed prompt for continuing our conversation above. Focus on information that would be helpful for continuing the conversation, including what we did, what we're doing, which files we're working on, and what we're going to do next considering new session will not have access to our conversation."
    const promptText = compacting.prompt ?? [defaultPrompt, ...compacting.context].join("\n\n")
    const result = await processor.process({
      user: userMessage,
      agent,
      abort: input.abort,
      sessionID: input.sessionID,
      tools: {},
      system: [],
      messages: [
        ...MessageV2.toModelMessages(input.messages, model),
        {
          role: "user",
          content: [
            {
              type: "text",
              text: promptText,
            },
          ],
        },
      ],
      model,
    })

    if (result === "continue" && input.auto) {
      const continueMsg = await Session.updateMessage({
        id: Identifier.ascending("message"),
        role: "user",
        sessionID: input.sessionID,
        time: {
          created: Date.now(),
        },
        agent: userMessage.agent,
        model: userMessage.model,
      })
      await Session.updatePart({
        id: Identifier.ascending("part"),
        messageID: continueMsg.id,
        sessionID: input.sessionID,
        type: "text",
        synthetic: true,
        text: "Continue if you have next steps",
        time: {
          start: Date.now(),
          end: Date.now(),
        },
      })
    }
    if (processor.message.error) return "stop"
    Bus.publish(Event.Compacted, { sessionID: input.sessionID })
    return "continue"
  }

  export const create = fn(
    z.object({
      sessionID: Identifier.schema("session"),
      agent: z.string(),
      model: z.object({
        providerID: z.string(),
        modelID: z.string(),
      }),
      auto: z.boolean(),
    }),
    async (input) => {
      const { config, mode, threshold } = await getCompactionSettings()
      if (mode === "sliding") {
        if (config.compaction?.auto === false && input.auto) return
        const model = await Provider.getModel(input.model.providerID, input.model.modelID)
        const messages = await Session.messages({ sessionID: input.sessionID })
        const result = await slideSession({
          sessionID: input.sessionID,
          messages,
          model,
          threshold,
        })
        if (!input.auto) {
          await setManualSlides(input.sessionID, result.removed)
        }
        log.info("sliding", {
          removed: result.removed,
          manual: result.manual,
          automatic: result.automatic,
          remainingEstimate: result.remainingEstimate,
          target: result.target,
        })
        Bus.publish(Event.Compacted, { sessionID: input.sessionID })
        return
      }
      const msg = await Session.updateMessage({
        id: Identifier.ascending("message"),
        role: "user",
        model: input.model,
        sessionID: input.sessionID,
        agent: input.agent,
        time: {
          created: Date.now(),
        },
      })
      await Session.updatePart({
        id: Identifier.ascending("part"),
        messageID: msg.id,
        sessionID: msg.sessionID,
        type: "compaction",
        auto: input.auto,
      })
    },
  )
}
