/** The brains the pipeline can score with.
 *
 * Mirrors PROVIDERS in pipeline/publikclip_pipeline/scoring/llm.py — the
 * `mode` strings are what get passed to `--llm`, and `secret` is the field
 * in ~/.publikclip/secrets.json. Keep the two lists in step; the pipeline is
 * the authority and will reject a mode it does not know.
 *
 * Endpoints with no preset (self-hosted vLLM, Together, DeepSeek, LM Studio)
 * run through the pipeline's `custom` mode, configured with
 * PUBLIKCLIP_LLM_BASE_URL / PUBLIKCLIP_LLM_MODEL. That one is deliberately
 * not settable from here: the base URL decides where every transcript is
 * sent, so it is not a thing the UI should be able to rewrite.
 */

export interface Brain {
  mode: string
  label: string
  /** secrets.json field, or null when the backend needs no key. */
  secret: string | null
  placeholder: string
  signup: string
  /** One line the settings modal shows under the field. */
  note: string
  /** Whether it can look at video frames (the T2 visual pass). */
  vision: boolean
}

export const BRAINS: Brain[] = [
  {
    mode: 'gemini',
    label: 'gemini',
    secret: 'gemini_api_key',
    placeholder: 'AIza…',
    signup: 'aistudio.google.com → Get API key',
    note: 'Full-quality scoring, and the backend the rubric was tuned against. Free tier exists, but Google uses free-tier content to improve its products.',
    vision: true
  },
  {
    mode: 'nvidia',
    label: 'nvidia',
    secret: 'nvidia_api_key',
    placeholder: 'nvapi-…',
    signup: 'build.nvidia.com — free, no card',
    note: 'Open models on NVIDIA-hosted GPUs. Free tier needs no credit card. Scores are labeled third-party.',
    vision: true
  },
  {
    mode: 'openrouter',
    label: 'openrouter',
    secret: 'openrouter_api_key',
    placeholder: 'sk-or-…',
    signup: 'openrouter.ai/keys',
    note: 'One key, most models. Pay per token at each model’s own rate.',
    vision: true
  },
  {
    mode: 'openai',
    label: 'openai',
    secret: 'openai_api_key',
    placeholder: 'sk-…',
    signup: 'platform.openai.com/api-keys',
    note: 'Paid only — no free tier.',
    vision: true
  },
  {
    mode: 'groq',
    label: 'groq',
    secret: 'groq_api_key',
    placeholder: 'gsk_…',
    signup: 'console.groq.com/keys',
    note: 'Very fast, text only — the visual pass is skipped and recorded as a missing signal.',
    vision: false
  },
  {
    mode: 'ollama',
    label: 'ollama',
    secret: null,
    placeholder: '',
    signup: 'ollama.com',
    note: 'Fully local, zero cost, nothing leaves the machine. Scores are labeled local estimate — small models judge humor less reliably.',
    vision: false
  }
]

export const KEYED_BRAINS = BRAINS.filter((b) => b.secret !== null)

/** Modes the picker offers: everything keyed, plus the keyless local one. */
export function brainFor(mode: string): Brain | undefined {
  return BRAINS.find((b) => b.mode === mode)
}
