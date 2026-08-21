import { useEffect, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { KEYED_BRAINS, type Brain } from '../brains'

/** Post-onboarding key management — the onboarding-only input was a gap.
 *
 * One field per backend the pipeline knows about. Endpoints with no preset
 * go through the pipeline's `custom` mode instead, configured by env var:
 * its base URL decides where every transcript is POSTed, so it is
 * deliberately not something this window can rewrite. */

interface Props {
  onClose: () => void
}

interface SetupState {
  keys?: Record<string, boolean>
}

function SecretField({
  field,
  placeholder,
  saved,
  onSaved
}: {
  field: string
  placeholder: string
  saved: boolean
  onSaved: () => void
}) {
  const [key, setKey] = useState('')
  const [justSaved, setJustSaved] = useState(false)

  async function save() {
    if (!key.trim()) return
    await invoke('save_provider_key', { field, key })
    setKey('')
    setJustSaved(true)
    onSaved()
  }

  return (
    <div className="ig-form">
      <input
        placeholder={placeholder}
        type="password"
        value={key}
        onChange={(e) => setKey(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && save()}
        className="mono"
      />
      <button className="btn-secondary" onClick={save} disabled={!key.trim()}>
        {justSaved ? 'saved ✓' : saved ? 'replace' : 'save'}
      </button>
    </div>
  )
}

function BrainRow({
  brain,
  saved,
  onSaved
}: {
  brain: Brain
  saved: boolean
  onSaved: () => void
}) {
  return (
    <div style={{ marginTop: 18 }}>
      <p className="audit-label">
        {brain.label.toUpperCase()}
        {saved && <span className="mono"> · key saved ✓</span>}
        {!brain.vision && <span className="mono"> · text only</span>}
      </p>
      <SecretField
        field={brain.secret as string}
        placeholder={`${brain.placeholder}  (${brain.signup})`}
        saved={saved}
        onSaved={onSaved}
      />
      <p className="ig-message mono">{brain.note}</p>
    </div>
  )
}

export default function KeyModal({ onClose }: Props) {
  const [keys, setKeys] = useState<Record<string, boolean>>({})

  function refresh() {
    invoke<SetupState>('get_setup_state').then((s) => setKeys(s.keys ?? {}))
  }

  useEffect(refresh, [])

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header className="modal-head">
          <p className="audit-kicker">THE BRAIN</p>
          <button className="btn-ghost" onClick={onClose}>close ✕</button>
        </header>
        <p className="ig-intro">
          Pick a backend per run. Keys live in{' '}
          <span className="mono">~/.publikclip/secrets.json</span> (owner-only on macOS and
          Linux) and each one only ever talks to its own provider. Ollama needs no key and
          nothing leaves the machine.
        </p>

        {KEYED_BRAINS.map((brain) => (
          <BrainRow
            key={brain.mode}
            brain={brain}
            saved={!!keys[brain.secret as string]}
            onSaved={refresh}
          />
        ))}

        <div style={{ marginTop: 26 }}>
          <p className="audit-label">
            PEXELS (STOCK VISUALS)
            {keys['pexels_api_key'] && <span className="mono"> · key saved ✓</span>}
          </p>
          <SecretField
            field="pexels_api_key"
            placeholder="Pexels API key (free — pexels.com/api)"
            saved={!!keys['pexels_api_key']}
            onSaved={refresh}
          />
        </div>

        <p className="ig-message mono" style={{ marginTop: 22 }}>
          Applies to new runs; a job mid-flight keeps the brain it started with. For an
          endpoint not listed here, set PUBLIKCLIP_LLM_BASE_URL and PUBLIKCLIP_LLM_MODEL,
          then run with <span className="mono">--llm custom</span>.
        </p>
      </div>
    </div>
  )
}
