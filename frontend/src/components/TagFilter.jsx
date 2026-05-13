import { useState } from 'react'
import { ChevronDown, ChevronRight, X } from 'lucide-react'
import clsx from 'clsx'

const FILTER_SECTIONS = [
  {
    key: 'layout_type',
    label: 'Layout',
    options: ['dashboard', 'landing', 'onboarding', 'form', 'card', 'other']
  },
  {
    key: 'color_mood',
    label: 'Color Mood',
    options: ['minimal', 'dark', 'colorful', 'warm', 'corporate', 'playful']
  },
  {
    key: 'industry',
    label: 'Industry',
    options: ['fintech', 'saas', 'ecommerce', 'health', 'education', 'social', 'travel', 'media', 'productivity', 'crypto', 'other']
  },
  {
    key: 'complexity',
    label: 'Complexity',
    options: ['low', 'medium', 'high']
  }
]

const COLOR_MOOD_DOTS = {
  minimal: '#a1a1aa',
  dark: '#18181b',
  colorful: '#818cf8',
  warm: '#fb923c',
  corporate: '#3b82f6',
  playful: '#f472b6'
}

function FilterSection({ section, activeValues, onToggle }) {
  const [open, setOpen] = useState(true)

  return (
    <div className="border-b border-[#2a2a2a] last:border-b-0">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full py-3 px-1 text-left group"
      >
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider group-hover:text-zinc-200 transition-colors">
          {section.label}
        </span>
        <span className="text-zinc-600 group-hover:text-zinc-400 transition-colors">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
      </button>

      {open && (
        <div className="flex flex-wrap gap-1.5 pb-3 px-1">
          {section.options.map((opt) => {
            const isActive = activeValues.includes(opt)
            const dot = section.key === 'color_mood' ? COLOR_MOOD_DOTS[opt] : null

            return (
              <button
                key={opt}
                onClick={() => onToggle(section.key, opt)}
                className={clsx(
                  'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
                  'border transition-all duration-150',
                  isActive
                    ? 'bg-brand-600/20 border-brand-500/60 text-brand-300'
                    : 'bg-[#252525] border-[#2a2a2a] text-zinc-400 hover:text-zinc-200 hover:border-[#3a3a3a]'
                )}
              >
                {dot && (
                  <span
                    className="w-2 h-2 rounded-full shrink-0 border border-white/10"
                    style={{ background: dot }}
                  />
                )}
                {opt}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default function TagFilter({ activeFilters, onFilterChange, visible }) {
  const hasActive = Object.values(activeFilters).some((v) => v.length > 0)

  const handleToggle = (key, value) => {
    const current = activeFilters[key] || []
    const updated = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value]
    onFilterChange({ ...activeFilters, [key]: updated })
  }

  const handleClearAll = () => {
    const cleared = Object.fromEntries(Object.keys(activeFilters).map((k) => [k, []]))
    onFilterChange(cleared)
  }

  if (!visible) return null

  return (
    <aside className="w-52 shrink-0 bg-[#141414] border border-[#2a2a2a] rounded-xl p-4 self-start sticky top-24 animate-slideUp">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-bold text-zinc-300 uppercase tracking-widest">Filters</h2>
        {hasActive && (
          <button
            onClick={handleClearAll}
            className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-red-400 transition-colors"
          >
            <X size={10} />
            Clear all
          </button>
        )}
      </div>

      {/* Active filter count */}
      {hasActive && (
        <div className="mb-3 px-2 py-1.5 bg-brand-600/10 border border-brand-500/20 rounded-lg">
          <p className="text-[10px] text-brand-400">
            {Object.values(activeFilters).flat().length} filter
            {Object.values(activeFilters).flat().length !== 1 ? 's' : ''} active
          </p>
        </div>
      )}

      {/* Sections */}
      <div className="space-y-0">
        {FILTER_SECTIONS.map((section) => (
          <FilterSection
            key={section.key}
            section={section}
            activeValues={activeFilters[section.key] || []}
            onToggle={handleToggle}
          />
        ))}
      </div>
    </aside>
  )
}
