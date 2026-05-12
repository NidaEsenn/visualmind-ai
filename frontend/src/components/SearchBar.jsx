import { useEffect, useRef, useState, useCallback } from 'react'
import { Search, X, Loader2, Clock, ImageIcon, SlidersHorizontal } from 'lucide-react'
import clsx from 'clsx'

export default function SearchBar({
  query,
  onSearch,
  isLoading,
  queryTime,
  resultCount,
  searchMode,
  onModeChange,
  alpha,
  onAlphaChange
}) {
  const [inputValue, setInputValue] = useState(query)
  const [showAlphaPanel, setShowAlphaPanel] = useState(false)
  const inputRef = useRef(null)
  const debounceRef = useRef(null)

  // Sync external query changes (e.g. clicking example queries)
  useEffect(() => {
    setInputValue(query)
  }, [query])

  // Keyboard shortcut: "/" to focus
  useEffect(() => {
    const handler = (e) => {
      if (
        e.key === '/' &&
        document.activeElement !== inputRef.current &&
        document.activeElement.tagName !== 'INPUT' &&
        document.activeElement.tagName !== 'TEXTAREA'
      ) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [])

  const triggerSearch = useCallback(
    (value) => {
      if (value.trim()) {
        onSearch(value.trim())
      }
    },
    [onSearch]
  )

  const handleChange = (e) => {
    const val = e.target.value
    setInputValue(val)
    clearTimeout(debounceRef.current)
    if (val.trim()) {
      debounceRef.current = setTimeout(() => triggerSearch(val), 300)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      clearTimeout(debounceRef.current)
      triggerSearch(inputValue)
    }
    if (e.key === 'Escape') {
      inputRef.current?.blur()
    }
  }

  const handleClear = () => {
    setInputValue('')
    onSearch('')
    inputRef.current?.focus()
  }

  const hasResults = resultCount !== null && resultCount !== undefined

  return (
    <div className="w-full space-y-3">
      {/* Search input row */}
      <div className="flex items-center gap-3">
        {/* Mode toggle */}
        <div className="flex items-center bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-1 shrink-0">
          <button
            onClick={() => onModeChange('text')}
            className={clsx(
              'px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150',
              searchMode === 'text'
                ? 'bg-brand-600 text-white shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            )}
          >
            Text
          </button>
          <button
            onClick={() => onModeChange('hybrid')}
            className={clsx(
              'px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150',
              searchMode === 'hybrid'
                ? 'bg-brand-600 text-white shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            )}
          >
            Hybrid
          </button>
        </div>

        {/* Main input */}
        <div className="relative flex-1">
          {/* Left icon */}
          <div className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none">
            {isLoading ? (
              <Loader2 size={18} className="text-brand-500 animate-spin" />
            ) : (
              <Search size={18} className="text-zinc-500" />
            )}
          </div>

          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Search UI designs, components, color moods..."
            className={clsx(
              'w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl',
              'pl-11 pr-20 py-3.5 text-sm text-white placeholder-zinc-500',
              'transition-all duration-150',
              'focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500/40'
            )}
          />

          {/* Right side actions */}
          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
            {inputValue && (
              <button
                onClick={handleClear}
                className="p-1.5 text-zinc-500 hover:text-zinc-200 rounded-lg hover:bg-white/5 transition-colors"
                title="Clear"
              >
                <X size={14} />
              </button>
            )}
            {/* Shortcut hint */}
            {!inputValue && (
              <kbd className="hidden sm:flex items-center justify-center w-6 h-6 text-[10px] font-mono text-zinc-600 bg-[#252525] border border-[#2a2a2a] rounded">
                /
              </kbd>
            )}
          </div>
        </div>

        {/* Alpha/settings toggle for hybrid */}
        {searchMode === 'hybrid' && (
          <button
            onClick={() => setShowAlphaPanel((v) => !v)}
            className={clsx(
              'p-2.5 rounded-lg border transition-all duration-150 shrink-0',
              showAlphaPanel
                ? 'bg-brand-600/20 border-brand-500/50 text-brand-400'
                : 'bg-[#1a1a1a] border-[#2a2a2a] text-zinc-400 hover:text-zinc-200 hover:border-[#3a3a3a]'
            )}
            title="Adjust alpha"
          >
            <SlidersHorizontal size={16} />
          </button>
        )}
      </div>

      {/* Hybrid alpha slider */}
      {searchMode === 'hybrid' && showAlphaPanel && (
        <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl px-5 py-4 animate-slideUp">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-xs text-zinc-400 font-medium">Search balance</span>
            <span className="text-xs font-mono text-brand-400 bg-brand-500/10 px-2 py-0.5 rounded">
              {alpha.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={alpha}
            onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
            className="w-full accent-brand-500"
          />
          <div className="flex justify-between mt-1.5">
            <span className="text-[10px] text-zinc-600">Dense (semantic)</span>
            <span className="text-[10px] text-zinc-600">BM25 (keyword)</span>
          </div>
        </div>
      )}

      {/* Results meta row */}
      {hasResults && (
        <div className="flex items-center gap-3 text-xs text-zinc-500 px-1 animate-slideUp">
          <span className="flex items-center gap-1.5">
            <ImageIcon size={12} />
            <span>
              <span className="text-zinc-300 font-medium">{resultCount}</span>{' '}
              {resultCount === 1 ? 'result' : 'results'}
            </span>
          </span>
          {queryTime !== null && queryTime !== undefined && (
            <>
              <span className="text-zinc-700">·</span>
              <span className="flex items-center gap-1.5">
                <Clock size={12} />
                <span>{queryTime}ms</span>
              </span>
            </>
          )}
        </div>
      )}
    </div>
  )
}
