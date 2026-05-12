import { useEffect, useState } from 'react'
import { X, Sparkles, Trash2, Calendar, Hash, ExternalLink, AlertTriangle } from 'lucide-react'
import clsx from 'clsx'

const TAG_LABEL_MAP = {
  ui_pattern: 'UI Patterns',
  color_mood: 'Color Mood',
  industry: 'Industry',
  layout_type: 'Layout',
  complexity: 'Complexity',
  primary_colors: 'Colors',
  has_illustration: 'Illustration',
  has_chart: 'Has Chart',
  has_data_viz: 'Data Viz'
}

function TagPill({ value, accent = false }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize border',
        accent
          ? 'bg-brand-600/20 border-brand-500/40 text-brand-300'
          : 'bg-[#252525] border-[#333] text-zinc-400'
      )}
    >
      {String(value)}
    </span>
  )
}

function TagGroup({ label, value }) {
  if (value === null || value === undefined || value === '') return null
  const values = Array.isArray(value) ? value : [value]
  if (values.length === 0) return null

  return (
    <div className="space-y-1.5">
      <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {values.map((v, i) => (
          <TagPill key={i} value={v} accent={label === 'UI Patterns' || label === 'Industry'} />
        ))}
      </div>
    </div>
  )
}

export default function ImageModal({ image, onClose, onSimilarSearch, onDelete }) {
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  // Escape key to close
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handler)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', handler)
      document.body.style.overflow = ''
    }
  }, [onClose])

  if (!image) return null

  const imageId = image.id || image.image_id || ''
  const tags = image?.tags || image?.metadata?.tags || {}
  const score = image?.score ?? image?.similarity_score ?? null
  const createdAt = image?.created_at || image?.metadata?.created_at
  const filename = image?.filename || image?.metadata?.filename || 'Untitled'

  const imageUrl =
    image?.url ||
    image?.thumbnail_url ||
    (imageId ? `http://localhost:8000/images/${imageId}/file` : null)

  const handleDelete = async () => {
    if (!confirmDelete) {
      setConfirmDelete(true)
      return
    }
    setIsDeleting(true)
    try {
      await onDelete(imageId)
      onClose()
    } catch {
      setIsDeleting(false)
      setConfirmDelete(false)
    }
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return null
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    } catch {
      return dateStr
    }
  }

  // Flatten tags for display
  const tagEntries = Object.entries(tags).filter(
    ([, v]) => v !== null && v !== undefined && v !== '' && !(Array.isArray(v) && v.length === 0)
  )

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fadeIn"
      onClick={onClose}
    >
      <div
        className="relative flex flex-col lg:flex-row w-full max-w-6xl max-h-[90vh] bg-[#141414] border border-[#2a2a2a] rounded-2xl overflow-hidden shadow-2xl animate-fadeIn"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 p-2 bg-black/40 hover:bg-black/70 backdrop-blur-sm border border-white/10 text-zinc-400 hover:text-white rounded-lg transition-all"
        >
          <X size={16} />
        </button>

        {/* Image panel */}
        <div className="flex-1 bg-[#0f0f0f] flex items-center justify-center min-h-[300px] max-h-[90vh] overflow-hidden">
          {imageUrl ? (
            <img
              src={imageUrl}
              alt={filename}
              className="max-w-full max-h-full object-contain"
              style={{ maxHeight: '70vh' }}
            />
          ) : (
            <div className="flex flex-col items-center gap-3 text-zinc-700">
              <ExternalLink size={40} />
              <p className="text-sm">No preview available</p>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="w-full lg:w-72 xl:w-80 shrink-0 border-t lg:border-t-0 lg:border-l border-[#2a2a2a] overflow-y-auto">
          <div className="p-5 space-y-5">
            {/* Filename */}
            <div>
              <h2 className="text-sm font-semibold text-white truncate" title={filename}>
                {filename}
              </h2>
              {imageId && (
                <p className="text-[10px] font-mono text-zinc-600 mt-0.5 truncate" title={imageId}>
                  <Hash size={9} className="inline mr-0.5 -mt-0.5" />
                  {imageId.slice(0, 20)}…
                </p>
              )}
            </div>

            {/* Meta row */}
            <div className="space-y-2">
              {score !== null && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-zinc-500">Similarity</span>
                  <span className="flex items-center gap-1 text-xs font-bold text-brand-400 bg-brand-600/10 px-2 py-0.5 rounded-full">
                    <Sparkles size={10} />
                    {Math.round(score * 100)}%
                  </span>
                </div>
              )}
              {createdAt && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-zinc-500 flex items-center gap-1.5">
                    <Calendar size={11} />
                    Added
                  </span>
                  <span className="text-xs text-zinc-400">{formatDate(createdAt)}</span>
                </div>
              )}
            </div>

            {/* Divider */}
            <div className="border-t border-[#2a2a2a]" />

            {/* Tags */}
            {tagEntries.length > 0 ? (
              <div className="space-y-4">
                <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                  Metadata
                </p>
                {tagEntries.map(([key, value]) => (
                  <TagGroup
                    key={key}
                    label={TAG_LABEL_MAP[key] || key.replace(/_/g, ' ')}
                    value={value}
                  />
                ))}
              </div>
            ) : (
              <div className="text-xs text-zinc-600 italic">No metadata available</div>
            )}

            {/* Divider */}
            <div className="border-t border-[#2a2a2a]" />

            {/* Actions */}
            <div className="space-y-2 pb-1">
              <button
                onClick={() => {
                  onSimilarSearch(image)
                  onClose()
                }}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium rounded-lg transition-colors"
              >
                <Sparkles size={14} />
                Find Similar Images
              </button>

              {confirmDelete ? (
                <div className="space-y-2">
                  <div className="flex items-center gap-2 p-2.5 bg-red-500/10 border border-red-500/20 rounded-lg">
                    <AlertTriangle size={13} className="text-red-400 shrink-0" />
                    <p className="text-[11px] text-red-400">This cannot be undone.</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setConfirmDelete(false)}
                      className="flex-1 px-3 py-2 bg-[#252525] hover:bg-[#2a2a2a] text-zinc-300 text-xs font-medium rounded-lg transition-colors border border-[#333]"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleDelete}
                      disabled={isDeleting}
                      className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white text-xs font-medium rounded-lg transition-colors"
                    >
                      {isDeleting ? (
                        <span>Deleting...</span>
                      ) : (
                        <>
                          <Trash2 size={12} />
                          Confirm
                        </>
                      )}
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  onClick={handleDelete}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-transparent hover:bg-red-500/10 border border-[#2a2a2a] hover:border-red-500/40 text-zinc-500 hover:text-red-400 text-sm font-medium rounded-lg transition-all"
                >
                  <Trash2 size={14} />
                  Delete Image
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
