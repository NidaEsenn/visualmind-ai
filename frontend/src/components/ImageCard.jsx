import { useRef } from 'react'
import { Layers, Sparkles } from 'lucide-react'
import clsx from 'clsx'

const COLOR_MOOD_MAP = {
  minimal: '#a1a1aa',
  dark: '#3f3f46',
  colorful: '#818cf8',
  warm: '#fb923c',
  corporate: '#3b82f6',
  playful: '#f472b6'
}

function ScoreBadge({ score }) {
  if (score === null || score === undefined) return null
  const pct = Math.round(score * 100)
  return (
    <span className="flex items-center gap-1 px-2 py-0.5 bg-brand-600/90 backdrop-blur-sm text-white text-[10px] font-bold rounded-full shadow-lg">
      <Sparkles size={9} />
      {pct}%
    </span>
  )
}

export default function ImageCard({ image, onClick, onSimilarSearch }) {
  const imgRef = useRef(null)

  const handleImgLoad = () => {
    if (imgRef.current) imgRef.current.style.opacity = '1'
  }

  const tags = image?.tags || image?.metadata?.tags || {}
  const uiPatterns = tags.ui_patterns || []
  const colorMood = tags.color_mood || image?.color_mood
  const industry = tags.industry || image?.industry
  const score = image?.score ?? image?.similarity_score ?? null

  const imageUrl =
    image?.thumbnail_url ||
    image?.url ||
    (image?.id ? `http://localhost:8000/images/${image.id}/thumbnail` : null)

  const moodColor = COLOR_MOOD_MAP[colorMood] || '#6366f1'

  return (
    <div
      className={clsx(
        'group relative bg-[#1a1a1a] rounded-xl overflow-hidden cursor-pointer',
        'border border-[#2a2a2a] hover:border-[#3a3a3a]',
        'transition-all duration-200 ease-out',
        'hover:scale-[1.02] hover:shadow-xl hover:shadow-black/40'
      )}
      onClick={() => onClick(image)}
    >
      {/* Thumbnail */}
      <div className="relative overflow-hidden bg-[#141414]">
        {imageUrl ? (
          <img
            ref={imgRef}
            src={imageUrl}
            alt={image?.filename || 'UI design'}
            loading="lazy"
            onLoad={handleImgLoad}
            className="w-full h-auto block object-cover"
            style={{ opacity: 0, transition: 'opacity 0.3s ease' }}
            onError={(e) => {
              e.target.style.display = 'none'
            }}
          />
        ) : (
          <div className="w-full h-40 flex items-center justify-center">
            <Layers size={32} className="text-zinc-700" />
          </div>
        )}

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/50 transition-all duration-200 flex items-start justify-between p-2.5 pointer-events-none group-hover:pointer-events-auto">
          {/* Score badge - top right */}
          <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <ScoreBadge score={score} />
          </div>
        </div>

        {/* Similar button - centered on hover */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none group-hover:pointer-events-auto">
          <button
            onClick={(e) => {
              e.stopPropagation()
              onSimilarSearch(image)
            }}
            className="px-3 py-1.5 bg-white/10 hover:bg-brand-600 backdrop-blur-sm border border-white/20 hover:border-brand-500 text-white text-xs font-medium rounded-lg transition-all duration-150 flex items-center gap-1.5"
          >
            <Sparkles size={12} />
            Similar
          </button>
        </div>

        {/* Industry badge - top left on hover */}
        {industry && (
          <div className="absolute top-2 left-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <span className="px-2 py-0.5 bg-black/60 backdrop-blur-sm text-zinc-300 text-[10px] font-medium rounded-full border border-white/10 capitalize">
              {industry}
            </span>
          </div>
        )}

        {/* Color mood dot - bottom right corner */}
        {colorMood && (
          <div
            className="absolute bottom-2 right-2 w-2.5 h-2.5 rounded-full border border-white/20 shadow-md opacity-70 group-hover:opacity-100 transition-opacity"
            style={{ background: moodColor }}
            title={colorMood}
          />
        )}
      </div>

      {/* Bottom strip */}
      {uiPatterns.length > 0 && (
        <div className="px-2.5 py-2 flex items-center gap-1 flex-wrap">
          {uiPatterns.slice(0, 2).map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 bg-[#252525] border border-[#333] text-zinc-500 text-[10px] rounded-full capitalize"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
