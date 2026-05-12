import Masonry from 'react-masonry-css'
import { Search, Layers } from 'lucide-react'
import ImageCard from './ImageCard'

const BREAKPOINTS = {
  default: 5,
  1536: 5,
  1280: 4,
  1024: 3,
  768: 2,
  640: 1
}

function SkeletonCard({ height = 200 }) {
  return (
    <div
      className="rounded-xl bg-[#1a1a1a] border border-[#2a2a2a] animate-pulse overflow-hidden"
      style={{ height }}
    >
      <div className="w-full h-full bg-gradient-to-b from-[#222] to-[#1a1a1a]" />
    </div>
  )
}

// Variable heights for skeletons to mimic masonry
const SKELETON_HEIGHTS = [220, 160, 280, 200, 240, 180, 300, 220, 170, 250, 190, 230]

export default function ImageGrid({ results, isLoading, onImageClick, onSimilarSearch, isBrowseMode }) {
  if (isLoading) {
    return (
      <Masonry
        breakpointCols={BREAKPOINTS}
        className="masonry-grid"
        columnClassName="masonry-grid-column"
      >
        {SKELETON_HEIGHTS.map((h, i) => (
          <SkeletonCard key={i} height={h} />
        ))}
      </Masonry>
    )
  }

  if (!results || results.length === 0) {
    const isEmpty = results !== null && results.length === 0
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <div className="w-16 h-16 rounded-2xl bg-[#1a1a1a] border border-[#2a2a2a] flex items-center justify-center mb-4">
          {isBrowseMode && !isEmpty ? (
            <Layers size={28} className="text-zinc-700" />
          ) : (
            <Search size={28} className="text-zinc-700" />
          )}
        </div>
        <p className="text-zinc-400 font-medium mb-1">
          {isBrowseMode && !isEmpty
            ? 'Upload images to get started'
            : isBrowseMode && isEmpty
            ? 'No images match these filters'
            : 'No results found'}
        </p>
        <p className="text-zinc-600 text-sm">
          {isBrowseMode && !isEmpty
            ? 'Drag and drop images or click Upload'
            : 'Try adjusting your filters or search query'}
        </p>
      </div>
    )
  }

  return (
    <Masonry
      breakpointCols={BREAKPOINTS}
      className="masonry-grid"
      columnClassName="masonry-grid-column"
    >
      {results.map((image) => (
        <ImageCard
          key={image.id || image.image_id}
          image={image}
          onClick={onImageClick}
          onSimilarSearch={onSimilarSearch}
        />
      ))}
    </Masonry>
  )
}
