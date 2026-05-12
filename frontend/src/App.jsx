import { useCallback, useEffect, useRef, useState } from 'react'
import { SlidersHorizontal, Brain, Database, AlertCircle, X, Search, LayoutGrid } from 'lucide-react'
import clsx from 'clsx'

import SearchBar from './components/SearchBar'
import TagFilter from './components/TagFilter'
import ImageGrid from './components/ImageGrid'
import ImageModal from './components/ImageModal'
import UploadZone from './components/UploadZone'

import {
  searchText,
  searchHybrid,
  searchSimilar,
  uploadImages,
  deleteImage,
  getStats,
  listImages,
} from './api'

const EXAMPLE_QUERIES = [
  'fintech onboarding screens',
  'dark minimal dashboards with sidebar nav',
  'warm e-commerce product cards',
  'brutalist landing pages with large typography',
  'SaaS pricing tables',
]

const INITIAL_FILTERS = {
  layout_type: [],
  color_mood: [],
  industry: [],
  complexity: [],
}

function ErrorBanner({ message, onDismiss }) {
  if (!message) return null
  return (
    <div className="flex items-center gap-3 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-xl text-sm text-red-400">
      <AlertCircle size={15} className="shrink-0" />
      <span className="flex-1">{message}</span>
      <button onClick={onDismiss} className="p-1 hover:bg-red-500/20 rounded-lg transition-colors">
        <X size={13} />
      </button>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)
  const [browseImages, setBrowseImages] = useState(null)
  const [isSearchLoading, setIsSearchLoading] = useState(false)
  const [isBrowseLoading, setIsBrowseLoading] = useState(false)
  const [searchMode, setSearchMode] = useState('text')
  const [activeFilters, setActiveFilters] = useState(INITIAL_FILTERS)
  const [selectedImage, setSelectedImage] = useState(null)
  const [showFilters, setShowFilters] = useState(true)
  const [imageCount, setImageCount] = useState(null)
  const [queryTime, setQueryTime] = useState(null)
  const [alpha, setAlpha] = useState(0.7)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null)
  const [error, setError] = useState(null)

  const lastQueryRef = useRef('')
  const hasQuery = query.trim().length > 0

  // -------------------------------------------------------------------------
  // Browse / gallery fetch
  // -------------------------------------------------------------------------
  const fetchBrowse = useCallback(async (filters = activeFilters) => {
    setIsBrowseLoading(true)
    try {
      const data = await listImages(filters)
      setBrowseImages(data?.images ?? [])
      setImageCount(data?.total ?? 0)
    } catch {
      setBrowseImages([])
    } finally {
      setIsBrowseLoading(false)
    }
  }, [activeFilters])

  useEffect(() => {
    fetchBrowse()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // -------------------------------------------------------------------------
  // Search
  // -------------------------------------------------------------------------
  const buildFilters = useCallback(() => {
    const active = {}
    Object.entries(activeFilters).forEach(([key, values]) => {
      if (values.length > 0) active[key] = values
    })
    return Object.keys(active).length > 0 ? active : null
  }, [activeFilters])

  const runSearch = useCallback(
    async (q, mode = searchMode, currentAlpha = alpha) => {
      if (!q?.trim()) {
        setSearchResults(null)
        setQueryTime(null)
        lastQueryRef.current = ''
        return
      }
      lastQueryRef.current = q
      setIsSearchLoading(true)
      setError(null)
      const t0 = performance.now()
      try {
        let data
        const filters = buildFilters()
        if (mode === 'hybrid') {
          data = await searchHybrid(q.trim(), 20, currentAlpha)
        } else {
          data = await searchText(q.trim(), 20, filters)
        }
        const items = data?.results ?? data?.images ?? (Array.isArray(data) ? data : [])
        setSearchResults(items)
        setQueryTime(Math.round(performance.now() - t0))
      } catch (err) {
        setError(err.message || 'Search failed. Check that the backend is running.')
        setSearchResults([])
        setQueryTime(null)
      } finally {
        setIsSearchLoading(false)
      }
    },
    [searchMode, alpha, buildFilters]
  )

  // Re-run when filters change
  useEffect(() => {
    if (lastQueryRef.current) {
      runSearch(lastQueryRef.current)
    } else {
      fetchBrowse(activeFilters)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeFilters])

  const handleSearch = useCallback((q) => {
    setQuery(q)
    runSearch(q)
  }, [runSearch])

  const clearSearch = useCallback(() => {
    setQuery('')
    setSearchResults(null)
    setQueryTime(null)
    lastQueryRef.current = ''
  }, [])

  const handleModeChange = useCallback((mode) => {
    setSearchMode(mode)
    if (lastQueryRef.current) runSearch(lastQueryRef.current, mode, alpha)
  }, [runSearch, alpha])

  const handleAlphaChange = useCallback((val) => {
    setAlpha(val)
    if (lastQueryRef.current) runSearch(lastQueryRef.current, searchMode, val)
  }, [runSearch, searchMode])

  const handleSimilarSearch = useCallback(async (image) => {
    const id = image?.id || image?.image_id
    if (!id) return
    setIsSearchLoading(true)
    setError(null)
    setQuery('')
    lastQueryRef.current = ''
    const t0 = performance.now()
    try {
      const data = await searchSimilar(id, 20)
      const items = data?.results ?? data?.images ?? (Array.isArray(data) ? data : [])
      setSearchResults(items)
      setQueryTime(Math.round(performance.now() - t0))
    } catch (err) {
      setError(err.message || 'Similar search failed.')
      setSearchResults([])
    } finally {
      setIsSearchLoading(false)
    }
  }, [])

  // -------------------------------------------------------------------------
  // Upload
  // -------------------------------------------------------------------------
  const handleUpload = useCallback(async (files) => {
    setIsUploading(true)
    setUploadProgress({ processed: 0, total: files.length, done: false })
    setError(null)
    try {
      const data = await uploadImages(files)
      const uploaded = data?.image_ids?.length ?? 0
      const duplicates = data?.duplicates?.length ?? 0
      setUploadProgress({ processed: files.length, total: files.length, uploaded, duplicates, done: true })
      // Refresh gallery
      await fetchBrowse()
      setTimeout(() => setUploadProgress(null), 4000)
    } catch (err) {
      setError(err.message || 'Upload failed.')
      setUploadProgress(null)
    } finally {
      setIsUploading(false)
    }
  }, [fetchBrowse])

  const handleDelete = useCallback(async (imageId) => {
    await deleteImage(imageId)
    setSearchResults((prev) => prev ? prev.filter((img) => (img.id || img.image_id) !== imageId) : prev)
    setBrowseImages((prev) => prev ? prev.filter((img) => img.id !== imageId) : prev)
    getStats().then((s) => setImageCount(s?.total_images ?? s?.count ?? s?.image_count ?? null)).catch(() => {})
  }, [])

  // -------------------------------------------------------------------------
  // Derived display state
  // -------------------------------------------------------------------------
  const isSimilarMode = !hasQuery && searchResults !== null
  const displayImages = hasQuery || isSimilarMode ? searchResults : browseImages
  const isGridLoading = hasQuery || isSimilarMode ? isSearchLoading : isBrowseLoading
  const totalImages = browseImages?.length ?? imageCount ?? 0

  return (
    <div className="min-h-screen bg-[#0f0f0f]">
      {/* ===================== NAVBAR ===================== */}
      <header className="sticky top-0 z-30 bg-[#0f0f0f]/90 backdrop-blur-md border-b border-[#1e1e1e]">
        <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-4">
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center shadow-lg shadow-brand-600/30">
              <Brain size={15} className="text-white" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-brand-400 to-indigo-300 bg-clip-text text-transparent hidden sm:block">
              VisualMind AI
            </span>
          </div>

          <div className="flex-1" />

          {imageCount !== null && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[#1a1a1a] border border-[#2a2a2a] rounded-full text-xs text-zinc-400 shrink-0">
              <Database size={11} className="text-brand-400" />
              <span className="font-medium text-zinc-300">{imageCount.toLocaleString()}</span>
              <span className="hidden sm:inline">images</span>
            </div>
          )}

          <button
            onClick={() => setShowFilters((v) => !v)}
            className={clsx(
              'flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium transition-all duration-150',
              showFilters
                ? 'bg-brand-600/20 border-brand-500/40 text-brand-300'
                : 'bg-[#1a1a1a] border-[#2a2a2a] text-zinc-400 hover:text-zinc-200 hover:border-[#3a3a3a]'
            )}
          >
            <SlidersHorizontal size={14} />
            <span className="hidden sm:inline">Filters</span>
          </button>

          <UploadZone
            onUpload={handleUpload}
            isUploading={isUploading}
            uploadProgress={uploadProgress}
          />
        </div>
      </header>

      {/* ===================== MAIN ===================== */}
      <main className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-6">
        <div className="mb-6">
          <SearchBar
            query={query}
            onSearch={handleSearch}
            isLoading={isSearchLoading}
            queryTime={queryTime}
            resultCount={hasQuery ? searchResults?.length ?? null : null}
            searchMode={searchMode}
            onModeChange={handleModeChange}
            alpha={alpha}
            onAlphaChange={handleAlphaChange}
          />
        </div>

        {error && (
          <div className="mb-4">
            <ErrorBanner message={error} onDismiss={() => setError(null)} />
          </div>
        )}

        <div className="flex gap-5 items-start">
          <TagFilter
            activeFilters={activeFilters}
            onFilterChange={setActiveFilters}
            visible={showFilters}
          />

          <div className="flex-1 min-w-0">
            {/* Mode header */}
            <div className="flex items-center justify-between mb-4 min-h-[24px]">
              {hasQuery ? (
                <div className="flex items-center gap-2">
                  <Search size={13} className="text-zinc-500" />
                  <span className="text-sm text-zinc-400">Results for</span>
                  <span className="text-sm font-medium text-white">"{query}"</span>
                  {searchResults !== null && (
                    <span className="text-xs text-zinc-600">· {searchResults.length} found</span>
                  )}
                  {queryTime && (
                    <span className="text-xs text-zinc-700">· {queryTime}ms</span>
                  )}
                </div>
              ) : isSimilarMode ? (
                <div className="flex items-center gap-2">
                  <Search size={13} className="text-zinc-500" />
                  <span className="text-sm text-zinc-400">Similar images</span>
                  {searchResults !== null && (
                    <span className="text-xs text-zinc-600">· {searchResults.length} found</span>
                  )}
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <LayoutGrid size={13} className="text-zinc-500" />
                  <span className="text-sm font-medium text-zinc-300">Library</span>
                  {totalImages > 0 && (
                    <span className="text-xs text-zinc-600">· {totalImages} images</span>
                  )}
                </div>
              )}

              {(hasQuery || isSimilarMode) && (
                <button
                  onClick={clearSearch}
                  className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  <X size={12} />
                  Clear search
                </button>
              )}
            </div>

            {/* Example queries — only when library is empty */}
            {!hasQuery && !isSimilarMode && !isGridLoading && browseImages?.length === 0 && (
              <div className="mb-8">
                <p className="text-xs font-semibold text-zinc-600 uppercase tracking-widest mb-3">
                  Try searching for
                </p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_QUERIES.map((eq) => (
                    <button
                      key={eq}
                      onClick={() => handleSearch(eq)}
                      className="px-3.5 py-2 bg-[#1a1a1a] hover:bg-[#222] border border-[#2a2a2a] hover:border-brand-500/40 text-zinc-400 hover:text-brand-300 text-xs font-medium rounded-lg transition-all duration-150"
                    >
                      {eq}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <ImageGrid
              results={displayImages}
              isLoading={isGridLoading}
              onImageClick={setSelectedImage}
              onSimilarSearch={(img) => {
                clearSearch()
                handleSimilarSearch(img)
              }}
              isBrowseMode={!hasQuery && !isSimilarMode}
            />
          </div>
        </div>
      </main>

      {selectedImage && (
        <ImageModal
          image={selectedImage}
          onClose={() => setSelectedImage(null)}
          onSimilarSearch={(img) => {
            setSelectedImage(null)
            clearSearch()
            handleSimilarSearch(img)
          }}
          onDelete={handleDelete}
        />
      )}
    </div>
  )
}
