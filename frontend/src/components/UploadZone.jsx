import { useEffect, useRef, useState } from 'react'
import { Upload, CloudUpload, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import clsx from 'clsx'

const ACCEPT_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif']
const MAX_FILES = 50

export default function UploadZone({ onUpload, isUploading, uploadProgress }) {
  const [isDragging, setIsDragging] = useState(false)
  const [dragCounter, setDragCounter] = useState(0)
  const fileInputRef = useRef(null)

  // Document-level drag events
  useEffect(() => {
    const handleDragEnter = (e) => {
      e.preventDefault()
      setDragCounter((c) => c + 1)
      if (e.dataTransfer.types.includes('Files')) {
        setIsDragging(true)
      }
    }

    const handleDragLeave = (e) => {
      e.preventDefault()
      setDragCounter((c) => {
        const next = c - 1
        if (next <= 0) setIsDragging(false)
        return Math.max(0, next)
      })
    }

    const handleDragOver = (e) => {
      e.preventDefault()
    }

    const handleDrop = (e) => {
      e.preventDefault()
      setIsDragging(false)
      setDragCounter(0)

      const files = Array.from(e.dataTransfer.files)
        .filter((f) => ACCEPT_TYPES.includes(f.type))
        .slice(0, MAX_FILES)

      if (files.length > 0) {
        onUpload(files)
      }
    }

    document.addEventListener('dragenter', handleDragEnter)
    document.addEventListener('dragleave', handleDragLeave)
    document.addEventListener('dragover', handleDragOver)
    document.addEventListener('drop', handleDrop)

    return () => {
      document.removeEventListener('dragenter', handleDragEnter)
      document.removeEventListener('dragleave', handleDragLeave)
      document.removeEventListener('dragover', handleDragOver)
      document.removeEventListener('drop', handleDrop)
    }
  }, [onUpload])

  const handleFileInput = (e) => {
    const files = Array.from(e.target.files)
      .filter((f) => ACCEPT_TYPES.includes(f.type))
      .slice(0, MAX_FILES)
    if (files.length > 0) {
      onUpload(files)
    }
    // Reset input so same files can be re-selected
    e.target.value = ''
  }

  const { processed = 0, total = 0, success = 0, duplicates = 0, errors = 0, done = false } =
    uploadProgress || {}

  return (
    <>
      {/* Upload button for navbar */}
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={isUploading}
        className={clsx(
          'flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-150',
          isUploading
            ? 'bg-brand-600/40 text-brand-300 cursor-not-allowed'
            : 'bg-brand-600 hover:bg-brand-700 text-white'
        )}
      >
        {isUploading ? (
          <>
            <Loader2 size={15} className="animate-spin" />
            <span className="hidden sm:inline">
              {total > 0 ? `${processed}/${total}` : 'Uploading...'}
            </span>
          </>
        ) : (
          <>
            <Upload size={15} />
            <span className="hidden sm:inline">Upload</span>
          </>
        )}
      </button>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPT_TYPES.join(',')}
        multiple
        onChange={handleFileInput}
        className="hidden"
      />

      {/* Upload progress toast */}
      {(isUploading || done) && (
        <div className="fixed bottom-6 right-6 z-40 bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4 shadow-2xl min-w-[260px] animate-slideUp">
          {isUploading ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2.5">
                <Loader2 size={16} className="text-brand-400 animate-spin shrink-0" />
                <span className="text-sm font-medium text-zinc-200">
                  Processing images...
                </span>
              </div>
              {total > 0 && (
                <>
                  <p className="text-xs text-zinc-500 pl-6">
                    {processed} of {total} images
                  </p>
                  <div className="w-full bg-[#2a2a2a] rounded-full h-1.5 overflow-hidden">
                    <div
                      className="h-full bg-brand-500 rounded-full transition-all duration-300"
                      style={{ width: `${total > 0 ? (processed / total) * 100 : 0}%` }}
                    />
                  </div>
                </>
              )}
            </div>
          ) : done ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2.5">
                <CheckCircle size={16} className="text-emerald-400 shrink-0" />
                <span className="text-sm font-medium text-zinc-200">Upload complete</span>
              </div>
              <div className="pl-6 space-y-1 text-xs">
                {success > 0 && (
                  <p className="text-emerald-400">
                    {success} image{success !== 1 ? 's' : ''} added
                  </p>
                )}
                {duplicates > 0 && (
                  <p className="text-amber-400">
                    {duplicates} duplicate{duplicates !== 1 ? 's' : ''} skipped
                  </p>
                )}
                {errors > 0 && (
                  <p className="text-red-400 flex items-center gap-1">
                    <XCircle size={11} />
                    {errors} failed
                  </p>
                )}
              </div>
            </div>
          ) : null}
        </div>
      )}

      {/* Full-page drag overlay */}
      {isDragging && (
        <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
          <div className="relative flex flex-col items-center gap-5 p-12 border-2 border-dashed border-brand-500/60 rounded-2xl bg-brand-600/10 mx-8">
            <div className="w-16 h-16 rounded-2xl bg-brand-600/20 flex items-center justify-center">
              <CloudUpload size={32} className="text-brand-400" />
            </div>
            <div className="text-center">
              <p className="text-xl font-semibold text-white mb-1">Drop images here</p>
              <p className="text-sm text-zinc-400">PNG, JPEG, WebP, GIF — up to 50 files</p>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
