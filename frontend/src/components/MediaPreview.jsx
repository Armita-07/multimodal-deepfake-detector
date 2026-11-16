
import { useEffect, useState } from 'react'

export default function MediaPreview({ selectedFile, onRemove }) {
  const [previewUrl, setPreviewUrl] = useState('')

  useEffect(() => {
    if (!selectedFile) return

    const url = URL.createObjectURL(selectedFile.file)
    setPreviewUrl(url)

    return () => {
      URL.revokeObjectURL(url)
      setPreviewUrl('')
    }
  }, [selectedFile])

  if (!selectedFile) return null

  const { file, mediaType } = selectedFile

  const sizeInMB = (file.size / (1024 * 1024)).toFixed(2)

  return (
    <section className="mt-8 rounded-2xl border border-slate-700 bg-slate-900 p-6">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-xl font-semibold">
            Selected Media
          </h3>

          <p className="mt-2 break-all text-sm text-slate-300">
            {file.name}
          </p>

          <p className="mt-1 text-xs text-slate-500">
            {mediaType.toUpperCase()} • {sizeInMB} MB
          </p>
        </div>

        <button
          type="button"
          onClick={onRemove}
          className="rounded-lg border border-red-500/50 px-4 py-2 text-sm text-red-400 transition hover:bg-red-500/10"
        >
          Remove File
        </button>
      </div>

      {previewUrl && mediaType === 'image' && (
        <img
          src={previewUrl}
          alt="Preview of selected media"
          className="mx-auto max-h-96 max-w-full rounded-lg object-contain"
        />
      )}

      {previewUrl && mediaType === 'video' && (
        <video
          src={previewUrl}
          controls
          preload="metadata"
          className="mx-auto max-h-96 w-full rounded-lg"
        >
          Your browser does not support video playback.
        </video>
      )}

      {previewUrl && mediaType === 'audio' && (
        <div className="rounded-lg bg-slate-800 p-6">
          <p className="mb-4 text-center text-slate-300">
            Audio Preview
          </p>

          <audio
            src={previewUrl}
            controls
            preload="metadata"
            className="w-full"
          >
            Your browser does not support audio playback.
          </audio>
        </div>
      )}

      <div className="mt-6 flex items-center gap-2 text-sm text-emerald-400">
        <span aria-hidden="true">✓</span>
        File selected successfully
      </div>
    </section>
  )
}
