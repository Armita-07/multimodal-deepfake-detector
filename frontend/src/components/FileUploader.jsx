
import { useRef, useState } from 'react'

const ACCEPTED_TYPES = {
  'image/jpeg': 'image',
  'image/png': 'image',
  'image/webp': 'image',
  'video/mp4': 'video',
  'video/webm': 'video',
  'audio/mpeg': 'audio',
  'audio/wav': 'audio',
  'audio/x-wav': 'audio',
  'audio/ogg': 'audio',
}

const MAX_FILE_SIZE = 100 * 1024 * 1024 // 100 MB

export default function FileUploader({ onFileSelect }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState('')

  function validateFile(file) {
    if (!file) return

    setError('')

    if (!Object.hasOwn(ACCEPTED_TYPES, file.type)) {
      setError('Unsupported file type. Use JPG, PNG, WEBP, MP4, WEBM, MP3, WAV or OGG.')
      return
    }

    if (file.size === 0) {
      setError('The selected file is empty.')
      return
    }

    if (file.size > MAX_FILE_SIZE) {
      setError('File is too large. Maximum supported size is 100 MB.')
      return
    }

    onFileSelect({
      file,
      mediaType: ACCEPTED_TYPES[file.type],
    })
  }

  function handleDrop(event) {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(false)

    if (event.dataTransfer.files.length !== 1) {
      setError('Please upload exactly one file at a time.')
      return
    }

    validateFile(event.dataTransfer.files[0])
  }

  function handleInputChange(event) {
    const file = event.target.files?.[0]
    validateFile(file)

    // Allow selecting the same file again.
    event.target.value = ''
  }

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={(event) => {
        event.preventDefault()
        setIsDragging(false)
      }}
      onDrop={handleDrop}
      className={`rounded-2xl border-2 border-dashed p-10 text-center transition-colors ${
        isDragging
          ? 'border-cyan-400 bg-cyan-950/40'
          : 'border-slate-600 bg-slate-900'
      }`}
    >
      <div className="mb-4 text-5xl" aria-hidden="true">
        📁
      </div>

      <h3 className="text-xl font-semibold">
        Upload Media for Analysis
      </h3>

      <p className="mt-3 text-slate-400">
        Drag and drop a file here, or browse your device.
      </p>

      <p className="mt-2 text-sm text-slate-500">
        Images: JPG, PNG, WEBP
        <br />
        Videos: MP4, WEBM
        <br />
        Audio: MP3, WAV, OGG
      </p>

      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,.mp4,.webm,.mp3,.wav,.ogg"
        onChange={handleInputChange}
        className="sr-only"
        aria-label="Select a media file for analysis"
      />

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-7 rounded-lg bg-cyan-500 px-8 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400"
      >
        Choose File
      </button>

      <p className="mt-4 text-xs text-slate-500">
        Maximum file size: 100 MB
      </p>

      {error && (
        <p role="alert" className="mt-4 text-sm text-red-400">
          {error}
        </p>
      )}
    </div>
  )
}
