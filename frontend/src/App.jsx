
import { useEffect, useRef, useState } from 'react'
import FileUploader from './components/FileUploader'
import MediaPreview from './components/MediaPreview'
import AnalysisProgress from './components/AnalysisProgress'
import ResultsDashboard from './components/ResultsDashboard'
import { analyzeMedia, IS_DEMO } from './services/api'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const controllerRef = useRef(null)

  useEffect(() => {
    return () => {
      controllerRef.current?.abort()
    }
  }, [])

  function cancelCurrentRequest() {
    controllerRef.current?.abort()
    controllerRef.current = null
  }

  function handleFileSelect(fileData) {
    cancelCurrentRequest()
    setSelectedFile(fileData)
    setResult(null)
    setError('')
    setStatus('idle')
  }

  function handleRemoveFile() {
    cancelCurrentRequest()
    setSelectedFile(null)
    setResult(null)
    setError('')
    setStatus('idle')
  }

  async function handleAnalyze() {
    if (!selectedFile || status === 'loading') return

    const controller = new AbortController()
    controllerRef.current = controller

    setStatus('loading')
    setError('')
    setResult(null)

    try {
      const analysisResult = await analyzeMedia(
        selectedFile,
        controller.signal
      )

      if (controller.signal.aborted) return

      setResult(analysisResult)
      setStatus('success')
    } catch (err) {
      if (controller.signal.aborted) return

      setError(err.message || 'An unexpected error occurred.')
      setStatus('error')
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
      }
    }
  }

  function handleCancel() {
    cancelCurrentRequest()
    setStatus('idle')
    setError('')
  }

  return (
    <div className="min-h-screen bg-slate-950 text-white">

      {/* Navigation */}
      <nav className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-6 py-5 md:px-8">
        <h1 className="text-2xl font-bold text-cyan-400">
          Deepfake Detector
        </h1>

        <span className="text-sm text-slate-400">
          Multimodal AI Detection System
        </span>
      </nav>

      <main className="mx-auto max-w-5xl px-6 py-16">

        <div className="text-center">
          <div className="mb-6 inline-block rounded-full border border-cyan-800 bg-cyan-950 px-4 py-2 text-sm text-cyan-300">
            AI-Powered Media Verification
          </div>

          <h2 className="text-4xl font-bold leading-tight md:text-5xl">
            Detect AI-Generated
            <br />
            <span className="text-cyan-400">
              Images, Videos & Audio
            </span>
          </h2>

          <p className="mx-auto mt-6 max-w-2xl text-lg text-slate-400">
            Upload your media and analyze it using our
            multimodal deepfake detection system.
            Receive predictions and interpretable results.
          </p>

          <div className="mt-5">
            {IS_DEMO ? (
              <span className="inline-block rounded-full border border-amber-700 bg-amber-950 px-4 py-2 text-sm text-amber-300">
                Demo Mode — Simulated Predictions
              </span>
            ) : (
              <span className="inline-block rounded-full border border-emerald-800 bg-emerald-950 px-4 py-2 text-sm text-emerald-300">
                Live API Mode
              </span>
            )}
          </div>
        </div>

        <div className="mx-auto mt-12 max-w-2xl">

          {status !== 'success' && (
            <>
              <FileUploader
                onFileSelect={handleFileSelect}
              />

              <MediaPreview
                selectedFile={selectedFile}
                onRemove={handleRemoveFile}
              />
            </>
          )}

          {selectedFile && status === 'idle' && (
            <button
              type="button"
              onClick={handleAnalyze}
              className="mt-6 w-full rounded-xl bg-cyan-500 px-6 py-4 text-lg font-semibold text-slate-950 transition hover:bg-cyan-400"
            >
              Analyze Media
            </button>
          )}

          {status === 'loading' && (
            <AnalysisProgress onCancel={handleCancel} />
          )}

          {status === 'error' && (
            <div
              role="alert"
              className="mt-6 rounded-xl border border-red-800 bg-red-950/30 p-5"
            >
              <h3 className="font-semibold text-red-400">
                Analysis Failed
              </h3>

              <p className="mt-2 text-sm text-slate-300">
                {error}
              </p>

              <button
                type="button"
                onClick={handleAnalyze}
                className="mt-4 rounded-lg border border-red-600 px-5 py-2 text-sm text-white hover:bg-red-950"
              >
                Retry Analysis
              </button>
            </div>
          )}

          {status === 'success' && (
            <ResultsDashboard
              result={result}
              onReset={handleRemoveFile}
            />
          )}
        </div>

        <p className="mt-10 text-center text-sm text-slate-500">
          Detection results are model-based estimates,
          not definitive proof of authenticity.
        </p>

      </main>
    </div>
  )
}

export default App
