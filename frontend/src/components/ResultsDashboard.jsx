import { useEffect, useState } from 'react'

export default function ResultsDashboard({ result, selectedFile, onReset }) {
  if (!result) return null

  const isFake = result.label === 'FAKE'
  const fakePercent = (result.fake_probability * 100).toFixed(1)
  const realPercent = ((result.real_probability ?? (1 - result.fake_probability)) * 100).toFixed(1)
  const confidencePercent = (result.confidence * 100).toFixed(1)

  const statusColor = isFake ? 'text-rose-400' : 'text-emerald-400'
  const statusBg = isFake ? 'bg-rose-500/10' : 'bg-emerald-500/10'
  const statusBorder = isFake ? 'border-rose-500/30' : 'border-emerald-500/30'
  const barGradient = isFake ? 'from-amber-500 to-rose-500' : 'from-cyan-500 to-emerald-400'

  // Original media preview URL
  const [originalMediaUrl, setOriginalMediaUrl] = useState('')
  const [viewMode, setViewMode] = useState('side-by-side') // 'side-by-side' | 'blend' | 'toggle'
  const [blendOpacity, setBlendOpacity] = useState(60) // 0 to 100
  const [activeToggle, setActiveToggle] = useState('heatmap') // 'original' | 'heatmap'

  useEffect(() => {
    if (!selectedFile?.file) return
    const url = URL.createObjectURL(selectedFile.file)
    setOriginalMediaUrl(url)

    return () => {
      URL.revokeObjectURL(url)
    }
  }, [selectedFile])

  const vizUrl = result.visualization_url || result.heatmap_url
  const metrics = result.metrics || {}
  const videoFrameMetrics = metrics.video_frame_metrics
  const dossierName = result.report_path ? result.report_path.split(/[\\/]/).pop() : 'Audit Dossier'

  return (
    <div className="w-full space-y-8 animate-fadeIn">

      {/* Top Header & Navigation Strip */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight text-white">
              Forensic Analysis Dossier
            </h2>
            <span className="rounded-full border border-cyan-800/60 bg-cyan-950/70 px-3 py-0.5 text-xs font-semibold text-cyan-300">
              {result.modality.toUpperCase()}
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Multimodal detection results, explainability telemetry & audit verification
          </p>
        </div>

        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/90 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:bg-slate-700 hover:text-white"
        >
          <span>↺</span> Analyze New Media
        </button>
      </div>

      {/* Primary Classification Verdict Banner */}
      <div className={`rounded-2xl border ${statusBorder} ${statusBg} p-6 md:p-8 shadow-2xl backdrop-blur-sm`}>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">

          {/* Left Column: Verdict Badge & Icon */}
          <div className="lg:col-span-5 flex items-start gap-5">
            <div className={`flex h-20 w-20 shrink-0 items-center justify-center rounded-2xl border ${statusBorder} bg-slate-900/90 text-4xl shadow-inner`}>
              <span className={statusColor} aria-hidden="true">
                {isFake ? '⚠️' : '✓'}
              </span>
            </div>

            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Classification Verdict
              </span>
              <h3 className={`text-3xl md:text-4xl font-black ${statusColor} tracking-tight mt-0.5`}>
                {isFake ? 'Likely Manipulated' : 'Likely Authentic'}
              </h3>
              <p className="text-sm text-slate-300 mt-1 font-medium">
                {isFake ? 'Synthetic or deepfake manipulation detected.' : 'No synthetic anomalies detected.'}
              </p>
              <div className="mt-2.5 inline-flex items-center gap-2 rounded-lg bg-slate-900/70 px-3 py-1 text-xs text-slate-300 border border-slate-800">
                <span className="text-slate-400">Model Confidence:</span>
                <span className="font-bold text-white">{confidencePercent}%</span>
              </div>
            </div>
          </div>

          {/* Right Column: Probability Gauge & Metrics */}
          <div className="lg:col-span-7 space-y-4">
            <div>
              <div className="flex items-center justify-between text-sm font-medium mb-2">
                <span className="text-slate-300">Estimated Fake Probability:</span>
                <span className={`text-2xl font-black ${statusColor}`}>{fakePercent}%</span>
              </div>

              {/* Progress bar with 50% Threshold marker */}
              <div className="relative h-4 overflow-hidden rounded-full bg-slate-900 border border-slate-700/60 shadow-inner">
                <div
                  className="absolute top-0 bottom-0 z-10 w-0.5 bg-yellow-400/90 shadow-sm"
                  style={{ left: '50%' }}
                  title="Decision Boundary (50%)"
                />
                <div
                  className={`h-full rounded-full bg-gradient-to-r ${barGradient} transition-all duration-700 ease-out`}
                  style={{ width: `${fakePercent}%` }}
                />
              </div>

              <div className="mt-2 flex justify-between text-xs text-slate-400 font-medium">
                <span>0% (Authentic)</span>
                <span className="text-yellow-400 font-semibold">Decision Boundary: 50%</span>
                <span>100% (Manipulated)</span>
              </div>
            </div>

            {/* Quick KPI stats strip */}
            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="rounded-xl border border-slate-800/80 bg-slate-900/70 p-3.5">
                <span className="text-xs text-slate-400">Authentic Likelihood</span>
                <p className="text-xl font-bold text-slate-200 mt-0.5">{realPercent}%</p>
              </div>
              <div className="rounded-xl border border-slate-800/80 bg-slate-900/70 p-3.5">
                <span className="text-xs text-slate-400">Synthetic Likelihood</span>
                <p className={`text-xl font-bold mt-0.5 ${statusColor}`}>{fakePercent}%</p>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Visual Forensic Explanation & Comparison Showcase */}
      {vizUrl && (
        <div className="rounded-2xl border border-slate-700/80 bg-slate-900/90 p-6 md:p-8 shadow-2xl space-y-6">

          {/* Section Header */}
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-xl">🔍</span>
                <h3 className="text-xl md:text-2xl font-bold text-white tracking-tight">
                  Visual Forensic Analysis & Comparison
                </h3>
              </div>
              <p className="text-sm text-slate-400 mt-1">
                {result.modality === 'image' && 'Spatial Occlusion Sensitivity — isolating pixels influencing model weights.'}
                {result.modality === 'video' && 'Temporal Anomaly Trajectory — per-frame fake probabilities against decision threshold.'}
                {result.modality === 'audio' && 'Acoustic Signal Waveform & Mel-Frequency Spectrogram energy density.'}
              </p>
            </div>

            {/* Mode Controls for Images */}
            {result.modality === 'image' && originalMediaUrl && (
              <div className="flex rounded-xl border border-slate-700 bg-slate-950 p-1 text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => setViewMode('side-by-side')}
                  className={`rounded-lg px-3.5 py-1.5 transition ${
                    viewMode === 'side-by-side'
                      ? 'bg-cyan-500 text-slate-950 shadow-md font-bold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Side-by-Side
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('blend')}
                  className={`rounded-lg px-3.5 py-1.5 transition ${
                    viewMode === 'blend'
                      ? 'bg-cyan-500 text-slate-950 shadow-md font-bold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Overlay Blend
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('toggle')}
                  className={`rounded-lg px-3.5 py-1.5 transition ${
                    viewMode === 'toggle'
                      ? 'bg-cyan-500 text-slate-950 shadow-md font-bold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Toggle View
                </button>
              </div>
            )}
          </div>

          {/* ===================== VIDEO MODE ===================== */}
          {result.modality === 'video' && (
            <div className="space-y-6">

              {/* Two Column Side-by-Side Layout for Video & Graph */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">

                {/* Left: Video Player */}
                {originalMediaUrl && (
                  <div className="lg:col-span-5 flex flex-col rounded-xl border border-slate-800 bg-slate-950/80 p-4 shadow-lg">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                      Original Video Stream
                    </span>
                    <div className="flex-1 flex items-center justify-center rounded-lg bg-black overflow-hidden border border-slate-800/80 max-h-[380px]">
                      <video
                        src={originalMediaUrl}
                        controls
                        className="max-h-[360px] w-auto max-w-full rounded"
                      />
                    </div>
                  </div>
                )}

                {/* Right: Dark-Mode Temporal Anomaly Timeline Graph */}
                <div className={`${originalMediaUrl ? 'lg:col-span-7' : 'lg:col-span-12'} flex flex-col rounded-xl border border-cyan-900/40 bg-slate-950/80 p-4 shadow-lg`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                      Temporal Anomaly Trajectory
                    </span>
                    <span className="text-xs text-slate-400 font-mono">16-Frame Sampling</span>
                  </div>
                  <div className="flex-1 flex items-center justify-center rounded-lg bg-slate-950 overflow-hidden border border-slate-800/80 p-2 min-h-[300px]">
                    <img
                      src={vizUrl}
                      alt="Temporal Anomaly Timeline Graph"
                      className="w-full h-auto max-h-[360px] object-contain rounded"
                    />
                  </div>
                </div>

              </div>

              {/* Face Detection Notice if No Faces Detected */}
              {videoFrameMetrics && videoFrameMetrics.has_faces_detected === false && (
                <div className="rounded-xl border border-amber-900/60 bg-amber-950/30 p-4 text-xs text-amber-200">
                  <p className="font-semibold text-amber-300 flex items-center gap-2">
                    <span>⚠️</span> Facial Feature Detection Note:
                  </p>
                  <p className="mt-1 text-slate-300 leading-relaxed">
                    MTCNN scanned 16 sampled frames but detected <strong>0 human faces</strong> (common in 3D animations, cartoons, or extreme profile angles). Deepfake facial classification requires visible facial landmarks, so the baseline remains at the neutral 50% boundary.
                  </p>
                </div>
              )}

              {/* Video Metric Summary Strip */}
              {videoFrameMetrics && (
                <div className="rounded-xl border border-slate-800/90 bg-slate-950/90 p-5 space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="rounded-lg bg-slate-900/80 border border-slate-800 p-3.5">
                      <span className="text-xs text-slate-400 font-medium">Frames Evaluated</span>
                      <p className="text-2xl font-black text-white mt-1">
                        {videoFrameMetrics.total_frames_analyzed} <span className="text-sm font-normal text-slate-400">frames</span>
                      </p>
                    </div>

                    <div className="rounded-lg bg-slate-900/80 border border-slate-800 p-3.5">
                      <span className="text-xs text-slate-400 font-medium">Suspicious Frames (≥ 50%)</span>
                      <p className={`text-2xl font-black mt-1 ${videoFrameMetrics.suspicious_frames_count > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                        {videoFrameMetrics.suspicious_frames_count} / {videoFrameMetrics.total_frames_analyzed}
                      </p>
                    </div>

                    <div className="rounded-lg bg-slate-900/80 border border-slate-800 p-3.5">
                      <span className="text-xs text-slate-400 font-medium">Sampling Architecture</span>
                      <p className="text-lg font-bold text-slate-200 mt-1">
                        MTCNN + EfficientNet-B0
                      </p>
                    </div>
                  </div>

                  {/* Sampled Frame Pills */}
                  {videoFrameMetrics.frame_scores && (
                    <div className="border-t border-slate-800/80 pt-4">
                      <div className="flex items-center justify-between mb-2.5">
                        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                          Sampled Frame Deepfake Probabilities:
                        </span>
                        <div className="flex items-center gap-3 text-xs">
                          <span className="flex items-center gap-1.5 text-slate-400">
                            <span className="h-2 w-2 rounded-full bg-slate-600"></span> Authentic (&lt;50%)
                          </span>
                          <span className="flex items-center gap-1.5 text-rose-400">
                            <span className="h-2 w-2 rounded-full bg-rose-500"></span> Suspicious (≥50%)
                          </span>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        {videoFrameMetrics.frame_scores.map((score, index) => {
                          const isSuspicious = score >= 0.5
                          return (
                            <div
                              key={index}
                              title={`Frame ${index + 1}: ${(score * 100).toFixed(1)}% Fake`}
                              className={`rounded-lg px-3 py-1 text-xs font-mono font-semibold transition ${
                                isSuspicious
                                  ? 'bg-rose-950/80 text-rose-300 border border-rose-800/80 shadow-sm'
                                  : 'bg-slate-800/80 text-slate-300 border border-slate-700/80'
                              }`}
                            >
                              F{index + 1}: <span className="font-bold">{(score * 100).toFixed(0)}%</span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

            </div>
          )}

          {/* ===================== IMAGE MODE ===================== */}
          {result.modality === 'image' && (
            <div className="space-y-6">

              {/* Side-by-Side View */}
              {viewMode === 'side-by-side' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
                  <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-4">
                    <span className="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                      Original Input Image
                    </span>
                    <div className="rounded-lg bg-black/60 overflow-hidden border border-slate-800 flex items-center justify-center min-h-[320px] max-h-[440px]">
                      {originalMediaUrl && (
                        <img
                          src={originalMediaUrl}
                          alt="Original input"
                          className="max-h-[420px] w-auto max-w-full object-contain rounded"
                        />
                      )}
                    </div>
                  </div>

                  <div className="rounded-xl border border-cyan-900/40 bg-slate-950/80 p-4">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                        Occlusion Sensitivity Heatmap
                      </span>
                      <span className="text-xs text-slate-400 font-mono">Jet Colormap (6×6 Grid)</span>
                    </div>
                    <div className="rounded-lg bg-black/60 overflow-hidden border border-slate-800 flex items-center justify-center min-h-[320px] max-h-[440px]">
                      <img
                        src={vizUrl}
                        alt="Occlusion heatmap"
                        className="max-h-[420px] w-auto max-w-full object-contain rounded"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Blend Mode */}
              {viewMode === 'blend' && (
                <div className="flex flex-col items-center space-y-4">
                  <div className="relative mx-auto max-h-[440px] w-full max-w-2xl overflow-hidden rounded-xl border border-slate-800 bg-slate-950 p-2">
                    {originalMediaUrl && (
                      <img
                        src={originalMediaUrl}
                        alt="Original base"
                        className="mx-auto max-h-[420px] w-auto object-contain rounded"
                      />
                    )}
                    <img
                      src={vizUrl}
                      alt="Heatmap overlay"
                      style={{ opacity: blendOpacity / 100 }}
                      className="absolute inset-0 mx-auto h-full w-auto object-contain pointer-events-none transition-opacity duration-75"
                    />
                  </div>

                  <div className="w-full max-w-lg space-y-2">
                    <div className="flex justify-between text-xs text-slate-400 font-medium">
                      <span>Original Image (0%)</span>
                      <span className="text-cyan-400 font-bold">Heatmap Blend: {blendOpacity}%</span>
                      <span>Heatmap (100%)</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={blendOpacity}
                      onChange={(e) => setBlendOpacity(Number(e.target.value))}
                      className="w-full accent-cyan-400 cursor-pointer"
                    />
                  </div>
                </div>
              )}

              {/* Toggle View */}
              {viewMode === 'toggle' && (
                <div className="flex flex-col items-center space-y-4">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setActiveToggle('original')}
                      className={`rounded-lg px-4 py-2 text-xs font-bold transition ${
                        activeToggle === 'original'
                          ? 'bg-slate-700 text-white shadow'
                          : 'bg-slate-800 text-slate-400 hover:text-white'
                      }`}
                    >
                      Original
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveToggle('heatmap')}
                      className={`rounded-lg px-4 py-2 text-xs font-bold transition ${
                        activeToggle === 'heatmap'
                          ? 'bg-cyan-500 text-slate-950 shadow'
                          : 'bg-slate-800 text-slate-400 hover:text-white'
                      }`}
                    >
                      Heatmap Overlay
                    </button>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-950 p-3 max-h-[440px] w-full max-w-2xl flex items-center justify-center">
                    <img
                      src={activeToggle === 'original' ? originalMediaUrl : vizUrl}
                      alt="Toggled view"
                      className="max-h-[400px] w-auto max-w-full object-contain rounded"
                    />
                  </div>
                </div>
              )}

              {/* Image Artifact Status Notice */}
              {result.visualization && result.visualization.has_artifacts === false && (
                <div className="rounded-xl border border-emerald-900/60 bg-emerald-950/25 p-4 text-xs text-emerald-200">
                  <p className="font-semibold text-emerald-300 flex items-center gap-2">
                    <span>✓</span> Authentic Media Verified: No Manipulation Clusters
                  </p>
                  <p className="mt-1 text-slate-300 leading-relaxed">
                    The model evaluated the image across the 6×6 occlusion grid and found <strong>zero synthetic manipulation clusters</strong> (all regional sensitivity deltas &lt; 3%). Because this photo is classified as authentic, the overlay shows no artificial anomaly highlights.
                  </p>
                </div>
              )}

              {/* Heatmap Legend */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-4 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-400">
                <div className="flex items-center gap-2.5">
                  <span className="font-semibold text-slate-200">Influence Scale:</span>
                  <span className="h-3 w-7 rounded bg-blue-600" title="Neutral"></span>
                  <span>Neutral</span>
                  <span className="h-3 w-8 rounded bg-gradient-to-r from-emerald-500 via-yellow-400 to-rose-600"></span>
                  <span className="text-rose-400 font-semibold">Manipulated Artifact Driver</span>
                </div>
                <span className="text-slate-500 font-mono">Gaussian-smoothed Occlusion Sensitivity (σ = 12)</span>
              </div>

            </div>
          )}

          {/* ===================== AUDIO MODE ===================== */}
          {result.modality === 'audio' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
                {originalMediaUrl && (
                  <div className="lg:col-span-4 flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-950/80 p-5">
                    <div>
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 block">
                        Original Speech Playback
                      </span>
                      <audio src={originalMediaUrl} controls className="w-full mt-3" />
                    </div>

                    <div className="mt-6 pt-4 border-t border-slate-800 text-xs space-y-2 text-slate-400">
                      <div className="flex justify-between">
                        <span>Sampling Rate:</span>
                        <span className="font-bold text-white">16 kHz Mono</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Temporal Slice:</span>
                        <span className="font-bold text-white">10.0 Seconds</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Acoustic Backbone:</span>
                        <span className="font-bold text-white">Facebook Wav2Vec 2.0</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className={`${originalMediaUrl ? 'lg:col-span-8' : 'lg:col-span-12'} rounded-xl border border-cyan-900/40 bg-slate-950/80 p-4`}>
                  <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 mb-3 block">
                    Forensic Acoustic Waveform & Mel-Spectrogram
                  </span>
                  <div className="rounded-lg bg-slate-950 p-2 border border-slate-800 flex items-center justify-center">
                    <img
                      src={vizUrl}
                      alt="Audio Spectrogram"
                      className="w-full h-auto max-h-[380px] object-contain rounded"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      )}

      {/* Bottom Row: Pipeline Metrics & Disclaimer in Wide Side-by-Side Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Left Column: Forensic Pipeline Metrics */}
        <div className="lg:col-span-7 rounded-2xl border border-slate-800/90 bg-slate-900/90 p-6 shadow-xl space-y-4">
          <h4 className="text-sm font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
            <span>⚙️</span> Forensic Pipeline Metrics & Audit Chain
          </h4>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-xl bg-slate-950/80 border border-slate-800 p-3">
              <span className="text-[11px] text-slate-400 uppercase">Modality</span>
              <p className="mt-1 font-bold text-white text-sm uppercase">{result.modality}</p>
            </div>

            <div className="rounded-xl bg-slate-950/80 border border-slate-800 p-3">
              <span className="text-[11px] text-slate-400 uppercase">Fusion Strategy</span>
              <p className="mt-1 font-bold text-white text-xs truncate" title={result.fusion_method}>
                {result.fusion_method || 'Late Fusion'}
              </p>
            </div>

            <div className="rounded-xl bg-slate-950/80 border border-slate-800 p-3">
              <span className="text-[11px] text-slate-400 uppercase">Threshold</span>
              <p className="mt-1 font-bold text-white text-sm">50.0% (0.50)</p>
            </div>

            <div className="rounded-xl bg-slate-950/80 border border-slate-800 p-3">
              <span className="text-[11px] text-slate-400 uppercase">Audit Dossier</span>
              <p className="mt-1 text-xs text-slate-300 font-mono truncate" title={result.report_path}>
                {dossierName}
              </p>
            </div>
          </div>

          {result.explanation && (
            <p className="text-xs text-slate-300 border-t border-slate-800/80 pt-3 leading-relaxed">
              <span className="font-bold text-slate-200">Analytical Summary: </span>
              {result.explanation}
            </p>
          )}
        </div>

        {/* Right Column: Forensic & Academic Disclaimer */}
        <div className="lg:col-span-5 rounded-2xl border border-amber-900/50 bg-amber-950/20 p-6 shadow-xl flex flex-col justify-between">
          <div>
            <h4 className="text-sm font-bold text-amber-300 flex items-center gap-2">
              <span>⚠️</span> Forensic & Regulatory Disclaimer
            </h4>
            <p className="mt-3 text-xs leading-relaxed text-slate-300">
              AI deepfake detectors are probabilistic estimators. High scores indicate statistical similarity to known synthetic generative artifacts, but do not constitute legal proof of tampering. Certified expert verification is recommended for legal proceedings.
            </p>
          </div>

          <div className="mt-4 pt-3 border-t border-amber-900/40 text-[11px] text-amber-400/80 font-medium">
            Project: Multimodal Deepfake & AI-Generated Content Detector
          </div>
        </div>

      </div>

      {/* Primary Action Button */}
      <div className="pt-2">
        <button
          type="button"
          onClick={onReset}
          className="w-full rounded-2xl bg-gradient-to-r from-cyan-500 to-cyan-400 px-6 py-4 text-base font-extrabold text-slate-950 transition hover:brightness-110 shadow-xl shadow-cyan-950/50"
        >
          Analyze Another File
        </button>
      </div>

    </div>
  )
}
