
export default function ResultsDashboard({ result, onReset }) {
  if (!result) return null

  const isFake = result.label === 'FAKE'
  const fakePercent = (result.fake_probability * 100).toFixed(1)
  const confidencePercent = (result.confidence * 100).toFixed(1)

  const statusColor = isFake ? 'text-red-400' : 'text-emerald-400'
  const barColor = isFake ? 'bg-red-400' : 'bg-emerald-400'

  return (
    <section className="mt-8 rounded-2xl border border-slate-700 bg-slate-900 p-6 md:p-8">

      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-2xl font-bold text-white">
          Analysis Result
        </h3>

        {result.demo && (
          <span className="rounded-full border border-amber-700 bg-amber-950 px-3 py-1 text-xs font-semibold text-amber-300">
            SIMULATED DEMO DATA
          </span>
        )}
      </div>

      <div className="rounded-xl border border-slate-700 bg-slate-800 p-6">
        <p className="text-sm text-slate-400">
          Classification
        </p>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h4 className={`text-3xl font-bold ${statusColor}`}>
              {isFake ? 'Likely Fake' : 'Likely Real'}
            </h4>

            <p className="mt-2 text-sm text-slate-400">
              Media Type: {result.modality.toUpperCase()}
            </p>
          </div>

          <span className={`text-4xl ${statusColor}`} aria-hidden="true">
            {isFake ? '⚠' : '✓'}
          </span>
        </div>

        <div className="mt-7">
          <div className="mb-3 flex items-center justify-between gap-3">
            <span className="text-sm text-slate-300">
              Fake Probability
            </span>

            <span className="text-xl font-semibold text-white">
              {fakePercent}%
            </span>
          </div>

          <div
            role="progressbar"
            aria-label="Estimated fake probability"
            aria-valuenow={Number(fakePercent)}
            aria-valuemin={0}
            aria-valuemax={100}
            className="h-3 overflow-hidden rounded-full bg-slate-700"
          >
            <div
              className={`${barColor} h-full rounded-full transition-all`}
              style={{ width: `${fakePercent}%` }}
            />
          </div>
        </div>

        <div className="mt-6 border-t border-slate-700 pt-5">
          <p className="text-sm text-slate-400">
            Classification Confidence
          </p>

          <p className="mt-2 text-2xl font-bold text-white">
            {confidencePercent}%
          </p>

          <p className="mt-1 text-xs text-slate-500">
            Derived from the model score; not a guarantee of correctness.
          </p>
        </div>
      </div>

      {result.heatmap_url && (
        <div className="mt-6 rounded-xl border border-slate-700 p-5">
          <h4 className="mb-4 text-lg font-semibold text-white">
            Explanation Heatmap
          </h4>

          <img
            src={result.heatmap_url}
            alt="Model-generated visual explanation heatmap"
            className="mx-auto max-h-96 max-w-full rounded-lg object-contain"
          />

          <p className="mt-3 text-xs text-slate-400">
            Highlighted regions indicate areas influencing the model.
            They do not prove manipulation.
          </p>
        </div>
      )}

      <div className="mt-6 rounded-xl border border-slate-700 bg-slate-950 p-5">
        <h4 className="font-semibold text-cyan-400">
          Analysis Information
        </h4>

        <p className="mt-3 text-sm leading-relaxed text-slate-300">
          {result.explanation ||
            'The detection model returned a classification score. No additional explanation was provided.'}
        </p>
      </div>

      <div className="mt-6 rounded-xl border border-amber-900/60 bg-amber-950/20 p-5">
        <h4 className="font-semibold text-amber-300">
          Important Disclaimer
        </h4>

        <p className="mt-2 text-sm leading-relaxed text-slate-300">
          {result.demo
            ? 'This result is simulated and does not represent an actual AI detection.'
            : 'This is a model-based estimate, not definitive proof that media is real or manipulated. False positives and false negatives are possible.'}
        </p>
      </div>

      <button
        type="button"
        onClick={onReset}
        className="mt-7 w-full rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400"
      >
        Analyze Another File
      </button>
    </section>
  )
}
