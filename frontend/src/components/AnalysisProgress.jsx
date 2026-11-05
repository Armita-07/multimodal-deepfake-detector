
export default function AnalysisProgress({ onCancel }) {
  return (
    <section
      role="status"
      aria-live="polite"
      className="mt-8 rounded-2xl border border-cyan-800 bg-slate-900 p-8 text-center"
    >
      <div
        aria-hidden="true"
        className="mx-auto mb-5 h-12 w-12 animate-spin rounded-full border-4 border-slate-700 border-t-cyan-400"
      />

      <h3 className="text-xl font-semibold text-white">
        Analyzing Your Media
      </h3>

      <p className="mt-3 text-sm text-slate-400">
        Processing your file and preparing the detection result.
      </p>

      <p className="mt-2 text-xs text-slate-500">
        Processing time depends on file size and backend availability.
      </p>

      <button
        type="button"
        onClick={onCancel}
        className="mt-6 rounded-lg border border-slate-600 px-6 py-2 text-sm text-slate-300 hover:bg-slate-800"
      >
        Cancel Analysis
      </button>
    </section>
  )
}
