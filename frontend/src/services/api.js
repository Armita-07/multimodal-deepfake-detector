
const API_MODE = import.meta.env.VITE_API_MODE || 'mock'

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
).replace(/\/$/, '')

export const IS_DEMO = API_MODE === 'mock'

function delay(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Analysis cancelled', 'AbortError'))
      return
    }

    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)

    function onAbort() {
      clearTimeout(timer)
      reject(new DOMException('Analysis cancelled', 'AbortError'))
    }

    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

// Demonstration values only.
// These scores are NOT calculated from the uploaded file.
function createMockResult(mediaType) {
  const scores = {
    image: 0.86,
    video: 0.72,
    audio: 0.18,
  }

  const fakeProbability = scores[mediaType]

  return {
    label: fakeProbability >= 0.5 ? 'FAKE' : 'REAL',
    fake_probability: fakeProbability,
    confidence: Math.max(
      fakeProbability,
      1 - fakeProbability
    ),
    modality: mediaType,
    explanation:
      'This is simulated demonstration data. No AI model has analyzed the uploaded file.',
    heatmap_url: null,
    demo: true,
  }
}

function normalizeResult(data, mediaType) {
  if (!data || typeof data !== 'object') {
    throw new Error('Invalid response from detection API.')
  }

  // Provisional API contract: probability between 0 and 1.
  const probability = data.fake_probability

  if (
    typeof probability !== 'number' ||
    !Number.isFinite(probability) ||
    probability < 0 ||
    probability > 1
  ) {
    throw new Error(
      'Backend must return fake_probability as a number between 0 and 1.'
    )
  }

  const expectedLabel = probability >= 0.5 ? 'FAKE' : 'REAL'
  const label = data.label?.toUpperCase() ?? expectedLabel

  if (label !== expectedLabel) {
    throw new Error('Backend label and probability disagree.')
  }

  return {
    label,
    fake_probability: probability,
    confidence: Math.max(probability, 1 - probability),
    modality: data.modality || mediaType,
    explanation: data.explanation || null,
    heatmap_url: data.heatmap_url || null,
    demo: false,
  }
}

export async function analyzeMedia(selectedFile, signal) {
  const { file, mediaType } = selectedFile

  if (API_MODE === 'mock') {
    await delay(1600, signal)
    return createMockResult(mediaType)
  }

  if (API_MODE !== 'live') {
    throw new Error('Invalid VITE_API_MODE configuration.')
  }

  const formData = new FormData()
  formData.append('file', file)

  let response

  try {
    response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
      signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error

    throw new Error(
      'Cannot connect to the backend. Check that FastAPI is running and CORS is configured.'
    )
  }

  if (!response.ok) {
    throw new Error(`Analysis failed (HTTP ${response.status}).`)
  }

  let data

  try {
    data = await response.json()
  } catch {
    throw new Error('Backend returned invalid JSON.')
  }

  return normalizeResult(data, mediaType)
}
