
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/$/, '')

// Real backend integration — no mock predictions
export const IS_DEMO = false

function normalizeResult(data, mediaType) {
  if (!data || typeof data !== 'object') {
    throw new Error('Invalid response from detection server.')
  }

  const fakeProbability = data.fake_probability
  const confidence = data.confidence

  if (
    typeof fakeProbability !== 'number' ||
    !Number.isFinite(fakeProbability) ||
    fakeProbability < 0 ||
    fakeProbability > 1
  ) {
    throw new Error('Backend returned an invalid fake probability.')
  }

  if (
    typeof confidence !== 'number' ||
    !Number.isFinite(confidence) ||
    confidence < 0 ||
    confidence > 1
  ) {
    throw new Error('Backend returned an invalid confidence score.')
  }

  const label = String(data.label || '').toUpperCase()

  if (!['REAL', 'FAKE'].includes(label)) {
    throw new Error('Backend returned an invalid prediction label.')
  }

  return {
    label,
    fake_probability: fakeProbability,
    confidence,
    modality: data.modality || mediaType,
    explanation: data.explanation || null,
    heatmap_url: null,
    demo: false,
    modalities_used: data.modalities_used || [],
    fusion_method: data.fusion_method || null,
    report_path: data.report_path || null,
  }
}

export async function analyzeMedia(selectedFile, signal) {
  if (!selectedFile?.file) {
    throw new Error('Please select a file before analysis.')
  }

  const formData = new FormData()
  formData.append('file', selectedFile.file)

  let response

  try {
    response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
      signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') {
      throw error
    }

    throw new Error(
      `Cannot connect to the backend at ${API_BASE_URL}. ` +
      'Ensure FastAPI is running.'
    )
  }

  if (!response.ok) {
    let message = `Analysis failed (HTTP ${response.status}).`

    try {
      const errorData = await response.json()

      if (typeof errorData.detail === 'string') {
        message = errorData.detail
      } else if (Array.isArray(errorData.detail)) {
        message = errorData.detail
          .map(item => item.msg || 'Validation error')
          .join('; ')
      }
    } catch {
      // Keep the original HTTP error message.
    }

    throw new Error(message)
  }

  let data

  try {
    data = await response.json()
  } catch {
    throw new Error('Backend returned invalid JSON.')
  }

  return normalizeResult(data, selectedFile.mediaType)
}
