import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred'
    return Promise.reject(new Error(message))
  }
)

/**
 * Upload images to a collection.
 * @param {File[]} files
 * @param {string|null} collectionId
 */
export async function uploadImages(files, collectionId = null) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  if (collectionId) {
    formData.append('collection_id', collectionId)
  }
  const response = await api.post('/api/images/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return response.data
}

/**
 * Browse all images with optional filters (gallery mode).
 * @param {object|null} filters - e.g. { layout_type: ["dashboard"] }
 * @param {number} limit
 */
export async function listImages(filters = null, limit = 200) {
  const params = { limit }
  const active = filters
    ? Object.fromEntries(Object.entries(filters).filter(([, v]) => v?.length > 0))
    : {}
  if (Object.keys(active).length > 0) {
    params.filters = JSON.stringify(active)
  }
  const response = await api.get('/api/images', { params })
  return response.data
}

/**
 * Search images by text query.
 * @param {string} query
 * @param {number} k
 * @param {object|null} filters
 */
export async function searchText(query, k = 20, filters = null) {
  const params = { q: query, k }
  const active = filters
    ? Object.fromEntries(Object.entries(filters).filter(([, v]) => v?.length > 0))
    : {}
  if (Object.keys(active).length > 0) {
    params.filters = JSON.stringify(active)
  }
  const response = await api.get('/api/search/text', { params })
  return response.data
}

/**
 * Search for visually similar images.
 * @param {string} imageId
 * @param {number} k
 */
export async function searchSimilar(imageId, k = 20) {
  const response = await api.get(`/api/search/similar/${imageId}`, { params: { k } })
  return response.data
}

/**
 * Hybrid search combining dense and BM25 retrieval.
 * @param {string} query
 * @param {number} k
 * @param {number} alpha - weight for dense (0=BM25 only, 1=dense only)
 */
export async function searchHybrid(query, k = 20, alpha = 0.7) {
  const response = await api.get('/api/search/hybrid', { params: { q: query, k, alpha } })
  return response.data
}

/**
 * Get image metadata by ID.
 * @param {string} imageId
 */
export async function getImage(imageId) {
  const response = await api.get(`/api/images/${imageId}`)
  return response.data
}

/**
 * Delete an image by ID.
 * @param {string} imageId
 */
export async function deleteImage(imageId) {
  const response = await api.delete(`/api/images/${imageId}`)
  return response.data
}

/**
 * Get system statistics.
 */
export async function getStats() {
  const response = await api.get('/stats')
  return response.data
}

/**
 * Health check.
 */
export async function getHealth() {
  const response = await api.get('/health')
  return response.data
}

export default api
