import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const acceptedTypes = ['image/jpeg', 'image/png', 'image/webp']

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [predictions, setPredictions] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  function chooseFile(nextFile) {
    setError('')
    setPredictions([])
    if (!nextFile) return
    if (!acceptedTypes.includes(nextFile.type)) {
      setFile(null)
      setPreview('')
      setError('Please select a JPG, PNG, or WebP image.')
      return
    }
    if (nextFile.size > 8 * 1024 * 1024) {
      setError('Images must be 8 MB or smaller.')
      return
    }
    setFile(nextFile)
    setPreview(URL.createObjectURL(nextFile))
  }

  async function classify() {
    if (!file) return
    setLoading(true)
    setError('')
    setPredictions([])
    const payload = new FormData()
    payload.append('image', file)
    try {
      const response = await fetch('/api/classify', { method: 'POST', body: payload })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Something went wrong.')
      setPredictions(data.predictions)
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the classifier API.')
    } finally {
      setLoading(false)
    }
  }

  function onDrop(event) {
    event.preventDefault()
    chooseFile(event.dataTransfer.files[0])
  }

  return (
    <main>
      <header>
        <a className="brand" href="/" aria-label="Focus home"><span>✦</span> FOCUS</a>
        <p>Techviman Image Classifier</p>
      </header>
      <section className="hero">
        <div className="eyebrow">MOBILENET V3 · IMAGENET</div>
        <h1>See what your<br /><em>images know.</em></h1>
        <p className="intro">Drop in an image. Our model looks closer, then returns the objects it recognizes with a clear confidence score.</p>
      </section>

      <section className="workspace" aria-label="Image classifier">
        <div
          className={`dropzone ${preview ? 'has-preview' : ''}`}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex="0"
          onKeyDown={(event) => event.key === 'Enter' && inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => chooseFile(event.target.files[0])} />
          {preview ? <img src={preview} alt="Selected image preview" /> : <><div className="upload-icon">↑</div><strong>Drop your image here</strong><span>or click to browse</span><small>JPG, PNG or WebP · up to 8 MB</small></>}
          {preview && <div className="change-image">Change image</div>}
        </div>

        <div className="results">
          <div className="results-heading"><span>ANALYSIS</span><span>{predictions.length ? 'COMPLETE' : 'WAITING'}</span></div>
          {error && <div className="message error">{error}</div>}
          {!predictions.length && !error && <div className="empty"><span>◌</span><p>Your predictions will appear here.</p></div>}
          {predictions.map((prediction, index) => <div className="prediction" key={prediction.label}>
            <div><span className="rank">0{index + 1}</span><strong>{prediction.label}</strong></div>
            <div className="score"><span>{prediction.confidence}%</span><i><b style={{ width: `${prediction.confidence}%` }} /></i></div>
          </div>)}
          <button className="classify" disabled={!file || loading} onClick={classify}>
            {loading ? 'ANALYZING…' : 'ANALYZE IMAGE'} <span>→</span>
          </button>
        </div>
      </section>
      <footer><span>Developed by TECHVIMAN</span><span>Images are processed for your request and never stored.</span></footer>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>)
