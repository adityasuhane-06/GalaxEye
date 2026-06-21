import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'

export default function ImageUploader({ label, icon, onFileSelect, preview, accept }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0])
    }
  }, [onFileSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept || { 'image/*': ['.png', '.jpg', '.jpeg', '.tif', '.tiff'] },
    maxFiles: 1,
  })

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              key="preview"
              className="upload-preview"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <img src={preview} alt={label} />
              <p style={{
                textAlign: 'center',
                marginTop: '0.5rem',
                fontSize: '0.8rem',
                color: 'var(--color-accent)',
              }}>
                {label} loaded. Click to change.
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{ textAlign: 'center' }}
            >
              <div className="upload-zone-icon">{icon}</div>
              <p style={{ fontWeight: 600 }}>{label}</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                {isDragActive ? 'Drop the file here...' : 'Drag & drop or click to browse'}
              </p>
              <p style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>
                PNG, JPG, or TIFF (512x512 recommended)
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}
