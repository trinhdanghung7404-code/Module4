const validate = (schema) => (req, res, next) => {
  const result = schema.safeParse(req.body)
  if (!result.success) {
    const issues = result.error.issues || result.error.errors || []
    return res.status(400).json({
      success: false,
      message: 'Validation error',
      errors: issues.map(e => ({
        field: e.path[0],
        message: e.message
      }))
    })
  }
  req.body = result.data
  next()
}

module.exports = validate