const { z } = require('zod')

const createCategorySchema = z.object({
  name: z.string().min(3, 'Tên danh mục ít nhất 3 ký tự'),
  description: z.string().optional()
})

const updateCategorySchema = z.object({
  name: z.string().min(3, 'Tên danh mục ít nhất 3 ký tự').optional(),
  description: z.string().optional()
})

module.exports = { createCategorySchema, updateCategorySchema }