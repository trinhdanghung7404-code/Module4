const router = require('express').Router();
const categoryController = require('./category.controller');
const  validate  = require('../../middlewares/validate.middleware');
const {authenticate, authorize} = require('../../middlewares/auth.middleware');
const { createCategorySchema, updateCategorySchema } = require('./category.validation');

// Any unauthenticated user can view categories
router.get('/', categoryController.getAll);
router.get('/:id', categoryController.getById);

// Only authenticated users can create, update, or delete categories
router.post('/', authenticate, authorize('ADMIN','STAFF','USER'), validate(createCategorySchema), categoryController.create);
router.put('/:id', authenticate, authorize('ADMIN','STAFF','USER'), validate(updateCategorySchema), categoryController.updateById);
router.delete('/:id', authenticate, authorize('ADMIN','STAFF','USER'), categoryController.remove);

module.exports = router;
