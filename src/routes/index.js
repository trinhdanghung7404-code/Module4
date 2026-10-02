const express = require('express');
const rateLimit = require('express-rate-limit');
const authRoute = require('../modules/auth/auth.route');
const categoryRoute = require('../modules/category/category.route');
const router = express.Router();

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 20,
  message: { success: false, message: 'Quá nhiều request, thử lại sau' }
});

router.use('/auth', authLimiter, authRoute);
router.use('/categories', categoryRoute);

module.exports = router;
