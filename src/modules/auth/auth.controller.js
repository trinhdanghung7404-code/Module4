const authService = require('./auth.service')

const register = async (req, res, next) => {
  try {
    const user = await authService.register(req.body)
    res.status(201).json({ success: true, data: user })
  } catch (err) {
    next(err)
  }
}

const login = async (req, res, next) => {
  try {
    const result = await authService.login(req.body)
    res.json({ success: true, data: result })
  } catch (err) {
    next(err)
  }
}

const refresh = async (req, res, next) => {
  try {
    const { refreshToken } = req.body
    const result = await authService.refresh(refreshToken)
    res.json({ success: true, data: result })
  } catch (err) {
    next(err)
  }
}

const logout = async (req, res, next) => {
  try {
    const { refreshToken } = req.body
    await authService.logout(refreshToken)
    res.json({ success: true, message: 'Đăng xuất thành công' })
  } catch (err) {
    next(err)
  }
}

module.exports = { register, login, refresh, logout }