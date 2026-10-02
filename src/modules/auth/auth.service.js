const bcrypt = require('bcryptjs')
const jwt = require('jsonwebtoken')
const prisma = require('../../config/db')

const generateTokens = (userId, role) => {
  const accessToken = jwt.sign(
    { userId, role },
    process.env.JWT_ACCESS_SECRET,
    { expiresIn: process.env.JWT_ACCESS_EXPIRES }
  )
  const refreshToken = jwt.sign(
    { userId },
    process.env.JWT_REFRESH_SECRET,
    { expiresIn: process.env.JWT_REFRESH_EXPIRES }
  )
  return { accessToken, refreshToken }
}

const register = async ({ name, email, password }) => {
  const existing = await prisma.user.findUnique({ where: { email } })
  if (existing) {
    const error = new Error('Email đã được sử dụng')
    error.status = 409
    throw error
  }

  const hashed = await bcrypt.hash(password, 10)
  const user = await prisma.user.create({
    data: { name, email, password: hashed }
  })

  return { id: user.id, name: user.name, email: user.email, role: user.role }
}

const login = async ({ email, password }) => {
  const user = await prisma.user.findUnique({ where: { email } })
  if (!user) {
    const error = new Error('Email hoặc mật khẩu không đúng')
    error.status = 401
    throw error
  }

  const isMatch = await bcrypt.compare(password, user.password)
  if (!isMatch) {
    const error = new Error('Email hoặc mật khẩu không đúng')
    error.status = 401
    throw error
  }

  const { accessToken, refreshToken } = generateTokens(user.id, user.role)

  // Lưu refresh token vào DB
  await prisma.refreshToken.create({
    data: {
      token: refreshToken,
      userId: user.id,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
    }
  })

  return {
    user: { id: user.id, name: user.name, email: user.email, role: user.role },
    accessToken,
    refreshToken
  }
}

const refresh = async (token) => {
  if (!token) {
    const error = new Error('Không có refresh token')
    error.status = 401
    throw error
  }

  const stored = await prisma.refreshToken.findUnique({ where: { token } })
  if (!stored || stored.expiresAt < new Date()) {
    const error = new Error('Refresh token không hợp lệ hoặc đã hết hạn')
    error.status = 401
    throw error
  }

  const payload = jwt.verify(token, process.env.JWT_REFRESH_SECRET)
  const user = await prisma.user.findUnique({ where: { id: payload.userId } })

  const { accessToken, refreshToken: newRefreshToken } = generateTokens(user.id, user.role)

  // Xóa token cũ, lưu token mới (Refresh Token Rotation)
  await prisma.refreshToken.delete({ where: { token } })
  await prisma.refreshToken.create({
    data: {
      token: newRefreshToken,
      userId: user.id,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
    }
  })

  return { accessToken, refreshToken: newRefreshToken }
}

const logout = async (token) => {
  if (!token) return
  await prisma.refreshToken.deleteMany({ where: { token } })
}

module.exports = { register, login, refresh, logout }