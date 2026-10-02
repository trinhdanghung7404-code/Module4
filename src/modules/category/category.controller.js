const  categoryService = require('./category.service');

const create = async (req, res, next) => {
    try {
        const category = await categoryService.create(req.body);
        res.status(201).json({ success: true, data: category })
    } catch (err) {
        next(err);
    }
}

const getAll = async (req, res, next) => {
    try {
        const categories = await categoryService.getAll();
        res.json({ success: true, data: categories });
    } catch (err) {
        next(err);
    }
}

const getById = async (req, res, next) => {
    try {
        const category = await categoryService.getById(req.params.id);
        res.json({ success: true, data: category });
    } catch (err) {
        next(err);
    }
}

const updateById = async (req, res, next) => {
    try {
        const category = await categoryService.updateById(req.params.id, req.body);
        res.json({ success: true, data: category });
    } catch (err) {
        next(err);
    }
}

const remove = async (req, res, next) => {
    try {
        await categoryService.remove(req.params.id);
        res.json({ success: true, message: 'Danh mục đã được xóa' });
    } catch (err) {
        next(err);
    }
}

module.exports = { create, getAll, getById, updateById, remove };