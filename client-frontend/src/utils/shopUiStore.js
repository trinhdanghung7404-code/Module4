import { fetchCategories } from "../api/shopApi";

/**
 * Small external store for the shared shop state: the category list plus the
 * current filter/page. Components read it through useSyncExternalStore (see
 * hooks/useShopUi.js), which keeps the header and the grid in sync without a
 * context provider and without "setState inside an effect" cascades — the store
 * notifies subscribers from async callbacks only.
 */

const DEFAULT_FILTER = { categoryId: 0, search: "", sort: "newest" };

const listeners = new Set();

let snapshot = {
  categories: [],
  categoriesStatus: "idle",
  categoriesError: null,
  filter: DEFAULT_FILTER,
  page: 0,
};

let inflight = null;

function set(patch) {
  snapshot = { ...snapshot, ...patch };
  listeners.forEach((listener) => listener());
}

export const shopUiStore = {
  subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },

  getSnapshot: () => snapshot,

  setFilter(patch) {
    const previous = snapshot.filter;
    const changed = Object.keys(patch).some((key) => patch[key] !== previous[key]);
    if (!changed) return;

    snapshot = {
      ...snapshot,
      filter: { ...previous, ...patch },
      // A new filter set makes the current page number meaningless.
      page: 0,
    };
    listeners.forEach((listener) => listener());
  },

  setPage(page) {
    if (page === snapshot.page) return;
    set({ page });
  },

  resetFilters() {
    this.setFilter({ ...DEFAULT_FILTER });
  },

  /** Idempotent: safe to call from every mounted view; only one request goes out. */
  loadCategories(force = false) {
    if (inflight) return inflight;
    if (!force && snapshot.categoriesStatus === "ready") {
      return Promise.resolve(snapshot.categories);
    }

    set({ categoriesStatus: "loading", categoriesError: null });

    // Never rejects: a failed catalog load is surfaced through state and the
    // header's "Thử lại" button, not as an unhandled promise rejection.
    inflight = fetchCategories()
      .then((data) => {
        inflight = null;
        set({
          categories: Array.isArray(data) ? data : [],
          categoriesStatus: "ready",
          categoriesError: null,
        });
        return data;
      })
      .catch((error) => {
        inflight = null;
        set({
          categories: [],
          categoriesStatus: "error",
          categoriesError: error.message,
        });
        return null;
      });

    return inflight;
  },
};
