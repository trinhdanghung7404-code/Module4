import { useEffect, useSyncExternalStore } from "react";
import { shopUiStore } from "../utils/shopUiStore";

const EMPTY = [];

function getServerSnapshot() {
  return shopUiStore.getSnapshot();
}

export function useShopUi() {
  const snapshot = useSyncExternalStore(
    shopUiStore.subscribe,
    shopUiStore.getSnapshot,
    getServerSnapshot
  );

  // Every view asks, the store de-duplicates: see shopUiStore.loadCategories().
  useEffect(() => {
    shopUiStore.loadCategories();
  }, []);

  return {
    ...snapshot,
    setFilter: shopUiStore.setFilter,
    setPage: shopUiStore.setPage,
    resetFilters: shopUiStore.resetFilters,
    reloadCategories: shopUiStore.loadCategories,
  };
}

export function useCategories() {
  const { categories, categoriesStatus, categoriesError, reloadCategories } =
    useShopUi();

  return {
    categories: categories ?? EMPTY,
    status: categoriesStatus,
    error: categoriesError,
    reload: () => reloadCategories(true),
  };
}
