import { useMemo, useSyncExternalStore } from "react";
import { cartStore } from "../utils/cartStore";

function getServerSnapshot() {
  return [];
}

export function useCart() {
  const lines = useSyncExternalStore(
    cartStore.subscribe,
    cartStore.getSnapshot,
    getServerSnapshot
  );

  const totals = useMemo(() => {
    const subtotal = lines.reduce(
      (sum, line) => sum + Number(line.price || 0) * line.quantity,
      0
    );
    const count = lines.reduce((sum, line) => sum + line.quantity, 0);
    return { subtotal, count };
  }, [lines]);

  return {
    lines,
    ...totals,
    add: cartStore.add,
    setQuantity: cartStore.setQuantity,
    remove: cartStore.remove,
    clear: cartStore.clear,
    syncWithProducts: cartStore.syncWithProducts,
  };
}
