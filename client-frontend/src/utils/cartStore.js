const STORAGE_KEY = "shop.cart.v1";

/**
 * Cart lives in localStorage only: the backend has no Order or cart endpoint yet,
 * so a server-side cart would have nowhere to be saved.
 *
 * Each line stores a snapshot (name, price, image) next to the product id. That is
 * deliberate: it keeps the cart renderable when a product was deleted or the API is
 * down, but it also means a price change in the admin is NOT reflected in an existing
 * cart until syncWithProducts() runs. Never treat these numbers as final — the price
 * shown at checkout time must come from a fresh product fetch.
 */

function toPositiveInt(value) {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

function normalizeLine(raw) {
  if (!raw || typeof raw !== "object") return null;

  const productId = toPositiveInt(raw.productId);
  if (!productId) return null;

  let quantity = toPositiveInt(raw.quantity) ?? 1;
  const stock = toPositiveInt(raw.stock);
  if (stock && quantity > stock) quantity = stock;

  return {
    productId,
    name: typeof raw.name === "string" ? raw.name : `Sản phẩm #${productId}`,
    price: Number.isFinite(Number(raw.price)) ? Number(raw.price) : 0,
    imageUrl: typeof raw.imageUrl === "string" ? raw.imageUrl : null,
    quantity,
    stock,
  };
}

function readStorage() {
  if (typeof window === "undefined") return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.map(normalizeLine).filter(Boolean);
  } catch {
    // Corrupted or tampered JSON: an empty cart is a safe recovery, and losing a
    // demo cart is cheaper than crashing the whole app on first paint.
    return [];
  }
}

let lines = readStorage();
const listeners = new Set();

function notify({ persist }) {
  if (persist) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(lines));
    } catch {
      // Private mode / quota exceeded: the cart still works for this tab session.
    }
  }
  listeners.forEach((listener) => listener());
}

function findIndex(productId) {
  return lines.findIndex((line) => line.productId === productId);
}

export const cartStore = {
  wiredStorageEvents: false,

  subscribe(listener) {
    listeners.add(listener);

    if (!cartStore.wiredStorageEvents && typeof window !== "undefined") {
      cartStore.wiredStorageEvents = true;
      // Keeps two open tabs in sync. The write is skipped here, otherwise every tab
      // would echo the change back and the storage event would never settle.
      window.addEventListener("storage", (event) => {
        if (event.key !== STORAGE_KEY) return;
        lines = readStorage();
        notify({ persist: false });
      });
    }

    return () => listeners.delete(listener);
  },

  getSnapshot: () => lines,

  add(product, quantity = 1) {
    const productId = toPositiveInt(product?.id ?? product?.productId);
    if (!productId) return { ok: false, reason: "Sản phẩm không hợp lệ." };

    const requested = toPositiveInt(quantity) ?? 1;
    const stock = toPositiveInt(product?.stock ?? product?.quantity);
    const index = findIndex(productId);
    const current = index >= 0 ? lines[index].quantity : 0;

    if (stock && current + requested > stock) {
      if (current >= stock) {
        return { ok: false, reason: `Chỉ còn ${stock} sản phẩm trong kho.` };
      }
      const capped = [...lines];
      capped[index] = { ...capped[index], quantity: stock, stock };
      lines = capped;
      notify({ persist: true });
      return { ok: true, clampedTo: stock };
    }

    const next = [...lines];
    if (index >= 0) {
      next[index] = { ...next[index], quantity: current + requested, stock };
    } else {
      next.push(
        normalizeLine({
          productId,
          name: product.name,
          price: product.price,
          imageUrl: product.imageUrl ?? product.images?.[0]?.imageUrl ?? null,
          quantity: requested,
          stock,
        })
      );
    }

    lines = next;
    notify({ persist: true });
    return { ok: true };
  },

  setQuantity(productId, quantity) {
    const index = findIndex(toPositiveInt(productId));
    if (index < 0) return;

    const target = toPositiveInt(quantity);
    if (!target) {
      // An empty or nonsense value arrives while the user is mid-edit in the number
      // input. Dropping the line here would delete a product because of a keystroke,
      // so it is ignored; removal has its own explicit button.
      return;
    }

    const stock = lines[index].stock;
    const capped = stock && target > stock ? stock : target;
    const next = [...lines];
    next[index] = { ...next[index], quantity: capped };
    lines = next;
    notify({ persist: true });
  },

  remove(productId) {
    const target = toPositiveInt(productId);
    lines = lines.filter((line) => line.productId !== target);
    notify({ persist: true });
  },

  clear() {
    lines = [];
    notify({ persist: true });
  },

  /**
   * Refreshes cached snapshots from live product data and clamps quantities to stock.
   *
   * `pruneMissing` defaults to false on purpose. A paginated grid only knows the 12
   * products on screen; pruning against that subset would silently delete every cart
   * line the user is not currently looking at. Only a caller that fetched exactly the
   * cart's own product ids (CartPage) may pass true.
   *
   * Returns the ids that were dropped, so the caller can tell the user what happened.
   */
  syncWithProducts(products, { pruneMissing = false } = {}) {
    if (!Array.isArray(products) || products.length === 0) return [];

    const byId = new Map(products.map((p) => [Number(p.id), p]));
    const missing = [];
    let changed = false;

    const next = lines
      .map((line) => {
        const product = byId.get(line.productId);
        if (!product) {
          missing.push(line.productId);
          return line;
        }


        const stock = toPositiveInt(product.quantity);
        const quantity = stock && line.quantity > stock ? stock : line.quantity;
        const price = Number.isFinite(Number(product.price))
          ? Number(product.price)
          : line.price;
        const imageUrl = product.imageUrl ?? line.imageUrl;
        const name = product.name ?? line.name;

        if (
          quantity !== line.quantity ||
          price !== line.price ||
          imageUrl !== line.imageUrl ||
          name !== line.name ||
          stock !== line.stock
        ) {
          changed = true;
        }

        return { ...line, name, price, imageUrl, quantity, stock };
      })
      .filter((line) => !pruneMissing || !missing.includes(line.productId));

    if (changed || (pruneMissing && missing.length > 0)) {
      lines = next;
      notify({ persist: true });
    }

    return pruneMissing ? missing : [];
  },
};

