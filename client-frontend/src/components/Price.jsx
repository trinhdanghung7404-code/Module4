import { formatPrice } from "../utils/format";

/**
 * Renders an em dash instead of "NaN ₫" when the API omits price. Cheap guard, but
 * it keeps a malformed row from breaking the whole grid layout.
 */
export default function Price({ value, className = "" }) {
  const amount = Number(value);

  if (!Number.isFinite(amount)) {
    return <span className={`price price--empty ${className}`.trim()}>—</span>;
  }

  return <span className={`price ${className}`.trim()}>{formatPrice(amount)}</span>;
}
