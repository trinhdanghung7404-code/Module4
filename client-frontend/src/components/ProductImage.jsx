import { useState } from "react";

/**
 * Two failure modes have to be handled here, and the backend exposes both:
 *   - imageUrl is null   (the product has no FRONT image uploaded yet)
 *   - imageUrl is broken (Cloudinary URL from a deleted or foreign account)
 * A dead <img> collapses the grid, so a missing source and a failed load both fall
 * back to the same neutral tile.
 */
export default function ProductImage({ src, alt, className = "", ratio = "4 / 3" }) {
  const [failed, setFailed] = useState(false);

  if (!src || failed) {
    return (
      <div
        className={`image-fallback ${className}`.trim()}
        style={{ aspectRatio: ratio }}
        role="img"
        aria-label={alt}
      >
        <span>{alt ? alt.charAt(0).toUpperCase() : "?"}</span>
      </div>
    );
  }

  return (
    <img
      className={className}
      src={src}
      alt={alt}
      loading="lazy"
      style={{ aspectRatio: ratio }}
      onError={() => setFailed(true)}
    />
  );
}
