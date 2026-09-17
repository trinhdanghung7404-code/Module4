/**
 * Một ô nhập + nhãn + chỗ hiện lỗi.
 *
 * Tach tu CheckoutPage ra de trang tai khoan dung chung: form nguoi nhan o hai
 * noi giong het ve kieu dang va ve CSS, nam trong hai file thi the nao cung lech
 * nhau sau vai lan sua.
 */
export function Hint({ text }) {
  return text ? <span className="field__hint">{text}</span> : null;
}

export default function Field({ id, label, value, onChange, error, hint, ...inputProps }) {
  return (
    <label className="field" htmlFor={id}>
      <span className="field__label">{label}</span>
      <input
        id={id}
        className="field__input"
        value={value}
        onChange={onChange}
        aria-invalid={error ? "true" : undefined}
        aria-describedby={error ? `${id}-error` : undefined}
        {...inputProps}
      />
      {error ? (
        <span className="field__error" id={`${id}-error`}>
          {error}
        </span>
      ) : (
        <Hint text={hint} />
      )}
    </label>
  );
}
