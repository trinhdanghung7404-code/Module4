const API_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8080/api/admin";

function getErrorMessage(data) {
  if (typeof data === "string") {
    return data || "Có lỗi xảy ra";
  }

  if (data?.message) {
    return data.message;
  }

  if (data && typeof data === "object") {
    return Object.values(data)[0] ?? "Có lỗi xảy ra";
  }

  return "Có lỗi xảy ra";
}

async function post(path, payload) {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const text = await response.text();
  let data = text;

  try {
    data = JSON.parse(text);
  } catch {
    // Backend hiện trả chuỗi ở trường hợp thành công.
  }

  if (!response.ok) {
    throw new Error(getErrorMessage(data));
  }

  return data;
}

export function loginAdmin(payload) {
  return post("/login", payload);
}

export function registerAdmin(payload) {
  return post("/register", payload);
}
