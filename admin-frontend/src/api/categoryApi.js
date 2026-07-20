const API_URL =
  import.meta.env.VITE_API_URL ??
  "http://localhost:8080/api/admin";

async function handleResponse(response) {
  const text = await response.text();

  let data;

  try {
    data = JSON.parse(text);
  } catch {
    data = text;
  }

  if (!response.ok) {
    throw new Error(
      data?.message ||
        (typeof data === "object"
          ? Object.values(data)[0]
          : data) ||
        "Có lỗi xảy ra"
    );
  }

  return data;
}

export async function getCategories() {
  const response = await fetch(`${API_URL}/categories`);
  return handleResponse(response);
}

export async function createCategory(name) {
  const response = await fetch(`${API_URL}/categories`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ name }),
  });

  return handleResponse(response);
}