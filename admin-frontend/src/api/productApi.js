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
    const message =
      data?.message ||
      (typeof data === "object" && data !== null
        ? Object.values(data)[0]
        : data) ||
      "Có lỗi xảy ra";

    throw new Error(message);
  }

  return data;
}

export async function getProducts() {
  const response = await fetch(`${API_URL}/products`);
  return handleResponse(response);
}

export async function createProduct(formData) {
  const response = await fetch(`${API_URL}/products`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(response);
}
