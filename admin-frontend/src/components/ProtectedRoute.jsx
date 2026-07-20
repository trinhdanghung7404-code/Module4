import { Navigate, Outlet } from "react-router-dom";

function ProtectedRoute() {
  const isLoggedIn =
    localStorage.getItem("adminLoggedIn") === "true";

  return isLoggedIn
    ? <Outlet />
    : <Navigate to="/login" replace />;
}

export default ProtectedRoute;