import { Navigate, Outlet } from "react-router-dom";
import { isLoggedIn } from "../utils/adminSession";

function ProtectedRoute() {
  return isLoggedIn()
    ? <Outlet />
    : <Navigate to="/login" replace />;
}

export default ProtectedRoute;