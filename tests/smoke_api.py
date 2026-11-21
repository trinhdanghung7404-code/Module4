import threading
import time
import requests
import uvicorn
import app as m


def main():
    cfg = uvicorn.Config(m.app, host="127.0.0.1", port=8099, log_level="warning")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    time.sleep(1.5)
    base = "http://127.0.0.1:8099"
    try:
        for pid in ["B1", "B22"]:
            s = requests.post(f"{base}/api/session", data={"product_id": pid}).json()
            sid = s["session_id"]
            with open(f"data/reference_images/{pid}/front.jpg", "rb") as f:
                u = requests.post(
                    f"{base}/api/upload/{sid}",
                    data={"view": "front"},
                    files={"image": ("front.jpg", f, "image/jpeg")},
                )
            p = requests.post(f"{base}/api/session/{sid}/preprocess").json()
            d = requests.post(f"{base}/api/session/{sid}/detect").json()
            pf = p["results"]["front"]
            df = d["results"]["front"]
            print(
                f"{pid} (sid={sid}): upload={u.status_code} | "
                f"ignore_ratio={pf['ignore_ratio']}% reshoot={pf['should_reshoot']} | "
                f"detect={df['severity']} ({df['defect_count']} defects)"
            )
    finally:
        srv.should_exit = True


if __name__ == "__main__":
    main()
