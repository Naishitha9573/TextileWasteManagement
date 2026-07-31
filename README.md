# Textile Waste Intelligence Platform — Startup Guide

## Why "Server Connection Failed" Happens
This error appears when the **backend FastAPI server is not running**.  
The frontend (React) cannot call `/api/...` endpoints if the Python server is offline.

You must **always start both servers** before opening the browser.

---

## How to Start the Platform (2 Steps)

### Step 1 — Start the Backend (FastAPI)

Open a **new PowerShell / CMD window** and run:

```powershell
cd "C:\Users\sama\OneDrive\Desktop\Textile\Backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Expected output:**
```
INFO:  Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:  Started reloader process ...
```

Leave this terminal **open and running**.

---

### Step 2 — Start the Frontend (React + Vite)

Open a **second PowerShell / CMD window** and run:

```powershell
cd "C:\Users\sama\OneDrive\Desktop\Textile\Frontend"
npm run dev
```

**Expected output:**
```
  VITE v8.1.5  ready in 1407 ms
  ➜  Local:   http://localhost:5173/
```

Leave this terminal **open and running**.

---

## Open the App

Go to your browser and open:
**http://localhost:5173**

---

## Demo Login Credentials

| Username         | Password             | Role                       |
|------------------|----------------------|----------------------------|
| `recycler`       | `recycler123`        | Recycling Facility Operator |
| `sustainability` | `sustainability123`  | Sustainability Manager      |
| `manufacturer`   | `manufacturer123`    | Textile Manufacturer        |
| `admin`          | `admin123`           | Administrator               |

---

## API Documentation (Swagger)

While the backend is running, visit:
**http://127.0.0.1:8000/docs**

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Server Connection Failed" on login | Backend is not running. Do Step 1 above. |
| Browser shows blank page | Frontend is not running. Do Step 2 above. |
| Port 8000 already in use | Kill existing process: `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` |
| Port 5173 already in use | Kill with: `netstat -ano \| findstr :5173` then `taskkill /PID <pid> /F` |
| Login returns 401 Unauthorized | Database may be missing users. Delete `Backend/textile_waste.db` and restart backend — it auto-seeds users. |

---

## Docker (Optional — runs everything together)

If you have Docker Desktop running:

```powershell
cd "C:\Users\sama\OneDrive\Desktop\Textile"
docker-compose up --build
```

Access:  
- Frontend: http://localhost:3000  
- Backend API: http://localhost:8000/docs
