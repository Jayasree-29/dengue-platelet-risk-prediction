

   cd backend
   # Build with Python 3.8.10 (uses Windows Launcher)
   py -3.8 -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt

 **Run the API server:**
    ```bash
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
    ```
    *Note: `0.0.0.0` means the server listens on all network interfaces. You can access it at `http://localhost:8000`.*
   
### frontend
   cd ..
   npm install

   npm run dev

   *The dashboard will be available at `http://localhost:5173`*
