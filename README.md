# 🛡️ Warranty Fraud Detector

An AI-powered, multi-agent invoice verification and warranty fraud detection platform. It uses image forensics, fuzzy logic matching, and LLMs (Google Gemini) to analyze uploaded invoices (images or PDFs), detect tampering or fraud, and produce automated approval decisions.

---

## 🏗️ Architecture & Verification Pipeline

```
          [ Upload PDF or Image ]
                     │
                     ▼
          [ Frontend (Streamlit) ]
                     │
                     ▼
         [ FastAPI Backend (:8000) ]
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[ 🖼️ Forensics Engine ]     [ 📝 Gemini OCR Agent ]
 ├── ELA Tampering Check     ├── Invoice # & Amount
 ├── Copy-Move Keypoints     ├── Product & Seller
 └── EXIF Metadata Audit     └── Full Raw Text
       └─────────────┬─────────────┘
                     ▼
     [ ⚙️ Validation Pipeline Agents ]
      ├── 📦 Product Match (Fuzz fallback)
      ├── 🏪 Seller Check (Fuzz lookup)
      ├── 📅 Date Validation (Warranty window)
      └── 🔁 Duplicate Check (Invoice No matching)
                     │
                     ▼
         [ ⚖️ Risk engine scoring ]
                     │
                     ▼
        [ 🤖 Gemini AI Decision Agent ]
                     │
                     ▼
       [ 📋 SQLite Database / UI ]
```

---

## ✨ Key Features

- **📄 Full PDF Support & Preview**: Accepts PDFs alongside JPEG/PNG invoices. Renders a clean PDF preview (first page) on the frontend using PyMuPDF.
- **🔬 Advanced Image Forensics**:
  - **Error Level Analysis (ELA)**: Recompresses images to identify pixel-level alterations.
  - **Copy-Move Forgery Detection**: Detects copied-and-pasted text/regions using ORB keypoint feature matching.
  - **EXIF Metadata Audit**: Checks for editing tool fingerprints (e.g. Photoshop, Canva) without penalizing screenshots (which contain no EXIF data).
- **🧠 Fuzzy OCR Validation**: Product name and seller validation use `rapidfuzz` (`token_set_ratio` & `partial_ratio`) fallback against raw OCR text. If the structured OCR parser fails to extract fields but the items are found in the raw text, it resolves successfully.
- **🏪 Expanded Marketplace Support**: Native authorized-seller support for **Blinkit** (including matching `"BLINK COMMERCE PRIVATE LIMITED"`) alongside Amazon, Flipkart, Croma, and Reliance Digital.
- **🔄 Smart Duplicate Logic**: Flags duplicate attempts based strictly on the `invoice_number` in the database, allowing users to register multiple different purchases under the same email and mobile.
- **🗑️ Registration Management**: Frontend includes a "🗑️ Delete Registration" button inside expanders to manage/remove entries dynamically.

---

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have Python installed, then install the package requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create or edit `.env` in the root folder:
```ini
# Gemini API Key (Required for OCR and Decision Agent)
GEMINI_API_KEY=AIzaSy...

# Optional Configurations
WARRANTY_MONTHS=12
DATABASE_URL=sqlite:///./warranty_fraud.db
```

### 3. Start the FastAPI backend
```bash
uvicorn app.main:app --reload
```

### 4. Start the Streamlit UI
```bash
streamlit run app/frontend/app.py
```

*Or use the one-command startup script (Linux/macOS):*
```bash
bash app/frontend/start.sh
```

---

## 🌐 Services Summary

| Service | URL | Description |
|---|---|---|
| **Streamlit Web UI** | `http://localhost:8501` | Main User Dashboard & Registrations List |
| **FastAPI Backend** | `http://localhost:8000` | Orchestration Engine API |
| **API Docs (Swagger)** | `http://localhost:8000/docs` | Interactive Swagger API documentation |
| **Registrations List**| `http://localhost:8000/registrations` | Fetch recent database entries |

---

## ⚖️ Risk Score & Decision Matrix

| Agent | Max Risk | Validation Metrics |
|---|---|---|
| **ELA** | 25 | Localized pixel differences & compression |
| **Metadata** | 20 | EXIF editing software fingerprints (0 risk for screenshots) |
| **Copy-Move** | 25 | Keypoint cluster matches across the image |
| **Product Match** | 20 | Fuzzy token matching (set to 0 risk if single word matches) |
| **Seller Valid** | 15 | Authorized seller validation (0 risk for 'Other' marketplaces) |
| **Date Valid** | 15 | Expired warranty window or future dates |
| **Duplicate** | 40 | Exact invoice number duplicate check in the database |
| **Total Capped** | **100** | |

### Decision Thresholds
- **APPROVE**: Risk Score < `20`
- **MANUAL REVIEW**: Risk Score `20` - `50`
- **REJECT**: Risk Score > `50`

---

## 🧪 Running Unit Tests
Execute the unit test suite covering forensics, OCR validation, fuzzy checks, duplicate checks, and deletion:
```bash
pytest app/testing/test_agents.py -v
```