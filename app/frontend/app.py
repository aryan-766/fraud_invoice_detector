"""
frontend/app.py
Streamlit UI for the Warranty Fraud Detection Platform.
Run: streamlit run frontend/app.py
"""
import time
from datetime import date
import html
import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Warranty Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Global ─────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* ─── Typography ──────────────────────────────────── */
h1, h2, h3 { color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
p, label, .stMarkdown { color: #8b949e; }

/* ─── Cards ───────────────────────────────────────── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 0.5rem;
}

/* ─── Decision badges ─────────────────────────────── */
.badge-approve  { background:#0d1117; border:1px solid #238636; color:#3fb950; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1rem; }
.badge-review   { background:#0d1117; border:1px solid #9e6a03; color:#d29922; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1rem; }
.badge-reject   { background:#0d1117; border:1px solid #da3633; color:#f85149; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1rem; }

/* ─── Risk bar ────────────────────────────────────── */
.risk-track {
    background: #21262d;
    border-radius: 8px;
    height: 10px;
    margin: 6px 0 12px;
    overflow: hidden;
}
.risk-fill-low    { height:10px; border-radius:8px; background:#238636; }
.risk-fill-medium { height:10px; border-radius:8px; background:#9e6a03; }
.risk-fill-high   { height:10px; border-radius:8px; background:#da3633; }

/* ─── OCR chip ────────────────────────────────────── */
.ocr-chip {
    display:inline-block;
    background:#21262d;
    border:1px solid #30363d;
    border-radius:6px;
    padding:2px 10px;
    font-size:0.82rem;
    color:#e6edf3;
    margin:2px 4px 2px 0;
}

/* ─── OCR raw text box ────────────────────────────── */
.ocr-raw-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #c9d1d9;
    line-height: 1.5;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
.ocr-confidence-high { color: #3fb950; font-weight: 600; }
.ocr-confidence-low  { color: #d29922; font-weight: 600; }

/* ─── Upload area ─────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 10px !important;
}

/* ─── Inputs ──────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ─── Primary button ──────────────────────────────── */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg,#1f6feb,#388bfd) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 2rem !important;
    width: 100%;
}

/* ─── Divider ─────────────────────────────────────── */
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def risk_bar(score: float) -> str:
    pct = min(score, 100)
    cls = "risk-fill-low" if pct < 30 else ("risk-fill-medium" if pct < 60 else "risk-fill-high")
    return f"""
    <div class="risk-track">
        <div class="{cls}" style="width:{pct}%;"></div>
    </div>"""


def badge(decision: str) -> str:
    cls_map = {"APPROVE": "badge-approve", "MANUAL_REVIEW": "badge-review", "REJECT": "badge-reject"}
    label_map = {"APPROVE": "✅ APPROVED", "MANUAL_REVIEW": "⚠️ MANUAL REVIEW", "REJECT": "❌ REJECTED"}
    cls   = cls_map.get(decision, "badge-review")
    label = label_map.get(decision, decision)
    return f'<span class="{cls}">{label}</span>'


def render_ocr_data(ocr: dict, title: str = "📄 OCR Extracted Data") -> None:
    """Display structured OCR fields, confidence, and raw text from API response."""
    st.markdown(f"**{title}**")
    confidence = ocr.get("ocr_confidence", 0.0)
    conf_cls = "ocr-confidence-high" if confidence >= 0.5 else "ocr-confidence-low"
    model = ocr.get("ocr_model") or "—"
    conf_label = model if confidence >= 0.5 else "extraction failed"
    st.markdown(
        f"**OCR Confidence:** <span class='{conf_cls}'>{confidence:.0%}</span> "
        f"<small style='color:#8b949e'>(model: {model})</small>",
        unsafe_allow_html=True,
    )

    if ocr.get("error"):
        st.error(f"OCR error: {ocr['error']}")

    chips = ""
    for label, val in [
        ("Invoice #", ocr.get("invoice_number")),
        ("Seller",    ocr.get("seller")),
        ("Product",   ocr.get("product")),
        ("Date",      ocr.get("purchase_date")),
        ("Amount",    ocr.get("amount")),
    ]:
        v = val or "—"
        chips += f'<span class="ocr-chip"><b>{label}:</b> {v}</span>'
    st.markdown(chips, unsafe_allow_html=True)

    raw_text = html.escape((ocr.get("raw_text") or "").strip())
    if raw_text:
        st.markdown("**📝 Raw Text from Invoice**")
        st.markdown(f'<div class="ocr-raw-box">{raw_text}</div>', unsafe_allow_html=True)
    else:
        st.caption("No raw text extracted from invoice.")

    if ocr.get("error"):
        st.error(f"OCR error: {ocr['error']}")


def run_ocr_preview(invoice_file) -> dict | None:
    """Call /ocr API and return extracted data."""
    mime = invoice_file.type or "application/octet-stream"
    files = {"invoice": (invoice_file.name or "invoice.jpg", invoice_file.getvalue(), mime)}
    try:
        resp = requests.post(f"{API_BASE}/ocr", files=files, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"OCR API error {resp.status_code}: {resp.text[:300]}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure FastAPI is running.")
    except Exception as ex:
        st.error(f"OCR failed: {ex}")
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛡️ Warranty Fraud Detector")
    st.markdown("---")

    alive = api_health()
    if alive:
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline — start FastAPI first")
        st.code("uvicorn app.main:app --reload", language="bash")

    st.markdown("---")
    page = st.radio("Navigation", ["🔍 Verify Invoice", "📋 Registrations", "ℹ️ How It Works"])
    st.markdown("---")
    st.caption("Built with FastAPI · Gemini OCR · OpenCV · OpenCV Forensics")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Verify Invoice
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔍 Verify Invoice":
    st.markdown("## 🔍 Invoice Verification")
    st.markdown("Submit your purchase details and invoice to register your warranty.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="card"><div class="card-title">📋 Registration Details</div>', unsafe_allow_html=True)

        st.markdown("**Invoice Upload**")
        invoice_file = st.file_uploader(
            "Upload Invoice (JPG, PNG, PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
            help="Max 10 MB",
            key="invoice_upload",
        )

        if invoice_file:
            if invoice_file.type == "application/pdf" or invoice_file.name.lower().endswith(".pdf"):
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(stream=invoice_file.getvalue(), filetype="pdf")
                    if len(doc) > 0:
                        page = doc[0]
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                        img_data = pix.tobytes("png")
                        st.image(img_data, caption="Uploaded PDF Invoice Preview (Page 1)", use_container_width=True)
                    else:
                        st.warning("Uploaded PDF is empty.")
                except Exception as e:
                    st.warning(f"Could not render PDF preview: {e}")
                    st.info(f"📄 PDF Uploaded: {invoice_file.name}")
            else:
                st.image(invoice_file, caption="Uploaded Invoice Preview", use_container_width=True)
            upload_key = f"{invoice_file.name}:{invoice_file.size}"
            if st.session_state.get("ocr_preview_file") != upload_key:
                st.session_state.pop("ocr_preview", None)
                st.session_state["ocr_preview_file"] = upload_key
            if alive:
                if st.button("🔍 Extract OCR Text", use_container_width=True):
                    with st.spinner("Running Gemini OCR…"):
                        preview = run_ocr_preview(invoice_file)
                        if preview:
                            st.session_state["ocr_preview"] = preview
            else:
                st.caption("Start FastAPI to preview OCR before verification.")

        preview = st.session_state.get("ocr_preview")
        if preview and invoice_file:
            st.markdown('<div class="card"><div class="card-title">🔍 OCR Preview</div>', unsafe_allow_html=True)
            render_ocr_data(preview, title="Extracted from uploaded image")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.form("verify_form"):
            st.markdown("**Your Details**")
            name   = st.text_input("Full Name",    placeholder="")
            mobile = st.text_input("Mobile",       placeholder="+91 9876543210")
            email  = st.text_input("Email",        placeholder="abc@example.com")

            st.markdown("**Product Details**")
            # Dropdown containing Ambrane products
            AMBRANE_PRODUCTS = [
                "Ambrane PowerMini 20 Fast Charging Power Bank",
    "Ambrane Stylo N10 Slim & Compact Power Bank",
    "Ambrane Stellar Ultra High Capacity Power Bank",
    "Ambrane Powerlit Ultra 240 Laptop Power Bank",
    "Ambrane Aerosync Snap Wireless Power Bank",
    "Ambrane Aerosync Snap2 Wireless Power Bank",
    "Ambrane AeroSync PB 12 MagSafe Power Bank",
    "Ambrane MiniCharge 20 Compact Power Bank",
    "Ambrane Powerlit Pocket Power Bank",
    "Ambrane Charge GaN Fast Charging Wall Adapter",
    "Ambrane i45W Mobile Fast Charging Adapter",
    "Ambrane ATA-03 Universal Travel Socket Adaptor",
    "Ambrane AC4CL-15 4-in-1 Fast Charging Braided Cable",
    "Ambrane MiniVac 02 Portable Wireless Vacuum Cleaner",
    "Ambrane CarLink Stream CarPlay Adapter",
    "Ambrane Wireless CarPlay Streaming Adapter",
    "Ambrane Smartstrip Extension Board",
    "Ambrane Charge HUB Multi-Port Charger",
    "Ambrane Hitz Wired Headset",
    "Ambrane EP-21 Wired Earphones",
    "Ambrane Stringz Earphones",
    "Ambrane SliQ Pro Wireless Mouse",
    "Ambrane StarLight Portable Light"
            ]
            product_name = st.selectbox("Product Name", AMBRANE_PRODUCTS)
            marketplace  = st.selectbox("Marketplace", ["Ambrane India",
                "Amazon", "Flipkart", "Croma", "Reliance Digital",
                "Myntra", "Meesho", "Blinkit","Other"
            ])
            purchase_date_val = st.date_input("Purchase Date", value=date.today())
            purchase_date = str(purchase_date_val) if purchase_date_val else ""

            submitted = st.form_submit_button("🚀 Verify & Register")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if submitted:
            # ── Validation ─────────────────────────────────────────────────
            errors = []
            if not name.strip():        errors.append("Name is required.")
            if not mobile.strip():      errors.append("Mobile is required.")
            if not email.strip():       errors.append("Email is required.")
            if not product_name.strip():errors.append("Product name is required.")
            if invoice_file is None:    errors.append("Please upload an invoice.")
            if not alive:               errors.append("FastAPI server is not running.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # ── Progress ───────────────────────────────────────────────
                steps = [
                    "📤 Uploading invoice…",
                    "🔍 Running OCR extraction…",
                    "🔬 Forensics analysis (ELA + metadata)…",
                    "📦 Product name matching…",
                    "🏪 Seller validation…",
                    "📅 Date validation…",
                    "🔁 Duplicate detection…",
                    "⚖️ Risk engine scoring…",
                    "🤖 AI decision agent…",
                ]
                progress_bar = st.progress(0)
                status_text  = st.empty()

                mime = invoice_file.type or "application/octet-stream"
                files   = {"invoice": (invoice_file.name or "invoice.jpg", invoice_file.getvalue(), mime)}
                payload = {
                    "name": name, "mobile": mobile, "email": email,
                    "product_name": product_name, "marketplace": marketplace,
                    "purchase_date": purchase_date,
                }

                for i, step in enumerate(steps[:-1]):
                    status_text.markdown(f"**{step}**")
                    progress_bar.progress(int((i + 1) / len(steps) * 90))
                    time.sleep(0.4)

                status_text.markdown(f"**{steps[-1]}**")
                try:
                    resp = requests.post(f"{API_BASE}/verify", data=payload, files=files, timeout=120)
                    progress_bar.progress(100)
                    status_text.empty()

                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["last_result"] = data
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text[:300]}")
                        data = None
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API. Make sure FastAPI is running.")
                    data = None
                except Exception as ex:
                    st.error(f"Unexpected error: {ex}")
                    data = None

        # ── Display Result ────────────────────────────────────────────────
        data = st.session_state.get("last_result")
        if data:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Verification Result</div>', unsafe_allow_html=True)

            # Decision badge
            st.markdown(badge(data["decision"]), unsafe_allow_html=True)
            st.markdown(f"**Registration ID:** `#{data.get('registration_id','—')}`")
            st.markdown(f"*{data.get('summary', '')}*")
            st.markdown("---")

            # Risk score
            score = data["risk_score"]
            score_colour = "#3fb950" if score < 30 else ("#d29922" if score < 60 else "#f85149")
            st.markdown(f"**Risk Score:** <span style='color:{score_colour};font-size:1.3rem;font-weight:700'>{score:.0f}/100</span>", unsafe_allow_html=True)
            st.markdown(risk_bar(score), unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {data.get('confidence', '—')}%")
            st.markdown("---")

            # OCR Data
            render_ocr_data(data.get("ocr_data", {}))
            st.markdown("---")

            # Risk breakdown
            st.markdown("**⚖️ Risk Breakdown**")
            bd = data.get("breakdown", {})
            cols = st.columns(4)
            items = [
                ("ELA",      bd.get("ela_risk", 0)),
                ("Metadata", bd.get("metadata_risk", 0)),
                ("Copy-Move",bd.get("copy_move_risk", 0)),
                ("Product",  bd.get("product_risk", 0)),
                ("Seller",   bd.get("seller_risk", 0)),
                ("Date",     bd.get("date_risk", 0)),
                ("Duplicate",bd.get("duplicate_risk", 0)),
            ]
            for i, (label, val) in enumerate(items):
                with cols[i % 4]:
                    colour = "#3fb950" if val == 0 else ("#d29922" if val < 20 else "#f85149")
                    st.markdown(
                        f"<small style='color:#8b949e'>{label}</small><br>"
                        f"<b style='color:{colour}'>{val:.0f}</b>",
                        unsafe_allow_html=True,
                    )
            st.markdown("---")

            # Reasons
            if data.get("reasons"):
                st.markdown("**🗒️ Reasons**")
                for r in data["reasons"]:
                    icon = "⚠️" if any(w in r.lower() for w in ["risk","flag","detect","tamper","dup"]) else "✅"
                    st.markdown(f"{icon} {r}")

            st.markdown('</div>', unsafe_allow_html=True)
        elif not submitted:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem; color:#484f58;">
                <div style="font-size:3rem;">🛡️</div>
                <p style="margin-top:1rem;">Fill the form and upload your invoice to begin verification.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page: Registrations
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Registrations":
    st.markdown("## 📋 Recent Registrations")
    st.markdown("---")

    if not alive:
        st.error("API is offline. Start FastAPI first.")
    else:
        if st.button("🔄 Refresh"):
            st.rerun()
        try:
            rows = requests.get(f"{API_BASE}/registrations?limit=30", timeout=10).json()
            if not rows:
                st.info("No registrations yet.")
            else:
                for r in rows:
                    dec = r["decision"]
                    icon = {"APPROVE":"✅","MANUAL_REVIEW":"⚠️","REJECT":"❌"}.get(dec, "❓")
                    colour= {"APPROVE":"#3fb950","MANUAL_REVIEW":"#d29922","REJECT":"#f85149"}.get(dec,"#8b949e")
                    with st.expander(f"{icon} #{r['id']} — {r['name']} | {r['product']} | Risk: {r['risk_score']:.0f}"):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.markdown(f"**Email:** {r['email']}")
                        c2.markdown(f"**Invoice No:** `{r.get('invoice_number') or '—'}`")
                        c3.markdown(f"**Decision:** <span style='color:{colour}'>{dec}</span>", unsafe_allow_html=True)
                        c4.markdown(f"**Registered:** {r['created_at'][:10]}")

                        st.markdown("---")
                        if st.button("🗑️ Delete Registration", key=f"del_{r['id']}", use_container_width=True):
                            try:
                                res = requests.delete(f"{API_BASE}/registrations/{r['id']}", timeout=10)
                                if res.status_code == 200:
                                    st.toast(f"Registration #{r['id']} deleted successfully!", icon="🗑️")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete: {res.text}")
                            except Exception as ex:
                                st.error(f"Error deleting: {ex}")
        except Exception as e:
            st.error(f"Error fetching registrations: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Page: How It Works
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ How It Works":
    st.markdown("## ℹ️ How It Works")
    st.markdown("---")

    steps = [
        ("1️⃣ OCR Extraction",      "Gemini 2.0 Flash reads invoice text via API and extracts invoice number, seller, product name, date, amount, and full raw text."),
        ("2️⃣ Forensics Analysis",  "Three checks: **Metadata** (editing tool fingerprints in EXIF), **ELA** (Error Level Analysis for tampered pixels), **Copy-Move** (cloned region detection via ORB keypoints)."),
        ("3️⃣ Product Matching",    "RapidFuzz fuzzy-matches the user-entered product name against the OCR-extracted product. Below 80% similarity → risk flag."),
        ("4️⃣ Seller Validation",   "The invoice seller is checked against an authorized-sellers database for the chosen marketplace."),
        ("5️⃣ Date Validation",     "Future dates are rejected. Invoices older than the warranty window are flagged."),
        ("6️⃣ Duplicate Detection", "Invoice number, email, and mobile are checked for prior registrations in PostgreSQL."),
        ("7️⃣ Risk Engine",         "Sub-scores are aggregated: ELA (0-25) + Metadata (0-20) + Copy-Move (0-25) + Product (0-20) + Seller (0-15) + Date (0-15) + Duplicate (0-40). Capped at 100."),
        ("8️⃣ AI Decision",         "Gemini receives the full risk report and flags, then returns a structured APPROVE / MANUAL_REVIEW / REJECT decision with confidence and reasons."),
    ]

    for title, desc in steps:
        st.markdown(f'<div class="card"><div class="card-title">{title}</div><p style="color:#c9d1d9;margin:0">{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Risk Thresholds")
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="card" style="border-color:#238636"><div class="card-title" style="color:#3fb950">✅ APPROVE</div><p style="color:#c9d1d9">Score &lt; 20</p></div>', unsafe_allow_html=True)
    col2.markdown('<div class="card" style="border-color:#9e6a03"><div class="card-title" style="color:#d29922">⚠️ MANUAL REVIEW</div><p style="color:#c9d1d9">Score 20–50</p></div>', unsafe_allow_html=True)
    col3.markdown('<div class="card" style="border-color:#da3633"><div class="card-title" style="color:#f85149">❌ REJECT</div><p style="color:#c9d1d9">Score &gt; 50</p></div>', unsafe_allow_html=True)