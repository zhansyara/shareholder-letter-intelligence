# Shareholder Letter Intelligence

A Streamlit MVP for comparing annual shareholder letters across companies and years.

## What is included

- White-blue interface
- Industry selection and industry-specific analysis lenses
- PDF upload
- Optional OpenAI-powered semantic analysis
- Offline heuristic demo mode
- Management-tone radar
- Strategy constellation
- Strategic-emphasis heatmap
- Commitments and risk extraction
- Annotated letter view
- Editable long-term performance dataset
- CSV comparison export

## Installation

### 1. Install Python

Use Python 3.11, 3.12, or 3.13.

### 2. Open a terminal in this folder

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run streamlit_app.py
```

A browser window should open automatically.

## OpenAI mode

The app runs without an API key in **Demo / heuristic** mode.

For stronger analysis, select **OpenAI API** in the sidebar and paste your API key. You can also set it as an environment variable:

**macOS / Linux**
```bash
export OPENAI_API_KEY="your-key"
```

**Windows PowerShell**
```powershell
$env:OPENAI_API_KEY="your-key"
```

## Recommended first dataset

Start with one industry and 4-6 letters:

- 2-3 companies
- 2 annual letters per company
- the same or adjacent fiscal years

This is sufficient to test year-over-year and peer comparison.

## Important limitation

This MVP extracts text from text-based PDFs. Scanned PDFs require OCR. The annotated-letter view highlights matched text excerpts rather than drawing overlays on the original PDF page.
