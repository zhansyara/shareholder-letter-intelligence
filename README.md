# SHAREHOLDER LETTER INTELLIGENCE

Turn management narratives into structured investment insight.

Upload shareholder letters to evaluate tone, strategic priorities, management commitments and long-term credibility! 
Get every conclusion grounded in the source document!

## What is included

- AI-powered analysis of shareholder letters
- Company and industry-specific evaluation framework
- PDF upload and automatic text extraction
- Management tone and communication assessment
- Strategic priorities and capital allocation analysis
- Risk, opportunity and commitment identification
- Interactive visualizations and comparative dashboards
- Annotated letter with evidence-backed insights
- Long-term financial performance tracking
- Exportable comparison results (CSV)

## Screenshots

<img width="1358" height="718" alt="image" src="https://github.com/user-attachments/assets/cb6f645a-02b6-44a9-b438-c348488dc7c1" />
<img width="1358" height="718" alt="image" src="https://github.com/user-attachments/assets/dff6f8b7-0341-4978-a42e-739656bc8453" />
<img width="1358" height="656" alt="image" src="https://github.com/user-attachments/assets/edd21a1e-42ce-4745-b48b-abb4b5737a7c" />
<img width="1358" height="715" alt="image" src="https://github.com/user-attachments/assets/4d465faf-842b-4186-bccc-bf9ce60ff3d4" />

## Live Demo
https://shareholder-letter-intelligence.streamlit.app


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
