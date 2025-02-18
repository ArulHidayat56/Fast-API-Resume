from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from typing import List
from bs4 import BeautifulSoup

app = FastAPI(
    title="Medical SOAP to JSON API",
    description="API otomatis untuk mengonversi catatan SOAP menjadi resume HTML lalu JSON",
    version="1.0.0"
)

class SOAPRequest(BaseModel):
    soap_text: str

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token="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"
)

SECTION_TITLES = [
    "Keluhan Utama",
    "Alasan Pasien Dirawat",
    "Riwayat Penyakit",
    "Pemeriksaan Fisik",
    "Pemeriksaan Penunjang",
    "Diagnosis Primer",
    "Diagnosis Sekunder",
    "Prosedur Terapi dan Tindakan yang Telah Dikerjakan",
    "Obat yang Diberikan Saat Dirawat",
    "Obat yang Diberikan Setelah Pasien Keluar Rumah Sakit",
    "Instruksi / Tindak Lanjut"
]

def generate_html_resume(soap_text: str) -> str:
    prompt = (
        "Buatkan resume medis berbasis SOAP berikut dalam format tabel HTML yang memiliki bagian berikut: "
        + ", ".join(SECTION_TITLES) + ". Harap isi setiap bagian sesuai urutan dengan label yang jelas dan isi yang sesuai.\n"
        f"SOAP: {soap_text}"
    )
    response = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=2048, temperature=0.7, top_p=0.9)
    return response.choices[0].message["content"].strip()

def html_to_json(html_content: str) -> dict:
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = []
    table = soup.find('table')
    if not table:
        raise Exception("No table found in HTML")
    for i, row in enumerate(table.find_all('tr')):
        cells = row.find_all('td')
        if cells:
            title = SECTION_TITLES[i] if i < len(SECTION_TITLES) else f"Bagian {i+1}"
            content = " ".join(cell.get_text(strip=True) for cell in cells)
            sections.append({"title": title, "content": content})
    if not sections:
        raise Exception("No sections parsed from HTML")
    return {"title": "RESUME MEDIS", "sections": sections}

@app.post("/full-process/")
async def full_process(request: SOAPRequest):
    try:
        html_resume = generate_html_resume(request.soap_text)
        if not html_resume:
            raise HTTPException(status_code=500, detail="AI did not return a valid HTML resume")
        json_resume = html_to_json(html_resume)
        if not json_resume["sections"]:
            raise HTTPException(status_code=500, detail="Failed to parse sections from HTML")
        return JSONResponse(content=json_resume)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")
