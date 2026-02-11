from pydantic import BaseModel
from typing import Dict, Any, Optional

class MetricDetail(BaseModel):
    valor: float
    classificacao: str
    descricao: str

class AnalysisMetadata(BaseModel):
    idade: int
    exercicio: str
    duracao_video: str

class AnalysisResponse(BaseModel):
    metadata: AnalysisMetadata
    metricas: Dict[str, MetricDetail]
    eventos: Dict[str, int]
    frames_analisados: int
    status: str
    screenshots: Optional[list[str]] = None
