from datetime import date

from pydantic import BaseModel, ConfigDict


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    CDPHId: int
    ChemicalId: int
    ChemicalName: str | None = None
    CasNumber: str | None = None
    ProductName: str | None = None
    CompanyName: str | None = None
    BrandName: str | None = None
    PrimaryCategory: str | None = None
    SubCategory: str | None = None
    InitialDateReported: date | None = None
    MostRecentDateReported: date | None = None
    DiscontinuedDate: date | None = None
    ChemicalDateRemoved: date | None = None