from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, DateTime, Integer, String, func

class Base(DeclarativeBase):
    pass

class MetaData(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_name = Column(String, nullable=False)
    doc_type = Column(String)
    file_path = Column(String, nullable=False)
    indexing_status = Column(String, nullable=False, default="indexing")
    indexed_chunks = Column(Integer, nullable=False, default=0)
    indexing_error = Column(String)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    
