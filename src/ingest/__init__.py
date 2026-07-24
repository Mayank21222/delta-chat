from src.ingest.base import AdapterRegistry
from src.ingest.pdf_native import PDFNativeAdapter
from src.ingest.pdf_scanned import PDFScannedAdapter
from src.ingest.dwg import DWGAdapter

AdapterRegistry.register("pdf_native", PDFNativeAdapter())
AdapterRegistry.register("pdf_scanned", PDFScannedAdapter())
AdapterRegistry.register("dwg", DWGAdapter())
