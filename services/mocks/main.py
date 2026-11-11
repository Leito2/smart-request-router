"""Run the P1/P3 mocks: python -m services.mocks.main  (port 8010)."""
import uvicorn

from services.mocks.app import create_mock_app

if __name__ == "__main__":
    uvicorn.run(create_mock_app(), host="0.0.0.0", port=8010)
