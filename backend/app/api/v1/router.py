from fastapi import APIRouter

from app.api.v1.endpoints import audio

api_router = APIRouter()
api_router.include_router(audio.router)