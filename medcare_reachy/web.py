"""Settings-page API, mounted on the app's settings server (port 8042 on the robot)."""

from pydantic import BaseModel

from medcare_reachy.settings_store import SettingsError, public_view


class ConfigUpdate(BaseModel):
    app_url: str | None = None
    device_token: str | None = None
    language: str | None = None
    capture_fps: float | None = None
    vision_on_server: bool | None = None
    checkin_ack: bool | None = None
    checkin_gestures: bool | None = None
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None


def register_routes(api, service) -> None:
    @api.get("/api/config")
    def get_config():
        return public_view(service.store.load())

    @api.post("/api/config")
    def save_config(update: ConfigUpdate):
        from fastapi.responses import JSONResponse

        try:
            settings = service.store.save(update.model_dump())
        except SettingsError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        service.restart()
        return {"config": public_view(settings), "status": service.status()}

    @api.get("/api/status")
    def get_status():
        return service.status()

    @api.post("/api/restart")
    def restart():
        service.restart()
        return service.status()
