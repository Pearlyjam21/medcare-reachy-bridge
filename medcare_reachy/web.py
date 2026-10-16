"""Settings-page API, mounted on the app's settings server (port 8042 on the robot)."""

from pydantic import BaseModel

from medcare_reachy.service import ServiceBusy
from medcare_reachy.settings_store import SettingsError, public_view


class ExpressRequest(BaseModel):
    affect: str | None = None     # "concern", "joy" or null (no posture)
    move: str | None = None       # a recorded emotion move, e.g. "downcast1", or null
    seconds: float = 6.0          # how long the posture is held before it is lifted (1-20)


class ConfigUpdate(BaseModel):
    app_url: str | None = None
    device_token: str | None = None
    language: str | None = None
    capture_fps: float | None = None
    vision_on_server: bool | None = None
    checkin_ack: bool | None = None
    checkin_gestures: bool | None = None
    multimodal_emotion: bool | None = None
    follow_face: bool | None = None
    emotion_compute: str | None = None


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

    @api.post("/api/express")
    def express(request: ExpressRequest):
        """A test of Reachy's body language on the real robot, from the laptop, without a chat: the posture
        (`affect`: concern, joy) and one recorded emotion move (`move`, media.MOVE_LIBRARIES "emotion"), as the live
        chat's `express` message would show them, for `seconds`. Refused (409) while a reminder or a chat runs."""
        from fastapi.responses import JSONResponse

        try:
            return service.express(request.affect, request.move, request.seconds)
        except ServiceBusy as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)
