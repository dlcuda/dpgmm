from pydantic import BaseModel


class BaseModelWithTensorsAllowed(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
