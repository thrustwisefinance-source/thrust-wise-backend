"""
Base response envelope matching the frontend's ApiResponse<T>
(types/common.types.ts): { data: T, success: boolean, message?: string }.

All schemas serialize with camelCase aliases so JSON keys match the
TypeScript interfaces exactly.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


T = TypeVar("T")


class ApiResponse(CamelModel, Generic[T]):
    data: T
    success: bool = True
    message: str | None = None
