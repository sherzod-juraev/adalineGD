from pydantic import BaseModel, field_validator, model_validator
from fastapi import HTTPException, status
from numpy import ndarray, issubdtype, number, array


class AdalineGDIn(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list[float]]
    y: list[int]

    @field_validator('X')
    def verify_X(cls, X):
        X = array(X)
        if X.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='A 2D matrix must be entered'
            )
        if not issubdtype(X.dtype, number):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='There should be no NaN values.'
            )
        return X

    @field_validator('y')
    def verify_y(cls, y):
        y = array(y)
        if y.ndim != 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='target values must be a vector'
            )
        if not issubdtype(y.dtype, number):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='There must be no NaN values.'
            )
        return y

    @model_validator(mode='after')
    def verify_object(self):
        if self.X.shape[0] != self.y.shape[0]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='The sample and target values are not the same size.'
            )
        return self


class AdalineGDOut(BaseModel):

    fit: bool
    w_: list


class AdalineGDPredict(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list]

    @field_validator('X')
    def verify_X(cls, X):
        X = array(X)
        if X.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='X must be a 2D matrix'
            )
        return X