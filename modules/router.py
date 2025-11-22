from fastapi import APIRouter, status
from .scheme import AdalineGDIn, AdalineGDPredict, AdalineGDOut
from adalineGD import AdalineGD


modules_router = APIRouter()

adaline_gd = AdalineGD(n_iter=2000)


@modules_router.post(
    '/',
    summary='AdalineGD fit',
    status_code=status.HTTP_200_OK,
    response_model=AdalineGDOut
)
async def model_fit(
        adalineGD_scheme: AdalineGDIn
) -> AdalineGDOut:
    adalinegd_out = AdalineGDOut(
        fit=adaline_gd.fit(adalineGD_scheme.X, adalineGD_scheme.y),
        w_=adaline_gd.w_
    )
    return adalinegd_out


@modules_router.post(
    '/predict',
    summary='AdalineGD predict',
    status_code=status.HTTP_200_OK,
    response_model=int
)
async def model_predict(
        adalineGD_scheme: AdalineGDPredict
) -> int:
    return adaline_gd.predict(adalineGD_scheme.X)