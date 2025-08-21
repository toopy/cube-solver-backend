from contextlib import asynccontextmanager
from functools import partial
from typing import List

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware

from backends.depends import get_backend_name
from backends.registry import (
    get_registry,
    init_registry,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_registry(app)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
def solve_cube(
    color_input: List[List[str]],
    backend_name: str = Depends(partial(get_backend_name, app)),
):
    try:
        solver = get_registry(app).get(backend_name)
        actions = solver.solve(color_input)
        return actions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # except Exception as e:
    #     # évite de leak les traces internes en prod
    #     raise HTTPException(status_code=500, detail="Inference error")
