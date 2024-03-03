from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

app = FastAPI(
    title="명언제조기",
    description="Get a real quote said by oldman.",
    servers=[
        {
            "url": "https://outdoor-shannon-miscellaneous-advances.trycloudflare.com",
        },
    ],
)

# 서버실행 명령어
# uvicorn main:app --reload

# cloudflared tunnel 실행
# cloudflared tunnel --url http://127.0.0.1:8000


class Quote(BaseModel):
    quote: str = Field(description="The quote that someone said.")
    year: int = Field(description="The year when Someone said the quote.")


@app.get(
    "/quote",
    summary="Returns a random quote by someone",
    description="upon receiving a GET request this endpoint will return a real quiote said by someone",
    response_description="A Quote object that contains the quote said by someone and the date when the quote was said.",
    response_model=Quote,
)
def get_quote(request: Request):
    print(request.headers)
    return {
        "quote": "인생은 짧습니다. 먹고싶은것 마음껏 드세요.",
        "year": 1950,
    }
