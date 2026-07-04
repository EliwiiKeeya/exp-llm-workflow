# -*- encoding: utf-8 -*-
# @File			: model.py
# @Date			: 2026/07/04 21:09:22
# @Author		: Eliwii_Keeya

import os

from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import BaseRateLimiter

load_dotenv()


class CustomRateLimiterChatOpenAI(BaseRateLimiter):
    def __init__(self) -> None:
        super().__init__()

    def acquire(self, *, blocking: bool = True) -> bool:
        return super().acquire(blocking=blocking)

    async def aacquire(self, *, blocking: bool = True) -> bool:
        return await super().aacquire(blocking=blocking)


class CustomChat(ChatOpenAI):
    def __init__(self) -> None:
        OPENAI_MODEL = os.getenv("OPENAI_MODEL")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

        if not OPENAI_MODEL or not OPENAI_API_KEY or not OPENAI_BASE_URL:
            raise ValueError("Missing OpenAI configuration.")

        super().__init__(
            rate_limiter=CustomRateLimiterChatOpenAI(),
            model=OPENAI_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL,
        )
