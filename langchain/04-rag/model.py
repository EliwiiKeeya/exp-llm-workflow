# -*- encoding: utf-8 -*-
# @File			: model.py
# @Date			: 2026/07/04 21:09:22
# @Author		: Eliwii_Keeya

import os
import abc
import time
import threading
import asyncio
from typing import Optional

from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.rate_limiters import BaseRateLimiter

load_dotenv()


class CustomRateLimiterMeta(abc.ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class CustomRateLimiter(BaseRateLimiter, metaclass=CustomRateLimiterMeta):
    """An in memory rate limiter based on a token bucket algorithm.

    This is an in memory rate limiter, so it cannot rate limit across
    different processes.

    The rate limiter only allows time-based rate limiting and does not
    take into account any information about the input or the output, so it
    cannot be used to rate limit based on the size of the request.

    It is thread and coroutine safe and can be used in either a sync or async context.

    The in memory rate limiter is based on a token bucket. The bucket is filled
    with tokens at a given rate. Each request consumes a token. If there are
    not enough tokens in the bucket, the request is blocked until there are
    enough tokens.

    These tokens have nothing to do with LLM tokens. They are just
    a way to keep track of how many requests can be made at a given time.

    Current limitations:

    - The rate limiter is not designed to work across different processes. It is
        an in-memory rate limiter, but it is thread safe.
    - The rate limiter only supports time-based rate limiting. It does not take
        into account the size of the request or any other factors.
    """

    def __init__(
        self,
        peak_burst_size: float,
        committed_burst_size: float,
        check_every_n_seconds: float,
        peak_information_rate: Optional[float] = None,
        committed_information_rate: Optional[float] = None,
        eps: float = 1e-9
    ) -> None:
        """A rate limiter based on a token bucket.

        These tokens have nothing to do with LLM tokens. They are just
        a way to keep track of how many requests can be made at a given time.

        This rate limiter is designed to work in a threaded environment.

        It works by filling up a bucket with tokens at a given rate. Each
        request consumes a given number of tokens. If there are not enough
        tokens in the bucket, the request is blocked until there are enough
        tokens.

        Args:
            peak_burst_size: The maximum number of tokens that can be in the bucket.
                Will be raised at least `1`. Used to prevent bursts of requests.
            committed_burst_size: The maximum number of tokens that can be in the bucket.
                Will be raised at least `1`. Used to prevent bursts of requests.
            peak_information_rate: The rate at which tokens are added to the peak bucket.
                If None, it will be set to the peak burst size.
            committed_information_rate: The rate at which tokens are added to the committed bucket.
                If None, it will be set to the committed burst size.
            check_every_n_seconds: Check whether the tokens are available
                every this many seconds. Can be a float to represent
                fractions of a second.
            eps: A small value to avoid floating point precision issues.
        """
        # Maximum Number of requests that we can make per second.
        # EPS is used to avoid floating point precision issues.
        self.peak_burst_size = max(peak_burst_size, 1) + eps
        self.committed_burst_size = max(committed_burst_size, 1) + eps

        # Number of requests that we can make per second.
        # If the PIR is None, we will set it to the PBS.
        if peak_information_rate is None:
            self.peak_information_rate = peak_burst_size
        else:
            self.peak_information_rate = peak_information_rate

        # If the CIR is None, we will set it to the CBS.
        if committed_information_rate is None:
            self.committed_information_rate = committed_burst_size
        else:
            self.committed_information_rate = committed_information_rate

        # Number of tokens in the bucket.
        self.T_p = 0.0
        self.T_c = 0.0

        # A lock to ensure that tokens can only be consumed by one thread
        # at a given time.
        self._consume_lock = threading.Lock()
        self._a_consume_lock = asyncio.Lock()

        # The last time we tried to consume tokens.
        # Initialize to avoid a burst.
        # Begin fulling the bucket with tokens at the start of the program.
        self.last: float = time.monotonic()

        self.check_every_n_seconds = check_every_n_seconds

    def _consume(self) -> bool:
        """Try to consume a token.

        Returns:
            True means that the tokens were consumed, and the caller can proceed to
            make the request. A False means that the tokens were not consumed, and
            the caller should try again later.
        """
        with self._consume_lock:
            now = time.monotonic()

            elapsed = now - self.last
            self.last = now

            # Produce tokens based on the elapsed time and the rate limits.
            # Make sure that we don't exceed the bucket size.
            # This is used to prevent bursts of requests.
            if self.T_p < self.peak_burst_size:
                self.T_p = min(
                    self.T_p + elapsed * self.peak_information_rate,
                    self.peak_burst_size
                )

            if self.T_c < self.committed_burst_size:
                self.T_c = min(
                    self.T_c + elapsed * self.committed_information_rate,
                    self.committed_burst_size
                )

            if self.T_p < 1:
                return False
            elif self.T_p >= 1 and self.T_c < 1:
                self.T_p -= 1
                return False
            else:
                self.T_p -= 1
                self.T_c -= 1
                return True

    def acquire(self, *, blocking: bool = True) -> bool:
        """Attempt to acquire a token from the rate limiter.

        This method blocks until the required tokens are available if `blocking`
        is set to `True`.

        If `blocking` is set to `False`, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If `True`, the method will block until the tokens are available.
                If `False`, the method will return immediately with the result of
                the attempt.

        Returns:
            `True` if the tokens were successfully acquired, `False` otherwise.
        """
        if not blocking:
            return self._consume()

        while not self._consume():
            time.sleep(self.check_every_n_seconds)

        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        """Attempt to acquire a token from the rate limiter. Async version.

        This method blocks until the required tokens are available if `blocking`
        is set to `True`.

        If `blocking` is set to `False`, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If `True`, the method will block until the tokens are available.
                If `False`, the method will return immediately with the result of
                the attempt.

        Returns:
            `True` if the tokens were successfully acquired, `False` otherwise.
        """
        async with self._a_consume_lock:
            if not blocking:
                return self._consume()

            while not self._consume():  # noqa: ASYNC110
                # This code ignores the ASYNC110 warning which is a false positive in this
                # case.
                # There is no external actor that can mark that the Event is done
                # since the tokens are managed by the rate limiter itself.
                # It needs to wake up to re-fill the tokens.
                # https://docs.astral.sh/ruff/rules/async-busy-wait/
                await asyncio.sleep(self.check_every_n_seconds)
            return True


class CustomRateLimiterDeepSeekV4(CustomRateLimiter):
    def __init__(self) -> None:
        # DeepSeek V4 Flash rate limits
        super().__init__(
            peak_burst_size=1,
            committed_burst_size=3 / 60,
            check_every_n_seconds=0.1
        )


class CustomChatDeepSeekV4Flash(ChatOpenAI):
    def __init__(self) -> None:
        OPENAI_MODEL = "deepseek-v4-flash"
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_BASE_URL = "https://api.modelarts-maas.com/openai/v1"

        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY configuration.")

        super().__init__(
            rate_limiter=CustomRateLimiterDeepSeekV4(),
            model=OPENAI_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL,
        )


class CustomRateLimiterDeepSeekV3(CustomRateLimiter):
    def __init__(self) -> None:
        # DeepSeek V3.2 rate limits
        super().__init__(
            peak_burst_size=700 / 60,
            committed_burst_size=700 / 60,
            check_every_n_seconds=0.1
        )


class CustomChatDeepSeekV3(ChatOpenAI):
    def __init__(self) -> None:
        OPENAI_MODEL = "deepseek-v3.2"
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_BASE_URL = "https://api.modelarts-maas.com/openai/v1"

        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY configuration.")

        super().__init__(
            rate_limiter=CustomRateLimiterDeepSeekV3(),
            model=OPENAI_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL,
        )


class CustomRateLimiterOpenPanguFlash(CustomRateLimiter):
    def __init__(self) -> None:
        # OpenPangu Flash rate limits
        super().__init__(
            peak_burst_size=100 / 60,
            committed_burst_size=100 / 60,
            check_every_n_seconds=0.1
        )


class CustomChatOpenPanguFlash(ChatOpenAI):
    def __init__(self) -> None:
        OPENAI_MODEL = "openpangu-2.0-flash"
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_BASE_URL = "https://api.modelarts-maas.com/openai/v1"

        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY configuration.")

        super().__init__(
            rate_limiter=CustomRateLimiterOpenPanguFlash(),
            model=OPENAI_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL,
        )


class CustomBGEM3Embeddings(OpenAIEmbeddings):
    def __init__(self) -> None:
        OPENAI_MODEL = "bge-m3"
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_BASE_URL = "https://api.modelarts-maas.com/v1"

        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY configuration.")

        super().__init__(
            model=OPENAI_MODEL,
            api_key=SecretStr(OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL,
        )


if __name__ == '__main__':
    import dotenv
    import requests
    import json
    import os
    dotenv.load_dotenv()
    url = "https://api.modelarts-maas.com/v1/embeddings"  # API地址
    api_key = os.getenv("OPENAI_API_KEY")  # 获取MAAS_API_KEY环境变量

    # Send request.
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    texts = ["这是一只小猫", "这是一只小狗"]
    data = {
        "model": "bge-m3",  # model参数
        "input": texts,  # input类型可为string or string[]
        "encoding_format": "float"  # 取值范围："float","base64"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)

    # Print result.
    print(response.status_code)
    print(response.text)
