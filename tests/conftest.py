import copy
import logging

import httpx
import pytest

from react_lite.agent import Agent
from react_lite.events import Events
from react_lite.llm import Reply
from react_lite.tools import ExchangeRates, Tools

XML = b"""<?xml version="1.0" encoding="windows-1251"?>
<ValCurs Date="15.10.2026" name="Foreign Currency Market">
  <Valute ID="R01235"><CharCode>USD</CharCode><Nominal>1</Nominal><Value>90,1234</Value></Valute>
  <Valute ID="R01820"><CharCode>JPY</CharCode><Nominal>100</Nominal><Value>62,5000</Value></Valute>
</ValCurs>"""


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, tools, budget):
        self.requests.append(copy.deepcopy(messages))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(messages)
        return reply


@pytest.fixture
def make_agent():
    clients = []

    def make(replies, handler=None, **kwargs):
        client = httpx.Client(
            transport=httpx.MockTransport(
                handler or (lambda request: httpx.Response(200, content=XML)),
            )
        )
        clients.append(client)
        model = ScriptedModel(replies)
        logger = logging.getLogger(f"test.{id(model)}")
        logger.addHandler(logging.NullHandler())
        events = Events(logger)
        agent = Agent(model, Tools(ExchangeRates(client, attempts=1)), events, **kwargs)
        return agent, model

    yield make
    for client in clients:
        client.close()


@pytest.fixture
def final():
    return Reply("Готово")
