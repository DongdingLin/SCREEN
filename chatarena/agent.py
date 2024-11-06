from typing import List, Union
import re
from tenacity import RetryError
import logging
import uuid
from abc import abstractmethod
import asyncio

from .backends import IntelligenceBackend, load_backend
from .message import Message, SYSTEM_NAME
from .config import AgentConfig, Configurable, BackendConfig

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Agent(Configurable):

    @abstractmethod
    def __init__(self, name: str, role_desc: str, role_desc_in_transition_turn: str, role_desc_after_transition_turn: str, transition_turn: int, global_prompt: str = None, visual_path: str = None, second_visual_path: str = None, *args, **kwargs):
        super().__init__(name=name, role_desc=role_desc, role_desc_in_transition_turn=role_desc_in_transition_turn, role_desc_after_transition_turn=role_desc_after_transition_turn, transition_turn=transition_turn, global_prompt=global_prompt, visual_path=visual_path, second_visual_path=second_visual_path, **kwargs)
        self.name = name
        self.role_desc = role_desc
        self.role_desc_in_transition_turn = role_desc_in_transition_turn
        self.role_desc_after_transition_turn = role_desc_after_transition_turn
        self.transition_turn = transition_turn
        self.global_prompt = global_prompt
        self.visual_path = visual_path
        self.second_visual_path = second_visual_path
class Player(Agent):
    """
    Player of the game. It can takes the observation from the environment and return an action
    """

    def __init__(self, name: str, role_desc: str, role_desc_in_transition_turn: str, role_desc_after_transition_turn: str, transition_turn: int, backend: Union[BackendConfig, IntelligenceBackend], global_prompt: str = None, visual_path: str = None, second_visual_path: str = None, **kwargs):

        if isinstance(backend, BackendConfig):
            backend_config = backend
            backend = load_backend(backend_config)
        elif isinstance(backend, IntelligenceBackend):
            backend_config = backend.to_config()
        else:
            raise ValueError(f"backend must be a BackendConfig or an IntelligenceBackend, but got {type(backend)}")

        assert name != SYSTEM_NAME, f"Player name cannot be {SYSTEM_NAME}, which is reserved for the system."

        # Register the fields in the _config
        super().__init__(name=name, role_desc=role_desc, role_desc_in_transition_turn=role_desc_in_transition_turn, role_desc_after_transition_turn=role_desc_after_transition_turn, transition_turn=transition_turn, backend=backend_config, global_prompt=global_prompt, visual_path=visual_path, second_visual_path=second_visual_path, **kwargs)

        self.backend = backend

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            role_desc_in_transition_turn=self.role_desc_in_transition_turn,
            role_desc_after_transition_turn=self.role_desc_after_transition_turn,
            transition_turn=self.transition_turn,
            backend=self.backend.to_config(),
            global_prompt=self.global_prompt,
            visual_path=self.visual_path,
            second_visual_path=self.second_visual_path,
        )

    def act(self, observation: List[Message]) -> str:
        """
        Call the agents to generate a response (equivalent to taking an action).
        """
        try:
            response = self.backend.query(agent_name=self.name, role_desc=self.role_desc, role_desc_in_transition_turn=self.role_desc_in_transition_turn, role_desc_after_transition_turn=self.role_desc_after_transition_turn, transition_turn=self.transition_turn, history_messages=observation, global_prompt=self.global_prompt, visual_path=self.visual_path, second_visual_path=self.second_visual_path, request_msg=None)
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}. "
                            f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION

        return response

    def __call__(self, observation: List[Message]) -> str:
        return self.act(observation)

    async def async_act(self, observation: List[Message]) -> str:
        """
        Async call the agents to generate a response (equivalent to taking an action).
        """
        try:
            response = self.backend.async_query(agent_name=self.name, role_desc=self.role_desc, role_desc_in_transition_turn=self.role_desc_in_transition_turn, role_desc_after_transition_turn=self.role_desc_after_transition_turn, transition_turn=self.transition_turn, history_messages=observation, global_prompt=self.global_prompt, visual_path=self.visual_path, second_visual_path=self.second_visual_path, request_msg=None)
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}. "
                            f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION

        return response

    def reset(self):
        self.backend.reset()


class Moderator(Player):
    """
    A special type of player that moderates the conversation (usually used as a component of environment).
    """

    def __init__(self, role_desc: str, role_desc_in_transition_turn: str, role_desc_after_transition_turn: str, transition_turn: int, backend: Union[BackendConfig, IntelligenceBackend], terminal_condition: str, global_prompt: str = None, **kwargs):
        name = "Moderator"
        super().__init__(name=name, role_desc=role_desc, backend=backend, global_prompt=global_prompt, role_desc_in_transition_turn=role_desc_in_transition_turn, role_desc_after_transition_turn=role_desc_after_transition_turn, transition_turn=transition_turn, **kwargs)

        self.terminal_condition = terminal_condition

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            role_desc_in_transition_turn=self.role_desc_in_transition_turn,
            role_desc_after_transition_turn=self.role_desc_after_transition_turn,
            transition_turn=self.transition_turn,
            backend=self.backend.to_config(),
            terminal_condition=self.terminal_condition,
            global_prompt=self.global_prompt,
        )

    def is_terminal(self, history: List[Message], *args, **kwargs) -> bool:
        """
        check whether the conversation is over
        """
        # If the last message is the signal, then the conversation is over
        if history[-1].content == SIGNAL_END_OF_CONVERSATION:
            return True

        try:
            request_msg = Message(agent_name=self.name, content=self.terminal_condition, turn=-1)
            response = self.backend.query(agent_name=self.name, role_desc=self.role_desc, role_desc_in_transition_turn=self.role_desc_in_transition_turn, role_desc_after_transition_turn=self.role_desc_after_transition_turn, transition_turn=self.transition_turn, history_messages=history, global_prompt=self.global_prompt, request_msg=request_msg, *args, **kwargs)
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}.")
            return True

        if re.match(r"yes|y|yea|yeah|yep|yup|sure|ok|okay|alright", response, re.IGNORECASE):
            # print(f"Decision: {response}. Conversation is ended by moderator.")
            return True
        else:
            return False
