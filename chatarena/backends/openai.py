from typing import List
import os
import re
import logging
from tenacity import retry, stop_after_attempt, wait_random, wait_random_exponential
import base64
from .base import IntelligenceBackend
from ..message import Message, SYSTEM_NAME, MODERATOR_NAME
from colorama import Fore, Style
try:
    import openai
    from openai import OpenAI
except ImportError:
    is_openai_available = False
    logging.warning("openai package is not installed")
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if openai.api_key is None:
        logging.warning("OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY")
        is_openai_available = False
    else:
        is_openai_available = True

# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gpt-3.5-turbo"

END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|endoftext|>", END_OF_MESSAGE)  # End of sentence token
BASE_PROMPT = f"The messages always end with the token {END_OF_MESSAGE}."

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class OpenAIChat(IntelligenceBackend):
    """
    Interface to the ChatGPT style model with system, user, assistant roles separation
    """
    stateful = False
    type_name = "openai-chat"

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = DEFAULT_MODEL, merge_other_agents_as_one_user: bool = True, **kwargs):
        """
        instantiate the OpenAIChat backend
        args:
            temperature: the temperature of the sampling
            max_tokens: the maximum number of tokens to sample
            model: the model to use
            merge_other_agents_as_one_user: whether to merge messages from other agents as one user message
        """
        assert is_openai_available, "openai package is not installed or the API key is not set"
        super().__init__(temperature=temperature, max_tokens=max_tokens, model=model,
                         merge_other_agents_as_one_user=merge_other_agents_as_one_user, **kwargs)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.merge_other_agent_as_user = merge_other_agents_as_one_user

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=60))  # Modified retry strategy
    def _get_response(self, messages):
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1/chat/completions",
        )
        chat_completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=STOP,  
        )
        response = chat_completion.choices[0].message.content
        response = response.strip()
        return response

    def query(self, agent_name: str, role_desc: str, role_desc_in_transition_turn: str, role_desc_after_transition_turn: str, transition_turn: int, history_messages: List[Message], global_prompt: str = None, request_msg: Message = None, visual_path: str = None, second_visual_path: str = None, *args, **kwargs) -> str:
        """
        format the input and call the ChatGPT/GPT-4 API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            role_desc_in_transition_turn: the description of the role of the agent in the transition turn
            role_desc_after_transition_turn: the description of the role of the agent after the transition turn
            transition_turn: the turn number of the transition
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            visual_path: the path of the visual information for the agent
            second_visual_path: the path of the second visual information for the agent
            request_msg: the request from the system to guide the agent's next response
        """

        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"{global_prompt.strip()}\n\nYour name: {agent_name}\n\nYour role: {role_desc}"
            system_prompt_in_transition_turn = f"{global_prompt.strip()}\n\nYour name: {agent_name}\n\nYour role: {role_desc_in_transition_turn}"
            system_prompt_after_transition_turn = f"{global_prompt.strip()}\n\nYour name: {agent_name}\n\nYour role: {role_desc_after_transition_turn}"
        else:
            system_prompt = f"You are {agent_name}.\n\nYour role: {role_desc}"
            system_prompt_in_transition_turn = f"You are {agent_name}.\n\nYour role: {role_desc_in_transition_turn}"
            system_prompt_after_transition_turn = f"You are {agent_name}.\n\nYour role: {role_desc_after_transition_turn}"

        all_messages = [(SYSTEM_NAME, system_prompt)]
        all_messages_in_transition_turn = [(SYSTEM_NAME, system_prompt_in_transition_turn)]
        all_messages_after_transition_turn = [(SYSTEM_NAME, system_prompt_after_transition_turn)]
        for msg in history_messages:
            if msg.agent_name == SYSTEM_NAME:
                all_messages.append((SYSTEM_NAME, msg.content))
                all_messages_in_transition_turn.append((SYSTEM_NAME, msg.content))
                all_messages_after_transition_turn.append((SYSTEM_NAME, msg.content))
            else:  # non-system messages are suffixed with the end of message token
                all_messages.append((msg.agent_name, f"{msg.content}{END_OF_MESSAGE}"))
                all_messages_in_transition_turn.append((msg.agent_name, f"{msg.content}{END_OF_MESSAGE}"))
                all_messages_after_transition_turn.append((msg.agent_name, f"{msg.content}{END_OF_MESSAGE}"))

        if request_msg is not None:
            all_messages.append((SYSTEM_NAME, request_msg.content))
            all_messages_in_transition_turn.append((SYSTEM_NAME, request_msg.content))
            all_messages_after_transition_turn.append((SYSTEM_NAME, request_msg.content))
        else:  # The default request message that reminds the agent its role and instruct it to speak
            all_messages.append((SYSTEM_NAME, f"Now you speak, {agent_name}.{END_OF_MESSAGE}"))
            all_messages_in_transition_turn.append((SYSTEM_NAME, f"Now you speak, {agent_name}.{END_OF_MESSAGE}"))
            all_messages_after_transition_turn.append((SYSTEM_NAME, f"Now you speak, {agent_name}.{END_OF_MESSAGE}"))

        # Add the visual information if it exists
        if visual_path is not None and os.path.exists(visual_path):
            base64_image = encode_image(visual_path)
            base64_image_in_transition_turn = encode_image(second_visual_path)
            base64_image_after_transition_turn = encode_image(second_visual_path)
        else:
            base64_image = None
            base64_image_in_transition_turn = None
            base64_image_after_transition_turn = None
            
        messages = []
        
        # print (Fore.GREEN + f"All Messages: \n{all_messages}" + Style.RESET_ALL + "\n")
        # print (Fore.GREEN + f"All Messages In Transition Turn: \n{all_messages_in_transition_turn}" + Style.RESET_ALL + "\n")
        # print (Fore.GREEN + f"All Messages After Transition Turn: \n{all_messages_after_transition_turn}" + Style.RESET_ALL + "\n")

        if len(all_messages) <= transition_turn*2-1:
            final_use_messages = all_messages
            final_base64_image = base64_image
        elif len(all_messages) == transition_turn*2 or len(all_messages) == transition_turn*2+1:
            final_use_messages = all_messages_in_transition_turn
            final_base64_image = base64_image_in_transition_turn
        elif len(all_messages) > transition_turn*2+1:
            final_use_messages = all_messages_after_transition_turn
            final_base64_image = base64_image_after_transition_turn
        else:
            raise ValueError(f"Invalid length of all messages: {len(all_messages)}")
        
        for i, msg in enumerate(final_use_messages):
            if i == 0:
                assert msg[0] == SYSTEM_NAME  # The first message should be from the system
                messages.append({"role": "system", "content": msg[1]})
            else:
                if msg[0] == agent_name:
                    messages.append({"role": "assistant", "content": msg[1]})
                else:
                    if messages[-1]["role"] == "user":  # last message is from user
                        if self.merge_other_agent_as_user:
                            if final_base64_image is not None:
                                messages[-1]["content"] = [{"type": "text", "text": f"{messages[-1]['content']}\n\n[{msg[0]}]: {msg[1]}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_base64_image}"}}]
                            else:
                                messages[-1]["content"] = f"{messages[-1]['content']}\n\n[{msg[0]}]: {msg[1]}"
                        else:
                            if final_base64_image is not None:
                                messages.append({"role": "user", "content": [{"type": "text", "text": f"[{msg[0]}]: {msg[1]}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_base64_image}"}}]})
                            else:
                                messages.append({"role": "user", "content": f"[{msg[0]}]: {msg[1]}"})
                    elif messages[-1]["role"] == "assistant":  # consecutive assistant messages
                        # Merge the assistant messages
                        messages[-1]["content"] = f"{messages[-1]['content']}\n{msg[1]}"
                    elif messages[-1]["role"] == "system":
                        messages.append({"role": "user", "content": f"[{msg[0]}]: {msg[1]}"})
                    else:
                        raise ValueError(f"Invalid role: {messages[-1]['role']}")

        # print (Fore.GREEN + f"Now the agent is {agent_name}" + Style.RESET_ALL)
        # print (Fore.GREEN + f"Messages: \n{messages}" + Style.RESET_ALL)
        # input("Press Enter to continue...")
        

        response = self._get_response(messages, *args, **kwargs)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()
        response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()
        # Remove the tailing end of message token
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
        
        return response
