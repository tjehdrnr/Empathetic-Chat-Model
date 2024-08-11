import logging, uuid
import time
from backend.message import Message
from typing import Optional, Dict, Union


LEN_ID = 8
PREFIX_USER      = "### 명령어: "
PREFIX_ASSISTANT = "### 응답: "

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentStore:

    def __init__(self, user_id: Optional[str] = None, **kwargs):
        self.start_time = time.time()
        self.messages = []
        self.history = []
        if user_id is not None:
            self.session_id = \
                str(uuid.uuid5(uuid.NAMESPACE_DNS, user_id))[:LEN_ID]
        # Maximum number of messages to be stored
        self.max_messages = kwargs.get('max_messages', 100)
        assert self.max_messages % 2 == 0

    
    def add(self, role: str, content: str) -> Message:
        """
        Add new message and conversation history
        """
        self._check_capacity()

        if self.session_id is not None:
            message = Message(role, content, self.session_id)
        else:
            message = Message(role, content)
        self.messages.append(message)
        
        current_time = time.time()
        relative_time = current_time - self.start_time

        logger.info(f"Added message: {message}")
        
        # Add history when assistant turn
        if role == 'assistant':
            pair = self.messages[-2:]
            self.history.append(
                {    
                    "text": PREFIX_USER + pair[0].message['content'] + '\n' + \
                            PREFIX_ASSISTANT + pair[1].message['content'],
                    "relative_time": relative_time,
                    "_id": pair[0].metadatas['_id'], # user input '_id'
                }

            )
            logger.info(f"Updated history: {self.history[-1]}")
        
        return message

    
    def clear(self):
        """
        Delete all messages and conversation history
        """
        self.start_time = time.time()
        self.messages = []
        self.history = []

        logger.info(f"Cleared all messages and history.")

            
    def delete(self, _id: str) -> Union[int, None]:
        """
        Delete pair of user input and assistant response.
        If user clicked delete button, this method receives message's id(metadatas: '_id').
        """
        # Find the index of the message to delete
        msg_trg_idx, his_trg_idx = None, None
        for i, obj in enumerate(self.messages):
            if obj.metadatas['_id'] == _id:
                msg_trg_idx = i
                break
        
        # Find the index of the history to delete
        for i, dic in enumerate(self.history):
            if isinstance(dic, Dict) and  dic['_id'] == _id:
                his_trg_idx = i
                break
        
        if msg_trg_idx is not None and his_trg_idx is not None:
            if msg_trg_idx + 1 < len(self.messages):
                _user_input = self.messages[msg_trg_idx]
                _response = self.messages[msg_trg_idx + 1]
                
                assert(_user_input.message['role'] == 'user')
                assert(_response.message['role'] == 'assistant')

                self.messages = self.messages[:msg_trg_idx] + self.messages[msg_trg_idx + 2:]
                logger.info(f"Deleted user message: {_user_input}")
                logger.info(f"Deleted assistant message: {_response}")
            else:
                raise IndexError(f"Index out of bounds. Please check the index value")
            _history = self.history[his_trg_idx]

            self.history[his_trg_idx] = ""
            logger.info(f"Deleted history: {_history}")

            return his_trg_idx
        else:
            raise ValueError(f"Can not find the index of '{_id}'")
        

    def count(self) -> Dict:
        """
        Returns the number of messages and conversation history for each session
        """
        user_lengths, ai_lengths = [], []
        for obj in self.messages:
            if obj.message['role'] == 'user':
                user_lengths.append(len(obj.message['content'].split()))
            else:
                ai_lengths.append(len(obj.message['content'].split()))

        return {
            'n_messages': len(self.messages),
            'n_history': sum([isinstance(h, Dict) for h in self.history]),
            'user_avg': sum(user_lengths) // len(user_lengths) if user_lengths else 0,
            'assistant_avg': sum(ai_lengths) // len(ai_lengths) if ai_lengths else 0,
            'user_max': max(user_lengths) if user_lengths else 0,
            'user_min': min(user_lengths) if user_lengths else 0,
            'assistant_max': max(ai_lengths) if ai_lengths else 0,
            'assistant_min': min(ai_lengths) if ai_lengths else 0,
        }
    

    def _check_capacity(self):
        """
        If message capacity is full, delete half of past messages and history
        """
        if self.messages and len(self.messages) % self.max_messages == 0:
            n_delete = self.max_messages // 2
            self.messages = self.messages[-n_delete:]
            self.history = self.history[-n_delete // 2:]

            logger.info(
                f"Message capacity is full. Deleted past {n_delete} messages.")
