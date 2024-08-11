import uuid
import time
from typing import Optional


LEN_ID=8

class Message:
    __slots__ = ["message", "metadatas"]

    def __init__(self, role: str, content: str, session_id: Optional[str] = None):

        self.message = {"role": role, "content": content}
        self.metadatas = {
            "_id": str(uuid.uuid4())[:LEN_ID],
            "timestamp": time.time(),
        }
        if session_id is not None:
            self.metadatas["session_id"] = session_id
    
    def __repr__(self):
        return (
            f"Message(message={self.message}, "
            f"metadatas={self.metadatas})"
        )