import asyncio
import logging
from typing import AsyncGenerator, Optional
import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
from openai.types import FileObject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIAssistant:
    def __init__(
        self, assistant_id: str, thread_id: str, stream_tool_outputs: bool
    ) -> None:
        self.client = AsyncOpenAI()
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.uploaded_file: Optional[FileObject] = None
        self.stream_tool_outputs = stream_tool_outputs

    async def _initialize_thread(self) -> None:
        if self.thread_id:
            logger.info(f"Retrieving thread: {self.thread_id}")
            await self.client.beta.threads.retrieve(thread_id=self.thread_id)
        else:
            logger.info("Creating new thread")
            thread = await self.client.beta.threads.create()
            self.thread_id = thread.id
            logger.info(f"New thread created: {self.thread_id}")

    async def upload_file(self, file_path: str) -> FileObject:
        logger.info(f"Uploading file: {file_path}")
        with open(file_path, "rb") as file:
            response = await self.client.files.create(file=file, purpose="assistants")
        logger.info(f"File uploaded: {response.id}")
        self.uploaded_file = response
        return response

    async def chat(self, user_input: str) -> AsyncGenerator[str, None]:
        await self._initialize_thread()
        message_params = {
            "thread_id": self.thread_id,
            "role": "user",
            "content": user_input,
        }
        if self.uploaded_file:
            message_params["attachments"] = [
                {
                    "file_id": self.uploaded_file.id,
                    "tools": [{"type": "code_interpreter"}],
                }
            ]
        logger.info(f"Adding user message to thread: {user_input}")
        await self.client.beta.threads.messages.create(**message_params)
        logger.info("Sending user message to assistant")
        stream = await self.client.beta.threads.runs.create(
            thread_id=self.thread_id, assistant_id=self.assistant_id, stream=True
        )
        logger.info("Receiving assistant messages")
        async for event in stream:
            if event.event == "thread.message.delta":
                if event.data.delta.content:
                    for content in event.data.delta.content:
                        if content.type == "text":
                            yield content.text.value
            elif event.event == "thread.run.step.delta" and self.stream_tool_outputs:
                if event.data.delta.step_details:
                    step_details = event.data.delta.step_details
                    if step_details.type == "tool_calls":
                        for tool_call in step_details.tool_calls:
                            if tool_call.type == "code_interpreter":
                                yield tool_call.code_interpreter.input
                                if tool_call.code_interpreter.outputs:
                                    for output in tool_call.code_interpreter.outputs:
                                        yield output.content


async def main() -> None:
    # Load environment variables from .env file
    load_dotenv()

    thread_id = os.getenv("OPENAI_THREAD_ID", "")  # Get thread_id from env or use empty string
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")  # Get assistant_id from env
    stream_tool_outputs = os.getenv("STREAM_TOOL_OUTPUTS", "True").lower() == "true"  # Get stream_tool_outputs from env

    if not assistant_id:
        raise ValueError("OPENAI_ASSISTANT_ID must be set in the environment or .env file")

    assistant = OpenAIAssistant(
        assistant_id, thread_id, stream_tool_outputs
    )
    user_input = "Hello world!"
    while True:
        user_input = user_input or input("Enter your response (or 'quit' to exit): ")
        if user_input.lower().strip() == "quit":
            break
        print("Assistant: ", end="", flush=True)
        async for message in assistant.chat(user_input):
            print(message, end="", flush=True)
        print("\n")
        user_input = ""


if __name__ == "__main__":
    asyncio.run(main())
