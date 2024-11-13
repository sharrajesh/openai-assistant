import asyncio
import logging
import os
from pathlib import Path
from tempfile import gettempdir
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types import FileObject

from s3_uploader import S3Uploader

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
        self.s3_uploader = S3Uploader(
            os.getenv("AWS_ACCESS_KEY_ID"),
            os.getenv("AWS_SECRET_ACCESS_KEY"),
            os.getenv("S3_BUCKET_NAME"),
            os.getenv("AWS_REGION", "us-east-1"),
        )

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
                                if tool_call.code_interpreter.input:
                                    if self.stream_tool_outputs:
                                        yield tool_call.code_interpreter.input
                                if tool_call.code_interpreter.outputs:
                                    for output in tool_call.code_interpreter.outputs:
                                        if output.type == "logs":
                                            print(output.logs, end="", flush=True)
                                        else:
                                            logger.info(f"Output: {output.value}")
            elif event.event == "thread.message.completed":
                message = event.data
                for content_block in message.content:
                    if content_block.text.annotations:
                        for annotation in content_block.text.annotations:
                            object_name = annotation.text.split("/")[-1]
                            if annotation.type == "file_path":
                                yield await self._create_s3_download_url(
                                    annotation.file_path.file_id, object_name
                                )
                            elif annotation.type == "file":
                                yield await self._create_s3_download_url(
                                    annotation.file_id, object_name
                                )

    async def _upload_to_s3(self, file_path: str) -> str:
        return await self.s3_uploader.upload_file(file_path)

    async def _create_s3_download_url(self, file_id: str, object_name: str) -> str:
        file_path = Path(gettempdir()) / object_name
        logger.info(f"Creating temporary file: {file_path}")
        try:
            file_data = await self.client.files.content(file_id)
            file_path.write_bytes(file_data.read())
            s3_url = await self._upload_to_s3(str(file_path))
            return f"\n\nDownload file: [{object_name}]({s3_url})\n"
        except Exception as e:
            logger.error(f"Error handling file: {e}")
            return f"\nError creating download link: {str(e)}\n"
        finally:
            try:
                file_path.unlink(missing_ok=True)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


async def main() -> None:
    load_dotenv()

    thread_id = os.getenv("OPENAI_THREAD_ID", "")
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")  # Get assistant_id from env
    stream_tool_outputs = True
    if not assistant_id:
        raise ValueError(
            "OPENAI_ASSISTANT_ID must be set in the environment or .env file"
        )
    assistant = OpenAIAssistant(assistant_id, thread_id, stream_tool_outputs)
    user_input = "Create a hello world python scenario"
    while True:
        print("Assistant: ", end="", flush=True)
        async for message in assistant.chat(user_input):
            print(message, end="", flush=True)
        print("\n")
        user_input = input("Enter your response (or 'quit' to exit): ")
        if user_input.lower().strip() == "quit":
            break
    print(f"Conversation ended. Final thread ID: {assistant.thread_id}")


if __name__ == "__main__":
    asyncio.run(main())
