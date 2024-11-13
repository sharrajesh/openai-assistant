import asyncio
import logging
import os

import aioboto3
from botocore.config import Config

logger = logging.getLogger(__name__)


class S3Uploader:
    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        bucket_name: str,
        region: str,
    ):
        self.session = aioboto3.Session()
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
        self.aws_config = Config(signature_version="s3v4", region_name=region)

    async def upload_file(self, file_path: str) -> str:
        object_name = os.path.basename(file_path)
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                config=self.aws_config,
            ) as s3_client:
                await s3_client.upload_file(file_path, self.bucket_name, object_name)
                url = await s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": object_name},
                    ExpiresIn=3600,
                )
                return url
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise


async def main():
    from dotenv import load_dotenv

    load_dotenv()

    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket = os.getenv("S3_BUCKET_NAME")

    if not all([aws_key, aws_secret, bucket]):
        print("Missing environment variables:")
        print(f"AWS_ACCESS_KEY_ID: {'✓' if aws_key else '✗'}")
        print(f"AWS_SECRET_ACCESS_KEY: {'✓' if aws_secret else '✗'}")
        print(f"S3_BUCKET_NAME: {'✓' if bucket else '✗'}")
        return

    uploader = S3Uploader(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        bucket_name=bucket,
        region=os.getenv("AWS_REGION", "us-east-1"),
    )

    test_file = "test.txt"
    with open(test_file, "w") as f:
        f.write("Test content")

    try:
        url = await uploader.upload_file(test_file)
        print(f"File uploaded successfully. Download URL: {url}")
    except Exception as e:
        print(f"Error uploading file: {e}")
    finally:
        os.remove(test_file)


if __name__ == "__main__":
    asyncio.run(main())
