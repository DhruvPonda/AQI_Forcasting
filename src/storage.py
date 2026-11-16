"""Storage abstraction supporting both local disk and AWS S3 data lake layouts."""

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import AWS_CONFIG, BASE_DIR, STORAGE_CONFIG
from src.utils import logger

STORAGE_MODE = os.environ.get("STORAGE_MODE", STORAGE_CONFIG.get("mode", "local")).lower()
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", AWS_CONFIG.get("bucket", "aqi-pipeline-default"))
AWS_REGION = os.environ.get("AWS_REGION", AWS_CONFIG.get("region", "ap-south-1"))
LOCAL_BASE_DIR = BASE_DIR / STORAGE_CONFIG.get("local_dir", "data")


class StorageManager:
    """Handles object persistence to either AWS S3 or Local File System."""

    def __init__(self, mode: str = STORAGE_MODE, bucket: str = S3_BUCKET, region: str = AWS_REGION):
        self.mode = mode.lower()
        self.bucket = bucket
        self.region = region
        self._s3_client = None

        if self.mode == "s3":
            try:
                import boto3
                self._s3_client = boto3.client("s3", region_name=self.region)
                logger.info(f"Initialized S3 StorageManager for bucket: {self.bucket} in {self.region}")
            except Exception as e:
                logger.warning(f"Could not initialize S3 client: {e}. Falling back to local storage.")
                self.mode = "local"

        if self.mode == "local":
            LOCAL_BASE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized Local StorageManager at: {LOCAL_BASE_DIR}")

    def _get_local_path(self, key: str) -> Path:
        # Standardize key (strip leading slashes)
        clean_key = key.lstrip("/").replace("\\", "/")
        path = LOCAL_BASE_DIR / clean_key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, key: str, data: Any) -> None:
        """Saves a Python object as JSON."""
        clean_key = key.lstrip("/").replace("\\", "/")
        json_str = json.dumps(data, indent=2, default=str)
        if self.mode == "s3":
            self._s3_client.put_object(
                Bucket=self.bucket,
                Key=clean_key,
                Body=json_str.encode("utf-8"),
                ContentType="application/json",
            )
        else:
            path = self._get_local_path(clean_key)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)

    def load_json(self, key: str) -> Optional[Any]:
        """Loads a JSON object by key."""
        clean_key = key.lstrip("/").replace("\\", "/")
        if self.mode == "s3":
            try:
                resp = self._s3_client.get_object(Bucket=self.bucket, Key=clean_key)
                content = resp["Body"].read().decode("utf-8")
                return json.loads(content)
            except Exception as e:
                logger.debug(f"S3 get_object failed for key {clean_key}: {e}")
                return None
        else:
            path = self._get_local_path(clean_key)
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def save_parquet(self, key: str, df: pd.DataFrame) -> None:
        """Saves a pandas DataFrame to Parquet format."""
        clean_key = key.lstrip("/").replace("\\", "/")
        if not clean_key.endswith(".parquet"):
            clean_key += ".parquet"

        if self.mode == "s3":
            buf = io.BytesIO()
            df.to_parquet(buf, engine="pyarrow", index=False)
            buf.seek(0)
            self._s3_client.put_object(
                Bucket=self.bucket,
                Key=clean_key,
                Body=buf.getvalue(),
                ContentType="application/octet-stream",
            )
        else:
            path = self._get_local_path(clean_key)
            df.to_parquet(path, engine="pyarrow", index=False)

    def load_parquet(self, key: str) -> Optional[pd.DataFrame]:
        """Loads a Parquet file as a pandas DataFrame."""
        clean_key = key.lstrip("/").replace("\\", "/")
        if not clean_key.endswith(".parquet"):
            clean_key += ".parquet"

        if self.mode == "s3":
            try:
                resp = self._s3_client.get_object(Bucket=self.bucket, Key=clean_key)
                buf = io.BytesIO(resp["Body"].read())
                return pd.read_parquet(buf, engine="pyarrow")
            except Exception as e:
                logger.debug(f"S3 get_object failed for parquet key {clean_key}: {e}")
                return None
        else:
            path = self._get_local_path(clean_key)
            if not path.exists():
                return None
            return pd.read_parquet(path, engine="pyarrow")

    def save_bytes(self, key: str, data: bytes) -> None:
        """Saves arbitrary bytes (e.g. pickled model)."""
        clean_key = key.lstrip("/").replace("\\", "/")
        if self.mode == "s3":
            self._s3_client.put_object(
                Bucket=self.bucket,
                Key=clean_key,
                Body=data,
                ContentType="application/octet-stream",
            )
        else:
            path = self._get_local_path(clean_key)
            with open(path, "wb") as f:
                f.write(data)

    def load_bytes(self, key: str) -> Optional[bytes]:
        """Loads arbitrary bytes."""
        clean_key = key.lstrip("/").replace("\\", "/")
        if self.mode == "s3":
            try:
                resp = self._s3_client.get_object(Bucket=self.bucket, Key=clean_key)
                return resp["Body"].read()
            except Exception as e:
                logger.debug(f"S3 get_object failed for bytes key {clean_key}: {e}")
                return None
        else:
            path = self._get_local_path(clean_key)
            if not path.exists():
                return None
            with open(path, "rb") as f:
                return f.read()

    def file_exists(self, key: str) -> bool:
        """Checks if a file exists."""
        clean_key = key.lstrip("/").replace("\\", "/")
        if self.mode == "s3":
            try:
                self._s3_client.head_object(Bucket=self.bucket, Key=clean_key)
                return True
            except Exception:
                return False
        else:
            path = self._get_local_path(clean_key)
            return path.exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        """Lists all keys matching a prefix."""
        clean_prefix = prefix.lstrip("/").replace("\\", "/")
        if self.mode == "s3":
            keys = []
            paginator = self._s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=clean_prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        else:
            base = LOCAL_BASE_DIR / clean_prefix
            if not base.exists():
                return []
            if base.is_file():
                return [clean_prefix]
            keys = []
            for item in base.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(LOCAL_BASE_DIR).as_posix()
                    keys.append(rel)
            return keys

    def get_storage_stats(self) -> Dict[str, Any]:
        """Calculates storage summary metrics for monitoring."""
        if self.mode == "s3":
            total_size = 0
            file_count = 0
            try:
                paginator = self._s3_client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self.bucket):
                    for obj in page.get("Contents", []):
                        file_count += 1
                        total_size += obj["Size"]
            except Exception as e:
                logger.error(f"Failed to query S3 bucket stats: {e}")
            return {
                "mode": "s3",
                "bucket": self.bucket,
                "region": self.region,
                "file_count": file_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
        else:
            file_count = 0
            total_size = 0
            if LOCAL_BASE_DIR.exists():
                for item in LOCAL_BASE_DIR.rglob("*"):
                    if item.is_file():
                        file_count += 1
                        total_size += item.stat().st_size
            return {
                "mode": "local",
                "local_dir": str(LOCAL_BASE_DIR),
                "file_count": file_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }


# Default singleton instance
storage = StorageManager()
