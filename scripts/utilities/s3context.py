import logging
import os
from pathlib import Path
import shutil
import tempfile

import boto3

def parse_s3_path(s3_path):
    assert s3_path.startswith("s3://")
    s3_path = s3_path.replace("s3://", "")
    parts = s3_path.split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    fname = parts[-1]
    return bucket, key, fname


class S3FileWriter(object):
    """ 1. Map s3 path to local path, return local path
        2. Write to local path
        3. On exit of context manager save to s3
    """

    def __init__(self, s3_path):
        if s3_path.startswith("s3"):
            s3_path_parts = parse_s3_path(s3_path)
            self.bucket = s3_path_parts[0]
            self.key = s3_path_parts[1]
            self.fname = s3_path_parts[2]
        else:
            self.final_file = s3_path
            self.fname = s3_path.split("/")[-1]
            Path(self.final_file).parent.mkdir(parents=True, exist_ok=True)

        self.tmpdir = tempfile.mkdtemp()
        self.local_file = "{}/{}".format(self.tmpdir, self.fname)

    def __enter__(self):
        return self.local_file

    def __exit__(self, type, value, traceback):
        if hasattr(self, "bucket"):
            s3 = boto3.resource("s3")
            bucket = s3.Bucket(self.bucket)
            bucket.upload_file(self.local_file, "{}".format(self.key))
        else:
            shutil.move(self.local_file, self.final_file)
        shutil.rmtree(self.tmpdir)


class S3Cache:
    def __init__(self, s3_path, local_path):
        # Convert path to string for compatibility with pathlib
        s3_path = str(s3_path)
        local_path = str(local_path)
        if s3_path.startswith("s3"):
            s3_path_parts = parse_s3_path(s3_path)
            self.bucket = s3_path_parts[0]
            self.key = s3_path_parts[1]
            if self.key:
                self.prefix = "s3://{}/{}".format(self.bucket, self.key)
            else:
                self.prefix = "s3://{}".format(self.bucket)
            self.cache_dir = os.path.expanduser(local_path)

    def fetch(self, s3_path, force_download=False):
        s3_path = str(s3_path)
        if not s3_path.startswith(self.prefix):
            logging.warning(
                "cache doesnt understand this path %s. Attempting to interpret it as a local file",
                s3_path,
            )
            return s3_path
        _, key, _ = parse_s3_path(s3_path)
        local_path = s3_path.replace(self.prefix, self.cache_dir)
        if force_download and Path(local_path).exists():
            Path(local_path).unlink()
        if force_download or not os.path.exists(local_path):
            logging.info("cache downloading %s to %s", s3_path, local_path)
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            s3 = boto3.client("s3")
            s3.download_file(self.bucket, key, local_path)
        return local_path

    def sync(self, s3_path):
        _, key, _ = parse_s3_path(s3_path)
        dir_path = s3_path.replace(self.prefix, self.cache_dir)
        local_paths = []
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(self.bucket)
        for object in bucket.objects.filter(Prefix = key):
            if object.key.endswith('/'):
                continue
            local_paths.append(self.fetch('s3://' + self.bucket + '/' + object.key))

        return dir_path

    def upload(self, local_path):
        local_path = os.path.expanduser(local_path)
        assert local_path.startswith(self.cache_dir)
        key = local_path[len(self.cache_dir):]
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(self.bucket)
        bucket.upload_file(local_path, key)

CACHE = S3Cache("s3://transformers-sample/", "~/transformer-samples/")
ZL_CACHE = S3Cache("s3://zoomlion-sample/", "~/zoomlion-sample/")

