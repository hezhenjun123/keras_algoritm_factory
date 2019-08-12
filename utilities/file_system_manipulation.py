import os
import boto3


def is_s3_path(path):
    return path.startswith('s3')

def split_s3_path(s3_path):
    r"""
    Splits S3 path into bucket and key pair
    ----------
    s3_path: str
        S3 path to split
    """
    if not is_s3_path(s3_path):
        raise ValueError('Incorrect s3 path provided, must start with s3://')
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key

def directory_to_file_list(path):
    if is_s3_path(path):
        if os.path.splitext(path)[1]!='':
            paths = [path]
        else:
            bucket,key = split_s3_path(path)
            s3 = boto3.client('s3')
            s3_files = []
            continuation_token =  None
            #gets all objects since max returned by 1 call is 1000
            while True:
                tkn={}
                if continuation_token is not None:
                  tkn['ContinuationToken']=continuation_token
                response = s3.list_objects_v2(Bucket=bucket,
                                            Prefix=key,
                                            MaxKeys=1000,
                                            **tkn)
                s3_files.extend(response['Contents'])
                if not response['IsTruncated']:
                    break
                continuation_token = response['NextContinuationToken']

            paths = [os.path.join('s3://',bucket,s3_path['Key'])
                     for s3_path in s3_files
                     if os.path.splitext(s3_path['Key'])[1]!='']
    else:
        if not os.path.isdir(path):
            paths = [path]
        else:
            paths = []
            for (dirpath, dirnames, filenames) in os.walk(path):
                paths.extend(os.path.join(dirpath, filename) for filename in filenames)
    return paths


def s3_to_local(s3_path,local_path):
    if not is_s3_path(s3_path):
        raise ValueError('Incorrect s3 path provided, must start with s3://')
    return s3_dir_to_local(s3_path,local_path)

def s3_dir_to_local(s3_path,local_path):
    if not is_s3_path(s3_path):
        raise ValueError('Incorrect s3 path provided, must start with s3://')
    if not local_path.endswith('/'): local_path+='/'
    s3_paths =  directory_to_file_list(s3_path)
    client = boto3.client("s3")
    downloaded_files = []
    for s3_file_path in s3_paths:
        local_file_path = s3_file_path.replace(s3_path,local_path)
        if local_file_path.endswith('/'): local_file_path=local_file_path[:-1]
        s3_file_to_local(s3_file_path,local_file_path,client)
        downloaded_files.append(local_file_path)
    return downloaded_files

def s3_file_to_local(s3_path,local_path=None,client=None):
    r"""
    Downloads file from S3 to /tmp/ directory locally
    Parameters
    ----------
    file_path: str
        Path to a file on S3
    """
    if not is_s3_path(s3_path):
        raise ValueError('Incorrect s3 path provided, must start with s3://')
    if local_path is None:
        local_path = os.path.join("/tmp", os.path.basename(s3_path))
    os.makedirs(os.path.dirname(local_path),exist_ok=True)
    bucket_name, file_path = split_s3_path(s3_path)
    if  client is None : client = boto3.client("s3")
    client.download_file(bucket_name, file_path, local_path)
    if not os.path.exists(path=local_path):
        raise IOError(
            "File {} not found, download failed".format(local_path))
    

def local_to_s3(local_path, s3_path):
    r"""
    Uploads local file or directory to S3
    Parameters
    ----------
    local_path: str
        Path to a file/directory to upload
    s3_path: str
        S3 path to upload to
    """
    if is_s3_path(local_path):
        raise ValueError('local_path must not be an s3 path')

    if os.path.isdir(local_path):
        return local_dir_to_s3(local_path, s3_path)
    else:
        return local_file_to_s3(local_path, s3_path)


def local_dir_to_s3(local_path, s3_path):
    r"""
    Uploads files from local directory to S3
    Parameters
    ----------
    local_path: str
        Path to a directory to upload
    s3_path: str
        S3 path to upload to
    """
    if is_s3_path(local_path):
        raise ValueError('local_path must not be an s3 path')
    if not os.path.isdir(local_path):
        raise ValueError('provided local path is not a directory')
    if not local_path.endswith('/'): local_path+='/'
    local_paths = directory_to_file_list(local_path)
    client = boto3.client("s3")
    # make sure the order is the same every time
    local_paths = list(sorted(local_paths))
    uploaded_files = []
    for full_path in local_paths:
        upload_path = os.path.join(s3_path, full_path.replace(local_path, ''))
        local_file_to_s3(full_path, upload_path, client)
        uploaded_files.append(upload_path)
    return uploaded_files


def local_file_to_s3(local_path, s3_path, client=None):
    r"""
    Uploads local file to S3
    Parameters
    ----------
    local_path: str
        Path to a file to upload
    s3_path: str
        S3 path to upload to including full file name
    bucket: boto3 Bucket object
        Optional bucket object for uploading
        Gets created automatically if not provided 
    """
    if is_s3_path(local_path):
        raise ValueError('local_path must not be an s3 path')
    if os.path.isdir(local_path):
        raise ValueError('provided local path is a directory not a file')
    bucket_name, key = split_s3_path(s3_path)
    if client is None : client = boto3.client("s3")
    with open(local_path, "rb") as data:
        print(f"Uploading {local_path} to {s3_path}")
        client.put_object(Key=key, Body=data,Bucket=bucket_name)
    return s3_path

def copy_file(source_path,destination_path):
    if is_s3_path(source_path)  and is_s3_path(destination_path):
        raise ValueError('Moving files on s3 is not supported')
    elif is_s3_path(source_path):
        return s3_to_local(source_path,destination_path)
    elif is_s3_path(destination_path):
        return local_to_s3(source_path,destination_path)
    else:
        raise ValueError('Moving files locally is not supported')
