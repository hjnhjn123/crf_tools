# -*- coding: utf-8 -*-

import os

import boto3


def get_boto3_client(service, **kwargs):
    s3_config = {}
    if os.environ.get('AWS_ACCESS_KEY_ID'):
        s3_config['aws_access_key_id'] = os.environ.get('AWS_ACCESS_KEY_ID')
    if os.environ.get('AWS_SECRET_ACCESS_KEY'):
        s3_config['aws_secret_access_key'] = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if s3_config.get('AWS_ACCESS_REGION'):
        s3_config['region_name'] = s3_config.get('AWS_ACCESS_REGION')
    else:
        s3_config['region_name'] = 'us-east-1'

    if os.environ.get('NLP_MODE') and os.environ.get('NLP_MODE') == 'dev':
        if service.lower() == 's3':
            s3_config['endpoint_url'] = os.environ['NLP_DEV_S3_ENDPOINT_URL']
        if service.lower() == 'sqs':
            s3_config['endpoint_url'] = os.environ['NLP_DEV_SQS_ENDPOINT_URL']
    return boto3.client(service, **s3_config)


def get_boto3_service(service):
    s3_config = {}
    if os.environ.get('AWS_ACCESS_KEY_ID'):
        s3_config['aws_access_key_id'] = os.environ.get('AWS_ACCESS_KEY_ID')
    if os.environ.get('AWS_SECRET_ACCESS_KEY'):
        s3_config['aws_secret_access_key'] = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if s3_config.get('AWS_ACCESS_REGION'):
        s3_config['region_name'] = s3_config.get('AWS_ACCESS_REGION')
    else:
        s3_config['region_name'] = 'us-east-1'

    if os.environ.get('NLP_MODE') and os.environ.get('NLP_MODE') == 'dev':
        if service.lower() == 's3':
            s3_config['endpoint_url'] = os.environ['NLP_DEV_S3_ENDPOINT_URL']
        if service.lower() == 'sqs':
            s3_config['endpoint_url'] = os.environ['NLP_DEV_SQS_ENDPOINT_URL']
    return boto3.resource(service, **s3_config)


def sqs_send_msg(data, queue_name='360_nlp_output'):
    sqs = get_boto3_service('sqs')
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    queue.send_message(MessageBody=data)


def sqs_get_msgs(queue_name='360_nlp_input'):
    sqs = get_boto3_service('sqs')
    queues = sqs.get_queue_by_name(QueueName=queue_name)
    return queues


def s3_get_file(bucket, key, file):
    s3 = get_boto3_service('s3')
    bucket = s3.Bucket(bucket)
    with open(file, 'wb') as data:
        bucket.download_fileobj(key, data)
