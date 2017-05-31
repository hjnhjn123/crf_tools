# -*- coding: utf-8 -*-

import boto3


def sqs_send_msg(data, queue_name='360_nlp_output', online='aws'):
    sqs = boto3.resource('sqs') if online == 'aws' else boto3.resource('sqs', endpoint_url='http://192.168.27.190:4576')
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    queue.send_message(MessageBody=data)


def sqs_get_msgs(queue_name='360_nlp_input', online='aws'):
    sqs = boto3.resource('sqs') if online == 'aws' else boto3.resource('sqs', endpoint_url='http://192.168.27.190:4576')
    queues = sqs.get_queue_by_name(QueueName=queue_name, )
    return queues


def s3_get_file(bucket, key, file, online='aws'):
    s3 = boto3.resource('s3') if online == 'aws' else boto3.resource('s3', endpoint_url='http://192.168.27.190:4572')
    bucket = s3.Bucket(bucket)
    with open(file, 'wb') as data:
        bucket.download_fileobj(key, data)
