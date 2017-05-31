# -*- coding: utf-8 -*-

import boto3


def sqs_send_msg(data, queue_name='360_nlp_output', online=True):
    sqs = boto3.resource('sqs') if online else boto3.resource('sqs', endpoint_url='http://192.168.27.190:4576')
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    queue.send_message(MessageBody=data)


def sqs_get_msgs(queue_name='360_nlp_input', online=True):
    sqs = boto3.resource('sqs') if online else boto3.resource('sqs', endpoint_url='http://192.168.27.190:4576')
    queues = sqs.get_queue_by_name(QueueName=queue_name, )
    return queues


def s3_get_file(bucket, key, file, online=True):
    s3 = boto3.client('s3') if online else boto3.resource('s3', endpoint_url='http://192.168.27.190:4572')
    file = s3.download_file(bucket, key, file)
    return file