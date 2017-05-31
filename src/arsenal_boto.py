# -*- coding: utf-8 -*-

import boto3

def sqs_send_a_msg(data, queue_name='360_nlp_output', online=True):
    sqs = boto3.resource('sqs') if online else boto3.resource('sqs', endpoint_url='http://192.168.27.190:4576')
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    queue.send_message(MessageBody=data)


def sqs_get_msgs(queue_name='360_nlp_input', online=True):
    sqs = boto3.resource('sqs') if online else boto3.resource('sqs', endpoint_url='http://192.168.27.190:4576')
    queues = sqs.get_queue_by_name(QueueName=queue_name, )
    return queues