# -*- coding: utf-8 -*-

import boto3

def sqs_send_a_msg(data, queue_name='360_nlp_output'):
    sqs = boto3.resource('sqs')
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    queue.send_message(MessageBody=data)


def sqs_get_msgs(queue_name='360_nlp_input'):
    sqs = boto3.resource('sqs')
    queues = sqs.get_queue_by_name(QueueName=queue_name)
    return queues