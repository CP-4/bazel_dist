import sys
import csv
import argparse
from threading import Semaphore
import threading, collections, queue, os, os.path
import wavTranscriber
import socket
import time
import numpy as np
from timeit import default_timer as timer
from datetime import datetime
from pytz import timezone

audio_queue = queue.Queue()
audio_length_queue = queue.Queue()
inference_time_queue = queue.Queue()

n_thread = 0
n_audio = 0
i_audio = 0
recv_n_audio = 0

_fetch_audio_sema = Semaphore(1)
_read_write_audio_length_sema = Semaphore(1)
_read_write_inference_time_sema = Semaphore(1)
_read_write_i_audio_sema = Semaphore(1)
_model_loded = queue.Queue()


def start_server():
    HOST = 'localhost'
    PORT = 8070
    recv_n_audio = 0

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind((HOST, PORT))
    except socket.error as msg:
        s.close()
        print('Bind Fail Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
        sys.exit()

    s.listen(1)
    # print('listening...')
    while recv_n_audio < n_audio:

        recv_n_audio += 1
        c, addr = s.accept()
        # print ('Got connection from', addr)
        # send a thank you message to the client.
        c.send(str.encode('connected'))

        audio = bytearray()

        while True:
            l = c.recv(1024)
            if not l: break
            audio.extend(l)

        # with open('output.wav','wb') as f:
        #     f.write(audio)

        audio_queue.put(audio)
        print("added to Queue: " + str(audio_queue.qsize()))
        # print(audio_queue.qsize())

        # Close the connection with the client
        c.close()

    print('server_thread end')


def start_client(n_audio, audio_path='audio.wav'):

    port = 8070
    print(audio_path)
    for i in range(n_audio):
        s = socket.socket()
        s.connect(('127.0.0.1', port))
        rec = s.recv(1024)

        if rec == b'connected':
            with open(audio_path, 'rb') as f:
                for l in f: s.sendall(l)
        s.close()
    print('client_thread end')


def check_i_audio():
    global i_audio
    if i_audio < n_audio:
        i_audio += 1
        return True
    else:
        return False


def fetch_audio():

    threadName = threading.currentThread().name

    while True:
        if not audio_queue.empty():
            # print(threadName + ':' + 'audio_queue_before_get:' + str(audio_queue.qsize()))

            audio = audio_queue.get()
            return audio


def test_ds_inst(model_path):

    threadName = threading.currentThread().name
    sample_rate = 16000
    # Point to a path containing the pre-trained models & resolve ~ if used
    dirName = os.path.expanduser(args.model)
    # print(dirName)

    # Resolve all the paths of model files
    output_graph, alphabet, lm, trie = wavTranscriber.resolve_models(dirName)
    # print(output_graph, alphabet, lm, trie)

    # Load output_graph, alpahbet, lm and trie
    model_retval = wavTranscriber.load_model(output_graph, alphabet, lm, trie)
    print(threadName + ': Model loded . . . ')

    _model_loded.put(threadName)

    run_ds = False
    with _read_write_i_audio_sema:
        run_ds = check_i_audio()

    while run_ds:
        # audio, sample_rate, audio_length = wavTranscriber.read_wave('audio.wav')

        with _fetch_audio_sema:
            audio = fetch_audio()

        # print(threadName + ':' + 'audio_queue_after_get:' + str(audio_queue.qsize()))
        audio = np.frombuffer(audio, dtype=np.int16)
        output = wavTranscriber.stt(model_retval[0], audio, sample_rate)
        # print(threadName + ':' + output[0])
        print('Inference took %0.3fs for %0.3fs audio file.' % (output[2], output[1]))
        print(i_audio, n_audio)

        with _read_write_audio_length_sema:
            audio_length_queue.put(output[1])

        with _read_write_inference_time_sema:
            inference_time_queue.put(output[2])

        with _read_write_i_audio_sema:
            run_ds = check_i_audio()

    print(threadName + ' end')

if __name__ == '__main__':

    args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Test Suite for DeepSpeech')
    parser.add_argument('--model', required=True,
                        help='Path to directory that contains all model files (output_graph, lm, trie and alphabet)')
    parser.add_argument('--nthread', required=True,
                        help='Number of DeepSpeech instances to initialize')
    parser.add_argument('--audio', required=False,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--naudio', required=False,
                        help='Number of times the audio file is sent to Deepspeech server')
    args = parser.parse_args()

    # print(n_thread, n_audio)

    arg_thread = int(args.nthread)
    n_audio = int(args.naudio)
    tz = timezone('Asia/Kolkata')

    for n_thread in range (1, arg_thread):

        for _ in range(1):

            thread_start_time = datetime.now(timezone('UTC')).astimezone(tz)

            print('=======================================')
            print('n_thread: ' + str(n_thread))
            print('=======================================')
            ds_thread = [None] * n_thread
            i_audio = 0


            server_thread = threading.Thread(target=start_server, args=(), name='server_thread')
            server_thread.start()

            time.sleep(1)

            client_thread = threading.Thread(target=start_client, args=(n_audio, args.audio, ), name='client_thread')
            client_thread.start()

            abs_inference_time = 0.0
            inference_start = timer()

            for i in range(n_thread):
                ds_thread[i] = threading.Thread(target=test_ds_inst, args=(args.model, ), name='test_DS' + str(i))
                ds_thread[i].start()

            # while _model_loded.qsize() != n_thread:
            #     pass

            
            # print(str(_model_loded.qsize()) + 'models loaded, starting client')
            # client_thread = threading.Thread(target=start_client, args=(n_audio, ), name='client_thread')
            # client_thread.start()

            # abs_inference_time = 0.0
            # inference_start = timer()

            client_thread.join()
            server_thread.join()
            for i in range(n_thread):
                ds_thread[i].join()


            thread_end_time = datetime.now(timezone('UTC')).astimezone(tz)

            inference_end = timer() - inference_start
            abs_inference_time += inference_end

            total_inferene_time = 0
            total_audio_time = 0
            inference_time_list = list(inference_time_queue.queue)
            inference_time_list = [ round(elem, 2) for elem in inference_time_list ]
            audio_length = 0

            n_inference = inference_time_queue.qsize()


            for i in range(inference_time_queue.qsize()):
                inference_time = inference_time_queue.get()
                total_inferene_time += inference_time
                audio_length = audio_length_queue.get()
                total_audio_time += audio_length

            avg_inference_time = total_inferene_time/n_inference


            fieldnames = [thread_start_time, thread_end_time, n_thread, n_audio, round(audio_length, 2), round(avg_inference_time, 2), round(total_audio_time, 2), round(abs_inference_time, 2), round(total_inferene_time, 2)]
            fieldnames.extend(inference_time_list)

            print(fieldnames)
            
            print("updating CSV file")
            with open('test_results.csv', 'a', encoding='utf-8') as test_csv:
                writer = csv.writer(test_csv)
                writer.writerow(fieldnames)
