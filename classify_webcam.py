import sys
import os
import numpy as np
import copy
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def predict(image_data):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/trained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    metin1= "  ******************************* SIGN2WORD *******************************"
    metin2= "    Version:              1.0                                              "
    metin3= "    Tasarim tarihi:       10.04.2018                                       "
    metin4= "    Tasarim yeri:         Marmara universitesi, teknoloji fakultesi        "
    metin5= "    Tasarim amaci         Vize Odev Projesi                                "
    metin5= "    Tasarim ekibi:        Fedi Salhi 170214925, Mehmet Aksu 170214925      "
    metin6= "    E-mail:               fadi.salhi@outlook.fr                            "
    metin7= "                                                                           "
    metin8= "  **************************************************************************"
    c = 0
    cap = cv2.VideoCapture(0)
    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if ret:
            x1, y1, x2, y2 = 120, 100, 480, 350
            img_cropped = img[y1:y2, x1:x2]

            c += 1
            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            a = cv2.waitKey(33)
            if i == 4:
                res_tmp, score = predict(image_data)
                res = res_tmp
                i = 0
                if mem == res:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['bos']:
                    if res == 'bosluk':
                        sequence += ' '
                    elif res == 'sil':
                        sequence = sequence[:-1]
                    else:
                        sequence += res
                    consecutive = 0
            i += 1
            cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            cv2.putText(img,'SIGN2WORD V1.0', (70,70),cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,100),3)
            cv2.putText(img,'Fedi Salhi', (450,420),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,100),2)
            cv2.putText(img,'Mehmet Aksu', (430,450),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,100),2)

            
            cv2.putText(img, 'Dogruluk = %.2f' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("SIGN2WORD_VIDEO", img)
            img_sequence = np.zeros((390,1200,3), np.uint8)
            cv2.putText(img_sequence, '%s' % metin1, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),  2)
            cv2.putText(img_sequence, '%s' % metin2, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(img_sequence, '%s' % metin3, (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(img_sequence, '%s' % metin4, (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(img_sequence, '%s' % metin5, (10,190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(img_sequence, '%s' % metin6, (10,230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(img_sequence, '%s' % metin7, (10,270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(img_sequence, '%s' % metin8, (10,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            
            cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,350), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255),2)
            cv2.imshow('SIGN2WORD_YAZI', img_sequence)
        else:
            break
cv2.VideoCapture(0).release()
