# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf, uos, gc
import time

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((320, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("Road_QuantPruned.tflite", load_to_fb=uos.stat('Road_QuantPruned.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "Road_QuantPruned.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    # default settings just do one detection... change them to search the image...
    for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        img.draw_rectangle(obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))
        l1,l2 = [],[]
        for i in range(len(predictions_list)):
            #print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))
            l1.append(predictions_list[i][1])
            l2.append(predictions_list[i][0])
    print("The sign is",l2[l1.index(max(l1))])
   # print("The confidence interval is",max(l1))
   # print(clock.fps(), "fps")
#    time.sleep(0.5)

