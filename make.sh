#!/bin/bash

NAME=*
EXENAME=detect_track

g++ -std=c++11 -I/opt/pylon/include -Wl,--enable-new-dtags -Wl,-rpath,/opt/pylon/lib -L/opt/pylon/lib -Wl,-E -lpylonbase -lpylonutility -lGenApi_gcc_v3_1_Basler_pylon -lGCBase_gcc_v3_1_Basler_pylon -I/usr/local/include/opencv -I/usr/local/include -Wl,--enable-new-dtags -Wl,-rpath,/usr/local/lib -L/usr/local/lib -Wl,-E -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_dnn -lopencv_highgui -lopencv_tracking $NAME.cpp -o $EXENAME
