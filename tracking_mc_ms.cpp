#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/common.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/viz/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <pylon/PylonIncludes.h>

using namespace cv;
using namespace dnn;

using namespace Pylon;
// Namespace for using GenApi objects.
using namespace GenApi;
// Namespace for using cout.
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

// Structures
struct objectInfo
    {
        Rect trackBox;
        float trackConfidence;
        int trackClassId;
    };

// Global variables 
static const size_t c_maxCamerasToUse = 2;


// Function Declarations
float confThreshold, nmsThreshold;
std::vector<std::string> classes;
inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale, bool swapRB);
objectInfo postprocess(Mat& frame, const std::vector<Mat>& out, Net& net, int backend, int masina);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
void callback(int pos, void* userdata);


  
int main(int argc, char** argv) 
{
    confThreshold = 0.5;
    nmsThreshold = 0.4;
    float scale = 0.00392;
    bool swapRB = true;
    int inpWidth = 416;
    int inpHeight = 416;
    std::string modelPath = findFile("/home/pi/darknet/yolov3-tiny_final.weights");
    std::string configPath = findFile("/home/pi/darknet/yolov3-tiny.cfg");
    
    // Sa konzole:
    std::istringstream ss(argv[1]);
    int masina;
    if (!(ss >> masina)) {
        std::cerr << "Invalid number: " << argv[1] << '\n';
    } else if (!ss.eof()) {
        std::cerr << "Trailing characters after number: " << argv[1] << '\n';
    }
    
    std::string file = "/home/pi/darknet/classes.names";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
      CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
      classes.push_back(line);
    }
    
    // Load a model.
    Net net = readNet(modelPath, configPath);
    int backend = 0;
    net.setPreferableBackend(backend);
    //net.setPreferableTarget(parser.get<int>("target"));
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    
    // Create a window
    static const std::string kWinName = "Basler dart / rpi4 detection tracking";
    namedWindow(kWinName, WINDOW_NORMAL);
    setWindowProperty(kWinName, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    int initialConf = (int)(confThreshold * 100);
    createTrackbar("Confidence threshold [%]", kWinName, &initialConf, 99, callback);
    
    // Tracker
    Ptr<Tracker> tracker0 = TrackerMedianFlow::create();
    Ptr<Tracker> tracker1 = TrackerMedianFlow::create();
 
    // Initialization of vectors
    objectInfo noObject = {};
    vector<objectInfo> objectNew;
    objectNew.push_back(objectInfo());
    objectNew.push_back(objectInfo());
    vector<objectInfo> objectOld;
    objectOld.push_back(objectInfo());
    objectOld.push_back(objectInfo());
    std::vector<Rect> bbox;
    Rect bboxIni0;
    Rect bboxIni1;
    bbox.push_back(bboxIni0);
    bbox.push_back(bboxIni1);
    std::vector<float> confidence;
    confidence.push_back(0);
    confidence.push_back(0);
    std::vector<int> classId;
    classId.push_back(0);
    classId.push_back(0);
    Rect2d bbox0;
    Rect2d bbox1;
  
  
  
    bool lost = true;
    bool camFound0 = false;
    bool camFound1 = false;
    int tick = 0;
    
    
    
    // Before using any pylon methods, the pylon runtime must be initialized. 
    PylonInitialize();
    
    // Get the transport layer factory.
    CTlFactory& tlFactory = CTlFactory::GetInstance();
      
    // Get all attached devices and exit application if no device is found.
    DeviceInfoList_t devices;
    if ( tlFactory.EnumerateDevices(devices) == 0 )
    {
        throw RUNTIME_EXCEPTION( "No camera present.");
    }

    // Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    CInstantCameraArray cameras( min( devices.size(), c_maxCamerasToUse));
    
    // Stereo-vision
    String_t reverseCamera = "23358352"; 
    bool alternate = false;
    
    // Create and attach all Pylon Devices.
    for ( size_t i = 0; i < cameras.GetSize(); ++i)
    {
        cameras[ i ].Attach( tlFactory.CreateDevice( devices[ i ]));
        // Print the model name of the camera.
        cout << "Using device: " << cameras[ i ].GetDeviceInfo().GetModelName() << ", SN: " << cameras[ i ].GetDeviceInfo().GetSerialNumber() << endl;
        cameras[ i ].Open();
        
        INodeMap& nodemap = cameras[ i ].GetNodeMap();
        
        if (devices[ i ].GetSerialNumber() == reverseCamera)
        {
            CBooleanPtr(nodemap.GetNode("ReverseX"))->SetValue(true);
            // Enable Reverse Y, if available
            CBooleanPtr(nodemap.GetNode("ReverseY"))->SetValue(true);
        }
        
        CIntegerParameter( nodemap, "Width").SetValue( 1600, IntegerValueCorrection_Nearest);
        CIntegerParameter( nodemap, "Height").SetValue( 1200, IntegerValueCorrection_Nearest);
        CEnumParameter(nodemap, "PixelFormat").SetValue("RGB8");
        CEnumParameter(nodemap, "OverlapMode").SetValue("Off");

        
        cameras[ i ].Close();
    }
    
    cameras[0].StartGrabbing(GrabStrategy_LatestImages);
    cameras[1].StartGrabbing(GrabStrategy_LatestImages);

    cout << "Please wait. Images are being grabbed." << endl;
    // This smart pointer will receive the grab result data.
    CGrabResultPtr ptrGrabResult1;
    CGrabResultPtr ptrGrabResult2;
    // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
    // when c_countOfImagesToGrab images have been retrieved.

    while ( cameras[0].IsGrabbing())
    {
        
        
        // Start timer
        //double timer = (double)getTickCount();  
        // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        cameras[0].RetrieveResult( 5000, ptrGrabResult1, TimeoutHandling_ThrowException);
        cameras[1].RetrieveResult( 5000, ptrGrabResult2, TimeoutHandling_ThrowException);
        void *slika1 = ptrGrabResult1->GetBuffer();
        void *slika2 = ptrGrabResult2->GetBuffer();

        Mat frameFull1(1200,1600, CV_8UC3, slika1);
        Mat frame1;
        resize(frameFull1, frame1, Size(416,312));
        cvtColor(frame1, frame1, COLOR_BGR2RGB);
    
        Mat frameFull2(1200,1600, CV_8UC3, slika2);
        Mat frame2;
        resize(frameFull2, frame2, Size(416,312));
        cvtColor(frame2, frame2, COLOR_BGR2RGB);
        
        std::vector<Mat> frameArr;
        frameArr.push_back(frame1);
        frameArr.push_back(frame2);
        
        
        if (lost == true)
        {
            putText(frameArr[0], "Detecting Machine...", Point(30,60), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,0,0), 0.8);

            for (int cam = 0; cam < 2; cam++)
            {
            
                preprocess(frameArr[cam], net, Size(inpWidth, inpHeight), scale, swapRB);
                
                std::vector<Mat> outs;
                net.forward(outs, outNames);
                       
                objectNew[cam] = postprocess(frameArr[cam], outs, net, backend, masina);

                cout << "New bbox coord: " << objectNew[cam].trackBox.x << ", "  << objectNew[cam].trackBox.y << " | Old bbox coord: " << objectOld[cam].trackBox.x << ", " << objectOld[cam].trackBox.y << endl;
                
                if (objectNew[cam].trackBox != noObject.trackBox)
                {
                    bbox[cam] = objectNew[cam].trackBox;
                    classId[cam] = objectNew[cam].trackClassId;
                    confidence[cam] = objectNew[cam].trackConfidence;

                    if (cam == 0)
                    {
                        bbox0 = static_cast<Rect_<double>>(bbox[cam]);
                        tracker0 = TrackerMedianFlow::create();
                        tracker0->init(frameArr[cam], bbox0);
                        camFound0 = true;
                    }
                    else
                    {
                        bbox1 = static_cast<Rect_<double>>(bbox[cam]);
                        tracker1 = TrackerMedianFlow::create();
                        tracker1->init(frameArr[cam], bbox1);
                        camFound1 = true;
                    }
                    lost = false;
                }
            }
            cout << camFound0 << "|" << camFound1 << endl;

        }
        else
        {
            // Update the tracking result
            if (camFound0 == true && camFound1 == true)
            {
                bbox0 = static_cast<Rect_<double>>(bbox[0]);
                bbox1 = static_cast<Rect_<double>>(bbox[1]);

                bool ok0 = tracker0->update(frameArr[0], bbox0);
                
                if (ok0)
                {
                    // Tracking success : Draw the tracked object
                    drawPred(classId[0], confidence[0], bbox0.x, bbox0.y, bbox0.x + bbox0.width, bbox0.y + bbox0.height, frameArr[0]);
                }
                else
                {
                    // Tracking failure detected.
                    putText(frameArr[0], "Trackin failure detected!", Point(30,60), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,0,0), 0.8);
                    
                    
                }
                
                bool ok1 = tracker1->update(frameArr[1], bbox1);
                
                if (ok1)
                {
                    // Tracking success : Draw the tracked object
                    drawPred(classId[1], confidence[1], bbox1.x, bbox1.y, bbox1.x + bbox1.width, bbox1.y + bbox1.height, frameArr[1]);
                }
                else
                {
                    // Tracking failure detected.
                    putText(frameArr[1], "Trackin failure detected!", Point(30,60), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,0,0), 0.8);
                    
                    
                }
                if (ok0 == false && ok1 == false)
                {
                    lost = true;
                    camFound0 = false;
                    camFound1 = false;
                    tracker0.release();
                    tracker1.release();
                }
                    
                
            }
            else if(camFound0 == true && camFound1 == false)
            {
                bbox0 = static_cast<Rect_<double>>(bbox[0]);
                
                bool ok0 = tracker0->update(frameArr[0], bbox0);
               

                
                if (ok0)
                {
                    // Tracking success : Draw the tracked object
                    drawPred(classId[0], confidence[0], bbox0.x, bbox0.y, bbox0.x + bbox0.width, bbox0.y + bbox0.height, frameArr[0]);
                }
                else
                {
                    // Tracking failure detected.
                    putText(frameArr[0], "Trackin failure detected!", Point(30,60), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,0,0), 0.8);
            
                    lost = true;
                    camFound0 = false;
                    
                    tracker0.release();
                }

                
            }
            else if(camFound0 == false && camFound1 == true)
            {
                bbox1 = static_cast<Rect_<double>>(bbox[1]);
                
                bool ok1 = tracker1->update(frameArr[1], bbox1);
                
                
                if (ok1)
                {
                    // Tracking success : Draw the tracked object
                    drawPred(classId[1], confidence[1], bbox1.x, bbox1.y, bbox1.x + bbox1.width, bbox1.y + bbox1.height, frameArr[1]);
                }
                else
                {
                    // Tracking failure detected.
                    putText(frameArr[1], "Trackin failure detected!", Point(30,60), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,0,0), 0.8);
            
                    lost = true;
                    camFound1 = false;
                    
                    tracker1.release();
                }
                
            }
            
        }
        putText(frameArr[0], "Tracking: Machine " + SSTR(masina+1), Point(30,30), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,0,0), 0.8);
      
        // Display frame.
        Mat frameMat[] = {frameArr[0], frameArr[1]};
        Mat frameCon;
        hconcat( frameMat, 2, frameCon );
        imshow(kWinName, frameCon);
        
        
      
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
    
    }

    PylonTerminate();

    return 0;
}

inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale, bool swapRB)
{
    static Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;
    blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        resize(frame, frame, inpSize);
        Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

objectInfo postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend, int masina)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    
    std::vector<int> classIdsOut;
    std::vector<float> confidencesOut;
    std::vector<Rect> boxesOut;
    
    objectInfo objectNew;
    
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);

                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
             
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
               

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    }

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != DNN_BACKEND_OPENCV))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;

        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }

        boxesOut = nmsBoxes;
        classIdsOut = nmsClassIds;
        confidencesOut = nmsConfidences;
        
    }    
    std::vector<int>::iterator it = std::find(classIdsOut.begin(), classIdsOut.end(), masina);
   
    if (it != classIdsOut.end())
    {
        int index = std::distance(classIdsOut.begin(), it);
        objectNew = {boxesOut[index], confidencesOut[index], classIdsOut[index]};
    }
        

    //for (size_t idx = 0; idx < boxes.size(); ++idx)
    //{
        //Rect box = boxes[idx];
        //drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 //box.x + box.width, box.y + box.height, frame);
    //}
    return objectNew;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}

