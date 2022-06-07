# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

# Report - Write up

### FP.1 Match 3D Objects

Implement "matchBoundingBoxes" which takes the input of both previous and current dataframes and gives the ids of the matched regions of interest (i.e bounding ID) as the output. Matches are the ones with highest correspondences of the keypoints
Code : camFusion_Student.cpp Line : 278-317
```c++    
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    for(auto it = prevFrame.boundingBoxes.begin(); it != prevFrame.boundingBoxes.end(); it++)
    {
        std::vector<vector<cv::DMatch>::iterator> query;
        for(auto it1 = matches.begin(); it1 != matches.end(); it1++) 
        {
            int prevKeyPointIdx = it1->queryIdx;
            if(it->roi.contains(prevFrame.keypoints.at(prevKeyPointIdx).pt)) 
            {
                query.push_back(it1);
            }
        }

        std::multimap<int, int> result;
        for(auto it2 = query.begin(); it2 != query.end(); it2++) {
            int currKeyPointIdx = (*it2)->trainIdx;
            for(auto it3 = currFrame.boundingBoxes.begin(); it3 != currFrame.boundingBoxes.end(); it3++)
            {
                if(it3->roi.contains(currFrame.keypoints.at(currKeyPointIdx).pt)) {
                    int boxId = it3->boxID;
                    result.insert(std::pair<int, int>(boxId, currKeyPointIdx));
                }
            }
        }

        int max_count = 0;
        int id_max = 10000;
        if(result.size() > 0) {
            for(auto it_4 = result.begin(); it_4 != result.end(); it_4++)
            {
                if(result.count(it_4->first) > max_count) {
                    max_count = result.count(it_4->first);
                    id_max = it_4->first;
                }  
            }
            bbBestMatches.insert(std::pair<int, int>(it->boxID, id_max));
        }
    }
}
```
### FP.2 Compute Lidar-based TTC

Compute TTC in second for all matched 3D objects using Lidar measurements from the matched bounding boxes between previous and current frame. 
In order to deal with the outlier Lidar points and the threshold for Point.y values is set and remove those points on the edge. Remaining points in middle area are sorted with Point.x value , only consider Lidar Points with ego lane, then get the mean distance to get stable output.
Code : camFusion_Student.cpp Line : 222-276
```c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,  std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT=1/frameRate;
    double laneWidth=4.0;
    vector<double> lidarPointsCurrX,lidarPointsPrevX;
    for(auto it =lidarPointsPrev.begin();it!=lidarPointsPrev.end();++it){
        if(abs(it->y)<=laneWidth/2.0){
            lidarPointsPrevX.push_back(it->x);
        }
    }
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {

        if (abs(it->y) <= laneWidth / 2.0){
            lidarPointsCurrX.push_back(it->x);
        }
    }
    double XCurrSum = accumulate(lidarPointsCurrX.begin(), lidarPointsCurrX.end(), 0.0);
    double XPrevSum = accumulate(lidarPointsPrevX.begin(), lidarPointsPrevX.end(), 0.0);
    double CurrMean = XCurrSum / lidarPointsCurrX.size();
    double PrevMean = XPrevSum / lidarPointsPrevX.size();
    double Curraccum = 0.0;
    double Prevaccum = 0.0;
    for_each(begin(lidarPointsCurrX), std::end(lidarPointsCurrX), [&](const double d) {
        Curraccum += (d - XCurrSum) * (d - XCurrSum);
    });
    double CurrStd = sqrt(Curraccum / (lidarPointsCurrX.size() - 1));

    for_each(begin(lidarPointsPrevX), std::end(lidarPointsPrevX), [&](const double d) {
        Prevaccum += (d - XPrevSum) * (d - XPrevSum);
    });

    double PrevStd = sqrt(Prevaccum / (lidarPointsPrevX.size() - 1));
    int CurrCount = 0;
    int PrevCount = 0;
    double CurrAns = 0;
    double PrevAns = 0;
    for (int i = 0; i < lidarPointsCurrX.size(); ++i) {
        if (abs(lidarPointsCurrX[i] - CurrMean) < 3 * CurrStd) {
            CurrAns += lidarPointsCurrX[i];
            ++CurrCount;
        }
    
    for (int i = 0; i < lidarPointsPrevX.size(); ++i) {
        if (abs(lidarPointsPrevX[i] - PrevMean) < 3 * PrevStd) {
            PrevAns += lidarPointsPrevX[i];
            ++PrevCount;
        }
    
    double curMean = CurrAns / CurrCount;
    double preMean = PrevAns / PrevCount;
    TTC = curMean * dT / (preMean -curMean);
}
```
### FP.3 Associate Keypoint Correspondences with Bounding Boxes
TTC computation based on camera measurements  by computing the distance between keypoints within a bounding box and using associated keypoint correspondences to the bounding boxes which enclose them.
All the matches statify the condition are added to a vector with respect to the bounding box.
Code : camFusion_Student.cpp Line :  134-170
```c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<double> dist;
    for(auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        int currKptIndex = (*it).trainIdx;
        const auto &currKeyPoint = kptsCurr[currKptIndex];

        if(boundingBox.roi.contains(currKeyPoint.pt))
        {
            int prevKptIndex = (*it).queryIdx;
            const auto &prevKeyPoint = kptsPrev[prevKptIndex];

            dist.push_back(cv::norm(currKeyPoint.pt - prevKeyPoint.pt));
        }
    }
    int pair_num =  dist.size();
    double distMean = std::accumulate(dist.begin(), dist.end(), 0.0) / pair_num;
    for(auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        int currKptIndex = (*it).trainIdx;
        const auto &currKeyPoint = kptsCurr[currKptIndex];

        if(boundingBox.roi.contains(currKeyPoint.pt))
        {
            int prevKptIndex = (*it).queryIdx;
            const auto &prevKeyPoint = kptsPrev[prevKptIndex];
            double temp = cv::norm(currKeyPoint.pt - prevKeyPoint.pt);
            double threshold = distMean * 1.3;
            if(temp < threshold)
            {
                boundingBox.keypoints.push_back(currKeyPoint);
                boundingBox.kptMatches.push_back(*it);
            }
        }
    }
}
```
### FP.4 Compute Camera-based TTC
Compute TTC in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame. Median of distance Ratios is used to reduce the impact of outliers of keypoints.
Code : CamFusion_Student.cpp Line : 174-220
```c++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { 
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { 
            double minDist = 90.0; // min. required distance default 100

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { 
                // avoid division by zero
                double distRatio = distCurr/distPrev;
                distRatios.push_back(distRatio);
            }
        } 
    }     
    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}
```
### FP.5 Performance Evaluation 1
Q:Find examples where the TTC estimate of the Lidar sensor does not seem plausible. 
Describe your observations and provide a sound argumentation why you think this happened.
Ans:TTC from Lidar is not correct because of some outliers and some unstable points from preceding vehicle'front mirror,
```c++
    TTC = curMean * dT / (preMean -curMean);// camFusion_student.cpp //Line 275
```
caused by small interval of (preMean - curMean) than the last sample, those need to be filtered out . 
Here we adapt taking of median value is useful for reducing the imapct of outliers to get more reliable and stable lidar points.
 Then get a more accurate results.

|     LiDAR TTC (s) |    Median-LiDAR TTC (s)) | 
|        ---        |       ---         |
|      12.86        |      12.69        | 
|      12.66        |      11.92        |     
|      28.88        |      20.88        |
|      14.56        |      14.28        |
|      15.49        |      14.75        |
|      16.54        |      13.92        |
|      14.87        |      12.14        |
|      16.39        |      12.45        |
|      13.49        |      12.15        |
|      17.82        |      13.42        |
|      12.94        |      12.46        |
|      11.59        |      11.48        |
|       9.83        |       9.05        |
|      10.79        |       9.18        |
|      10.15        |       8.61        | 
|      11.12        |       9.21        | 
|      11.40        |      12.48        | 
|      13.00        |       9.75        | 


    

### FP.6 Performance Evaluation 2
Q:Run several detector / descriptor combinations and look at the differences in TTC estimation. 
Find out which methods perform best and also include several examples where camera-based TTC estimation is way off.
As with Lidar, describe your observations again and also look into potential reasons.
Ans: In the mid-term project, the top 3 detector/descriptor has been seletected in terms of their performance on accuracy and speed. 
So here, I use them one by one for Camera TTC estimate.

| Rank | Detector | Descriptor |Reasons|
|:-----|----------|------------|-------|
| 1    | FAST     | ORB        | Relative good speed and accuracy|
| 2    | FAST     | BRIEF      | Higher speed & relatively good accuracy|
| 3    | BRISK    | BRIEF      | Higher accuracy|

<br>

##### 1. FAST + ORB

|     LiDAR TTC (s) |    Camera TTC (s) | 
|        ---        |       ---         |
|      12.69        |      12.20        | 
|      11.92        |      12.93        |     
|      20.88        |      16.39        |
|      14.28        |      14.06        |
|      14.75        |   **_-inf_**      |
|      13.92        |      35.57        |
|      12.14        |      12.05        |
|      12.45        |      12.00        |
|      12.15        |      13.10        |
|      13.42        |      14.90        |
|      12.46        |      14.21        |
|      11.48        |      12.76        |
|       9.05        |      13.40        |
|       9.18        |      10.90        |
|       8.61        |      10.41        | 
|       9.21        |      11.53        | 
|      12.48        |      11.76        | 
|       8.75        |      13.82        | 

<br>

##### 2. FAST + BRIEF

|     LiDAR TTC (s) |    Camera TTC (s) | 
|        ---        |       ---         |
|      12.69        |      11.18        | 
|      11.92        |      13.00        |     
|      20.88        |      15.00        |
|      14.28        |      13.81        |
|      14.75        |   **_-inf_**      |
|      13.92        |      23.85        |
|      12.14        |      12.046       |
|      12.46        |      12.61        |
|      12.15        |      14.41        |
|      13.42        |      14.96        |
|      12.46        |      13.75        |
|      11.48        |      12.71        |
|       9.05        |      12.98        |
|       9.18        |      11.48        |
|       8.61        |      12.40        | 
|       9.21        |      12.61        | 
|      12.48        |      10.89        | 
|       8.75        |      13.88        | 

<br>

##### 3. BRISK + ORB

|     LiDAR TTC (s) |    Camera TTC (s) | 
|        ---        |       ---         |
|      12.69        |      27.14        | 
|      11.92        |      18.60        |     
|      20.88        |      18.75        |
|      14.28        |      18.05        |
|      14.75        |      20.61        |
|      13.92        |      19.53        |
|      12.45        |      17.87        |
|      12.14        |      17.44        |
|      12.45        |      14.88        |
|      13.15        |      12.53        |
|      12.42        |      13.57        |
|      11.48        |      15.89        |
|       9.05        |      11.82        |
|       9.18        |      12.29        |
|       8.61        |      12.29        | 
|       9.21        |      11.49        | 
|      12.48        |      13.23        | 
|       8.75        |      17.97        |  

<br>

Both FAST+ORB & FAST+BRIEF  gives -inf values,it is caused by the distribution of keypoints didn't satisfy distance thrshold.
BRISK+BRIEF gives the most reasonable results however some TTC times are longer than LiDAR's.

