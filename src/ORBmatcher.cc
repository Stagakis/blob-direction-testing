/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "ORBmatcher.h"
#include <future>
#include<limits.h>
#include <iostream>
#include <fstream>

#include<helpers.h>

#include<BlobExtractor.h>


#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include <opencv2/core/eigen.hpp>

#include<chrono>
#include<stdint-gcc.h>
using namespace std;
std::vector<cv::DMatch> ORB_SLAM2::ORBmatcher::matches = std::vector<cv::DMatch>();
#define use_timings true

#if(use_timings == true)
#define TIME_ACCUM(Variable, x) std::chrono::steady_clock::time_point t_accum = std::chrono::steady_clock::now(); x ; Variable += std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t_accum).count()
#define TIME_ACCUM2(Variable, x) std::chrono::steady_clock::time_point t_accum2 = std::chrono::steady_clock::now(); x ; Variable += std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t_accum2).count()
std::chrono::steady_clock::time_point t2;
#define TIME(x) t2 = std::chrono::steady_clock::now(); x ; std::cout << #x << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t2).count() << std::endl
#define TIME2(name, x) t2 = std::chrono::steady_clock::now(); x ; std::cout << name <<  std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t2).count() << std::endl

#else
#define TIME(x) x
#define TIME2(name, x) x
#define TIME_ACCUM(Variable, x) x
#define TIME_ACCUM2(Variable, x) x
#endif

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor)
            r*=th;

        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                    const int dist =  DescriptorDistance(dKF,dF);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF]=pMP;

                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        if(mbCheckOrientation)
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float ur = u-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

/*
void filter_keypoints_indeces2(const vector<KeyPoint>& all_kp, vector<int>& out_kp_indeces, int color, const cv::Mat& mask_img){
    for(int i = 0; i< all_kp.size(); i++){
        Point2d test_point = all_kp[i].pt/4;
        if(mask_img.at<uchar>(test_point) == color || mask_img.at<uchar>(test_point) == 255){
            out_kp_indeces.push_back(i);
        }
        else if(mask_img.at<uchar>(test_point) == color || mask_img.at<uchar>(test_point) == 255){
            out_kp_indeces.push_back(i);
        }
        else if(mask_img.at<uchar>(test_point) == color || mask_img.at<uchar>(test_point) == 255){
            out_kp_indeces.push_back(i);
        }
        else if(mask_img.at<uchar>(test_point) == color || mask_img.at<uchar>(test_point) == 255){
            out_kp_indeces.push_back(i);
        }
        else if(mask_img.at<uchar>(test_point) == color || mask_img.at<uchar>(test_point) == 255){
            out_kp_indeces.push_back(i);
        }
        else if(mask_img.at<uchar>(test_point) == color || mask_img.at<uchar>(test_point) == 255){
            out_kp_indeces.push_back(i);
        }

    }
}
*/

void filter_keypoints_indeces(const vector<cv::KeyPoint>& all_kp, vector<size_t>& out_kp_indeces, int color,
                                     cv::Mat& mask_img, cv::Rect& bb){

    cv::Rect bb_orig = cv::Rect(bb.x*4, bb.y*4, bb.width*4, bb.height*4);
    int range = 1;
    for(int k = 0; k< all_kp.size(); ++k){
        if(!bb_orig.contains(all_kp.operator[](k).pt)) {            continue;        }

        //int pt_x = (int)all_kp.operator[](k).pt.x/4;
        //int pt_y = (int)all_kp.operator[](k).pt.y/4; //TODO check is there is a speed difference
        int pt_x = static_cast<int>(all_kp.operator[](k).pt.x)/4;
        int pt_y = static_cast<int>(all_kp.operator[](k).pt.y)/4;

        //TIME_ACCUM(point2i_creation, Point2i test_point = all_kp->operator[](k).pt;);

        //if(!bb.contains(Point2i(pt_x,pt_y)/4)) {            continue;        }
        //*//


        for(int i = -range, found = false; i < range+1 && !found; i++){
            for(int j = -range; j < range+1 && !found; j++){
                int row = pt_y-j;
                int col = pt_x-i;
                const uchar* data = mask_img.ptr(row);
                bool belongs = data[col] == color || data[col] == 255;
                if(belongs){
                    out_kp_indeces.push_back(k);
                    found = true;
                    break;
                }
            }
        }

        //*/
    }
}

float calculate_angle(const cv::Mat& image) {
    cv::Vec2i start_point(0, 0);
    cv::Vec2i end_point(0, 0);
    int before_points = 0;
    int after_points = 0;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) == 105) {
                start_point.val[0] += i;
                start_point.val[1] += j;
                before_points++;
            }

            else if (image.at<uchar>(i, j) == 190) {
                end_point.val[0] += i;
                end_point.val[1] += j;
                after_points++;
            }

            else if (image.at<uchar>(i, j) == 255) {
                start_point.val[0] += i;
                start_point.val[1] += j;
                end_point.val[0] += i;
                end_point.val[1] += j;
            }
        }
    }

    end_point /= after_points;
    start_point /= before_points;
    auto dir = end_point - start_point;
    float angle = atan2(-dir.val[1], dir.val[0]);
    if(angle<0) angle += 360;
    return angle;
}

int ORBmatcher::SearchByThreeFrameProcessing(Frame &CurrentFrame, const Frame &LastFrame, bool positiveChange) {
    // Rotation Histogram (to check rotation consistency per blob)
    bool mbCheckOrientation = false; //TODO OVERWRITING DEFAULT BOOL
/*    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
*/
    double keypoint_filtering_timing = 0;
    double blob_extraction_timing = 0;
    double matching_timing = 0;

    int nmatches = 0;
    BlobExtractor* blextr;
    cv::Mat blob;
    cv::Rect blob_bb;
    std::vector<future<void>> futures;
    std::vector<size_t> indeces_prev(200), indeces_cur(200);

    cv::Mat diff_img, cur, prev;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    if(positiveChange) {
        threshold(CurrentFrame.diff_img, cur, 0, -1, CV_THRESH_TOZERO);
        threshold(LastFrame.diff_img, prev, 0, -1, CV_THRESH_TOZERO);
        cur.convertTo(cur, CV_8UC1);
        prev.convertTo(prev, CV_8UC1);
        cv::imwrite("../../debug_folder/cur8" + to_string(time(0))+".jpg",cur);
        cv::imwrite("../../debug_folder/prev8"+ to_string(time(0))+".jpg",prev);
        //cv::adaptiveThreshold(cur, cur,THRESHOLD_VALUE_CUR,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,-3);
        //cv::adaptiveThreshold(prev, prev,THRESHOLD_VALUE_PREV,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,-3);

        threshold(cur,cur,25,THRESHOLD_VALUE_CUR,CV_THRESH_BINARY+CV_THRESH_OTSU);
        threshold(prev,prev,25,THRESHOLD_VALUE_PREV,CV_THRESH_BINARY+CV_THRESH_OTSU);

        cv::imwrite("../../debug_folder/cur" + to_string(time(0))+".jpg",cur);
        cv::imwrite("../../debug_folder/prev"+ to_string(time(0))+".jpg",prev);
    }
    else{
        threshold(CurrentFrame.diff_img, cur, 0, -1, CV_THRESH_TOZERO_INV);
        threshold(LastFrame.diff_img, prev, 0, -1, CV_THRESH_TOZERO_INV);

        cur.convertTo(cur, CV_8UC1, -1);
        prev.convertTo(prev, CV_8UC1, -1);
        //cv::imwrite("../../debug_folder/cur8" + to_string(time(0))+".jpg",cur);
        //cv::imwrite("../../debug_folder/prev8"+ to_string(time(0))+".jpg",prev);
        cv::adaptiveThreshold(cur, cur,THRESHOLD_VALUE_CUR,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,-3);
        cv::adaptiveThreshold(prev, prev,THRESHOLD_VALUE_PREV,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,-3);

        //cv::imwrite("../../debug_folder/cur" + to_string(time(0))+".jpg",cur);
        //cv::imwrite("../../debug_folder/prev"+ to_string(time(0))+".jpg",prev);
    }
    std::cout << "-Initialization:        " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count() << std::endl;

/*
    start = std::chrono::steady_clock::now();
    Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> a1;
    Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> a2;
    cv2eigen(cur, a1);
    cv2eigen(prev, a2);
    auto msparse_cur = a1.sparseView() + a2.sparseView();
    std::cout << "-EigenSparseAddingMats: " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count() << fixed << std::endl;
*/

    //start = std::chrono::steady_clock::now();
    Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> b1;
    Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> b2;
    cv2eigen(cur, b1);
    cv2eigen(prev, b2);
    b1 += b2;
    eigen2cv(b1, diff_img);
    //std::cout << "-EigenAddingMatricesOP: " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count() << fixed << std::endl;



    //TIME2("-Sparsecifications:OPCV ",
    //cv::SparseMat sparse1(cur); cv::SparseMat sparse2(prev);
    //);
    //std::vector<cv::Point> non_zero;
    //findNonZero(diff_img, non_zero);
    //cout<<"NonZeroElementsInCur  " << non_zero.size()/((float)(diff_img.rows*diff_img.cols)) * 100 << endl;

    //cv::imwrite("../../debug_folder/diff1.jpg",cur);
    //cv::imwrite("../../debug_folder/diff2.jpg",prev);
    cv::imwrite("../../debug_folder/diff_all_" + to_string(time(0)) + ".jpg",diff_img);
    //cv::imwrite("../debug_folder/pos_diff"+to_string(frame_counter) + ".jpg",diff_img);
    TIME2("-InitializingBlextr:    ", blextr = new BlobExtractor(diff_img));

    /*// CONCURRENT WAY
    //auto start = std::chrono::steady_clock::now();
    while(blextr->GetNextBlob(diff_img, blob, blob_bb)) {
        indeces_prev.clear();
        indeces_cur.clear();
        futures.push_back(
                std::async(std::launch::async, handle_blob_async, &LastFrame, &CurrentFrame, blob, blob_bb, &nmatches));
        continue;
    }

    threshold(CurrentFrame.diff_img, cur, 40, 190, CV_THRESH_BINARY);
    threshold(LastFrame.diff_img, prev, 40, 105, CV_THRESH_BINARY);
    cv::add(cur, prev, diff_img, noArray(), CV_8U);
    //std::cout << "-Initialization:        " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count() << std::endl;
    blextr = new BlobExtractor(diff_img);
    //auto start = std::chrono::steady_clock::now();
    while(blextr->GetNextBlob(diff_img, blob, blob_bb)) {
        indeces_prev.clear();
        indeces_cur.clear();
        futures.push_back(
                std::async(std::launch::async, handle_blob_async, &LastFrame, &CurrentFrame, blob, blob_bb, &nmatches));
        continue;
    }
    //*/

    //*//    //Non-Concurrent
    int total_sum = 0;
    start = std::chrono::steady_clock::now();
    int blob_number = 0;
    //ofstream myfile;
    //myfile.open ("../debug_folder/pos_blobs_"+to_string(frame_counter) +".txt");
    while(blextr->GetNextBlob(diff_img, blob, blob_bb)){
        indeces_prev.clear();
        indeces_cur.clear();

        TIME_ACCUM(keypoint_filtering_timing ,
                filter_keypoints_indeces(LastFrame.mvKeys, indeces_prev, THRESHOLD_VALUE_PREV, blob, blob_bb);
        if(indeces_prev.empty()) continue;
        filter_keypoints_indeces(CurrentFrame.mvKeys, indeces_cur, THRESHOLD_VALUE_CUR, blob, blob_bb);
        );
        //cv::imwrite("../debug_folder/pos_blob_" + to_string(frame_counter) +"_"+ to_string(blob_number) + ".jpg",blob);
        //myfile << to_string(indeces_prev.size()) + " " + to_string(indeces_cur.size())+"\n";
        //blob_number++;
/*        if(indeces_prev.size() * indeces_cur.size() > 250){
            continue;
            cv::imwrite("../../debug_folder/sd/Blob_w"
            + std::to_string(blob_bb.width) + "_h"
            + std::to_string(blob_bb.height)+"_"
            + std::to_string(blextr->num_of_blobs)
            + "_angle" + std::to_string(calculate_angle(blob(blob_bb))) +
            +".png",blob);
        }*/

        TIME_ACCUM2(matching_timing,
        for(size_t i : indeces_prev)
        {
            MapPoint* pMP = LastFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!LastFrame.mvbOutlier[i])
                {

                    int bestDist = 256;
                    int bestIdx2 = -1;
 
                    const cv::Mat dMP = pMP->GetDescriptor();
                    for(size_t i2 : indeces_cur)
                    {

                        if(CurrentFrame.mvpMapPoints[i2]) {
                            continue;
                        }
                        total_sum++;
                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP,d);
                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH/2)
                    {
                        ORBmatcher::matches.push_back(DMatch(i, bestIdx2, bestDist));
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;
/*                        if(mbCheckOrientation)
                        {
                            float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }*/
                    }

                }
            }        
        }
/*        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }
        }
        for(int kys = 0; kys < HISTO_LENGTH ; kys++) rotHist[kys].clear();*/
        );
    }
    //*/

    //futures.clear();
    auto main_loop_timing = std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count();

    std::cout << "Blob number:            " << blextr->num_of_blobs << endl;
    if(use_timings) {
        std::cout << "-For loop:              " << main_loop_timing << fixed << std::endl;
        std::cout << "--AccumulatedFiltering: " << keypoint_filtering_timing << fixed << std::endl;
        std::cout << "--MatchingTiming     :  " << matching_timing << fixed << std::endl;
        std::cout << "--BlobExtractionAccum:  " << blextr->blob_extraction_total_time << fixed << std::endl;
        std::cout << "TotalSum:               " << total_sum << endl;
        std::cout << "Efficiency:             " << ((float) (nmatches)) / total_sum * 100.0f << endl;
    }
    //myfile.close();
    delete blextr;
    return nmatches;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    ORBmatcher::matches.clear();
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw;

    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

    cout << "LastFrame.N: " << LastFrame.N << endl;
    int total_sum = 0;
    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                if(invzc<0)
                    continue;

                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                int nLastOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                //cout<<" " << vIndices2.size() << " ";
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }
                    total_sum++;
                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);
                    
                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    ORBmatcher::matches.push_back(DMatch(i, bestIdx2, bestDist));
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }
    std::cout << "TotalSum:Projection     " << total_sum << endl;
    std::cout << "Efficiency:             " << ((float)(nmatches)) / total_sum * 100.0f << endl;
    return nmatches;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

int ORBmatcher::SearchByThreeFrameProcessing_test(Frame *CurrentFrame2, Frame *LastFrame2, bool positiveChange){
    Frame CurrentFrame = *CurrentFrame2;
    Frame LastFrame = *LastFrame2;
    // Rotation Histogram (to check rotation consistency per blob)
    bool mbCheckOrientation = false; //TODO OVERWRITING DEFAULT BOOL
/*    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
*/

    int nmatches = 0;
    BlobExtractor* blextr;
    cv::Mat blob;
    cv::Rect blob_bb;
    std::vector<future<void>> futures;
    std::vector<size_t> indeces_prev(200), indeces_cur(200);

    cv::Mat diff_img, cur, prev;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    if(positiveChange) {
        threshold(CurrentFrame.diff_img, cur, 0, -1, CV_THRESH_TOZERO);
        threshold(LastFrame.diff_img, prev, 0, -1, CV_THRESH_TOZERO);
        cur.convertTo(cur, CV_8UC1);
        prev.convertTo(prev, CV_8UC1);
        //cv::imwrite("../../debug_folder/cur8" + to_string(time(0))+".jpg",cur);
        //cv::imwrite("../../debug_folder/prev8"+ to_string(time(0))+".jpg",prev);
        //cv::adaptiveThreshold(cur, cur,THRESHOLD_VALUE_CUR,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,-3);
        //cv::adaptiveThreshold(prev, prev,THRESHOLD_VALUE_PREV,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,-3);

        threshold(cur,cur,25,THRESHOLD_VALUE_CUR,CV_THRESH_BINARY+CV_THRESH_OTSU);
        threshold(prev,prev,25,THRESHOLD_VALUE_PREV,CV_THRESH_BINARY+CV_THRESH_OTSU);

        //cv::imwrite("../../debug_folder/cur" + to_string(time(0))+".jpg",cur);
        //cv::imwrite("../../debug_folder/prev"+ to_string(time(0))+".jpg",prev);
    }
    else{
        threshold(CurrentFrame.diff_img, cur, 0, -1, CV_THRESH_TOZERO_INV);
        threshold(LastFrame.diff_img, prev, 0, -1, CV_THRESH_TOZERO_INV);

        cur.convertTo(cur, CV_8UC1, -1);
        prev.convertTo(prev, CV_8UC1, -1);
        //cv::imwrite("../../debug_folder/cur8" + to_string(time(0))+".jpg",cur);
        //cv::imwrite("../../debug_folder/prev8"+ to_string(time(0))+".jpg",prev);
        //cv::adaptiveThreshold(cur, cur,THRESHOLD_VALUE_CUR,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,-3);
        //cv::adaptiveThreshold(prev, prev,THRESHOLD_VALUE_PREV,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,-3);

        threshold(cur,cur,25,THRESHOLD_VALUE_CUR,CV_THRESH_BINARY+CV_THRESH_OTSU);
        threshold(prev,prev,25,THRESHOLD_VALUE_PREV,CV_THRESH_BINARY+CV_THRESH_OTSU);

        //cv::imwrite("../../debug_folder/cur" + to_string(time(0))+".jpg",cur);
        //cv::imwrite("../../debug_folder/prev"+ to_string(time(0))+".jpg",prev);
    }
    //std::cout << "-Initialization:        " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count() << std::endl;

/*
    start = std::chrono::steady_clock::now();
    Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> a1;
    Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> a2;
    cv2eigen(cur, a1);
    cv2eigen(prev, a2);
    auto msparse_cur = a1.sparseView() + a2.sparseView();
    std::cout << "-EigenSparseAddingMats: " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count() << fixed << std::endl;
*/

    //start = std::chrono::steady_clock::now();
    Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> b1;
    Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> b2;
    cv2eigen(cur, b1);
    cv2eigen(prev, b2);
    b1 += b2;
    eigen2cv(b1, diff_img);
    //std::cout << "-EigenAddingMatricesOP: " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count() << fixed << std::endl;



    //TIME2("-Sparsecifications:OPCV ",
    //cv::SparseMat sparse1(cur); cv::SparseMat sparse2(prev);
    //);
    //std::vector<cv::Point> non_zero;
    //findNonZero(diff_img, non_zero);
    //cout<<"NonZeroElementsInCur  " << non_zero.size()/((float)(diff_img.rows*diff_img.cols)) * 100 << endl;

    //cv::imwrite("../../debug_folder/diff1.jpg",cur);
    //cv::imwrite("../../debug_folder/diff2.jpg",prev);
    //cv::imwrite("../../debug_folder/diff_all_" + to_string(time(0)) + ".jpg",diff_img);
    //cv::imwrite("../debug_folder/pos_diff"+to_string(frame_counter) + ".jpg",diff_img);

    blextr = new BlobExtractor(diff_img);

    /*// CONCURRENT WAY
    //auto start = std::chrono::steady_clock::now();
    while(blextr->GetNextBlob(diff_img, blob, blob_bb)) {
        indeces_prev.clear();
        indeces_cur.clear();
        futures.push_back(
                std::async(std::launch::async, handle_blob_async, &LastFrame, &CurrentFrame, blob, blob_bb, &nmatches));
        continue;
    }

    threshold(CurrentFrame.diff_img, cur, 40, 190, CV_THRESH_BINARY);
    threshold(LastFrame.diff_img, prev, 40, 105, CV_THRESH_BINARY);
    cv::add(cur, prev, diff_img, noArray(), CV_8U);
    //std::cout << "-Initialization:        " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count() << std::endl;
    blextr = new BlobExtractor(diff_img);
    //auto start = std::chrono::steady_clock::now();
    while(blextr->GetNextBlob(diff_img, blob, blob_bb)) {
        indeces_prev.clear();
        indeces_cur.clear();
        futures.push_back(
                std::async(std::launch::async, handle_blob_async, &LastFrame, &CurrentFrame, blob, blob_bb, &nmatches));
        continue;
    }
    //*/

    //*//    //Non-Concurrent
    int total_sum = 0;
    start = std::chrono::steady_clock::now();
    int blob_number = 0;
    //ofstream myfile;
    //myfile.open ("../debug_folder/pos_blobs_"+to_string(frame_counter) +".txt");
    while(blextr->GetNextBlob(diff_img, blob, blob_bb)){
        indeces_prev.clear();
        indeces_cur.clear();


        filter_keypoints_indeces(LastFrame.mvKeys, indeces_prev, THRESHOLD_VALUE_PREV, blob, blob_bb);
        if(indeces_prev.empty()) continue;
        filter_keypoints_indeces(CurrentFrame.mvKeys, indeces_cur, THRESHOLD_VALUE_CUR, blob, blob_bb);

        //cv::imwrite("../debug_folder/pos_blob_" + to_string(frame_counter) +"_"+ to_string(blob_number) + ".jpg",blob);
        //myfile << to_string(indeces_prev.size()) + " " + to_string(indeces_cur.size())+"\n";
        //blob_number++;
/*        if(indeces_prev.size() * indeces_cur.size() > 250){
            continue;
            cv::imwrite("../../debug_folder/sd/Blob_w"
            + std::to_string(blob_bb.width) + "_h"
            + std::to_string(blob_bb.height)+"_"
            + std::to_string(blextr->num_of_blobs)
            + "_angle" + std::to_string(calculate_angle(blob(blob_bb))) +
            +".png",blob);
        }*/


        for(size_t i : indeces_prev)
        {
            MapPoint* pMP = LastFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!LastFrame.mvbOutlier[i])
                {

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    const cv::Mat dMP = pMP->GetDescriptor();
                    for(size_t i2 : indeces_cur)
                    {

                        if(CurrentFrame.mvpMapPoints[i2]) {
                            continue;
                        }
                        total_sum++;
                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP,d);
                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH/2)
                    {
                        ORBmatcher::matches.push_back(DMatch(i, bestIdx2, bestDist));
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;
/*                        if(mbCheckOrientation)
            {
                float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                if(rot<0.0)
                    rot+=360.0f;
                int bin = round(rot*factor);
                if(bin==HISTO_LENGTH)
                    bin=0;
                assert(bin>=0 && bin<HISTO_LENGTH);
                rotHist[bin].push_back(bestIdx2);
            }*/
                    }

                }
            }
        }
/*        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }
        }
        for(int kys = 0; kys < HISTO_LENGTH ; kys++) rotHist[kys].clear();*/

    }
    //*/

    //futures.clear();
    auto main_loop_timing = std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - start).count();

    //myfile.close();
    delete blextr;
    return nmatches;
}

} //namespace ORB_SLAM
