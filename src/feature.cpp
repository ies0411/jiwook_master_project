#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
int main(int argc, char const *argv[]) {
    std::string path1 = "/home/catkin_ws/src/jiwook_master_project/data/1.png";
    std::string path2 = "/home/catkin_ws/src/jiwook_master_project/data/2.png";
    cv::Mat img1 = cv::imread(path1, cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_COLOR);
    // cv::imshow("show1", img1);
    // cv::imshow("show2", img2);
    // cv::waitKey(0);

    // ORB feature extractor로 특징점을 뽑을 거야
    //초기화 - 변수만들거야
    // keypoint.pt = (x,y)
    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;
    cv::Mat descriptors_1, descriptors_2;
    // detect class
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    // Fast 로 특징점(코너) 추출
    detector->detect(img1, keypoint_1);
    detector->detect(img2, keypoint_2);
    // 디스크립터로 ID부여
    descriptor->compute(img1, keypoint_1, descriptors_1);
    descriptor->compute(img2, keypoint_2, descriptors_2);
    //
    cv::Mat detect_result_img;
    cv::drawKeypoints(img1, keypoint_1, detect_result_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    // cv::drawMatches(img1, keypoint_1, img2, keypoint_2, matches, img_match);
    // cv::imshow("draw result", detect_result_img);
    // cv::waitKey(0);

    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    cv::Mat result_match;
    cv::drawMatches(img1, keypoint_1, img2, keypoint_2, matches, result_match);
    cv::imshow("result match", result_match);
    // cv::waitKey(0);

    double min_distance = 100000, max_distance = 0;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance > max_distance) max_distance = matches[i].distance;
        if (matches[i].distance < min_distance) min_distance = matches[i].distance;
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(min_distance * 2, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }
    cv::Mat good_matches_img;
    cv::drawMatches(img1, keypoint_1, img2, keypoint_2, good_matches, good_matches_img);
    cv::imshow("good result match", good_matches_img);
    cv::waitKey(0);

    cv::Point2d principal_point(325.1, 249.7);  //카메라 광학 중심
    double focal_length = 521;                  //카메라 초점 거리,
    cv::Mat essential_matrix;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (int i = 0; i < (int)good_matches.size(); i++) {
        points1.push_back(keypoint_1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoint_2[good_matches[i].trainIdx].pt);
    }
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    std::cout << "essential_matrix : " << std::endl
              << essential_matrix << std::endl;

    cv::Mat R, t;
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R :" << std::endl
              << R << std::endl;
    std::cout << "t :" << std::endl
              << t << std::endl;

    return 0;
}
#if 0
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++) {
        // queryIdx : 매칭된 img1의 index
        // trainIdx : 매칭된 img2의 index
        // pt : pixel
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // F matrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental matrix : " << fundamental_matrix << endl;

    // E matrix
    Point2d principal_point(325.1, 249.7);  //카메라 광학 중심
    double focal_length = 521;              //카메라 초점 거리,
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential : " << essential_matrix << endl;

    //호모그래피 행렬 계산
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography : " << homography_matrix << endl;

    //필수 행렬에서 회전 및 변환 정보 복구
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R :" << endl
         << R << endl;
    cout << "t :" << endl
         << t << endl;

    /***triangle****/
}
Point2d pixel2cam(const Point2d& p, const Mat& K) {
    return Point2d(
        (p.x - K.at<double>(0.2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void triangulation(const vector<KeyPoint>& keypoint_1, const vector<KeyPoint>& keypoint_2,
                   const vector<DMatch>& matches, const Mat& R, const Mat& t, vector<Point3d>& points) {
    Mat T1 = (Mat_<float>(3, 4) << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> pts_1, pts_2;
    for (DMatch m : matches) {
        //픽셀 좌표를 카메라 좌표로 변환
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }
    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    //비균일 좌표로 변환
    for (int i = 0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);  //정규화
        Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(p);
    }
}

void budleAdjustment(const vector<Point3f> points_3d, const vector<Point2f> points_2d,
                     const Mat& K, Mat& R, Mat& t) {
    // initialize g2o
    // pose dimension is 6, The landmark dimension is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;
    // linear equation solver
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCSparse<Block::PoseMatrixType>());  // linear equation solver
                                                                                                                   // Matrix block solver
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex(se(3),R,t)
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2.0))));
    optimizer.addVertex(pose);

    // vertex
    int index = 1;
    // landmarks
    for (const Point3f p : points_3d) {
        g2o::VertexPointXYZ* point = new g2o::VertexPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);  // Marg must be set in g2o. See the content of Lecture 10
        optimizer.addVertex(point);
    }

    // camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters(
        K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // edge
    index = 1;
    for (const Point2f p : points_2d) {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "cost time :" << time_used.count() << endl;
    cout << "T=" << endl
         << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}

int main(int argc, char const* argv[]) {
    if (argc != 5) {
        cout << "wrong path" << endl;
        return -1;
    }
    // read img
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    findFeatureMatches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "matches size = " << matches.size() << endl;
#if ESSENTIAL
    //두 이미지 사이의 움직임 추정
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // E=t^R*스케일

    // t^ : skew symmetry matrix
    Mat t_x = (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
               t.at<double>(2, 0), 0, -t.at<double>(0, 0),
               -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    // essential matrix 직접구화기

    cout << "t^R=" << endl
         << t_x * R << endl;

    // epipolar constraints
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m : matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
#endif  // DEBUG

#if TRIANGULATION

    // triangulation
    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    //삼각점과 특징점 간의 재투영 관계 확인
    K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (int i = 0; i < matches.size(); i++) {
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        Point2d pt1_cam_3d(points[i].x / points[i].z, points[i].y / points[i].z);

        cout << "first camera frame :" << pt1_cam << endl;
        cout << "projected from 3D :" << pt1_cam_3d << ", d=" << points[i].z << endl;

        Point2f pt2_cam = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /= pt2_trans.at<double>(2, 0);
        cout << "second camera frame :" << pt2_cam << endl;
        cout << "reprojected from second frame: " << pt2_trans.t() << endl;
    }
#endif  // DEBUG
    /*******3d2d*******/
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);  //깊이 맵은 16비트 무부호 숫자, 단일 채널 이미지
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch m : matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) continue;  // bad depth
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs : " << pts_3d.size() << endl;

    Mat r, t;
    // OpenCV의 PnP 솔루션을 호출하고 EPNP, DLS 및 기타 방법을 선택하십시오.
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
    Mat R;
    // r은 Rodrigues 공식을 사용하여 행렬로 변환된 회전 벡터의 형태입니다.
    cv::Rodrigues(r, R);
    cout << "R=" << endl
         << R << endl;
    cout << "t=" << endl
         << t << endl;
    cout << "====BA====" << endl;
    budleAdjustment(pts_3d, pts_2d, K, R, t);

    return 0;
}


    return 0;
}
#endif
