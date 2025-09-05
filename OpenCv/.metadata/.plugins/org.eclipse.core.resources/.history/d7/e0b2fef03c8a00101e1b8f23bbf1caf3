package com.face.recognition;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImageSimilaritySSIM {
    public static boolean isSameImage(String imgPathA, String imgPathB, double ssimThreshold) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat a = Imgcodecs.imread(imgPathA);
        Mat b = Imgcodecs.imread(imgPathB);
        if (a.empty() || b.empty()) throw new IllegalArgumentException("Cannot read images");

        // Resize B to A if needed (optional, but SSIM assumes alignment)
        if (a.size().width != b.size().width || a.size().height != b.size().height) {
            Imgproc.resize(b, b, a.size());
        }

        Mat ga = new Mat(); Mat gb = new Mat();
        Imgproc.cvtColor(a, ga, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(b, gb, Imgproc.COLOR_BGR2GRAY);

        double ssim = computeSSIM(ga, gb);

        a.release(); b.release(); ga.release(); gb.release();
        return ssim >= ssimThreshold;
    }

    // Minimal SSIM implementation for single-channel images
    private static double computeSSIM(Mat img1, Mat img2) {
        Mat i1 = img1.clone();
        Mat i2 = img2.clone();

        i1.convertTo(i1, CvType.CV_32F);
        i2.convertTo(i2, CvType.CV_32F);

        Mat mu1 = new Mat(); Mat mu2 = new Mat();
        Imgproc.GaussianBlur(i1, mu1, new Size(11,11), 1.5);
        Imgproc.GaussianBlur(i2, mu2, new Size(11,11), 1.5);

        Mat mu1_2 = new Mat(); Core.multiply(mu1, mu1, mu1_2);
        Mat mu2_2 = new Mat(); Core.multiply(mu2, mu2, mu2_2);
        Mat mu1_mu2 = new Mat(); Core.multiply(mu1, mu2, mu1_mu2);

        Mat sigma1_2 = new Mat(); Mat tmp1 = new Mat();
        Imgproc.GaussianBlur(i1.mul(i1), tmp1, new Size(11,11), 1.5);
        Core.subtract(tmp1, mu1_2, sigma1_2);

        Mat sigma2_2 = new Mat(); Mat tmp2 = new Mat();
        Imgproc.GaussianBlur(i2.mul(i2), tmp2, new Size(11,11), 1.5);
        Core.subtract(tmp2, mu2_2, sigma2_2);

        Mat sigma12 = new Mat(); Mat tmp12 = new Mat();
        Imgproc.GaussianBlur(i1.mul(i2), tmp12, new Size(11,11), 1.5);
        Core.subtract(tmp12, mu1_mu2, sigma12);

        double C1 = 6.5025, C2 = 58.5225;

        Mat t1 = new Mat(); Mat t2 = new Mat(); Mat t3 = new Mat();
        Core.multiply(mu1_mu2, new Scalar(2), t1);
        Core.add(t1, new Scalar(C1), t1);
        Core.multiply(sigma12, new Scalar(2), t2);
        Core.add(t2, new Scalar(C2), t2);
        Core.multiply(t1, t2, t3);

        Mat t1_ = new Mat(); Mat t2_ = new Mat(); Mat t3_ = new Mat();
        Core.add(mu1_2, mu2_2, t1_);
        Core.add(t1_, new Scalar(C1), t1_);
        Core.add(sigma1_2, sigma2_2, t2_);
        Core.add(t2_, new Scalar(C2), t2_);
        Core.multiply(t1_, t2_, t3_);

        Mat ssim_map = new Mat();
        Core.divide(t3, t3_, ssim_map);
        Scalar mssim = Core.mean(ssim_map);

        // Single channel; use mssim.val
        double score = mssim.val;

        // Release
        i1.release(); i2.release(); mu1.release(); mu2.release();
        mu1_2.release(); mu2_2.release(); mu1_mu2.release();
        sigma1_2.release(); sigma2_2.release(); sigma12.release();
        tmp1.release(); tmp2.release(); tmp12.release();
        t1.release(); t2.release(); t3.release();
        t1_.release(); t2_.release(); t3_.release();
        ssim_map.release();

        return score;
    }
}
