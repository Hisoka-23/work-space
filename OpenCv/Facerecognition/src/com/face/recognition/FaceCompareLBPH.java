package com.face.recognition;

import org.opencv.core.*;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class FaceCompareLBPH<LBPHFaceRecognizer> {
    private final LBPHFaceRecognizer recog;
    private final int W = 100, H = 100;

    public FaceCompareLBPH(String modelPath) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        recog = LBPHFaceRecognizer.create(); // or with tuned params
        recog.read(modelPath);
        if (recog.empty()) throw new IllegalStateException("LBPH model not loaded: " + modelPath);
    }

    // Returns true if both face images are predicted as the same identity under threshold tau
    public boolean isSamePerson(String faceImgPathA, String faceImgPathB, double tau) {
        Mat a = loadAndPrep(faceImgPathA);
        Mat b = loadAndPrep(faceImgPathB);

        int[] la = new int[2]; double[] da = new double[2];
        int[] lb = new int[2]; double[] db = new double[2];

        // Predict labels/distances independently
        recog.predict(a, la, da);
        recog.predict(b, lb, db);

        // Strategy 1: if labels equal and both distances <= tau, treat as same person
        boolean same = (la == lb) && (da <= tau) && (db <= tau);

        // Cleanup
        a.release(); b.release();
        return same;
    }

    private Mat loadAndPrep(String p) {
        Mat img = Imgcodecs.imread(p, Imgcodecs.IMREAD_GRAYSCALE);
        if (img.empty()) throw new IllegalArgumentException("Cannot read image: " + p);
        Imgproc.equalizeHist(img, img);
        Mat resized = new Mat();
        Imgproc.resize(img, resized, new Size(W, H));
        img.release();
        return resized;
    }
}
