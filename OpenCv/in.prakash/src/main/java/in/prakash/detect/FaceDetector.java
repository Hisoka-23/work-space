package in.prakash.detect;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.util.ArrayList;
import java.util.List;

public class FaceDetector {
    private final CascadeClassifier classifier;
    private final Size minSize;

    public FaceDetector(String cascadePath) {
        this.classifier = new CascadeClassifier(cascadePath);
        this.minSize = new Size(80, 80); // tune for webcam
    }

    public List<Rect> detect(Mat grayEqualized) {
        MatOfRect faces = new MatOfRect();
        classifier.detectMultiScale(
                grayEqualized, faces,
                1.1, 3, Objdetect.CASCADE_SCALE_IMAGE,
                minSize, new Size());
        List<Rect> list = new ArrayList<>();
        for (Rect r : faces.toArray()) list.add(r);
        return list;
    }
}
