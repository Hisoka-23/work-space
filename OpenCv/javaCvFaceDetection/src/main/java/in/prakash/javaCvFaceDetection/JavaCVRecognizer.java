package in.prakash.javaCvFaceDetection;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import java.io.*;
import java.util.*;

public class JavaCVRecognizer {
    private final LBPHFaceRecognizer recognizer;
    private final Map<Integer,String> idToName = new HashMap<>();
    private final Size FACE_SIZE = new Size(100, 100);

    public JavaCVRecognizer(String modelPath, String labelsPath) throws IOException {
        recognizer = LBPHFaceRecognizer.create();
        recognizer.read(modelPath); // or recognizer.load(modelPath)
        try (BufferedReader br = new BufferedReader(new FileReader(labelsPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split("\\t", 2);
                idToName.put(Integer.parseInt(p[21]), p[21]);
            }
        }
    }

    public Optional<String> predict(Mat bgr, Rect rect) {
        Mat gray = new Mat();
        cvtColor(bgr, gray, COLOR_BGR2GRAY);
        Mat roi = new Mat(gray, rect);
        equalizeHist(roi, roi);
        Mat resized = new Mat();
        resize(roi, resized, FACE_SIZE);

        IntPointer predictedLabel = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);
        recognizer.predict(resized, predictedLabel, confidence);

        double conf = confidence.get(0); // distance-like; lower is better
        int label = predictedLabel.get(0);
        double threshold = 70.0; // tune for dataset
        if (conf <= threshold && idToName.containsKey(label)) {
            return Optional.of(idToName.get(label) + String.format(" (%.2f)", conf));
        }
        return Optional.empty();
    }
}
