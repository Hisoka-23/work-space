package in.prakash.recog;

import com.example.facerec.data.DatasetLoader;
import com.example.facerec.detect.FaceDetector;
import com.example.facerec.util.Labels;
import org.opencv.core.*;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.List;

public class FaceRecognizerService {
    private final File modelFile;
    private final Labels labels;
    private final Size fixedSize = new Size(200, 200);
    private LBPHFaceRecognizer lbph;
    private double threshold;

    public FaceRecognizerService(File modelFile, Labels labels, double threshold) {
        this.modelFile = modelFile;
        this.labels = labels;
        this.threshold = threshold;
        this.lbph = LBPHFaceRecognizer.create(1, 8, 8, 8, threshold);
        if (modelFile.exists()) {
            lbph.read(modelFile.getAbsolutePath());
        }
    }

    public void setThreshold(double thr) {
        this.threshold = thr;
        lbph.setThreshold(thr);
    }

    public void trainFromDataset(File datasetRoot, FaceDetector detector) {
        DatasetLoader loader = new DatasetLoader();
        DatasetLoader.Data data = loader.load(datasetRoot, detector, fixedSize);
        if (data.images.isEmpty()) throw new IllegalStateException("No training faces found");
        lbph = LBPHFaceRecognizer.create(1, 8, 8, 8, threshold);
        lbph.train(data.images, data.labels);
        modelFile.getParentFile().mkdirs();
        lbph.write(modelFile.getAbsolutePath());
        labels.saveMap(data.idToName);
    }

    public Mat normalizeFace(Mat grayROI) {
        Mat eq = new Mat();
        Imgproc.equalizeHist(grayROI, eq);
        Mat resized = new Mat();
        Imgproc.resize(eq, resized, fixedSize);
        return resized;
    }

    public String predictLabel(Mat grayROI) {
        Mat face = normalizeFace(grayROI);
        int[] label = new int[1];
        double[] conf = new double[1];
        lbph.predict(face, label, conf);
        String name = labels.nameFor(label);
        boolean unknown = (label == -1) || (conf > threshold);
        String caption = unknown ? "Unknown" : name;
        return caption + " (" + String.format("%.1f", conf) + ")";
    }
}
