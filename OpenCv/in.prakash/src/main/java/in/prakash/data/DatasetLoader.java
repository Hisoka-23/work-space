package in.prakash.data;

import com.example.facerec.detect.FaceDetector;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.*;

public class DatasetLoader {
    public static class Data {
        public final List<Mat> images;
        public final Mat labels;
        public final Map<Integer,String> idToName;

        public Data(List<Mat> images, Mat labels, Map<Integer,String> map) {
            this.images = images;
            this.labels = labels;
            this.idToName = map;
        }
    }

    public Data load(File datasetRoot, FaceDetector detector, Size fixed) {
        List<Mat> imgs = new ArrayList<>();
        List<Integer> lbls = new ArrayList<>();
        Map<Integer,String> idToName = new HashMap<>();

        File[] persons = datasetRoot.listFiles(File::isDirectory);
        if (persons == null) persons = new File;

        int nextId = 0;
        for (File personDir : persons) {
            int id = nextId++;
            idToName.put(id, personDir.getName());
            File[] files = personDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png"));
            if (files == null) continue;
            for (File f : files) {
                Mat color = Imgcodecs.imread(f.getAbsolutePath());
                if (color.empty()) continue;
                Mat gray = new Mat();
                Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
                Imgproc.equalizeHist(gray, gray);
                // detect one face; if multiple, take the largest
                Rect roi = largest(detector, gray);
                if (roi == null) continue;
                Mat face = new Mat(gray, roi);
                Imgproc.resize(face, face, fixed);
                imgs.add(face);
                lbls.add(id);
            }
        }
        Mat labels = new Mat(lbls.size(), 1, CvType.CV_32SC1);
        for (int i = 0; i < lbls.size(); i++) labels.put(i, 0, lbls.get(i));
        return new Data(imgs, labels, idToName);
    }

    private Rect largest(FaceDetector detector, Mat grayEq) {
        List<Rect> det = detector.detect(grayEq);
        if (det.isEmpty()) return null;
        return det.stream().max(Comparator.comparingInt(r -> r.width * r.height)).orElse(det.get(0));
        }
}
