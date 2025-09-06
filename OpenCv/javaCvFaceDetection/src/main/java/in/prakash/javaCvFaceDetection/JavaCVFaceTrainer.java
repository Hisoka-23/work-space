package in.prakash.javaCvFaceDetection;

import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.opencv_face.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

public class JavaCVFaceTrainer {
    private static final String CASCADE = "data/haarcascade_frontalface_alt2.xml";
    private static final Size FACE_SIZE = new Size(100, 100);

    public static void main(String[] args) throws Exception {
        String datasetDir = "datasets";
        String modelPath = "models/lbph.yml";
        String labelsPath = "models/labels.txt";

        CascadeClassifier detector = new CascadeClassifier(CASCADE);
        if (detector.empty()) throw new RuntimeException("Failed to load cascade: " + CASCADE);

        List<Mat> imagesList = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();
        Map<Integer,String> idToName = new LinkedHashMap<>();
        int nextId = 0;

        try (DirectoryStream<Path> people = Files.newDirectoryStream(Paths.get(datasetDir))) {
            for (Path personDir : people) {
                if (!Files.isDirectory(personDir)) continue;
                String person = personDir.getFileName().toString();
                int id = nextId++;
                idToName.put(id, person);

                List<Path> imgs = Files.list(personDir)
                        .filter(Files::isRegularFile)
                        .filter(p -> {
                            String s = p.getFileName().toString().toLowerCase();
                            return s.endsWith(".jpg")||s.endsWith(".jpeg")||s.endsWith(".png")||s.endsWith(".pgm");
                        }).collect(Collectors.toList());

                for (Path imgPath : imgs) {
                    Mat bgr = imread(imgPath.toString());
                    if (bgr == null || bgr.empty()) continue;

                    Mat gray = new Mat();
                    cvtColor(bgr, gray, COLOR_BGR2GRAY);

                    RectVector faces = new RectVector();
                    detector.detectMultiScale(gray, faces, 1.1, 5, 0, new Size(50, 50), new Size());

                    if (faces.size() == 0) continue;
                    Rect best = null;
                    for (int i=0;i<faces.size();i++){
                        Rect r = faces.get(i);
                        if (best == null || r.area() > best.area()) best = r;
                    }

                    Mat roi = new Mat(gray, best);
                    equalizeHist(roi, roi);
                    Mat resized = new Mat();
                    resize(roi, resized, FACE_SIZE);

                    imagesList.add(resized);
                    labelsList.add(id);
                }
            }
        }

        if (imagesList.isEmpty()) throw new RuntimeException("No faces found in " + datasetDir);

        // Convert to JavaCV types for training
        MatVector images = new MatVector(imagesList.size());
        Mat labels = new Mat(imagesList.size(), 1, CV_32SC1);
        IntRawIndexer labelsIdx = labels.createIndexer();
        for (int i=0;i<imagesList.size();i++){
            images.put(i, imagesList.get(i));
            labelsIdx.put(i, 0, labelsList.get(i));
        }
        labelsIdx.release();

        // Create and train LBPH recognizer
        LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create(1, 8, 8, 8, 100.0);
        recognizer.train(images, labels);

        // Save model
        new File("models").mkdirs();
        recognizer.save(modelPath); // or recognizer.write(modelPath)

        // Save labels map
        try (PrintWriter pw = new PrintWriter(new FileWriter(labelsPath))) {
            for (Map.Entry<Integer,String> e : idToName.entrySet()) {
                pw.println(e.getKey() + "\t" + e.getValue());
            }
        }

        System.out.println("Saved: " + modelPath + " and " + labelsPath);
    }
}
