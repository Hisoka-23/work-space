package in.prakash.controller;

import com.example.facerec.camera.CameraService;
import com.example.facerec.detect.FaceDetector;
import com.example.facerec.recog.FaceRecognizerService;
import com.example.facerec.util.ImageUtils;
import com.example.facerec.util.Labels;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.*;

import java.io.File;
import java.util.List;

public class MainController {
    @FXML private ImageView imageView;
    @FXML private Button btnStart, btnStop, btnTrain, btnEnroll;
    @FXML private Label lblStatus;
    @FXML private TextField txtThreshold;

    private CameraService camera;
    private FaceDetector detector;
    private FaceRecognizerService recognizer;
    private Labels labels;

    @FXML
    public void initialize() {
        detector = new FaceDetector("cascades/haarcascade_frontalface_alt2.xml");
        labels = new Labels(new File("model/labels.json"));
        recognizer = new FaceRecognizerService(new File("model/lbph-model.xml"), labels, 85.0);
        lblStatus.setText("Ready");
    }

    @FXML
    public void onStart() {
        double thr = parseThreshold();
        recognizer.setThreshold(thr);
        camera = new CameraService(0, frame -> {
            // Process frame: detect, recognize, annotate
            Mat display = frame.clone();
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(gray, gray);

            List<Rect> faces = detector.detect(gray);
            for (Rect r : faces) {
                Mat faceROI = new Mat(gray, r);
                String labelText = recognizer.predictLabel(faceROI);
                Imgproc.rectangle(display, r, new Scalar(0, 255, 0), 2);
                Imgproc.putText(display, labelText, new Point(r.x, Math.max(0, r.y - 8)),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0,255,0), 2);
            }
            Image img = ImageUtils.matToImage(display);
            Platform.runLater(() -> imageView.setImage(img));
        });
        camera.start();
        lblStatus.setText("Streaming");
    }

    @FXML
    public void onStop() {
        if (camera != null) camera.stop();
        lblStatus.setText("Stopped");
    }

    @FXML
    public void onTrain() {
        try {
            recognizer.trainFromDataset(new File("dataset"), detector);
            lblStatus.setText("Model trained");
        } catch (Exception ex) {
            lblStatus.setText("Train error: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    @FXML
    public void onEnroll() {
        TextInputDialog dlg = new TextInputDialog("person_name");
        dlg.setTitle("Enroll");
        dlg.setHeaderText("Enter person name to capture 20 samples");
        dlg.setContentText("Name:");
        String name = dlg.showAndWait().orElse(null);
        if (name == null || name.isBlank()) return;

        File dir = new File("dataset/" + name);
        dir.mkdirs();
        lblStatus.setText("Enrolling: " + name);

        // simple capture loop: take 20 samples over live video
        final int target = 20;
        final int[] taken = {0};
        camera.setFrameConsumer(frame -> {
            Mat show = frame.clone();
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(gray, gray);

            List<Rect> faces = detector.detect(gray);
            for (Rect r : faces) {
                Mat roi = new Mat(gray, r);
                Mat norm = recognizer.normalizeFace(roi);
                if (taken < target) {
                    Imgcodecs.imwrite(new File(dir, "img_" + System.nanoTime() + ".jpg").getAbsolutePath(), norm);
                    taken++;
                }
                Imgproc.rectangle(show, r, new Scalar(255, 0, 0), 2);
            }
            Image img = ImageUtils.matToImage(show);
            Platform.runLater(() -> imageView.setImage(img));

            if (taken >= target) {
                // revert to normal recognition consumer
                camera.setDefaultConsumer();
                Platform.runLater(() -> lblStatus.setText("Enrollment done for " + name));
            }
        });
    }

    private double parseThreshold() {
        try { return Double.parseDouble(txtThreshold.getText().trim()); }
        catch (Exception e) { return 85.0; }
    }
}

