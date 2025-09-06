package in.prakash.camera;

import javafx.util.Duration;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import java.util.concurrent.*;
import java.util.function.Consumer;

public class CameraService {
    private final int cameraIndex;
    private final VideoCapture capture;
    private final ScheduledExecutorService exec;
    private Consumer<Mat> frameConsumer;
    private Consumer<Mat> defaultConsumer;
    private volatile boolean running;

    public CameraService(int cameraIndex, Consumer<Mat> frameConsumer) {
        this.cameraIndex = cameraIndex;
        this.capture = new VideoCapture();
        this.exec = Executors.newSingleThreadScheduledExecutor();
        this.frameConsumer = frameConsumer;
        this.defaultConsumer = frameConsumer;
    }

    public void start() {
        if (running) return;
        running = true;
        capture.open(cameraIndex);
        exec.scheduleAtFixedRate(this::grab, 0, 33, TimeUnit.MILLISECONDS); // ~30fps
    }

    public void stop() {
        running = false;
        exec.shutdownNow();
        if (capture.isOpened()) capture.release();
    }

    private void grab() {
        if (!running) return;
        if (!capture.isOpened()) return;
        Mat frame = new Mat();
        if (capture.read(frame)) {
            Consumer<Mat> consumer = frameConsumer;
            if (consumer != null) consumer.accept(frame);
        }
    }

    public void setFrameConsumer(Consumer<Mat> consumer) {
        this.frameConsumer = consumer;
    }

    public void setDefaultConsumer() {
        this.frameConsumer = this.defaultConsumer;
    }
}
