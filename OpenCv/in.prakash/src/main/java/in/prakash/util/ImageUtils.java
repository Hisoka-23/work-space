package in.prakash.util;

import javafx.scene.image.Image;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.ByteArrayInputStream;

public class ImageUtils {
    public static Image matToImage(Mat mat) {
        MatOfByte mob = new MatOfByte();
        // BMP is usually a bit faster to encode/decode than PNG
        Imgcodecs.imencode(".bmp", mat, mob);
        return new Image(new ByteArrayInputStream(mob.toArray()));
    }
}

