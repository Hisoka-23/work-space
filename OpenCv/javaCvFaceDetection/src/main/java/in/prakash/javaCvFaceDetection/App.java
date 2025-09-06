package in.prakash.javaCvFaceDetection;

import java.awt.Image;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import demo.opencv.imgProcessor;

/**
 * Hello world!
 *
 */
public class App 
{
	
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	//imgProcessor proc;
	private JFrame frame;
	private  JLabel imageLabel;
	private CascadeClassifier faceDetector;
	
    public static void main( String[] args )
    {
       
		App app = new App();
		app.initGUI();//A method for initializing gui components
		
		app.loadCascade();//Lodding the cascades classifiers
		
		app.runMainLoop(args);
    	
    }
    
    
    private void loadCascade() {
		String cascadePath = "data/haarcascade_frontalface_alt2.xml";//path for the cascade classifier
		faceDetector = new CascadeClassifier(cascadePath);
	}
	private void initGUI() {
		// Loading gui components
		frame = new JFrame("Face detection");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(400,400);
		
		imageLabel = new JLabel();
		frame.add(imageLabel);
		frame.setVisible(true);
	}
	private void runMainLoop(String[] args) {
		imgProcessor imageProcessor = new imgProcessor();
		Mat webcamImage = new Mat();
		Image tempImage;
		
		VideoCapture capture = new VideoCapture(0);
		capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
		capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
		if(capture.isOpened()) {
			while(true) {
				capture.read(webcamImage);
				if(!webcamImage.empty()) {
					detectAndDrawFace(webcamImage);
					tempImage = imageProcessor.toBufferImage(webcamImage);//Convert image to buffer from our imgProcess class
					
					ImageIcon imageIcon= new ImageIcon(tempImage, "Captured Video");
					imageLabel.setIcon(imageIcon);
					frame.pack();
				}
				else {
					JOptionPane.showMessageDialog(frame, "Error ----- could not capture frames");
				}
			}
		}
		else {
			JOptionPane.showMessageDialog(frame, "Could not open webcam");
		}	
	}
	private void detectAndDrawFace(Mat image) {
		// Detect face and draw
		MatOfRect faceDetection = new MatOfRect();
		faceDetector.detectMultiScale(image, faceDetection,1.1,7,0,new Size(50,20),new Size());
		//Draw a box around each face
		for(Rect rect : faceDetection.toArray()) {
			Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width,rect.y + rect.height),
					new Scalar(0,255,0));
			//Put text around the rectangle
			Imgproc.putText(image, "Face", new Point(rect.x + rect.height, rect.y + rect.width / 10),
					1,1,new Scalar(124,252,0),1);
		}	
	}
    
}
