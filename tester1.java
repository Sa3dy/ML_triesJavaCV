package opencvJavaTester0;

import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_features2d.*;
import static org.bytedeco.javacpp.opencv_flann.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_nonfree.*;
import static org.bytedeco.javacpp.opencv_ml.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

import org.opencv.highgui.Highgui;

public class tester1 {
	public static void main(String[] args) {
		
		org.opencv.core.Mat imgOpenCV = Highgui.imread("imgtst.png");
		
		Mat imgJavaCV = imread("imgtst.png");
		
		System.out.println("imgJavaCV: " + imgJavaCV.rows() * imgJavaCV.cols());
		System.out.println("imgOpenCV: " + imgOpenCV.rows() * imgOpenCV.cols());
		
	}
}
