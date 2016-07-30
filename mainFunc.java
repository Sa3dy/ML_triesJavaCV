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

import org.bytedeco.javacpp.opencv_features2d.Feature2D;
import org.bytedeco.javacpp.annotation.Const;

public class mainFunc {
	
	static DescriptorMatcher matcher = new DescriptorMatcher();
	static DescriptorExtractor extractor = new DescriptorExtractor();
	
	SURF detector = new SURF(500);
	
	int dictionarySize = 1500;
	TermCriteria tc = new TermCriteria(CV_TERMCRIT_ITER, 10, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	BOWKMeansTrainer bowTrainer = new BOWKMeansTrainer(dictionarySize, tc, retries, flags);
	static BOWImgDescriptorExtractor bowDE = new BOWImgDescriptorExtractor(extractor, matcher);

	public static void main(String[] args) {
		
		Mat src = imread("imgtst.png");
		
		System.out.println("src: " + src.elemSize());
		
		SURF dSURF = new SURF();
		KeyPoint keypoints = new KeyPoint();
		dSURF.detect(src, keypoints);
		
		System.out.println("keypoints: " + keypoints.size());
		
		Mat dictionary = new Mat();
		
		FileStorage fs_reader = new FileStorage("vocabulary.xml", FileStorage.READ);

		read(fs_reader.get("vocabulary"), dictionary);

		System.out.println("dictionary.rows * dictionary.cols: " + dictionary.rows() * dictionary.cols());

		fs_reader.release();

		bowDE.setVocabulary(dictionary);
		
	}

}
