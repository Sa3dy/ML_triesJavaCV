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

import org.bytedeco.javacpp.opencv_core.FileStorage;

public class tester0 {

	public static void main(String[] args) {

		int i, j;
		Mat dictionary = new Mat();
		FeatureDetector fd = FeatureDetector.create("FAST");
		DescriptorExtractor extractor = DescriptorExtractor.create("FREAK");
		BFMatcher matcher = new BFMatcher();
		int dictionarySize = 1500;

		TermCriteria termcrit = new TermCriteria(CV_TERMCRIT_ITER, 10, 0.001);
		BOWKMeansTrainer bowTrainer = new BOWKMeansTrainer(dictionarySize, termcrit, 1,
				KMEANS_PP_CENTERS);
		BOWImgDescriptorExtractor bowDE = new BOWImgDescriptorExtractor(extractor, matcher);

		KeyPoint keypoint = new KeyPoint();
		
		FileStorage fs_reader = new FileStorage("vocabulary.xml", FileStorage.READ);
		read(fs_reader.get("vocabulary"), dictionary);
		System.out.println("dictionary.rows * dictionary.cols: " + dictionary.rows() * dictionary.cols());
		fs_reader.release();
		
		if (!dictionary.empty()) {
			System.out.println("Working On Training Images...");
			for (j = 1; j <= 4; j++) {
				for (i = 1; i <= 60; i++) {
					String path = "C:\\Users\\Sa3dyLAP\\Documents\\Visual Studio 2010\\Projects\\OpenCV_ML_Testing\\OpenCV_ML_Testing\\train\\"
							+ j + " (" + i + ").jpg";
					System.out.println("Processing file: " + j + " (" + i
							+ ").jpg");
					Mat img2 = imread(path, CV_LOAD_IMAGE_COLOR);
					System.out.println("Size of file: " + j + " (" + i
							+ ").jpg is: " + (img2.rows() * img2.cols()));
					fd.detect(img2, keypoint);
					Mat features = new Mat();
					extractor.compute(img2, keypoint, features);
					Mat feature = new Mat();
					features.convertTo(feature, CV_32FC1);
					bowTrainer.add(feature);
				}
			}

			System.out.println("Clustering features...");
			dictionary = bowTrainer.cluster();
			System.out.println("Features clustered.");
			
			FileStorage fs_writer = new FileStorage("vocabulary.xml", FileStorage.WRITE);
			write(fs_writer, "vocabulary", dictionary);
			fs_writer.release();
		}
		
		bowDE.setVocabulary(dictionary);

		CvMat labels = CvMat.create(480, 1, CV_32FC1);
		CvMat trainingData = CvMat.create(0, 1, CV_32FC1);
		Mat bowDescriptor1 = new Mat();
		Mat bowDesc1 = new Mat();
		KeyPoint keypoint1 = new KeyPoint();
		
		System.out.println("Working On Testing Images...");
		for (j = 1; j <= 4; j++) {
			for (i = 1; i <= 60; i++) {
				String path = "C:\\Users\\Sa3dyLAP\\Documents\\Visual Studio 2010\\Projects\\OpenCV_ML_Testing\\OpenCV_ML_Testing\\eval\\"
						+ j + " (" + i + ").jpg";
				Mat img2 = imread(path, CV_LOAD_IMAGE_COLOR);
				System.out.println("Processing file: " + j + " (" + i + ").jpg");
				fd.detect(img2, keypoint1);
				extractor.compute(img2, keypoint1, bowDescriptor1);
				bowDescriptor1.convertTo(bowDesc1, CV_32FC1);
				trainingData.put(new Mat(bowDesc1.reshape(1, 1)));
				labels.put((float)j);
				System.out.println("Processing file: " + j + " (" + i + ").jpg putted in the testingData.");
			}
		}
		if (CV_MAT_TYPE(trainingData.type()) != CV_32FC1) {
			System.out.println("train data must be floating-point matrix");
		}

		if (CV_MAT_TYPE(labels.type()) != CV_32FC1) {
			System.out.println("train data must be floating-point matrix");
		}
		
		System.out.println("SVM Params Initialization...");
		CvTermCriteria criteria = new CvTermCriteria();
		criteria.type(CV_TERMCRIT_ITER);
		criteria.max_iter(100);
		criteria.epsilon(0.000001);
		CvSVMParams params = new CvSVMParams();
		params.kernel_type(CvSVM.RBF);
		params.svm_type(CvSVM.C_SVC);
		params.gamma(0.50625000000000009);
		params.C(312.50000000000000);
		params.term_crit(criteria);
		System.out.println("SVM Params Initialized...");
		
		CvSVM svm = new CvSVM(params);
		
		Mat trainingDataMat = new Mat(trainingData);
		Mat labelsMat = new Mat(labels);
		
		System.out.println("Training SVM Started...");
		svm.train(trainingDataMat, labelsMat, new Mat(), new Mat(), params);
		String svmpath = "dictionary1.xml";
		svm.save(svmpath);

	}

}
