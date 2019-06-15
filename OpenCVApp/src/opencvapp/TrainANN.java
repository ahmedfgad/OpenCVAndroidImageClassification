package opencvapp;

import java.io.File;
import java.util.Arrays;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;

public class TrainANN {
    public static void main(String [] args){
        String currentDirectory = System.getProperty("user.dir");
        System.load(currentDirectory + "\\OpenCVDLL\\x64\\" + Core.NATIVE_LIBRARY_NAME + ".dll");

        Mat datasetHist = new Mat();
        Mat datasetLabels = new Mat();
        
        String [] classesNames = {"apple", "lemon", "mango", "raspberry"};
        
        for(int classIdx=0; classIdx<classesNames.length; classIdx++){
            String currClassName = classesNames[classIdx];
            String currClassDir = currentDirectory + "\\Dataset\\Train\\" + currClassName + "\\";
            System.out.println("Current Class Directory : " + currClassDir);
            
            File folder = new File(currClassDir);
            File[] listOfFiles = folder.listFiles();
            
            // Counter for the number of image being processed within the class.
            int imgCount = 0;

            for (File listOfFile : listOfFiles) {
                // Make sure we are working with a file and its extension is JPG
                if (listOfFile.isFile() && (currClassDir + listOfFile.getName()).endsWith(".jpg")) {
                    System.out.println("Class Index " + classIdx + "(" + currClassName + ")" + ", Image Index " + imgCount + "(" + listOfFile.getName() + ")");
                    
                    String currImgPath = currClassDir + listOfFile.getName();
                    System.out.println(currImgPath);
                    Mat imgBGR = Imgcodecs.imread(currImgPath);
    //                int numRows = imgBGR.rows();
    //                int numCols = imgBGR.cols();
    //                int numChannels = imgBGR.channels();
    //                System.out.println("Image Size : (" + numRows + ", " + numCols + ", " + numChannels + ")");

                    Mat imgHSV = new Mat();
                    Imgproc.cvtColor(imgBGR, imgHSV, Imgproc.COLOR_BGR2HSV);

                    // Preparing parameters of Imgproc.calcHist().
                    MatOfInt selectedChannels = new MatOfInt(0); // Selecting the first channel in HSV which is Hue.
                    Mat imgHist = new Mat();
                    MatOfInt histSize = new MatOfInt(180); // Number of bins in the histogram.
                    MatOfFloat ranges = new MatOfFloat(0f, 180f); // Uniform bin distribution. Start value of first bin is 0 and end value for last bin is 180.

                    // Doc: https://docs.opencv.org/3.1.0/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d
                    Imgproc.calcHist(Arrays.asList(imgHSV), selectedChannels, new Mat(), imgHist, histSize, ranges);

                    // Transposing the histogram Mat from being 1D column vector to be 1D row vector.
                    imgHist = imgHist.t();
                    System.out.println("Hue Channel Hist : " + imgHist.dump());
                    System.out.println("Image Hist Size : (" + imgHist.rows() + ", " + imgHist.cols()+ ")\n");

                    // Inserting the extracted histogram of the current image into the Mat collecting the histograms of all images.
                    datasetHist.push_back(imgHist);
                    datasetLabels.push_back(new MatOfInt(classIdx));
                    
                    imgCount++;
                }
            }
        }
        
        // Converting the type of the features & labels Mats into CV_32F because ANN accepts data of this type.
        datasetHist.convertTo(datasetHist, CvType.CV_32F);
        datasetLabels.convertTo(datasetLabels, CvType.CV_32F);
        
        System.out.println("Dataset Hist Size : (" + datasetHist.rows() + ", " + datasetHist.cols()+ ")");
        System.out.println("Dataset Label Size : (" + datasetLabels.rows() + ", " + datasetLabels.cols()+ ")");
        
        ANN_MLP ANN = ANN_MLP.create();

        Mat layerSizes = new Mat(4, 1, CvType.CV_8U);
        layerSizes.put(0, 0, 180); // 180 input neurons in the input layer because the histogram length is 180.
        layerSizes.put(1, 0, 60); // 60 hidden neurons in the first hidden layer.
        layerSizes.put(2, 0, 20); // 20 hidden neurons in the second hidden layer.
        layerSizes.put(3, 0, 1); // One output neuron in the output layer.
        ANN.setLayerSizes(layerSizes);
        System.out.println("Layers Sizes : \n" + layerSizes.dump());

        ANN.setActivationFunction(ANN_MLP.SIGMOID_SYM);
        ANN.setTrainMethod(ANN_MLP.BACKPROP);

        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 10000, 0.00000001);
        ANN.setTermCriteria(criteria);

        ANN.train(datasetHist, Ml.ROW_SAMPLE, datasetLabels);
        
        try{
            ANN.save(currentDirectory + "\\OpenCV_ANN_Fruits.yml");
            System.out.println("Model Saved Successfully.");
        } catch(Exception ex) {
            System.err.println("Error Saving Model.");
        }
    }    
}
