import org.opencv.core.Core;

public class DetectionCodeBarDemo {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // Загрузка библиотеки OpenCV
    }
    public static void main(String[] args) {
        DetectionCodeBar.detectedCodeBar();
    }
}