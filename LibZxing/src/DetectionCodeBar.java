import com.google.zxing.*;
import com.google.zxing.client.j2se.BufferedImageLuminanceSource;
import com.google.zxing.common.HybridBinarizer;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

import static org.opencv.core.Core.flip;

public class DetectionCodeBar {
    public static void detectedCodeBar() {

        VideoCapture capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            System.err.println("Unable to open video");
            System.exit(0);
        }

        Mat frame = new Mat();
        while (true) {
            capture.read(frame);
            if (frame.empty()) {
                break;
            }

            try {
                // Преобразование Mat в BufferedImage
                BufferedImage image = matToBufferedImage(frame);

                LuminanceSource source = new BufferedImageLuminanceSource(image);
                BinaryBitmap bitmap = new BinaryBitmap(new HybridBinarizer(source));
                Result result = new MultiFormatReader().decode(bitmap);
                System.out.println("Текст штрих-кода: " + result.getText());

                printContours(result, frame);

            } catch (NotFoundException _) {
            } catch (IOException e) {
                System.err.println("Ошибка преобразования изображения: " + e.getMessage());
            }

            HighGui.imshow("Frame", frame);
            int keyboard = HighGui.waitKey(30);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }
        }
        HighGui.waitKey();
        System.exit(0);
    }

    //Рисует по точкам области кодов
    private static void printContours(Result result, Mat frame) {
        ResultPoint[] resultPoints = result.getResultPoints();

        if (resultPoints != null && resultPoints.length >= 4) {
            Point[] points = new Point[resultPoints.length];

            for (int i = 0; i < resultPoints.length; i++) {
                points[i] = new Point(resultPoints[i].getX(), resultPoints[i].getY());
            }

            Imgproc.line(frame, points[0], points[1], new Scalar(0, 255, 0), 2);
            Imgproc.line(frame, points[1], points[2], new Scalar(0, 255, 0), 2);
            Imgproc.line(frame, points[2], points[3], new Scalar(0, 255, 0), 2);
            Imgproc.line(frame, points[3], points[0], new Scalar(0, 255, 0), 2);

        } else if (resultPoints != null && resultPoints.length >= 3) {
            Point[] points = new Point[resultPoints.length];

            for (int i = 0; i < resultPoints.length; i++) {
                points[i] = new Point(resultPoints[i].getX(), resultPoints[i].getY());
            }

            Imgproc.line(frame, points[0], points[1], new Scalar(0, 255, 0), 2);
            Imgproc.line(frame, points[1], points[2], new Scalar(0, 255, 0), 2);
            Imgproc.line(frame, points[2], points[0], new Scalar(0, 255, 0), 2);

        } else if (resultPoints != null && resultPoints.length >= 2) {
            Point[] points = new Point[resultPoints.length];

            for (int i = 0; i < resultPoints.length; i++) {
                points[i] = new Point(resultPoints[i].getX(), resultPoints[i].getY());
            }

            Imgproc.line(frame, points[0], points[1], new Scalar(0, 255, 0), 2);
            Imgproc.line(frame, points[1], points[0], new Scalar(0, 255, 0), 2);

        } else {
            System.out.println("Недостаточно точек для обводки штрих-кода.");
        }
    }

    // Преобразование Mat в BufferedImage
    private static BufferedImage matToBufferedImage(Mat matrix) throws IOException {

        MatOfByte mob = new MatOfByte();
        Imgcodecs.imencode(".jpg", matrix, mob); //или другой формат изображения

        byte[] byteArray = mob.toArray();
        BufferedImage bufImage = ImageIO.read(new ByteArrayInputStream(byteArray));

        return bufImage;
    }
}