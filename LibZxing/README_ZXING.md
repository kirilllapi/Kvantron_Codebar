# Библиотека Zxing 
Данная библиотека преобразует исходное изображение в бинарное изображение, после чего (в нашем случае, для проверки работы на основных видах штрихкодов) полученное изображение передается в метод ```decodeInternal```, в котором перебираются все декодеры (для каждого типа штрихкода), с вызовом метода (для 2D ```decode```, а для 1D ```decodeRow```) для нахождения особых признаков соответствующего штрихкода. У каждого штрихкода представлен свой локализатор, который ищет эти признаки на бинарном изображении.
## Для запуска использовать класс  ```DetectionCodeBarDemo```.
Для этого необходимо перезагрузить все зависимости ```pom.xml``` Maven.
(Добавлена библиотека OpenCV).
В классе **```DetectionBar```** прописаны методы распознование штрихкодов и отрисовки границ.
# Результат запуска: 

https://github.com/user-attachments/assets/c76faa39-608b-468c-9621-7978f273c666



# Описание ```DetectionBar```
```java
class DetectionCodeBar {
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
```
1. **```BufferedImageLuminanceSource```** - преобразование изображения в ЧБ с сохранением яркости пикселей, необходимого для дальнейшего определения, 
какие пиксели принадлежат темным или светлым областям.
### Взято из ```DetectionCodeBar```:
```java
LuminanceSource source = new BufferedImageLuminanceSource(image);
```

2. **```HybridBinarizer```** - преобразовывает полученное ЧБ изображение в бинарное изображение по порогу. В данном случае используется
глобальная и локальная адаптация яркости.
### Взято из ```DetectionCodeBar```:
```java
 BinaryBitmap bitmap = new BinaryBitmap(new HybridBinarizer(source));
```
- ***Глобальная адаптация яркости***: рассчитывает **среднюю яркость** всего изображения и использует его для базового порога,
который будет использоваться для преобразования в бинарное изображение
- ***Локальная адаптация яркости***: расчитывает по участкам для учитывания всяких проблем с освещением (блики, тени и т.д.)

### Описание класса ```HybridBinarizer```:
```java
public final class HybridBinarizer extends GlobalHistogramBinarizer {

  // This class uses 5x5 blocks to compute local luminance, where each block is 8x8 pixels.
  // So this is the smallest dimension in each axis we can accept.
  private static final int BLOCK_SIZE_POWER = 3;
  private static final int BLOCK_SIZE = 1 << BLOCK_SIZE_POWER; // ...0100...00
  private static final int BLOCK_SIZE_MASK = BLOCK_SIZE - 1;   // ...0011...11
  private static final int MINIMUM_DIMENSION = BLOCK_SIZE * 5;
  private static final int MIN_DYNAMIC_RANGE = 24;

  private BitMatrix matrix;

  public HybridBinarizer(LuminanceSource source) {
    super(source);
  }

  /**
   * Calculates the final BitMatrix once for all requests. This could be called once from the
   * constructor instead, but there are some advantages to doing it lazily, such as making
   * profiling easier, and not doing heavy lifting when callers don't expect it.
   */
  @Override
  public BitMatrix getBlackMatrix() throws NotFoundException {
    if (matrix != null) {
      return matrix;
    }
    LuminanceSource source = getLuminanceSource();
    int width = source.getWidth();
    int height = source.getHeight();
    if (width >= MINIMUM_DIMENSION && height >= MINIMUM_DIMENSION) {
      byte[] luminances = source.getMatrix();
      int subWidth = width >> BLOCK_SIZE_POWER;
      if ((width & BLOCK_SIZE_MASK) != 0) {
        subWidth++;
      }
      int subHeight = height >> BLOCK_SIZE_POWER;
      if ((height & BLOCK_SIZE_MASK) != 0) {
        subHeight++;
      }
      int[][] blackPoints = calculateBlackPoints(luminances, subWidth, subHeight, width, height);

      BitMatrix newMatrix = new BitMatrix(width, height);
      calculateThresholdForBlock(luminances, subWidth, subHeight, width, height, blackPoints, newMatrix);
      matrix = newMatrix;
    } else {
      // If the image is too small, fall back to the global histogram approach.
      matrix = super.getBlackMatrix();
    }
    return matrix;
  }

  @Override
  public Binarizer createBinarizer(LuminanceSource source) {
    return new HybridBinarizer(source);
  }

  /**
   * For each block in the image, calculate the average black point using a 5x5 grid
   * of the blocks around it. Also handles the corner cases (fractional blocks are computed based
   * on the last pixels in the row/column which are also used in the previous block).
   */
  private static void calculateThresholdForBlock(byte[] luminances,
                                                 int subWidth,
                                                 int subHeight,
                                                 int width,
                                                 int height,
                                                 int[][] blackPoints,
                                                 BitMatrix matrix) {
    int maxYOffset = height - BLOCK_SIZE;
    int maxXOffset = width - BLOCK_SIZE;
    for (int y = 0; y < subHeight; y++) {
      int yoffset = y << BLOCK_SIZE_POWER;
      if (yoffset > maxYOffset) {
        yoffset = maxYOffset;
      }
      int top = cap(y, subHeight - 3);
      for (int x = 0; x < subWidth; x++) {
        int xoffset = x << BLOCK_SIZE_POWER;
        if (xoffset > maxXOffset) {
          xoffset = maxXOffset;
        }
        int left = cap(x, subWidth - 3);
        int sum = 0;
        for (int z = -2; z <= 2; z++) {
          int[] blackRow = blackPoints[top + z];
          sum += blackRow[left - 2] + blackRow[left - 1] + blackRow[left] + blackRow[left + 1] + blackRow[left + 2];
        }
        int average = sum / 25;
        thresholdBlock(luminances, xoffset, yoffset, average, width, matrix);
      }
    }
  }

  private static int cap(int value, int max) {
    return value < 2 ? 2 : Math.min(value, max);
  }

  /**
   * Applies a single threshold to a block of pixels.
   */
  private static void thresholdBlock(byte[] luminances,
                                     int xoffset,
                                     int yoffset,
                                     int threshold,
                                     int stride,
                                     BitMatrix matrix) {
    for (int y = 0, offset = yoffset * stride + xoffset; y < BLOCK_SIZE; y++, offset += stride) {
      for (int x = 0; x < BLOCK_SIZE; x++) {
        // Comparison needs to be <= so that black == 0 pixels are black even if the threshold is 0.
        if ((luminances[offset + x] & 0xFF) <= threshold) {
          matrix.set(xoffset + x, yoffset + y);
        }
      }
    }
  }

  /**
   * Calculates a single black point for each block of pixels and saves it away.
   * See the following thread for a discussion of this algorithm:
   *  http://groups.google.com/group/zxing/browse_thread/thread/d06efa2c35a7ddc0
   */
  private static int[][] calculateBlackPoints(byte[] luminances,
                                              int subWidth,
                                              int subHeight,
                                              int width,
                                              int height) {
    int maxYOffset = height - BLOCK_SIZE;
    int maxXOffset = width - BLOCK_SIZE;
    int[][] blackPoints = new int[subHeight][subWidth];
    for (int y = 0; y < subHeight; y++) {
      int yoffset = y << BLOCK_SIZE_POWER;
      if (yoffset > maxYOffset) {
        yoffset = maxYOffset;
      }
      for (int x = 0; x < subWidth; x++) {
        int xoffset = x << BLOCK_SIZE_POWER;
        if (xoffset > maxXOffset) {
          xoffset = maxXOffset;
        }
        int sum = 0;
        int min = 0xFF;
        int max = 0;
        for (int yy = 0, offset = yoffset * width + xoffset; yy < BLOCK_SIZE; yy++, offset += width) {
          for (int xx = 0; xx < BLOCK_SIZE; xx++) {
            int pixel = luminances[offset + xx] & 0xFF;
            sum += pixel;
            // still looking for good contrast
            if (pixel < min) {
              min = pixel;
            }
            if (pixel > max) {
              max = pixel;
            }
          }
          // short-circuit min/max tests once dynamic range is met
          if (max - min > MIN_DYNAMIC_RANGE) {
            // finish the rest of the rows quickly
            for (yy++, offset += width; yy < BLOCK_SIZE; yy++, offset += width) {
              for (int xx = 0; xx < BLOCK_SIZE; xx++) {
                sum += luminances[offset + xx] & 0xFF;
              }
            }
          }
        }

        // The default estimate is the average of the values in the block.
        int average = sum >> (BLOCK_SIZE_POWER * 2);
        if (max - min <= MIN_DYNAMIC_RANGE) {
          // If variation within the block is low, assume this is a block with only light or only
          // dark pixels. In that case we do not want to use the average, as it would divide this
          // low contrast area into black and white pixels, essentially creating data out of noise.
          //
          // The default assumption is that the block is light/background. Since no estimate for
          // the level of dark pixels exists locally, use half the min for the block.
          average = min / 2;

          if (y > 0 && x > 0) {
            // Correct the "white background" assumption for blocks that have neighbors by comparing
            // the pixels in this block to the previously calculated black points. This is based on
            // the fact that dark barcode symbology is always surrounded by some amount of light
            // background for which reasonable black point estimates were made. The bp estimated at
            // the boundaries is used for the interior.

            // The (min < bp) is arbitrary but works better than other heuristics that were tried.
            int averageNeighborBlackPoint =
                (blackPoints[y - 1][x] + (2 * blackPoints[y][x - 1]) + blackPoints[y - 1][x - 1]) / 4;
            if (min < averageNeighborBlackPoint) {
              average = averageNeighborBlackPoint;
            }
          }
        }
        blackPoints[y][x] = average;
      }
    }
    return blackPoints;
  }
}
```
- ```calculateBlackPoints``` - рассчитывает средний порог яркости для каждого блока (ядра).
- ```thresholdBlock``` - пременяет пороговую яркость, для пикселей меньше порога присваивается черный цвет (0), а для больших белый (1). Можно сказать, производится пороговая сегментация штрихкода от фона.
### Описание класса ```BinaryBitmap```:
```java
public final class BinaryBitmap {

  private final Binarizer binarizer;
  private BitMatrix matrix;

  public BinaryBitmap(Binarizer binarizer) {
    if (binarizer == null) {
      throw new IllegalArgumentException("Binarizer must be non-null.");
    }
    this.binarizer = binarizer;
  }

  /**
   * @return The width of the bitmap.
   */
  public int getWidth() {
    return binarizer.getWidth();
  }

  /**
   * @return The height of the bitmap.
   */
  public int getHeight() {
    return binarizer.getHeight();
  }

  /**
   * Converts one row of luminance data to 1 bit data. May actually do the conversion, or return
   * cached data. Callers should assume this method is expensive and call it as seldom as possible.
   * This method is intended for decoding 1D barcodes and may choose to apply sharpening.
   *
   * @param y The row to fetch, which must be in [0, bitmap height)
   * @param row An optional preallocated array. If null or too small, it will be ignored.
   *            If used, the Binarizer will call BitArray.clear(). Always use the returned object.
   * @return The array of bits for this row (true means black).
   * @throws NotFoundException if row can't be binarized
   */
  public BitArray getBlackRow(int y, BitArray row) throws NotFoundException {
    return binarizer.getBlackRow(y, row);
  }

  /**
   * Converts a 2D array of luminance data to 1 bit. As above, assume this method is expensive
   * and do not call it repeatedly. This method is intended for decoding 2D barcodes and may or
   * may not apply sharpening. Therefore, a row from this matrix may not be identical to one
   * fetched using getBlackRow(), so don't mix and match between them.
   *
   * @return The 2D array of bits for the image (true means black).
   * @throws NotFoundException if image can't be binarized to make a matrix
   */
  public BitMatrix getBlackMatrix() throws NotFoundException {
    // The matrix is created on demand the first time it is requested, then cached. There are two
    // reasons for this:
    // 1. This work will never be done if the caller only installs 1D Reader objects, or if a
    //    1D Reader finds a barcode before the 2D Readers run.
    // 2. This work will only be done once even if the caller installs multiple 2D Readers.
    if (matrix == null) {
      matrix = binarizer.getBlackMatrix();
    }
    return matrix;
  }

  /**
   * @return Whether this bitmap can be cropped.
   */
  public boolean isCropSupported() {
    return binarizer.getLuminanceSource().isCropSupported();
  }

  /**
   * Returns a new object with cropped image data. Implementations may keep a reference to the
   * original data rather than a copy. Only callable if isCropSupported() is true.
   *
   * @param left The left coordinate, which must be in [0,getWidth())
   * @param top The top coordinate, which must be in [0,getHeight())
   * @param width The width of the rectangle to crop.
   * @param height The height of the rectangle to crop.
   * @return A cropped version of this object.
   */
  public BinaryBitmap crop(int left, int top, int width, int height) {
    LuminanceSource newSource = binarizer.getLuminanceSource().crop(left, top, width, height);
    return new BinaryBitmap(binarizer.createBinarizer(newSource));
  }

  /**
   * @return Whether this bitmap supports counter-clockwise rotation.
   */
  public boolean isRotateSupported() {
    return binarizer.getLuminanceSource().isRotateSupported();
  }

  /**
   * Returns a new object with rotated image data by 90 degrees counterclockwise.
   * Only callable if {@link #isRotateSupported()} is true.
   *
   * @return A rotated version of this object.
   */
  public BinaryBitmap rotateCounterClockwise() {
    LuminanceSource newSource = binarizer.getLuminanceSource().rotateCounterClockwise();
    return new BinaryBitmap(binarizer.createBinarizer(newSource));
  }

  /**
   * Returns a new object with rotated image data by 45 degrees counterclockwise.
   * Only callable if {@link #isRotateSupported()} is true.
   *
   * @return A rotated version of this object.
   */
  public BinaryBitmap rotateCounterClockwise45() {
    LuminanceSource newSource = binarizer.getLuminanceSource().rotateCounterClockwise45();
    return new BinaryBitmap(binarizer.createBinarizer(newSource));
  }

  @Override
  public String toString() {
    try {
      return getBlackMatrix().toString();
    } catch (NotFoundException e) {
      return "";
    }
  }
}
```
- ```crop``` - обрезка изображения.
- ```isRotateSupported()``` - проверяет на возможность переворота изображения.
- ```rotateCounterClockwise()``` - поворачивает изображение на 90 градусов против часовой стрелки. Если исходное изображение имеет определенную ориентацию, то оно будет перевернуто на 90 градусов влево. Возвращает новый объект ```BinaryBitmap```, который содержит повернутое изображение.
- ```rotateCounterClockwise45()``` - поворачивает изображение на 45 градусов против часовой стрелки. Также возвращает перевернутое изображение в ```BinaryBitmap```
3. *```MultiFormatReader```* - класс декодеров для всех видов штрихкодов, в который передается полученное бинарное изображение.
### Взято из ```DetectionCodeBar```:
 ```java
 Result result = new MultiFormatReader().decode(bitmap);
```
### Описание класса ```MultiFormatReader```:
```java
public final class MultiFormatReader implements Reader {

  private static final Reader[] EMPTY_READER_ARRAY = new Reader[0];

  private Map<DecodeHintType,?> hints;
  private Reader[] readers;

  /**
   * This version of decode honors the intent of Reader.decode(BinaryBitmap) in that it
   * passes null as a hint to the decoders. However, that makes it inefficient to call repeatedly.
   * Use setHints() followed by decodeWithState() for continuous scan applications.
   *
   * @param image The pixel data to decode
   * @return The contents of the image
   * @throws NotFoundException Any errors which occurred
   */
  @Override
  public Result decode(BinaryBitmap image) throws NotFoundException {
    setHints(null);
    return decodeInternal(image);
  }

  /**
   * Decode an image using the hints provided. Does not honor existing state.
   *
   * @param image The pixel data to decode
   * @param hints The hints to use, clearing the previous state.
   * @return The contents of the image
   * @throws NotFoundException Any errors which occurred
   */
  @Override
  public Result decode(BinaryBitmap image, Map<DecodeHintType,?> hints) throws NotFoundException {
    setHints(hints);
    return decodeInternal(image);
  }

  /**
   * Decode an image using the state set up by calling setHints() previously. Continuous scan
   * clients will get a <b>large</b> speed increase by using this instead of decode().
   *
   * @param image The pixel data to decode
   * @return The contents of the image
   * @throws NotFoundException Any errors which occurred
   */
  public Result decodeWithState(BinaryBitmap image) throws NotFoundException {
    // Make sure to set up the default state so we don't crash
    if (readers == null) {
      setHints(null);
    }
    return decodeInternal(image);
  }

  /**
   * This method adds state to the MultiFormatReader. By setting the hints once, subsequent calls
   * to decodeWithState(image) can reuse the same set of readers without reallocating memory. This
   * is important for performance in continuous scan clients.
   *
   * @param hints The set of hints to use for subsequent calls to decode(image)
   */
  public void setHints(Map<DecodeHintType,?> hints) {
    this.hints = hints;

    boolean tryHarder = hints != null && hints.containsKey(DecodeHintType.TRY_HARDER);
    @SuppressWarnings("unchecked")
    Collection<BarcodeFormat> formats =
        hints == null ? null : (Collection<BarcodeFormat>) hints.get(DecodeHintType.POSSIBLE_FORMATS);
    Collection<Reader> readers = new ArrayList<>();
    if (formats != null) {
      boolean addOneDReader =
          formats.contains(BarcodeFormat.UPC_A) ||
          formats.contains(BarcodeFormat.UPC_E) ||
          formats.contains(BarcodeFormat.EAN_13) ||
          formats.contains(BarcodeFormat.EAN_8) ||
          formats.contains(BarcodeFormat.CODABAR) ||
          formats.contains(BarcodeFormat.CODE_39) ||
          formats.contains(BarcodeFormat.CODE_93) ||
          formats.contains(BarcodeFormat.CODE_128) ||
          formats.contains(BarcodeFormat.ITF) ||
          formats.contains(BarcodeFormat.RSS_14) ||
          formats.contains(BarcodeFormat.RSS_EXPANDED);
      // Put 1D readers upfront in "normal" mode
      if (addOneDReader && !tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }
      if (formats.contains(BarcodeFormat.QR_CODE)) {
        readers.add(new QRCodeReader());
      }
      if (formats.contains(BarcodeFormat.DATA_MATRIX)) {
        readers.add(new DataMatrixReader());
      }
      if (formats.contains(BarcodeFormat.AZTEC)) {
        readers.add(new AztecReader());
      }
      if (formats.contains(BarcodeFormat.PDF_417)) {
        readers.add(new PDF417Reader());
      }
      if (formats.contains(BarcodeFormat.MAXICODE)) {
        readers.add(new MaxiCodeReader());
      }
      // At end in "try harder" mode
      if (addOneDReader && tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }
    }
    if (readers.isEmpty()) {
      if (!tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }

      readers.add(new QRCodeReader());
      readers.add(new DataMatrixReader());
      readers.add(new AztecReader());
      readers.add(new PDF417Reader());
      readers.add(new MaxiCodeReader());

      if (tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }
    }
    this.readers = readers.toArray(EMPTY_READER_ARRAY);
  }

  @Override
  public void reset() {
    if (readers != null) {
      for (Reader reader : readers) {
        reader.reset();
      }
    }
  }

  private Result decodeInternal(BinaryBitmap image) throws NotFoundException {
    if (readers != null) {
      for (Reader reader : readers) {
        if (Thread.currentThread().isInterrupted()) {
          throw NotFoundException.getNotFoundInstance();
        }
        try {
          return reader.decode(image, hints);
        } catch (ReaderException re) {
          // continue
        }
      }
      if (hints != null && hints.containsKey(DecodeHintType.ALSO_INVERTED)) {
        // Calling all readers again with inverted image
        image.getBlackMatrix().flip();
        for (Reader reader : readers) {
          if (Thread.currentThread().isInterrupted()) {
            throw NotFoundException.getNotFoundInstance();
          }
          try {
            return reader.decode(image, hints);
          } catch (ReaderException re) {
            // continue
          }
        }
      }
    }
    throw NotFoundException.getNotFoundInstance();
  }

}

```
- ```decode(BinaryBitmap image)``` - не принимает hint'ы для декодирования изображения и выполняет декодирование с помощью всех декодеров, настроенных по умолчанию.
- ```decodeWithState(BinaryBitmap image)``` - метод используется для избежания повторной инициализации декодеров каждый раз, когда вызывается метод декодирования.
- ```decode(BinaryBitmap image, Map<DecodeHintType,?> hints)``` - принимает hint'ы для определения соответствующего декодера.
- ```decodeInternal(BinaryBitmap image)``` - выполнение декодирования изображения декодерами, находящиеся в коллекции readers.
- ```reset()``` - сброс использованных декодеров.

### Описание метода ```setHints(Map<DecodeHintType,?> hints)```:
Метод для установки hint'ов (где hint'ы можно определить самим, если мы хотим читать только определенные типы штрихкодов. Для этого необходимо передать аргументом в классе ```detectedCodeBar``` набор необходимов hint'ов в метод ```decode(bitmap, HINTS)```), для определения, какой из декодеров должен быть использован. 
```java
public class DetectionCodeBar {
    public static void detectedCodeBar() {
    ...
                LuminanceSource source = new BufferedImageLuminanceSource(image);
                BinaryBitmap bitmap = new BinaryBitmap(new HybridBinarizer(source));
                    
                // hints для ограничения только QR-кодами
                EnumMap<DecodeHintType, Object> hints = new EnumMap<>(DecodeHintType.class);
                List<BarcodeFormat> formats = new ArrayList<>();
                formats.add(BarcodeFormat.QR_CODE);  // добавляем QR-код в List
                hints.put(DecodeHintType.POSSIBLE_FORMATS, formats);  // передаем List

                Result result = new MultiFormatReader().decode(bitmap, hints);
    ...
    }
}
```

В случае, если ни один из hint'ов не был установлен, то декодирование в ```decodeInternal(BinaryBitmap image)``` выполняют все декодеры.

```java
  public Result decode(BinaryBitmap image, Map<DecodeHintType,?> hints) throws NotFoundException {
    setHints(hints);
    return decodeInternal(image);
  }


  public void setHints(Map<DecodeHintType,?> hints) {
    this.hints = hints;

    boolean tryHarder = hints != null && hints.containsKey(DecodeHintType.TRY_HARDER);
    @SuppressWarnings("unchecked")
    Collection<BarcodeFormat> formats =
        hints == null ? null : (Collection<BarcodeFormat>) hints.get(DecodeHintType.POSSIBLE_FORMATS);
    Collection<Reader> readers = new ArrayList<>();
    if (formats != null) {
      boolean addOneDReader =
          formats.contains(BarcodeFormat.UPC_A) ||
          formats.contains(BarcodeFormat.UPC_E) ||
          formats.contains(BarcodeFormat.EAN_13) ||
          formats.contains(BarcodeFormat.EAN_8) ||
          formats.contains(BarcodeFormat.CODABAR) ||
          formats.contains(BarcodeFormat.CODE_39) ||
          formats.contains(BarcodeFormat.CODE_93) ||
          formats.contains(BarcodeFormat.CODE_128) ||
          formats.contains(BarcodeFormat.ITF) ||
          formats.contains(BarcodeFormat.RSS_14) ||
          formats.contains(BarcodeFormat.RSS_EXPANDED);
      // Put 1D readers upfront in "normal" mode
      if (addOneDReader && !tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }
      if (formats.contains(BarcodeFormat.QR_CODE)) {
        readers.add(new QRCodeReader());
      }
      if (formats.contains(BarcodeFormat.DATA_MATRIX)) {
        readers.add(new DataMatrixReader());
      }
      if (formats.contains(BarcodeFormat.AZTEC)) {
        readers.add(new AztecReader());
      }
      if (formats.contains(BarcodeFormat.PDF_417)) {
        readers.add(new PDF417Reader());
      }
      if (formats.contains(BarcodeFormat.MAXICODE)) {
        readers.add(new MaxiCodeReader());
      }
      // At end in "try harder" mode
      if (addOneDReader && tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }
    }
    if (readers.isEmpty()) {
      if (!tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }

      readers.add(new QRCodeReader());
      readers.add(new DataMatrixReader());
      readers.add(new AztecReader());
      readers.add(new PDF417Reader());
      readers.add(new MaxiCodeReader());

      if (tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }
    }
    this.readers = readers.toArray(EMPTY_READER_ARRAY);
  }
```
![image](https://github.com/user-attachments/assets/8096f03d-caef-4c08-bb0e-daad6d6b1745) ![image](https://github.com/user-attachments/assets/4581c04f-3416-4cf1-bf8a-2026fd762f07)


- В данном случае, если мы рассматриваем данные штрихкоды, то декодер будет определен в классе ```MultiFormatOneDReader()``` для соответстующего вида штрихкода.
```java
 // At end in "try harder" mode
      if (addOneDReader && tryHarder) {
        readers.add(new MultiFormatOneDReader(hints));
      }
```

### Описание ```MultiFormatOneDReader```:
```java
  public MultiFormatOneDReader(Map<DecodeHintType,?> hints) {
    @SuppressWarnings("unchecked")
    Collection<BarcodeFormat> possibleFormats = hints == null ? null :
        (Collection<BarcodeFormat>) hints.get(DecodeHintType.POSSIBLE_FORMATS);
    boolean useCode39CheckDigit = hints != null &&
        hints.get(DecodeHintType.ASSUME_CODE_39_CHECK_DIGIT) != null;
    Collection<OneDReader> readers = new ArrayList<>();
    if (possibleFormats != null) {
      if (possibleFormats.contains(BarcodeFormat.EAN_13) ||
          possibleFormats.contains(BarcodeFormat.UPC_A) ||
          possibleFormats.contains(BarcodeFormat.EAN_8) ||
          possibleFormats.contains(BarcodeFormat.UPC_E)) {
        readers.add(new MultiFormatUPCEANReader(hints));
      }
      if (possibleFormats.contains(BarcodeFormat.CODE_39)) {
        readers.add(new Code39Reader(useCode39CheckDigit));
      }
      if (possibleFormats.contains(BarcodeFormat.CODE_93)) {
        readers.add(new Code93Reader());
      }
      if (possibleFormats.contains(BarcodeFormat.CODE_128)) {
        readers.add(new Code128Reader());
      }
      if (possibleFormats.contains(BarcodeFormat.ITF)) {
        readers.add(new ITFReader());
      }
      if (possibleFormats.contains(BarcodeFormat.CODABAR)) {
        readers.add(new CodaBarReader());
      }
      if (possibleFormats.contains(BarcodeFormat.RSS_14)) {
        readers.add(new RSS14Reader());
      }
      if (possibleFormats.contains(BarcodeFormat.RSS_EXPANDED)) {
        readers.add(new RSSExpandedReader());
      }
    }
    if (readers.isEmpty()) {
      readers.add(new MultiFormatUPCEANReader(hints));
      readers.add(new Code39Reader());
      readers.add(new CodaBarReader());
      readers.add(new Code93Reader());
      readers.add(new Code128Reader());
      readers.add(new ITFReader());
      readers.add(new RSS14Reader());
      readers.add(new RSSExpandedReader());
    }
    this.readers = readers.toArray(EMPTY_ONED_ARRAY);
  }
```
По факту является тем же самым что и класс, только для 1D штрихкодов ```MultiFormatReader```
# Для каждого типа штрихкода прописан свой декодер (reader). Рассмотрим 1D штрихкоды представленные выше:
### Описание reader'а для CODE 39 ```Code39Reader```:
```java
public final class Code39Reader extends OneDReader {

  static final String ALPHABET_STRING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%";

  /**
   * These represent the encodings of characters, as patterns of wide and narrow bars.
   * The 9 least-significant bits of each int correspond to the pattern of wide and narrow,
   * with 1s representing "wide" and 0s representing narrow.
   */
  static final int[] CHARACTER_ENCODINGS = {
      0x034, 0x121, 0x061, 0x160, 0x031, 0x130, 0x070, 0x025, 0x124, 0x064, // 0-9
      0x109, 0x049, 0x148, 0x019, 0x118, 0x058, 0x00D, 0x10C, 0x04C, 0x01C, // A-J
      0x103, 0x043, 0x142, 0x013, 0x112, 0x052, 0x007, 0x106, 0x046, 0x016, // K-T
      0x181, 0x0C1, 0x1C0, 0x091, 0x190, 0x0D0, 0x085, 0x184, 0x0C4, 0x0A8, // U-$
      0x0A2, 0x08A, 0x02A // /-%
  };

  static final int ASTERISK_ENCODING = 0x094;

  private final boolean usingCheckDigit;
  private final boolean extendedMode;
  private final StringBuilder decodeRowResult;
  private final int[] counters;

  /**
   * Creates a reader that assumes all encoded data is data, and does not treat the final
   * character as a check digit. It will not decoded "extended Code 39" sequences.
   */
  public Code39Reader() {
    this(false);
  }

  /**
   * Creates a reader that can be configured to check the last character as a check digit.
   * It will not decoded "extended Code 39" sequences.
   *
   * @param usingCheckDigit if true, treat the last data character as a check digit, not
   * data, and verify that the checksum passes.
   */
  public Code39Reader(boolean usingCheckDigit) {
    this(usingCheckDigit, false);
  }

  /**
   * Creates a reader that can be configured to check the last character as a check digit,
   * or optionally attempt to decode "extended Code 39" sequences that are used to encode
   * the full ASCII character set.
   *
   * @param usingCheckDigit if true, treat the last data character as a check digit, not
   * data, and verify that the checksum passes.
   * @param extendedMode if true, will attempt to decode extended Code 39 sequences in the
   * text.
   */
  public Code39Reader(boolean usingCheckDigit, boolean extendedMode) {
    this.usingCheckDigit = usingCheckDigit;
    this.extendedMode = extendedMode;
    decodeRowResult = new StringBuilder(20);
    counters = new int[9];
  }

  @Override
  public Result decodeRow(int rowNumber, BitArray row, Map<DecodeHintType,?> hints)
      throws NotFoundException, ChecksumException, FormatException {

    int[] theCounters = counters;
    Arrays.fill(theCounters, 0);
    StringBuilder result = decodeRowResult;
    result.setLength(0);

    int[] start = findAsteriskPattern(row, theCounters);
    // Read off white space
    int nextStart = row.getNextSet(start[1]);
    int end = row.getSize();

    char decodedChar;
    int lastStart;
    do {
      recordPattern(row, nextStart, theCounters);
      int pattern = toNarrowWidePattern(theCounters);
      if (pattern < 0) {
        throw NotFoundException.getNotFoundInstance();
      }
      decodedChar = patternToChar(pattern);
      result.append(decodedChar);
      lastStart = nextStart;
      for (int counter : theCounters) {
        nextStart += counter;
      }
      // Read off white space
      nextStart = row.getNextSet(nextStart);
    } while (decodedChar != '*');
    result.setLength(result.length() - 1); // remove asterisk

    // Look for whitespace after pattern:
    int lastPatternSize = 0;
    for (int counter : theCounters) {
      lastPatternSize += counter;
    }
    int whiteSpaceAfterEnd = nextStart - lastStart - lastPatternSize;
    // If 50% of last pattern size, following last pattern, is not whitespace, fail
    // (but if it's whitespace to the very end of the image, that's OK)
    if (nextStart != end && (whiteSpaceAfterEnd * 2) < lastPatternSize) {
      throw NotFoundException.getNotFoundInstance();
    }

    if (usingCheckDigit) {
      int max = result.length() - 1;
      int total = 0;
      for (int i = 0; i < max; i++) {
        total += ALPHABET_STRING.indexOf(decodeRowResult.charAt(i));
      }
      if (result.charAt(max) != ALPHABET_STRING.charAt(total % 43)) {
        throw ChecksumException.getChecksumInstance();
      }
      result.setLength(max);
    }

    if (result.length() == 0) {
      // false positive
      throw NotFoundException.getNotFoundInstance();
    }

    String resultString;
    if (extendedMode) {
      resultString = decodeExtended(result);
    } else {
      resultString = result.toString();
    }

    float left = (start[1] + start[0]) / 2.0f;
    float right = lastStart + lastPatternSize / 2.0f;

    Result resultObject = new Result(
        resultString,
        null,
        new ResultPoint[]{
            new ResultPoint(left, rowNumber),
            new ResultPoint(right, rowNumber)},
        BarcodeFormat.CODE_39);
    resultObject.putMetadata(ResultMetadataType.SYMBOLOGY_IDENTIFIER, "]A0");
    return resultObject;
  }

  private static int[] findAsteriskPattern(BitArray row, int[] counters) throws NotFoundException {
    int width = row.getSize();
    int rowOffset = row.getNextSet(0);

    int counterPosition = 0;
    int patternStart = rowOffset;
    boolean isWhite = false;
    int patternLength = counters.length;

    for (int i = rowOffset; i < width; i++) {
      if (row.get(i) != isWhite) {
        counters[counterPosition]++;
      } else {
        if (counterPosition == patternLength - 1) {
          // Look for whitespace before start pattern, >= 50% of width of start pattern
          if (toNarrowWidePattern(counters) == ASTERISK_ENCODING &&
              row.isRange(Math.max(0, patternStart - ((i - patternStart) / 2)), patternStart, false)) {
            return new int[]{patternStart, i};
          }
          patternStart += counters[0] + counters[1];
          System.arraycopy(counters, 2, counters, 0, counterPosition - 1);
          counters[counterPosition - 1] = 0;
          counters[counterPosition] = 0;
          counterPosition--;
        } else {
          counterPosition++;
        }
        counters[counterPosition] = 1;
        isWhite = !isWhite;
      }
    }
    throw NotFoundException.getNotFoundInstance();
  }

  // For efficiency, returns -1 on failure. Not throwing here saved as many as 700 exceptions
  // per image when using some of our blackbox images.
  private static int toNarrowWidePattern(int[] counters) {
    int numCounters = counters.length;
    int maxNarrowCounter = 0;
    int wideCounters;
    do {
      int minCounter = Integer.MAX_VALUE;
      for (int counter : counters) {
        if (counter < minCounter && counter > maxNarrowCounter) {
          minCounter = counter;
        }
      }
      maxNarrowCounter = minCounter;
      wideCounters = 0;
      int totalWideCountersWidth = 0;
      int pattern = 0;
      for (int i = 0; i < numCounters; i++) {
        int counter = counters[i];
        if (counter > maxNarrowCounter) {
          pattern |= 1 << (numCounters - 1 - i);
          wideCounters++;
          totalWideCountersWidth += counter;
        }
      }
      if (wideCounters == 3) {
        // Found 3 wide counters, but are they close enough in width?
        // We can perform a cheap, conservative check to see if any individual
        // counter is more than 1.5 times the average:
        for (int i = 0; i < numCounters && wideCounters > 0; i++) {
          int counter = counters[i];
          if (counter > maxNarrowCounter) {
            wideCounters--;
            // totalWideCountersWidth = 3 * average, so this checks if counter >= 3/2 * average
            if ((counter * 2) >= totalWideCountersWidth) {
              return -1;
            }
          }
        }
        return pattern;
      }
    } while (wideCounters > 3);
    return -1;
  }

  private static char patternToChar(int pattern) throws NotFoundException {
    for (int i = 0; i < CHARACTER_ENCODINGS.length; i++) {
      if (CHARACTER_ENCODINGS[i] == pattern) {
        return ALPHABET_STRING.charAt(i);
      }
    }
    if (pattern == ASTERISK_ENCODING) {
      return '*';
    }
    throw NotFoundException.getNotFoundInstance();
  }

  private static String decodeExtended(CharSequence encoded) throws FormatException {
    int length = encoded.length();
    StringBuilder decoded = new StringBuilder(length);
    for (int i = 0; i < length; i++) {
      char c = encoded.charAt(i);
      if (c == '+' || c == '$' || c == '%' || c == '/') {
        char next = encoded.charAt(i + 1);
        char decodedChar = '\0';
        switch (c) {
          case '+':
            // +A to +Z map to a to z
            if (next >= 'A' && next <= 'Z') {
              decodedChar = (char) (next + 32);
            } else {
              throw FormatException.getFormatInstance();
            }
            break;
          case '$':
            // $A to $Z map to control codes SH to SB
            if (next >= 'A' && next <= 'Z') {
              decodedChar = (char) (next - 64);
            } else {
              throw FormatException.getFormatInstance();
            }
            break;
          case '%':
            // %A to %E map to control codes ESC to US
            if (next >= 'A' && next <= 'E') {
              decodedChar = (char) (next - 38);
            } else if (next >= 'F' && next <= 'J') {
              decodedChar = (char) (next - 11);
            } else if (next >= 'K' && next <= 'O') {
              decodedChar = (char) (next + 16);
            } else if (next >= 'P' && next <= 'T') {
              decodedChar = (char) (next + 43);
            } else if (next == 'U') {
              decodedChar = (char) 0;
            } else if (next == 'V') {
              decodedChar = '@';
            } else if (next == 'W') {
              decodedChar = '`';
            } else if (next == 'X' || next == 'Y' || next == 'Z') {
              decodedChar = (char) 127;
            } else {
              throw FormatException.getFormatInstance();
            }
            break;
          case '/':
            // /A to /O map to ! to , and /Z maps to :
            if (next >= 'A' && next <= 'O') {
              decodedChar = (char) (next - 32);
            } else if (next == 'Z') {
              decodedChar = ':';
            } else {
              throw FormatException.getFormatInstance();
            }
            break;
        }
        decoded.append(decodedChar);
        // bump up i again since we read two characters
        i++;
      } else {
        decoded.append(c);
      }
    }
    return decoded.toString();
  }

}
```
- ```decodeRow``` - распознает строку штрих-кода BitArray и преобразует её в текстовую строку. Основной метод.
- ```findAsteriskPattern``` - ищет начальный символ * в строке штрих-кода. По факту ищет начало штрих кода по своему особенному признаку
- ```toNarrowWidePattern```- преобразует массив ширин полос в числовой узор.
- ```patternToChar``` - сопоставляет числовой узор символа с соответствующим символом в ```CHARACTER_ENCODINGS```.
- ```decodeExtended``` - декодирует строки расширенного Code 39 (использует специальные символы $, /, %, +).
- ```recordPattern``` - записывает ширину черных и белых полос из строки BitArray в массив counters.
  
### Описание reader'а для CODE 93 ```Code93Reader```:
```java
public final class Code93Reader extends OneDReader {

  // Note that 'abcd' are dummy characters in place of control characters.
  static final String ALPHABET_STRING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%abcd*";
  private static final char[] ALPHABET = ALPHABET_STRING.toCharArray();

  /**
   * These represent the encodings of characters, as patterns of wide and narrow bars.
   * The 9 least-significant bits of each int correspond to the pattern of wide and narrow.
   */
  static final int[] CHARACTER_ENCODINGS = {
      0x114, 0x148, 0x144, 0x142, 0x128, 0x124, 0x122, 0x150, 0x112, 0x10A, // 0-9
      0x1A8, 0x1A4, 0x1A2, 0x194, 0x192, 0x18A, 0x168, 0x164, 0x162, 0x134, // A-J
      0x11A, 0x158, 0x14C, 0x146, 0x12C, 0x116, 0x1B4, 0x1B2, 0x1AC, 0x1A6, // K-T
      0x196, 0x19A, 0x16C, 0x166, 0x136, 0x13A, // U-Z
      0x12E, 0x1D4, 0x1D2, 0x1CA, 0x16E, 0x176, 0x1AE, // - - %
      0x126, 0x1DA, 0x1D6, 0x132, 0x15E, // Control chars? $-*
  };
  static final int ASTERISK_ENCODING = CHARACTER_ENCODINGS[47];

  private final StringBuilder decodeRowResult;
  private final int[] counters;

  public Code93Reader() {
    decodeRowResult = new StringBuilder(20);
    counters = new int[6];
  }

  @Override
  public Result decodeRow(int rowNumber, BitArray row, Map<DecodeHintType,?> hints)
      throws NotFoundException, ChecksumException, FormatException {

    int[] start = findAsteriskPattern(row);
    // Read off white space
    int nextStart = row.getNextSet(start[1]);
    int end = row.getSize();

    int[] theCounters = counters;
    Arrays.fill(theCounters, 0);
    StringBuilder result = decodeRowResult;
    result.setLength(0);

    char decodedChar;
    int lastStart;
    do {
      recordPattern(row, nextStart, theCounters);
      int pattern = toPattern(theCounters);
      if (pattern < 0) {
        throw NotFoundException.getNotFoundInstance();
      }
      decodedChar = patternToChar(pattern);
      result.append(decodedChar);
      lastStart = nextStart;
      for (int counter : theCounters) {
        nextStart += counter;
      }
      // Read off white space
      nextStart = row.getNextSet(nextStart);
    } while (decodedChar != '*');
    result.deleteCharAt(result.length() - 1); // remove asterisk

    int lastPatternSize = 0;
    for (int counter : theCounters) {
      lastPatternSize += counter;
    }

    // Should be at least one more black module
    if (nextStart == end || !row.get(nextStart)) {
      throw NotFoundException.getNotFoundInstance();
    }

    if (result.length() < 2) {
      // false positive -- need at least 2 checksum digits
      throw NotFoundException.getNotFoundInstance();
    }

    checkChecksums(result);
    // Remove checksum digits
    result.setLength(result.length() - 2);

    String resultString = decodeExtended(result);

    float left = (start[1] + start[0]) / 2.0f;
    float right = lastStart + lastPatternSize / 2.0f;

    Result resultObject = new Result(
        resultString,
        null,
        new ResultPoint[]{
            new ResultPoint(left, rowNumber),
            new ResultPoint(right, rowNumber)},
        BarcodeFormat.CODE_93);
    resultObject.putMetadata(ResultMetadataType.SYMBOLOGY_IDENTIFIER, "]G0");
    return resultObject;
  }

  private int[] findAsteriskPattern(BitArray row) throws NotFoundException {
    int width = row.getSize();
    int rowOffset = row.getNextSet(0);

    Arrays.fill(counters, 0);
    int[] theCounters = counters;
    int patternStart = rowOffset;
    boolean isWhite = false;
    int patternLength = theCounters.length;

    int counterPosition = 0;
    for (int i = rowOffset; i < width; i++) {
      if (row.get(i) != isWhite) {
        theCounters[counterPosition]++;
      } else {
        if (counterPosition == patternLength - 1) {
          if (toPattern(theCounters) == ASTERISK_ENCODING) {
            return new int[]{patternStart, i};
          }
          patternStart += theCounters[0] + theCounters[1];
          System.arraycopy(theCounters, 2, theCounters, 0, counterPosition - 1);
          theCounters[counterPosition - 1] = 0;
          theCounters[counterPosition] = 0;
          counterPosition--;
        } else {
          counterPosition++;
        }
        theCounters[counterPosition] = 1;
        isWhite = !isWhite;
      }
    }
    throw NotFoundException.getNotFoundInstance();
  }

  private static int toPattern(int[] counters) {
    int sum = 0;
    for (int counter : counters) {
      sum += counter;
    }
    int pattern = 0;
    int max = counters.length;
    for (int i = 0; i < max; i++) {
      int scaled = Math.round(counters[i] * 9.0f / sum);
      if (scaled < 1 || scaled > 4) {
        return -1;
      }
      if ((i & 0x01) == 0) {
        for (int j = 0; j < scaled; j++) {
          pattern = (pattern << 1) | 0x01;
        }
      } else {
        pattern <<= scaled;
      }
    }
    return pattern;
  }

  private static char patternToChar(int pattern) throws NotFoundException {
    for (int i = 0; i < CHARACTER_ENCODINGS.length; i++) {
      if (CHARACTER_ENCODINGS[i] == pattern) {
        return ALPHABET[i];
      }
    }
    throw NotFoundException.getNotFoundInstance();
  }

  private static String decodeExtended(CharSequence encoded) throws FormatException {
    int length = encoded.length();
    StringBuilder decoded = new StringBuilder(length);
    for (int i = 0; i < length; i++) {
      char c = encoded.charAt(i);
      if (c >= 'a' && c <= 'd') {
        if (i >= length - 1) {
          throw FormatException.getFormatInstance();
        }
        char next = encoded.charAt(i + 1);
        char decodedChar = '\0';
        switch (c) {
          case 'd':
            // +A to +Z map to a to z
            if (next >= 'A' && next <= 'Z') {
              decodedChar = (char) (next + 32);
            } else {
              throw FormatException.getFormatInstance();
            }
            break;
          case 'a':
            // $A to $Z map to control codes SH to SB
            if (next >= 'A' && next <= 'Z') {
              decodedChar = (char) (next - 64);
            } else {
              throw FormatException.getFormatInstance();
            }
            break;
          case 'b':
            if (next >= 'A' && next <= 'E') {
              // %A to %E map to control codes ESC to USep
              decodedChar = (char) (next - 38);
            } else if (next >= 'F' && next <= 'J') {
              // %F to %J map to ; < = > ?
              decodedChar = (char) (next - 11);
            } else if (next >= 'K' && next <= 'O') {
              // %K to %O map to [ \ ] ^ _
              decodedChar = (char) (next + 16);
            } else if (next >= 'P' && next <= 'T') {
              // %P to %T map to { | } ~ DEL
              decodedChar = (char) (next + 43);
            } else if (next == 'U') {
              // %U map to NUL
              decodedChar = '\0';
            } else if (next == 'V') {
              // %V map to @
              decodedChar = '@';
            } else if (next == 'W') {
              // %W map to `
              decodedChar = '`';
            } else if (next >= 'X' && next <= 'Z') {
              // %X to %Z all map to DEL (127)
              decodedChar = 127;
            } else {
              throw FormatException.getFormatInstance();
            }
            break;
          case 'c':
            // /A to /O map to ! to , and /Z maps to :
            if (next >= 'A' && next <= 'O') {
              decodedChar = (char) (next - 32);
            } else if (next == 'Z') {
              decodedChar = ':';
            } else {
              throw FormatException.getFormatInstance();
            }
            break;
        }
        decoded.append(decodedChar);
        // bump up i again since we read two characters
        i++;
      } else {
        decoded.append(c);
      }
    }
    return decoded.toString();
  }

  private static void checkChecksums(CharSequence result) throws ChecksumException {
    int length = result.length();
    checkOneChecksum(result, length - 2, 20);
    checkOneChecksum(result, length - 1, 15);
  }

  private static void checkOneChecksum(CharSequence result, int checkPosition, int weightMax)
      throws ChecksumException {
    int weight = 1;
    int total = 0;
    for (int i = checkPosition - 1; i >= 0; i--) {
      total += weight * ALPHABET_STRING.indexOf(result.charAt(i));
      if (++weight > weightMax) {
        weight = 1;
      }
    }
    if (result.charAt(checkPosition) != ALPHABET[total % 47]) {
      throw ChecksumException.getChecksumInstance();
    }
  }
}
```
- ```decodeRow``` - также является основным методом
- ```findAsteriskPattern``` - поиск начального символа * с использованием массива счетчиков полос. Ищет начало шртих кода.
- ```toPattern``` - преобразует массив полос (широкие и узкие) в числовой шаблон.
- ```patternToChar``` - находит соответствующий символ в массиве CHARACTER_ENCODINGS.
- ```checkChecksums``` - проверяет две контрольные суммы (C и K).
- ```decodeExtended``` - обрабатывает управляющие символы для расширенного набора символов.

### Описание reader'а для CodaBar  ```CodaBarReader```:
```java
public final class CodaBarReader extends OneDReader {

  // These values are critical for determining how permissive the decoding
  // will be. All stripe sizes must be within the window these define, as
  // compared to the average stripe size.
  private static final float MAX_ACCEPTABLE = 2.0f;
  private static final float PADDING = 1.5f;

  private static final String ALPHABET_STRING = "0123456789-$:/.+ABCD";
  static final char[] ALPHABET = ALPHABET_STRING.toCharArray();

  /**
   * These represent the encodings of characters, as patterns of wide and narrow bars. The 7 least-significant bits of
   * each int correspond to the pattern of wide and narrow, with 1s representing "wide" and 0s representing narrow.
   */
  static final int[] CHARACTER_ENCODINGS = {
      0x003, 0x006, 0x009, 0x060, 0x012, 0x042, 0x021, 0x024, 0x030, 0x048, // 0-9
      0x00c, 0x018, 0x045, 0x051, 0x054, 0x015, 0x01A, 0x029, 0x00B, 0x00E, // -$:/.+ABCD
  };

  // minimal number of characters that should be present (including start and stop characters)
  // under normal circumstances this should be set to 3, but can be set higher
  // as a last-ditch attempt to reduce false positives.
  private static final int MIN_CHARACTER_LENGTH = 3;

  // official start and end patterns
  private static final char[] STARTEND_ENCODING = {'A', 'B', 'C', 'D'};
  // some Codabar generator allow the Codabar string to be closed by every
  // character. This will cause lots of false positives!

  // some industries use a checksum standard but this is not part of the original Codabar standard
  // for more information see : http://www.mecsw.com/specs/codabar.html

  // Keep some instance variables to avoid reallocations
  private final StringBuilder decodeRowResult;
  private int[] counters;
  private int counterLength;

  public CodaBarReader() {
    decodeRowResult = new StringBuilder(20);
    counters = new int[80];
    counterLength = 0;
  }

  @Override
  public Result decodeRow(int rowNumber, BitArray row, Map<DecodeHintType,?> hints) throws NotFoundException {

    Arrays.fill(counters, 0);
    setCounters(row);
    int startOffset = findStartPattern();
    int nextStart = startOffset;

    decodeRowResult.setLength(0);
    do {
      int charOffset = toNarrowWidePattern(nextStart);
      if (charOffset == -1) {
        throw NotFoundException.getNotFoundInstance();
      }
      // Hack: We store the position in the alphabet table into a
      // StringBuilder, so that we can access the decoded patterns in
      // validatePattern. We'll translate to the actual characters later.
      decodeRowResult.append((char) charOffset);
      nextStart += 8;
      // Stop as soon as we see the end character.
      if (decodeRowResult.length() > 1 &&
          arrayContains(STARTEND_ENCODING, ALPHABET[charOffset])) {
        break;
      }
    } while (nextStart < counterLength); // no fixed end pattern so keep on reading while data is available

    // Look for whitespace after pattern:
    int trailingWhitespace = counters[nextStart - 1];
    int lastPatternSize = 0;
    for (int i = -8; i < -1; i++) {
      lastPatternSize += counters[nextStart + i];
    }

    // We need to see whitespace equal to 50% of the last pattern size,
    // otherwise this is probably a false positive. The exception is if we are
    // at the end of the row. (I.e. the barcode barely fits.)
    if (nextStart < counterLength && trailingWhitespace < lastPatternSize / 2) {
      throw NotFoundException.getNotFoundInstance();
    }

    validatePattern(startOffset);

    // Translate character table offsets to actual characters.
    for (int i = 0; i < decodeRowResult.length(); i++) {
      decodeRowResult.setCharAt(i, ALPHABET[decodeRowResult.charAt(i)]);
    }
    // Ensure a valid start and end character
    char startchar = decodeRowResult.charAt(0);
    if (!arrayContains(STARTEND_ENCODING, startchar)) {
      throw NotFoundException.getNotFoundInstance();
    }
    char endchar = decodeRowResult.charAt(decodeRowResult.length() - 1);
    if (!arrayContains(STARTEND_ENCODING, endchar)) {
      throw NotFoundException.getNotFoundInstance();
    }

    // remove stop/start characters character and check if a long enough string is contained
    if (decodeRowResult.length() <= MIN_CHARACTER_LENGTH) {
      // Almost surely a false positive ( start + stop + at least 1 character)
      throw NotFoundException.getNotFoundInstance();
    }

    if (hints == null || !hints.containsKey(DecodeHintType.RETURN_CODABAR_START_END)) {
      decodeRowResult.deleteCharAt(decodeRowResult.length() - 1);
      decodeRowResult.deleteCharAt(0);
    }

    int runningCount = 0;
    for (int i = 0; i < startOffset; i++) {
      runningCount += counters[i];
    }
    float left = runningCount;
    for (int i = startOffset; i < nextStart - 1; i++) {
      runningCount += counters[i];
    }
    float right = runningCount;

    Result result = new Result(
        decodeRowResult.toString(),
        null,
        new ResultPoint[]{
            new ResultPoint(left, rowNumber),
            new ResultPoint(right, rowNumber)},
        BarcodeFormat.CODABAR);
    result.putMetadata(ResultMetadataType.SYMBOLOGY_IDENTIFIER, "]F0");
    return result;
  }

  private void validatePattern(int start) throws NotFoundException {
    // First, sum up the total size of our four categories of stripe sizes;
    int[] sizes = {0, 0, 0, 0};
    int[] counts = {0, 0, 0, 0};
    int end = decodeRowResult.length() - 1;

    // We break out of this loop in the middle, in order to handle
    // inter-character spaces properly.
    int pos = start;
    for (int i = 0; i <= end; i++) {
      int pattern = CHARACTER_ENCODINGS[decodeRowResult.charAt(i)];
      for (int j = 6; j >= 0; j--) {
        // Even j = bars, while odd j = spaces. Categories 2 and 3 are for
        // long stripes, while 0 and 1 are for short stripes.
        int category = (j & 1) + (pattern & 1) * 2;
        sizes[category] += counters[pos + j];
        counts[category]++;
        pattern >>= 1;
      }
      // We ignore the inter-character space - it could be of any size.
      pos += 8;
    }

    // Calculate our allowable size thresholds using fixed-point math.
    float[] maxes = new float[4];
    float[] mins = new float[4];
    // Define the threshold of acceptability to be the midpoint between the
    // average small stripe and the average large stripe. No stripe lengths
    // should be on the "wrong" side of that line.
    for (int i = 0; i < 2; i++) {
      mins[i] = 0.0f;  // Accept arbitrarily small "short" stripes.
      mins[i + 2] = ((float) sizes[i] / counts[i] + (float) sizes[i + 2] / counts[i + 2]) / 2.0f;
      maxes[i] = mins[i + 2];
      maxes[i + 2] = (sizes[i + 2] * MAX_ACCEPTABLE + PADDING) / counts[i + 2];
    }

    // Now verify that all of the stripes are within the thresholds.
    pos = start;
    for (int i = 0; i <= end; i++) {
      int pattern = CHARACTER_ENCODINGS[decodeRowResult.charAt(i)];
      for (int j = 6; j >= 0; j--) {
        // Even j = bars, while odd j = spaces. Categories 2 and 3 are for
        // long stripes, while 0 and 1 are for short stripes.
        int category = (j & 1) + (pattern & 1) * 2;
        int size = counters[pos + j];
        if (size < mins[category] || size > maxes[category]) {
          throw NotFoundException.getNotFoundInstance();
        }
        pattern >>= 1;
      }
      pos += 8;
    }
  }

  /**
   * Records the size of all runs of white and black pixels, starting with white.
   * This is just like recordPattern, except it records all the counters, and
   * uses our builtin "counters" member for storage.
   * @param row row to count from
   */
  private void setCounters(BitArray row) throws NotFoundException {
    counterLength = 0;
    // Start from the first white bit.
    int i = row.getNextUnset(0);
    int end = row.getSize();
    if (i >= end) {
      throw NotFoundException.getNotFoundInstance();
    }
    boolean isWhite = true;
    int count = 0;
    while (i < end) {
      if (row.get(i) != isWhite) {
        count++;
      } else {
        counterAppend(count);
        count = 1;
        isWhite = !isWhite;
      }
      i++;
    }
    counterAppend(count);
  }

  private void counterAppend(int e) {
    counters[counterLength] = e;
    counterLength++;
    if (counterLength >= counters.length) {
      int[] temp = new int[counterLength * 2];
      System.arraycopy(counters, 0, temp, 0, counterLength);
      counters = temp;
    }
  }

  private int findStartPattern() throws NotFoundException {
    for (int i = 1; i < counterLength; i += 2) {
      int charOffset = toNarrowWidePattern(i);
      if (charOffset != -1 && arrayContains(STARTEND_ENCODING, ALPHABET[charOffset])) {
        // Look for whitespace before start pattern, >= 50% of width of start pattern
        // We make an exception if the whitespace is the first element.
        int patternSize = 0;
        for (int j = i; j < i + 7; j++) {
          patternSize += counters[j];
        }
        if (i == 1 || counters[i - 1] >= patternSize / 2) {
          return i;
        }
      }
    }
    throw NotFoundException.getNotFoundInstance();
  }

  static boolean arrayContains(char[] array, char key) {
    if (array != null) {
      for (char c : array) {
        if (c == key) {
          return true;
        }
      }
    }
    return false;
  }

  // Assumes that counters[position] is a bar.
  private int toNarrowWidePattern(int position) {
    int end = position + 7;
    if (end >= counterLength) {
      return -1;
    }

    int[] theCounters = counters;

    int maxBar = 0;
    int minBar = Integer.MAX_VALUE;
    for (int j = position; j < end; j += 2) {
      int currentCounter = theCounters[j];
      if (currentCounter < minBar) {
        minBar = currentCounter;
      }
      if (currentCounter > maxBar) {
        maxBar = currentCounter;
      }
    }
    int thresholdBar = (minBar + maxBar) / 2;

    int maxSpace = 0;
    int minSpace = Integer.MAX_VALUE;
    for (int j = position + 1; j < end; j += 2) {
      int currentCounter = theCounters[j];
      if (currentCounter < minSpace) {
        minSpace = currentCounter;
      }
      if (currentCounter > maxSpace) {
        maxSpace = currentCounter;
      }
    }
    int thresholdSpace = (minSpace + maxSpace) / 2;

    int bitmask = 1 << 7;
    int pattern = 0;
    for (int i = 0; i < 7; i++) {
      int threshold = (i & 1) == 0 ? thresholdBar : thresholdSpace;
      bitmask >>= 1;
      if (theCounters[position + i] > threshold) {
        pattern |= bitmask;
      }
    }

    for (int i = 0; i < CHARACTER_ENCODINGS.length; i++) {
      if (CHARACTER_ENCODINGS[i] == pattern) {
        return i;
      }
    }
    return -1;
  }
}
```
- ```decodeRow``` - распознает строку штрихкода из массива битов BitArray. Основной метод.
- ```validatePattern``` - проверяет, что все элементы штрихкода соответствуют допустимым размерам полос (узких и широких) на основе заданных порогов.
- ```setCounters``` - записывает длины всех чередующихся черных и белых сегментов в массив counters.
- ```counterAppend``` - добавляет очередной элемент в массив counters, при необходимости увеличивая его размер/
- ```findStartPattern``` - находит и возвращает позицию первого символа начала штрихкода, проверяя его соответствие допустимым шаблонам.
- ```toNarrowWidePattern``` - преобразует текущий участок массива счетчиков в числовой шаблон, определяя узкие и широкие полосы.

# Reader для QR-кода (2D штрихкода), использующийся в классе ```MultiFormatReader```:
```java
public class QRCodeReader implements Reader {

  private static final ResultPoint[] NO_POINTS = new ResultPoint[0];

  private final Decoder decoder = new Decoder();

  protected final Decoder getDecoder() {
    return decoder;
  }

  /**
   * Locates and decodes a QR code in an image.
   *
   * @return a String representing the content encoded by the QR code
   * @throws NotFoundException if a QR code cannot be found
   * @throws FormatException if a QR code cannot be decoded
   * @throws ChecksumException if error correction fails
   */
  @Override
  public Result decode(BinaryBitmap image) throws NotFoundException, ChecksumException, FormatException {
    return decode(image, null);
  }

  @Override
  public final Result decode(BinaryBitmap image, Map<DecodeHintType,?> hints)
      throws NotFoundException, ChecksumException, FormatException {
    DecoderResult decoderResult;
    ResultPoint[] points;
    if (hints != null && hints.containsKey(DecodeHintType.PURE_BARCODE)) {
      BitMatrix bits = extractPureBits(image.getBlackMatrix());
      decoderResult = decoder.decode(bits, hints);
      points = NO_POINTS;
    } else {
      DetectorResult detectorResult = new Detector(image.getBlackMatrix()).detect(hints);
      decoderResult = decoder.decode(detectorResult.getBits(), hints);
      points = detectorResult.getPoints();
    }

    // If the code was mirrored: swap the bottom-left and the top-right points.
    if (decoderResult.getOther() instanceof QRCodeDecoderMetaData) {
      ((QRCodeDecoderMetaData) decoderResult.getOther()).applyMirroredCorrection(points);
    }

    Result result = new Result(decoderResult.getText(), decoderResult.getRawBytes(), points, BarcodeFormat.QR_CODE);
    List<byte[]> byteSegments = decoderResult.getByteSegments();
    if (byteSegments != null) {
      result.putMetadata(ResultMetadataType.BYTE_SEGMENTS, byteSegments);
    }
    String ecLevel = decoderResult.getECLevel();
    if (ecLevel != null) {
      result.putMetadata(ResultMetadataType.ERROR_CORRECTION_LEVEL, ecLevel);
    }
    if (decoderResult.hasStructuredAppend()) {
      result.putMetadata(ResultMetadataType.STRUCTURED_APPEND_SEQUENCE,
                         decoderResult.getStructuredAppendSequenceNumber());
      result.putMetadata(ResultMetadataType.STRUCTURED_APPEND_PARITY,
                         decoderResult.getStructuredAppendParity());
    }
    result.putMetadata(ResultMetadataType.ERRORS_CORRECTED, decoderResult.getErrorsCorrected());
    result.putMetadata(ResultMetadataType.SYMBOLOGY_IDENTIFIER, "]Q" + decoderResult.getSymbologyModifier());
    return result;
  }

  @Override
  public void reset() {
    // do nothing
  }

  /**
   * This method detects a code in a "pure" image -- that is, pure monochrome image
   * which contains only an unrotated, unskewed, image of a code, with some white border
   * around it. This is a specialized method that works exceptionally fast in this special
   * case.
   */
  private static BitMatrix extractPureBits(BitMatrix image) throws NotFoundException {

    int[] leftTopBlack = image.getTopLeftOnBit();
    int[] rightBottomBlack = image.getBottomRightOnBit();
    if (leftTopBlack == null || rightBottomBlack == null) {
      throw NotFoundException.getNotFoundInstance();
    }

    float moduleSize = moduleSize(leftTopBlack, image);

    int top = leftTopBlack[1];
    int bottom = rightBottomBlack[1];
    int left = leftTopBlack[0];
    int right = rightBottomBlack[0];

    // Sanity check!
    if (left >= right || top >= bottom) {
      throw NotFoundException.getNotFoundInstance();
    }

    if (bottom - top != right - left) {
      // Special case, where bottom-right module wasn't black so we found something else in the last row
      // Assume it's a square, so use height as the width
      right = left + (bottom - top);
      if (right >= image.getWidth()) {
        // Abort if that would not make sense -- off image
        throw NotFoundException.getNotFoundInstance();
      }
    }

    int matrixWidth = Math.round((right - left + 1) / moduleSize);
    int matrixHeight = Math.round((bottom - top + 1) / moduleSize);
    if (matrixWidth <= 0 || matrixHeight <= 0) {
      throw NotFoundException.getNotFoundInstance();
    }
    if (matrixHeight != matrixWidth) {
      // Only possibly decode square regions
      throw NotFoundException.getNotFoundInstance();
    }

    // Push in the "border" by half the module width so that we start
    // sampling in the middle of the module. Just in case the image is a
    // little off, this will help recover.
    int nudge = (int) (moduleSize / 2.0f);
    top += nudge;
    left += nudge;

    // But careful that this does not sample off the edge
    // "right" is the farthest-right valid pixel location -- right+1 is not necessarily
    // This is positive by how much the inner x loop below would be too large
    int nudgedTooFarRight = left + (int) ((matrixWidth - 1) * moduleSize) - right;
    if (nudgedTooFarRight > 0) {
      if (nudgedTooFarRight > nudge) {
        // Neither way fits; abort
        throw NotFoundException.getNotFoundInstance();
      }
      left -= nudgedTooFarRight;
    }
    // See logic above
    int nudgedTooFarDown = top + (int) ((matrixHeight - 1) * moduleSize) - bottom;
    if (nudgedTooFarDown > 0) {
      if (nudgedTooFarDown > nudge) {
        // Neither way fits; abort
        throw NotFoundException.getNotFoundInstance();
      }
      top -= nudgedTooFarDown;
    }

    // Now just read off the bits
    BitMatrix bits = new BitMatrix(matrixWidth, matrixHeight);
    for (int y = 0; y < matrixHeight; y++) {
      int iOffset = top + (int) (y * moduleSize);
      for (int x = 0; x < matrixWidth; x++) {
        if (image.get(left + (int) (x * moduleSize), iOffset)) {
          bits.set(x, y);
        }
      }
    }
    return bits;
  }

  private static float moduleSize(int[] leftTopBlack, BitMatrix image) throws NotFoundException {
    int height = image.getHeight();
    int width = image.getWidth();
    int x = leftTopBlack[0];
    int y = leftTopBlack[1];
    boolean inBlack = true;
    int transitions = 0;
    while (x < width && y < height) {
      if (inBlack != image.get(x, y)) {
        if (++transitions == 5) {
          break;
        }
        inBlack = !inBlack;
      }
      x++;
      y++;
    }
    if (x == width || y == height) {
      throw NotFoundException.getNotFoundInstance();
    }
    return (x - leftTopBlack[0]) / 7.0f;
  }

}
```
- ```decode(BinaryBitmap image, Map<DecodeHintType,?> hints)``` - если в hints указан флаг ```PURE_BARCODE```, вызывается метод ```extractPureBits``` для обработки чистых изображений.
В противном случае используется ```Detector``` для поиска QR-кода в изображении, который анализирует изображение, находит координаты QR-кода, искажения, углы и выравнивает его.

```java
else {
      DetectorResult detectorResult = new Detector(image.getBlackMatrix()).detect(hints);
      decoderResult = decoder.decode(detectorResult.getBits(), hints);
      points = detectorResult.getPoints();
    }
```

### Описание класса ```Detector```:
```java
/*
 * Copyright 2007 ZXing authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.zxing.qrcode.detector;

import com.google.zxing.DecodeHintType;
import com.google.zxing.FormatException;
import com.google.zxing.NotFoundException;
import com.google.zxing.ResultPoint;
import com.google.zxing.ResultPointCallback;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.common.DetectorResult;
import com.google.zxing.common.GridSampler;
import com.google.zxing.common.PerspectiveTransform;
import com.google.zxing.common.detector.MathUtils;
import com.google.zxing.qrcode.decoder.Version;

import java.util.Map;

/**
 * <p>Encapsulates logic that can detect a QR Code in an image, even if the QR Code
 * is rotated or skewed, or partially obscured.</p>
 *
 * @author Sean Owen
 */
public class Detector {

  private final BitMatrix image;
  private ResultPointCallback resultPointCallback;

  public Detector(BitMatrix image) {
    this.image = image;
  }

  protected final BitMatrix getImage() {
    return image;
  }

  protected final ResultPointCallback getResultPointCallback() {
    return resultPointCallback;
  }

  /**
   * <p>Detects a QR Code in an image.</p>
   *
   * @return {@link DetectorResult} encapsulating results of detecting a QR Code
   * @throws NotFoundException if QR Code cannot be found
   * @throws FormatException if a QR Code cannot be decoded
   */
  public DetectorResult detect() throws NotFoundException, FormatException {
    return detect(null);
  }

  /**
   * <p>Detects a QR Code in an image.</p>
   *
   * @param hints optional hints to detector
   * @return {@link DetectorResult} encapsulating results of detecting a QR Code
   * @throws NotFoundException if QR Code cannot be found
   * @throws FormatException if a QR Code cannot be decoded
   */
  public final DetectorResult detect(Map<DecodeHintType,?> hints) throws NotFoundException, FormatException {

    resultPointCallback = hints == null ? null :
        (ResultPointCallback) hints.get(DecodeHintType.NEED_RESULT_POINT_CALLBACK);

    FinderPatternFinder finder = new FinderPatternFinder(image, resultPointCallback);
    FinderPatternInfo info = finder.find(hints);

    return processFinderPatternInfo(info);
  }

  protected final DetectorResult processFinderPatternInfo(FinderPatternInfo info)
      throws NotFoundException, FormatException {

    FinderPattern topLeft = info.getTopLeft();
    FinderPattern topRight = info.getTopRight();
    FinderPattern bottomLeft = info.getBottomLeft();

    float moduleSize = calculateModuleSize(topLeft, topRight, bottomLeft);
    if (moduleSize < 1.0f) {
      throw NotFoundException.getNotFoundInstance();
    }
    int dimension = computeDimension(topLeft, topRight, bottomLeft, moduleSize);
    Version provisionalVersion = Version.getProvisionalVersionForDimension(dimension);
    int modulesBetweenFPCenters = provisionalVersion.getDimensionForVersion() - 7;

    AlignmentPattern alignmentPattern = null;
    // Anything above version 1 has an alignment pattern
    if (provisionalVersion.getAlignmentPatternCenters().length > 0) {

      // Guess where a "bottom right" finder pattern would have been
      float bottomRightX = topRight.getX() - topLeft.getX() + bottomLeft.getX();
      float bottomRightY = topRight.getY() - topLeft.getY() + bottomLeft.getY();

      // Estimate that alignment pattern is closer by 3 modules
      // from "bottom right" to known top left location
      float correctionToTopLeft = 1.0f - 3.0f / modulesBetweenFPCenters;
      int estAlignmentX = (int) (topLeft.getX() + correctionToTopLeft * (bottomRightX - topLeft.getX()));
      int estAlignmentY = (int) (topLeft.getY() + correctionToTopLeft * (bottomRightY - topLeft.getY()));

      // Kind of arbitrary -- expand search radius before giving up
      for (int i = 4; i <= 16; i <<= 1) {
        try {
          alignmentPattern = findAlignmentInRegion(moduleSize,
              estAlignmentX,
              estAlignmentY,
              i);
          break;
        } catch (NotFoundException re) {
          // try next round
        }
      }
      // If we didn't find alignment pattern... well try anyway without it
    }

    PerspectiveTransform transform =
        createTransform(topLeft, topRight, bottomLeft, alignmentPattern, dimension);

    BitMatrix bits = sampleGrid(image, transform, dimension);

    ResultPoint[] points;
    if (alignmentPattern == null) {
      points = new ResultPoint[]{bottomLeft, topLeft, topRight};
    } else {
      points = new ResultPoint[]{bottomLeft, topLeft, topRight, alignmentPattern};
    }
    return new DetectorResult(bits, points);
  }

  private static PerspectiveTransform createTransform(ResultPoint topLeft,
                                                      ResultPoint topRight,
                                                      ResultPoint bottomLeft,
                                                      ResultPoint alignmentPattern,
                                                      int dimension) {
    float dimMinusThree = dimension - 3.5f;
    float bottomRightX;
    float bottomRightY;
    float sourceBottomRightX;
    float sourceBottomRightY;
    if (alignmentPattern != null) {
      bottomRightX = alignmentPattern.getX();
      bottomRightY = alignmentPattern.getY();
      sourceBottomRightX = dimMinusThree - 3.0f;
      sourceBottomRightY = sourceBottomRightX;
    } else {
      // Don't have an alignment pattern, just make up the bottom-right point
      bottomRightX = (topRight.getX() - topLeft.getX()) + bottomLeft.getX();
      bottomRightY = (topRight.getY() - topLeft.getY()) + bottomLeft.getY();
      sourceBottomRightX = dimMinusThree;
      sourceBottomRightY = dimMinusThree;
    }

    return PerspectiveTransform.quadrilateralToQuadrilateral(
        3.5f,
        3.5f,
        dimMinusThree,
        3.5f,
        sourceBottomRightX,
        sourceBottomRightY,
        3.5f,
        dimMinusThree,
        topLeft.getX(),
        topLeft.getY(),
        topRight.getX(),
        topRight.getY(),
        bottomRightX,
        bottomRightY,
        bottomLeft.getX(),
        bottomLeft.getY());
  }

  private static BitMatrix sampleGrid(BitMatrix image,
                                      PerspectiveTransform transform,
                                      int dimension) throws NotFoundException {

    GridSampler sampler = GridSampler.getInstance();
    return sampler.sampleGrid(image, dimension, dimension, transform);
  }

  /**
   * <p>Computes the dimension (number of modules on a size) of the QR Code based on the position
   * of the finder patterns and estimated module size.</p>
   */
  private static int computeDimension(ResultPoint topLeft,
                                      ResultPoint topRight,
                                      ResultPoint bottomLeft,
                                      float moduleSize) throws NotFoundException {
    int tltrCentersDimension = MathUtils.round(ResultPoint.distance(topLeft, topRight) / moduleSize);
    int tlblCentersDimension = MathUtils.round(ResultPoint.distance(topLeft, bottomLeft) / moduleSize);
    int dimension = ((tltrCentersDimension + tlblCentersDimension) / 2) + 7;
    switch (dimension & 0x03) { // mod 4
      case 0:
        dimension++;
        break;
        // 1? do nothing
      case 2:
        dimension--;
        break;
      case 3:
        throw NotFoundException.getNotFoundInstance();
    }
    return dimension;
  }

  /**
   * <p>Computes an average estimated module size based on estimated derived from the positions
   * of the three finder patterns.</p>
   *
   * @param topLeft detected top-left finder pattern center
   * @param topRight detected top-right finder pattern center
   * @param bottomLeft detected bottom-left finder pattern center
   * @return estimated module size
   */
  protected final float calculateModuleSize(ResultPoint topLeft,
                                            ResultPoint topRight,
                                            ResultPoint bottomLeft) {
    // Take the average
    return (calculateModuleSizeOneWay(topLeft, topRight) +
        calculateModuleSizeOneWay(topLeft, bottomLeft)) / 2.0f;
  }

  /**
   * <p>Estimates module size based on two finder patterns -- it uses
   * {@link #sizeOfBlackWhiteBlackRunBothWays(int, int, int, int)} to figure the
   * width of each, measuring along the axis between their centers.</p>
   */
  private float calculateModuleSizeOneWay(ResultPoint pattern, ResultPoint otherPattern) {
    float moduleSizeEst1 = sizeOfBlackWhiteBlackRunBothWays((int) pattern.getX(),
        (int) pattern.getY(),
        (int) otherPattern.getX(),
        (int) otherPattern.getY());
    float moduleSizeEst2 = sizeOfBlackWhiteBlackRunBothWays((int) otherPattern.getX(),
        (int) otherPattern.getY(),
        (int) pattern.getX(),
        (int) pattern.getY());
    if (Float.isNaN(moduleSizeEst1)) {
      return moduleSizeEst2 / 7.0f;
    }
    if (Float.isNaN(moduleSizeEst2)) {
      return moduleSizeEst1 / 7.0f;
    }
    // Average them, and divide by 7 since we've counted the width of 3 black modules,
    // and 1 white and 1 black module on either side. Ergo, divide sum by 14.
    return (moduleSizeEst1 + moduleSizeEst2) / 14.0f;
  }

  /**
   * See {@link #sizeOfBlackWhiteBlackRun(int, int, int, int)}; computes the total width of
   * a finder pattern by looking for a black-white-black run from the center in the direction
   * of another point (another finder pattern center), and in the opposite direction too.
   */
  private float sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY) {

    float result = sizeOfBlackWhiteBlackRun(fromX, fromY, toX, toY);

    // Now count other way -- don't run off image though of course
    float scale = 1.0f;
    int otherToX = fromX - (toX - fromX);
    if (otherToX < 0) {
      scale = fromX / (float) (fromX - otherToX);
      otherToX = 0;
    } else if (otherToX >= image.getWidth()) {
      scale = (image.getWidth() - 1 - fromX) / (float) (otherToX - fromX);
      otherToX = image.getWidth() - 1;
    }
    int otherToY = (int) (fromY - (toY - fromY) * scale);

    scale = 1.0f;
    if (otherToY < 0) {
      scale = fromY / (float) (fromY - otherToY);
      otherToY = 0;
    } else if (otherToY >= image.getHeight()) {
      scale = (image.getHeight() - 1 - fromY) / (float) (otherToY - fromY);
      otherToY = image.getHeight() - 1;
    }
    otherToX = (int) (fromX + (otherToX - fromX) * scale);

    result += sizeOfBlackWhiteBlackRun(fromX, fromY, otherToX, otherToY);

    // Middle pixel is double-counted this way; subtract 1
    return result - 1.0f;
  }

  /**
   * <p>This method traces a line from a point in the image, in the direction towards another point.
   * It begins in a black region, and keeps going until it finds white, then black, then white again.
   * It reports the distance from the start to this point.</p>
   *
   * <p>This is used when figuring out how wide a finder pattern is, when the finder pattern
   * may be skewed or rotated.</p>
   */
  private float sizeOfBlackWhiteBlackRun(int fromX, int fromY, int toX, int toY) {
    // Mild variant of Bresenham's algorithm;
    // see http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
    boolean steep = Math.abs(toY - fromY) > Math.abs(toX - fromX);
    if (steep) {
      int temp = fromX;
      fromX = fromY;
      fromY = temp;
      temp = toX;
      toX = toY;
      toY = temp;
    }

    int dx = Math.abs(toX - fromX);
    int dy = Math.abs(toY - fromY);
    int error = -dx / 2;
    int xstep = fromX < toX ? 1 : -1;
    int ystep = fromY < toY ? 1 : -1;

    // In black pixels, looking for white, first or second time.
    int state = 0;
    // Loop up until x == toX, but not beyond
    int xLimit = toX + xstep;
    for (int x = fromX, y = fromY; x != xLimit; x += xstep) {
      int realX = steep ? y : x;
      int realY = steep ? x : y;

      // Does current pixel mean we have moved white to black or vice versa?
      // Scanning black in state 0,2 and white in state 1, so if we find the wrong
      // color, advance to next state or end if we are in state 2 already
      if ((state == 1) == image.get(realX, realY)) {
        if (state == 2) {
          return MathUtils.distance(x, y, fromX, fromY);
        }
        state++;
      }

      error += dy;
      if (error > 0) {
        if (y == toY) {
          break;
        }
        y += ystep;
        error -= dx;
      }
    }
    // Found black-white-black; give the benefit of the doubt that the next pixel outside the image
    // is "white" so this last point at (toX+xStep,toY) is the right ending. This is really a
    // small approximation; (toX+xStep,toY+yStep) might be really correct. Ignore this.
    if (state == 2) {
      return MathUtils.distance(toX + xstep, toY, fromX, fromY);
    }
    // else we didn't find even black-white-black; no estimate is really possible
    return Float.NaN;
  }

  /**
   * <p>Attempts to locate an alignment pattern in a limited region of the image, which is
   * guessed to contain it. This method uses {@link AlignmentPattern}.</p>
   *
   * @param overallEstModuleSize estimated module size so far
   * @param estAlignmentX x coordinate of center of area probably containing alignment pattern
   * @param estAlignmentY y coordinate of above
   * @param allowanceFactor number of pixels in all directions to search from the center
   * @return {@link AlignmentPattern} if found, or null otherwise
   * @throws NotFoundException if an unexpected error occurs during detection
   */
  protected final AlignmentPattern findAlignmentInRegion(float overallEstModuleSize,
                                                         int estAlignmentX,
                                                         int estAlignmentY,
                                                         float allowanceFactor)
      throws NotFoundException {
    // Look for an alignment pattern (3 modules in size) around where it
    // should be
    int allowance = (int) (allowanceFactor * overallEstModuleSize);
    int alignmentAreaLeftX = Math.max(0, estAlignmentX - allowance);
    int alignmentAreaRightX = Math.min(image.getWidth() - 1, estAlignmentX + allowance);
    if (alignmentAreaRightX - alignmentAreaLeftX < overallEstModuleSize * 3) {
      throw NotFoundException.getNotFoundInstance();
    }

    int alignmentAreaTopY = Math.max(0, estAlignmentY - allowance);
    int alignmentAreaBottomY = Math.min(image.getHeight() - 1, estAlignmentY + allowance);
    if (alignmentAreaBottomY - alignmentAreaTopY < overallEstModuleSize * 3) {
      throw NotFoundException.getNotFoundInstance();
    }

    AlignmentPatternFinder alignmentFinder =
        new AlignmentPatternFinder(
            image,
            alignmentAreaLeftX,
            alignmentAreaTopY,
            alignmentAreaRightX - alignmentAreaLeftX,
            alignmentAreaBottomY - alignmentAreaTopY,
            overallEstModuleSize,
            resultPointCallback);
    return alignmentFinder.find();
  }

}

```
- ```Detector``` - ищет Alignment Pattern, который помогает выровнять QR-код, особенно если он искажён или имеет перспективные искажения.
- ```createTransform``` - использует PerspectiveTransform.quadrilateralToQuadrilateral, чтобы выровнять QR-код, независимо от его ориентации или искажений.
