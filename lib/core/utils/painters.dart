import 'dart:math';
import 'package:flutter/material.dart';


// --- Custom Painter for Bounding Boxes ---
class DetectionBoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> recognitions;
  final Size originalImageSize; // Actual w/h of the original image
  final Size previewSize; // w/h of the container displaying the scaled image
  final List<Color> boxColors;

  DetectionBoxPainter({
    required this.recognitions,
    required this.originalImageSize,
    required this.previewSize,
    required this.boxColors,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Size here refers to the canvas size, which should match previewSize
    if (originalImageSize == Size.zero || previewSize == Size.zero || originalImageSize.width == 0 || originalImageSize.height == 0) {
      print("DetectionBoxPainter: Invalid sizes, cannot paint. Original: $originalImageSize, Preview: $previewSize");
      return; // Avoid division by zero or painting with invalid dimensions
    }

    // Calculate scaling factors based on BoxFit.contain logic
    final double scaleX = previewSize.width / originalImageSize.width;
    final double scaleY = previewSize.height / originalImageSize.height;
    final double scale = min(scaleX, scaleY); // Use the smaller scale to fit aspect ratio

    // Calculate offsets to center the scaled image within the preview container
    final double offsetX = (previewSize.width - originalImageSize.width * scale) / 2.0;
    final double offsetY = (previewSize.height - originalImageSize.height * scale) / 2.0;

    for (int i = 0; i < recognitions.length; i++) {
      final recognition = recognitions[i];
      // Extract data safely
      final List<double>? rawBox = (recognition['raw_box'] as List<dynamic>?)?.cast<double>();
      final String label = recognition['label'] as String? ?? 'N/A';
      final double confidence = (recognition['confidence'] as num? ?? 0.0).toDouble();

      if (rawBox == null || rawBox.length != 4) {
        print("DetectionBoxPainter: Skipping invalid raw_box: $rawBox");
        continue; // Skip if box data is invalid
      }

      // Assuming raw_box is [ymin, xmin, ymax, xmax] normalized coordinates (0.0 to 1.0)
      // Scale coordinates to original image dimensions first
      final double imgYMin = rawBox[0] * originalImageSize.height;
      final double imgXMin = rawBox[1] * originalImageSize.width;
      final double imgYMax = rawBox[2] * originalImageSize.height;
      final double imgXMax = rawBox[3] * originalImageSize.width;

      // Scale and offset coordinates to match the preview display
      final Rect displayRect = Rect.fromLTRB(
        max(0, imgXMin * scale + offsetX), // Ensure rect starts within canvas
        max(0, imgYMin * scale + offsetY),
        min(previewSize.width, imgXMax * scale + offsetX), // Ensure rect ends within canvas
        min(previewSize.height, imgYMax * scale + offsetY),
      );

      // Skip drawing if the rectangle is invalid or too small
      if (displayRect.width <= 0 || displayRect.height <= 0) {
        print("DetectionBoxPainter: Skipping invalid displayRect: $displayRect");
        continue;
      }


      // Choose color
      final color = boxColors[i % boxColors.length];

      // Draw the bounding box
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5; // Slightly thicker stroke
      canvas.drawRect(displayRect, paint);

      // Prepare text painter for label and confidence
      final textPainter = TextPainter(
        text: TextSpan(
          text: ' $label (${(confidence * 100).toStringAsFixed(0)}%) ', // Add padding
          style: TextStyle(
            color: Colors.white,
            backgroundColor: color.withOpacity(0.85), // Semi-transparent background
            fontSize: 13.0,
            fontWeight: FontWeight.w500,
          ),
        ),
        textDirection: TextDirection.ltr,
      );

      // Layout and paint the text
      textPainter.layout();
      // Position text slightly below the top-left corner of the box
      double textX = displayRect.left + 2; // Small offset from left
      double textY = displayRect.top - textPainter.height - 2; // Position above box

      // Adjust text position if it goes off-screen top
      if (textY < 0) {
        textY = displayRect.top + 2; // Position inside box near top
      }
      // Adjust text position if it goes off-screen left (less common)
      if (textX < 0) {
        textX = 0;
      }
      // Adjust text position if it goes off-screen right
      if (textX + textPainter.width > previewSize.width) {
         textX = previewSize.width - textPainter.width;
         if (textX < 0) textX = 0; // Ensure it doesn't go negative if text is too wide
      }


      textPainter.paint(canvas, Offset(textX, textY));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionBoxPainter oldDelegate) {
    // Repaint only if recognitions or sizes change
    return oldDelegate.recognitions != recognitions ||
           oldDelegate.originalImageSize != originalImageSize ||
           oldDelegate.previewSize != previewSize;
  }
}
