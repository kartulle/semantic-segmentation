package com.karol.tflite.classification

import android.content.res.AssetManager
import android.graphics.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class SegmentationInterpreter(
    assetManager: AssetManager,
    private val modelPath: String = "unet_pet_simp_float32.tflite",
    private val inputSize: Int = 256,
) {

    private val interpreter: Interpreter = Interpreter(
        mappedModel(assetManager, modelPath),
        Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
        }
    )

    private val inputBuf: ByteBuffer =
        ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())
    private val outputBuf: ByteBuffer =
        ByteBuffer.allocateDirect(4 * inputSize * inputSize).order(ByteOrder.nativeOrder())
    private val tmpPixels = IntArray(inputSize * inputSize)

    private fun mappedModel(assetManager: AssetManager, assetPath: String): MappedByteBuffer {
        val fd = assetManager.openFd(assetPath)
        FileInputStream(fd.fileDescriptor).channel.use { ch ->
            return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.length)
        }
    }

    private fun encodeBitmapNHWC(bitmap: Bitmap) {
        inputBuf.clear()

        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        scaled.getPixels(tmpPixels, 0, inputSize, 0, 0, inputSize, inputSize)

        var i = 0
        while (i < tmpPixels.size) {
            val c = tmpPixels[i]
            inputBuf.putFloat(Color.red(c) / 255f)
            inputBuf.putFloat(Color.green(c) / 255f)
            inputBuf.putFloat(Color.blue(c) / 255f)
            i++
        }
        inputBuf.rewind()
    }
    private fun sigmoidStable(z: Float): Float {
        return if (z >= 0f) {
            val e = exp(-z)
            (1f / (1f + e)).coerceIn(0f, 1f)
        } else {
            val e = exp(z)
            (e / (1f + e)).coerceIn(0f, 1f)
        }
    }

    fun predictMask(bitmap: Bitmap, threshold: Float = 0.5f): Bitmap {
        encodeBitmapNHWC(bitmap)

        outputBuf.clear()
        interpreter.run(inputBuf, outputBuf)
        outputBuf.rewind()

        val mask = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val logit = outputBuf.getFloat()
                val p = sigmoidStable(logit)
                val v = if (p >= threshold) 255 else 0
                mask.setPixel(x, y, Color.argb(160, v, 0, 0))
            }
        }
        return mask
    }

    fun overlayOn(bitmap: Bitmap, threshold: Float = 0.5f): Bitmap {
        val maskSmall = predictMask(bitmap, threshold)
        val mask = Bitmap.createScaledBitmap(
            maskSmall,
            bitmap.width.coerceAtLeast(1),
            bitmap.height.coerceAtLeast(1),
            true
        )
        val out = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawBitmap(bitmap, 0f, 0f, null)
            drawBitmap(mask, 0f, 0f, Paint())
        }
        return out
    }

    fun close() {
        interpreter.close()
    }
}
