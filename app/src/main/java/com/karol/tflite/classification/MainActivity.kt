package com.karol.tflite.classification

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import android.view.View
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import com.karol.tflite.classification.databinding.ActivityMainBinding

class MainActivity : ComponentActivity() {

    private lateinit var binding: ActivityMainBinding

    private var selectedBitmap: Bitmap? = null
    private var selectedModel: String = "fp32"
    private var segmenter: SegmentationInterpreter? = null

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            selectedBitmap = bitmap
            binding.imageView.setImageBitmap(bitmap)

            binding.imageOverlay.setImageDrawable(null)
            binding.imageOverlay.visibility = View.GONE
        }
    }

    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        if (bitmap != null) {
            selectedBitmap = bitmap
            binding.imageView.setImageBitmap(bitmap)

            binding.imageOverlay.setImageDrawable(null)
            binding.imageOverlay.visibility = View.GONE
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            openGallery()
        } else {
            Toast.makeText(this, "Permission denied to read your storage.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.imageOverlay.visibility = View.GONE
        binding.imageOverlay.setImageDrawable(null)

        if (segmenter == null) {
            segmenter = SegmentationInterpreter(assets, "unet_pet_simp_float32.tflite", 256)
        }

        binding.modelSelector.setOnCheckedChangeListener { _, checkedId ->
            Toast.makeText(this, "Modelo fixo FP32 para segmentação", Toast.LENGTH_SHORT).show()
            segmenter?.close()
            segmenter = SegmentationInterpreter(assets, "unet_pet_simp_float32.tflite", 256)
        }

        binding.btnCamera.setOnClickListener {
            takePictureLauncher.launch(null)
        }

        binding.btnGallery.setOnClickListener {
            if (android.os.Build.VERSION.SDK_INT <= android.os.Build.VERSION_CODES.S_V2) {
                requestPermissionLauncher.launch(android.Manifest.permission.READ_EXTERNAL_STORAGE)
            } else {
                openGallery()
            }
        }

        binding.btnPredict.setOnClickListener {
            binding.txtResult.text = "Selected model: $selectedModel"
            runPrediction()
        }
    }

    private fun openGallery() {
        pickImageLauncher.launch("image/*")
    }

    private fun runPrediction() {
        val bitmap = selectedBitmap ?: run {
            binding.txtResult.text = "Please select an image first"
            return
        }
        val overlay = segmenter?.overlayOn(bitmap, 0.5f)
        if (overlay != null) {
            binding.imageOverlay.setImageBitmap(overlay)
            binding.imageOverlay.visibility = View.VISIBLE
            binding.txtResult.text = "Segmentação concluída"
        } else {
            binding.txtResult.text = "Falha na segmentação"
            binding.imageOverlay.visibility = View.GONE
        }

    }

    override fun onStop() {
        super.onStop()
    }
}