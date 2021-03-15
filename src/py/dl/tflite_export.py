import tensorflow as tf

# Convert the model

# saved_model_dir = '/work/jprieto/data/remote/EGower/jprieto/trained/keras_app/VGG19_extract_features_384_avgpool'
# saved_model_dir = '/work/jprieto/data/remote/EGower/jprieto/trained/trachoma_class_input_normals_healthy_sev23_class_features_noflips_normalsvstt_folds/trachoma_class_input_normals_healthy_sev23_class_features_noflips_normalsvstt/trachoma_class_input_normals_healthy_sev23_class_features_noflips_normalsvstt_fold0_train'
saved_model_dir = '/work/jprieto/data/remote/EGower/jprieto/trained/keras_app/ResNet50_extract_features_384_avgpool'


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

with open(saved_model_dir + ".tflite", 'wb') as f:
  f.write(tflite_model)
