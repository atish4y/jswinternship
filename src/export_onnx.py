"""
JSW Steel ERS - ONNX Model Export
===================================
Converts trained XGBoost model to ONNX format for Java/Tomcat deployment.

The ONNX model can be loaded in Java using ONNX Runtime (ORT) Java API,
enabling native JVM inference without Python dependencies in production.

Export pipeline:
    XGBoost (.json) → ONNX Conversion → Validation → .onnx file
"""

import sys
import os
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import xgboost as xgb

# Try to import ONNX tools
try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from feature_engineering import get_feature_columns


def export_to_onnx(model_dir, output_dir=None):
    """
    Load trained XGBoost model and export to ONNX format.
    """
    if not HAS_ONNX:
        print("  ERROR: onnxmltools and/or onnxruntime not installed.")
        print("  Install with: pip install onnxmltools onnxruntime")
        return None

    if output_dir is None:
        output_dir = model_dir

    print("=" * 70)
    print("  JSW Steel ERS - ONNX Model Export")
    print("=" * 70)

    # Load XGBoost model
    model_path = os.path.join(model_dir, 'xgboost_risk_model.json')
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        return None

    print(f"\n  [1/4] Loading XGBoost model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Define input shape
    feature_cols = get_feature_columns()
    n_features = len(feature_cols)
    print(f"  Model expects {n_features} features")

    # Convert to ONNX
    print(f"\n  [2/4] Converting to ONNX format...")
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    onnx_model = convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=12,
        name='ERS_FailurePrediction'
    )

    # Save ONNX model
    onnx_path = os.path.join(output_dir, 'xgboost_risk_model.onnx')
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Saved ONNX model: {onnx_path}")
    print(f"  File size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    # Validate ONNX model
    print(f"\n  [3/4] Validating ONNX model...")
    session = ort.InferenceSession(onnx_path)

    # Get input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    print(f"  Input:  name='{input_name}', shape={input_shape}, type={input_type}")

    for output in session.get_outputs():
        print(f"  Output: name='{output.name}', shape={output.shape}, type={output.type}")

    # Test with dummy input
    print(f"\n  [4/4] Testing inference...")
    dummy_input = np.random.randn(5, n_features).astype(np.float32)
    results = session.run(None, {input_name: dummy_input})

    print(f"  Test predictions (5 samples): {results[0]}")
    print(f"  Test probabilities shape: {np.array(results[1]).shape if len(results) > 1 else 'N/A'}")

    # Save ONNX metadata for Java integration
    onnx_meta = {
        'model_file': 'xgboost_risk_model.onnx',
        'input_name': input_name,
        'input_shape': str(input_shape),
        'input_type': 'float32',
        'feature_columns': feature_cols,
        'n_features': n_features,
        'output_classes': {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'CRITICAL'},
        'java_dependency': 'com.microsoft.onnxruntime:onnxruntime:1.15.0',
        'java_usage': '''
// Java Integration Example:
// Add Maven dependency:
//   <dependency>
//     <groupId>com.microsoft.onnxruntime</groupId>
//     <artifactId>onnxruntime</artifactId>
//     <version>1.15.0</version>
//   </dependency>

// Load model:
//   OrtEnvironment env = OrtEnvironment.getEnvironment();
//   OrtSession session = env.createSession("xgboost_risk_model.onnx");

// Run inference:
//   float[][] input = new float[1][18]; // 18 features
//   OnnxTensor tensor = OnnxTensor.createTensor(env, input);
//   OrtSession.Result result = session.run(
//       Collections.singletonMap("float_input", tensor));
//   long[] predictions = (long[]) result.get(0).getValue();
'''
    }

    meta_path = os.path.join(output_dir, 'onnx_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(onnx_meta, f, indent=2)
    print(f"\n  Saved ONNX metadata: {meta_path}")

    print(f"\n{'=' * 70}")
    print(f"  ONNX export complete!")
    print(f"  Model ready for Java/Tomcat deployment via ONNX Runtime")
    print(f"{'=' * 70}")

    return onnx_path


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "models")
    export_to_onnx(model_dir)


if __name__ == "__main__":
    main()
