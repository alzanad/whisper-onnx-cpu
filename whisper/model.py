import io
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import requests
import psutil
import onnx
from onnx.serialization import ProtoSerializer
import onnxruntime as ort
from whisper.decoding import detect_language as detect_language_function, decode as decode_function
from whisper.utils import onnx_dtype_to_np_dtype_convert
import json
from pathlib import Path


def model_download(name: str, onnx_file_save_path: str = '.') -> bytes:
    """تنزيل ملف نموذج ONNX من الخادم"""
    # الحصول على المجلد الحالي للسكريبت
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # تحميل الإعدادات
    config = load_config_for_model(current_dir)
    models_dir = config.get("models_directory", "whisper-models")
    
    # استخراج اسم النموذج من المسار إذا كان مساراً
    if '/' in name or '\\' in name:
        model_name = os.path.basename(name)
        model_dir = os.path.dirname(name)
        # إذا كان المسار نسبياً، اجعله مطلقاً
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(current_dir, model_dir)
        onnx_file_path = os.path.join(model_dir, f"{model_name}_11.onnx")
    else:
        model_name = name
        if onnx_file_save_path == '.':
            # استخدم مجلد النماذج من الإعدادات
            if not os.path.isabs(models_dir):
                onnx_file_save_path = os.path.join(current_dir, models_dir)
            else:
                onnx_file_save_path = models_dir
        elif not os.path.isabs(onnx_file_save_path):
            onnx_file_save_path = os.path.join(current_dir, onnx_file_save_path)
        onnx_file_path = os.path.join(onnx_file_save_path, f'{name}_11.onnx')
    
    if not os.path.exists(onnx_file_path):
        # إذا لم يجد الملف، أطبع رسالة خطأ وتوقف
        print(f"❌ خطأ: ملف النموذج غير موجود: {onnx_file_path}")
        print(f"📥 يجب تنزيل ملف النموذج ووضعه في المجلد المناسب")
        print(f"🔗 رابط التنزيل: https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/{model_name}_11.onnx")
        exit(1)
    
    # قراءة الملف الموجود
    serializer: ProtoSerializer = onnx._get_serializer(fmt='protobuf')
    onnx_graph: onnx.ModelProto = onnx.load(onnx_file_path)
    onnx_serialized_graph = serializer.serialize_proto(proto=onnx_graph)
    
    return onnx_serialized_graph

def load_config_for_model(current_dir):
    """تحميل ملف الإعدادات للنموذج"""
    # البحث أولاً في المجلد الحالي، ثم في المجلد الأب
    config_path = Path(current_dir) / "config.json"
    if not config_path.exists():
        config_path = Path(current_dir).parent / "config.json"
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # إعدادات افتراضية
    return {
        "models_directory": "whisper-models",
        "audio_directory": "audio",
        "output_directory": "output"
    }

def load_model(name: str):
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model (e.g., "whisper-models/tiny.en")

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """
    # الحصول على المجلد الحالي للسكريبت
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # تحميل الإعدادات
    config = load_config_for_model(current_dir)
    models_dir = config.get("models_directory", "whisper-models")
    
    # استخراج اسم النموذج من المسار أو الاسم
    if '/' in name or '\\' in name:
        # إذا كان مساراً، استخرج اسم النموذج من آخر جزء
        model_name = os.path.basename(name)
        model_dir = os.path.dirname(name)
        # إذا كان المسار نسبياً، اجعله مطلقاً
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(current_dir, model_dir)
        encoder_path = os.path.join(model_dir, f"{model_name}_encoder_11.onnx")
        decoder_path = os.path.join(model_dir, f"{model_name}_decoder_11.onnx")
    else:
        # إذا كان اسماً فقط، ابحث في مجلد النماذج من الإعدادات أولاً
        model_name = name
        
        # البحث في مجلد النماذج من الإعدادات
        if not os.path.isabs(models_dir):
            models_full_path = os.path.join(current_dir, models_dir)
        else:
            models_full_path = models_dir
            
        encoder_path = os.path.join(models_full_path, f"{name}_encoder_11.onnx")
        decoder_path = os.path.join(models_full_path, f"{name}_decoder_11.onnx")
        
        # إذا لم توجد في مجلد الإعدادات، ابحث في المجلد الحالي
        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            encoder_path = os.path.join(current_dir, f"{name}_encoder_11.onnx")
            decoder_path = os.path.join(current_dir, f"{name}_decoder_11.onnx")
    
    # التحقق من وجود ملفي النموذج المطلوبين
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print(f"❌ خطأ: النموذج '{name}' غير موجود")
        print(f"📁 ملفات مطلوبة:")
        print(f"   - {encoder_path}")
        print(f"   - {decoder_path}")
        print(f"📥 يجب تنزيل هذين الملفين ووضعهما في المجلد المناسب")
        print(f"🔗 روابط التنزيل:")
        print(f"   - https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/{model_name}_encoder_11.onnx")
        print(f"   - https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/{model_name}_decoder_11.onnx")
        exit(1)
    
    # تحديد أبعاد النموذج حسب الاسم
    dims_config = _get_model_dimensions(model_name)
    if not dims_config:
        print(f"❌ خطأ: نموذج غير مدعوم '{model_name}'")
        print(f"📋 النماذج المدعومة: {available_models()}")
        exit(1)
    
    dims = ModelDimensions(**dims_config)
    model = Whisper(dims=dims, model_name=model_name)
    return model

def _get_model_dimensions(name: str) -> dict:
    """إرجاع أبعاد النموذج حسب الاسم"""
    model_configs = {
        "tiny": {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4},
        "tiny.en": {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4},
        "base": {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6},
        "base.en": {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6},
        "small": {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12},
        "small.en": {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12},
        "medium": {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24},
        "medium.en": {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    }
    return model_configs.get(name)

def available_models() -> List[str]:
    """Returns the names of available models"""
    return ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium"]

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class OnnxAudioEncoder():
    def __init__(
        self,
        model: str,
    ):
        super().__init__()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.sess = \
            ort.InferenceSession(
                path_or_bytes=model_download(name=f'{model}_encoder'),
                sess_options=sess_options,
                providers=[
                    'CPUExecutionProvider'
                ],
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        mel: np.ndarray
    ) -> np.ndarray:
        result: np.ndarray = \
            self.sess.run(
                output_names=[
                    "output",
                ],
                input_feed={
                    "mel": mel.astype(self.inputs["mel"]),
                }
            )[0]
        return result


class OnnxTextDecoder():
    def __init__(
        self,
        model: str,
    ):
        super().__init__()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.sess = \
            ort.InferenceSession(
                path_or_bytes=model_download(name=f'{model}_decoder'),
                sess_options=sess_options,
                providers=[
                    'CPUExecutionProvider'
                ],
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        x: np.ndarray,
        xa: np.ndarray,
        kv_cache: np.ndarray,
        offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = \
            self.sess.run(
                output_names=[
                    "logits",
                    "output_kv_cache",
                    "cross_attention_qks",
                ],
                input_feed={
                    "tokens": x.astype(self.inputs["tokens"]),
                    "audio_features": xa.astype(self.inputs["audio_features"]),
                    "kv_cache": kv_cache.astype(self.inputs["kv_cache"]),
                    "offset": np.array([offset], dtype=self.inputs["offset"]),
                }
            )
        logits: np.ndarray = outputs[0]
        output_kv_cache: np.ndarray = outputs[1]
        cross_attention_qks: np.ndarray = outputs[2]
        return logits.astype(np.float32), output_kv_cache.astype(np.float32)


class Whisper():
    def __init__(
        self,
        dims: ModelDimensions,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name
        self.dims = dims
        self.encoder = OnnxAudioEncoder(model=model_name)
        self.decoder = OnnxTextDecoder(model=model_name)

    def embed_audio(
        self,
        mel: np.ndarray,
    ):
        return self.encoder(mel)

    def logits(
        self,
        tokens: np.ndarray,
        audio_features: np.ndarray,
    ):
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, audio_features, kv_cache=kv_cache, offset=0)
        return output

    def __call__(
        self,
        mel: np.ndarray,
        tokens: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, self.encoder(mel), kv_cache=kv_cache, offset=0)
        return output

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def new_kv_cache(
        self,
        n_group: int,
        length: int,
    ):
        if self.model_name == "tiny.en" or self.model_name == "tiny":
            size = [8, n_group, length, 384]
        elif self.model_name == "base.en" or self.model_name == "base":
            size = [12, n_group, length, 512]
        elif self.model_name == "small.en" or self.model_name == "small":
            size = [24, n_group, length, 768]
        elif self.model_name == "medium.en" or self.model_name == "medium":
            size = [48, n_group, length, 1024]
        else:
            raise ValueError(f"Unsupported model type: {self.type}")
        return np.zeros(size, dtype=np.float32)

    detect_language = detect_language_function
    # transcribe = transcribe_function
    decode = decode_function
