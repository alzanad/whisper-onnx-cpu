#!/usr/bin/env python3
"""
واجهة برمجية مرنة لـ Whisper ONNX
تسهل دمج البرنامج مع أي مشروع دون الاعتماد على ثوابت أو مسارات محددة


الاستخدام:
    from api import WhisperAPI
    
    # إنشاء واجهة مع إعدادات افتراضية
    whisper = WhisperAPI()
    
    # نسخ ملف صوتي
    result = whisper.transcribe("path/to/audio.mp3")
    
    # نسخ مع تخصيص الإعدادات
    result = whisper.transcribe(
        audio_file="audio.wav",
        model="base.en",
        language="en",
        output_formats=["txt", "srt"],
        verbose=True
    )
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict


@dataclass
class TranscriptionResult:
    """نتيجة عملية النسخ"""
    text: str
    language: str
    segments: List[Dict] = None
    success: bool = True
    error_message: str = None
    output_files: Dict[str, str] = None
    
    def to_dict(self) -> Dict:
        """تحويل النتيجة إلى قاموس"""
        return asdict(self)
    
    def to_json(self) -> str:
        """تحويل النتيجة إلى JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class WhisperAPI:
    """
    واجهة برمجية مرنة لـ Whisper ONNX
    تسمح بالنسخ الصوتي مع إعدادات قابلة للتخصيص
    """
    
    def __init__(
        self,
        whisper_dir: str = None,
        models_dir: str = None,
        audio_dir: str = None,
        output_dir: str = None,
        config_file: str = None
    ):
        """
        إنشاء واجهة Whisper API
        
        المعاملات:
            whisper_dir: مجلد برنامج Whisper (افتراضي: نفس مجلد هذا الملف)
            models_dir: مجلد النماذج (افتراضي: whisper-models)
            audio_dir: مجلد الملفات الصوتية (افتراضي: audio)
            output_dir: مجلد الإخراج (افتراضي: output)
            config_file: ملف الإعدادات (افتراضي: config.json)
        """
        # تحديد مجلد البرنامج
        if whisper_dir is None:
            # افتراضياً، مجلد whisper في نفس مستوى هذا الملف
            self.whisper_dir = Path(__file__).parent / "whisper"
        else:
            self.whisper_dir = Path(whisper_dir)
            
        # التحقق من وجود مجلد whisper
        if not self.whisper_dir.exists():
            raise FileNotFoundError(f"مجلد Whisper غير موجود: {self.whisper_dir}")
            
        # ملف transcribe.py
        self.transcribe_script = self.whisper_dir / "transcribe.py"
        if not self.transcribe_script.exists():
            raise FileNotFoundError(f"ملف transcribe.py غير موجود: {self.transcribe_script}")
        
        # المجلدات الافتراضية
        self.models_dir = models_dir or "whisper-models"
        self.audio_dir = audio_dir or "audio"
        self.output_dir = output_dir or "output"
        self.config_file = config_file or "config.json"
        
        # تحميل الإعدادات الافتراضية
        self.default_config = self._load_default_config()
    
    def _load_default_config(self) -> Dict:
        """تحميل الإعدادات الافتراضية"""
        config_path = self.whisper_dir / self.config_file
        if not config_path.exists():
            config_path = self.whisper_dir.parent / self.config_file
            
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # إزالة المفاتيح التي تبدأ بـ _
                return {k: v for k, v in config.items() if not k.startswith('_')}
            except (json.JSONDecodeError, IOError):
                pass
        
        # إعدادات افتراضية أساسية
        return {
            "default_model": "tiny.en",
            "default_language": "en",
            "output_formats": ["txt"],
            "verbose": False
        }
    
    def _resolve_path(self, file_path: str, base_dir: str) -> str:
        """حل المسار - إما مطلق أو نسبي من المجلد المحدد"""
        if not file_path:
            return ""
            
        path = Path(file_path)
        
        # إذا كان مساراً مطلقاً، استخدمه كما هو
        if path.is_absolute():
            return str(path)
        
        # إذا كان نسبياً، اربطه بالمجلد الأساسي
        if base_dir:
            base_path = Path(base_dir)
            if not base_path.is_absolute():
                base_path = self.whisper_dir.parent / base_path
            return str(base_path / file_path)
        
        # كحل أخير، اربطه بمجلد البرنامج
        return str(self.whisper_dir.parent / file_path)
    
    def _create_temp_config(self, **kwargs) -> str:
        """إنشاء ملف إعدادات مؤقت"""
        config = self.default_config.copy()
        config.update(kwargs)
        
        # إنشاء ملف مؤقت
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            return f.name
    

    
    def transcribe(
        self,
        audio_file: str,
        model: str = None,
        language: str = None,
        output_dir: str = None,
        output_formats: List[str] = None,
        verbose: bool = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        نسخ ملف صوتي إلى نص
        
        المعاملات:
            audio_file: مسار الملف الصوتي (مطلق أو نسبي)
            model: اسم النموذج أو مساره (افتراضي من الإعدادات)
            language: رمز اللغة (مثل en, ar) أو None للكشف التلقائي
            output_dir: مجلد الإخراج (افتراضي من الإعدادات)
            output_formats: قائمة صيغ الإخراج (txt, vtt, srt)
            verbose: عرض تفاصيل أكثر
            **kwargs: معاملات إضافية لـ transcribe.py
        
        العائد:
            TranscriptionResult: نتيجة عملية النسخ
        """
        
        # تحضير المعاملات
        model = model or self.default_config.get("default_model", "tiny.en")
        language = language or self.default_config.get("default_language", "en")
        output_dir = output_dir or self.output_dir
        output_formats = output_formats or self.default_config.get("output_formats", ["txt"])
        verbose = verbose if verbose is not None else self.default_config.get("verbose", False)
        
        # حل مسار الملف الصوتي
        audio_path = self._resolve_path(audio_file, self.audio_dir)
        
        # التحقق من وجود الملف
        if not os.path.exists(audio_path):
            return TranscriptionResult(
                text="",
                language="",
                success=False,
                error_message=f"الملف الصوتي غير موجود: {audio_path}"
            )
        
        # حل مسار الإخراج
        output_path = self._resolve_path(output_dir, "")
        
        # بناء الأمر
        cmd = [
            sys.executable,
            str(self.transcribe_script),
            "-a", audio_path,
            "-m", model,
            "-o", output_path,
            "--formats"] + output_formats
        
        if language and language != "auto":
            cmd.extend(["-l", language])
            
        if verbose:
            cmd.append("-v")
        
        # إضافة المعاملات الإضافية
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        try:
            # تشغيل الأمر
            result = subprocess.run(
                cmd,
                cwd=self.whisper_dir.parent,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                return TranscriptionResult(
                    text="",
                    language="",
                    success=False,
                    error_message=f"خطأ في التشغيل: {result.stderr}"
                )
            
            # استخراج النص من المخرجات
            output_lines = result.stdout.strip().split('\n')
            text = ""
            language_detected = language
            
            # البحث عن النص المستخرج
            text_started = False
            for line in output_lines:
                if "📝 النص المستخرج:" in line:
                    text_started = True
                elif text_started and line.startswith("="):
                    if text:  # إذا وجدنا النص، توقف عند الخط الثاني
                        break
                elif text_started and not line.startswith("="):
                    text += line + "\n"
                elif "🌍 اللغة المكتشفة:" in line:
                    # استخراج اللغة المكتشفة
                    try:
                        language_detected = line.split(":")[-1].strip()
                    except:
                        pass
            
            text = text.strip()
            
            # البحث عن ملفات الإخراج
            output_files = {}
            audio_name = Path(audio_path).stem
            
            for fmt in output_formats:
                output_file = Path(output_path) / f"{audio_name}.{fmt}"
                if output_file.exists():
                    output_files[fmt] = str(output_file)
            
            return TranscriptionResult(
                text=text,
                language=language_detected,
                success=True,
                output_files=output_files
            )
            
        except Exception as e:
            return TranscriptionResult(
                text="",
                language="",
                success=False,
                error_message=f"خطأ في العملية: {str(e)}"
            )
    
    def transcribe_batch(
        self,
        audio_files: List[str],
        **kwargs
    ) -> List[TranscriptionResult]:
        """
        نسخ عدة ملفات صوتية
        
        المعاملات:
            audio_files: قائمة بمسارات الملفات الصوتية
            **kwargs: معاملات النسخ (نفس معاملات transcribe)
        
        العائد:
            List[TranscriptionResult]: قائمة بنتائج النسخ
        """
        results = []
        for audio_file in audio_files:
            result = self.transcribe(audio_file, **kwargs)
            results.append(result)
        return results


# دوال مساعدة للاستخدام المباشر
def quick_transcribe(audio_file: str, model: str = "tiny.en", language: str = "en") -> str:
    """
    نسخ سريع لملف صوتي - يرجع النص فقط
    
    المعاملات:
        audio_file: مسار الملف الصوتي
        model: اسم النموذج
        language: رمز اللغة
    
    العائد:
        str: النص المستخرج
    """
    api = WhisperAPI()
    result = api.transcribe(audio_file, model=model, language=language)
    return result.text if result.success else ""


def transcribe_to_file(
    audio_file: str,
    output_file: str,
    model: str = "tiny.en",
    language: str = "en"
) -> bool:
    """
    نسخ ملف صوتي وحفظ النتيجة في ملف
    
    المعاملات:
        audio_file: مسار الملف الصوتي
        output_file: مسار ملف الإخراج
        model: اسم النموذج
        language: رمز اللغة
    
    العائد:
        bool: True إذا نجحت العملية
    """
    api = WhisperAPI()
    result = api.transcribe(audio_file, model=model, language=language)
    
    if result.success:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.text)
            return True
        except Exception:
            return False
    
    return False



