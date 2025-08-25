#!/usr/bin/env python3
"""
ملف التنفيذ الرئيسي لـ Whisper ONNX
يدعم CLI والإعدادات من ملف config.json

الاستخدام:
    python transcribe.py                                    # استخدام الإعدادات الافتراضية
    python transcribe.py -m tiny.en -l en -a audio.mp3     # تحديد المعاملات
    python transcribe.py -m whisper-models/tiny.en          # استخدام مسار كامل للنموذج
    python transcribe.py -a my_audio.wav                    # البحث عن الملف في مجلد audio/
    python transcribe.py --config custom_config.json       # استخدام ملف إعدادات مخصص

ملاحظات:
    - ملفات النماذج تُبحث في مجلد whisper-models/
    - الملفات الصوتية تُبحث في مجلد audio/ إذا لم يُحدد مسار كامل
    - النتائج تُحفظ في مجلد output/
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

# إضافة المجلد الأساسي للمسار
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from whisper.model import load_model, available_models
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram, SAMPLE_RATE, N_FRAMES, HOP_LENGTH
from whisper.decoding import decode, detect_language, DecodingOptions, DecodingResult
from whisper.tokenizer import get_tokenizer, LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import exact_div, format_timestamp, write_txt, write_vtt, write_srt
from whisper import utils

def load_config(config_path: str = None) -> dict:
    """تحميل ملف الإعدادات"""
    if config_path is None:
        # البحث أولاً في المجلد الحالي، ثم في المجلد الأب
        config_path = current_dir / "config.json"
        if not config_path.exists():
            config_path = project_root / "config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ خطأ: ملف الإعدادات غير موجود: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # إزالة المفاتيح التي تبدأ بـ _
        return {k: v for k, v in config.items() if not k.startswith('_')}
    except json.JSONDecodeError as e:
        print(f"❌ خطأ في قراءة ملف الإعدادات: {e}")
        return {}

def list_audio_files(audio_dir: str) -> list:
    """عرض قائمة بالملفات الصوتية المتاحة"""
    if not os.path.isabs(audio_dir):
        audio_dir_path = current_dir / audio_dir
    else:
        audio_dir_path = Path(audio_dir)
    if not audio_dir_path.exists():
        return []
    
    supported_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma', '.aac'}
    audio_files = []
    
    for file_path in audio_dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            audio_files.append(file_path.name)
    
    return sorted(audio_files)

def resolve_model_path(model_name: str, models_dir: str = "whisper-models") -> tuple:
    """حل مسار النموذج واستخراج اسم النموذج الأساسي"""
    # إذا كان مساراً كاملاً
    if '/' in model_name or '\\' in model_name:
        model_path = Path(model_name)
        base_name = model_path.name
        model_dir = model_path.parent
    else:
        # إذا كان اسماً فقط
        base_name = model_name
        # البحث في المجلد المحدد في الإعدادات (مسار مطلق)
        if not os.path.isabs(models_dir):
            model_dir = current_dir / models_dir
        else:
            model_dir = Path(models_dir)
    
    encoder_path = model_dir / f"{base_name}_encoder_11.onnx"
    decoder_path = model_dir / f"{base_name}_decoder_11.onnx"
    
    return str(model_dir / base_name), base_name, encoder_path, decoder_path

def resolve_audio_path(audio_name: str, audio_dir: str = "audio") -> str:
    """حل مسار الملف الصوتي - يدعم البحث الذكي بامتدادات مختلفة"""
    # إذا كان مساراً كاملاً، استخدمه كما هو
    if '/' in audio_name or '\\' in audio_name or os.path.exists(audio_name):
        return audio_name
    
    # الامتدادات المدعومة
    supported_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma', '.aac']
    
    # إذا كان اسماً فقط، ابحث في مجلد الصوتيات
    if not os.path.isabs(audio_dir):
        audio_dir_path = current_dir / audio_dir
    else:
        audio_dir_path = Path(audio_dir)
    
    # البحث المباشر أولاً
    direct_path = audio_dir_path / audio_name
    if direct_path.exists():
        return str(direct_path)
    
    # إذا لم يكن له امتداد، جرب الامتدادات المختلفة
    if '.' not in audio_name:
        for ext in supported_extensions:
            test_path = audio_dir_path / (audio_name + ext)
            if test_path.exists():
                return str(test_path)
    
    # البحث في المجلد الحالي كحل أخير
    if os.path.exists(audio_name):
        return audio_name
    
    # البحث بالامتدادات في المجلد الحالي
    if '.' not in audio_name:
        for ext in supported_extensions:
            test_path = audio_name + ext
            if os.path.exists(test_path):
                return test_path
    
    # إرجاع المسار الأصلي للتعامل مع الخطأ لاحقاً
    return audio_name

def transcribe(
    *,
    model,
    audio: Union[str, np.ndarray],
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper
    """
    from typing import List, Optional, Tuple, Union
    
    mel: np.ndarray = log_mel_spectrogram(audio)

    if decode_options.get("language", None) is None:
        if verbose:
            print("🔍 كشف اللغة...")
        segment = pad_or_trim(mel, N_FRAMES)
        _, probs = model.detect_language(segment)
        decode_options["language"] = max(probs, key=probs.get)
        if verbose is not None:
            print(f"🌍 اللغة المكتشفة: {LANGUAGES[decode_options['language']].title()}")

    mel = mel[np.newaxis, ...]
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    def decode_with_fallback(segment: np.ndarray) -> List[DecodingResult]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        kwargs = {**decode_options}
        t = temperatures[0]
        if t == 0:
            best_of = kwargs.pop("best_of", None)
        else:
            best_of = kwargs.get("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        results = model.decode(segment, options)

        kwargs.pop("beam_size", None)  # no beam search for t > 0
        kwargs.pop("patience", None)  # no patience for t > 0
        kwargs["best_of"] = best_of  # enable best_of for t > 0
        for t in temperatures[1:]:
            needs_fallback = [
                compression_ratio_threshold is not None
                and result.compression_ratio > compression_ratio_threshold
                or logprob_threshold is not None
                and result.avg_logprob < logprob_threshold
                for result in results
            ]
            if any(needs_fallback):
                options = DecodingOptions(**kwargs, temperature=t)
                retries = model.decode(segment[needs_fallback], options)
                for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                    results[original_index] = retries[retry_index]

        return results

    seek = 0
    input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx)  # mel frames per output token: 2
    time_precision = (input_stride * HOP_LENGTH / SAMPLE_RATE)  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def add_segment(*, start: float, end: float, text_tokens: np.ndarray, result: DecodingResult):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append({
            "id": len(all_segments),
            "seek": seek,
            "start": start,
            "end": end,
            "text": text,
            "tokens": result.tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        })
        if verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}", flush=True)

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    previous_seek_value = seek

    while seek < num_frames:
        timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
        segment = pad_or_trim(mel[:, :, seek:], N_FRAMES)
        segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

        decode_options["prompt"] = all_tokens[prompt_reset_since:]
        result = decode_with_fallback(segment)[0]
        tokens = result.tokens

        if no_speech_threshold is not None:
            # no voice activity check
            should_skip = result.no_speech_prob > no_speech_threshold
            if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                # don't skip if the logprob is high enough, despite the no_speech_prob
                should_skip = False

            if should_skip:
                seek += segment.shape[-1]  # fast-forward to the next segment boundary
                continue

        timestamp_tokens: np.ndarray = np.greater_equal(tokens, tokenizer.timestamp_begin)
        consecutive = np.add(np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0], 1)
        if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
            last_slice = 0
            for current_slice in consecutive:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = (sliced_tokens[0] - tokenizer.timestamp_begin)
                end_timestamp_position = (sliced_tokens[-1] - tokenizer.timestamp_begin)
                add_segment(
                    start=timestamp_offset + start_timestamp_position * time_precision,
                    end=timestamp_offset + end_timestamp_position * time_precision,
                    text_tokens=sliced_tokens[1:-1],
                    result=result,
                )
                last_slice = current_slice
            last_timestamp_position = (tokens[last_slice - 1] - tokenizer.timestamp_begin)
            seek += last_timestamp_position * input_stride
            all_tokens.extend(list(tokens[: last_slice + 1]))
        else:
            duration = segment_duration
            tokens = np.asarray(tokens) if isinstance(tokens, list) else tokens
            timestamps = tokens[np.ravel_multi_index(np.nonzero(timestamp_tokens), timestamp_tokens.shape)]
            if len(timestamps) > 0:
                # no consecutive timestamps but it has a timestamp; use the last one.
                # single timestamp at the end means no speech after the last timestamp.
                last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                duration = last_timestamp_position * time_precision

            add_segment(
                start=timestamp_offset,
                end=timestamp_offset + duration,
                text_tokens=tokens,
                result=result,
            )

            seek += segment.shape[-1]
            all_tokens.extend(list(tokens))

        if not condition_on_previous_text or result.temperature > 0.5:
            # do not feed the prompt tokens if a high temperature was used
            prompt_reset_since = len(all_tokens)

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language)
def create_output_dir(output_dir: str):
    """إنشاء مجلد الإخراج إذا لم يكن موجوداً"""
    os.makedirs(output_dir, exist_ok=True)

def save_output(result, audio_path: str, output_dir: str, formats: list):
    """حفظ النتائج بالصيغ المطلوبة"""
    audio_name = Path(audio_path).stem
    
    # استخراج النص والـ segments
    if isinstance(result, dict):
        text = result.get("text", "")
        segments = result.get("segments", [])
    elif hasattr(result, 'text'):
        text = result.text
        segments = []
    else:
        text = str(result)
        segments = []
    
    # إنشاء segments افتراضي إذا لم يكن متاحاً
    if not segments and text:
        segments = [{"start": 0.0, "end": 30.0, "text": text}]
    
    # حفظ النص العادي
    if "txt" in formats:
        txt_path = Path(output_dir) / f"{audio_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            write_txt(segments, file=f)
        print(f"💾 تم حفظ النص: {txt_path}")
    
    # حفظ VTT
    if "vtt" in formats:
        vtt_path = Path(output_dir) / f"{audio_name}.vtt"
        with open(vtt_path, 'w', encoding='utf-8') as f:
            write_vtt(segments, file=f)
        print(f"💾 تم حفظ VTT: {vtt_path}")
    
    # حفظ SRT  
    if "srt" in formats:
        srt_path = Path(output_dir) / f"{audio_name}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            write_srt(segments, file=f)
        print(f"💾 تم حفظ SRT: {srt_path}")

def main():
    parser = argparse.ArgumentParser(description="Whisper ONNX Transcription Tool")
    parser.add_argument("-m", "--model", type=str, help="اسم النموذج أو مساره")
    parser.add_argument("-l", "--language", type=str, help="رمز اللغة (مثل: en, ar)")
    parser.add_argument("-a", "--audio", type=str, help="مسار الملف الصوتي أو اسمه فقط")
    parser.add_argument("-o", "--output", type=str, help="مجلد الإخراج")
    parser.add_argument("--config", type=str, help="مسار ملف الإعدادات المخصص")
    parser.add_argument("--list-models", action="store_true", help="عرض النماذج المدعومة")
    parser.add_argument("--list-audio", action="store_true", help="عرض الملفات الصوتية المتاحة")
    parser.add_argument("--formats", nargs="+", choices=["txt", "vtt", "srt"], help="صيغ الإخراج")
    parser.add_argument("-v", "--verbose", action="store_true", help="عرض تفاصيل أكثر")
    
    args = parser.parse_args()
    
    # تحميل الإعدادات أولاً للحصول على المجلدات
    config = load_config(args.config)
    audio_dir = config.get("audio_directory", "audio")
    
    # عرض النماذج المدعومة
    if args.list_models:
        print("📋 النماذج المدعومة:")
        for model in available_models():
            print(f"   - {model}")
        return
    
    # عرض الملفات الصوتية المتاحة
    if args.list_audio:
        audio_files = list_audio_files(audio_dir)
        if audio_files:
            print(f"🎵 الملفات الصوتية المتاحة في '{audio_dir}':")
            for audio_file in audio_files:
                print(f"   - {audio_file}")
            print(f"\n💡 يمكنك استخدام: python transcribe.py -a filename")
        else:
            print(f"📭 لا توجد ملفات صوتية في '{audio_dir}'")
            print(f"💡 ضع ملفاتك الصوتية في مجلد '{audio_dir}' أو استخدم المسار الكامل")
        return
    
    # دمج الإعدادات مع معاملات CLI
    model_name = args.model or config.get("default_model", "tiny.en")
    language = args.language or config.get("default_language", "en")
    audio_name = args.audio or config.get("default_audio", "audio.mp3")
    output_dir = args.output or config.get("output_directory", "output")
    models_dir = config.get("models_directory", "whisper-models")
    audio_dir = config.get("audio_directory", "audio")
    output_formats = args.formats or config.get("output_formats", ["txt"])
    verbose = args.verbose or config.get("verbose", False)
    
    # حل مسار الملف الصوتي
    audio_path = resolve_audio_path(audio_name, audio_dir)
    
    if verbose:
        print(f"🎯 النموذج: {model_name}")
        print(f"🌍 اللغة: {language}")
        print(f"🎵 الملف الصوتي: {audio_name} -> {audio_path}")
        print(f"📁 مجلد الصوتيات: {audio_dir}")
        print(f"📁 مجلد الإخراج: {output_dir}")
    
    # التحقق من وجود الملف الصوتي
    if not os.path.exists(audio_path):
        print(f"❌ خطأ: الملف الصوتي غير موجود: {audio_path}")
        if audio_name != audio_path:
            print(f"🔍 تم البحث عن '{audio_name}' في:")
            print(f"   - مجلد الصوتيات: {audio_dir}/")
            print(f"   - المجلد الحالي")
            print(f"💡 تلميح: يمكنك وضع الملفات الصوتية في مجلد '{audio_dir}' واستخدام اسم الملف فقط")
            print(f"📁 الامتدادات المدعومة: .mp3, .wav, .flac, .m4a, .ogg, .wma, .aac")
        return
    
    # حل مسار النموذج
    model_path, base_model_name, encoder_path, decoder_path = resolve_model_path(model_name, models_dir)
    
    # التحقق من وجود ملفات النموذج
    if not encoder_path.exists() or not decoder_path.exists():
        print(f"❌ خطأ: النموذج '{model_name}' غير موجود")
        print(f"📁 ملفات مطلوبة:")
        print(f"   - {encoder_path}")
        print(f"   - {decoder_path}")
        print(f"📥 يجب تنزيل هذين الملفين ووضعهما في المجلد المناسب")
        print(f"🔗 استخدم أداة التنزيل: python download_models/download_models.py --model {base_model_name}")
        return
    
    if verbose:
        print(f"📂 مسار النموذج: {model_path}")
        print(f"🔧 اسم النموذج الأساسي: {base_model_name}")
    
    try:
        # تحميل النموذج
        print(f"🔄 تحميل النموذج: {base_model_name}")
        model = load_model(model_path)
        
        # تحميل الصوت
        print(f"🎵 تحميل الملف الصوتي: {audio_path}")
        audio = load_audio(audio_path)
        audio = pad_or_trim(audio)
        
        # تحويل إلى mel spectrogram
        mel = log_mel_spectrogram(audio)
        
        # النسخ باستخدام الدالة المحدثة
        print("⚡ بدء عملية النسخ...")
        result = transcribe(
            model=model,
            audio=audio_path,
            language=language if language != "auto" else None,
            verbose=verbose
        )
        
        # إنشاء مجلد الإخراج
        create_output_dir(output_dir)
        
        # حفظ النتائج
        save_output(result, audio_path, output_dir, output_formats)
        
        # عرض النص المستخرج
        print("\n📝 النص المستخرج:")
        print("=" * 50)
        print(result["text"])
        print("=" * 50)
        
        # عرض الـ segments إذا كان verbose
        if verbose and result.get("segments"):
            print(f"\n📊 تم استخراج {len(result['segments'])} مقطع:")
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                print(f"   [{start_time} --> {end_time}] {segment['text']}")
        
        print(f"✅ تمت العملية بنجاح!")
        
    except Exception as e:
        print(f"❌ خطأ في العملية: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

def cli():
    """نقطة دخول واجهة سطر الأوامر"""
    main()

if __name__ == "__main__":
    main()