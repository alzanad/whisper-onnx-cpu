# 🎤 Whisper ONNX CPU

> مشروع Whisper محوّل إلى صيغة ONNX للتشغيل السريع والفعّال على المعالجات المحلية بدون الحاجة لكروت الرسومات

## 📖 حول المشروع

هذا المشروع مبني على العمل الممتاز من المشروع الأصلي:
**[whisper-onnx-cpu](https://github.com/PINTO0309/whisper-onnx-cpu)**

**واجهت مشاكل كثيرة في المشروع الأصلي وكان معقد لذلك قمت بتبسيطه**

## ⚠️ ملاحظات مهمة حول النماذج والعربية

**النماذج العامة تعطي نتائج ضعيفة جداً مع اللغة العربية**. السبب أن هذه النماذج مدربة بشكل عام على عدة لغات ولا تركز على لغة واحدة بعينها، مما يجعل أداؤها ضعيفاً مع اللغات التي تتطلب فهماً خاصاً مثل العربية.

تم اختبار النماذج التالية مع النتائج:
- `tiny`: نتائج سيئة للغاية مع العربية
- `base`: نتائج سيئة للغاية مع العربية  
- `small`: نتائج سيئة للغاية مع العربية
**لم نختبر الباقي لعلمنا بسوء النتيجة**

## الميزات الرئيسية

- **أداء محسّن**: تشغيل سريع على المعالجات بفضل تحسينات ONNX
- **واجهة سطر أوامر قوية**: تحكم كامل في عملية التحويل مع إعدادات قابلة للتخصيص
- **إدارة ذكية للملفات**: بحث تلقائي في المجلدات المخصصة مع دعم امتدادات متعددة
- **صيغ إخراج متعددة**: حفظ بصيغ TXT، VTT، SRT
- **إعدادات مرنة**: ملف config.json لحفظ الإعدادات المفضلة

## البدء السريع

### المتطلبات الأساسية

**مطلوب Python 3.10 أو أحدث للحصول على أفضل أداء وتوافق.**

### إعداد البيئة

**ننصح بشدة بإنشاء بيئة افتراضية:**
```bash
python3.10 -m venv venv
source venv/bin/activate  # على Linux/Mac
# أو
venv\Scripts\activate     # على Windows
```

**إذا لم ترغب في إنشاء بيئة افتراضية، استخدم هذا الأمر:**
```bash
pip install --upgrade --pre transformers
```

### 1. تثبيت المتطلبات

```bash
pip install -r requirements.txt
```

### 2. تنزيل النماذج

```bash
cd download_models
python download_models.py --model tiny.en
```

### 3. تشغيل التحويل

```bash
cd whisper
python transcribe.py -m tiny.en -l en -a audio_file.mp3
```

## 📚 دليل الاستخدام المفصل

### 🎯 استخدام ملف transcribe.py

```bash
# الانتقال إلى مجلد whisper
cd whisper

# تشغيل بالإعدادات الافتراضية (من config.json)
python transcribe.py

# تحديد المعاملات يدوياً
python transcribe.py -m tiny.en -l en -a ../audio.mp3

# استخدام مسار كامل للنموذج
python transcribe.py -m ../whisper-models/tiny.en -l en -a ../audio.mp3

# تحديد مجلد الإخراج وصيغ متعددة
python transcribe.py -m tiny.en -l en -a ../audio.mp3 -o ../output --formats txt vtt srt

# عرض النماذج المدعومة
python transcribe.py --list-models

# تشغيل مع تفاصيل أكثر
python transcribe.py -m tiny.en -l en -a ../audio.mp3 -v
```

### 📋 المعاملات المتاحة

- `-m, --model`: اسم النموذج أو مساره الكامل
- `-l, --language`: رمز اللغة (en, ar, وغيرها)  
- `-a, --audio`: مسار الملف الصوتي
- `-o, --output`: مجلد الإخراج (افتراضي: output)
- `--formats`: صيغ الإخراج (txt, vtt, srt)
- `--config`: مسار ملف إعدادات مخصص
- `--list-models`: عرض النماذج المدعومة
- `-v, --verbose`: عرض تفاصيل أكثر

### ⚙️ ملف الإعدادات (config.json)

يمكن تعديل الإعدادات الافتراضية في ملف `config.json`:

```json
{
    "default_model": "tiny.en",
    "default_language": "en", 
    "default_audio": "audio.mp3",
    "models_directory": "whisper-models",
    "output_directory": "output",
    "output_formats": ["txt", "vtt", "srt"]
}
```
 إذا وضعت اﻹعدادات اﻹفتراضية فتستطيع تشغيل النظام ب:
 
 ```bash
python transcribe.py
 ```

### 🔽 تنزيل النماذج

النماذج منفصلة عن المشروع الأساسي. لتنزيل النماذج:

```bash
# الانتقال إلى مجلد أدوات التنزيل
cd download_models

# عرض النماذج المتاحة
python download_models.py --list

# تنزيل نموذج محدد
python download_models.py --model tiny.en

# تنزيل جميع النماذج
python download_models.py --all
```

النماذج ستُحفظ في مجلد `whisper-models` وسيجدها البرنامج تلقائياً.

## 🗂️ هيكل المشروع

```
whisper-onnx-cpu/
├── 📄 README.md                    # هذا الملف - دليل شامل للمشروع
├── 🔗 api.py                       # واجهة برمجية مرنة للدمج مع المشاريع
├── ⚙️ config.json                 # ملف الإعدادات العامة
├── 📋 requirements.txt            # متطلبات Python
├── 🎵 whisper/                    # المكتبة الأساسية
│   ├── 🚀 transcribe.py          # ملف التشغيل الرئيسي
│   ├── ⚙️ config.json           # إعدادات whisper
│   ├── 🎙️ audio.py              # معالجة الصوت
│   ├── 🧠 model.py               # إدارة النماذج
│   ├── 🔤 tokenizer.py           # معالجة النصوص
│   ├── 🔍 decoding.py            # فك التشفير
│   └── 📁 assets/                # ملفات النظام
├── 🤖 whisper-models/            # مجلد النماذج
│   ├── tiny.en_encoder_11.onnx
│   ├── tiny.en_decoder_11.onnx
│   └── ...
├── 🎵 audio/                     # الملفات الصوتية
├── 📝 output/                    # ملفات الإخراج
└── 📥 download_models/           # أدوات تنزيل النماذج
    ├── download_models.py
    └── models_config.json
```

## أمثلة الاستخدام المتقدمة

### تحويل سريع بالإعدادات الافتراضية
```bash
cd whisper
python transcribe.py
```

### تحديد النموذج واللغة والملف
```bash
python transcribe.py -m small -l ar -a arabic_speech.mp3
```

### حفظ بصيغ متعددة مع تفاصيل أكثر
```bash
python transcribe.py -m base.en -l en -a interview.wav \
  --formats txt vtt srt -v
```

### استخدام مسار كامل للنموذج
```bash
python transcribe.py -m ../whisper-models/medium -l ar \
  -a ../audio/lecture.m4a -o ../output
```

### مثال شامل متقدم
```bash
cd whisper
python transcribe.py \
  -m ../whisper-models/tiny.en \
  -l en \
  -a ../audio.mp3 \
  -o ../output \
  --formats txt vtt srt \
  -v
```

### استخدام ملف إعدادات مخصص  
```bash
python transcribe.py --config ../my_config.json
```

## � الواجهة البرمجية (API) - ملف api.py

يتضمن المشروع **واجهة برمجية مرنة وقوية** في ملف `api.py` تسمح بدمج قدرات Whisper في مشاريعك البرمجية بسهولة.

### أمثلة الاستخدام البرمجي

#### الاستخدام الأساسي
```python
from api import WhisperAPI

# إنشاء واجهة مع الإعدادات الافتراضية
whisper = WhisperAPI()

# نسخ ملف صوتي
result = whisper.transcribe("audio/my_audio.mp3")

if result.success:
    print(f"النص: {result.text}")
    print(f"اللغة: {result.language}")
else:
    print(f"خطأ: {result.error_message}")
```

#### استخدام متقدم مع تخصيص الإعدادات
```python
# تخصيص المجلدات والإعدادات
whisper = WhisperAPI(
    models_dir="my_models",
    audio_dir="my_audio",
    output_dir="my_output"
)

# نسخ مع إعدادات مخصصة
result = whisper.transcribe(
    audio_file="interview.wav",
    model="base.en",
    language="en",
    output_formats=["txt", "srt", "vtt"],
    verbose=True
)

# الوصول للملفات المحفوظة
if result.success:
    print(f"الملفات المحفوظة: {result.output_files}")
```

#### نسخ عدة ملفات دفعة واحدة
```python
# قائمة الملفات الصوتية
audio_files = ["file1.mp3", "file2.wav", "file3.m4a"]

# نسخ جميع الملفات
results = whisper.transcribe_batch(
    audio_files,
    model="tiny.en",
    language="en",
    output_formats=["txt"]
)

# معالجة النتائج
for i, result in enumerate(results):
    if result.success:
        print(f"ملف {i+1}: تم النسخ بنجاح")
        print(f"النص: {result.text[:100]}...")
    else:
        print(f"ملف {i+1}: فشل - {result.error_message}")
```

#### دوال الاستخدام السريع
```python
from api import quick_transcribe, transcribe_to_file

# نسخ سريع - يرجع النص مباشرة
text = quick_transcribe("audio.mp3", model="tiny.en", language="en")
print(text)

# نسخ وحفظ في ملف
success = transcribe_to_file(
    audio_file="audio.mp3",
    output_file="output.txt",
    model="base.en",
    language="en"
)
```

### 🎯 حالات الاستخدام المقترحة

- **تطبيقات الويب**: دمج النسخ الصوتي في تطبيقات Flask/Django
- **أتمتة المعالجة**: نسخ ملفات صوتية متعددة تلقائياً
- **تطبيقات سطح المكتب**: إضافة ميزة النسخ للتطبيقات المحلية
- **معالجة البيانات**: تحويل مكتبات صوتية كاملة إلى نصوص
- **التكامل مع خدمات أخرى**: ربط النسخ مع قواعد البيانات أو APIs

## �🔧 المتطلبات التقنية

- **Python**: 3.10 أو أحدث (موصى به بشدة)
- **ذاكرة**: 4GB RAM (موصى بـ 8GB للنماذج الكبيرة)
- **تخزين**: 100MB - 1GB حسب النموذج المستخدم
- **معالج**: أي معالج حديث (لا يتطلب GPU)

## 📄 الترخيص

هذا المشروع مرخص تحت رخصة MIT - راجع ملف [LICENSE](LICENSE) للتفاصيل.

---

<div align="center">

**إذا أعجبك المشروع، لا تنس إعطاؤه ⭐**

[🐛 المشاكل](../../issues) | [💬 المناقشات](../../discussions)

</div>
