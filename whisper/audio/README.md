# مجلد الملفات الصوتية

هذا المجلد مخصص لحفظ الملفات الصوتية التي تريد تحويلها إلى نص باستخدام Whisper ONNX.

## الاستخدام

ضع ملفاتك الصوتية في هذا المجلد، ثم استخدم اسم الملف فقط (بدون المسار الكامل):

```bash
# بدلاً من:
python transcribe.py -a /path/to/audio/myfile.mp3

# استخدم:
python transcribe.py -a myfile.mp3
# أو حتى بدون الامتداد:
python transcribe.py -a myfile
```

## الامتدادات المدعومة

- `.mp3` - MP3 Audio
- `.wav` - WAV Audio  
- `.flac` - FLAC Audio
- `.m4a` - M4A Audio
- `.ogg` - OGG Audio
- `.wma` - WMA Audio
- `.aac` - AAC Audio

## عرض الملفات المتاحة

```bash
python transcribe.py --list-audio
```

## أمثلة على الاستخدام

```bash
# نسخ ملف صوتي
python transcribe.py -a audio.mp3 -l ar

# نسخ بعدة صيغ
python transcribe.py -a myfile -l en --formats txt vtt srt

# استخدام الإعدادات الافتراضية
python transcribe.py

# عرض الملفات المتاحة
python transcribe.py --list-audio

# تشغيل مع تفاصيل أكثر
python transcribe.py -a myfile -v
```

## ملاحظات

- النظام يبحث في هذا المجلد أولاً، ثم في المجلد الحالي
- يمكن استخدام اسم الملف بدون امتداد إذا كان واضحاً
- الملف `audio.mp3` هو الملف الافتراضي المحدد في `config.json`
- يدعم النظام البحث الذكي للملفات بامتدادات مختلفة
