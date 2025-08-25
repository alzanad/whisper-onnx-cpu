# أدوات تنزيل النماذج

## الملفات

### `models_config.json`
- يحتوي على إعدادات النماذج وروابط التنزيل
- يضم أبعاد كل نموذج ومواقع ملفات encoder و decoder

### `download_models.py`
- أداة تنزيل النماذج من الخوادم
- يستخدم ملف `models_config.json` لمعرفة الروابط

## الاستخدام

```bash
# عرض النماذج المتاحة
python download_models.py --list

# تنزيل نموذج محدد
python download_models.py --model tiny.en

# تنزيل جميع النماذج
python download_models.py --all
```

## ملاحظة مهمة

- بعد التنزيل، ضع ملفات `.onnx` في مجلد المشروع الأساسي
- المشروع الأساسي سيتوقف برسالة واضحة إذا لم يجد النماذج المطلوبة

## مثال كامل

```bash
# 1. تنزيل النموذج
python download_models.py --model tiny.en

# 2. نقل إلى مجلد المشروع (إذا كانت في مجلد منفصل)
mv tiny.en_*.onnx /path/to/whisper-project/

# 3. تشغيل المشروع الأساسي
cd /path/to/whisper-project/
python -m whisper --audio audio.mp3 --model tiny.en
```
