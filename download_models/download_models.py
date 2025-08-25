#!/usr/bin/env python3
"""
🚨 تنويه مهم: هذا الملف منفصل تماماً عن مشروع Whisper الأساسي 🚨

أداة تنزيل نماذج Whisper ONNX
- مخصص فقط لتنزيل النماذج من الخوادم
- لا يعتمد عليه المشروع الأساسي
- استخدمه لتنزيل النماذج ثم ضع ملفات .onnx في مجلد المشروع
- المشروع الأساسي سيتوقف برسالة واضحة إذا لم يجد النماذج

الاستخدام:
    python download_models.py --list                # عرض النماذج المتاحة
    python download_models.py --model tiny.en       # تنزيل نموذج محدد
    python download_models.py --all                 # تنزيل جميع النماذج
"""

import io
import os
import json
import argparse
import requests
import onnx
from onnx.serialization import ProtoSerializer
from typing import Dict, Any

def load_model_config(config_path: str = 'models_config.json') -> Dict[str, Any]:
    """تحميل إعدادات النماذج من ملف JSON"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ملف الإعدادات غير موجود: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def download_model_file(url: str, save_path: str) -> bool:
    """تنزيل ملف نموذج واحد"""
    try:
        print(f"🔽 تنزيل النموذج من: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # إنشاء المجلد إذا لم يكن موجوداً
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # تنزيل الملف مع شريط التقدم
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📥 التقدم: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ تم تنزيل: {save_path}")
        return True
        
    except requests.RequestException as e:
        print(f"\n❌ خطأ في التنزيل: {e}")
        return False
    except Exception as e:
        print(f"\n❌ خطأ غير متوقع: {e}")
        return False

def download_model(model_name: str, save_dir: str = '.', config_path: str = 'models_config.json') -> bool:
    """تنزيل نموذج كامل (encoder + decoder)"""
    try:
        config = load_model_config(config_path)
        
        if model_name not in config:
            print(f"❌ النموذج '{model_name}' غير موجود في الإعدادات")
            # عرض النماذج المتاحة (تجاهل المفاتيح التي تبدأ بخط سفلي)
            available_models = [k for k in config.keys() if not k.startswith('_')]
            print(f"النماذج المتاحة: {', '.join(available_models)}")
            return False
        
        model_config = config[model_name]
        urls = model_config['urls']
        
        print(f"🚀 بدء تنزيل النموذج: {model_name}")
        
        success = True
        for component, url in urls.items():
            file_name = f"{model_name}_{component}_11.onnx"
            save_path = os.path.join(save_dir, file_name)
            
            # تخطي التنزيل إذا كان الملف موجوداً
            if os.path.exists(save_path):
                print(f"⏭️ الملف موجود بالفعل: {save_path}")
                continue
            
            if not download_model_file(url, save_path):
                success = False
                break
        
        if success:
            print(f"🎉 تم تنزيل النموذج '{model_name}' بنجاح!")
        else:
            print(f"💥 فشل في تنزيل النموذج '{model_name}'")
        
        return success
        
    except Exception as e:
        print(f"❌ خطأ في تنزيل النموذج: {e}")
        return False

def list_available_models(config_path: str = 'models_config.json'):
    """عرض قائمة النماذج المتاحة"""
    try:
        config = load_model_config(config_path)
        print("📋 النماذج المتاحة للتنزيل:")
        print("-" * 40)
        
        for model_name, model_config in config.items():
            # تجاهل المفاتيح التي تبدأ بخط سفلي (مثل _notice)
            if model_name.startswith('_'):
                continue
                
            dims = model_config['dimensions']
            print(f"🔹 {model_name}")
            print(f"   📊 المفردات: {dims['n_vocab']:,}")
            print(f"   🧠 حالة الصوت: {dims['n_audio_state']}")
            print(f"   📝 حالة النص: {dims['n_text_state']}")
            print()
            
    except Exception as e:
        print(f"❌ خطأ في عرض النماذج: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="أداة تنزيل نماذج Whisper ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  %(prog)s --list                    # عرض النماذج المتاحة
  %(prog)s --model tiny.en           # تنزيل نموذج tiny.en
  %(prog)s --model small --dir ../whisper-models  # تنزيل إلى مجلد محدد
  %(prog)s --all                     # تنزيل جميع النماذج
        """
    )
    
    parser.add_argument('--model', '-m', type=str, 
                       help='اسم النموذج المراد تنزيله')
    parser.add_argument('--dir', '-d', type=str, default='../whisper-models', 
                       help='مجلد الحفظ (افتراضي: ../whisper-models)')
    parser.add_argument('--config', '-c', type=str, default='models_config.json',
                       help='مسار ملف الإعدادات')
    parser.add_argument('--list', '-l', action='store_true',
                       help='عرض قائمة النماذج المتاحة')
    parser.add_argument('--all', '-a', action='store_true',
                       help='تنزيل جميع النماذج')
    
    args = parser.parse_args()
    
    # عرض النماذج المتاحة
    if args.list:
        list_available_models(args.config)
        return
    
    # تنزيل جميع النماذج
    if args.all:
        try:
            config = load_model_config(args.config)
            print("🚀 بدء تنزيل جميع النماذج...")
            
            # تجاهل المفاتيح التي تبدأ بخط سفلي (مثل _notice)
            model_names = [k for k in config.keys() if not k.startswith('_')]
            
            for model_name in model_names:
                print(f"\n{'='*50}")
                download_model(model_name, args.dir, args.config)
            
            print(f"\n🎉 انتهى تنزيل جميع النماذج!")
            
        except Exception as e:
            print(f"❌ خطأ في تنزيل جميع النماذج: {e}")
        return
    
    # تنزيل نموذج محدد
    if args.model:
        download_model(args.model, args.dir, args.config)
        return
    
    # لا توجد معاملات كافية
    parser.print_help()

if __name__ == '__main__':
    main()
