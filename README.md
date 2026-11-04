# ميزان الذهب | Gold Image Classifier

تطبيق بواجهة عربية لتصنيف **صور الذهب إلى حقيقي ومزيف** باستخدام التعلم العميق،
مع مسار متكامل لتدقيق البيانات، والتدريب، والتقييم، ومقارنة التجارب.

**أفضل دقة حالية: 88.45% على اختبار المشروع باستخدام دمج ResNet18 وEfficientNet-B0. هدف 90% لم يتحقق؛ لم تُضف صور ذهب جديدة.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?logo=flask&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.11.3-D00000?logo=keras&logoColor=white)

> **استخدام بحثي وتجريبي:** التصنيف تقدير بصري، وليس إثباتًا لأصالة المعدن أو عياره.
> لا يُستخدم لاتخاذ قرارات شراء أو بيع أو كبديل لفحص الذهب لدى مختص.

## المزايا

- واجهة عربية تدعم اتجاه الكتابة من اليمين إلى اليسار ومعاينة الصور.
- رفع صور JPG وPNG وWebP، وعرض التصنيف ودرجات الفئتين.
- نموذج CNN مدرّب مرفق لتشغيل التطبيق دون إعادة التدريب.
- تدقيق جودة الصور والتكرار القريب، واستبعاد النسخ المتداخلة من التدريب.
- مقارنة تجهيز الصور وزيادة تنوعها باستخدام مجموعة التحقق فقط.
- مسار MobileNetV2 بأوزان ImageNet وتدريب تدريجي مع تجميد BatchNormalization.
- اختبارات للتدريب والحفظ والتحميل وواجهة التصنيف وتطابق المعالجة.

## حالة النماذج والنتائج

النسخة النشطة تدمج **EfficientNet-B0 بنسبة 75% وResNet18 بنسبة 25%**،
مع متوسط المنظر الأصلي والانعكاس الأفقي وعتبة قرار 55%.
المدخلات 320×320، ويُعاد تحجيمها داخليًا إلى 224×224 لفرع ResNet18.
ترتيب الفئات: `fake = 0`، و`real = 1`.
تظل GoldCNN الصغيرة السابقة مرفقة للمقارنة والتراجع، ومدخلاتها 128×128.

| التجربة | دقة التحقق | دقة الاختبار |
| --- | ---: | ---: |
| CNN الأولية | 72.52% | 72.94% |
| استكمال التدريب بمعدل تعلم أقل | 74.50% | 73.93% |
| CNN بعد تنظيف التداخل وإعادة التدريب من الصفر | **75.50%** | **76.90%** |
| ResNet18 / ImageNet السابقة | 83.77% | 84.16% |
| ResNet18 مع Fine-Tuning إضافي | 86.09% | 84.82% |
| **دمج EfficientNet-B0 وResNet18 — النشطة** | **89.40%** | **88.45%** |
| MobileNetV2 / ImageNet | لم يُنفذ التدريب الفعلي في هذه البيئة | غير مقاسة |

نتائج النموذج المدمج النشط على **303 صور اختبار**:

| الفئة الفعلية ↓ / التنبؤ → | مزيف | حقيقي |
| --- | ---: | ---: |
| مزيف | 141 | 10 |
| حقيقي | 25 | 127 |

ارتفعت الدقة من 84.82% إلى **88.45%**، وانخفض قبول المزيف من 21 إلى **10**،
وبقي رفض الحقيقي عند 25. **لم نصل إلى 90%**: صُنفت 268 صورة صحيحة من 303،
بينما يتطلب بلوغ 90% على هذه المجموعة 273 صورة صحيحة على الأقل.
لم تُضف صور ذهب، ولم تُغيّر التصنيفات أو عضوية التقسيمات أو تُستبعد حالات من الاختبار.
الاختبار سابق الاستخدام في التطوير؛ النتيجة لا تثبت الأداء على مصدر صور مستقل.

التفاصيل: [تقرير استهداف الدقة](reports/Accuracy-target.md)،
[تقرير Fine-Tuning](reports/Fine-tuning.md)،
[تقرير ResNet18](reports/ResNet18-transfer.md)،
[تقرير هندسة البيانات](reports/Data-engineering.md)،
[مقاييس الاختبار الحالية](reports/selected_test_metrics.json)،
[قرار اختيار GoldCNN السابقة](reports/engineering_selection.json).

## التشغيل السريع

### المتطلبات

- Python 3.10 أو أحدث؛ اختُبر المشروع على Python 3.12 وLinux باستخدام CPU.
- Git واتصال لتنزيل المكتبات؛ لا يلزم GPU أو مفتاح API لتشغيل النموذج المرفق.

### التثبيت

```bash
git clone https://github.com/Aishaas1/AI-porject-new.git
cd AI-porject-new
python3 -m venv .venv
source .venv/bin/activate

# نسخة PyTorch المخصصة لـ CPU لتجنب تنزيل مكتبات CUDA غير اللازمة
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

python app.py
```

على Windows، فعّل البيئة باستخدام `.venv\Scripts\activate` بدل أمر `source`.
بعد التشغيل، افتح عنوان الخادم المحلي على المنفذ **5000** في متصفح جهاز التشغيل.
خادم Flask المرفق للتطوير؛ النشر العام يحتاج إعداد خادم إنتاج ومصادقة مناسبة.

### النموذج المرفق

- الأوزان النشطة: [`artifacts/gold_selected.pt`](artifacts/gold_selected.pt).
- ResNet18 المضبوطة السابقة: [`artifacts/gold_finetuned.pt`](artifacts/gold_finetuned.pt).
- ResNet18 السابقة: [`artifacts/gold_transfer.pt`](artifacts/gold_transfer.pt).
- أوزان المقارنة السابقة: [`artifacts/gold_engineered.pt`](artifacts/gold_engineered.pt).
- إعدادات اختيار النموذج النشط: [`artifacts/gold_selected.json`](artifacts/gold_selected.json).
- اختيار النسخة السابقة: [`artifacts/gold_engineered.json`](artifacts/gold_engineered.json).
- يحمّل التطبيق الدمج المختار أولًا، ثم يعود إلى النماذج السابقة عند غيابه.
- حجم ملف الدمج نحو 59 ميبيبايت، ولا يتطلب الاتصال بـ ImageNet عند التشغيل.
- حجم أوزان ResNet18 نحو 43 ميبيبايت؛ لا يحتاج تشغيلها إلى إعادة تنزيل ImageNet.
- لا يتضمن المستودع صور التدريب أو بقية الأوزان التجريبية.
- لا يُحفظ محتوى الصور المرفوعة على القرص.

لتحديد نموذج آخر صراحةً:

```bash
GOLD_MODEL_PATH=artifacts/gold_engineered.pt python app.py
```

عند غياب الأوزان يعرض التطبيق «بانتظار بيانات التدريب» ولا يُنشئ تصنيفات عشوائية.
استخدم أوزانًا من مصادر موثوقة فقط.

## ملف Google Colab الجاهز

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aishaas1/AI-porject-new/blob/main/notebooks/Gold_Classifier_Colab.ipynb)

افتح [Gold_Classifier_Colab.ipynb](notebooks/Gold_Classifier_Colab.ipynb) وشغّل خلاياه بالترتيب.
يتضمن تنزيل المشروع، وتثبيت مكتباته في بيئة Python معزولة، والتحقق من بصمة الأوزان،
وتشغيل الخادم، ورابط ngrok، واختبار صورة داخل الدفتر دون رابط عام.
لا يلزم تدريب أو GPU لتشغيل النموذج المرفق. احفظ رمز ngrok في أسرار Colab باسم `NGROK_AUTHTOKEN` وفعّل وصول الدفتر إليه؛
لا تضعه في الكود. يقرأ الدفتر السر مباشرة لتفادي اختلاف سلوك `getpass` بين الواجهات.

إعادة التدريب والتقييم وتنزيل حزمة النتائج متاحة في أقسام اختيارية معطلة افتراضيًا.
تستخدم إعادة التدريب نسخة منفصلة للمستودع كي لا تغيّر أوزان الواجهة العاملة.
خلية الإيقاف معطلة افتراضيًا أيضًا حتى لا تُغلق الخدمة مباشرةً عند «تشغيل الكل»؛
فعّلها عند الانتهاء. التشغيل قائم على CPU، والنفق يتوقف بانتهاء جلسة Colab.

تحققنا محليًا من صيغة الدفتر، والتثبيت في بيئة فارغة، وفحص الأوزان، وتشغيل الخادم،
ورفع صورة اختبار، وإعادة التشغيل والإيقاف. إعادة التدريب الاختيارية لم تُكرر ضمن فحص الدفتر. لم نجر جلسة Colab حية أو نفتح نفق ngrok عام أثناء التحقق؛
هذان يعتمدان على جلسة المستخدم ورمز حسابه. النتيجة المسجلة للنموذج **88.45%**؛
الدفتر لا يضمن الوصول إلى 90% عند إعادة التدريب.

## تشغيل الواجهة على Google Colab

لا تحتاج إلى إعادة تدريب النموذج أو تفعيل GPU. افتح دفتر Colab جديدًا وشغّل
الخلايا التالية بالترتيب. عنوان `127.0.0.1` وحده لا يفتح واجهة Colab في متصفحك؛
نستخدم ngrok لإنشاء رابط يصل إلى الخادم داخل الجلسة.

### 1. تنزيل المشروع وتثبيت المكتبات

```python
!git clone https://github.com/Aishaas1/AI-porject-new.git
%cd /content/AI-porject-new
!pip -q install -r requirements.txt pyngrok requests
```

استخدم جلسة جديدة لتجنب التعارض مع إصدارات PyTorch المثبتة سابقًا. إذا طلب
Colab إعادة تشغيل الجلسة بعد التثبيت، أعد تشغيلها ثم نفّذ هذه الخلية قبل المتابعة:

```python
%cd /content/AI-porject-new
```

إذا كان المستودع موجودًا بالفعل، انتقل إلى مجلده وحدّثه بواسطة
`!git pull origin main` بدل تكرار أمر التنزيل، مع الحفاظ على تعديلاتك المحلية.

### 2. إعداد ngrok باستخدام أسرار Colab

أنشئ حسابًا في [ngrok](https://dashboard.ngrok.com/signup)، وانسخ **Authtoken**.
افتح أيقونة المفتاح **Secrets / الأسرار** في Colab، وأضف سرًا اسمه
`NGROK_AUTHTOKEN` وقيمته الرمز، ثم فعّل **Notebook access**.

```python
from google.colab import userdata
from pyngrok import ngrok

def configure_ngrok_secret():
    try:
        token = userdata.get("NGROK_AUTHTOKEN")
    except Exception:
        raise RuntimeError("أضف السر NGROK_AUTHTOKEN وفعّل وصول الدفتر إليه.") from None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("السر يجب أن يكون نصًا غير فارغ؛ افحصه في إعدادات الأسرار.")
    try:
        ngrok.set_auth_token(token.strip())
    except Exception:
        raise RuntimeError("تعذر إعداد رمز ngrok؛ تحقق منه في الأسرار دون طباعته.") from None

configure_ngrok_secret()
```

لا ترسل الرمز في المحادثة ولا تحفظه داخل كود الدفتر. إذا ظهر خطأ
`AttributeError: 'dict' object has no attribute 'strip'` في نسخة قديمة، استبدل
خلية `getpass` بهذه الخلية؛ لا يلزم إعادة التثبيت أو التدريب.

### 3. تشغيل التطبيق وفتح رابط الواجهة

شغّل هذه الخلية مرة واحدة. قبل تكرارها، أوقف الخادم السابق باستخدام خلية الإيقاف
أدناه حتى لا يتعارض المنفذ 5000.

```python
import os
import subprocess
import time
import requests

os.environ["GOLD_MODEL_PATH"] = "artifacts/gold_selected.pt"

log = open("/content/gold_app.log", "w")
server = subprocess.Popen(
    ["python", "app.py"],
    stdout=log,
    stderr=subprocess.STDOUT,
    env=os.environ.copy()
)

ready = False
for _ in range(60):
    if server.poll() is not None:
        break
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=2)
        if response.ok:
            health = response.json()
            if health.get("model_ready") and health.get("checkpoint") == "gold_selected.pt":
                ready = True
                break
    except requests.RequestException:
        pass
    time.sleep(1)

if not ready:
    if server.poll() is None:
        server.terminate()
        server.wait(timeout=10)
    log.close()
    print(open("/content/gold_app.log").read())
    raise RuntimeError("لم يصبح الخادم جاهزًا؛ راجع السجل أعلاه قبل إعادة المحاولة.")

try:
    tunnel = ngrok.connect(5000, "http")
    print("افتح الواجهة:", tunnel.public_url)
except Exception:
    server.terminate()
    server.wait(timeout=10)
    log.close()
    raise
```

افتح الرابط المطبوع، ثم اختر صورة واضغط **«تحليل الصورة»**. إذا عرض ngrok
صفحة تنبيه، تابع إلى الموقع. التطبيق يحمّل النموذج المدمج المرفق تلقائيًا.

الرابط متاح لمن يملكه، والواجهة لا تتضمن تسجيل دخول؛ لا تشاركه علنًا أو ترفع
صورًا حساسة. يتوقف الوصول عند إغلاق النفق أو انتهاء جلسة Colab، ولا يُعد هذا
إعداد استضافة دائمة أو نشرًا إنتاجيًا.

### 4. إيقاف الواجهة

بعد نجاح تشغيل الخلية السابقة، أوقف النفق والخادم عند الانتهاء:

```python
ngrok.disconnect(tunnel.public_url)
server.terminate()
server.wait(timeout=10)
log.close()
```

إذا فشل التشغيل، يظهر سجل الخادم في الخلية. راجع أخطاء التثبيت أو غياب الأوزان
قبل المتابعة؛ لا يلزم تدريب النموذج مجددًا لحل مشكلة تشغيل الواجهة.

## واجهة HTTP

| المسار | الطريقة | الوظيفة |
| --- | --- | --- |
| `/` | GET | الواجهة العربية |
| `/health` | GET | حالة الخدمة واسم النموذج المحمّل |
| `/predict` | POST | تحليل صورة ضمن حقل `image` بصيغة multipart |

مثال من جهاز تشغيل الخادم:

```bash
curl http://127.0.0.1:5000/health
curl -X POST -F "image=@sample.jpg" http://127.0.0.1:5000/predict
```

يرجع التحليل `label` و`real_score` و`fake_score` ورسالة توضيح للنتيجة.
درجات softmax **غير معايرة**؛ ليست احتمالات مثبتة لأصالة الذهب.
حد الطلب 8 ميجابايت، وحد أبعاد الصورة 20 مليون بكسل.
يرجع الخادم `400` للصورة غير المقبولة، و`413` لتجاوز حجم الطلب، و`503` لغياب النموذج.

## البيانات وإعادة إنتاج التجارب

المصدر هو [GoldNet — Zobeir Raisi](https://github.com/zobeirraisi/GoldNet)،
عند الإصدار `f206c941a39d3126b6ce867e01abfad87e10f525`.
تُستخدم التصنيفات كما قدمها المصدر؛ لم يُجر فحص مادي مستقل للقطع.

### تجهيز GoldNet

من جذر المشروع، بعد تفعيل البيئة:

```bash
git clone https://github.com/zobeirraisi/GoldNet.git /tmp/GoldNet
git -C /tmp/GoldNet checkout f206c941a39d3126b6ce867e01abfad87e10f525
python prepare_goldnet.py --source /tmp/GoldNet --destination data/goldnet
python audit_data.py --data data/goldnet
```

اختر وجهة جديدة إذا كانت البيانات مجهزة بالفعل؛ سكربت التحضير يرفض استبدالها.
البنية الناتجة:

```text
data/goldnet/
├── train/
│   ├── fake/
│   └── real/
├── val/
│   ├── fake/
│   └── real/
├── test/
│   ├── fake/
│   └── real/
└── manifest.json
```

من 2,127 صورة أصلية، استُبعدت 116 نسخة متطابقة في البكسلات قبل التقسيم.
بعد مراجعة التكرار القريب استُبعدت خمس صور إضافية **من التحميل للتدريب فقط**،
دون حذف الملفات أو تعديل التصنيفات أو تغيير مجموعتي التحقق والاختبار.

| المجموعة المستخدمة | مزيف | حقيقي | الإجمالي |
| --- | ---: | ---: | ---: |
| التدريب المنظف | 702 | 699 | 1,401 |
| التحقق | 151 | 151 | 302 |
| الاختبار | 151 | 152 | 303 |

التقسيم طبقي ببذرة 42. تُستبعد أمثلة مجلد `pairs` في المصدر.
تُحفظ البصمات والتقسيمات في [goldnet_split.json](reports/goldnet_split.json)،
والاستبعادات المراجعة في [engineering_exclusions.json](reports/engineering_exclusions.json).
عتبة dHash ترشح صورًا للمراجعة، ولا تثبت اكتشاف جميع الصور المتشابهة.
الصورة المنخفضة الإضاءة التي رصدها التدقيق بقيت في التحقق دون استبعاد.

### تدريب النسخة المنظفة واختيار الأفضل

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train_engineered.py --data data/goldnet --epochs 30
python select_engineered.py
OMP_NUM_THREADS=4 python evaluate.py \
  --data data/goldnet/test \
  --model artifacts/gold_engineered.pt \
  --output reports/engineered_test_metrics.json
```

تقارن هذه الأوامر ثلاث تجارب من الصفر: التجهيز الأساسي، والحواف الانعكاسية مع
زيادة التنوع، ثم إضافة ضبط تباين الإضاءة. تستخدم جميعها العضوية المنظفة نفسها.
زيادة التنوع تشمل انعكاسًا أفقيًا ودورانًا خفيفًا وتغييرات بسيطة في الموضع والإضاءة
والتباين، وتطبق على التدريب فقط. تقييم التحقق والاختبار حتمي.

يحفظ التدريب أفضل الأوزان وفق دقة التحقق، مع خفض معدل التعلم عند توقف التحسن
وإيقاف مبكر. تُحسم التعادلات لصالح التجربة الأبسط والأسبق.
**في النتائج الحالية لم تتفوق زيادة التنوع أو معالجة الإضاءة على التجهيز الأساسي**،
لذلك لم تُفعّل في GoldCNN المختارة من هذه التجارب. هذه المقارنة تاريخية؛
مسار Fine-Tuning الأحدث يستخدم انعكاسًا أفقيًا أثناء التدريب كما هو موثق أدناه.

الأوامر تعيد توليد ملفات التجارب والتقارير، وقد تختلف النتائج قليلًا بين الأجهزة
وإصدارات المكتبات. راجع قائمة الاستبعادات قبل استخدام بيانات مختلفة؛ سكربت الاختيار
مخصص لأعداد تقسيم GoldNet الحالي. استخدم اختبارًا خارجيًا جديدًا لأي ادعاء نهائي بالتعميم.

### مسارات CNN السابقة

```bash
python train.py --data data/goldnet --output artifacts/gold.pt --epochs 10 --batch-size 32
python train.py --data data/goldnet --output artifacts/gold_refined.pt \
  --resume artifacts/gold.pt --learning-rate 0.0001 --epochs 20 --batch-size 32
```

هذه أوامر التجارب التاريخية، ولا تطبق استبعادات التكرار القريب الجديدة؛ استخدم
`train_engineered.py` لإعادة إنتاج المسار المنظف. خيار `--resume` يستعيد الأوزان
ويبدأ محسن Adam جديدًا، ولا يستعيد حالة المحسن السابقة.

## MobileNetV2 ونقل التعلم

يوفر المشروع Keras 3 بخلفية PyTorch؛ لا يلزم TensorFlow لهذا المسار.

1. تحميل `MobileNetV2(weights="imagenet", include_top=False)` بمدخلات 224×224.
2. تجميد العمود الفقري وتدريب رأس التصنيف ثلاث دورات بمعدل `0.001`.
3. فتح آخر 30 طبقة مع إبقاء كل طبقات `BatchNormalization` مجمدة.
4. استدعاء العمود الفقري بـ `training=False` لحماية إحصاءات BatchNormalization.
5. **إعادة `compile` بمحسن Adam جديد ومعدل `1e-5`**، ثم سبع دورات ضبط دقيق.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train_mobilenet.py \
  --data data/goldnet --head-epochs 3 --fine-epochs 7 --batch-size 32
OMP_NUM_THREADS=4 python evaluate_mobilenet.py
GOLD_MODEL_PATH=artifacts/gold_mobilenet.keras python app.py
```

تنزيل ImageNet الأول يحتاج HTTPS إلى `storage.googleapis.com`.
تعطل التنزيل في البيئة التي أُجريت فيها التجارب، لذلك **لا توجد نتيجة تدريب
MobileNetV2 مقاسة ضمن هذا الإصدار**. اختبارات هذا المسار تستخدم أوزانًا عشوائية
مؤقتة لاختبار التنفيذ فقط، ولا تُعرض على أنها نتائج ImageNet.

مسار MobileNetV2 الحالي يقرأ التقسيم الأساسي مباشرة ولا يطبق قائمة الاستبعادات
القريبة الخاصة بالمسار المنظف؛ يجب تطبيقها أو تجهيز تقسيم نظيف قبل تجربة مقارنة جديدة.
لا تقارن أي نتيجة تدريب جديدة دون التحقق من عضوية مجموعاتها.

يمكن استيراد ملف `.keras` من Colab بعد التأكد من البنية وترتيب الفئات والمعالجة:
المسار المرفق يتوقع رأس softmax ثنائيًا بالترتيب `fake, real`، وحجم 224×224،
ومعالجة `[0,255] → [-1,1]` **داخل النموذج**. ملف Colab بإعدادات مختلفة يحتاج
تكييف كود التحميل قبل استخدامه؛ صيغة الملف وحدها لا تضمن التوافق.

## هيكل المشروع

```text
.
├── app.py                       # واجهة Flask ومسارات HTTP
├── model.py                     # تحميل GoldCNN أو ResNet18 ومعالجتها
├── transfer_model.py            # ResNet18 والتجميد ومعالجة ImageNet
├── train_transfer.py            # تدريب الرأس ثم ضبط آخر كتلة
├── fine_tune.py                 # ضبط إضافي لآخر كتلتين
├── mobilenet.py                 # MobileNetV2 والتجميد وإعادة التجميع
├── prepare_goldnet.py            # تجهيز البيانات والتقسيم والبصمات
├── audit_data.py                 # تدقيق الجودة والتكرار القريب
├── data_engineering.py           # المعالجة الحتمية وزيادة تنوع التدريب
├── train_engineered.py           # تجارب التدريب على العضوية المنظفة
├── select_engineered.py          # اختيار النموذج وفق التحقق فقط
├── train.py / evaluate.py        # تدريب CNN وتقييمها
├── train_mobilenet.py            # نقل التعلم على مرحلتين
├── evaluate_mobilenet.py         # تقييم Keras
├── efficient_model.py           # EfficientNet والدمج ومتوسط المناظر
├── train_efficient.py           # تدريب EfficientNet بدقة 320×320
├── adapt_bn.py                  # مواءمة إحصاءات التطبيع بالتدريب فقط
├── select_candidate.py          # اختيار الدمج والعتبة من التحقق
├── artifacts/gold_selected.pt    # النموذج المدمج النشط
├── artifacts/gold_finetuned.pt   # نسخة ResNet18 السابقة
├── artifacts/gold_transfer.pt    # ResNet18 السابقة للتراجع
├── artifacts/gold_engineered.pt  # GoldCNN المنظفة السابقة
├── templates/index.html         # الواجهة العربية
├── reports/                     # التوثيق والتدقيق والمقاييس والتقسيمات
├── tests/                       # الاختبارات
└── requirements.txt
```

## الاختبارات

```bash
python -m pip install pytest==8.4.2
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

نجحت **17 اختبارًا** في البيئة المستخدمة. تغطي دورة تدريب اصطناعية، والتحقق من
تجميد طبقات MobileNetV2، وحفظ الأوزان وتحميلها، والمعالجة والاستبعادات، ورفع الصور.
قد تظهر تحذيرات إهمال من توافق Keras/PyTorch مع NumPy؛ لا تُفسر الاختبارات
الاصطناعية على أنها قياس لدقة تصنيف الذهب.

## حدود الاستخدام

- قد تكون القطع المطلية مطابقة بصريًا للذهب الحقيقي؛ الصورة لا تكشف تركيب المعدن.
- لا يستنتج التطبيق العيار أو الوزن أو السعر.
- الإضاءة والكاميرا والخلفية والسوق الذي جاءت منه الصور قد تؤثر في النتائج.
- لم تُعاير درجات التصنيف ولم يُجر اختبار مستقل على سوق أو مصدر صور جديد.
- تحسن الدقة الإجمالية لا يلغي خطر قبول قطع مزيفة؛ راجع مصفوفة الالتباس قبل الاستخدام.

## مصدر البيانات والترخيص

صور GoldNet تحت ترخيص [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)،
مع نسب العمل إلى **Zobeir Raisi**. لا تُعاد استضافة الصور في هذا المستودع؛
يوجد إشعار المصدر في [GoldNet-LICENSE-DATA](reports/GoldNet-LICENSE-DATA).
التحضير يزيل التكرار ويحدد تقسيمات واستبعادات؛ الصور الأصلية لا تُعدل على القرص.
المرجع كما يورده مستودع البيانات:

> Z. Raisi, “GoldFormer: A Texture-Aware Vision Transformer-Based Algorithm for
> Detecting Near-Identical Images,” *Algorithms*, 2026. DOI: 10.3390/a19070530.

ترخيص البيانات لا يمنح تلقائيًا ترخيصًا لشيفرة هذا المشروع؛ لم يُضف ترخيص مستقل
للشيفرة ضمن هذا الإصدار.


## نقل التعلم باستخدام ResNet18

أضيف مسار CNN مدرب مسبقًا على ImageNet من توزيعة PyTorch الرسمية، مختلف عن
MobileNetV2 الخاص بـ Keras. يستخدم الصور الكاملة بدقة 224×224 مع تطبيع ImageNet،
وعضوية التدريب المنظفة نفسها، ثم تدريب الرأس وضبط آخر كتلة بمعدل `1e-5`،
مع تجميد جميع معاملات وإحصاءات BatchNormalization.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train_transfer.py
OMP_NUM_THREADS=4 python evaluate.py --data data/goldnet/test --model artifacts/gold_transfer.pt --output reports/transfer_test_metrics.json
GOLD_MODEL_PATH=artifacts/gold_transfer.pt python app.py
```

يتطلب التدريب الأول تنزيل أوزان ResNet18 من `download.pytorch.org`،
ويتحقق torchvision من البصمة. تحميل الأوزان المحلية لاحقًا لا يتطلب الاتصال.
الاختيار عبر المرحلتين يستخدم دقة التحقق، وتكسر التعادلات بقلة أخطاء المزيف
المصنف كحقيقي على التحقق. لا يقرأ التدريب مجموعة الاختبار.
تفاصيل البروتوكول والنتائج في [تقرير ResNet18](reports/ResNet18-transfer.md).

## Fine-Tuning إضافي لآخر كتلتين من ResNet18

لتجربة ضبط أوسع انطلاقًا من أفضل أوزان ResNet18 الموجودة، مع الحفاظ على الأصل:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python fine_tune.py \
  --initial artifacts/gold_transfer.pt \
  --output artifacts/gold_finetuned.pt \
  --epochs 8
```

يفتح هذا المسار `layer3` و`layer4` والرأس، ويبقي جميع طبقات BatchNormalization
مجمدة، بمعدلات تعلم متدرجة `2e-6` و`5e-6` و`1e-5`. يطبق انعكاسًا أفقيًا على
التدريب فقط و`label_smoothing=0.05`، مع خفض كوسيني لمعدلات التعلم وقص معيار
التدرج. يعتمد الاختيار والإيقاف المبكر على التحقق فقط، ويدخل النموذج الأصلي
ضمن المرشحين لضمان عدم فقد أفضل نتيجة تحقق.

إعدادات المعالجة والتقسيم والاستبعادات هي نفسها لمسار ResNet18 الحالي.
لتقييم ملف التجربة بعد اكتمالها:

```bash
OMP_NUM_THREADS=4 python evaluate.py --data data/goldnet/test --model artifacts/gold_finetuned.pt --output reports/finetuned_test_metrics.json
```

هذا تقييم تجريبي على اختبار سابق الاستخدام؛ لا يُستخدم لتعديل عتبة التصنيف.
راجع [تقرير Fine-Tuning](reports/Fine-tuning.md) لمعرفة النتيجة والنسخة النشطة.


نتيجة Fine-Tuning الإضافي: **86.09% تحقق، و84.82% اختبار**.
النسخة `gold_finetuned.pt` هي الافتراضية عند غياب `gold_selected.pt`. التحسن في الاختبار محدود
(صورتان إضافيتان)، مع زيادة قبول المزيف من 18 إلى 21؛ راجع تقرير المقارنة.
للتراجع إلى ResNet18 السابقة استخدم:

```bash
GOLD_MODEL_PATH=artifacts/gold_transfer.pt python app.py
```

## تجربة استهداف 90% بالبيانات الحالية

يتوفر مسار تجريبي لـ EfficientNet-B0 بدقة 320×320 ودمج احتمالاتها مع ResNet18،
دون إضافة صور ذهب أو تغيير التصنيفات أو نقل صور بين المجموعات. أوزان ImageNet
الأولية تُحمّل من PyTorch، لكنها لا تضيف صورًا إلى مجموعة تدريب الذهب.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train_efficient.py
OMP_NUM_THREADS=4 python adapt_bn.py
OMP_NUM_THREADS=4 python select_candidate.py
OMP_NUM_THREADS=4 python evaluate.py --data data/goldnet/test --model artifacts/gold_selected.pt --output reports/selected_test_metrics.json
```

الاختيار يقرأ التحقق فقط: النموذجان المنفردان وثلاثة أوزان دمج، مع مقارنة المنظر الأصلي ومتوسطه مع الانعكاس الأفقي وخمس عتبات
قرار محددة مسبقًا. لا تُختار العتبة من نتائج الاختبار. تعرض الواجهة عتبة القرار
المحفوظة إلى جانب الدرجات؛ إذا كانت مختلفة عن 50% فقد يختلف التصنيف عن الفئة
صاحبة الدرجة الأعلى. الدرجات الخام لا تتحول إلى احتمالات معايرة بتغيير العتبة.

يمكن تشغيل مرشح محفوظ صراحةً باستخدام:

```bash
GOLD_MODEL_PATH=artifacts/gold_selected.pt python app.py
```

هذه الأوامر لا تضمن الوصول إلى 90%. التقييم النهائي يشمل جميع صور الاختبار
دون استبعاد الحالات الصعبة أو إضافة نتيجة «غير حاسم» لرفع الدقة حسابيًا.
راجع [تقرير التجربة](reports/Accuracy-target.md) ونتائج الاختيار قبل الاعتماد
على أي رقم. الاختبار سابق الاستخدام، ومقارنة 50 إعدادًا على التحقق تزيد احتمال
الملاءمة الزائدة؛ يلزم اختبار خارجي مستقل لادعاء التعميم.


**نتيجة تجربة الاستهداف:** 89.40% تحقق و88.45% اختبار؛ **هدف 90% لم يتحقق**.
النموذج `gold_selected.pt` هو الافتراضي، وعتبته 55%. التصنيف «حقيقي» يحتاج
درجة تتجاوز هذه العتبة؛ النسب ليست احتمالات معايرة. يوجد عبء استدلال أكبر بسبب
تشغيل شبكتين ومنظرين لكل صورة. للتراجع إلى النسخة السابقة:

```bash
GOLD_MODEL_PATH=artifacts/gold_finetuned.pt python app.py
```
