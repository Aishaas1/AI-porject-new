# ميزان الذهب | Gold Image Classifier

تطبيق بواجهة عربية لتصنيف **صور الذهب إلى حقيقي ومزيف** باستخدام التعلم العميق،
مع مسار متكامل لتدقيق البيانات، والتدريب، والتقييم، ومقارنة التجارب.

**أفضل نتيجة حالية: 84.16% على اختبار المشروع باستخدام ResNet18 / ImageNet.**

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

النسخة النشطة هي **ResNet18 بنقل التعلم من ImageNet** بعد التدريب على البيانات المنظفة، وليست MobileNetV2.
حجم مدخلاتها 224×224، وترتيب الفئات: `fake = 0`، و`real = 1`.
تظل GoldCNN الصغيرة السابقة مرفقة للمقارنة والتراجع، ومدخلاتها 128×128.

| التجربة | دقة التحقق | دقة الاختبار |
| --- | ---: | ---: |
| CNN الأولية | 72.52% | 72.94% |
| استكمال التدريب بمعدل تعلم أقل | 74.50% | 73.93% |
| CNN بعد تنظيف التداخل وإعادة التدريب من الصفر | **75.50%** | **76.90%** |
| **ResNet18 / ImageNet — النشطة** | **83.77%** | **84.16%** |
| MobileNetV2 / ImageNet | لم يُنفذ التدريب الفعلي في هذه البيئة | غير مقاسة |

نتائج ResNet18 النشطة على **303 صور اختبار**:

| الفئة الفعلية ↓ / التنبؤ → | مزيف | حقيقي |
| --- | ---: | ---: |
| مزيف | 133 | 18 |
| حقيقي | 30 | 122 |

تحسنت الدقة من 76.90% إلى **84.16%**، وانخفضت حالات اعتبار المزيف حقيقيًا
من 32 إلى **18**. كشفت مراجعة البيانات مشهدًا متداخلًا بين التدريب والاختبار في
التجارب القديمة؛ استُبعد من جميع تجارب التدريب المنظفة، بما فيها ResNet18.
مجموعة الاختبار مستخدمة في جولات تطوير سابقة، ولا تمثل اختبارًا خارجيًا أعمى؛
لا تعني النتيجة ضمان الدقة على مصادر صور جديدة.

التفاصيل: [تقرير ResNet18](reports/ResNet18-transfer.md)،
[تقرير هندسة البيانات](reports/Data-engineering.md)،
[مقاييس الاختبار](reports/engineered_test_metrics.json)،
[قرار اختيار النموذج](reports/engineering_selection.json).

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

- الأوزان النشطة: [`artifacts/gold_transfer.pt`](artifacts/gold_transfer.pt).
- أوزان المقارنة السابقة: [`artifacts/gold_engineered.pt`](artifacts/gold_engineered.pt).
- مقاييس النموذج النشط: [`artifacts/gold_transfer.json`](artifacts/gold_transfer.json).
- اختيار النسخة السابقة: [`artifacts/gold_engineered.json`](artifacts/gold_engineered.json).
- يحمّل التطبيق ResNet18 تلقائيًا عند توفرها، ثم يرجع إلى GoldCNN عند غيابها.
- حجم أوزان ResNet18 نحو 43 ميبيبايت؛ لا يحتاج تشغيلها إلى إعادة تنزيل ImageNet.
- لا يتضمن المستودع صور التدريب أو بقية الأوزان التجريبية.
- لا يُحفظ محتوى الصور المرفوعة على القرص.

لتحديد نموذج آخر صراحةً:

```bash
GOLD_MODEL_PATH=artifacts/gold_engineered.pt python app.py
```

عند غياب الأوزان يعرض التطبيق «بانتظار بيانات التدريب» ولا يُنشئ تصنيفات عشوائية.
استخدم أوزانًا من مصادر موثوقة فقط.

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
لذلك ليست مفعلة في النموذج المرفق. التحسن لا يُنسب إلى هذه المعالجات.

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
├── model.py                     # GoldCNN وتحميل أوزانها
├── mobilenet.py                 # MobileNetV2 والتجميد وإعادة التجميع
├── prepare_goldnet.py            # تجهيز البيانات والتقسيم والبصمات
├── audit_data.py                 # تدقيق الجودة والتكرار القريب
├── data_engineering.py           # المعالجة الحتمية وزيادة تنوع التدريب
├── train_engineered.py           # تجارب التدريب على العضوية المنظفة
├── select_engineered.py          # اختيار النموذج وفق التحقق فقط
├── train.py / evaluate.py        # تدريب CNN وتقييمها
├── train_mobilenet.py            # نقل التعلم على مرحلتين
├── evaluate_mobilenet.py         # تقييم Keras
├── artifacts/gold_engineered.pt  # النموذج المختار المرفق
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

نجحت **11 اختبارًا** في البيئة المستخدمة. تغطي دورة تدريب اصطناعية، والتحقق من
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
