---
title: ميزان — Gold Vision
emoji: 💎
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: gradio_app.py
pinned: false
---

# ميزان — واجهة Gradio العربية

قالب README لـ Hugging Face Spaces. انسخه إلى جذر Space مع ملفات المشروع
إذا أردت استضافة مستقلة عن جلسة Colab. يحتاج النشر إلى حسابك وصلاحياته؛ لم
يُنشأ Space تلقائيًا ضمن هذا المشروع.

انسخ أيضًا محتوى `deploy/gradio/requirements.txt` إلى `requirements.txt` في
جذر Space، وارفع `gradio_app.py` و`app.py` وملفات النماذج البرمجية ومجلدي
`static/` و`artifacts/` (الأوزان المختارة المرفقة). لا يلزم رفع صور التدريب.
ينفّذ SDK ملف `gradio_app.py` على المنفذ 7860 دون `--share`.

الشيفرة والمكتبات: https://github.com/Aishaas1/AI-porject-new
نتيجة اختبار النموذج المرفق 88.45%، وعتبة الحقيقي 55%. تقدير بصري لا يثبت
أصالة المعدن أو نقاءه؛ الدرجات غير معايرة. يحفظ Gradio ملفات رفع مؤقتة تُحذف
دوريًا، لذا لا تنشر صورًا حساسة على الواجهة العامة.
