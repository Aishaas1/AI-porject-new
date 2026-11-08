"""Arabic Gradio interface; shares the existing Flask inference/validation path."""
import argparse
import html
import os
from pathlib import Path
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parent.parent/'hf-cache'))
import gradio as gr
from PIL import Image,ImageOps,UnidentifiedImageError
from app import create_app

ROOT=Path(__file__).resolve().parent
IDLE='''<div class="empty-result"><div class="empty-ring">◇</div><h3>كل قراءة تبدأ بصورة</h3><p>أضف صورة واضحة للقطعة، ثم اضغط «تحليل الصورة» لعرض تقدير النموذج.</p></div>'''


def error_card(message):
    return '<div class="error-result"><h3>تعذّر إكمال التحليل</h3><p>'+html.escape(str(message))+'</p><p>اختر صورة أخرى وحاول مجددًا.</p></div>'


def result_card(result):
    real=float(result['real_score']);fake=float(result['fake_score']);threshold=float(result['decision_threshold'])*100
    is_real=result['label']=='حقيقي'
    title='يميل التقدير إلى الحقيقي' if is_real else 'يميل التقدير إلى المزيف'
    tag='حقيقي' if is_real else 'مزيف'
    circumference=2*3.14159265359*58
    offset=circumference*(1-real/100)
    return f'''<div class="result-head"><div><div class="eyebrow">قراءة النموذج</div><h3 class="result-label">{title}</h3></div><span class="result-tag {'fake' if not is_real else ''}">{tag}</span></div>
    <p class="result-caption">تقدير بصري للصورة المرفوعة، وليس حكمًا نهائيًا على أصالة القطعة.</p>
    <div class="gauge"><svg viewBox="0 0 144 144" aria-hidden="true"><circle cx="72" cy="72" r="58" fill="none" stroke="#294237" stroke-width="7"/><circle cx="72" cy="72" r="58" fill="none" stroke="#dfbf7b" stroke-width="7" stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"/></svg><div class="number">{real:.1f}%<small>درجة الحقيقي</small></div></div>
    <div class="score-row"><span>حقيقي</span><b dir="ltr">{real:.2f}%</b></div><div class="track"><div class="fill" style="width:{real}%"></div></div>
    <div class="score-row"><span>مزيف</span><b dir="ltr">{fake:.2f}%</b></div><div class="track"><div class="fill fake" style="width:{fake}%"></div></div>
    <div class="threshold"><span>عتبة تصنيف الحقيقي</span><b>{threshold:.0f}%</b></div><p class="result-caption">الدرجات غير معايرة. يُصنَّف الحقيقي عند تجاوز عتبته، وقد تختلف النتيجة عن الفئة صاحبة الدرجة الأعلى.</p>'''


class Predictor:
    def __init__(self,model_path=None):
        self.flask_app=create_app(model_path)
        self.health=self.flask_app.test_client().get('/health').get_json()
    def preview(self,path):
        if not path:return gr.update(value=None,visible=False),IDLE
        try:
            path=Path(path)
            if path.stat().st_size>8*1024*1024:raise ValueError('حجم الصورة يتجاوز 8 ميجابايت')
            with Image.open(path) as image:
                if image.format not in {'JPEG','PNG','WEBP'} or image.width*image.height>20_000_000:
                    raise ValueError('استخدم JPG أو PNG أو WebP حتى 20 مليون بكسل')
                image=ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((700,450))
                preview=image.copy()
            return gr.update(value=preview,visible=True),IDLE
        except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError) as error:
            return gr.update(value=None,visible=False),error_card(error)
    def predict(self,path):
        if not path:return error_card('اختر صورة أولًا')
        try:
            path=Path(path)
            if path.stat().st_size>8*1024*1024:return error_card('حجم الصورة يتجاوز 8 ميجابايت')
            with path.open('rb') as file:
                response=self.flask_app.test_client().post('/predict',data={'image':(file,path.name)})
            result=response.get_json()
            if response.status_code!=200:return error_card(result.get('error','تعذر تحليل الصورة'))
            return result_card(result)
        except (OSError,ValueError):
            return error_card('تعذر قراءة الصورة؛ اختر ملفًا سليمًا وحاول مجددًا')


def build_demo(model_path=None):
    predictor=Predictor(model_path)
    ready=predictor.health['model_ready']
    css=(ROOT/'static/gradio.css').read_text()
    theme=gr.themes.Base(primary_hue='amber',neutral_hue='slate',font=['Tahoma','Arial','sans-serif'])
    with gr.Blocks(theme=theme,css=css,title='ميزان | قراءة ذكية للذهب',analytics_enabled=False,fill_width=True,head='<style>html,body{background:#081b17!important;margin:0}</style>',delete_cache=(3600,3600)) as demo:
        gr.HTML('''<header id="topbar"><div class="brand"><span class="brand-mark">◇</span><div><strong>ميزان</strong><small>GOLD VISION LAB</small></div></div><div class="status-pill">'''+('جاهز لقراءة صورتك' if ready else 'بانتظار أوزان النموذج')+'''</div></header>
        <section class="hero"><svg class="hero-gem" viewBox="0 0 240 240" fill="none" stroke="#dfbf7b" stroke-width="1"><path d="M30 85 70 35h100l40 50-90 125Z M30 85h180 M70 35l-5 50 55 125 55-125-5-50 M65 85l55-50 55 50"/><circle cx="120" cy="120" r="114" opacity=".4"/></svg><div class="eyebrow">عين على التفاصيل · قراءة بالذكاء الاصطناعي</div><h1>تفاصيل صغيرة.<br><span>قراءة أكثر وضوحًا.</span></h1><p>صورة واحدة تمنحك تقديرًا بصريًا للقطعة. ارفع صورتك، واستكشف درجات الحقيقي والمزيف في مساحة واضحة وسهلة.</p><div class="hero-badges"><span>واجهة عربية</span><span>معاينة قبل التحليل</span><span>نتيجة مع درجاتها</span></div></section>''')
        with gr.Row(elem_id='workspace'):
            with gr.Column(scale=1,min_width=310,elem_id='input-panel'):
                gr.HTML('<div class="section-heading"><span class="step">01</span><div><h2>أضف صورة القطعة</h2><p>إضاءة متوازنة، تفاصيل قريبة، وخلفية بسيطة.</p></div></div>')
                upload=gr.File(label='اختر صورة أو اسحبها هنا',file_types=['.jpg','.jpeg','.png','.webp'],type='filepath',height=155,elem_id='upload-area')
                preview=gr.Image(show_label=False,interactive=False,visible=False,height=230,elem_id='preview')
                with gr.Row(elem_id='action-row'):
                    analyse=gr.Button('تحليل الصورة  ←',variant='primary',interactive=ready,elem_id='analyse',scale=3)
                    reset=gr.Button('بدء جديد',elem_id='reset',scale=1)
                gr.HTML('<div class="tip">JPG · PNG · WebP &nbsp; | &nbsp; حتى 8 ميجابايت<br>تجنّب الظلال القوية والمرشحات اللونية للحفاظ على تفاصيل الصورة.</div>')
            with gr.Column(scale=1,min_width=340,elem_id='result-panel'):
                gr.HTML('<div class="section-heading"><span class="step">02</span><div><h2>اقرأ النتيجة</h2><p>تقدير واضح، مع توضيح درجات التصنيف.</p></div></div>')
                result=gr.HTML(IDLE,elem_id='result-card')
        gr.HTML('''<section class="stats"><div class="stat"><b>88.45%</b><span>دقة تجربة المشروع المسجلة</span></div><div class="stat"><b>303</b><span>صور في مجموعة الاختبار</span></div><div class="stat"><b>55%</b><span>عتبة النموذج المنشور</span></div></section><div class="notice"><span class="icon">ⓘ</span><span>قراءة بصرية مساعدة، وليست شهادة أصالة. الصور وحدها لا تثبت نقاء الذهب أو عياره؛ استعن بفحص مختص قبل اتخاذ قرار شراء. نتيجة الاختبار تخص بيانات المشروع وليست ضمانًا لصور جديدة.</span></div><div class="bottom"><span>ميزان · مساحة لفهم الصورة</span><span>BUILT WITH GRADIO · GOLD VISION LAB</span></div>''')
        upload.change(predictor.preview,inputs=upload,outputs=[preview,result],api_name=False)
        analyse.click(predictor.predict,inputs=upload,outputs=result,api_name='analyse',concurrency_limit=1)
        reset.click(lambda:(None,gr.update(value=None,visible=False),IDLE),outputs=[upload,preview,result],api_name=False)
    return demo.queue(default_concurrency_limit=1,max_size=12)


def main():
    parser=argparse.ArgumentParser(description='واجهة Gradio العربية لتصنيف صور الذهب')
    parser.add_argument('--share',action='store_true',help='إنشاء رابط Gradio مؤقت عبر الإنترنت')
    parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    build_demo().launch(server_name='0.0.0.0',server_port=args.port,share=args.share,
                        show_error=False,show_api=False,max_file_size='8mb')


if __name__=='__main__':main()
