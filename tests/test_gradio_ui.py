import io
from pathlib import Path
from PIL import Image
from app import create_app
import pytest
pytest.importorskip("gradio")
from gradio_app import Predictor,IDLE,result_card,error_card,build_demo
from model import GoldCNN
import torch


def checkpoint(tmp_path):
    path=tmp_path/'test.pt'
    torch.save(dict(classes=['fake','real'],state_dict=GoldCNN().state_dict()),path)
    return path


def test_predictor_matches_original_api(tmp_path):
    torch.set_num_threads(2)
    path=checkpoint(tmp_path)
    image=tmp_path/'photo.png'
    Image.new('RGB',(70,110),(210,160,70)).save(image)
    predictor=Predictor(path)
    with image.open('rb') as file:
        expected=create_app(path).test_client().post('/predict',data={'image':(file,'photo.png')}).get_json()
    rendered=predictor.predict(image)
    assert f"{expected['real_score']:.2f}%" in rendered
    assert f"{expected['fake_score']:.2f}%" in rendered
    assert f"{expected['decision_threshold']*100:.0f}%" in rendered
    preview,card=predictor.preview(image)
    assert preview['visible'] is True and card==IDLE
    assert predictor.preview(None)[0]['visible'] is False
    assert 'اختر صورة' in predictor.predict(None)
    broken=tmp_path/'broken.png';broken.write_text('<script>alert(1)</script>')
    assert 'تعذر' in predictor.predict(broken) or 'سليمة' in predictor.predict(broken)
    too_big=tmp_path/'large.png';too_big.write_bytes(b'x'*(8*1024*1024+1))
    assert '8 ميجابايت' in predictor.predict(too_big)


def test_decision_threshold_and_error_escaping():
    rendered=result_card(dict(label='مزيف',real_score=54.,fake_score=46.,decision_threshold=.55))
    assert 'يميل التقدير إلى المزيف' in rendered
    assert '54.00%' in rendered and '55%' in rendered
    assert '<script>' not in error_card('<script>x</script>')
    assert '&lt;script&gt;' in error_card('<script>x</script>')


def test_missing_model_disables_button(tmp_path):
    demo=build_demo(tmp_path/'missing.pt')
    buttons=[c for c in demo.config['components'] if c.get('props',{}).get('elem_id')=='analyse']
    assert len(buttons)==1 and buttons[0]['props']['interactive'] is False
