# Create your views here.
from django.http import HttpResponse, HttpResponseServerError
from django.shortcuts import render,redirect
from .forms import UploadForm
from .models import Upload
from scipy.io.wavfile import write
#from ttsai.tts import tts#, wav2sp_emb
from django.views.decorators.csrf import requires_csrf_token
from ttsai.tts import tts

@requires_csrf_token
def my_customized_server_error(request, template_name='500.html'):
    import sys
    from django.views import debug
    error_html = debug.technical_500_response(request, *sys.exc_info()).content
    return HttpResponseServerError(error_html)


def home_func(request):
	return render(request, 'home.html')

def author_func(request):
	return render(request, 'author.html')

def multi_func(request):
    import os
    import subprocess

    #保存PATH
    source = "media/media/"    

    #結果保存
    speech_result = ""

    if request.method == 'POST':
    	#フォルダーの初期化
        
        cmd = f'rm -r {source}*'
        subprocess.call(cmd, shell=True)
    	
        #アップロードファイルの保存
        form = UploadForm(request.POST,request.FILES)
        form.save()

        #アップロードしたファイル名を取得
        #ファイル名と拡張子を分割(ext->拡張子(.py))
        transcribe_file = request.FILES['document'].name
        name, ext = os.path.splitext(transcribe_file)

        #ファイルの変換処理
        f_input = source + transcribe_file
        f_output = source + name + "_16k.wav"
        upload_file_name = name + ".wav"
        
        cmd = 'ffmpeg -i ' + f_input + ' -ar 16000 -ac 1 ' + f_output
        subprocess.call(cmd, shell=True)

        pca_feats = wav2sp_emb(f_output)
        text=request.POST['text']

        gen_wav = tts(pca_feats=pca_feats, text=text)

        out_file = f'{source}gen.wav'

        write(f"{out_file}", 24000, gen_wav)

        """
        #作業用ファイルの削除
        cmd = 'rm -f ' + f_input #+ ' ' + f_output     
        subprocess.call(cmd, shell=True)
        print(f_output)
        """
        return render(request, 'multi.html', {
        'form': form,
        'file_name': out_file,
        'tts_result':'coming soon',
    })
    else:
        form = UploadForm()
        return render(request, 'multi.html', {
        	'tts_result': 'coming soon',
        	'form': form,
        	})

def demo_func(request):
    import os
    import subprocess
    from sklearn.preprocessing import StandardScaler
    import base64

    #保存PATH
    source = "media/media/"    

    #結果保存
    speech_result = ""

    if request.method == 'POST':
    	#フォルダーの初期化
        cmd = f'rm -r {source}*'
        subprocess.call(cmd, shell=True)
    	
        #アップロードファイルの保存
        form = UploadForm(request.POST,request.FILES)
        form.save()

        text=request.POST['text']
        gen_wav = tts(text=text)

        f_output = f'{source}gen_test.wav'

        data_format = b'data:audio/wav;base64,'

        wav_format = b'RIFF\x96P\x06\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\xc0]\x00\x00\x80\xbb\x00\x00\x02\x00\x10\x00datarP\x06\x00'

        scaler = StandardScaler()
        scaler.fit(gen_wav.reshape(-1, 1))

        norm_wav = scaler.transform(gen_wav.reshape(-1, 1)).squeeze()

        norm_wav = norm_wav * 1700
        ndarr = norm_wav.astype('int16')
        to_bytes = ndarr.tobytes('F')
        data = base64.b64encode(wav_format + to_bytes)
        data_uri = data_format + data

        #write(f"{f_output}", 24000, gen_wav)

        return render(request, 'demo.html', {
        'form': form,
        'file_name': f_output, #f_output,
        'tts_result':'coming soon',
        'data_uri': data_uri.decode(),
    })
    else:
        form = UploadForm()
        return render(request, 'demo.html', {
        	'tts_result': 'coming soon',
        	'form': form,
        	})



