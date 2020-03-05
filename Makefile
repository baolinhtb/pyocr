# written by junying
# 2020-01-28
# ctpn was built in python2, it doesn't work in python3
# devenv: cuda-9.0, cudnn-7.0.5, tensorflow-1.5.0

prerequisites:
	@pip install -r requirements.txt

install.gpu: prerequisites
	@pip install tensorflow-gpu==1.5.0
	@chmod +x ./ctpn/lib/utils/make.sh
	@cd ./ctpn/lib/utils/ && ./make.sh

install.cpu: prerequisites
	@pip install tensorflow==1.5.0
	@chmod +x ./ctpn/lib/utils/make_cpu.sh
	@cd ./ctpn/lib/utils/ && ./make_cpu.sh

test.rest:
	@python rest.py

test.infer:
	@python infer.py

test.ocr:
	@curl -F "img=@samples/1.jpg" http://0.0.0.0:5000/ocr

test.idcard:
	@curl -F "img=@samples/1.jpg" http://0.0.0.0:5000/idcard

