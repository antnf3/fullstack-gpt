wsl2 
 pip, venv 설치여부 확인
 1.
python3 -m pip
/usr/bin/python3: No module named pip

2. 설치
sudo apt install python3-pip
sudo apt install python3-venv

3. 가상환경설정
python -m venv ./env
source ./env/bin/activate
pip install -r requirements.txt



## LCEL
[LangChain Expression Language(LCEL)](https://python.langchain.com/docs/expression_language/)