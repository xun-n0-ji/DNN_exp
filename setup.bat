@echo off
REM uv���g�p����Python�̉��z�����쐬���A�p�b�P�[�W���C���X�g�[������o�b�`�t�@�C��

echo === uv���g�p���ĉ��z�����쐬���Ă��܂� ===
uv venv -p 3.10

echo === ���z�����A�N�e�B�x�[�g���Ă��܂� ===
call .venv\Scripts\activate

echo === requirements.txt����p�b�P�[�W���C���X�g�[�����Ă��܂� ===
uv pip install -r requirements.txt

echo === �C���X�g�[�����ꂽ�p�b�P�[�W��\�����܂� ===
uv pip list

echo === �������܂��� ===
echo ���z�����I������ɂ� 'deactivate' �Ɠ��͂��Ă�������

pause